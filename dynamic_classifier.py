import cv2
import os
import pickle
import numpy
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from hand_detector import HandDetector
from utils import draw_graph
from utils import write_report


class DynamicClassifier:
    h_detector = HandDetector()
    multi_label_binarizer = MultiLabelBinarizer()
    model_name = "dynamic_classifier"
    training_data_path = f'training_data/{model_name}'
    processed_data_path = f'processed_data/{model_name}'
    feature_vector = None
    output_classes = None
    model: Sequential = None
    checkpoint_callback: ModelCheckpoint = None
    checkpoint_path = f'classification_models/{model_name}/cp.ckpt'
    multilabel_path = f'classification_models/{model_name}/mlb.pkl'
    s1, s2 = None, None

    def __init__(self):
        if self.s1 is None or self.s2 is None:
            self.s1, self.s2 = get_feature_shape()
        try:
            os.mkdir(f'classification_models/{self.model_name}')
        except FileExistsError:
            pass
        try:
            os.mkdir(f'outputs/')
        except FileExistsError:
            pass

    def transform_samples(self):
        only_files = [f for f in os.listdir(self.training_data_path) if
                      os.path.isfile(os.path.join(self.training_data_path, f))]
        for each in only_files:
            count = 0
            temp_file = os.path.join(self.training_data_path, each)
            cap = cv2.VideoCapture(temp_file)
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                data = self.h_detector.detect_hands(frame, True)
                while count < 5 and data is None:
                    data = self.h_detector.detect_hands(frame, True)
                    count = count + 1
                if data is None:
                    continue
                frames.append(data.flatten())
            cap.release()
            np_frames = numpy.array(frames).reshape(-1, 63)
            base_name = each[:-5]
            path = os.path.join(self.processed_data_path, base_name + '.csv')
            if len(np_frames) < 20:
                continue
            print('i', end='')
            numpy.savetxt(path, np_frames, delimiter=",")

    def load_data(self):
        only_files = [f for f in os.listdir(self.processed_data_path) if
                      os.path.isfile(os.path.join(self.processed_data_path, f))]
        samples = []
        expected = []
        for each in only_files:
            path = os.path.join(self.processed_data_path, each)
            try:
                data = numpy.loadtxt(path, delimiter=",")
            except UserWarning:
                print(each)
                continue
            # track tip of index finger
            processed_data_list = process_feature(data)
            for processed_data in processed_data_list:
                samples.append(processed_data)
                expected.append(each[:-11])
        data_x = numpy.array(samples)
        data_y = numpy.array(expected).reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=40)
        y_train = self.multi_label_binarizer.fit_transform(y_train)
        y_test = self.multi_label_binarizer.transform(y_test)
        self.feature_vector = get_feature_shape()
        self.output_classes = y_test.shape[1]
        with open(self.multilabel_path, 'wb') as f:
            pickle.dump(self.multi_label_binarizer, f)
        return x_train, x_test, y_train, y_test

    def model_create(self):
        self.model = Sequential()
        self.model.add(LSTM(73, input_shape=self.feature_vector,
                            dropout=0.2,
                            return_sequences=True))
        self.model.add(LSTM(23))
        self.model.add(Dense(self.output_classes, activation="softmax"))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=["categorical_accuracy"])
        self.checkpoint_callback = ModelCheckpoint(filepath=self.checkpoint_path,
                                                   save_weights_only=True,
                                                   verbose=1)
        return

    def model_fit(self, epochs=100, batch_size=10):
        x_train, x_test, y_train, y_test = self.load_data()
        self.model_create()
        print("[INFO] training network...")
        history = self.model.fit(x_train, y_train,
                                 validation_data=(x_test, y_test),
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 callbacks=[self.checkpoint_callback])
        predictions = self.model.predict(x_test)
        fig_path = os.path.join('outputs', f'{self.model_name}_fig.png')
        history_path = os.path.join('outputs', f'{self.model_name}_report.txt')
        draw_graph(history, epochs, fig_path)
        report = write_report(y_test=y_test, predictions=predictions, mlb=self.multi_label_binarizer,
                              file_path=history_path)
        print(report)
        return

    def load_model(self):
        with open(self.multilabel_path, 'rb') as f:
            self.multi_label_binarizer = pickle.load(f)
        self.output_classes = len(self.multi_label_binarizer.classes_)
        self.feature_vector = get_feature_shape()
        self.model_create()
        self.model.load_weights(self.checkpoint_path).expect_partial()
        return

    def predict(self, features):
        if features is None:
            return None
        processed_features = process_feature(features).reshape((-1, self.s1, self.s2))
        predictions = self.model.predict(processed_features)
        index = predictions.argmax(axis=1)
        predict_vec = numpy.zeros(predictions.shape)
        confidence = []
        for e, i in enumerate(index):
            predict_vec[e][i] = 1
            confidence.append(predictions[e][i])
        try:
            transform = self.multi_label_binarizer.inverse_transform(predict_vec)
            if transform == [()]:
                return "No Match"
            ret = [[transform[i][0], e] for i, e in enumerate(confidence)]
            return ret
        except KeyError:
            return "No Match"


def process_feature(feature):
    """
    :param feature: seq of feature vector (21,3)
    :return:
    """
    f = feature.reshape(-1, 21, 3)
    feature_seq = []
    frame_len = 12
    for index in range(0, len(f) - frame_len, 5):
        f_to_track = [0, 4, 8, 12, 16, 20]
        extracted = f[index:index + frame_len, f_to_track, :] - f[index + 1:index + 1 + frame_len, f_to_track, :]
        norms = numpy.linalg.norm(extracted, axis=2).reshape((frame_len, len(f_to_track), 1))
        processed = extracted / norms
        processed = processed.reshape((frame_len, len(f_to_track) * 3))
        feature_seq.append(processed)
    return numpy.array(feature_seq)


def get_feature_shape():
    np_rand = numpy.random.random((30, 63))
    f = numpy.array(process_feature(np_rand))
    return f[0].shape
