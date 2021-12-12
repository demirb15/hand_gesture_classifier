import math
import os
import pickle

import numpy
from cv2 import cv2
from numpy import ndarray
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
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

    def __init__(self):
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
            base_name = each[:-4]
            path = os.path.join(self.processed_data_path, base_name + '.csv')
            numpy.savetxt(path, np_frames, delimiter=",")

    def load_data(self):
        only_files = [f for f in os.listdir(self.processed_data_path) if
                      os.path.isfile(os.path.join(self.processed_data_path, f))]
        samples = []
        expected = []
        for each in only_files:
            path = os.path.join(self.processed_data_path, each)
            data = numpy.loadtxt(path, delimiter=",")
            # track tip of index finger
            processed_data_list = self.process_feature(data, feature_to_track=8)
            for processed_data in processed_data_list:
                samples.append(processed_data)
                expected.append(each[:-12])
        data_x = numpy.array(samples)
        data_x = data_x.reshape(-1, data_x.shape[1])
        data_y = numpy.array(expected).reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=42)
        y_train = self.multi_label_binarizer.fit_transform(y_train)
        y_test = self.multi_label_binarizer.transform(y_test)
        self.feature_vector = x_test.shape[1]
        self.output_classes = y_test.shape[1]
        with open(self.multilabel_path, 'wb') as f:
            pickle.dump(self.multi_label_binarizer, f)
        return x_train, x_test, y_train, y_test

    """
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

    def model_create(self):
        self.model = Sequential()
        self.model.add(Dense(101, input_shape=(self.feature_vector,),
                             kernel_initializer='he_uniform', activation='relu'))
        self.model.add(Dropout(0.2))
        # self.model.add(Dense(90, activation="relu"))
        self.model.add(Dense(42, activation="relu"))
        self.model.add(Dense(self.output_classes, activation="softmax"))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='nadam',
                           metrics=["categorical_accuracy"])
        self.checkpoint_callback = ModelCheckpoint(filepath=self.checkpoint_path,
                                                   save_weights_only=True,
                                                   verbose=1)
        return

    def load_model(self):
        with open(self.multilabel_path, 'rb') as f:
            self.multi_label_binarizer = pickle.load(f)
        self.output_classes = len(self.multi_label_binarizer.classes_)
        temp = numpy.zeros(63).reshape(-1, 63)
        self.feature_vector = self.process_feature(temp).shape[0]
        self.model_create()
        self.model.load_weights(self.checkpoint_path).expect_partial()
        return

    def predict(self, features):
        if features is None:
            return None
        predictions = self.model.predict(self.process_feature(features).reshape(-1, self.feature_vector))
        index = predictions.argmax(axis=1)
        predict_vec = numpy.zeros(predictions.shape)
        predict_vec[0][index] = 1
        try:
            transform = self.multi_label_binarizer.inverse_transform(predict_vec)
            if transform == [()]:
                return "No Match"
            return transform
        except KeyError:
            return "No Match"
    """

    @staticmethod
    def process_feature(feature, feature_to_track=8):
        """
        :param feature: seq of feature vector (21,3)
        :param feature_to_track: feature that will be tracked
        :return:
        """
        f = feature.reshape(-1, 21, 3)
        feature_seq = []
        for index in range(0, len(f) - 18, 5):
            extracted = f[index:index + 15, feature_to_track, :] - f[index + 1:index + 16, feature_to_track, :]
            temp_seq = []
            for each in extracted:
                e_d = euclidean_distance(each)
                temp_seq.append(each / e_d)
            feature_seq.append(numpy.array(temp_seq).flatten())
        return feature_seq


def euclidean_distance(f):
    return numpy.sqrt(f[0] ** 2 + f[1] ** 2 + f[2] ** 2)
