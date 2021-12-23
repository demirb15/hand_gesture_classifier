import os
import pickle
import numpy
from tensorflow.keras import Sequential
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from hand_detector import HandDetector
from utils import draw_graph
from utils import write_report
from cv2 import imread, IMREAD_COLOR


class StaticClassifier:
    h_detector = HandDetector()
    multi_label_binarizer = MultiLabelBinarizer()
    model_name = "static_classifier"
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
            frame = imread(temp_file, IMREAD_COLOR)
            data = self.h_detector.detect_hands(frame, True)
            while count < 10 and data is None:
                data = self.h_detector.detect_hands(frame, True)
                count += 1
            if data is None:
                continue
            base_name = each[:-4]
            path = os.path.join(self.processed_data_path, base_name + '.csv')
            numpy.savetxt(path, data, delimiter=",")

    def load_data(self):
        only_files = [f for f in os.listdir(self.processed_data_path) if
                      os.path.isfile(os.path.join(self.processed_data_path, f))]
        samples = []
        expected = []
        for each in only_files:
            path = os.path.join(self.processed_data_path, each)
            data = numpy.loadtxt(path, delimiter=",")
            samples.append(self.process_feature(data))
            expected.append(each[:-11])

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

    def model_fit(self, epochs=100, batch_size=10):
        x_train, x_test, y_train, y_test = self.load_data()
        self.model_create()
        print("[INFO] training network...")
        history = self.model.fit(x_train, y_train,
                                 validation_data=(x_test, y_test),
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 callbacks=[self.checkpoint_callback])
        print(min(history.history['val_loss']))
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
        k_initializer = initializers.he_normal
        self.model.add(Dense(73, input_shape=(self.feature_vector,),
                             kernel_initializer=k_initializer, activation='relu'))
        self.model.add(Dropout(0.2))
        # self.model.add(Dense(90, activation="relu"))
        self.model.add(Dense(42, kernel_initializer=k_initializer, activation="relu"))
        self.model.add(Dense(self.output_classes, kernel_initializer=k_initializer, activation="softmax"))
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
            return transform[0][0]
        except KeyError:
            return "No Match"

    @staticmethod
    def process_feature(feature: numpy.array):
        """
        Process feature vector
        :param feature: feature vector (21,3)
        :return: processed feature vector
        """
        demo = True
        if demo:
            return demo_process_feature_3_0(feature)
        f = feature.reshape(21, 3)
        # normalize to wrist, hence now each point a vector
        f = f - f[0]
        # blacklist
        # indexes = 0
        indexes = (0, 1, 7, 9, 11, 13, 15, 19)
        f = numpy.delete(f, indexes, axis=0)
        norms = numpy.linalg.norm(f, axis=1).reshape(len(f), 1)
        f = f / norms
        return f.flatten()


def demo_process_feature(feature: numpy.array):
    f = feature.reshape(21, 3)
    zeros = numpy.zeros(3)
    f_map = [zeros,
             f[0], f[1], f[2], f[3],
             f[0], f[5], f[6], f[7],
             f[0], f[9], f[10], f[11],
             f[0], f[13], f[14], f[15],
             f[0], f[17], f[18], f[19]]
    f = f - f_map
    i = 7
    indexes = (0, i, i + 4, i + 8, i + 12)
    f = numpy.delete(f, indexes, axis=0)
    norms = numpy.linalg.norm(f, axis=1).reshape(len(f), 1)
    f = f / norms
    return f.flatten()


def demo_process_feature_2_0(feature: numpy.array):
    f = feature.reshape(21, 3)
    zeros = numpy.zeros(3)
    f_map = [zeros,
             f[0], f[1], zeros, f[3],
             f[0], f[5], zeros, f[6],
             f[0], f[9], zeros, f[10],
             f[0], f[13], zeros, f[15],
             f[0], f[17], zeros, f[18]]
    """
    0
    1,2,3,4,
    5,6,7,8,
    9,10,11,12,
    13,14,15,16,
    17,18,19,20,
    """
    f = f - f_map
    i = 7
    indexes = (0, i, i + 4, i + 8, i + 12)
    f = numpy.delete(f, indexes, axis=0)
    norms = numpy.linalg.norm(f, axis=1).reshape(len(f), 1)
    f = f / norms
    return f.flatten()


def demo_process_feature_3_0(feature: numpy.array):
    f = feature.reshape(21, 3)
    zeros = numpy.full(3, 42)
    f_map = numpy.array([zeros,
                         zeros, f[0], zeros, f[2],
                         f[0], f[5], zeros, f[6],
                         f[0], f[9], zeros, f[10],
                         f[0], f[13], zeros, f[14],
                         f[0], f[17], zeros, f[18]])
    f = f - f_map
    indexes = numpy.unique(numpy.where(numpy.array(f_map) == zeros))
    f = numpy.delete(f, indexes, axis=0)
    norms = numpy.linalg.norm(f, axis=1).reshape(len(f), 1) + 1e-19
    f = f / norms
    return f.flatten()
