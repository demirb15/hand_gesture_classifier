import math
import os
import pickle

import numpy
from cv2 import cv2
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from hand_detector import HandDetector


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

    def transform_samples(self):
        only_files = [f for f in os.listdir(self.training_data_path) if
                      os.path.isfile(os.path.join(self.training_data_path, f))]
        for each in only_files:
            count = 0
            temp_file = os.path.join(self.training_data_path, each)
            frame = cv2.imread(temp_file, cv2.IMREAD_COLOR)
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
            samples.append(process_feature(data))
            expected.append(each[:-11])

        data_x = numpy.array(samples)
        data_x = data_x.reshape(-1, data_x.shape[1])
        data_y = numpy.array(expected).reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33)
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
        print(history)
        return

    def model_create(self):
        self.model = Sequential()
        self.model.add(Dense(420, input_shape=(self.feature_vector,),
                             kernel_initializer='he_uniform', activation='relu'))
        self.model.add(Dense(210, activation="relu"))
        self.model.add(Dense(69, activation="relu"))
        self.model.add(Dense(self.output_classes, activation="softmax"))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=["categorical_accuracy"])
        self.checkpoint_callback = ModelCheckpoint(filepath=self.checkpoint_path,
                                                   save_weights_only=True,
                                                   verbose=1)
        return


def euclidean_distance(f):
    return numpy.sqrt(f[0] ** 2 + f[1] ** 2 + f[2] ** 2)


def process_feature(feature: numpy.array):
    """
    Process feature vector
    :param feature: feature vector (21,3)
    :return: processed feature vector
    """
    f = feature.reshape(21, 3)
    # normalize to wrist, hence now each point a vector
    f = f - f[0]
    # blacklist
    indexes = (0, 1, 7, 9, 11, 13, 15, 19)
    f = numpy.delete(f, indexes, axis=0)
    for index in range(len(f)):
        f[index] = f[index] / euclidean_distance(f[index])
    return f.flatten()
