from gesture_classifiers.base_model import BaseClassifierModel
import numpy
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D

from gesture_classifiers.classifier_errors import NoTrainingData


class DynamicHandGestureClassifier(BaseClassifierModel):
    def __init__(self, model_name='dynamic_hand_gesture_classifier'):
        super().__init__(model_name=model_name)
        self.feature_vector_size = 3

    def create_model(self):
        self.model = Sequential()
        self.model.add(Conv1D(160, activation='tanh', kernel_size=1, input_shape=(30, self.feature_vector_size)))
        self.model.add(LSTM(80))
        self.model.add(Dense(self.num_classes * 4, activation="sigmoid"))
        self.model.add(Dense(self.num_classes, activation="softmax"))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam',
                           metrics=["categorical_accuracy"])
        super().create_model()

    def data_reader(self):
        try:
            only_files = [f for f in os.listdir('saved_data/dynamic') if
                          os.path.isfile(os.path.join('saved_data/dynamic', f))]
        except FileNotFoundError:
            raise NoTrainingData
        data = []
        expected = []
        for each in only_files:
            temp_file = os.path.join('saved_data/dynamic', each)
            with open(temp_file) as file:
                loaded_array = numpy.loadtxt(file, delimiter=",")
                name = each[:-6]
                while name[-1] != '_':
                    name = name[:-1]
                if name[-1] == '_':
                    name = name[:-1]
                expected.append(name)
                loaded_array = frame_normalizer(loaded_array)

                data.append(loaded_array)

        data = numpy.array(data).reshape((-1, 30, 3))
        expected = numpy.array(expected).reshape(-1, 1)
        return data, expected


def frame_normalizer(input_array: numpy.array):
    return input_array - input_array[0]


def data_cutter(input_array: numpy.array):
    mask = []
    for each in input_array:
        mask.append(each[:3])
    return numpy.array(mask)
