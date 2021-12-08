from gesture_classifiers.base_model import BaseClassifierModel
import numpy
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from gesture_classifiers.classifier_errors import NoTrainingData


class StaticHandGestureClassifier(BaseClassifierModel):
    def __init__(self, model_name='static_hand_gesture_classifier'):
        super().__init__(model_name=model_name)
        self.feature_vector_size = 63

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(120, input_shape=(self.feature_vector_size,),
                             kernel_initializer='he_uniform', activation='relu'))
        self.model.add(Dense(60, activation="relu"))
        self.model.add(Dense(self.num_classes, activation="softmax"))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=["categorical_accuracy"])
        super().create_model()

    def data_reader(self):
        try:
            only_files = [f for f in os.listdir('saved_data/static') if
                          os.path.isfile(os.path.join('saved_data/static', f))]
        except FileNotFoundError:
            raise NoTrainingData
        data = numpy.zeros((1, 63))
        expected = []
        for each in only_files:
            temp_file = os.path.join('saved_data/static', each)
            with open(temp_file) as file:
                loaded_array = numpy.loadtxt(file, delimiter=",")
                name = each[:-6]
                if name[-1] == '_':
                    name = name[:-1]
                for i in range(loaded_array.shape[0]):
                    expected.append(name)
                data = numpy.concatenate((data, loaded_array), axis=0)
        data = numpy.delete(data, 0, axis=0)
        data = numpy.array(data).reshape(-1, 63)
        expected = numpy.array(expected).reshape(-1, 1)
        return data, expected
