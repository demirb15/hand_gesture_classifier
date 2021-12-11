import math
import os

import numpy
from cv2 import cv2
from sklearn.preprocessing import MultiLabelBinarizer

from hand_detector import HandDetector


class StaticClassifier:
    h_detector = HandDetector()
    multi_label_binarizer = MultiLabelBinarizer()
    model_name = "static_classifier"
    training_data_path = f'training_data/{model_name}'
    processed_data_path = f'processed_data/{model_name}'

    def transform_samples(self):
        only_files = [f for f in os.listdir(self.training_data_path) if
                      os.path.isfile(os.path.join(self.training_data_path, f))]
        for each in only_files:
            temp_file = os.path.join(self.training_data_path, each)
            frame = cv2.imread(temp_file, cv2.IMREAD_COLOR)
            data = self.h_detector.detect_hands(frame, True)
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
