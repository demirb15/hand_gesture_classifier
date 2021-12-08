import os
import pickle

import numpy
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.python.keras.callbacks import ModelCheckpoint

from gesture_classifiers.classifier_errors import *


class BaseClassifierModel:
    multi_label_binarizer = MultiLabelBinarizer()
    model = None
    feature_vector_size: int = None
    num_classes: int = None
    check_points: str = None
    multi_labels: str = None
    cp_callback = ModelCheckpoint

    def __init__(self, model_name=None):
        self.model_name = model_name
        if not os.path.isdir('classification_models'):
            os.mkdir('classification_models')
        self.save_folder = f'classification_models/{model_name}/'
        if not os.path.isdir(self.save_folder):
            os.mkdir(self.save_folder)
        self.check_points = f'{self.save_folder}cp.ckpt'
        self.multi_labels = f'{self.save_folder}mlb.pkl'
        if not os.path.isdir('output'):
            os.mkdir('output')

    def create_model(self):
        self.cp_callback = ModelCheckpoint(filepath=self.save_folder,
                                           save_weights_only=True,
                                           verbose=1)

    def load_model(self):
        model_dir = 'classification_models'
        if not os.path.isdir(os.path.join(model_dir)):
            raise NoDirExist
        if self.model_name is None:
            raise NoModelName
        saved_model_path = os.path.join('classification_models', self.model_name)
        if not os.path.isdir(saved_model_path):
            raise NoSavedModelExist(saved_model_path)
        if not os.path.isfile(self.check_points) or not os.path.isfile(self.multi_labels):
            raise NoWeightsAvailable

        with open(self.multi_labels, 'rb') as f:
            self.multi_label_binarizer, self.feature_vector_size = pickle.load(f)
        self.num_classes = self.multi_label_binarizer.classes_
        self.create_model()
        if self.model is None:
            raise NoModelError
        self.model.load_weights(self.check_points).expect_partial()

    def train_model(self, save: bool = False, epoch_num: int = 100, batch_size: int = 60):
        x_train, x_test, y_train, y_test = self.load_data()
        self.create_model()
        history = self.model_fit(x_train, x_test, y_train, y_test, epoch_num=epoch_num, batch_size=batch_size)
        predictions = self.model_test(x_test)
        self.draw_graph(history, epoch_num=epoch_num, save=save)
        self.get_report(y_test, predictions, save=save)

    def load_data(self, test_size: float = 0.33):
        data_x, data_y = self.data_reader()
        if data_x is None or data_y is None:
            raise NoDataReader
        x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=test_size)
        y_train = self.multi_label_binarizer.fit_transform(y_train)
        y_test = self.multi_label_binarizer.transform(y_test)
        self.num_classes = y_test.shape[1]
        return x_train, x_test, y_train, y_test

    def model_fit(self, x_train, x_test, y_train, y_test, epoch_num, batch_size):
        history = self.model.fit(x_train,
                                 y_train,
                                 validation_data=(x_test, y_test),
                                 epochs=epoch_num,
                                 batch_size=batch_size,
                                 callbacks=[self.cp_callback])
        return history

    def model_test(self, x_test):
        predictions = self.model.predict(x_test)
        return predictions

    def draw_graph(self, history, epoch_num, save: bool = False):
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(numpy.arange(0, epoch_num), history.history["loss"], label="train_loss")
        plt.plot(numpy.arange(0, epoch_num), history.history["val_loss"], label="val_loss")
        plt.plot(numpy.arange(0, epoch_num), history.history["categorical_accuracy"], label="train_acc")
        plt.plot(numpy.arange(0, epoch_num), history.history["val_categorical_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        if not save:
            return
        save_location = os.path.join('output')
        file_fig = os.path.join(save_location, f'{self.model_name}_fig.png')
        if os.path.exists(file_fig):
            os.remove(file_fig)
        plt.savefig(file_fig)

    def get_report(self, y_test, predictions, save):
        class_report = classification_report(y_test.argmax(axis=1),
                                             predictions.argmax(axis=1),
                                             target_names=[str(x) for x in self.multi_label_binarizer.classes_])
        if not save:
            return
        save_location = os.path.join('output')
        file_report = os.path.join(save_location, f'{self.model_name}_report.txt')
        if os.path.exists(file_report):
            os.remove(file_report)
        with open(file_report, "w+") as f:
            f.write(class_report)
        return class_report

    def data_reader(self):
        return None, None

    def model_summery(self):
        return self.model.summary()
