from gesture_classifiers import DynamicHandGestureClassifier
from gesture_classifiers.static_hand_gesture_classifier import StaticHandGestureClassifier


def run():
    model = StaticHandGestureClassifier()
    model.train_model(save=True, epoch_num=10)
    model = DynamicHandGestureClassifier()
    model.train_model(save=True, epoch_num=10)


run()
