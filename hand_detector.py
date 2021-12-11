import mediapipe
import numpy
from cv2 import cv2


class HandDetector:
    solutions_hands = mediapipe.solutions.hands

    def __init__(self, mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.hands = self.solutions_hands.Hands(static_image_mode=mode,
                                                max_num_hands=max_hands,
                                                min_detection_confidence=detection_confidence,
                                                min_tracking_confidence=tracking_confidence)

    def detect_hands(self, image, is_rgb=True):
        if not is_rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        process_result = self.hands.process(image)
        hand_landmarks = process_result.multi_hand_landmarks
        norm_data = formalize(hand_landmarks)
        if norm_data is None:
            return None
        return norm_data[0]


def formalize(hand_landmarks):
    multi_hand_arr = []
    if hand_landmarks:
        for handLMs in hand_landmarks:
            single_hand_arr = []
            for index, lm in enumerate(handLMs.landmark):
                temp_arr = numpy.array([lm.x, lm.y, lm.z])
                single_hand_arr.append(temp_arr)
            single_hand_arr = numpy.asarray(single_hand_arr).flatten()
            multi_hand_arr.append(single_hand_arr)
        return numpy.array(multi_hand_arr)
    return None
