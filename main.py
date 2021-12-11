import cv2
import numpy

from hand_detector import HandDetector
from static_classifier import StaticClassifier

test = numpy.loadtxt('processed_data/static_classifier/german_two_155448.csv', delimiter=",")
if __name__ == '__main__':
    test = numpy.array(test)
    sc = StaticClassifier()
    sc.load_model()
    capture = cv2.VideoCapture(0)
    cv2.namedWindow("DISPLAY", cv2.WINDOW_FULLSCREEN)
    detector = HandDetector()
    classify = False
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            cv2.destroyAllWindows()
            break
        key = cv2.waitKey(1)
        if key is ord('q'):
            cv2.destroyAllWindows()
            break
        elif key is ord('s'):
            classify = not classify
        if classify:
            print(sc.predict(detector.detect_hands(frame, False)))
        cv2.imshow("DISPLAY", frame)
    capture.release()
