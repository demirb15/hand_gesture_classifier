import cv2
from hand_detector import HandDetector
from static_classifier import StaticClassifier

if __name__ == '__main__':
    sc = StaticClassifier()
    x_train, x_test, y_train, y_test = sc.load_data()
    exit()
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
            print(detector.detect_hands(frame, False))
        cv2.imshow("DISPLAY", frame)
    capture.release()
