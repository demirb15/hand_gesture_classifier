import cv2
import numpy

from dynamic_classifier import DynamicClassifier
from hand_detector import HandDetector
from static_classifier import StaticClassifier

if __name__ == '__main__':
    dc = DynamicClassifier()
    sc = StaticClassifier()
    # dc.model_fit(100)
    # sc.model_fit(300)
    # exit()
    dc.load_model()
    sc.load_model()
    capture = cv2.VideoCapture(0)
    cv2.namedWindow("DISPLAY", cv2.WINDOW_FULLSCREEN)
    detector = HandDetector()
    classify = False
    cache = []
    counter = 0
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
            f = detector.detect_hands(frame, False)
            if f is not None:
                cache.append(f)
                counter = 0
            else:
                counter += 1
                if counter > 20:
                    cache.clear()
            prediction = sc.predict(f)
            if prediction is not None:
                print('static:: ', prediction[0][0], end=' ')
                if prediction[0][0] == 'raised_fist':
                    cache.clear()

            if len(cache) > 20:
                np_cache = numpy.array(cache)
                d_prediction = dc.predict(np_cache)
                if d_prediction is not None:
                    print('dynamic:: ', d_prediction, end='')
                cache.pop(0)
            if prediction is not None:
                print('')
        cv2.imshow("DISPLAY", frame)
    capture.release()
