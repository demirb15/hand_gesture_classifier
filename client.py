import base64
import socketio
from cv2 import cv2

from hand_detector import HandDetector

sio = socketio.Client()
client = None


class Client:
    classify = False
    capture = cv2.VideoCapture(0)
    cv2.namedWindow("DISPLAY", cv2.WINDOW_FULLSCREEN)
    detector = HandDetector()

    def __init__(self):
        pass


@sio.event
def connect():
    print('connection established')


@sio.event
def disconnect():
    print('disconnected from server')


@sio.on('dynamic_classification')
def on_classification(data):
    print(data)


@sio.on('static_classification')
def on_classification(data):
    print(data)


def send_hand_landmarks(data):
    if data is not None:
        sio.emit('hand_landmarks', base64.b64encode(data))
        return
    sio.emit('lost_hand', {'response': ''})
    return


def main():
    sio.connect('http://127.0.0.1:5000')
    while client.capture.isOpened():
        if not sio.connected:
            sio.connect('http://127.0.0.1:5000')
        ret, frame = client.capture.read()
        if not ret:
            cv2.destroyAllWindows()
            break
        key = cv2.waitKey(1)
        if key is ord('q'):
            cv2.destroyAllWindows()
            break
        elif key is ord('s'):
            client.classify = not client.classify
        if client.classify:
            f = client.detector.detect_hands(frame, False)
            if f is not None:
                send_hand_landmarks(f)
            else:
                sio.disconnect()
        cv2.imshow("DISPLAY", frame)
    client.capture.release()
    sio.disconnect()

    return


if __name__ == '__main__':
    client = Client()
    main()
