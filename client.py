import time

from connection_status import Connection
import ast
import base64
import socket
import threading
import imutils
import cv2
import requests

HOST = socket.gethostbyname(socket.getfqdn())
PORT = 6969
BUFFER_SIZE = 65536

CAMERA_CAPTURE_FPS = 15
CAMERA_CAPTURE_WIDTH = 1280
CAMERA_CAPTURE_HEIGHT = 720


def listener(client: socket.socket):
    while message := client.recv(BUFFER_SIZE):
        decoded_message = message.decode(encoding='UTF-8')
        dict_message = ast.literal_eval(decoded_message)
        static_c = dict_message['static']
        dynamic_c = dict_message['dynamic']
        print(static_c)
        print(dynamic_c)


def main():
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFFER_SIZE)
    res = requests.get('http://127.0.0.1:5000/')
    if res.text == Connection.SUCCESS.value:
        client.connect((HOST, PORT))
    else:
        return
    x_thread = threading.Thread(target=listener, args=(client,), daemon=True)
    x_thread.start()
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)
    capture.set(4, 480)
    capture.set(cv2.CAP_PROP_FPS, CAMERA_CAPTURE_FPS)
    cv2.namedWindow("SENDING VIDEO")
    start_time = time.time()
    frame_counter = 0
    fps = 0
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            cv2.destroyAllWindows()
            break
        key = cv2.waitKey(1) & 0xFF
        if key is ord('q'):
            cv2.destroyWindow("SENDING VIDEO")
            break
        if time.time() - start_time >= 1:
            fps = frame_counter
            frame_counter = 0
            start_time = time.time()
        cv2.putText(frame, "FPS: " + str(fps), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        frame_counter += 1
        cv2.imshow("SENDING VIDEO", frame)
        # process img
        frame = imutils.resize(frame, width=400)
        encoded, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        message = base64.b64encode(buffer)
        client.send(message)
    capture.release()
    client.close()


if __name__ == '__main__':
    main()
