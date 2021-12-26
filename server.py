import base64
import binascii
import socket
import threading
import time
import numpy
import cv2

from connection_status import Connection
from dynamic_classifier import DynamicClassifier
from hand_detector import HandDetector
from static_classifier import StaticClassifier
from flask import Flask, request


class ClassificationServer:
    HOST = socket.gethostbyname(socket.getfqdn())
    PORT = 6969
    BUFFER_SIZE = 65536
    detector = HandDetector()
    s_classifier = StaticClassifier()
    d_classifier = DynamicClassifier()
    max_connection = 5
    open_connections = []

    def __init__(self, app):
        self.s_classifier.load_model()
        self.d_classifier.load_model()
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.BUFFER_SIZE)
        self.app = app

    def run_server(self):
        with self as this:
            this.app.run()

    def __enter__(self):
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_server()

    def start_server(self):
        self.server.bind((self.HOST, self.PORT))
        self.server.listen(self.max_connection)

    def stop_server(self):
        for each in self.open_connections:
            each.join(5)
        self.server.close()

    def is_available(self):
        return len(self.open_connections) < self.max_connection

    def accept_conn(self, ip=None):
        communication_socket, address = self.server.accept()
        if address[0] != ip:
            return Connection.NO_MATCH
        print(f'Connected to {address}')
        return communication_socket

    def classify(self, communication_socket: socket):
        cache = []
        last_package_timestamp = time.perf_counter()
        current_package_timestamp = time.perf_counter()
        try:
            while packet := communication_socket.recv(self.BUFFER_SIZE):
                if current_package_timestamp - last_package_timestamp > 3:
                    cache = cache[3:]
                try:
                    data = base64.b64decode(packet)
                except binascii.Error as e:
                    print(e)
                    continue
                np_data = numpy.frombuffer(data, dtype=numpy.uint8)
                frame = cv2.imdecode(np_data, 1)
                if frame is not None:
                    landmarks = self.detector.detect_hands(frame, False)
                    if landmarks is not None:
                        last_package_timestamp = current_package_timestamp
                        current_package_timestamp = time.perf_counter()
                        sc_prediction = self.s_classifier.predict(landmarks)
                        cache.append(landmarks)
                        if len(cache) > 20:
                            dc_prediction = self.d_classifier.predict(numpy.array(cache))
                            res = str(
                                {
                                    'static': sc_prediction,
                                    'dynamic': dc_prediction
                                }
                            ).encode(encoding='UTF-8')
                            try:
                                communication_socket.send(res)
                            except BrokenPipeError:
                                communication_socket.close()
                                break
                            cache.pop(0)
            communication_socket.close()
        except ConnectionResetError:
            communication_socket.close()
            return Connection.DISCONNECTED.value

    def thread_conn(self, incoming_ip):
        incorrect_conn = 0
        while conn := self.accept_conn(incoming_ip):
            if conn == Connection.NO_MATCH:
                incorrect_conn += 1
                conn.close()
                continue
            if incorrect_conn > 5:
                break
            self.classify(conn)
            conn.close()
            break
        th_id = next((th for th in self.open_connections if th.ident == threading.get_ident()), None)
        self.open_connections.remove(th_id)
        return Connection.DISCONNECTED.value


app = Flask(__name__)
classifier_server: ClassificationServer


@app.route('/')
def ask_to_connect():
    global classifier_server
    incoming_ip = request.remote_addr
    if not classifier_server.is_available():
        return Connection.TRY_AGAIN_LATER.value
    new_thread = threading.Thread(target=classifier_server.thread_conn, args=(incoming_ip,), daemon=True)
    classifier_server.open_connections.append(new_thread)
    new_thread.start()
    return Connection.SUCCESS.value


def main():
    global classifier_server, app
    classifier_server = ClassificationServer(app)
    classifier_server.run_server()


if __name__ == '__main__':
    main()
