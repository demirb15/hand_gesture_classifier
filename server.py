import base64
import eventlet
import numpy
import socketio
from dynamic_classifier import DynamicClassifier
from static_classifier import StaticClassifier

sio = socketio.Server()
app = socketio.WSGIApp(sio, static_files={'/': {}})


class Server:
    dc = DynamicClassifier()
    sc = StaticClassifier()
    memory = {'0': []}

    def __init__(self):
        self.dc.load_model()
        self.sc.load_model()

    def classify(self, sid, features):
        self.memory[sid].append(features)
        guesses = [self.sc.predict(f) for f in self.memory[sid][-3:]]
        most_common = max(set(guesses), key=guesses.count)
        if len(self.memory[sid]) > 16:
            dynamic_gesture = self.dc.predict(numpy.array(self.memory[sid]))
            sio.emit('static_classification', {'response': f'{most_common}, {dynamic_gesture}'})
            self.memory[sid] = self.memory[sid][-8:]


@sio.event
def connect(sid, environ):
    print('connected to ', sid)
    server.memory[sid] = []


@sio.on('hand_landmarks')
def on_hand_landmarks(sid, data):
    decoded = base64.b64decode(data)
    features = numpy.frombuffer(decoded, dtype=numpy.float64)
    server.classify(sid, features)


@sio.on('lost_hand')
def on_hand_landmarks(sid, data):
    print("lost hand", data)


@sio.event
def disconnect(sid):
    del server.memory[sid]
    sio.disconnect(sid)
    print('disconnected ', sid)


if __name__ == '__main__':
    server = Server()
    eventlet.wsgi.server(eventlet.listen(('127.0.0.1', 5000)), app)
