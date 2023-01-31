import zmq


class Poller:

    def __init__(self):
        self.poller = zmq.Poller()

    def register(self, subscriber):
        self.poller.register(subscriber.socket, zmq.POLLIN)

    def poll(self, subscriber):
        sockets = dict(self.poller.poll(10))
        if sockets:
            if sockets.get(subscriber.socket) == zmq.POLLIN:
                return True
        return False
