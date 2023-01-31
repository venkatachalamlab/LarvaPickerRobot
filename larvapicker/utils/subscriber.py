import zmq


class Subscriber:

    def __init__(self, name, port='5001', context=None):
        self.name = str(name)
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect('tcp://localhost:' + port)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, self.name)

    def recv(self, prt=True):
        msg = self.socket.recv_string()
        requests = msg.split(' ')
        if len(requests) > 2:
            cmd, args = requests[1], requests[2:]
            if prt:
                argc = []
                for arg in args:
                    if len(arg) < 25:
                        argc.append(arg)
                    else:
                        break
                if len(argc) > 0:
                    print(f'{self.name} received message: {cmd}({argc})')
                else:
                    print(f'{self.name} received message: {cmd}')
        else:
            cmd, args = requests[1], None
            if prt:
                print(f'{self.name} received message: {cmd}')
        return cmd, args
