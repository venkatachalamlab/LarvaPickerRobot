import zmq


class Publisher:

    def __init__(self, name, port='5000', context=None):
        self.name = str(name)
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind('tcp://*:' + port)

    def send(self, msg, prt=True):
        self.socket.send_string(msg)
        message = msg.split(' ')
        dest, cmd = message[0], message[1]
        if prt:
            if len(message) > 2:
                args = []
                for arg in message[2:]:
                    if len(arg) < 25:
                        args.append(arg)
                    else:
                        break
                if len(args) > 0:
                    print(f'{self.name} sending message to {dest}: {cmd}({args})')
                else:
                    print(f'{self.name} sending message to {dest}: {cmd}')
            else:
                print(f'{self.name} sending message to {dest}: {cmd}')
