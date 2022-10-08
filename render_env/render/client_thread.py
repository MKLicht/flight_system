import json
import socket
from threading import Thread


class ClientThread(Thread):
    def __init__(self, render_app, host, port):
        super(ClientThread, self).__init__()
        self.host = host
        self.port = port
        self.state_dict = None
        self.render_app = render_app

    def run(self):
        client_socket = None
        while True:
            if client_socket is None:
                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                client_socket.connect((self.host, self.port))
                print('{} established.'.format(client_socket.getpeername()))
                while True:
                    req = json.dumps({
                        'id': self.render_app.focus,
                        'action': self.render_app.action
                    })
                    res = client_socket.recv(4096)
                    client_socket.send(req.encode('utf-8'))
                    self.state_dict = json.loads(res.decode('utf-8'))
            except ConnectionResetError:
                print('{} reset.'.format(client_socket.getpeername()))
                client_socket.close()
                client_socket = None
            except ConnectionRefusedError:
                print('{} refused.'.format(client_socket.getsockname()))
