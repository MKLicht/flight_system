import json
import socket
from threading import Thread


class ServerThread(Thread):
    def __init__(self, host, port):
        super(ServerThread, self).__init__()
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((host, port))
        self.client_socket_list = []

    def run(self):
        self.socket.listen(10)
        while True:
            client_socket, address_info = self.socket.accept()
            print('{} established.'.format(client_socket.getpeername()))
            self.client_socket_list.append(client_socket)

    def send(self, state):
        for client_socket in self.client_socket_list:
            try:
                req = json.dumps(state)
                client_socket.send(req.encode('utf-8'))
                client_socket.recv(1024).decode('utf-8')
            except ConnectionResetError:
                print('{} reset.'.format(client_socket.getpeername()))
                self.client_socket_list.remove(client_socket)
                client_socket.close()


class InteractServerThread(ServerThread):
    def __init__(self, actions, mask, localhost='localhost', port=9020):
        super(InteractServerThread, self).__init__(localhost, port)
        self.actions = actions
        self.mask = mask

    def send(self, state):
        for client_socket in self.client_socket_list:
            try:
                req = json.dumps(state)
                client_socket.send(req.encode('utf-8'))
                res = client_socket.recv(1024).decode('utf-8')
                res_dict = json.loads(res)
                self.interact(res_dict)
            except ConnectionResetError:
                print('{} reset.'.format(client_socket.getpeername()))
                self.client_socket_list.remove(client_socket)
                client_socket.close()

    def interact(self, res_dict):
        fighter_id = res_dict['id']
        if fighter_id >= 0 and self.mask[fighter_id]:
            self.actions[fighter_id] = res_dict['action']
