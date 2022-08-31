"""simple_udp.py"""
from threading import Thread
from socket import socket, AF_INET, SOCK_DGRAM

class SimpleUDP(Thread):
    def __init__(self, recv_adapter="0.0.0.0", port=0):
        super(SimpleUDP, self).__init__()
        self._stop = False

        self.sock = socket(AF_INET, SOCK_DGRAM)
        if port:
            self.sock.bind((recv_adapter, port))
            self.sock.setblocking(False)

        self.sendto = self.sock.sendto
        self.recvfrom = self.sock.recvfrom

    def terminate(self):
        self._stop = True
        self.sock.close()

    def callback(self, func, *args, **kwargs):
        self._func = func
        self._args = args
        self._kwargs = kwargs

        self.start()

    def run(self):
        while not self._stop:
            try:
                self._func(*self._args, **self._kwargs)
            except BlockingIOError:
                continue
            except OSError:
                break