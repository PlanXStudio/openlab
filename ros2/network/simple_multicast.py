"""simple_multicast.py"""
from socket import socket, inet_aton, timeout
from socket import AF_INET, SOCK_DGRAM
from socket import SOL_SOCKET, SO_REUSEADDR, INADDR_ANY
from socket import IPPROTO_IP, IP_ADD_MEMBERSHIP, IP_MULTICAST_TTL
import struct

class SimpleMulticast:
    def __init__(self, mcast_group, timeout=0.2):
        self.sock = socket(AF_INET, SOCK_DGRAM)
        self.sock.settimeout(timeout)

        self._mcast_group = mcast_group

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sock.close()

    def send(self, data, addr=()):
        self.sock.sendto(data, addr if addr else self._mcast_group)

    def recv(self, buf_size=1500):
        try:
            return self.sock.recvfrom(buf_size)
        except timeout:
            return (None, None)

class SimpleMulticastReceiver(SimpleMulticast):
    def __init__(self, mcast_group=('224.1.1.1', 49003), timeout=0.2):
        super(SimpleMulticastReceiver, self).__init__(mcast_group, timeout)

        self.sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self.sock.bind(('', mcast_group[1]))
        mreq = struct.pack("4sL", inet_aton(mcast_group[0]), INADDR_ANY)
        self.sock.setsockopt(IPPROTO_IP, IP_ADD_MEMBERSHIP, mreq)

class SimpleMulticastSender(SimpleMulticast):
    def __init__(self, mcast_group=('224.1.1.1', 49003), timeout=0.2, ttl=2):
        super(SimpleMulticastSender, self).__init__(mcast_group, timeout)

        ttl = struct.pack('b', ttl)
        self.sock.setsockopt(IPPROTO_IP, IP_MULTICAST_TTL, ttl)