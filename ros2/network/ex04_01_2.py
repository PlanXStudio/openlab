"""ex04_01_2.py"""
from socket import socket, AF_INET, SOCK_STREAM
import struct
import time

class SimpleTCPClient:
    def __init__(self, server_ip="127.0.0.1", server_port=49001):
        self._conn_sock = socket(AF_INET, SOCK_STREAM)
        self._conn_sock.connect((server_ip, server_port))

    def send(self, fmt, *data):
        self._conn_sock.send(fmt.encode())
        time.sleep(0.1)
        self._conn_sock.send(struct.pack(fmt, *data))
        self._conn_sock.close()

def main():
    fmt = "11sIf"
    msg = ["hi, Python!", 1024, 3.14]
    
    client = SimpleTCPClient()

    msg[0] = msg[0].encode()
    client.send(fmt, *msg)

if __name__ == "__main__":
    main()