"""ex04_01_1.py"""
from socket import socket, AF_INET, SOCK_STREAM,  SOL_SOCKET, SO_REUSEADDR
import struct

class SimpleTCPServer:
    def __init__(self, adapter="0.0.0.0", server_port=49001):       
        self._sock = socket(AF_INET, SOCK_STREAM)
        self._sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self._sock.bind((adapter, server_port))
        self._sock.listen(5)

    def __del__(self):
        self._sock.close()

    def receive(self, buf_size=1500):
        conn_sock, addr = self._sock.accept()

        fmt = conn_sock.recv(buf_size)
        data = conn_sock.recv(buf_size)
        conn_sock.close()

        return addr, struct.unpack(fmt, data)

def main():
    srv = SimpleTCPServer()
    data = srv.receive()    
    print(f"{data[0]}, {data[1][0].decode()}, {data[1][1]}, {data[1][2]:.3}")

if __name__ == "__main__":
    main()