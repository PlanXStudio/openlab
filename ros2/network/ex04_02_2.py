"""ex04_02_2.py"""
from socket import socket, AF_INET, SOCK_STREAM
import struct
import time

class SimpleEchoClient:
    ECHO_DATA = "0123456789"

    def __init__(self, server_ip="127.0.0.1", server_port=49001):
        conn_sock = socket(AF_INET, SOCK_STREAM)
        
        conn_sock.connect((server_ip, server_port))

        conn_sock.send(self.ECHO_DATA.encode())
        data = conn_sock.recv(1500)

        t = struct.unpack("d", data[:8])[0]
        data = data[8:].decode()

        if (self.ECHO_DATA == data):
            print(f"{time.ctime(t)}: ECHO OK")
        else:
            print("ECHO FAIL")

        conn_sock.close()

def main():
    SimpleEchoClient()

if __name__ == "__main__":
    main()