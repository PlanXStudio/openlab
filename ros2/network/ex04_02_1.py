"""ex04_02_1.py"""
from socket import socket, AF_INET, SOCK_STREAM,  SOL_SOCKET, SO_REUSEADDR
from threading import Thread
import struct
import time

class SimpleEchoServer(Thread):
    def __init__(self, ip="0.0.0.0", port=49001):
        super(SimpleEchoServer, self).__init__()

        self._sock = socket(AF_INET, SOCK_STREAM)
        
        self._sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        self._sock.bind((ip, port))
        self._sock.listen(5)
        self._sock.setblocking(False)

        self.start()
    
    def terminate(self):
        self._sock.close()

    def __work_thread(self, sock):
        data = sock.recv(1500)
        t = time.time()
        data = struct.pack("d", t) + data
        print(t, data[8:].decode())
        sock.send(data)
        sock.close()

    def run(self):
        while True:
            try:
                conn_sock, addr = self._sock.accept()
                print(f"Client IP: {addr[0]}, Port: {addr[1]}")
            except BlockingIOError:
                continue
            except OSError:
                break

            Thread(target=self.__work_thread, args=(conn_sock,)).start()

def main():
    echo_srv = SimpleEchoServer()
    input("Press the Enter key to exit.\n")
    echo_srv.terminate()
    
if __name__ == "__main__":
    main()