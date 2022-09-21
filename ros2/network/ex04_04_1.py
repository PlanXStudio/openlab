"""ex04_04_1.py"""
from simple_udp import SimpleUDP

class UDPEchoServer(SimpleUDP):
    def __init__(self):
        super(UDPEchoServer, self).__init__(port=49002)
        self.callback(self.echo, 1500)

    def echo(self, buf_size):
        data, addr = self.recvfrom(buf_size)
        self.sendto(data, addr)
        print(f"{addr = }, data = {data.decode()}")

def main():
    echo = UDPEchoServer()
    input("Press the Enter key to exit.\n")
    echo.terminate()

if __name__ == "__main__":
    main()