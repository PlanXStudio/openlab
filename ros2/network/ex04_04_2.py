"""ex04_04_2.py"""
from simple_udp import SimpleUDP

class UDPEchoClient(SimpleUDP):
    def __init__(self, server_ip="127.0.0.1", server_port=49002):
        super(UDPEchoClient, self).__init__()
        self.callback(self.echo, server_ip, server_port, 1500)

    def echo(self, server_ip, server_port, buf_size):
        data = input("data: ").encode()
        self.sendto(data, (server_ip, server_port))
        recv_data, addr = self.recvfrom(buf_size)
        print(f"recv data: {recv_data.decode()}")

        self.terminate()

def main():
    echo = UDPEchoClient()

if __name__ == "__main__":
    main()