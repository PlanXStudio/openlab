"""ex04_03_2.py"""
from socket import socket, AF_INET, SOCK_DGRAM

ECHO_SERVER_IP = "127.0.0.1"
ECHO_SERVER_PORT = 49002

def main():
    sock = socket(AF_INET, SOCK_DGRAM)

    data = input("data: ").encode()
    sock.sendto(data, (ECHO_SERVER_IP, ECHO_SERVER_PORT))
    recv_data = sock.recv(1500).decode()
    print(f"{recv_data = }")

    sock.close()

if __name__ == "__main__":
    main()