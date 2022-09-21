"""ex04_03_1.py"""
from socket import socket, AF_INET, SOCK_DGRAM

ADAPTER = "0.0.0.0"
ECHO_SERVER_PORT = 49002

def main():
    sock = socket(AF_INET, SOCK_DGRAM)
    sock.bind((ADAPTER, ECHO_SERVER_PORT))

    data, addr = sock.recvfrom(1500)
    sock.sendto(data, addr)
    
    print(f"{addr = }, data = {data.decode()}")

    sock.close()

if __name__ == "__main__":
    main()