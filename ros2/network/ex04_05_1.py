"""ex04_05_1.py"""
from socket import socket, inet_aton, AF_INET, SOCK_DGRAM, SOL_SOCKET, SO_REUSEADDR, INADDR_ANY, IPPROTO_IP, IP_ADD_MEMBERSHIP
import struct

MCAST_GROUP = '224.1.1.1'
SERVER_ADDR = ('', 49003)

def main():
    sock = socket(AF_INET, SOCK_DGRAM)

    sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    sock.bind(SERVER_ADDR)

    mreq = struct.pack("4sL", inet_aton(MCAST_GROUP), INADDR_ANY)
    sock.setsockopt(IPPROTO_IP, IP_ADD_MEMBERSHIP, mreq)

    while True:
        data, addr = sock.recvfrom(10240)
        print(f"{addr =}, {data = }")

        sock.sendto("ack".encode(), addr)
        
if __name__ == "__main__":
    main()