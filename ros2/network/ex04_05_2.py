"""ex04_05_2.py"""
from socket import socket, timeout, AF_INET, SOCK_DGRAM, IP_MULTICAST_TTL, IPPROTO_IP
import struct

MCAST_GROUP = ('224.1.1.1', 49003)

def main():
    sock = socket(AF_INET, SOCK_DGRAM)
    sock.settimeout(0.2)
    
    ttl = struct.pack('b', 2)
    sock.setsockopt(IPPROTO_IP, IP_MULTICAST_TTL, ttl)

    data = "echo data".encode()
    sock.sendto(data, MCAST_GROUP)

    while True:
        try:
            data, addr = sock.recvfrom(32)
        except timeout:
            break
        else:
            print(f"{addr =}, {data = }")
    
    sock.close()

if __name__ == "__main__":
    main()