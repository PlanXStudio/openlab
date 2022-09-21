"""ex03_03_1.py"""
from socket import socket, AF_PACKET, SOCK_RAW, htons

def send_link(dst, src, payload, type="0800", interface="eth0"):
    s = socket(AF_PACKET, SOCK_RAW, htons(3))
    s.bind((interface, 3))

    frame = bytearray.fromhex(dst + src + type) + payload.encode()

    return s.send(frame)

def main():
    dst = "00155d11c9ca"
    src = "00155d11c9ca"
    payload = input("payload: ")

    send_link(dst, src, payload)

if __name__ == "__main__":
    main()