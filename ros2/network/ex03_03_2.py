"""ex03_03_2.py"""
from socket import socket, AF_PACKET, SOCK_RAW, htons

def ba2hs(ar, fmt=''):
    return fmt.join(f"{b:02x}" for b in ar)

def recv_link(interface="eth0"):
    s = socket(AF_PACKET, SOCK_RAW, htons(3))
    s.bind((interface, 0))
    
    frame = s.recv(1516)

    dst = ba2hs(frame[:6], ':')
    src = ba2hs(frame[6:12], ':')
    type = ba2hs(frame[12:14]) 

    try:
        payload = frame[14:].decode()
    except UnicodeDecodeError:
        return

    print(f"{dst = }")
    print(f"{src = }")
    print(f"{type = }")        
    print(f"{payload = }")

def main():
    while True:
        try:
            recv_link()
        except KeyboardInterrupt:
            break
        
if __name__ == "__main__":
    main()