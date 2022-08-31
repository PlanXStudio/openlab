"""ex03_08_1.py"""
from simple_multicast import SimpleMulticastReceiver

def main():
    with SimpleMulticastReceiver() as mrecv:
        while True:
            try:
                data, addr = mrecv.recv()
                if data:
                    print(f"{addr =}, {data = }")
                    mrecv.send("ack".encode(), addr)
            except KeyboardInterrupt:
                break

if __name__ == "__main__":
    main()