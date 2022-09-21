"""ex04_06_2.py"""
from simple_multicast import SimpleMulticastSender

def main():   
    with SimpleMulticastSender() as msnd:
        msnd.send("echo data".encode())

        while True:
            try:
                data, addr = msnd.recv()
                if data:
                    print(f"{addr =}, {data = }")
            except KeyboardInterrupt:
                break
            
if __name__ == "__main__":
    main()