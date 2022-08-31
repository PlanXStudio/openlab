"""ex02_09.py"""
class Automotive:
    def __init__(self):
        self.name = "Automotive"
        self.stop()

    def forward(self):
        self.state = 1
        print("forward")

    def backward(self):
        self.state = -1
        print("backward")

    def trun(angle):
        print(f"call trun {angle}")

    def stop(self):
        self.state = 0

    def getName(self):
        return self.name

class Truck(Automotive):
    def __init__(self):
        super(Truck, self).__init__()
        self.name = "Truck"
    
    def trun(self, angle):
        assert(-1 <= angle <= 1)
        Automotive.trun(angle)

def main():
    truck = Truck()
    print(truck.getName())
    truck.forward()
    truck.trun(-0.2)
    truck.backward()    
    del truck
    
if __name__ == "__main__":
    main()