"""ex02_12.py"""
class Automotive:
    def __init__(self):
        from stream_logger import StreamLogger
        self.logger = StreamLogger()

    def get_logger(self):
        return self.logger

class Truck(Automotive):
    def foo(self):
        self.get_logger().debug("it's debug message")
    
    def boo(self):
        self.get_logger().error("it's error message")
    
def main():
    truck = Truck()
    truck.foo()
    truck.boo()

if __name__ == '__main__':
    main()