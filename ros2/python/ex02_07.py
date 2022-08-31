"""ex02_07.py"""
class Account:
    def __init__( self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __iter__(self):
        return iter(self.name)

from itertools import zip_longest

def main():
    acc1 = Account("Python")
    acc2 = Account("ROS2")

    print(acc1, acc2)
    for c in zip_longest(acc1, acc2, fillvalue=''):
        print(c[0], c[1])

if __name__ == '__main__':
    main()
