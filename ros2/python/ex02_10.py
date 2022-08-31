"""ex02_10.py"""
import random
from datetime import datetime

class Die:
    SIDES= 6

    def __init__( self ):
        random.seed(datetime.now())
        self.roll()

    def roll( self ):
        self.value = random.randrange(self.SIDES) + 1
        return self.value

    @property
    def value(self):
        return self._n
    
    @value.setter
    def value(self, n):
        self._n = n

class Dice:
    def __init__( self ):
        self._dice = (Die(), Die())

    def roll(self):
        for d in self._dice:
            d.roll()

    @property
    def total(self):
        return self._dice[0].value + self._dice[1].value

    @property
    def pair( self ):
        return self._dice 

class CrapsDice(Dice):
    def hardways(self):
        return self._dice[0].value == self._dice[1].value

    def isPoint(self, value):
        return self.total == value

def main():
    craps = CrapsDice()

    point = int(input("Enter point number: "))

    for i in range(10):
        print(f"*** {i} ***")
        craps.roll()      
        if craps.total in [3, 7, 12]:
            print(">>> craps <<<")
            break

        if craps.isPoint(point):
            print("<><><>point<><><>")
            if craps.hardways():
                print("=-=-=hardways=-=-=")
            break

if __name__ == '__main__':
    main()