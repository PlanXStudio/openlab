"""ex02_06_3.py"""
from sys import path
from ex02_06_1 import foo

def boo():
    print(f"__name__ is {__name__}")

def main():
    boo()
    foo()
    print(path)

if __name__ == "__main__":
    main()