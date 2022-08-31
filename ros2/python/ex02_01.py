"""ex02_01.py"""
import numpy as np
import matplotlib.pyplot as plt 

def show(x, y):
    plt.plot(x, y)
    plt.show()

def get_sin_data(r, n):
    x = np.linspace(0, r*np.pi, n)
    y = np.sin(x)
    return (x, y)

def main():
    x, y = get_sin_data(5, 100)
    show(x, y)

if __name__ == '__main__':
    main()
