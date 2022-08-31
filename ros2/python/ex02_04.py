"""ex02_04.py"""
import random
import math

def creat_data(n):
    return [random.normalvariate(0.3, 1.2) for _ in range(n)]

def calc_mean_val_std(data):
    mean = sum(data) / len(data)
    vsum  = 0
    for d in data:
        vsum = vsum + (d - mean)**2
    variance = vsum / len(data)
    std = math.sqrt(variance)
    
    return (mean, variance, std)

def main():
    data = creat_data(1000000)
    mean, ver, std = calc_mean_val_std(data)
    print(f"|{mean = }|{ver = :>30}|{std = :^10.2f}|")

if __name__ == "__main__":
    main()
