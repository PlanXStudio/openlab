"""ex02_02.py"""
import numpy as np

def creat_data(n):
    return np.random.standard_normal(n)

def calc_mean_val_std(data):
    return (np.mean(data), np.var(data), np.std(data))

def main():
    data = creat_data(1000000)
    mean, var, std = calc_mean_val_std(data)
    print(f"|{mean = }|{var = :>30}|{std = :^10.2f}|")

if __name__ == "__main__":
    main()