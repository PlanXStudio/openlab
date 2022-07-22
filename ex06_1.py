from pop import Pilot
from pop import LiDAR
from threading import Thread
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2

def Lidar():
    lidar = LiDAR.Rplidar()
    lidar.connect()
    lidar.startMotor()
    
    def close():
        lidar.stopMotor()

    Lidar.close = close

    def _inner():
        return np.array(lidar.getVectors())
    
    return _inner


def on_lidar(car, lidar):
    on_lidar.is_stop = False

    DIST = 40 * 10 #unit mm
    WIDTH, HEIGHT = 1280, 720

    fig = plt.figure(figsize=(WIDTH/100, HEIGHT/100), dpi=100)
    ax = fig.add_subplot(111, projection='polar')
    fig.tight_layout()

    while not on_lidar.is_stop:
        V = lidar()
        
        angle = np.radians(V[:,0])
        dist = V[:,1]
        dist = np.where(dist <= DIST, dist, DIST)

        ax.plot(angle, dist, color='pink', linestyle='--', linewidth=1.0, marker='o', markeredgecolor='red', markersize=2.0, alpha=0.6)
        fig.canvas.draw()

        cv2.imshow("lidar", cv2.cvtColor(np.array(fig.canvas.renderer._renderer), cv2.COLOR_RGB2BGR))
        plt.cla()
        ax.set_theta_zero_location("N")

        if cv2.waitKey(10) == 27:
            break


def main():  
    car = Pilot.AutoCar()
    lidar = Lidar()

    t = Thread(target=on_lidar, args=(car, lidar))
    t.daemon = True
    t.start()

    input()

    on_lidar.is_stop = True
    Lidar.close()

if __name__ == "__main__":
    main()