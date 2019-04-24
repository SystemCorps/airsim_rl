import airsim
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import math


class dronePPO:
    def __init__(self, init_pose, init_std, num_timesteps, seed):

        self.init_pose = init_pose
        self.init_std = init_std
        self.num_timesteps = num_timesteps
        self.seed = seed

        self.client = None

        self.airsim_connect()



    def airsim_connect(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Takeoff
        self.client.takeoffAsync().join()




def main():
    # Connecting AirSim-UE
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    client.takeoffAsync().join()

    # initial pose after takeoff - mean values and std for randomization using Gaussian Distribution
    # 4-DOF (x, y, z, yaw)
    x_mean = 0
    y_mean = 0
    z_mean = 1.5
    yaw_mean =  0
    init_pose = [x_mean, y_mean, z_mean, yaw_mean]

    x_std = 1
    y_std = 1
    z_std = 0.3
    yaw_std = math.pi
    init_std = [x_std, y_std, z_std, yaw_std]

    num_timesteps = 0
    seed = 0






if __name__ == '__main__':
    main()