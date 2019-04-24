import airsim
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import math

from baselines import logger
from baselines import bench
from baselines.common import set_global_seeds
from mpi4py import MPI
import os.path as osp


class DronePPO:
    def __init__(self, init_mean, init_std, num_timesteps, seed):

        self.init_mean = init_mean
        self.init_std = init_std
        self.num_timesteps = num_timesteps
        self.seed = seed

        self.client = None
        self.init_pose = None       # Initial pose for each episode
        self.linSpeed = None
        self.ready = False

        self.airsimConnect()


    def airsimConnect(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Takeoff
        self.client.takeoffAsync().join()


    def initPoseRnd(self):
        self.init_pose = np.random.normal(self.init_mean, self.init_std)


    def calLinSpeed(self):
        state = self.client.getMultirotorState()

        vx = state.kinematics_estimated.linear_velocity.x_val
        vy = state.kinematics_estimated.linear_velocity.y_val
        vz = state.kinematics_estimated.linear_velocity.z_val
        linSpeed = (vx**2 + vy**2 + vz**2)**0.5

        return linSpeed


    def mvToInitPose(self):
        self.initPoseRnd()
        # To the initial position and yaw (heading)
        self.client.moveToPositionAsync(self.init_pose[0],
                                        self.init_pose[1],
                                        self.init_pose[2])
        self.client.rotateToYawAsync(self.init_pose[3])

        linSpeed = self.calLinSpeed()

        while(linSpeed < 0.1):
            linSpeed = self.calLinSpeed()
            time.sleep(0.03)

        self.ready = True


    def train(self):
        from baselines.ppo1 import pposgd_simple, cnn_policy
        import baselines.common.tf_util as U
        rank = MPI.COMM_WORLD.Get_rank()
        sess = U.single_threaded_session()
        sess.__enter__()
        if rank == 0:
            logger.configure()
        else:
            logger.configure(format_strs=[])
        workerseed = self.seed + 10000 * MPI.COMM_WORLD.Get_rank() if self.seed is not None else None
        set_global_seeds(workerseed)
        def policy_fn(name, ob_space, ac_space):
            return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
        



        env = bench.Monitor(env, logger.get_dir() and
                            osp.join(logger.get_dir(), str(rank)))











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
    init_mean = [x_mean, y_mean, z_mean, yaw_mean]

    x_std = 1
    y_std = 1
    z_std = 0.3
    yaw_std = math.pi/2
    init_std = [x_std, y_std, z_std, yaw_std]

    num_timesteps = 0
    seed = 0






if __name__ == '__main__':
    main()