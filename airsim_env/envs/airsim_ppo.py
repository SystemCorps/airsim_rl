import airsim
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Tuple, Box, Discrete, MultiDiscrete, Dict
from gym.spaces.box import Box
import numpy as np
import logging
import math

from airsim_env.envs.airsim_client import *

# Ref https://github.com/fshamshirdar/gym-airsim

# Baselines PPO requires below list from simulation environment
# action space (Box), observation space(Box), observation (Img), reward, is new
# "is new" - if collision is caused -> reset simulation to initial state

logger = logging.getLogger(__name__)


class AirSimPPO():
    metadata = {'render.modes': ['human']}
    pitch_l = -20.0
    pitch_u = 20.0
    roll_l = -1.0
    roll_u = 1.0
    throttle_l = 0.0
    throttle_u = 100.0
    yaw_l = -180.0
    yaw_u = 180.0
    duration = 0.1  # 10 Hz update rate
    actions = [[pitch_l, pitch_u],
               [roll_l, roll_u],
               [throttle_l, throttle_u],
               [yaw_l, yaw_u],
               [duration, duration]]

    def __init__(self, action_range=actions):
        # action_range without odometry : pitch, roll, throttle, yaw_rate, duration
        self.action_space = Box(low=np.array([action_range[0][0], action_range[1][0],
                                                     action_range[2][0], action_range[3][0],
                                                     action_range[4][0]]),
                                       high=np.array([action_range[0][1], action_range[1][1],
                                                      action_range[2][1], action_range[3][1],
                                                      action_range[4][1]]),
                                       dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self.state = np.zeros((84, 84, 1), dtype=np.uint8)
        self._seed()

        x_mean = -8
        y_mean = -11
        z_mean = -3
        yaw_mean = 0
        init_mean = [x_mean, y_mean, z_mean, yaw_mean]

        x_std = 1
        y_std = 1
        z_std = 0.3
        yaw_std = 30
        init_std = [x_std, y_std, z_std, yaw_std]


        self.sim= AirSimClient(init_mean, init_std)
        self.sim.mvToInitPose()

        self.goal = [30, 20, -2.0]
        self.dist_before = 100
        self.steps = 0
        self.no_episode = 0
        self.reward_sum = 0


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reward_cal(self, goal):
        dist_goal = np.sqrt(np.power((self.goal[0]-goal.x_val), 2) +
                           np.power((self.goal[1] - goal.y_val), 2) +
                           np.power((self.goal[2] - goal.z_val), 2))

        r = -1
        r = r + (self.dist_before - dist_goal)

        return r, dist_goal


    def goal_dir(self, goal, pos):
        pitch, roll, yaw = self.toEulerianAngle(self.sim.client.simGetGroundTruthKinematics().orientation)
        yaw = math.degrees(yaw)
        pos_angle = math.atan2(goal[1] - pos.y_val, goal[0] - pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360
        goal = math.radians(pos_angle - yaw)

        return ((math.degrees(goal) - 180) % 360) - 180


    # Input: Action, Output: Observation, Reward, New episode
    def _step(self, action):
        self.steps += 1
        collided = self.sim.exec_action(action)
        position = self.sim.client.simGetGroundTruthKinematics().position

        if collided == True:
            done = True
            reward = -100.0
            dist = np.sqrt(np.power(self.goal[0] - position.x_val, 2) +
                           np.power(self.goal[1] - position.y_val, 2) +
                           np.power(self.goal[2] - position.z_val, 2))

        else:
            done = False
            reward, dist = self.reward_cal(position)

        if dist < 1:
            done = True
            reward = 100.0

        self.dist_before = dist
        self.reward_sum += reward

        if self.reward_sum < -100:
            done = True

        info = {"x_pos":position.x_val, "y_pos":position.y_val, "z_pos":position.z_val}
        self.state = self.sim.getDroneCam()

        return self.state, reward, done, info


    def _reset(self):
        self.sim.simReset()
        self.steps = 0
        self.reward_sum = 0
        self.no_episode += 1

        position = self.sim.client.simGetGroundTruthKinematics().position
        goal = self.goal_dir(self.goal, position)
        self.state = self.sim.getDroneCam()

        return self.state


    def _render(self, mode='human', close=False):
        return



    @staticmethod
    def toEulerianAngle(self, q):
        z = q.z_val
        y = q.y_val
        x = q.x_val
        w = q.w_val
        ysqr = y * y

        # roll (x-axis rotation)
        t0 = +2.0 * (w*x + y*z)
        t1 = +1.0 - 2.0*(x*x + ysqr)
        roll = math.atan2(t0, t1)

        # pitch (y-axis rotation)
        t2 = +2.0 * (w*y - z*x)
        if (t2 > 1.0):
            t2 = 1
        if (t2 < -1.0):
            t2 = -1.0
        pitch = math.asin(t2)

        # yaw (z-axis rotation)
        t3 = +2.0 * (w*z + x*y)
        t4 = +1.0 - 2.0 * (ysqr + z*z)
        yaw = math.atan2(t3, t4)

        return (pitch, roll, yaw)