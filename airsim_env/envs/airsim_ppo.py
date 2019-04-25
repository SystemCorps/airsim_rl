
import airsim
import gym
import numpy as np
import logging

# Ref https://github.com/fshamshirdar/gym-airsim

# Baselines PPO requires below list from simulation environment
# action space (Box), observation space(Box), observation (Img), reward, is new
# "is new" - if collision is caused -> reset simulation to initial state

logger = logging.getLogger(__name__)

class AirSimPPO():
    metadata = {'render.modes': ['human']}

    def __init__(self, action_range):
        # action_range without odometry : pitch, roll, throttle, yaw_rate, duration
        self.ac_space = gym.spaces.Box(low=np.array([action_range[0][0], action_range[1][0],
                                                     action_range[2][0], action_range[3][0],
                                                     action_range[4][0]]),
                                       high=np.array([action_range[0][1], action_range[1][1],
                                                      action_range[2][1], action_range[3][1],
                                                      action_range[4][1]]),
                                       dtype=np.float32)
        self.ob_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        self._seed()
        
        
