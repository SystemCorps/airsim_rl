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

    def __init__(self, actions):
        self.ac_space = gym.spaces.Box(low=np.array([actions[0][0], actions[1][0], actions[2][0]]),
                                       high=np.array([actions[0][1], actions[1][1], actions[2][2]]),
                                       dtype=np.float32)
        
        