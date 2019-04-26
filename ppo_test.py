
#!/usr/bin/env python3

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import airsim_env
import math
import gym


def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import pposgd_simple, cnn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank() if seed is not None else None
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space): #pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)

    env = wrap_deepmind(env)
    env.seed(workerseed)

    pposgd_simple.learn(env, policy_fn,
        max_timesteps=int(num_timesteps * 1.1),
        timesteps_per_actorbatch=256,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64,
        gamma=0.99, lam=0.95,
        schedule='linear'
    )




def main():

    env_name = 'AirSimPPO-v0'
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

    # Action space lower and upper bounds
    # in degree
    pitch_l = -20.0
    pitch_u = 20.0
    roll_l = -1.0
    roll_u = 1.0
    yaw_l = -180.0
    yaw_u = 180.0
    duration = 0.1      # 10 Hz update rate
    actions = [[pitch_l, pitch_u],
               [roll_l, roll_u],
               [yaw_l, yaw_u],
               [duration, duration]]


    num_timesteps = 1000
    seed = 0

    train(env_name, num_timesteps, seed)



if __name__ == '__main__':
    main()