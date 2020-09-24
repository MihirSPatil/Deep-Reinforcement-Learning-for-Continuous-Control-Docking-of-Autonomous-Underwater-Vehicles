#!/usr/bin/env python3

import rlpyt

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
import rospy
from rlpyt.envs.gym import *
import gym
from openai_ros.task_envs.deepleng import deepleng_docking


def build_and_train(env_id="DeeplengDocking-v1", run_ID=0, cuda_idx=None):
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        eval_env_kwargs=dict(id=env_id),
        batch_T=1,  # One time-step per sampler iteration.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(51e3),
        eval_max_trajectories=50,
    )
    algo = SAC()  # Run with defaults.
    agent = SacAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=int(1e6),
        log_interval_steps=int(1e4),
        affinity=dict(cuda_idx=cuda_idx),
    )
    config = dict(env_id=env_id)
    name = "sac_" + env_id
    log_dir = "example_2"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":

    rospy.init_node('rlpyt_docker', anonymous=True)
    build_and_train()
    # print("imported rlpyt")
    # # env = make('DeeplengDocking-v1') #makes the environment with the rlpyt wrapper
    # # print("imported garage")
    # env = gym.make('DeeplengDocking-v1')
    # # env = GymEnvWrapper(env)
    # print("Gym environment done")
