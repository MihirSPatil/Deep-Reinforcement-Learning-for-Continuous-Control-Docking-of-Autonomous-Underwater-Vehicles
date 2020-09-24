#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import gym
from openai_ros.task_envs.deepleng import deepleng_docking
from stable_baselines.bench import Monitor
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
# from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.callbacks import BaseCallback

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
class SbPpo2():
    '''stable baselines PPO2'''
    def __init__(self):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('deepleng_control')
        self.outdir = pkg_path + '/saved_models/'

        # env = gym.make('LunarLanderContinuous-v2')
        self.env = gym.make('DeeplengDocking-v1')

        # env = Monitor(env, outdir)
        self.env.seed(1)

    def __call__(self, *args, **kwargs):

        # eval_callback = EvalCallback(env, best_model_save_path=eval_dir,
        #                              log_path=eval_dir, eval_freq=500,
        #                              deterministic=True, render=False)

        model = PPO2(MlpPolicy,
                     self.env,
                     n_steps=1024,
                     nminibatches=32,
                     verbose=1,
                     lam=0.98,
                     gamma=0.999,
                     noptepochs=4,
                     ent_coef=0.01,
                     tensorboard_log="/home/dfki.uni-bremen.de/mpatil/Documents/baselines_log",
                     seed=1)

        model.learn(total_timesteps=int(1e6), log_interval=50, tb_log_name="ppo_Docker")

        model.save(self.outdir + "ppo_deepleng")
        # model.save("/home/dfki.uni-bremen.de/mpatil/Desktop/ppo_LunarLander")
        # del model

        print("Closing environment")
        self.env.close()

def main():
    rospy.init_node('SbPpo2_docker', anonymous=True)
    train = SbPpo2()
    train()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")

if __name__ == '__main__':
    main()



