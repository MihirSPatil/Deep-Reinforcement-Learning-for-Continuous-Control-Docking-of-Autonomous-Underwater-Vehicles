#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import gym
from openai_ros.task_envs.deepleng import deepleng_docking
from stable_baselines.bench import Monitor
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.callbacks import BaseCallback


class SbInfer():
    '''stable baselines Inference script'''
    def __init__(self):
        # rospack = rospkg.RosPack()
        # pkg_path = rospack.get_path('deepleng_control')
        # self.outdir = pkg_path + '/saved_models/'

        self.env = gym.make('DeeplengDocking-v1')

        # env = Monitor(env, outdir)
    def __call__(self, *args, **kwargs):
        # model = PPO2.load(self.outdir + "ppo_deepleng")
        model = DDPG.load("/home/dfki.uni-bremen.de/mpatil/Documents/ddpg_stable_baselines")
        # mean_reward, std_reward = evaluate_policy(model, self.env, n_eval_episodes=10)
        # print("Mean Reward: {}, Std. reward".format(mean_reward, std_reward))

        print("Enjoy the trained agent")
        obs = self.env.reset()
        for i in range(1000):
            action, _states = model.predict(obs)
            # print("action:", action)
            obs, rewards, dones, info = self.env.step(action)
            # if dones:
            #     break

        self.env.close()

def main():
    rospy.init_node('SbInfer', anonymous=True)
    infer = SbInfer()
    infer()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")

if __name__ == '__main__':
    main()
