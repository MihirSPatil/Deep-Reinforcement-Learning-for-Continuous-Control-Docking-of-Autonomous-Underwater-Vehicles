#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import gym
from openai_ros.task_envs.deepleng import deepleng_docking
import torch
from spinup import ppo_pytorch as ppo

class SpinUpPpo():
    '''spinning-up openai PPO'''
    def __init__(self):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('deepleng_control')
        self.outdir = pkg_path + '/spinup_logs/'

        self.env = lambda: gym.make('DeeplengDocking-v1')

        # env = Monitor(env, outdir)
        # self.env.seed(1)

    def __call__(self, *args, **kwargs):

        ac_kwargs = dict(hidden_sizes=[400, 300, 300, 200, 100], activation=torch.nn.ReLU)

        logger_kwargs = dict(output_dir=self.outdir, exp_name='SU_ppo_docker')

        ppo(env_fn=self.env,
            ac_kwargs=ac_kwargs,
            steps_per_epoch=5000,
            epochs=250,
            logger_kwargs=logger_kwargs)
def main():
    rospy.init_node('SpinUpPpo_docker', anonymous=True)
    train = SpinUpPpo()
    train()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")

if __name__ == '__main__':
    main()
