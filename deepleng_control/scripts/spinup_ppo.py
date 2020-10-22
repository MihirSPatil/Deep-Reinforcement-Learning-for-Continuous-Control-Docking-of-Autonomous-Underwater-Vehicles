#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import gym
import os
from deepleng_gym.task_envs.deepleng import deepleng_docking
import torch
from spinup import ppo_pytorch as ppo

class SpinUpPpo():
    '''spinning-up openai PPO'''
    def __init__(self, expt_name):

        self.outdir = str(os.path.expanduser('~')) + "/" + expt_name
        try:
            os.makedirs(path)
            print("Directory ", path, " Created ")
        except FileExistsError:
            print("Directory ", path, " already exists")

        self.expt_name = expt_name

        self.env = lambda: gym.make('DeeplengDocking-v2')

        # self.env = wrappers.Monitor(self.env, self.outdir, force=True)
        # self.env.seed(1)

    def __call__(self, *args, **kwargs):

        ac_kwargs = dict(hidden_sizes=[400, 300, 200, 100], activation=torch.nn.ReLU)

        logger_kwargs = dict(output_dir=self.outdir, exp_name=self.expt_name)

        ppo(env_fn=self.env,
            ac_kwargs=ac_kwargs,
            steps_per_epoch=250,
            epochs=400,
            logger_kwargs=logger_kwargs)
def main():
    rospy.init_node('SpinUpPpo_docker', anonymous=True)
    train = SpinUpPpo(os.environ["expt_name"])
    train()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")

if __name__ == '__main__':
    main()
