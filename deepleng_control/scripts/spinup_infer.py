#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import gym
import os
from deepleng_gym.task_envs.deepleng import deepleng_docking
import torch
from spinup.utils.test_policy import load_pytorch_policy, run_policy


class SpInfer():
    '''stable baselines Inference script'''
    def __init__(self):
        # rospack = rospkg.RosPack()
        # pkg_path = rospack.get_path('deepleng_control')
        # self.outdir = pkg_path + '/saved_models/'

        self.env = gym.make('DeeplengDocking-v2')

        # env = Monitor(env, outdir)
    def __call__(self, *args, **kwargs):

        # get_action = load_pytorch_policy(model_path, '', deterministic=False)
        # run_policy(self.env, get_action, num_episodes=10, render=False)

        ac = torch.load('/home/dfki.uni-bremen.de/mpatil/Downloads/spinup_logs/pyt_save/model.pt')


        print("Enjoy the trained agent")

        for ep in range(10):
            print('ep:', ep)
            eps = {}
            xs = []
            ys = []
            pitches = []
            yaws = []
            x_lin = []
            y_lin = []
            z_lin = []
            x_ang = []
            y_ang = []
            z_ang = []
            obs = self.env.reset()
            for steps in range(50):
                action = ac.act(torch.as_tensor(obs, dtype=torch.float32))
                # print("action:", action)
                obs, rewards, done, info = self.env.step(action)
                xs.append(obs[0])
                ys.append(obs[1])
                pitches.append(obs[2])
                yaws.append(obs[3])
                x_lin.append(obs[4])
                y_lin.append(obs[5])
                z_lin.append(obs[6])
                x_ang.append(obs[7])
                y_ang.append(obs[8])
                z_ang.append(obs[9])
                # print("Observation:", obs)
                if done:
                    print('done:', done)
                    # eps[ep] = [xs, ys, pitches, yaws, x_lin, y_lin, z_lin, x_ang, y_ang, z_ang]
                    # print(eps)
                    break
        self.env.close()

def main():
    rospy.init_node('SpInfer', anonymous=True)
    infer = SpInfer()
    infer()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")

if __name__ == '__main__':
    main()
