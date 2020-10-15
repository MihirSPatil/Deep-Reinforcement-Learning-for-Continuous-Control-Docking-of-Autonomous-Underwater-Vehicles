#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import gym
from deepleng_gym.task_envs.deepleng import deepleng_docking
from stable_baselines.bench import Monitor
from stable_baselines import DDPG
from stable_baselines.common.evaluation import evaluate_policy

from stable_baselines.common.env_checker import check_env



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
        # model = DDPG.load("/home/dfki.uni-bremen.de/mpatil/Documents/lander_stable_baselines")
        model = DDPG.load("/home/dfki.uni-bremen.de/mpatil/cluster_logs/saved_models/ddpg_stable_baselines_128_7_net")
        mean_reward, std_reward = evaluate_policy(model,
                                                  self.env,
                                                  n_eval_episodes=10,
                                                  deterministic=True,
                                                  return_episode_rewards=True)
        print("Mean Reward: {}, Std. reward".format(mean_reward, std_reward))

        print("Enjoy the trained agent")

        # for ep in range(5):
        #     print('ep:', ep)
        #     eps = {}
        #     xs = []
        #     ys = []
        #     pitches = []
        #     yaws = []
        #     x_lin = []
        #     y_lin = []
        #     z_lin = []
        #     x_ang = []
        #     y_ang = []
        #     z_ang = []
        #     obs = self.env.reset()
        #     for steps in range(50):
        #         action, _states = model.predict(obs, deterministic=True)
        #         # print("action:", action)
        #         obs, rewards, done, info = self.env.step(action)
        #         xs.append(obs[0])
        #         ys.append(obs[1])
        #         pitches.append(obs[2])
        #         yaws.append(obs[3])
        #         x_lin.append(obs[4])
        #         y_lin.append(obs[5])
        #         z_lin.append(obs[6])
        #         x_ang.append(obs[7])
        #         y_ang.append(obs[8])
        #         z_ang.append(obs[9])
        #         # print("Observation:", obs)
        #         if done:
        #             print('done:', done)
        #             eps[ep] = [xs, ys, pitches, yaws, x_lin, y_lin, z_lin, x_ang, y_ang, z_ang]
        #             print(eps)
        #             break
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
