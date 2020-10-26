#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import os
import gym
from stable_baselines.common.env_checker import check_env
from deepleng_gym.task_envs.deepleng import deepleng_docking
from stable_baselines.bench import Monitor
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines import SAC
from stable_baselines.common.env_checker import check_env


class SbSac():
    '''stable baselines SAC'''

    def __init__(self, expt_name):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('deepleng_control')
        outdir = pkg_path + '/monitor_logs/' + expt_name

        # env = gym.make('LunarLanderContinuous-v2')
        env = gym.make('DeeplengDocking-v2')
        self.expt_name = expt_name
        self.env = Monitor(env, outdir)

    def __call__(self):

        policy_kwargs = dict(layers=[400, 300, 200, 100])

        # check_env(self.env)
        model = TD3(MlpPolicy,
                     self.env,
                     policy_kwargs=policy_kwargs,
                     tensorboard_log="/home/dfki.uni-bremen.de/mpatil/Documents/baselines_log",
                     verbose=1)

        time_steps = 3e4
        model.learn(total_timesteps=int(time_steps),
                    log_interval=50,
                    tb_log_name="sac_Docker_" + self.expt_name)
        model.save("/home/dfki.uni-bremen.de/mpatil/Documents/sac_stable_baselines_" + self.expt_name)

        print("Closing environment")
        self.env.close()


def main():
    rospy.init_node('SbSac_docker', anonymous=True)
    expt_name = os.environ["expt_name"]
    train = SbSac(expt_name)
    train()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")


if __name__ == '__main__':
    main()
