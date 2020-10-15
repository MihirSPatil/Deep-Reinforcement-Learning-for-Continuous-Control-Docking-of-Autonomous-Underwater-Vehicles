#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import os
import gym
from stable_baselines.common.env_checker import check_env
from deepleng_gym.task_envs.deepleng import deepleng_docking
from stable_baselines.bench import Monitor
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines.common.env_checker import check_env


# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
class SbDdpg():
    '''stable baselines Ddpg'''

    def __init__(self, expt_name):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('deepleng_control')
        outdir = pkg_path + '/monitor_logs/' + expt_name

        # self.env = gym.make('LunarLanderContinuous-v2')
        env = gym.make('DeeplengDocking-v2')
        self.expt_name = expt_name
        self.env = Monitor(env, outdir)

    def __call__(self):

        policy_kwargs = dict(layers=[400, 300, 200, 100])
        n_actions = self.env.action_space.shape[-1]
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.2) * np.ones(n_actions))

        # check_env(self.env)
        model = DDPG(MlpPolicy,
                     self.env,
                     policy_kwargs=policy_kwargs,
                     action_noise=action_noise,
                     memory_limit=50000,
                     tensorboard_log="/home/dfki.uni-bremen.de/mpatil/Documents/baselines_log",
                     verbose=1)

        time_steps = 1e4
        model.learn(total_timesteps=int(time_steps),
                    log_interval=50,
                    tb_log_name="ddpg_Docker_" + self.expt_name)
        model.save("/home/dfki.uni-bremen.de/mpatil/Documents/ddpg_stable_baselines_" + self.expt_name)

        print("Closing environment")
        self.env.close()


def main():
    rospy.init_node('SbDdpg_docker', anonymous=True)
    expt_name = os.environ["expt_name"]
    train = SbDdpg(expt_name)
    train()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")


if __name__ == '__main__':
    main()
