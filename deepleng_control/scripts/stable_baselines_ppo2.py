#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import os
import gym
from deepleng_gym.task_envs.deepleng import deepleng_docking
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

    def __init__(self, expt_name):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('deepleng_control')
        outdir = pkg_path + '/monitor_logs/' + expt_name

        # env = gym.make('LunarLanderContinuous-v2')
        env = gym.make('DeeplengDocking-v2')
        self.expt_name = expt_name
        self.env = Monitor(self.env, outdir)

    def __call__(self, *args, **kwargs):
        # eval_callback = EvalCallback(env, best_model_save_path=eval_dir,
        #                              log_path=eval_dir, eval_freq=500,
        #                              deterministic=True, render=False)
        policy_kwargs = dict(layers=[400, 300, 200, 100])
        model = PPO2(MlpPolicy,
                     self.env,
                     policy_kwargs=policy_kwargs,
                     verbose=1,
                     tensorboard_log="home/dfki.uni-bremen.de/mpatil/Documents/baselines_log")

        model.learn(total_timesteps=int(1e5),
                    log_interval=50,
                    tb_log_name="ppo_Docker_" + self.expt_name)

        model.save("/home/dfki.uni-bremen.de/mpatil/Documents/ppo_stable_baselines_" + self.expt_name)

        # del model

        print("Closing environment")
        self.env.close()


def main():
    rospy.init_node('SbPpo2_docker', anonymous=True)
    expt_name = os.environ["expt_name"]
    train = SbPpo2(expt_name)
    train()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")


if __name__ == '__main__':
    main()



