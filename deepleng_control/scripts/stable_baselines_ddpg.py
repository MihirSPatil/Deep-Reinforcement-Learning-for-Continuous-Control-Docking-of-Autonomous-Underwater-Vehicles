#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import gym
from openai_ros.task_envs.deepleng import deepleng_docking
from stable_baselines.bench import Monitor
from stable_baselines import DDPG
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise, NormalActionNoise
from stable_baselines.common.env_checker import check_env
from stable_baselines.common.callbacks import BaseCallback


# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
class SbDdpg():
    '''stable baselines Ddpg'''

    def __init__(self):
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('deepleng_control')
        self.outdir = pkg_path + '/saved_models/'

        # env = gym.make('LunarLanderContinuous-v2')
        self.env = gym.make('DeeplengDocking-v1')

        # env = Monitor(env, outdir)

    def __call__(self, *args, **kwargs):
        # eval_callback = EvalCallback(env, best_model_save_path=eval_dir,
        #                              log_path=eval_dir, eval_freq=500,
        #                              deterministic=True, render=False)
        policy_kwargs = dict(layers=[400, 300, 200, 100])
        n_actions = self.env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

        model = DDPG(MlpPolicy,
                     self.env,
                     policy_kwargs=policy_kwargs,
                     param_noise=param_noise,
                     action_noise=action_noise,
                     seed=1,
                     tensorboard_log="/home/dfki.uni-bremen.de/mpatil/Documents/baselines_log",
                     verbose=1)

        time_steps = 1e4
        model.learn(total_timesteps=int(time_steps), log_interval=50, tb_log_name="ddpg_Docker")
        model.save("/home/dfki.uni-bremen.de/mpatil/Documents/ddpg_stable_baselines")

        # print("Enjoy the trained agent")
        # obs = env.reset()
        # for i in range(10000):
        #     action, _states = model.predict(obs)
        #     # print("action:", action)
        #     obs, rewards, dones, info = env.step(action)
        #     env.render()
        #     if dones:
        #         obs = env.reset()
        print("Closing environment")
        self.env.close()


def main():
    rospy.init_node('SbDdpg_docker', anonymous=True)
    train = SbDdpg()
    train()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")


if __name__ == '__main__':
    main()
