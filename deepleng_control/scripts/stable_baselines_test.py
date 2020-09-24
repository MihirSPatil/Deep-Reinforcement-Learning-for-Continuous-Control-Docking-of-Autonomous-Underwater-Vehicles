#!/usr/bin/env python3

import rospy

import gym
from openai_ros.task_envs.deepleng import deepleng_docking
from stable_baselines.common.env_checker import check_env

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines import PPO2

# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])


if __name__ == "__main__":

    rospy.init_node('stable_baselines_docker', anonymous=True)
    env = gym.make('DeeplengDocking-v1')
    # obs = env.reset()
    # print("Observation: {}".format(obs))
    # print("Observation type: {}".format(type(obs)))
    # print("Observation space: {}".format(env.observation_space))
    # obs_new, _, _, _ = env.step()
    # print("new_observation: {}".format(obs_new))
    # check_env(env)

    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="/home/dfki.uni-bremen.de/mpatil/Documents")
    model.learn(total_timesteps=1000)
    #model.save("/home/dfki.uni-bremen.de/mpatil/Desktop/ddpg_deepleng")
    #del model

    # model = PPO2.load("/home/dfki.uni-bremen.de/mpatil/Desktop/ppo2_deepleng")
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    # print("Mean Reward: {}, Std. reward".format(mean_reward, std_reward))
    #
    print("Enjoy the trained agent")
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        # print("action:", action)
        obs, rewards, dones, info = env.step(action)
