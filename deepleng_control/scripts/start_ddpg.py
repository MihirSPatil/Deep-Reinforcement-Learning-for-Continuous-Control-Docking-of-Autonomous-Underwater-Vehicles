#!/usr/bin/env python

from __future__ import print_function
import gym
from gym import wrappers
import rospy
import rospkg
import random
import torch
import numpy as np
from collections import deque
from ddpg_agent import Agent
# import our training environment
from openai_ros.task_envs.deepleng import deepleng_docking

if __name__ == '__main__':

    rospy.init_node('deepleng_docker', anonymous=True)

    # Create the Gym environment
    env = gym.make('DeeplengDocking-v1')
    rospy.loginfo("Gym environment done")
    agent = Agent(state_size=17, action_size=5, random_seed=2)

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('deepleng_control')
    outdir = pkg_path + '/training_results'
    # env = wrappers.Monitor(env, outdir, force=True)
    # rospy.loginfo("Monitor Wrapper started")
    max_episodes = 50
    max_timesteps = 25


    def ddpg(n_episodes=max_episodes, n_timesteps=max_timesteps):
        scores = []
        for episode in range(1, n_episodes + 1):
            state = env.reset()
            agent.reset()
            score = 0
            print("==========================================================================")
            print("Episode no. {}".format(episode))
            print("==========================================================================")
            for stp in range(1, n_timesteps + 1):
                # print("___________________________________________________________________________")
                print("Step no. {}".format(stp))
                # print("Current state: {}".format([round(elem, 2) for elem in state]))
                print("Current state: {}".format(state))
                action = agent.act(np.array(state))
                print("Action taken: {}".format(action))
                next_state, reward, done, _ = env.step(action)
                print("Reward for action: {}".format(reward))
                print("Next state: {}".format(next_state))
                agent.step(state, action, reward, next_state, done)
                state = np.array(next_state)
                score += reward
                if done:
                    # print("Done")
                    break
                print("___________________________________________________________________________")
            scores.append(score)
            torch.save(agent.actor_local.state_dict(), '/home/mihir/Desktop/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), '/home/mihir/Desktop/checkpoint_critic.pth')

        return scores


    scores = ddpg()

    env.close()
