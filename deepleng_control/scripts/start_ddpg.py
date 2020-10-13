#!/usr/bin/env python3

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


class DdpgDeepleng():

    def __init__(self):
        # Create the Gym environment
        self.env = gym.make('DeeplengDocking-v1')
        rospy.loginfo("Gym environment done")
        self.agent = Agent(state_size=13, action_size=3, random_seed=2)

        # Set the logging system
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('deepleng_control')
        outdir = pkg_path + '/training_results'
        # env = wrappers.Monitor(env, outdir, force=True)
        # rospy.loginfo("Monitor Wrapper started")
        self.max_episodes = 200
        self.max_timesteps = 1000

    def __call__(self, *args, **kwargs):
        scores = []
        for episode in range(1, self.max_episodes + 1):
            state = self.env.reset()
            self.agent.reset()
            score = 0
            print("==========================================================================")
            print("Episode no. {}".format(episode))
            print("==========================================================================")
            for stp in range(1, self.max_timesteps + 1):
                # print("___________________________________________________________________________")
                print("Step no. {}".format(stp))
                # print("Current state: {}".format([round(elem, 2) for elem in state]))
                print("Current state: {}".format(state))
                action = self.agent.act(np.array(state))
                print("Action taken: {}".format(action))
                next_state, reward, done, _ = self.env.step(action)
                print("Reward for action: {}".format(reward))
                print("Next state: {}".format(next_state))
                self.agent.step(state, action, reward, next_state, done)
                state = np.array(next_state)
                score += reward
                if done:
                    # print("Done")
                    break
                print("___________________________________________________________________________")
            scores.append(score)
            torch.save(self.agent.actor_local.state_dict(),
                       '/home/dfki.uni-bremen.de/mpatil/Desktop/checkpoint_actor.pth')
            torch.save(self.agent.critic_local.state_dict(),
                       '/home/dfki.uni-bremen.de/mpatil/Desktop/checkpoint_critic.pth')
        self.env.close()
        return scores


def main():
    rospy.init_node('DdpgDeepleng_docker', anonymous=True)
    train = DdpgDeepleng()
    scores = train()
    print("Scores: ", scores)
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")

if __name__ == '__main__':
    main()