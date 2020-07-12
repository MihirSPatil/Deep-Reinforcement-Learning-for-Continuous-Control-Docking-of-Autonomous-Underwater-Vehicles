'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning
 
Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA
 
        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import random
import numpy as np

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, str(action)), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))            
        '''
        oldv = self.q.get((state, str(action)), None)
        if oldv is None:
            self.q[(state, str(action))] = reward
        else:
            self.q[(state, str(action))] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, self.actions)]
        action = np.random.uniform(low=-60, high=60, size=(5,))
        return action

    def learn(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, self.actions)])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)
