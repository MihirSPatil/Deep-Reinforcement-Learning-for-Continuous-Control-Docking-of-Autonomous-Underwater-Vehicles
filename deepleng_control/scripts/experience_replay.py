'''
## Replay Memory ##
# Adapted from: https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py
# Creates replay memory buffer to add experiences to and sample batches of experiences from
'''

import numpy as np

class ReplayMemory:
    def __init__(self, args, state_dims, action_dims):
        self.buffer_size = args.replay_mem_size
        self.min_buffer_size = args.initial_replay_mem_size
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.batch_size = args.batch_size
        self.count = 0
        self.current = 0

        # preallocate memory
        self.actions = np.empty((self.buffer_size,) + self.action_dims, dtype = np.float32)
        self.rewards = np.empty(self.buffer_size, dtype = np.float32)
        self.states = np.empty((self.buffer_size,) + self.state_dims, dtype = np.float32)
        self.terminals = np.empty(self.buffer_size, dtype = np.bool)   
        
        self.state_batch = np.empty((self.batch_size,) + self.state_dims, dtype = np.float32)
        self.next_state_batch = np.empty((self.batch_size,) + self.state_dims, dtype = np.float32)
        
        
    def add(self, action, reward, state, terminal):        
        assert state.shape == self.state_dims
        assert action.shape == self.action_dims

        self.actions[self.current, ...] = action
        self.rewards[self.current] = reward
        self.states[self.current, ...] = state
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.buffer_size
        
  
    def getState(self, index):
        # Returns the state at position 'index'.
        return self.states[index, ...]
         

    def getMinibatch(self):
        # memory should be initially populated with random actions up to 'min_buffer_size'
        assert self.count >= self.min_buffer_size, "Replay memory does not contain enough samples to start learning, take random actions to populate replay memory"
                
        # sample random indexes
        indexes = []
        # do until we have a full batch of states
        while len(indexes) < self.batch_size:
            # find random index 
            while True:
                # sample one index
                index = np.random.randint(1, self.count)
                # check index is ok
                # if state and next state wrap over current pointer, then get new one (as state from current pointer position will not be from same episode as state from previous position)
                if index == self.current:
                    continue
                # if state and next state wrap over episode end, i.e. current state is terminal, then get new one (note that next state can be terminal)
                if self.terminals[index-1]:
                    continue
                # index is ok to use
                break
            
            # Populate states and next_states with selected state and next_state
            # NB! having index first is fastest in C-order matrices
            self.state_batch[len(indexes), ...] = self.getState(index - 1)
            self.next_state_batch[len(indexes), ...] = self.getState(index)
            indexes.append(index)   
        
        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]
        
        return self.state_batch, actions, rewards, self.next_state_batch, terminals
    
if  __name__ == '__main__':
    ### For testing ###
    import argparse
    import gym    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--replay_mem_size", type=int, default=100)
    parser.add_argument("--initial_replay_mem_size", type=int, default=1, help="Initial size of replay memory (populated by random actions) before learning starts")
    args = parser.parse_args()
    
    states = []
    actions = []
    rewards = []
    
    #Populate experience buffer
    env = gym.make('Pendulum-v0')
    state_dims = env.observation_space.shape
    num_actions = env.action_space.shape
    
    # Initialise replay memory and state buffer
    replay_mem = ReplayMemory(args, state_dims, num_actions)
 
    env.reset()
     
    for i in range(0,100):
        env.render(mode='rgb_array')
        action = env.action_space.sample()
        state, reward, terminal, _ = env.step(action)
        states.append(state)
        actions.append(actions)
        rewards.append(reward)
        replay_mem.add(action, reward, state, terminal)
        if terminal:
            env.reset()
             
    env.close()
        
    states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch = replay_mem.getMinibatch()
    
