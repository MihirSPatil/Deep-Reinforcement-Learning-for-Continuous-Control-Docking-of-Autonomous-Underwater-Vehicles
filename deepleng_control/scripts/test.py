'''
## Test ##
# Test a trained DDPG network. This can be run alongside training by running 'run_every_new_ckpt.sh'.
@author: Mark Sinton (msinto93@gmail.com) 
'''

import os
import sys
import rospy
import gym
import tensorflow as tf
import numpy as np
import scipy.stats as ss


from network import Actor, Actor_BN
from openai_ros.task_envs.deepleng import deepleng_docking


class DDPGTest():
    def __init__(self):

        self.env = 'DeeplengDocking-v1'
        self.render = False
        self.random_seed = 999999
        self.num_eps_test = 100
        self.max_ep_length = 80
        self.dense1_size = 400
        self.dense2_size = 300
        self.final_layer_init = 0.003
        self.use_batch_norm = False
        self.ckpt_dir = "./ckpt"
        self.ckpt_file = None
        self.results_dir = "./test_results"
        self.log_dir = "./logs/test"


    def test(self):
        # Create environment
        env = gym.make(self.env)
        state_dims = env.observation_space.shape
        action_dims = env.action_space.shape
        action_bound_low = env.action_space.low
        action_bound_high = env.action_space.high

        # Set random seeds for reproducability
        env.seed(self.random_seed)
        np.random.seed(self.random_seed)
        tf.set_random_seed(self.random_seed)

        # Define input placeholder
        state_ph = tf.placeholder(tf.float32, ((None,) + state_dims))

        # Create policy (actor) network
        if self.use_batch_norm:
            actor = Actor_BN(state_ph, state_dims, action_dims, action_bound_low, action_bound_high, self, is_training=False, scope='actor_main')
        else:
            actor = Actor(state_ph, state_dims, action_dims, action_bound_low, action_bound_high, self, scope='actor_main')

        # Create session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Load ckpt file
        loader = tf.train.Saver()
        if self.ckpt_file is not None:
            ckpt = self.ckpt_dir + '/' + self.ckpt_file
        else:
            ckpt = tf.train.latest_checkpoint(self.ckpt_dir)

        loader.restore(sess, ckpt)
        sys.stdout.write('%s restored.\n\n' % ckpt)
        sys.stdout.flush()

        ckpt_split = ckpt.split('-')
        train_ep = ckpt_split[-1]

        # Create summary writer to write summaries to disk
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        # Create summary op to save episode reward to Tensorboard log
        reward_var = tf.Variable(0.0, trainable=False)
        tf.summary.scalar("Average Test Reward", reward_var)
        summary_op = tf.summary.merge_all()


        # Start testing

        rewards = []

        for test_ep in range(self.num_eps_test):
            state = env.reset()
            ep_reward = 0
            step = 0
            ep_done = False

            while not ep_done:
                if self.render:
                    env.render()
                action = sess.run(actor.output, {state_ph:np.expand_dims(state, 0)})[0]     # Add batch dimension to single state input, and remove batch dimension from single action output
                state, reward, terminal, _ = env.step(action)

                ep_reward += reward
                step += 1

                # Episode can finish either by reaching terminal state or max episode steps
                if terminal or step == self.max_ep_length:
                    sys.stdout.write('\x1b[2K\rTest episode {:d}/{:d}'.format(test_ep, self.num_eps_test))
                    sys.stdout.flush()
                    rewards.append(ep_reward)
                    ep_done = True

        mean_reward = np.mean(rewards)
        error_reward = ss.sem(rewards)

        sys.stdout.write('\x1b[2K\rTesting complete \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(mean_reward, error_reward))
        sys.stdout.flush()

        # Log average episode reward for Tensorboard visualisation
        summary_str = sess.run(summary_op, {reward_var: mean_reward})
        summary_writer.add_summary(summary_str, train_ep)

        # Write results to file
        if self.results_dir is not None:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            output_file = open(self.results_dir + '/' + self.env + '.txt', 'a')
            output_file.write('Training Episode {}: \t Average reward = {:.2f} +/- {:.2f} /ep \n\n'.format(train_ep, mean_reward, error_reward))
            output_file.flush()
            sys.stdout.write('Results saved to file \n\n')
            sys.stdout.flush()

        env.close()
    

def main():
    rospy.init_node('Ddpg_tester', anonymous=True)
    ddpg = DDPGTest()
    ddpg.test()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")


if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
