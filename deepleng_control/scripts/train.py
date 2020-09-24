#!/usr/bin/env python3
import os
import sys
import argparse
import gym
import tensorflow as tf
import numpy as np
import time
import rospy

from network import Critic, Actor, Critic_BN, Actor_BN
from experience_replay import ReplayMemory
from ou_noise import OrnsteinUhlenbeckActionNoise
from openai_ros.task_envs.deepleng import deepleng_docking


class ddpg_train():
    def __init__(self):
        # Environment parameters
        self.env = 'DeeplengDocking-v1'
        self.render = False
        self.random_seed = 99999999
        self.batch_size = 64
        self.num_eps_train = 200
        self.max_ep_length = 1000
        self.replay_mem_size = 1000000
        self.initial_replay_mem_size = 50000
        self.noise_scale = 0.1
        self.discount_rate = 0.99
        self.critic_learning_rate = 0.001
        self.actor_learning_rate = 0.0001
        self.critic_l2_lambda = 0.0
        self.dense1_size = 400
        self.dense2_size = 300
        self.final_layer_init = 0.003
        self.tau = 0.001
        self.use_batch_norm = False
        self.save_ckpt_step = 50
        self.ckpt_dir = "./ckpt/"
        self.ckpt_file = None
        self.log_dir = "./logs/train"

    def update_target_network(self, network_params, target_network_params, tau=1.0):
        # When tau=1.0, we perform a hard copy of parameters, otherwise a soft copy

        # Create ops which update target network parameters with (fraction of) main network parameters
        op_holder = []
        for from_var, to_var in zip(network_params, target_network_params):
            op_holder.append(to_var.assign((tf.multiply(from_var, self.tau) + tf.multiply(to_var, 1. - self.tau))))

        return op_holder

    def train(self):
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

        # Initialise replay memory
        replay_mem = ReplayMemory(self, state_dims, action_dims)

        # Initialise Ornstein-Uhlenbeck Noise generator
        exploration_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dims))
        noise_scaling = self.noise_scale * (action_bound_high - action_bound_low)

        # Define input placeholders
        state_ph = tf.placeholder(tf.float32, ((None,) + state_dims))
        action_ph = tf.placeholder(tf.float32, ((None,) + action_dims))
        target_ph = tf.placeholder(tf.float32, (None, 1))  # Target Q-value - for critic training
        action_grads_ph = tf.placeholder(tf.float32, (
                (None,) + action_dims))  # Gradient of critic's value output wrt action input - for actor training
        is_training_ph = tf.placeholder_with_default(True, shape=None)

        # Create value (critic) network + target network
        if self.use_batch_norm:
            critic = Critic_BN(state_ph, action_ph, state_dims, action_dims, self, is_training=is_training_ph,
                               scope='critic_main')
            critic_target = Critic_BN(state_ph, action_ph, state_dims, action_dims, self, is_training=is_training_ph,
                                      scope='critic_target')
        else:
            critic = Critic(state_ph, action_ph, state_dims, action_dims, self, scope='critic_main')
            critic_target = Critic(state_ph, action_ph, state_dims, action_dims, self, scope='critic_target')

        # Create policy (actor) network + target network
        if self.use_batch_norm:
            actor = Actor_BN(state_ph, state_dims, action_dims, action_bound_low, action_bound_high, self,
                             is_training=is_training_ph, scope='actor_main')
            actor_target = Actor_BN(state_ph, state_dims, action_dims, action_bound_low, action_bound_high, self,
                                    is_training=is_training_ph, scope='actor_target')
        else:
            actor = Actor(state_ph, state_dims, action_dims, action_bound_low, action_bound_high, self,
                          scope='actor_main')
            actor_target = Actor(state_ph, state_dims, action_dims, action_bound_low, action_bound_high, self,
                                 scope='actor_target')

        # Create training step ops
        critic_train_step = critic.train_step(target_ph)
        actor_train_step = actor.train_step(action_grads_ph)

        # Create ops to update target networks
        update_critic_target = self.update_target_network(critic.network_params, critic_target.network_params, self.tau)
        update_actor_target = self.update_target_network(actor.network_params, actor_target.network_params, self.tau)

        # Create session
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        # Define saver for saving model ckpts
        model_name = self.env + '.ckpt'
        checkpoint_path = os.path.join(self.ckpt_dir, model_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        saver = tf.train.Saver()

        # Load ckpt file if given
        if self.ckpt_file is not None:
            loader = tf.train.Saver()  # Restore all variables from ckpt
            ckpt = self.ckpt_dir + '/' + self.ckpt_file
            ckpt_split = ckpt.split('-')
            step_str = ckpt_split[-1]
            start_ep = int(step_str)
            loader.restore(sess, ckpt)
        else:
            start_ep = 0
            sess.run(tf.global_variables_initializer())
            # Perform hard copy (tau=1.0) of initial params to target networks
            sess.run(self.update_target_network(critic.network_params, critic_target.network_params))
            sess.run(self.update_target_network(actor.network_params, actor_target.network_params))

        # Create summary writer to write summaries to disk
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

        # Create summary op to save episode reward to Tensorboard log
        ep_reward_var = tf.Variable(0.0, trainable=False)
        tf.summary.scalar("Episode Reward", ep_reward_var)
        summary_op = tf.summary.merge_all()

        ## Training

        # Initially populate replay memory by taking random actions
        sys.stdout.write('\nPopulating replay memory with random actions...\n')
        sys.stdout.flush()
        env.reset()

        for random_step in range(1, self.initial_replay_mem_size + 1):
            if self.render:
                env.render()
            action = env.action_space.sample()
            state, reward, terminal, _ = env.step(action)
            replay_mem.add(action, reward, state, terminal)

            if terminal:
                env.reset()

            sys.stdout.write('\x1b[2K\rStep {:d}/{:d}'.format(random_step, self.initial_replay_mem_size))
            sys.stdout.flush()

        sys.stdout.write('\n\nTraining...\n')
        sys.stdout.flush()

        for train_ep in range(start_ep + 1, self.num_eps_train + 1):
            # Reset environment and noise process
            state = env.reset()
            exploration_noise.reset()

            train_step = 0
            episode_reward = 0
            duration_values = []
            ep_done = False

            sys.stdout.write('\n')
            sys.stdout.flush()

            while not ep_done:
                train_step += 1
                start_time = time.time()
                ## Take action and store experience
                if self.render:
                    env.render()
                if self.use_batch_norm:
                    action = sess.run(actor.output, {state_ph: np.expand_dims(state, 0), is_training_ph: False})[
                        0]  # Add batch dimension to single state input, and remove batch dimension from single action output
                else:
                    action = sess.run(actor.output, {state_ph: np.expand_dims(state, 0)})[0]
                action += exploration_noise() * noise_scaling
                state, reward, terminal, _ = env.step(action)
                replay_mem.add(action, reward, state, terminal)

                episode_reward += reward

                ## Train networks
                # Get minibatch
                states_batch, actions_batch, rewards_batch, next_states_batch, terminals_batch = replay_mem.getMinibatch()

                # Critic training step
                # Predict actions for next states by passing next states through policy target network
                future_action = sess.run(actor_target.output, {state_ph: next_states_batch})
                # Predict target Q values by passing next states and actions through value target network
                future_Q = sess.run(critic_target.output, {state_ph: next_states_batch, action_ph: future_action})[:,
                           0]  # future_Q is of shape [batch_size, 1], need to remove second dimension for ops with terminals_batch and rewards_batch which are of shape [batch_size]
                # Q values of the terminal states is 0 by definition
                future_Q[terminals_batch] = 0
                targets = rewards_batch + (future_Q * self.discount_rate)
                # Train critic
                sess.run(critic_train_step,
                         {state_ph: states_batch, action_ph: actions_batch, target_ph: np.expand_dims(targets, 1)})

                # Actor training step
                # Get policy network's action outputs for selected states
                actor_actions = sess.run(actor.output, {state_ph: states_batch})
                # Compute gradients of critic's value output wrt actions
                action_grads = sess.run(critic.action_grads, {state_ph: states_batch, action_ph: actor_actions})
                # Train actor
                sess.run(actor_train_step, {state_ph: states_batch, action_grads_ph: action_grads[0]})

                # Update target networks
                sess.run(update_critic_target)
                sess.run(update_actor_target)

                # Display progress
                duration = time.time() - start_time
                duration_values.append(duration)
                ave_duration = sum(duration_values) / float(len(duration_values))

                sys.stdout.write(
                    '\x1b[2K\rEpisode {:d}/{:d} \t Steps = {:d} \t Reward = {:.3f} \t ({:.3f} s/step)'.format(train_ep,
                                                                                                              self.num_eps_train,
                                                                                                              train_step,
                                                                                                              episode_reward,
                                                                                                              ave_duration))
                sys.stdout.flush()

                if terminal or train_step == self.max_ep_length:
                    # Log total episode reward and begin next episode
                    summary_str = sess.run(summary_op, {ep_reward_var: episode_reward})
                    summary_writer.add_summary(summary_str, train_ep)
                    ep_done = True

            if train_ep % self.save_ckpt_step == 0:
                saver.save(sess, checkpoint_path, global_step=train_ep)
                sys.stdout.write('\n Checkpoint saved.')
                sys.stdout.flush()

        env.close()


def main():
    rospy.init_node('Ddpg_docker', anonymous=True)
    ddpg = ddpg_train()
    ddpg.train()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        print("Shutting down")


if __name__ == '__main__':
    main()
