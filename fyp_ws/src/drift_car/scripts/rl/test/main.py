#!/usr/bin/env python
import argparse
import os
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import style
import time

from agents import QNAgent
style.use('fivethirtyeight')

fig = plt.figure(figsize=(20, 20))
plt.ylabel('Reward')
plt.xlabel('Episode count')
plt.ion()

def refresh_chart(rewards):
    fig.clear()
    plt.plot(rewards)
    fig.canvas.draw()

def train(config, env):
    all_rewards = []
    steps_taken = []
    all_losses = []
    total_step_count = 0
    epsilon = 1.0

    tf.reset_default_graph()
    with tf.Session() as sess:
        agent = QNAgent(sess, config)
        sess.run(tf.global_variables_initializer())

        gradBuffer = sess.run(tf.trainable_variables())
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0

        plt.show()
        fig.canvas.draw()

        for episode_count in range(1, config.total_episodes + 1):
            step_count = 0
            episode_buffer = []
            running_reward = 0
            done = False

            s = env.reset()
            while step_count < config.max_episode_length and not done:
                if config.render_env:
                    env.render()

                step_count += 1
                total_step_count += 1

                a_dist = sess.run(agent.output, feed_dict={agent.state_in:[s]})
                action = np.random.choice(a_dist[0], p=a_dist[0])
                action = np.argmax(a_dist == action)

                next_state, reward, done, _ = env.step(action)
                if config.verbose:
                        print("Post Action", action, " on step count", step_count, "total_step_count", total_step_count, "next_state", next_state, "reward", reward, "done", done)
                d_int = 1 if done else 0
                running_reward += reward
                episode_buffer.append(
                    [s, action, reward, next_state, d_int])
                s = next_state

            ep_history = np.array(episode_buffer)
            ep_history[:,2] = discount_rewards(ep_history[:,2], config.gamma)
            feed_dict = { agent.reward_holder:ep_history[:,2], agent.action_holder:ep_history[:,1], agent.state_in:np.vstack(ep_history[:,0])}
            grads = sess.run(agent.gradients, feed_dict=feed_dict)
            for idx, grad in enumerate(grads):
                gradBuffer[idx] += grad

            if episode_count % config.update_freq == 0 and episode_count != 0:
                feed_dict= dictionary = dict(zip(agent.gradient_holders, gradBuffer))
                _ = sess.run(agent.update_batch, feed_dict=feed_dict)
                for ix,grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
            
            all_rewards.append(running_reward)
            steps_taken.append(step_count)
            if episode_count % 100 == 0:
                mean = np.mean(all_rewards[-100:])
                print(mean)
                refresh_chart(all_rewards)
                if mean > 195:
                    print("Solved in ", str(episode_count), " episodes")

def discount_rewards(r, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(xrange(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Integer hyperparams.
    parser.add_argument(
        '-bs',
        "--batch_size",
        help='The batch size for training',
        type=int,
        default=100)
    parser.add_argument(
        '-te',
        '--total_episodes',
        help='Total number of episodes to run algorithm (Default: 20k)',
        type=int,
        default=20000)
    parser.add_argument('--max_episode_length', help='Length of each episode',
                        type=int, default=999)
    parser.add_argument(
        '--pretrain_steps',
        help='Number of steps to run algorithm. Default 10k steps'
        ' without updating networks',
        type=int,
        default=10000)
    parser.add_argument(
        '-re',
        '--render_env',
        help='Render learning environment',
        action='store_true')

    # Float hyperparameters.
    parser.add_argument(
        '-g',
        '--gamma',
        help='Discount factor',
        type=float,
        default=0.95)
    parser.add_argument(
        '-emi',
        '--epsilon_min',
        help='Minimum allowable value for epsilon',
        type=float,
        default=0.0)
    parser.add_argument('--tau', help='Controls update rate of target network',
                        type=float, default=0.1)
    # parser.add_argument('-e_decay', '--epsilon_decay', help="Rate of epsilon decay", 
    #     type=float, default=0.98)
    parser.add_argument(
        '-lr',
        '--learning_rate',
        help='Learning rate of algorithm',
        type=float,
        default=0.001)

    # Intervals.
    parser.add_argument(
        '-uf',
        '--update_freq',
        help='Determines how often(steps) target network updates toward primary network. (Default: 50 steps)',
        type=int,
        default=50)
    parser.add_argument(
        '--save_model_interval',
        help='How often to save model. (Default: 5 ep)',
        type=int,
        default=5)
    parser.add_argument(
        '--epsilon_update_interval',
        help='How often to update epsilon (Default: 4 ep)',
        type=int,
        default=4)

    parser.add_argument(
        '--chart_refresh_interval',
        help='Number of episodes between chart updates (Default: 100 ep)',
        type=int,
        default=100)

    # Model load/save and path.
    parser.add_argument(
        '-lm',
        '--load_model',
        help='Load saved model parameters',
        action='store_true')
    parser.add_argument(
        '-sm',
        '--save_model',
        help='Periodically save model parameters',
        action='store_true')
    parser.add_argument('--model_path', help='Path of saved model parameters',
                        default='./model')
    parser.add_argument('--verbose', help='Verbose output', action='store_true')
    
    config = parser.parse_args()

    env = gym.make('CartPole-v0')
    #env = gym.make('DriftCarGazeboEnv-v0')
    
    # Additional network params.
    vars(config)['a_size'] = env.action_space.n
    vars(config)['s_size'] = env.observation_space.shape[0]
    vars(config)['h_size'] = 200
    vars(config)['o_size'] = 200
    # Train the network.
    train(config, env)
