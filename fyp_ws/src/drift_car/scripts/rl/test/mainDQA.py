#!/usr/bin/env python
import argparse
import os
import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import style
import time

from agents import DQNAgent
style.use('fivethirtyeight')

fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(211)
ax1.set_xlabel('Episode count')
ax1.set_ylabel('Reward')
ax2 = fig.add_subplot(212)
ax2.set_xlabel('Episode count')
ax2.set_ylabel('Loss')
plt.ion()


def refresh_chart(rewards, mean_loss):
    ax1.clear()
    ax2.clear()
    ax1.plot(rewards)
    ax2.plot(mean_loss)
    fig.canvas.draw()

def train(config, env):
    all_rewards = []
    steps_taken = []
    all_losses = []
    total_step_count = 0
    epsilon = 1.0
    annealing_rate = (epsilon - config.epsilon_min) / config.total_episodes

    tf.reset_default_graph()
    with tf.Session() as sess:
        agent = DQNAgent(sess, config)
        sess.run(tf.global_variables_initializer())

        plt.show()
        fig.canvas.draw()

        for episode_count in range(1, config.total_episodes + 1):
            step_count = 0
            episode_buffer = []
            running_reward = 0
            episode_loss = []
            done = False

            s = env.reset()
            while step_count < config.max_episode_length and not done:
                if config.render_env:
                    env.render()

                step_count += 1
                total_step_count += 1

                if np.random.randn(1) < epsilon or total_step_count < config.pretrain_steps:
                    action = np.random.randint(0, config.a_size)
                else:
                    action = agent.take_action(s)

                next_state, reward, done, _ = env.step(action)
                running_reward += reward

                if config.verbose:
                    print("Post Action", action, " on step count", step_count, "total_step_count", total_step_count, "next_state", next_state, "reward", reward, "done", done)

                d_int = 1 if done else 0
                episode_buffer.append([s, action, reward, next_state, d_int])
                s = next_state

                if total_step_count > config.pretrain_steps: 
                    if epsilon > config.epsilon_min:
                        epsilon -= annealing_rate
                    if total_step_count % config.update_freq == 0:
                        loss = agent.update_agent()
                        episode_loss.append(np.mean(loss))

            agent.add_experiences(episode_buffer)
            all_rewards.append(running_reward)
            steps_taken.append(step_count)
            if len(episode_loss) != 0:
                all_losses.append(np.mean(episode_loss))

            if episode_count % 100 == 0:
                rMat = np.resize(np.array(all_rewards),[len(all_rewards)//100,100])
                rMean = np.average(rMat,1)
                refresh_chart(rMean,all_losses)

                # meanR = np.mean(all_rewards[-100:])
                # print(meanR, epsilon)
                # refresh_chart(all_rewards, all_losses)
                # if meanR > 195:
                #     print("Solved in ", str(episode_count), " episodes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Integer hyperparams.
    parser.add_argument(
        '-bs',
        "--batch_size",
        help='The batch size for training',
        type=int,
        default=32)
    parser.add_argument(
        '-te',
        '--total_episodes',
        help='Total number of episodes to run algorithm (Default: 20k)',
        type=int,
        default=10000)
    parser.add_argument(
        '--max_episode_length',
        help='Length of each episode',
        type=int,
        default=500)
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
        default=0.99)
    parser.add_argument(
        '-emi',
        '--epsilon_min',
        help='Minimum allowable value for epsilon',
        type=float,
        default=0.0)
    parser.add_argument(
        '--tau',
        help='Controls update rate of target network',
        type=float,
        default=0.001)
    parser.add_argument(
        '-e_decay', 
        '--epsilon_decay', 
        help="Rate of epsilon decay", 
        type=float, 
        default=0.98)
    parser.add_argument(
        '-lr',
        '--learning_rate',
        help='Learning rate of algorithm',
        type=float,
        default=0.00001)

    # Intervals.
    parser.add_argument(
        '-uf',
        '--update_freq',
        help='Determines how often(steps) target network updates toward primary network. (Default: 50 steps)',
        type=int,
        default=4)
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
    parser.add_argument(
        '--model_path', 
        help='Path of saved model parameters',
        default='./model')

    # Others 
    parser.add_argument(
        '--verbose', 
        help='Verbose output', 
        action='store_true')
    
    config = parser.parse_args()

    env = gym.make('CartPole-v0')
    #env = gym.make('DriftCarGazeboEnv-v0')
    
    # Additional network params.
    vars(config)['a_size'] = env.action_space.n
    vars(config)['s_size'] = env.observation_space.shape[0]
    vars(config)['h_size'] = 512
    vars(config)['o_size'] = 512
    # Train the network.
    train(config, env)
