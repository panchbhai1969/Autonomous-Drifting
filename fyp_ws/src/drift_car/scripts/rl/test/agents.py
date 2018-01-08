#!/usr/bin/env python
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import datetime

from network_models import DQNetwork
from utilities import target_network_update_apply, target_network_update_ops, ExperienceReplayBuffer

try:
    xrange = xrange
except:
    xrange = range

class QNAgent():
    def __init__(self, sess, config):
        lr = config.learning_rate
        s_size = config.s_size
        a_size = config.a_size
        h_size = config.h_size

        #TODO move network to network file
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden1 = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden2 = slim.fully_connected(hidden1, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden3 = slim.fully_connected(hidden2, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden3, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output, 1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

class DQNAgent():
    def __init__(self, sess, config):
        self.primary_Q_network = DQNetwork(
            config.learning_rate,
            config.s_size,
            config.a_size,
            config.h_size,
            "primary")
        self.target_Q_network = DQNetwork(
            config.learning_rate,
            config.s_size,
            config.a_size,
            config.h_size,
            "target")

        self.saver = tf.train.Saver(save_relative_paths=True)
        tvars = tf.trainable_variables()
        self.targetOps = target_network_update_ops(tvars, config.tau)
        self.experience_buffer = ExperienceReplayBuffer()

        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.sess = sess
        
        #self.summary_writer = tf.summary.FileWriter('./summary/'+str(datetime.datetime.now()), self.sess.graph)
        #self.merged_summary = tf.summary.merge_all()

    def take_action(self, state):
        return self.sess.run(self.primary_Q_network.predict, 
                feed_dict={self.primary_Q_network.state_input: [state]})[0]

    def add_experiences(self, experiences):
        # Experience: [s, action, reward, next_state, d_int]
        self.experience_buffer.add(experiences)

    def update_agent(self):
        train_batch = self.experience_buffer.sample_batch(self.batch_size)

        # Double DQN
        max_next_state_action = self.sess.run(self.primary_Q_network.predict, feed_dict={
            self.primary_Q_network.state_input: np.vstack(train_batch[:, 3])})
        target_network_Q_values = self.sess.run(self.target_Q_network.Qout, feed_dict={
            self.target_Q_network.state_input: np.vstack(train_batch[:, 3])})
        end_multiplier = -(train_batch[:, 4] - 1)

        Q_values_next_state = target_network_Q_values[range(self.batch_size), max_next_state_action]
        target_Q_values = train_batch[:,2] + (self.gamma * Q_values_next_state * end_multiplier)

        l, _ = self.sess.run([self.primary_Q_network.prediction_loss, self.primary_Q_network.update], feed_dict={
                self.primary_Q_network.state_input: np.vstack(train_batch[:, 0]),
                self.primary_Q_network.targetQ: target_Q_values, 
                self.primary_Q_network.actions: train_batch[:,1]
            })
                    
        # Update the target network.
        target_network_update_apply(self.sess, self.targetOps)

        return l