#!/usr/bin/env python
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from utilities import huber_loss

# Dueling DQN model.
class DQNetwork():
    def __init__(self, lr, s_size, a_size, h_size, name):
        # Forward pass of the network.
        with tf.name_scope(name):
            self.state_input = tf.placeholder(
                shape=[None, s_size], dtype=tf.float32, name='state_input')

        hidden1 = slim.fully_connected(self.state_input, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden2 = slim.fully_connected(hidden1, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden3 = slim.fully_connected(hidden2, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        hidden4 = slim.fully_connected(hidden3, h_size, biases_initializer=None, activation_fn=tf.nn.relu)

        self.streamAC, self.streamVC = tf.split(hidden4, 2, 1)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, a_size]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)

        # Evaluate loss and backward pass.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32, name='target_q')
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32, name='action_taken')
        self.actions_onehot = tf.one_hot(self.actions, a_size, dtype=tf.float32, name='action_one_hot')
        self.Q = tf.reduce_sum(self.Qout * self.actions_onehot, axis=1)
        self.td_error = tf.square(self.targetQ - self.Q)
        #self.prediction_loss = tf.reduce_mean(huber_loss(self.predicted_Q - self.target_Q))
        self.prediction_loss = tf.reduce_mean(self.td_error)

        # Optimizer
        with tf.name_scope(name+'-train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update = self.optimizer.minimize(self.prediction_loss)