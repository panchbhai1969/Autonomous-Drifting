#!/usr/bin/env python
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from utilities import huber_loss

# Dueling DQN model.


class DQNetwork():
    def __init__(self, lr, s_size, a_size, h_size, o_size, name):
        if o_size % 2 != 0:
            raise ValueError('Number of outputs from final layer must be even')

        # Forward pass of the network.
        # output: batch_size x s_size
        with tf.name_scope(name):
            self.state_input = tf.placeholder(
                shape=[None, s_size], dtype=tf.float32, name='state_input')

        #initializer = tf.truncated_normal_initializer(0, 0.02)
        # output: batch_size x h_size
        layer1 = self.linear(self.state_input, h_size, tf.nn.relu, name + "-layer1")
        # Layer 2.
        layer2 = self.linear(layer1, h_size, tf.nn.relu, name + "-layer2")
        # Layer 3
        layer3 = self.linear(layer2, h_size, tf.nn.relu, name + "-layer3")

        # Layer 4
        layer4 = self.linear(layer3, h_size, tf.nn.relu, name + "-layer4")

        with tf.name_scope(name+"-value_stream"):
            value = self.linear(layer4, o_size, tf.nn.relu, name+'-value1')
            value = self.linear(value, 1, scope=name +'-value2')

        with tf.name_scope(name+'-adv_stream'):
            advantage = self.linear(layer4, o_size, tf.nn.relu, name+'-adv1')
            advantage = self.linear(advantage, a_size, scope=name+'-adv2')

        with tf.name_scope(name+"-Qout"):
            self.Qout = value + tf.subtract(advantage, tf.reduce_mean(advantage, axis=1, keep_dims=True))
        # Predict action that maximizes Q-value.
        self.action_predicted = tf.argmax(self.Qout, axis=1)

        # Evaluate loss and backward pass.
        self.target_Q = tf.placeholder(shape=[None], dtype=tf.float32, name='target_q')
        self.actions_taken = tf.placeholder(shape=[None], dtype=tf.int32, name='action_taken')
        self.action_taken_one_hot = tf.one_hot(
            self.actions_taken, a_size, 1.0, 0.0, dtype=tf.float32, name='action_one_hot')
        self.predicted_Q = tf.reduce_sum(self.Qout * self.action_taken_one_hot, axis=1)

        with tf.name_scope(name):
            self.prediction_loss = tf.reduce_mean(
                huber_loss(self.predicted_Q - self.target_Q))
            if name == "primary":
                tf.summary.scalar("huber-loss", self.prediction_loss)

        # Optimizer
        with tf.name_scope(name+'-train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update = self.optimizer.minimize(self.prediction_loss)

    def linear(self, x, out_size, activation_fn=None, scope='linear'):
        shape = x.get_shape().as_list()
        with tf.name_scope(scope):
            W = tf.Variable(tf.truncated_normal([shape[1], out_size], mean=0), name="W")
            b = tf.Variable(tf.constant(0.02, shape=[out_size]), name="b")
            out = tf.contrib.layers.batch_norm(tf.nn.bias_add(tf.matmul(x, W), b), center=True, scale=True, is_training=True, scope=scope+"/batch_norm")
            #out = tf.nn.bias_add(tf.matmul(x, W), b)
            if activation_fn != None:
                out = activation_fn(out)
            tf.summary.histogram("weights", W)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", out)
            return out
