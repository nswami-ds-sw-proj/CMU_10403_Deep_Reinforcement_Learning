import os
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.layers import Dense, Flatten, Input, Concatenate, Lambda, Activation
from keras.models import Model, load_model
from keras.regularizers import l2
from utils.util import ZFilter
import time

HIDDEN1_UNITS = 400
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 400

import logging
log = logging.getLogger('root')

class PENN:
    """
    (P)robabilistic (E)nsemble of (N)eural (N)etworks
    """

    def __init__(self, num_nets, state_dim, action_dim, learning_rate):
        """
        :param num_nets: number of networks in the ensemble
        :param state_dim: state dimension
        :param action_dim: action dimension
        :param learning_rate:
        """

        self.sess = tf.Session()
        self.num_nets = num_nets
        self.state_dim = state_dim
        self.action_dim = action_dim
        K.set_session(self.sess)

        # Log variance bounds
        self.max_logvar = tf.Variable(-3 * np.ones([1, self.state_dim]), dtype=tf.float32)
        self.min_logvar = tf.Variable(-7 * np.ones([1, self.state_dim]), dtype=tf.float32)

        # Create or load networks
        self.networks = [self.create_network(n) for n in range(self.num_nets)]

        # Define list of tensors / operations
        self.trainable_weights = [net.trainable_weights for net in self.networks]
        self.inputs = [net.input for net in self.networks]
        self.outputs = [self.get_output(net.output) for net in self.networks]
        self.targets = [tf.placeholder(tf.float32, [None, self.state_dim]) for _ in self.networks]

        # Minimize -loglike Guassian loss, while also log RMSE loss
        self.guas_losses = [self.get_loss(targ, out)
                            for (targ, out) in zip(self.targets, self.outputs)]

        self.rmse_losses = [self.get_rmse(targ, out)
                            for (targ, out) in zip(self.targets, self.outputs)]
        self.train_op = [tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=weights)
                         for (loss, weights) in zip(self.guas_losses, self.trainable_weights)]

        self.sess.run(tf.global_variables_initializer())

    def get_output(self, output):
        """
        Argument:
          output: tf variable representing the output of the keras models, i.e., model.output
        Return:
          mean and log variance tf tensors
        Note that you will still have to call sess.run on these tensors in order to get the actual output.
        """
        mean = output[:, 0:self.state_dim]
        raw_v = output[:, self.state_dim:]
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - raw_v)
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)
        return mean, logvar

    def get_rmse(self, targ, output):
        # TODO: write your code here
        mu = output[0]
        logvar = output[1]

        rmse = K.sqrt(K.mean(K.square(targ - mu), axis=1))

        return tf.reduce_mean(rmse)

    def get_loss(self, targ, output):
        # TODO: write your code here

        mu = output[0]
        logvar = output[1]

        diff = (mu - targ)

        loss = tf.reduce_mean(diff * K.exp(-logvar) * diff, axis=1)
        loss += tf.reduce_mean(logvar, axis=-1)

        return tf.reduce_mean(loss)

    def create_network(self, n):
        I = Input(shape=[self.state_dim + self.action_dim], name='input')
        h1 = Dense(HIDDEN1_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(I)
        h2 = Dense(HIDDEN2_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h1)
        h3 = Dense(HIDDEN3_UNITS, activation='relu', kernel_regularizer=l2(0.0001))(h2)
        O = Dense(2 * self.state_dim, activation='linear', kernel_regularizer=l2(0.0001))(h3)
        model = Model(input=I, output=O)
        return model

    def train(self, inputs, targets, batch_size=128, epochs=5):
        """
        Training the Probabilistic Ensemble (Algorithm 3)
          :param inputs : state and action inputs. Assumes that inputs are standardized.
          :prams targets: resulting states
        """

        D_size = len(inputs)
        data = [np.random.choice(D_size, size=D_size, replace=True) for _ in range(self.num_nets)]

        rmse = []
        gaus = []

        for e in range(epochs):

            rmse_e = []
            gaus_e = []

            batches = [[] for _ in range(self.num_nets)]
            for n in range(self.num_nets):
                net_data = np.random.permutation(data[n])

                for i in range(0, len(net_data), batch_size):
                    batch = net_data[i:i + batch_size]
                    batches[n].append(batch)

            for batch_ind in range(len(batches[0])):
                feed_dict = {}
                for n in range(self.num_nets):
                    feed_dict[self.inputs[n]] = inputs[batches[n][batch_ind]]
                    feed_dict[self.targets[n]] = targets[batches[n][batch_ind]]

                x = self.sess.run([self.rmse_losses, self.guas_losses, self.train_op],
                                    feed_dict=feed_dict)

                rmse_e.append(x[0])
                gaus_e.append(x[1])

            rmse.append(np.mean(rmse_e))
            gaus.append(np.mean(gaus_e))

        np.save("rmse_ens", np.array(rmse))
        np.save("gaus_ens", np.array(gaus))

        return (np.mean(rmse), np.mean(gaus))
