import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
import gym
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reinforce import Reinforce


def test_model(model, env, gamma):
    rewards = []
    for i in range(100):
        print("TESTING ROUND", i)
        _, _, _, cum_reward = model.generate_episode(env, gamma=gamma)
        rewards.append(cum_reward)
    print("###########TESTING DONE##############")
    return np.mean(rewards), np.std(rewards)


def loss_fn(returns, actions):

    # log_actions = tf.math.log(tf.clip_by_value(actions, 1e-15, 1.0))
    log_actions = tf.math.log(actions)
    returns_by_actions = tf.math.multiply(returns, log_actions)
    summed = tf.math.reduce_sum(returns_by_actions, axis=1)
    reduced = tf.math.reduce_mean(summed)
    return tf.math.negative(reduced)


def critic_loss_fn(returns, values):
    return K.mean(tf.math.square(tf.math.subtract(returns, values)))


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here.

    def __init__(self, lr, critic_lr, n=20):
        # Initializes A2C.
        # Args:
        # - model: The actor model.
        # - lr: Learning rate for the actor model.
        # - critic_model: The critic model.
        # - critic_lr: Learning rate for the critic model.
        # - n: The value of N in N-step A2C.
        self.lr = lr
        self.critic_lr = critic_lr
        self.model = Sequential()
        lecun = keras.initializers.lecun_uniform()
        vski1 = keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
        vski2 = keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
        vski3 = keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
        vski4 = keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
        vski5 = keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
        vski6 = keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
        vski7 = keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
        vski8 = keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=None)

        self.model.add(Dense(16, activation=tf.nn.relu, input_shape=(
            8, ), bias_initializer='zeros', kernel_initializer=vski1))
        self.model.add(Dense(16, activation=tf.nn.relu,
                             bias_initializer='zeros', kernel_initializer=vski2))
        self.model.add(Dense(16, activation=tf.nn.relu,
                             bias_initializer='zeros', kernel_initializer=vski3))
        self.model.add(Dense(4, activation=tf.nn.softmax,
                             bias_initializer='zeros', kernel_initializer=vski4))
        adam_actor = keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.9,
                                           beta_2=0.999, amsgrad=False)

        self.model.compile(optimizer=adam_actor, loss=loss_fn)
        self.critic_model = Sequential()
        self.critic_model.add(Dense(16, activation=tf.nn.relu, input_shape=(
            8, ), bias_initializer='zeros', kernel_initializer=vski5))
        self.critic_model.add(Dense(16, activation=tf.nn.relu,
                                    bias_initializer='zeros', kernel_initializer=vski6))
        self.critic_model.add(Dense(16, activation=tf.nn.relu,
                                    bias_initializer='zeros', kernel_initializer=vski7))
        # self.critic_model.add(Dense(16, activation=tf.nn.relu,
        #                             bias_initializer='zeros', kernel_initializer=lecun))
        self.critic_model.add(Dense(1, activation='linear',
                                    bias_initializer='zeros', kernel_initializer=vski8))
        adam_critic = keras.optimizers.Adam(learning_rate=self.critic_lr, beta_1=0.9,
                                            beta_2=0.999, amsgrad=False)
        self.critic_model.compile(optimizer=adam_critic, loss=critic_loss_fn)

        self.n = n

        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using A2C.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states, theta_returns, R_t, _ = self.generate_episode(env, gamma=gamma)
        self.model.fit(states, theta_returns, verbose=False)
        self.critic_model.fit(states, R_t, verbose=False)
        return

    def generate_episode(self, env, render=False, gamma=1.0):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        # TODO: Implement this method.
        done = False
        state = env.reset()
        states = []
        actions = []
        rewards = []
        values = []
        while not done:

            if render:
                env.render()

            states.append(state)

            action_dist = self.model.predict(np.array([state]))[0]

            action = np.random.choice(env.action_space.n, 1, p=action_dist)
            action = action[0]
            actions.append(action)
            value = self.critic_model.predict(np.array([state]))[0][0]
            values.append(value)
            state, reward, done, _ = env.step(action)
            rewards.append(reward/1000.0)
        print('reward = ', np.sum(rewards)*1000)
        sum_reward_100_episodes = np.sum(rewards)*1000

        T = len(rewards)
        V_end = 0
        R_t = np.zeros((T, env.action_space.n))
        values_matrix = np.zeros((T, env.action_space.n))  # In T x 4 form for theta optimization
        for t in range(T-1, -1, -1):
            if (t + self.n) >= T:
                V_end = 0
            else:
                assert((t + self.n) < T)
                V_end = values[t+self.n]
            R_t[t][actions[t]] = V_end * (gamma**(self.n))

            for k in range(self.n):

                if (t+k) < T:
                    R_t[t][actions[t]] += (gamma**(k)) * (rewards[t+k])

            values_matrix[t][actions[t]] = values[t]

        theta_returns = R_t - values_matrix

        R_t = np.sum(R_t, axis=1)

        # R_t -= R_t.mean()
        # R_t /= R_t.std()
        # theta_stats = np.sum(theta_returns, axis=1)
        # theta_stats = (theta_stats - theta_stats.mean()) / theta_stats.std()
        # for t in range(T):
        #     theta_returns[t][actions[t]] = theta_stats[t]
        return np.array(states), theta_returns, R_t, sum_reward_100_episodes


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The actor's learning rate.")
    parser.add_argument('--critic-lr', dest='critic_lr', type=float,
                        default=1e-4, help="The critic's learning rate.")
    parser.add_argument('--n', dest='n', type=int,
                        default=20, help="The value of N in N-step A2C.")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    parser_group = parser.add_mutually_exclusive_group(required=False)
    parser_group.add_argument('--render', dest='render',
                              action='store_true',
                              help="Whether to render the environment.")
    parser_group.add_argument('--no-render', dest='render',
                              action='store_false',
                              help="Whether to render the environment.")
    parser.set_defaults(render=False)

    return parser.parse_args()


def main(args):
    # Parse command-line arguments.
    args = parse_arguments()
    num_episodes = args.num_episodes
    lr = args.lr
    critic_lr = args.critic_lr
    n = args.n
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    test_mean_reward = []
    test_reward_std = []

    x_space = []

    # TODO: Create the model.
    # current 5e-4
    AI = A2C(1e-4, 1e-4, 1)
    # TODO: Train the model using A2C and plot the learning curves.
    print(AI.n)
    for i in range(num_episodes):
        print('episode', i)
        if i % 500 == 0:
            mean, std = test_model(AI, env, 0.99)
            test_mean_reward.append(mean)
            test_reward_std.append(std)
            print("MEAN REWARD", mean)
            print("STDEV", std)

            x_space.append(i)
            fig = plt.figure()
            plt.plot(x_space, test_mean_reward)
            plt.xlabel('Number of episodes trained on')
            plt.ylabel('Mean cumulative reward on 100 test episodes')
            plt.errorbar(x_space, test_mean_reward, yerr=test_reward_std)
            fig.savefig('a2c-50.png')
        AI.train(env, 0.99)
    fig = plt.figure()
    plt.plot(x_space, test_mean_reward)
    plt.xlabel('Number of episodes trained on')
    plt.ylabel('Mean cumulative reward on 100 test episodes')
    plt.errorbar(x_space, test_mean_reward, yerr=test_reward_std)
    # plt.errorbar(x_space, test_mean_reward, yerr=test_reward_std)
    fig.savefig('a2c-50.png')


if __name__ == '__main__':
    main(sys.argv)
