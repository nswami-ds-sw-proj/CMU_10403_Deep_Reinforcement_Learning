import sys
import argparse
import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dense, BatchNormalization
import gym
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
# model.predcit = [0.25 , 0.3 , 0.5]
# np.random.seed(0)
# tf.random.set_seed(0)


def test_model(model, env, gamma):
    rewards = []
    for i in range(100):
        _, _, _, cum_reward = model.generate_episode(env, gamma=gamma)
        rewards.append(cum_reward)

    return np.mean(rewards), np.std(rewards)


def loss_fn(returns, actions):

    # log_actions = tf.math.log(tf.clip_by_value(actions, 1e-15, 1.0))
    # [0.3, 0.2, 0.1 .4] [0, 0, .425, 0] ---> 0 , 0 , .1,  0  --> [.1]
    log_actions = tf.math.log(actions)
    returns_by_actions = tf.math.multiply(returns, log_actions)  # T x 4
    summed = tf.math.reduce_sum(returns_by_actions, axis=1)  # T x 1
    reduced = tf.math.reduce_mean(summed)
    return tf.math.negative(reduced)


class Reinforce(object):
    # Implementation of the policy gradient method REINFORCE.

    def __init__(self, lr, env):
        self.env = env
        self.model = Sequential()
        self.lr = lr

        vski1 = keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
        vski2 = keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
        vski3 = keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=None)
        vski4 = keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_avg', distribution='uniform', seed=None)

        self.model.add(Dense(16, activation=tf.nn.relu, input_shape=(
            8,), bias_initializer='zeros', kernel_initializer=vski1))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(16, activation=tf.nn.relu,
                             bias_initializer='zeros', kernel_initializer=vski2))
        self.model.add(Dense(16, activation=tf.nn.relu,
                             bias_initializer='zeros', kernel_initializer=vski3))
        # self.model.add(BatchNormalization())
        self.model.add(Dense(4, activation=tf.nn.softmax,
                             bias_initializer='zeros', kernel_initializer=vski4))
        adam = keras.optimizers.Adam(learning_rate=self.lr, beta_1=0.9,
                                     beta_2=0.999, amsgrad=False)

        self.model.compile(optimizer=adam, loss=loss_fn)
        # TODO: Define any training operations and optimizers here, initialize
        #       your variables, or alternately compile your model here.

    def train(self, env, gamma=1.0):
        # Trains the model on a single episode using REINFORCE.
        # TODO: Implement this method. It may be helpful to call the class
        #       method generate_episode() to generate training data.
        states, actions, returns, _ = self.generate_episode(env, gamma=gamma)

        self.model.train_on_batch(states, returns)

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
        while not done:

            if render:
                env.render()

            states.append(state)

            action_dist = self.model.predict(np.array([state]))[0]

            action = np.random.choice(env.action_space.n, 1, p=action_dist)[0]
            actions.append(action)
            state, reward, done, _ = env.step(action)
            rewards.append(reward/1000.0)
        print(np.sum(rewards)*1000)
        sum_reward_100_episodes = np.sum(rewards)*1000
        T = len(rewards)
        returns = np.zeros((T, env.action_space.n))
        for t in range(T-1, -1, -1):
            for k in range(t, T):
                returns[t][actions[t]] += (gamma**(k-t))*rewards[k]
        returns = np.sum(returns, axis=1)

        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        #
        new_returns = np.zeros((T, env.action_space.n))
        #
        for t in range(T):
            new_returns[t][actions[t]] = returns[t]
        #
        return np.array(states), np.array(actions), new_returns, sum_reward_100_episodes


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-episodes', dest='num_episodes', type=int,
                        default=50000, help="Number of episodes to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=5e-4, help="The learning rate.")

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
    render = args.render

    # Create the environment.
    env = gym.make('LunarLander-v2')
    # env.seed(0)

    test_mean_reward = []
    test_reward_std = []

    # TODO: Create the model.
    # current 1e-3
    AI = Reinforce(75e-5, env)
    # TODO: Train the model using REINFORCE and plot ithe learning curve.
    x_space = []
    print('75e5')
    for i in range(num_episodes):
        print(i)
        if (i % 500) == 0:
            mean, std = test_model(AI, env, 0.99)
            test_mean_reward.append(mean)
            test_reward_std.append(std)
            x_space.append(i)
            print("MEAN REWARD", mean)
            fig = plt.figure()
            plt.xlabel('Number of episodes trained on')
            plt.ylabel('Mean cumulative reward on 100 test episodes')
            plt.errorbar(x_space, test_mean_reward, yerr=test_reward_std)
            fig.savefig('reinforce-75e-5.png')
        AI.train(env, 0.99)
    fig = plt.figure()
    plt.errorbar(x_space, test_mean_reward, yerr=test_reward_std)
    fig.savefig('reinforce-75e-5.png')
    fig = plt.figure()
    plt.errorbar(x_space, test_mean_reward, yerr=test_reward_std)
    plt.xlabel('Number of episodes trained on')
    plt.ylabel('Mean cumulative reward on 100 test episodes')

    # plt.errorbar(x_space, test_mean_reward, yerr=test_reward_std)
    fig.savefig('reinforce-75e-5.png')


if __name__ == '__main__':
    main(sys.argv)
