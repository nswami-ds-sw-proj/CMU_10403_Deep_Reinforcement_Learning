

from collections import OrderedDict 
import gym
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import random
import tensorflow as tf

"""### Make the TF Policy Model
We'll use the same architecture for each of the problems. By implementing a function that creates the model here, you won't need to implement it again for each problem.
"""

def make_model():
  model = Sequential()      
  # WRITE CODE HERE
  model.add(Dense(10, activation=tf.nn.tanh, input_dim=4))
  model.add(Dense(2, activation=tf.nn.softmax))
  # Add layers to the model:
  # a fully connected layer with 10 units
  # a tanh activation
  # another fully connected layer with 2 units (the number of actions)
  # a softmax activation (so the output is a proper distribution)
  model.compile(loss='categorical_crossentropy',
                     optimizer=tf.train.AdamOptimizer(),
                     metrics=['accuracy'])
  
  # We expect the model to have four weight variables (a kernel and bias for
  # both layers)
  assert len(model.weights) == 4, 'Model should have 4 weights.'
  return model

"""### Test the model
To confirm that the model is correct, we'll use it to solve a binary classification problem. The target function $f: \mathbb{R}^4 \rightarrow {0, 1}$ indicates whether the sum of the vector coordinates is positive:
$$f(x) = \delta \left(\sum_{i=1}^4 x_i > 0 \right)$$

You should achieve an accuracy of at least 98%.
"""

model = make_model()
for t in range(20):
  X = np.random.normal(size=(1000, 4))  # some random data
  is_positive = np.sum(X, axis=1) > 0  # A simple binary function
  Y = np.zeros((1000, 2))
  Y[np.arange(1000), is_positive.astype(int)] = 1  # one-hot labels
  history = model.fit(X, Y, epochs=10, batch_size=256, verbose=0)
  loss = history.history['loss'][-1]
  acc = history.history['acc'][-1]
  print('(%d) loss= %.3f; accuracy = %.1f%%' % (t, loss, 100 * acc))

"""### Interacting with the OpenAI Gym
http://gym.openai.com/docs/#environments
"""

def action_to_one_hot(env, action):
    action_vec = np.zeros(env.action_space.n)
    action_vec[action] = 1
    return action_vec    
      
      
def generate_episode(env, policy):
  """Collects one rollout from the policy in an environment. The environment
  should implement the OpenAI Gym interface. A rollout ends when done=True. The
  number of states and actions should be the same, so you should not include
  the final state when done=True.

  Args:
    env: an OpenAI Gym environment.
    policy: a keras model
  Returns:
    states: a list of states visited by the agent.
    actions: a list of actions taken by the agent. While the original actions
      are discrete, it will be helpful to use a one-hot encoding. The actions
      that you return should be one-hot vectors (use action_to_one_hot())
    rewards: the reward received by the agent at each step.
  """
  done = False
  state = env.reset()

  states = []
  actions = []
  rewards = []
  while not done:
      # WRITE CODE HERE
      states.append(state)
      action = policy.predict_classes(np.array([state]))[0]
      state, reward, done, info = env.step(action)
      action = action_to_one_hot(env, action)
      actions.append(action)
      rewards.append(reward)
  return np.array(states), np.array(actions), np.array(rewards)

"""### Test the data collection
Run the following cell and make sure you see "Test passed!"
"""

# Create the environment.
env = gym.make('CartPole-v0')
policy = make_model()
states, actions, rewards = generate_episode(env, policy)
assert len(states) == len(actions), 'Number of states and actions should be equal.'
assert len(actions) == len(rewards), 'Number of actions and rewards should be equal.'
assert len(actions[0]) == 2, 'Actions should use one-hot encoding.'
print('Test passed!')

"""Bayesian Optimization
"""

MIN_X = -1.
MAX_X = 1.

# Squared exponential kernel
def kernel(X1, X2, l=0.2, sigma_f=0.01):
  sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
  return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)

def posterior_predictive(X_test, X_train, Y_train, l=0.2, sigma_f=0.01, sigma_y=1e-4):
  mu_test = 0.*X_test
  cov_test = np.eye(len(X_test))
  ###############################################################
  K = kernel(X_train, X_train, l=l, sigma_f=sigma_f)
  K_star = kernel(X_train, X_test, l=l, sigma_f=sigma_f)
  K_star_star = kernel(X_test, X_test, l=l, sigma_f=sigma_f)
  K_y = K + sigma_y * np.eye(len(X_train))
  mu_test = np.dot(np.dot(np.transpose(K_star), np.linalg.inv(K_y)), Y_train)
  cov_test = K_star_star - np.dot(np.dot(np.transpose(K_star), np.linalg.inv(K_y)), K_star)
  ###############################################################
  return mu_test, cov_test

def gp_sample(mu, cov):
  Y = 0.
  ###############################################################
  Y = np.random.multivariate_normal(mu.ravel(), cov)
  ###############################################################
  return Y

# This class defines the function we want to optimize
class FitnessFunction:
  def __init__(self):
    self._X = np.sort(np.random.uniform(MIN_X, MAX_X, size=(200,1)), 0)
    mu = np.zeros(self._X.shape)
    cov = kernel(self._X, self._X)
    self._Y = gp_sample(mu, cov)
    self._max_y = np.max( self._Y)

  def evaluate(self, x, sigma_y=1e-4):
    X = np.array(x).reshape(-1, 1)
    Y, _ = posterior_predictive(X, self._X, self._Y, sigma_y=sigma_y)
    return Y[0]

  def regret(self, y):
    return self._max_y - y

def gp_greedy(X_train, Y_train):
  X_train = np.array(X_train)
  Y_train = np.array(Y_train)
  new_x = 0.
  ###############################################################
  grid = np.linspace(MIN_X, MAX_X, num=100).reshape(100, 1)
  mu, cov = posterior_predictive(grid, X_train, Y_train)
  new_x = grid[np.argmax(mu)]
  ###############################################################
  return new_x

def gp_ucb(X_train, Y_train):
  X_train = np.array(X_train)
  Y_train = np.array(Y_train)
  new_x = 0.
  ###############################################################
  grid = np.linspace(MIN_X, MAX_X, num=100).reshape(100, 1)
  mu, cov = posterior_predictive(grid, X_train, Y_train)
  new_x = grid[np.argmax(mu + 1.96 * np.diagonal(cov))]
  ###############################################################
  return new_x

def gp_thompson(X_train, Y_train):
  X_train = np.array(X_train)
  Y_train = np.array(Y_train)
  new_x = 0.
  ###############################################################
  grid = np.linspace(MIN_X, MAX_X, num=100).reshape(100, 1)
  mu, cov = posterior_predictive(grid, X_train, Y_train)
  f = gp_sample(mu, cov)
  new_x = grid[np.argmax(f)]
  ###############################################################
  return new_x

def bayes_opt(n_trials=200, n_steps=50):
  acquisition_funcs = {
      'greedy': gp_greedy,
      'ucb': gp_ucb,
      'thompson': gp_thompson,
  }
  
  for acquisition_label in ['greedy', 'ucb', 'thompson']:
    acquisition_func = acquisition_funcs[acquisition_label]
    
    regret = []
    for trial in range(n_trials):
      fitness = FitnessFunction()
      
      X_trial = [np.random.uniform(-1, 1, size=(1))]
      Y_trial = [fitness.evaluate(X_trial[0])]
      trial_regret = [fitness.regret(Y_trial[0])]
      for t in range(n_steps-1):
        new_x = acquisition_func(X_trial, Y_trial)
        new_y = fitness.evaluate(new_x)
        X_trial.append(new_x)
        Y_trial.append(new_y)
        trial_regret.append(fitness.regret(new_y))
      regret.append(trial_regret)
    plt.plot(np.arange(n_steps), np.mean(np.array(regret), 0), label=acquisition_label)
  plt.legend()

bayes_opt()

"""## CMA-ES 
"""

class CMAES:
    def __init__(self, env, L, n, p, sigma, noise, reward_fn=None):
        """
        Args:
          env: environment with which to evaluate different weights
          L: number of episodes used to evaluate a given set of weights
          n: number of members (weights) in each generation
          p: proportion of members used to update the mean and covariance
          sigma: initial std
          noise: additional noise to add to covariance
          reward_fn: if specified, this reward function is used instead of the 
            default environment reward. This will be used in Problem 3, when the
            reward function will come from the discriminator. The reward
            function should be a function of the state and action.
        """

        self.env = env
        self.model = make_model()
        # Let d be the dimension of the 1d vector of weights.
        self.d = sum(int(np.product(w.shape)) for w in self.model.weights)
        self.mu = np.zeros(self.d)
        self.S = sigma**2 * np.eye(self.d)

        self.L = L
        self.n = n
        self.p = p
        self.noise = noise
        self.reward_fn = reward_fn


    def populate(self):
        """
        """
        self.population = []

        self.population = np.random.multivariate_normal(self.mu, self.S, size=self.n)


    def set_weight(self, member):
        ind = 0
        weights = []
        for w in self.model.weights:
          if len(w.shape) > 1:
            mat = member[ind:ind+int(w.shape[0]*w.shape[1])]
            mat = np.reshape(mat, w.shape)
            ind += int(w.shape[0]*w.shape[1])
          else:
            mat = member[ind:ind+int(w.shape[0])]
            ind += int(w.shape[0])
          weights.append(mat)

        self.model.set_weights(weights)
        

    def evaluate(self, member, num_episodes):
        """
        """
        self.set_weight(member)
        return self.evaluate_policy(self.model, num_episodes)
    
    def evaluate_policy(self, policy, num_episodes):
        episode_rewards = []
        for episode in range(num_episodes):
          # WRITE CODE HERE
          state, action, reward = generate_episode(self.env, policy)
          if self.reward_fn == None:
            episode_rewards.append(sum(reward))
          else:
            reward = []
            for i in range(len(state)):
              reward.append(self.reward_fn(state, action))
            episode_rewards.append(sum(reward) / len(reward))

        return np.mean(episode_rewards)

    def train(self):
        """
        Perform an iteration of CMA-ES by evaluating all members of the
        current population and then updating mu and S with the top self.p
        proportion of members. Note that self.populate automatically deletes
        all the old members, so you don't need to worry about deleting the
        "bad" members.

        """
        self.populate()
        # WRITE CODE HERE
        rewards = []
        for member in self.population:
          reward = self.evaluate(member, self.L)
          rewards.append(reward)

        order = np.argsort(rewards)
        self.population = self.population[order]
        rewards = np.array(rewards)[order]
        top_count = int(self.p * self.n)
        elites = self.population[self.n - top_count - 1:]

        self.mu = np.mean(elites, axis=0)
        self.S = (self.noise * np.eye(self.d)) + np.cov(elites, rowvar=False)
        best_member = self.population[-1]

        best_r = self.evaluate(best_member, 10)
        mu_r = self.evaluate(self.mu, 10)
        return mu_r, best_r

iterations = 200  # Use 10 for debugging, and then try 200 once you've got it working.
#pop_size_vec = [50]             # Start with a population size of 50. Once that
pop_size_vec = [20, 50, 100]  # works, try varying the population size.
                              
data = {pop_size: [] for pop_size in pop_size_vec}

for pop_size in pop_size_vec:
  print('Population size:', pop_size)
  env = gym.make('CartPole-v0')
  optimizer = CMAES(env,
                    L=1,  # number of episodes for evaluation
                    n=pop_size,  # population size
                    p=0.25,  # proportion of population to keep
                    sigma=10,  # initial std dev
                    noise=0.25)  # noise

  for t in range(iterations):
      mu_r, best_r = optimizer.train()
      data[pop_size].append((mu_r, best_r))
      print('(%d) avg member rew = %.2f; best member rew = %.2f' % (t, mu_r, best_r))

      if mu_r >= 200: break

plt.figure(figsize=(12, 4))
# plt.subplot(121)  # Left plot will show performance vs number of iterations
# for pop_size, values in data.items():
#   mu_r = np.array(values)[:, 1]  # Use the performance of the best point
#   x = np.arange(len(mu_r)) + 1
#   plt.plot(x, mu_r, label=str(pop_size))
#   plt.ylabel('average return', fontsize=16)
#   plt.xlabel('num. iterations', fontsize=16)

#Pt2
plt.subplot(121)  # Right plot will show performance vs number of points evaluated
for pop_size, values in data.items():
  mu_r = np.array(values)[:, 1]  # Use the performance of the best point
  x = pop_size * (np.arange(len(mu_r)) + 1)
  plt.plot(x, mu_r, label=str(pop_size))
  plt.ylabel('best return', fontsize=16)
  plt.xlabel('num. points evaluated', fontsize=16)

#Pt1
# plt.subplot(121)  # Left plot will show performance vs number of iterations
# for pop_size, values in data.items():
#   mu_r = np.array(values)[:, 0]  # Use the performance of the best point
#   x = np.arange(len(mu_r)) + 1
#   plt.plot(x, mu_r, label=str(pop_size))
#   plt.ylabel('average return', fontsize=16)
#   plt.xlabel('num. iterations', fontsize=16)

#Pt1 and Pt2
plt.subplot(122)  # Left plot will show performance vs number of iterations
for pop_size, values in data.items():
  mu_r = np.array(values)[:, 1]  # Use the performance of the best point
  x = np.arange(len(mu_r)) + 1
  plt.plot(x, mu_r, label=str(pop_size))
  plt.ylabel('best return', fontsize=16)
  plt.xlabel('num. iterations', fontsize=16)

plt.legend()
plt.tight_layout()
plt.savefig('cmaes_pop_size.png')
plt.show()

