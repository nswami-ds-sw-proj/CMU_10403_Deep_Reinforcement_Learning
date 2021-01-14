# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import torch

from .ReplayBuffer import ReplayBuffer
from .ActorNetwork import ActorNetwork
from .CriticNetwork import CriticNetwork

BUFFER_SIZE = 1000000
BATCH_SIZE = 1024
GAMMA = 0.98                    # Discount for rewards.
TAU = 0.05                      # Target network update rate.
LEARNING_RATE_ACTOR = 0.0001
LEARNING_RATE_CRITIC = 0.001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EpsilonNormalActionNoise(object):
    """A class for adding noise to the actions for exploration."""

    def __init__(self, mu, sigma, epsilon):
        """Initialize the class.

        Args:
            mu: (float) mean of the noise (probably 0).
            sigma: (float) std dev of the noise.
            epsilon: (float) probability in range [0, 1] with
            which to add noise.
        """
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon

    def __call__(self, action):
        """With probability epsilon, adds random noise to the action.
        Args:
            action: a batched tensor storing the action.
        Returns:
            noisy_action: a batched tensor storing the action.
        """
        if np.random.uniform() > self.epsilon:
            return action + np.random.normal(self.mu, self.sigma)
        else:
            return np.random.uniform(-1.0, 1.0, size=action.shape)


class DDPG(object):
    """A class for running the DDPG algorithm."""

    def __init__(self, env, outfile_name):
        """Initialize the DDPG object.

        Args:
            env: an instance of gym.Env on which we aim to learn a policy.
            outfile_name: (str) name of the output filename.
        """
        action_dim = len(env.action_space.low)
        state_dim = len(env.observation_space.low)
        np.random.seed(1337)
        self.env = env

        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.target_actor = ActorNetwork(state_dim, action_dim).to(device)
        self.target_critic = CriticNetwork(state_dim, action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.optimize_c = torch.optim.Adam(self.critic.parameters(),
                                           lr=LEARNING_RATE_CRITIC)
        self.optimize_a = torch.optim.Adam(self.actor.parameters(),
                                           lr=LEARNING_RATE_ACTOR)

        self.mse = torch.nn.MSELoss()

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self.outfile = outfile_name

    def evaluate(self, num_episodes, episode=None, t_means=None, t_stds=None):
        """Evaluate the policy. Noise is not added during evaluation.

        Args:
            num_episodes: (int) number of evaluation episodes.
        Returns:
            success_rate: (float) fraction of episodes that were successful.
            average_return: (float) Average cumulative return.
        """

        self.actor.eval()
        self.critic.eval()

        test_rewards = []
        success_vec = []
        #plt.figure(figsize=(12, 12))
        for i in range(num_episodes):
            s_vec = []
            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            success = False
            while not done:
                s_vec.append(s_t)

                state = np.array([state])

                a_t = self.actor(torch.from_numpy(state).float().to(device)).cpu().detach().numpy()

                a_t = np.squeeze(a_t)

                new_s, r_t, done, info = self.env.step(a_t)
                if done and "goal" in info["done"]:
                    success = True
                new_s = np.array(new_s)
                total_reward += r_t
                s_t = new_s
                step += 1

                state = new_s

            success_vec.append(success)
            test_rewards.append(total_reward)
            """if i < 9:
                plt.subplot(3, 3, i+1)
                s_vec = np.array(s_vec)
                pusher_vec = s_vec[:, :2]
                puck_vec = s_vec[:, 2:4]
                goal_vec = s_vec[:, 4:]
                plt.plot(pusher_vec[:, 0], pusher_vec[:, 1], '-o', label='pusher')
                plt.plot(puck_vec[:, 0], puck_vec[:, 1], '-o', label='puck')
                plt.plot(goal_vec[:, 0], goal_vec[:, 1], '*', label='goal', markersize=10)
                plt.plot([0, 5, 5, 0, 0], [0, 0, 5, 5, 0], 'k-', linewidth=3)
                plt.fill_between([-1, 6], [-1, -1], [6, 6], alpha=0.1,
                                 color='g' if success else 'r')
                plt.xlim([-1, 6])
                plt.ylim([-1, 6])
                if i == 0:
                    plt.legend(loc='lower left', fontsize=28, ncol=3, bbox_to_anchor=(0.1, 1.0))
                if i == 8:
                    # Comment out the line below to disable plotting.
                    continue
                    plt.show()
            """

        if episode != None:
            t_means[int(episode / 100)] = np.mean(success_vec)
            t_stds[int(episode / 100)] = np.std(success_vec)

        return np.mean(success_vec), np.mean(test_rewards), t_means, t_stds

    def train(self, num_episodes, hindsight=False):
        """Runs the DDPG algorithm.

        Args:
            num_episodes: (int) Number of training episodes.
            hindsight: (bool) Whether to use HER.
        """

        mu = 0
        sigma = 0.01
        epsilon = 0.2
        total_successes = 0
        test_means = np.zeros(int(num_episodes / 100))
        test_stds = np.zeros(int(num_episodes / 100))

        for i in range(num_episodes):

            self.actor.train()
            self.critic.train()
            self.target_actor.eval()
            self.target_critic.eval()

            state = self.env.reset()
            s_t = np.array(state)
            total_reward = 0.0
            done = False
            step = 0
            loss = 0.0
            store_states = []
            store_actions = []

            # sigma *= 0.999
            # epsilon *= 0.995

            noise = EpsilonNormalActionNoise(mu=mu, sigma=sigma, epsilon=epsilon)
            while not done:
                # Collect one episode of experience, saving the states and actions
                # to store_states and store_actions, respectively.

                self.actor.eval()
                self.critic.eval()
                self.target_actor.eval()
                self.target_critic.eval()

                store_states.append(state)
                state = np.array([state])

                action = self.actor(torch.from_numpy(state).float().to(device)).cpu().detach().numpy()
                action = np.squeeze(action)

                action = noise(action)
                store_actions.append(action)
                new_s, reward, done, _ = self.env.step(action)
                total_reward += reward

                if done and reward == 0:
                    total_successes += 1

                transition = (state, action, reward, new_s, done)
    
                self.replay_buffer.append(transition)

                state = np.array(new_s)

                sample = self.replay_buffer.sample(BATCH_SIZE)
                #print(sample)
                if len(sample) < 2:
                    continue

                states = []
                actions = []
                rewards = []
                new_states = []
                dones = []
                for transition in sample:
                    states.append(transition[0])
                    actions.append(transition[1])
                    rewards.append([transition[2]])
                    new_states.append(transition[3])
                    dones.append([transition[4]])

                self.actor.train()
                self.critic.train()

                states = torch.FloatTensor(states).to(device)
                actions = torch.FloatTensor(actions).to(device)
                rewards = torch.FloatTensor(rewards).to(device)
                new_states = torch.FloatTensor(new_states).to(device)
                dones = torch.FloatTensor(dones).to(device)

                states = torch.squeeze(states, dim=1)

                q = self.critic(states, actions)
                q_next_s = self.target_critic(new_states, self.target_actor(new_states).detach())
                q_p = rewards + (1 - dones) * GAMMA * q_next_s

                c_loss = self.mse(q, q_p)
                a_loss = -self.critic(states, self.actor(states)).mean()
                self.optimize_a.zero_grad()
                a_loss.backward()
                self.optimize_a.step()
                self.optimize_c.zero_grad()
                c_loss.backward()
                self.optimize_c.step()

                self.update_targets()

                total_reward += reward
                loss += a_loss.mean()
                step += 1

            if hindsight:
                # For HER, we also want to save the final next_state.
                store_states.append(new_s)
                self.add_hindsight_replay_experience(store_states,
                                                     store_actions)
            del store_states, store_actions
            store_states, store_actions = [], []

            # Logging
            print("Episode %d: Total reward = %d Successes: %d" %
                  (i, total_reward, total_successes))
            print("\tTD loss = %.2f" % (loss / step,))
            #print("\tSteps = %d; Info = %s" % (step, info['done']))
            if i % 100 == 0:
                successes, mean_rewards, test_means, test_stds = self.evaluate(
                    20, i, test_means, test_stds)
                print('Evaluation: success = %.2f; return = %.2f' % (successes, mean_rewards))
                with open(self.outfile, "a") as f:
                    f.write("%.2f, %.2f,\n" % (successes, mean_rewards))
                np.save("ep" + str(i) + "_means.npy", test_means)
                np.save("ep" + str(i) + "_stds.npy", test_stds)

    def update_targets(self):
        for weight, t_weight in zip(self.critic.parameters(), self.target_critic.parameters()):
            new_weight = (1 - TAU) * t_weight.data + TAU * weight.data
            t_weight.data.copy_(new_weight)

        for weight, t_weight in zip(self.actor.parameters(), self.target_actor.parameters()):
            new_weight = (1 - TAU) * t_weight.data + TAU * weight.data
            t_weight.data.copy_(new_weight)

    def add_hindsight_replay_experience(self, states, actions):
        """Relabels a trajectory using HER.

        Args:
            states: a list of states.
            actions: a list of states.
        """
        T = len(actions)
        her_states, her_rewards = self.env.apply_hindsight(states)
    
        assert(len(her_states)==T+1)
        for i in range(T):
            if (i==T-1):
                transition = (np.array([her_states[i]]), actions[i], her_rewards[i], her_states[i+1], True)
                
            else:
                #assert(her_rewards[i]==-1)
                transition = (np.array([her_states[i]]), actions[i], her_rewards[i], her_states[i+1], False)
            self.replay_buffer.append(transition)

