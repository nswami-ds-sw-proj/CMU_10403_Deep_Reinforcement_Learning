#!/usr/bin/env python
import keras, tensorflow as tf, numpy as np, gym 
import sys, copy, argparse, collections, random, operator, os

#learning rate scheduler for cartpole
def schedule_cartpole(epoch, lr):
	if epoch != 0 and epoch % 10000 == 0:
		return max(0.5 * lr, 0.00005)
	return lr

#learning rate scheduler for mountaincar
def schedule_mountaincar(epoch, lr):
	if epoch != 0 and epoch % 5000 == 0:
		return max(0.75 * lr, 0.0001)
	return lr

class QNetwork():

	# This class essentially defines the network architecture. 
	# The network should take in state of the world as an input, 
	# and output Q values of the actions available to the agent as the output. 

	def __init__(self, environment_name):
		# hyperparamaters
		self.lr = 0.001 if environment_name == "CartPole-v0" else 0.00025
		self.gamma = 0.99 if environment_name == "CartPole-v0" else 0.9
		self.epsilon = 1.0 
		self.epsilon_min = 0.05 
		self.decay = .995 if environment_name == "CartPole-v0" else .9995
		self.num_episodes = 50000
		self.buffer_size = 50000 if environment_name == "CartPole-v0" else 100000
		self.batch_size = 32 

		# number of steps to take for a given action
		self.action_repeat = 4 if environment_name == "CartPole-v0" else 20
		# number of episodes that pass before updating the target network
		self.target_update = 1 if environment_name == "CartPole-v0" else 100
		# number of most recent transitions to have in each batch
		self.live_update = 0 if environment_name == "CartPole-v0" else 4
		# if none, every transition has equal sampling probability, otherwise 
		# this is how much more likely we are to sample higher-reward transitions
		self.sample_scalar = None if environment_name == "CartPole-v0" else 5

		self.optimizer = keras.optimizers.Adam(lr=self.lr) 
		self.env = gym.make(environment_name)

		space_size = len(self.env.observation_space.high)

		# the training model
		self.model = keras.Sequential()
		self.model.add(keras.layers.Dense(24, activation='relu', init='lecun_uniform', input_dim=space_size))
		self.model.add(keras.layers.Dense(48, activation='relu', init='lecun_uniform'))
		self.model.add(keras.layers.Dense(self.env.action_space.n, activation='linear'))
		self.model.compile(loss="mse", optimizer=self.optimizer)

		# the target model for for Double DQN
		self.target_model = keras.Sequential()
		self.target_model.add(keras.layers.Dense(24, activation='relu', input_dim=space_size))
		self.target_model.add(keras.layers.Dense(48, activation='relu'))
		self.target_model.add(keras.layers.Dense(self.env.action_space.n, activation='linear'))
		self.target_model.compile(loss="mse", optimizer=self.optimizer)

		# setting target model to have the same weights as the training model
		self.target_model.set_weights(self.model.get_weights())

	def save_model_weights(self, suffix):
		# Helper function to save your model / weights. 
		self.model.save_weights(suffix + ".h5")

	def load_model(self, model_file):
		# Helper function to load an existing model.
		# e.g.: torch.save(self.model.state_dict(), model_file)
		self.model = keras.models.load_model(model_file)

	def load_model_weights(self, weight_file):
		# Helper funciton to load model weights. 
		# e.g.: self.model.load_state_dict(torch.load(model_file))
		self.model.load_weights(weight_file)

	# predicts from the training model
	def predict(self, data):
		return self.model.predict(np.array([data]))[0]

	# predicts from the target model
	def target_predict(self, data):
		return self.target_model.predict(np.array([data]))[0]

	#steps the environment self.action_repeat steps
	def env_step(self, action):
		reward = 0.0
		for i in range(self.action_repeat):
			next_state, new_reward, done, info = self.env.step(action)
			reward += new_reward
			if done: break
		return next_state, reward, done, info


class Replay_Memory():

	def __init__(self, memory_size=50000, burn_in=10000):

		# The memory essentially stores transitions recorder from the agent
		# taking actions in the environment.

		# Burn in episodes define the number of episodes that are written into the memory from the 
		# randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced. 
		# A simple (if not the most efficient) was to implement the memory is as a list of transitions. 
		
		# storage of the actual memory
		self.memory = collections.deque(maxlen=memory_size)
		# burn in size
		self.burn_in = burn_in

	def sample_batch(self, batch_size=32, recent_update=0, scale_positive=None):
		# This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples. 
		# You will feed this to your model to train.
		memory = list(self.memory)

		# if scale_positive is not None then we upscale higher-reward transitions
		if scale_positive != None:
			rewards = np.array([x[2] for x in memory])
			sample = np.ones(len(memory))
			sample[np.where(rewards > np.median(rewards))] *= scale_positive
			sample /= sum(sample)
			choices = list(np.random.choice(len(memory),
								size = batch_size - recent_update,
								p = sample))
			sampled_data = []
			for i in choices:
				sampled_data.append(memory[i])
		else:
			sampled_data = random.sample(memory, batch_size - recent_update)

		# if we choose to sampple the most recent transitions, then this adds those in
		if recent_update != 0:
			sampled_data.extend(memory[-recent_update:])
		return sampled_data

	def append(self, transition):
		# Appends transition to the memory. 	
		self.memory.append(transition)


class DQN_Agent():

	# In this class, we will implement functions to do the following. 
	# (1) Create an instance of the Q Network class.
	# (2) Create a function that constructs a policy from the Q values predicted by the Q Network. 
	#		(a) Epsilon Greedy Policy.
	# 		(b) Greedy Policy. 
	# (3) Create a function to train the Q Network, by interacting with the environment.
	# (4) Create a function to test the Q Network's performance on the environment.
	# (5) Create a function for Experience Replay.
	
	def __init__(self, environment_name, render=False, replay_memory=True):

		# Create an instance of the network itself, as well as the memory. 
		# Here is also a good place to set environmental parameters,
		# as well as training parameters - number of episodes / iterations, etc. 
		self.network = QNetwork(environment_name)
		self.environment_name = environment_name
		self.render = render
		self.replay = replay_memory
		if replay_memory:
			self.replay_memory = Replay_Memory(memory_size=self.network.buffer_size)

		if self.environment_name == "MountainCar-v0":
			self.replay_memory.burn_in = 50000


	def epsilon_greedy_policy(self, q_values, epsilon=None):
		# Creating epsilon greedy probabilities to sample from.      
		if epsilon == None: epsilon = self.network.epsilon       
		rand = np.random.uniform()
		if rand < epsilon: return np.random.randint(0, len(q_values))
		return np.argmax(q_values)


	def greedy_policy(self, q_values):
		# Creating greedy policy for test time. 
		return np.argmax(q_values)


	def train(self):
		# In this function, we will train our network. 
		# If training without experience replay_memory, then you will interact with the environment 
		# in this function, while also updating your network parameters. 

		# When use replay memory, you should interact with environment here, and store these 
		# transitions to memory, while also updating your model.

		# burn in the memory and reset the environment
		self.burn_in_memory()
		self.network.env.reset()

		# if the environment is carpole, use the cartpole learning rate scheduler
		# otherwise use the mountaincar scheduer
		if self.environment_name == "CartPole-v0":
			scheduler = keras.callbacks.LearningRateScheduler(schedule_cartpole, verbose=0)
		else:
			scheduler = keras.callbacks.LearningRateScheduler(schedule_mountaincar, verbose=0)

		action_space_size = self.network.env.action_space.n
		state_size = self.network.env.observation_space.shape[0]
		success_sum = 0

		test_reward = np.zeros(int(self.network.num_episodes / 100))
		td_error = np.zeros(int(self.network.num_episodes / 100))

		# iterate through all of the episodes
		for episode in range(self.network.num_episodes):

			# decay epsilon
			self.network.epsilon *= self.network.decay
			self.network.epsilon = max(self.network.epsilon, self.network.epsilon_min)

			done = False
			self.network.env.reset()

			# this will be the training data
			x = []
			y = []

			episode_length = 0
			episode_reward = 0.0

			# iterate through a new episode
			while not done:
				# render the episode if we want to see it, renders every 100 episodes
				if self.render and episode != 0 and episode % 100 == 0: self.network.env.render()

				state = self.network.env.state
				q_values = self.network.predict(state)
				action = self.epsilon_greedy_policy(q_values)

				next_state, reward, done, _ = self.network.env_step(action)

				episode_reward += reward
				transition = (state, action, reward, next_state, done)

				# if we have a replay memory, add the transition to the memory
				if self.replay:
					self.replay_memory.append(transition)
				else:
					# using both the target model and training model
					new_y = [0.0 for val in range(action_space_size)]
					pred_target = self.network.target_predict(next_state)
					pred_model = self.network.predict(next_state)

					# use just the reward if in a terminal state, otherwise we take the smaller of the 
					# max of the target model and training model's predictions
					if done:
						new_y[action] = reward
					else:
						new_y[action] = reward + self.network.gamma * min(np.max(pred_target), np.max(pred_model))
				
					# building the training dataset
					y.append(new_y)
					x.append(state)

				episode_length += 1

			if self.replay:
				transitions = self.replay_memory.sample_batch(batch_size=self.network.batch_size,
																recent_update=self.network.live_update,
																scale_positive=self.network.sample_scalar)

				# iterate through the sampled transitions
				for i in range(len(transitions)):
					(state, action, reward, next_state, done) = transitions[i]

					# using both the target model and training model
					new_y = [0.0 for val in range(action_space_size)]
					pred_target = self.network.target_predict(next_state)
					pred_model = self.network.predict(next_state)

					# use just the reward if in a terminal state, otherwise we take the smaller of the 
					# max of the target model and training model's predictions
					if done:
						new_y[action] = reward
					else:
						new_y[action] = reward + self.network.gamma * min(np.max(pred_target), np.max(pred_model))
					
					# building the training dataset
					y.append(new_y)
					x.append(state)

			x = np.array(x)
			y = np.array(y)

			# recording the number of successes
			if self.environment_name != "CartPole-v0" and episode_reward > -200: success_sum += 1
			if self.environment_name == "CartPole-v0" and episode_reward == 200: success_sum += 1

			# if we have a lr scheduler then we fit the model using it
			if scheduler != None:
				self.network.model.fit(x, y, batch_size=self.network.batch_size, 
									epochs=1, verbose=0, callbacks=[scheduler])
			else:
				self.network.model.fit(x, y, batch_size=self.network.batch_size, 
									epochs=1, verbose=0)

			# calculating TD error
			if episode % 100 == 0:
				transitions = self.replay_memory.sample_batch(batch_size=100)

				total_error = 0.0

				for i in range(len(transitions)):
					(state, action, reward, next_state, done) = transitions[i]
					pred_s_next = np.max(self.network.predict(next_state))
					pred_s = np.max(self.network.predict(state))

					total_error += reward + self.network.gamma * pred_s_next - pred_s

				td_error[int(episode / 100)] = total_error / 100.0

			# updating the target model weights using the training model
			if episode % self.network.target_update == 0:
				self.network.target_model.set_weights(self.network.model.get_weights())
				
			# print training metrics every 10 episodes
			if episode % 100 == 0:
				print("episode: " + str(episode) + ", train reward: " + str(round(episode_reward, 3)) + ", epsilon: " + str(self.network.epsilon)[:5] + ", successes: " + str(success_sum))

			# save the model weights every 1000 episodes
			if episode != 0 and episode % 1000 == 0:
				self.network.save_model_weights("ep_" + str(episode) + "_" + self.environment_name)

			if episode % 100 == 0:
				new_test_reward = self.test()
				print("episode:", str(episode), "loss:", str(new_test_reward))
				test_reward[int(episode / 100)] = new_test_reward
				np.save(str(self.environment_name) + "_" + "test_reward.npy", test_reward)
				np.save(str(self.environment_name) + "_" + "td_error.npy", td_error)

		# save the final model
		self.network.save_model_weights("ep_" + str(self.network.num_episodes) + "_" + self.environment_name)

		self.network.env.reset()


	def test(self, model_file=None):
		# Evaluate the performance of your agent over 100 episodes, by calculating cummulative rewards for the 100 episodes.
		# Here you need to interact with the environment, irrespective of whether you are using a memory. 
		
		# if we feed in a model file, we will load using that model file
		if model_file != None:
			self.network.model.load_model(model_file)

		# initializing the reward array
		rewards = [0.0 for x in range(100)]

		# iterating through the 100 episodes
		for i in range(100):
			done = False
			self.network.env.reset()

			ep_reward = []

			# run the episode using an epsilon greedy policy with epislon = 0.05
			while not done:
				state = self.network.env.state
				q_values = self.network.predict(state)
				action = self.epsilon_greedy_policy(q_values, epsilon=0.05)
				next_state, reward, done, _ = self.network.env.step(action)
				ep_reward.append(reward)

			rewards[i] = np.sum(ep_reward)

		# return the mean reward
		return np.mean(rewards)

	def burn_in_memory(self):
		# Initialize your replay memory with a burn_in number of episodes / transitions.

		# if the model isn't using a replay memory, then we return
		if not self.replay: return
		self.network.env.reset()

		# initialize the memory with a random policy of size burn_in
		done = False
		for i in range(self.replay_memory.burn_in):
			if done: self.network.env.reset()
			state = self.network.env.state
			action = self.network.env.action_space.sample()
			next_state, reward, done, _ = self.network.env_step(action)
			self.replay_memory.append((state, action, reward, next_state, done))


# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop. 
def test_video(agent, env, epi):
	# Usage: 
	# 	you can pass the arguments within agent.train() as:
	# 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.network.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        q_values = agent.network.predict(state)
        action = agent.epsilon_greedy_policy(q_values, 0.05)
        next_state, reward, done, info = agent.network.env_step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.network.env.close()


def parse_arguments():
	parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
	parser.add_argument('--env',dest='env',type=str)
	parser.add_argument('--render',dest='render',type=int,default=0)
	parser.add_argument('--train',dest='train',type=int,default=1)
	parser.add_argument('--model',dest='model_file',type=str)
	return parser.parse_args()


def main(args):

	random.seed(10403)

	args = parse_arguments()
	environment_name = args.env

	# Setting the session to allow growth, so it doesn't allocate all GPU memory. 
	try:
		gpu_ops = tf.GPUOptions(allow_growth=True)
		config = tf.ConfigProto(gpu_options=gpu_ops)
		sess = tf.Session(config=config)

		# Setting this as the default tensorflow session. 
		keras.backend.tensorflow_backend.set_session(sess)
	except:
		None

	print("TRAINING")

	agent = DQN_Agent(str(args.env), render=args.render)

	agent.network.env.seed(10403)

	if args.model_file:
		agent.network.load_model_weights(str(args.model_file))
	if args.train:
		agent.train()

	# agent = DQN_Agent("CartPole-v0")
	# agent.network.model.load_weights("ep_36000_CartPole-v0.h5")
	# test_video(agent, agent.environment_name, 0)

if __name__ == '__main__':
	main(sys.argv)

