

# Instructions
To run this implementation of Deep Deterministic Policy Gradients, run the file run.py. 
To edit any of the architectures for the Actor or Critic Network, edit their respective _init_ functions in their respective files.
To edit any other hyperparamters of this implementation, including whether to train on Hindsight Experience Replay, edit them appropriately in ddpg.py.
This implementation was trained to solve the Pushing2d-v0 environment from OpenAI's Gym toolkit. 

# Background
Deep Deterministic Policy Gradients is an off-policy reinforcement learning algorithm tailored for continuous action spaces, consisting of actor and critic networks as in 
Advantage-Actor-Critic, but also accompanied by target actor (policy) and critic (value function (Q)) networks, which slowly track the training progress of the original actor/critic networks. 
In this implementation we rollout the algorithm for a given number of timesteps, store each transition and value yielded in a replay buffer (See ReplayBuffer.py), and
train the networks on sampled mini-batches of these transitions. This implementation additionally has the option of including Hindsight Experience Replay, an option to additionally
store copies of transitions with different goals in mind, in order to alter the reward yielded by these transitions, reducing the reward sparsity of these transitions and providing more efficient
data augmentation for training. 

# Reference Links

https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b
https://arxiv.org/pdf/1707.01495.pdf










