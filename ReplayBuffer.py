from collections import deque
import random


class ReplayBuffer():

    def __init__(self, memory_size, burn_in=None):

        # The memory essentially stores transitions recorder from the agent
        # taking actions in the environment.

        # Burn in episodes define the number of episodes that are written into the memory from the
        # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
        # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

        # storage of the actual memory
        self.memory = deque(maxlen=memory_size)
        self.burn_in = burn_in
        if burn_in == None:
            self.burn_in = int(.2 * memory_size)

    def __len__(self):
        return len(list(self.memory))

    def sample(self, batch_size):
        # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
        # You will feed this to your model to train.
        memory = list(self.memory)
        sampled_data = random.sample(memory, min(batch_size, len(self.memory)))
        return sampled_data

    def append(self, transition):
        # Appends transition to the memory.

        self.memory.append(transition)
