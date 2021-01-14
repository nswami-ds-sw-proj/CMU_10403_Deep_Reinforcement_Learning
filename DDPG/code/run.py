import gym
import envs
from algo.ddpg import DDPG
import numpy as np
from matplotlib import pyplot as plt


def main():
    #print(np.load('ep4500_means.npy')[39])
    #assert(False)
   # mean = np.load('ep12100_means.npy')[:121]
   # std = np.load('ep12100_stds.npy')[:121]
   # print(mean)
   # assert(False) 
   # container = plt.errorbar(list(range(0, 12000, 100)), mean, yerr=std, ecolor='red')
   # plt.xlim(0, 12000)
   # plt.ylim(-0.1, 1.1)
   # plt.xlabel("Episodes")
   # plt.ylabel("Test Mean Reward")
   # plt.title("DDPG+HER Test Mean Reward by Episode")
 
   # plt.savefig("DDPG-HER.png")
    
   env = gym.make('Pushing2D-v0')
   algo = DDPG(env, 'ddpg_log.txt')
   algo.train(50000, hindsight=True)


if __name__ == '__main__':
    main()
