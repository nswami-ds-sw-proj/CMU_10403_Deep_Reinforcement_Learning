import matplotlib.pyplot as plt
import numpy as np
import gym
import envs
import os
import logging

from utils import config
from agent import Agent, RandomPolicy
from mpc import MPC
from cem import CEMOptimizer
from model import PENN

# Logging
now = config.Now()
log = logging.getLogger('root')
log.setLevel('INFO')
log.addHandler(config.MyHandler())

INFO = 1

# Training params
TASK_HORIZON = 40
PLAN_HORIZON = 5

# CEM params
POPSIZE = 200
NUM_ELITES = 20
MAX_ITERS = 5

# Model params
LR = 1e-3

# Dims
STATE_DIM = 8


class ExperimentGTDynamics(object):
    def __init__(self, env_name="Pushing2D-v1", mpc_params=None):
        self.env = gym.make(env_name)
        self.task_horizon = TASK_HORIZON
        self.agent = Agent(self.env)
        self.warmup = False

        # Use Ground Truth Dynamics
        mpc_params["use_gt_dynamics"] = True

        # Model Predictive Contoal (MPC)
        self.cem_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                              use_random_optimizer=False)
        self.random_policy = MPC(self.env, PLAN_HORIZON, None, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                                 use_random_optimizer=True)

    def test(self, num_episodes, optimizer="cem"):
        samples = []
        for j in range(num_episodes):
            if j == 0 or (j + 1) % INFO == 0: log.info("Test episode {}".format(j))
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if optimizer == "cem" else self.random_policy
                )
            )
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0 for sample in samples])
        return avg_return, avg_success


class ExperimentModelDynamics:
    def __init__(self, env_name="Pushing2D-v1", num_nets=1, mpc_params=None):
        self.env = gym.make(env_name)
        self.task_horizon = TASK_HORIZON

        self.agent = Agent(self.env)
        mpc_params["use_gt_dynamics"] = False
        self.model = PENN(num_nets, STATE_DIM, len(self.env.action_space.sample()), LR)  # dynamics model f(s, a, phi)
        self.cem_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                              use_random_optimizer=False)
        self.random_policy = MPC(self.env, PLAN_HORIZON, self.model, POPSIZE, NUM_ELITES, MAX_ITERS, **mpc_params,
                                 use_random_optimizer=True)

    def test(self, num_episodes, optimizer="cem"):
        samples = []
        for j in range(num_episodes):
            if j == 0 or (j + 1) % INFO == 0: log.info("Test episode {}".format(j))
            samples.append(
                self.agent.sample(
                    self.task_horizon, self.cem_policy if optimizer == "cem" else self.random_policy
                )
            )
        avg_return = np.mean([sample["reward_sum"] for sample in samples])
        avg_success = np.mean([sample["rewards"][-1] == 0 for sample in samples])
        return avg_return, avg_success

    def model_warmup(self, num_episodes, num_epochs):
        """ 
        Train a single probabilistic model using a random policy 
            :param num_episodes: randomly sampled episodes for training the a single probabilistic network
            :param num_epochs: epochs to pre-train the ensemble of networks
        """
        traj_obs, traj_acs, traj_rews = [], [], []

        samples = []
        for i in range(num_episodes):
            if i == 0 or (i + 1) % 100 == 0: log.info("Warm up episode %d" % (i))
            samples.append(self.agent.sample(self.task_horizon, self.random_policy))
            
            traj_obs.append(samples[-1]["obs"])
            traj_acs.append(samples[-1]["ac"])
            traj_rews.append(samples[-1]["rewards"])
        
        self.cem_policy.train(
            [sample["obs"] for sample in samples],
            [sample["ac"] for sample in samples],
            [sample["rewards"] for sample in samples],
            epochs=num_epochs
        )
        self.random_policy.has_been_trained = True # This is newly added 

        self.traj_obs, self.traj_acs, self.traj_rews = traj_obs, traj_acs, traj_rews
        self.warmup = True

    def train(self, num_train_epochs, num_episodes_per_epoch, evaluation_interval):
        """ 
        MBRL with PETS (Algorithm 5)
        """
        if self.warmup:
            traj_obs, traj_acs, traj_rews = self.traj_obs, self.traj_acs, self.traj_rews
        else:
            traj_obs, traj_acs, traj_rews = [], [], [], []

        cme_test, rnd_test = [], []
        rmse = []
        gaus = []
        for i in range(num_train_epochs):
            log.info("####################################################################")
            log.info("Starting training epoch %d." % (i + 1))

            samples = []
            for j in range(num_episodes_per_epoch):
                samples.append(
                    self.agent.sample(
                        self.task_horizon, self.cem_policy
                    )
                )
            log.info("Rewards obtained: {}, Rollout Length: {}".format(
                [sample["reward_sum"] for sample in samples], len(samples[-1]["obs"])))
            traj_obs.extend([sample["obs"] for sample in samples])
            traj_acs.extend([sample["ac"] for sample in samples])
            traj_rews.extend([sample["rewards"] for sample in samples])

            (rmse_loss, gaus_loss) = self.cem_policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples],
                epochs=5
            )

            rmse.append(rmse_loss)
            gaus.append(gaus_loss)

            # Q1.2.7: Test with both CEM + MPC & Random Policy + MPC
            if i == 0 or (i + 1) % evaluation_interval == 0:
                avg_return, avg_success = self.test(20, optimizer="cem")
                log.info("Test success CEM + MPC: {}".format(avg_success))
                cme_test.append([avg_return, avg_success])
                avg_return, avg_success = self.test(20, optimizer="random")
                log.info("Test success Random + MPC: {}".format(avg_success))
                rnd_test.append([avg_return, avg_success])

        np.save("rmse_mbrl", np.array(rmse))
        np.save("gaus_mbrl", np.array(gaus))

        return cme_test, rnd_test


def test_cem_gt_dynamics(num_episode=50):
    log.info("### Q1.1.1: CEM (without MPC)")
    mpc_params = {'use_mpc': False, 'num_particles': 1}
    exp = ExperimentGTDynamics(env_name="Pushing2D-v1", mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode, optimizer="cem")
    log.info("CEM PushingEnv: avg_reward: {}, avg_success: {}".format(avg_reward, avg_success))
    
    log.info("### Q1.1.2: Random Policy (without MPC)")
    mpc_params = {"use_mpc": False, "num_particles": 1}
    exp = ExperimentGTDynamics(env_name="Pushing2D-v1", mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode, optimizer="random")
    log.info("CEM PushingEnv: avg_reward: {}, avg_success: {}".format(avg_reward, avg_success))
   
    log.info("### Q1.1.2: Random Policy + MPC")
    mpc_params = {"use_mpc": True, "num_particles": 1}
    exp = ExperimentGTDynamics(env_name="Pushing2D-v1", mpc_params=mpc_params)
    avg_reward, avg_success = exp.test(num_episode, optimizer="random")
    log.info("CEM PushingEnv: avg_reward: {}, avg_success: {}".format(avg_reward, avg_success))
   
    log.info("### Q1.1.3:")
    for env_name in ["Pushing2D-v1", "Pushing2DNoisyControl-v1"]:

        # CEM (without MPC)
        mpc_params = {"use_mpc": False, "num_particles": 1}
        exp = ExperimentGTDynamics(env_name=env_name, mpc_params=mpc_params)
        avg_reward, avg_success = exp.test(num_episode, optimizer="cem")
        log.info("CEM {}: avg_reward: {}, avg_success: {}".format(env_name[:-3], avg_reward, avg_success))

        # CEM + MPC 
        mpc_params = {"use_mpc": True, "num_particles": 1}
        exp = ExperimentGTDynamics(env_name=env_name, mpc_params=mpc_params)
        avg_reward, avg_success = exp.test(num_episode, optimizer="cem")
        log.info("MPC {}: avg_reward: {}, avg_success: {}".format(env_name[:-3], avg_reward, avg_success))


def train_single_dynamics(num_test_episode=50):
    log.info("### Q1.2.1: Train a single dynamics model f(s, a, phi) using a random policy")
    num_nets = 1
    num_episodes = 1000
    num_epochs = 100
    mpc_params = {"use_mpc": True, "num_particles": 6}
    exp = ExperimentModelDynamics(env_name="Pushing2D-v1", num_nets=num_nets, mpc_params=mpc_params)
    
    log.info("### Q1.2.2: Train from 1000 randomly sampled episodes with 100 epochs")
    exp.model_warmup(num_episodes=num_episodes, num_epochs=num_epochs)
    
    log.info("### Q1.2.3: Test with Random Policy for %d episodes" % num_test_episode)
    avg_reward, avg_success = exp.test(num_test_episode, optimizer="random")
    log.info("Single + Random: avg_reward: {}, avg_success: {}".format(avg_reward, avg_success))

    log.info("### Q1.2.4: Test with CEM for %d episodes" % num_test_episode)
    avg_reward, avg_success = exp.test(num_test_episode, optimizer="cem")
    log.info("Single + CEM: avg_reward: {}, avg_success: {}".format(avg_reward, avg_success))


def train_pets():
    # MBRL with PETS (Algorithm 5)
    log.info("### Q1.2.6: Train an ensemble of probabilistic dynamics model (PETS)")
    num_nets = 2
    num_epochs = 500
    evaluation_interval = 50
    num_episodes_per_epoch = 1

    mpc_params = {"use_mpc": True, "num_particles": 6}
    exp = ExperimentModelDynamics(env_name="Pushing2D-v1", num_nets=num_nets, mpc_params=mpc_params)

    log.info("### Q1.2.7: Train from 100 randomly sampled episodes + 500 MPC epochs")
    exp.model_warmup(num_episodes=100, num_epochs=10)
    cme_test, rnd_test = exp.train(
        num_train_epochs=num_epochs,
        num_episodes_per_epoch=num_episodes_per_epoch,
        evaluation_interval=evaluation_interval)

    log.info("### Q1.2.8: Plot CEM + MPC v.s. Random Policy + MPC over training")
    for i, name in enumerate(["Returns", "Success Rate"]):
        plt.figure(figsize=(6, 4))
        plt.plot(np.array(cme_test)[:, i], label="CEM + MPC")
        plt.plot(np.array(rnd_test)[:, i], label="Random + MPC")
        plt.xlabel("Training Epochs")
        plt.ylabel(name)
        plt.savefig("out/%d-cem-vs-random-%s.png" % (num_nets, name), bbox_inches="tight")


if __name__ == "__main__":
    # test_cem_gt_dynamics(50)    # Q1.1.1
    # train_single_dynamics(50)   # Q1.2.1
    # train_pets()                # Q1.2.6

    rmse = np.load("rmse_mbrl.npy")
    gaus = np.load("gaus_mbrl.npy")

    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 11

    plt.plot(list(range(len(rmse))), rmse)
    plt.title("RMSE by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.savefig("rmse_mbrl.png")

    plt.clf()

    plt.rcParams['font.size'] = 12

    plt.plot(list(range(len(gaus))), gaus)
    plt.title("Gaussian Loss by Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Gaussian Loss")
    plt.savefig("gaus_mbrl.png")