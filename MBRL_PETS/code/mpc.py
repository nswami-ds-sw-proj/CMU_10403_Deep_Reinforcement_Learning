import os
import tensorflow as tf
import numpy as np
from cem import CEMOptimizer
from randopt import RandomOptimizer
import gym
import copy
import logging
import random
log = logging.getLogger('root')


class MPC:
    def __init__(self, env, plan_horizon, model, popsize, num_elites, max_iters,
                 num_particles=6,
                 use_gt_dynamics=True,
                 use_mpc=True,
                 use_random_optimizer=False):
        """ 
        Model Predictive Control (MPC) Class
            :param env:
            :param plan_horizon:
            :param model: The learned dynamics model to use, which can be None if use_gt_dynamics is True
            :param popsize: Population size
            :param num_elites: CEM parameter
            :param max_iters: CEM parameter
            :param num_particles: Number of trajectories for TS1
            :param use_gt_dynamics: Whether to use the ground truth dynamics from the environment
            :param use_mpc: Whether to use only the first action of a planned trajectory
            :param use_random_optimizer: Whether to use CEM or take random actions
            :param num_particles: Number of particles 
        """
        self.env = env
        self.plan_horizon = plan_horizon
        self.use_gt_dynamics = use_gt_dynamics
        self.use_mpc = use_mpc
        self.use_random_optimizer = use_random_optimizer
        self.num_particles = num_particles
        self.num_nets = None if model is None else model.num_nets

        self.state_dim, self.action_dim = 8, env.action_space.shape[0]
        self.ac_ub, self.ac_lb = env.action_space.high, env.action_space.low

        # Set up optimizer
        self.model = model

        if use_gt_dynamics:
            self.predict_next_state = self.predict_next_state_gt
            assert num_particles == 1
        else:
            self.predict_next_state = self.predict_next_state_model

        # Initialize your planner with the relevant arguments.
        # Use different optimizers for cem and random actions respectively
        optimizer = RandomOptimizer if self.use_random_optimizer else CEMOptimizer
        self.optimizer = optimizer(
            self.get_action_cost,
            out_dim=self.plan_horizon*self.action_dim,
            popsize=popsize,
            num_elites=num_elites,
            max_iters=max_iters,
            lower_bound=np.tile(self.ac_lb, [self.plan_horizon]),
            upper_bound=np.tile(self.ac_ub, [self.plan_horizon])
        )

        # Controller initialization: action planner
        self.curr_obs = np.zeros(self.state_dim)
        self.acs_buff = np.array([]).reshape(0, self.action_dim)  # planned action buffer
        self.prev_sol = np.tile(np.zeros(self.action_dim), 
            self.plan_horizon)  # size: (action_dim * plan_horizon,)
        self.init_var = np.tile(np.ones(self.action_dim) * (0.5 ** 2),   # std = 0.5
            self.plan_horizon)  # size: (action_dim * plan_horizon,)

        # Controller initialization: dynamics model: predict transition s' = s + f(s, a; phi)
        self.train_in = np.array([]).reshape([0, self.state_dim + self.action_dim])
        self.train_targs = np.array([]).reshape([0, self.state_dim])
        self.has_been_trained = False

    def obs_cost_fn(self, state):
        """ Cost function of the current state """
        # Weights for different terms
        W_PUSHER = 1
        W_GOAL = 2
        W_DIFF = 5

        pusher_x, pusher_y = state[0], state[1]
        box_x, box_y = state[2], state[3]
        goal_x, goal_y = self.goal[0], self.goal[1]

        pusher_box = np.array([box_x - pusher_x, box_y - pusher_y])
        box_goal = np.array([goal_x - box_x, goal_y - box_y])
        d_box = np.sqrt(np.dot(pusher_box, pusher_box))
        d_goal = np.sqrt(np.dot(box_goal, box_goal))
        diff_coord = np.abs(box_x / box_y - goal_x / goal_y)
        # the -0.4 is to adjust for the radius of the box and pusher
        return W_PUSHER * np.max(d_box - 0.4, 0) + W_GOAL * d_goal + W_DIFF * diff_coord

    def get_action_cost(self, ac_seqs):
        """ 
        Evaluate the policy (for each member of the CEM population):
            calculate the cost of a state and action sequence pair as the sum of 
            cost(s_t, a_t) w.r.t. to the policy rollout for each particle, 
            and then aggregate over all particles.

        Arguments:
            sac_seqs: shape = (popsize, plan_horizon * action_dim)
        """
        popsize = np.shape(ac_seqs)[0]
        # One cost per member of population per particle
        total_costs = np.zeros([popsize, self.num_particles])
        cur_obs = np.tile(self.curr_obs[None], [popsize * self.num_particles, 1])

        # ac_seqs.shape: (popsize, T x |A|),
        # reshape to: (popsize, T, |A|)
        ac_seqs = np.reshape(ac_seqs, [-1, self.plan_horizon, self.action_dim])
        # transpose to: (T, popsize, |A|), then ac_seqs[t] gives you action_t of all popsize
        ac_seqs = np.transpose(ac_seqs, [1, 0, 2]) 

        for t in range(self.plan_horizon):
            # action of current time step, of all popsize
            cur_acs = ac_seqs[t]
            # action_costs = 0.1 * np.tile(np.sum(np.square(cur_acs), axis=1)[:, None], [1, self.num_particles])
            next_obs = self.predict_next_state(cur_obs, cur_acs)
            delta_costs = np.apply_along_axis(self.obs_cost_fn, -1, next_obs)
            delta_costs = delta_costs.reshape((popsize, -1)) # + action_costs  # think: do we need action noise ???

            total_costs += delta_costs
            cur_obs = next_obs

        return np.mean(np.where(np.isnan(total_costs), 1e6 * np.ones_like(total_costs), total_costs), axis=1)

    def predict_next_state_model(self, states, actions):
        """ 
        For Q1.2.2,
        Given a list of state action pairs, use a single learned dynamics model to predict the next state.
            :param states  : [self.popsize * self.num_particles, self.state_dim]
            :param actions : [self.popsize, self.action_dim]

        For Q1.2.6, 
        Trajectory Sampling with TS1 (Algorithm 4) using an ensemble of learned dynamics model to predict the next state.
            :param states  : [self.popsize * self.num_particles, self.state_dim]
            :param actions : [self.popsize, self.action_dim]

        Tip: Q1.2.2 is actually a special case of TS1 where # of networks is 1.
             If you implement TS1 correctly, it should work for both Q1.2.2 and Q1.2.6.
        """
        P = self.num_particles

        # a = []
        # for action in actions:
        #     for p in range(P):
        #         a.append(action)
        # a = np.array(a)

        # s_a = np.concatenate([states, a], axis=1)
        # next_states = self.model.sess.run(self.model.outputs,
        #                                 {self.model.inputs[0] : s_a})[0]

        # mu = next_states[0]
        # logvar = next_states[1]

        # outs = np.random.normal(mu, np.sqrt(np.exp(logvar)), size=mu.shape)
        
        # return outs + states


        popsize = actions.shape[0]
        S = np.random.choice(len(self.model.networks), size=len(states), replace=True)
        run_data = [[] for _ in range(self.model.num_nets)]
        next_states = []
        for i in range(len(S)):
            net = S[i]
            state = states[i]
            action = actions[int(i / P)]
            s_a = np.concatenate([state, action])

            run_data[net].append(s_a)

        run_data = [np.array(x) for x in run_data]
        feed_dict = {}
        for n in range(len(run_data)):
            feed_dict[self.model.inputs[n]] = run_data[n]

        next_states = self.model.sess.run(self.model.outputs,
                                            feed_dict)
        mu = []
        logvar = []
        indices = [0 for _ in range(self.model.num_nets)]
        for s in range(len(actions)):

            for p in range(P):
                net_ind = S[s * P + p]

                mu_particle = next_states[net_ind][0][indices[net_ind]]
                logvar_particle = next_states[net_ind][1][indices[net_ind]]
                mu.append(mu_particle)
                logvar.append(logvar_particle)
                
                indices[net_ind] += 1

        mu = np.array(mu)
        logvar = np.array(logvar)

        return states + np.random.normal(mu, np.sqrt(np.exp(logvar)), mu.shape)

    def predict_next_state_gt(self, states, actions):
        """ Given a list of state action pairs, use the ground truth dynamics to predict the next state"""
        # TODO: write your code here
        next_states = []
        for i in range(len(states)):
            state = states[i]
            action = actions[i]

            next_state = self.env.get_nxt_state(state, action)
            next_states.append(next_state)

        return np.array(next_states)

    def train(self, obs_trajs, acs_trajs, rews_trajs, epochs=5):
        """
        Take the input obs, acs, rews and append to existing transitions the train model.

        Arguments:
          obs_trajs: states
          acs_trajs: actions
          rews_trajs: rewards (NOTE: this may not be used)
          epochs: number of epochs to train for
        """
        
        log.info("Train dynamics model with CEM for %d epochs" % epochs)
        new_train_in, new_train_targs = [], []
        for obs, acs in zip(obs_trajs, acs_trajs):
            # input (state, action)
            new_train_in.append(np.concatenate([obs[:-1, 0:-2], acs], axis=-1))  
            # predict dynamics: f(s, a, phi)
            new_train_targs.append(obs[1:, 0:-2]-obs[:-1, 0:-2])
        self.train_in = np.concatenate([self.train_in] + new_train_in, axis=0)
        self.train_targs = np.concatenate([self.train_targs] + new_train_targs, axis=0)
        (rmse, gaus) = self.model.train(self.train_in, self.train_targs, epochs=epochs)
        self.has_been_trained = True

        return rmse, gaus

    def reset(self):
        # size: (action_dim * plan_horizon, )
        self.prev_sol = np.tile(np.zeros(self.action_dim), self.plan_horizon) 

    def act(self, state, t):
        """
        Choose the action given current state using planning (CEM / Random) with or without MPC.
        
        Tip: You need to fill acs_buff to return the action 

        Arguments:
          state: current state
          t: current timestep
        """

        if not self.has_been_trained and not self.use_gt_dynamics:
            # Use random policy in warmup stage (for Q1.2)
            return np.random.uniform(self.ac_lb, self.ac_ub, self.ac_lb.shape)
        
        if self.acs_buff.shape[0] > 0:
            action, self.acs_buff = self.acs_buff[0], self.acs_buff[1:]
            return action

        self.curr_obs = state[0:-2]
        self.goal = state[-2:]

        if self.use_mpc:
            # Use MPC with CEM / Random Policy for planning (generate action sequences)
            # Review the MPC part in the writeup carefully.
            # Save only the first action (at timestep t) in self.acs_buff
            # Think carefully about what to keep in self.prev_sol for the next timestep (t+1):
            # keep the future actions (\mu) planned by CEM / Random policy
            # for t+1, ..., t+T-1 and initialize the action at t+T to 0
            soln = self.optimizer.solve(self.prev_sol, self.init_var)  # size: (action_dim * plan_horizon,)
            self.prev_sol = np.concatenate([np.copy(soln)[self.action_dim:], np.zeros(self.action_dim)])
            # MPC: only use the first/next action planned
            self.acs_buff = soln[:self.action_dim].reshape(-1, self.action_dim)
        else:
            # TODO: write your code here
            # Otherwise, directly use CEM / Random Policy without MPC
            # i.e. use planned future actions up to self.plan_horizon
            self.acs_buff = self.optimizer.solve(self.prev_sol, self.init_var).reshape(-1, self.action_dim)  

        return self.act(state, t)