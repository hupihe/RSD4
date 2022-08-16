import gym
from gym import spaces
import numpy as np
import pandas as pd
import random
import math

class scheduler_data_cs(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, arrival_data_path, channel_data_path, lamb, user_num=4, time_step=100, tau_max=6, seed1=1, seed2=1, partial=False, homogeneous=False, is_evaluate = False, train_area = 0.8) :
        super(scheduler_data_cs, self).__init__()

        self.rng = random.Random(seed1)
        self.rng_np = np.random.default_rng(seed1)

        self.N = user_num
        self.channelN = 4

        self.arrival_dataset = pd.read_csv(arrival_data_path)
        self.channel_dataset = pd.read_csv(channel_data_path)
        self.arrival_dataset_sample = self.arrival_dataset.sample(n = self.N, axis=0, random_state = seed1)
        self.channel_dataset_sample = self.channel_dataset.sample(n = self.N, axis=0, random_state = seed1)

        self.arrival_dataset_length = int(self.arrival_dataset.to_numpy(dtype='int').shape[1])
        self.channel_dataset_length = int(self.channel_dataset.to_numpy(dtype='int').shape[1])
        self.time_step = time_step
        self.max_time_step = int(self.arrival_dataset_length * train_area)
        self.train_length = self.max_time_step
        self.eval_length = self.arrival_dataset_length - self.train_length
        self.is_evaluate = is_evaluate
        if self.is_evaluate :
            self.arrivla_position = self.train_length
            if self.arrival_position + self.time_step > self.arrival_dataset_length :
                self.atom_queue = self.arrival_dataset_sample.iloc[:, self.arrival_position: self.arrival_dataset_length].to_numpy(dtype='int')
            else :
                self.atom_queue = self.arrival_dataset_sample.iloc[:, self.arrival_position: self.arrival_position + self.time_step].to_numpy(dtype='int')
        else :
            self.arrival_position = 0
            if self.arrival_position + self.time_step > self.train_length :
                self.atom_queue = self.arrival_dataset_sample.iloc[:, self.arrival_position : self.train_length].to_numpy(dtype = 'int')
            else :
                self.atom_queue = self.arrival_dataset_sample.iloc[:, self.arrival_position: self.arrival_position + self.time_step].to_numpy(dtype='int')
        self.packet_queue = self.atom_queue.copy()
        if self.is_evaluate :
            self.channel_position = self.train_length
            if self.channel_position + self.time_step > self.channel_dataset_length :
                self.channel_atom_queue = self.channel_dataset_sample.iloc[:, self.channel_position: self.arrival_dataset_length].to_numpy(dtype='int')
            else :
                self.channel_atom_queue = self.channel_dataset_sample.iloc[:, self.channel_position: self.channel_position + self.time_step].to_numpy(dtype='int')
        else :
            self.channel_position = 0
            if self.channel_position + self.time_step > self.train_length :
                self.channel_atom_queue = self.channel_dataset_sample.iloc[:, self.channel_position : self.train_length].to_numpy(dtype = 'int')
            else :
                self.channel_atom_queue = self.channel_dataset_sample.iloc[:, self.channel_position: self.channel_position + self.time_step].to_numpy(dtype='int')
        self.channel_queue = self.channel_atom_queue.copy()
        while self.packet_queue.shape[1] < self.time_step :
            self.packet_queue = np.concatenate((self.packet_queue, self.atom_queue), axis = 1)
        self.packet_queue = self.packet_queue[:, : self.time_step]
        while self.channel_queue.shape[1] < self.time_step :
            self.channel_queue = np.concatenate((self.channel_queue, self.channel_atom_queue), axis = 1)
        self.channel_queue = self.channel_queue[:, : self.time_step]

        self.lamb = lamb
        self.A_n = [self.rng.randint(2, 5) for i in range(self.N)]
        self.tau_max = tau_max
        self.tau_n = [self.rng.randint(1, self.tau_max) for i in range(self.N)]
        self.action_dim = np.sum(self.tau_n) + self.N


        if homogeneous:
            self.beta = [1 for i in range(self.N)]
            self.d = [1 for i in range(self.N)]
        else:
            self.beta = [self.rng.randint(1, 5) for i in range(self.N)]
            self.d = [self.rng.uniform(1, 2) for i in range(self.N)]
        self.channel_state_space = list(range(1, self.channelN + 1))#TODO
        self.partial_obserbvable = partial

        self.t = 0
        self.cum_energy = 0
        self.throughput = 0
        self.transition_matrix = np.asarray([[self.rng.uniform(0, 1) for i in range(self.N)] for j in range(self.N)])
        for j, r in enumerate(self.transition_matrix):
            r /= sum(r)

        self.rng = random.Random(seed2)
        self.rng_np = np.random.default_rng(seed2)

        self.Buffer = np.zeros(self.action_dim, dtype='int')
        self.success = np.zeros(self.action_dim, dtype='int')
        self.action_history = np.zeros((self.tau_max + 1, self.action_dim)) - 1
        self.channel_state = np.array([self.rng.randint(0, self.channelN - 1) for i in range(self.channelN)], dtype='int')#TODO
        self.action_space = spaces.Box(-1, 1, (self.action_dim,), dtype=np.float32)
        if self.partial_obserbvable:
            self.observation_space = spaces.MultiDiscrete([2 for _ in range(self.action_dim)])
        else:
            self.observation_space = spaces.MultiDiscrete([2 for _ in range(self.action_dim)] + [self.N for _ in range(self.N)])

    def reset(self):
        self.Buffer = np.zeros(self.action_dim, dtype='int')
        self.success = np.zeros(self.action_dim, dtype='int')
        self.action_history = np.zeros((self.tau_max + 1, self.action_dim)) - 1
        self.channel_state = np.array([self.rng.randint(0, self.channelN - 1) for i in range(self.channelN)], dtype='int')#TODO

        if self.is_evaluate :
            self.arrival_position = (self.arrival_position + self.time_step)
            if self.arrival_position > self.arrival_dataset_length :
                self.arrival_position = self.train_length + (self.arrival_position - self.arrival_dataset_length) % self.eval_length

            if self.arrival_position + self.time_step > self.arrival_dataset_length :
                self.atom_queue = self.arrival_dataset_sample.iloc[:, self.train_length : self.arrival_dataset_length].to_numpy(dtype='int')
                self.packet_queue = self.arrival_dataset_sample.iloc[:, self.arrival_position : self.arrival_dataset_length].to_numpy(dtype = 'int')
            else :
                self.atom_queue = self.arrival_dataset_sample.iloc[:, self.arrival_position: self.arrival_position + self.time_step].to_numpy(dtype='int')
                self.packet_queue = self.atom_queue.copy()
        else :
            self.arrival_position = (self.arrival_position + self.time_step) % self.max_time_step
            if self.arrival_position + self.time_step > self.arrival_dataset_length :
                self.atom_queue = self.arrival_dataset_sample.iloc[:, 0 : self.train_length].to_numpy(dtype = 'int')
                self.packet_queue = self.arrival_dataset_sample.iloc[:, self.arrival_position : self.train_length].to_numpy(dtype = 'int')
            else :
                self.atom_queue = self.arrival_dataset_sample.iloc[:, self.arrival_position: self.arrival_position + self.time_step].to_numpy(dtype='int')
                self.packet_queue = self.atom_queue.copy()

        while self.packet_queue.shape[1] < self.time_step :
            self.packet_queue = np.concatenate((self.packet_queue, self.atom_queue), axis = 1)
        self.packet_queue = self.packet_queue[:, : self.time_step]

        if self.is_evaluate :
            self.channel_position = (self.channel_position + self.time_step)
            if self.channel_position > self.channel_dataset_length :
                self.channel_position = self.train_length + (self.channel_position - self.channel_dataset_length) % self.eval_length

            if self.channel_position + self.time_step > self.channel_dataset_length :
                self.channel_atom_queue = self.channel_dataset_sample.iloc[:, self.train_length : self.channel_dataset_length].to_numpy(dtype='int')
                self.channel_queue = self.channel_dataset_sample.iloc[:, self.channel_position : self.channel_dataset_length].to_numpy(dtype = 'int')
            else :
                self.channel_atom_queue = self.channel_dataset_sample.iloc[:, self.channel_position: self.channel_position + self.time_step].to_numpy(dtype='int')
                self.channel_queue = self.channel_atom_queue.copy()
        else :
            self.channel_position = (self.channel_position + self.time_step) % self.max_time_step
            if self.channel_position + self.time_step > self.channel_dataset_length :
                self.channel_atom_queue = self.channel_dataset_sample.iloc[:, 0 : self.train_length].to_numpy(dtype = 'int')
                self.channel_queue = self.channel_dataset_sample.iloc[:, self.channel_position : self.train_length].to_numpy(dtype = 'int')
            else :
                self.channel_atom_queue = self.channel_dataset_sample.iloc[:, self.channel_position: self.channel_position + self.time_step].to_numpy(dtype='int')
                self.channel_queue = self.channel_atom_queue.copy()

        while self.channel_queue.shape[1] < self.time_step :
            self.channel_queue = np.concatenate((self.channel_queue, self.channel_atom_queue), axis = 1)
        self.channel_queue = self.channel_queue[:, : self.time_step]

        self.t = 0
        self.cum_energy = 0
        self.throughput = 0

        if self.partial_obserbvable:
            return self.Buffer
        else:
            return np.concatenate((self.Buffer, self.channel_state), axis=0)

    # def energy_fuc(self, i, e, n):
    #     tmp = math.exp(-2 * e / self.d[n] ** 3 / self.channel_state_space[i])
    #     return 2 / (1 + tmp) - 1

    def energy_fuc_cs(self, i, e, n):
        if self.channel_state_space[i] < self.N:
            return 0
        tmp = math.exp(-2 * e / self.d[n] ** 3 / self.channel_state_space[i])
        return 2 / (1 + tmp) - 1

    def step(self, action):
        reward = 0
        energy = 0
        vec_reward = [0 for _ in range(self.N)]
        vec_energy = [0 for _ in range(self.N)]
        for i in range(self.tau_max):
            self.action_history[self.tau_max - i, :] = self.action_history[self.tau_max - i - 1, :]
        self.action_history[0, :] = action

        self.action = action + 1
        self.action /= 2
        self.success = np.zeros(self.action_dim, dtype='int')
        # pack trainsimission
        for i in range(self.N):
            for j in range(np.sum(self.tau_n[:i], dtype='int') + i, np.sum(self.tau_n[:i + 1], dtype='int') + i + 1):
                #print('t:', self.t, 'pos:', j, 'action:', action[j], 'Buffer:', self.Buffer[j])
                energy += self.action[j] * self.Buffer[j]
                vec_energy[i] += self.action[j] * self.Buffer[j]
                prob = self.energy_fuc_cs(self.channel_state[i], self.action[j], i)
                # if self.t < 10:x
                #    print(self.t, self.action[j], prob)
                for k in range(self.Buffer[j]):
                    if prob > self.rng.uniform(0, 1):
                        self.success[j] += 1
                        reward += self.beta[i]
                        vec_reward[i] += self.beta[i]
                        self.Buffer[j] -= 1
                if j > np.sum(self.tau_n[:i]) + i:
                    self.Buffer[j - 1] = self.Buffer[j]

        self.cum_energy += energy
        self.throughput += reward
        reward -= self.lamb * energy
        vec_result = []
        for i in range(self.N):
            vec_result.append(vec_reward[i] - self.lamb * vec_energy[i])
        # packet arrival
        for i in range(self.N):
            self.Buffer[np.sum(self.tau_n[:i + 1]) + i] = self.packet_queue[i, self.t]
            #self.Buffer[np.sum(self.tau_n[:i + 1]) + i] = 1

        self.Buffer_copy = self.Buffer

        # update channel state
        for i in range(self.N):
            self.channel_state[i] = self.channel_queue[i, self.t]
        # for i in range(self.N):
        #     self.channel_state[i] = self.rng.choices(list(range(self.N)), weights=self.transition_matrix[self.channel_state[i], :], k=1)[0]

        # update channel history
        # for i in range(self.tau_max):
        #     self.channel_history[self.tau_max - i, :] = self.channel_history[self.tau_max - i - 1, :]
        # self.channel_history[0, :] = self.channel_state

        self.t += 1


        if self.partial_obserbvable:
            state = self.Buffer
        else:
            state = np.concatenate((self.Buffer, self.channel_state), axis=0)

        return state, reward, (self.t >= self.time_step), {'throughput': self.throughput / self.t, 'energy': self.cum_energy / self.t, 'reward': vec_result}