import gym
from gym import spaces
import numpy as np
import random
import math

class scheduler_csi(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, lamb, user_num=4, time_step=100, tau_max=6, seed1=1, seed2=1, partial=False, homogeneous=False, period=0, cntype=0, square=1, poisson=True):
        super(scheduler_csi, self).__init__()

        self.rng = random.Random(seed1)
        self.rng_np = np.random.default_rng(seed1)

        self.lamb = lamb
        self.period = period
        self.cntype = cntype
        self.square = square
        self.poisson = poisson
        self.N = user_num
        if self.poisson:
            self.A_n = [self.rng.randint(2, 5) for i in range(self.N)]
        else:
            self.A_n = self.rng_np.random(self.N)
        self.tau_max = tau_max
        self.tau_n = [self.rng.randint(1, self.tau_max) for i in range(self.N)]
        self.action_dim = np.sum(self.tau_n) + self.N
        if homogeneous:
            self.beta = [1 for i in range(self.N)]
            self.d = [1 for i in range(self.N)]
        else:
            self.beta = [self.rng.randint(1, 5) for i in range(self.N)]
            self.d = [self.rng.uniform(1, 2) for i in range(self.N)]
        self.channel_state_space = list(range(1, self.N + 1))
        self.time_step = time_step
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
        self.channel_history = np.zeros((self.tau_max + 1, self.N), dtype='int')
        self.action_history = np.zeros((self.tau_max + 1, self.action_dim)) - 1
        if self.cntype < 0:
            self.channel_state = np.zeros(self.N, dtype='int')
        else:
            self.channel_state = np.array([self.rng.randint(0, self.N - 1) for i in range(self.N)], dtype='int')
        if self.period > 0:
            if self.square > 0:
                self.atom_queue = np.transpose(np.repeat(np.array([self.A_n,[0]*self.N]), [self.square, self.period-self.square], axis=0)) * 2
            else:
                self.atom_queue = np.transpose(self.rng_np.poisson(lam=self.A_n, size=(self.period, self.N)))
            self.packet_queue = self.atom_queue.copy()
            while self.packet_queue.shape[1] < self.time_step:
                self.packet_queue = np.concatenate((self.packet_queue, self.atom_queue), axis=1)
            self.packet_queue = self.packet_queue[:,:self.time_step]
        elif self.period == 0:
            if self.poisson:
                self.packet_queue = np.transpose(self.rng_np.poisson(lam=self.A_n, size=(self.time_step, self.N)))
            else:
                self.packet_queue = (self.rng_np.random((self.N, self.time_step)) < self.A_n.reshape(self.N, 1)).astype(int)
        else:
            repeat_num = -self.period
            length = int(self.time_step/repeat_num) + 1
            self.atom_queue = np.transpose(self.rng_np.poisson(lam=self.A_n, size=(length, self.N)))
            self.packet_queue = np.repeat(self.atom_queue, repeat_num, axis=1)[:,:self.time_step]

        self.action_space = spaces.Box(-1, 1, (self.action_dim,), dtype=np.float32)
        if self.partial_obserbvable:
            self.observation_space = spaces.MultiDiscrete([1000 for _ in range(self.action_dim)])
        else:
            self.observation_space = spaces.MultiDiscrete([1000 for _ in range(self.action_dim)] + [self.N for _ in range(self.N)])

    def reset(self):
        self.Buffer = np.zeros(self.action_dim, dtype='int')
        self.success = np.zeros(self.action_dim, dtype='int')
        self.channel_history = np.zeros((self.tau_max + 1, self.N), dtype='int')
        self.action_history = np.zeros((self.tau_max + 1, self.action_dim)) - 1
        if self.cntype < 0:
            self.channel_state = np.zeros(self.N, dtype='int')
        else:
            self.channel_state = np.array([self.rng.randint(0, self.N - 1) for i in range(self.N)], dtype='int')
        if self.period > 0:
            if self.square > 0:
                self.atom_queue = np.transpose(
                    np.repeat(np.array([self.A_n, [0] * self.N]), [self.square, self.period - self.square], axis=0)) * 2
            else:
                self.atom_queue = np.transpose(self.rng_np.poisson(lam=self.A_n, size=(self.period, self.N)))
            self.packet_queue = self.atom_queue.copy()
            while self.packet_queue.shape[1] < self.time_step:
                self.packet_queue = np.concatenate((self.packet_queue, self.atom_queue), axis=1)
            self.packet_queue = self.packet_queue[:, :self.time_step]
        elif self.period == 0:
            if self.poisson:
                self.packet_queue = np.transpose(self.rng_np.poisson(lam=self.A_n, size=(self.time_step, self.N)))
            else:
                self.packet_queue = (self.rng_np.random((self.N, self.time_step)) < self.A_n.reshape(self.N, 1)).astype(int)
        else:
            repeat_num = -self.period
            length = int(self.time_step / repeat_num) + 1
            self.atom_queue = np.transpose(self.rng_np.poisson(lam=self.A_n, size=(length, self.N)))
            self.packet_queue = np.repeat(self.atom_queue, repeat_num, axis=1)[:, :self.time_step]
        self.t = 0
        self.cum_energy = 0
        self.throughput = 0

        if self.partial_obserbvable:
            return self.Buffer.copy()
        else:
            return np.concatenate((self.Buffer, self.channel_state), axis=0)

    def energy_fuc(self, i, e, n):
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
                prob = self.energy_fuc(self.channel_state[i], self.action[j], i)
                #print(i, self.action[j], self.channel_state[i], prob)
                # if self.t < 10:
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

        # update channel state
        for i in range(self.N):
            if self.cntype > 0:
                dist = self.t ** self.cntype
                self.channel_state[i] = (self.channel_state[i] + dist) % self.N
            elif self.cntype < 0:
                pd = -self.cntype
                self.channel_state[i] = 0 if self.t % pd < pd/2 else 1
            else:
                self.channel_state[i] = self.rng.choices(list(range(self.N)), weights=self.transition_matrix[self.channel_state[i], :], k=1)[0]

        # update channel history
        for i in range(self.tau_max):
            self.channel_history[self.tau_max - i, :] = self.channel_history[self.tau_max - i - 1, :]
        self.channel_history[0, :] = self.channel_state

        self.t += 1

        if self.partial_obserbvable:
            state = self.Buffer.copy()
        else:
            state = np.concatenate((self.Buffer, self.channel_state), axis=0)

        return state, reward, (self.t >= self.time_step), {'throughput': self.throughput / self.t, 'energy': self.cum_energy / self.t, 'reward': vec_result}