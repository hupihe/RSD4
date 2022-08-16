import gym
from gym import spaces
import numpy as np
import random
import math

class scheduler(gym.Env):
    metadata = {'render.modes': ['console']}

    def __init__(self, user_num=4, time_step=100, seed1=1, seed2=1, partial=False, render=False):
        super(scheduler, self).__init__()

        random.seed(seed1)
        np.random.seed(seed1)
        self.N = user_num
        self.A_n = np.random.rand(self.N)
        self.tau_n = [random.randint(1, 6) for i in range(self.N)]
        self.action_dim = np.sum(self.tau_n) + self.N
        self.beta = [random.randint(1, 5) for i in range(self.N)]
        self.d = [random.uniform(1, 2) for i in range(self.N)]
        self.channel_state_space = list(range(1, self.N + 1))
        self.time_step = time_step
        self.partial_obserbvable = partial

        self.t = 0
        self.cum_energy = 0
        self.throughput = 0
        self.action = np.zeros(self.action_dim)

        # self.transition_matrix = np.asarray([[.4, .3, .2, .1], [.25, .3, .25, .2], [.2, .25, .3, .25], [.1, .2, .3, .4]])
        self.transition_matrix = np.asarray([[random.uniform(0, 1) for i in range(self.N)] for j in range(self.N)])
        for j, r in enumerate(self.transition_matrix):
            r /= sum(r)

        if render:
            print('#'*20+'Initialization Info'+'#'*20)
            print('A_n:', self.A_n)
            print('tau_n:', self.tau_n)
            print('beta:', self.beta)
            print('d:', self.d)
            print('B:', self.B)
            print('transition_matrix:', self.transition_matrix)
            print('#' * 59)

        random.seed(seed2)
        np.random.seed(seed2)

        self.Buffer = np.zeros(self.action_dim, dtype='int')
        self.Buffer_mirror = self.Buffer.copy()
        self.channel_state = np.array([random.randint(0, self.N - 1) for i in range(self.N)], dtype='int')
        self.packet_queue = (np.random.rand(self.N, self.time_step) < self.A_n.reshape(self.N, 1)).astype(int)
        self.action_space = spaces.Box(-1, 1, (self.action_dim,), dtype=np.float32)
        # self.observation_space = spaces.Box(0, self.N, (self.action_dim + self.N + self.N*self.window_length,), dtype='int')
        if self.partial_obserbvable:
            self.observation_space = spaces.MultiDiscrete([2 for _ in range(self.action_dim)])
        else:
            self.observation_space = spaces.MultiDiscrete([2 for _ in range(self.action_dim)] + [self.N for _ in range(self.N)])

    def reset(self):
        self.channel_state = np.array([random.randint(0, self.N - 1) for i in range(self.N)], dtype='int')
        self.Buffer = np.zeros(self.action_dim, dtype='int')
        self.Buffer_mirror = self.Buffer.copy()
        self.packet_queue = (np.random.rand(self.N, self.time_step) < self.A_n.reshape(self.N, 1)).astype(int)
        self.t = 0
        self.cum_energy = 0
        self.throughput = 0
        self.action = np.zeros(self.action_dim)

        if self.partial_obserbvable:
            return self.Buffer
        else:
            return np.concatenate((self.Buffer, self.channel_state), axis=0)

    def energy_fuc(self, i, e, n):
        tmp = math.exp(-2 * e / self.d[n] ** 3 / self.channel_state_space[i])
        return 2 / (1 + tmp) - 1

    def step(self, action):
        reward = 0
        energy = 0

        action += 1
        action /= 2

        self.action = action
        # pack trainsimission
        for i in range(self.N):
            user_energy = 0
            for j in range(np.sum(self.tau_n[:i], dtype='int') + i, np.sum(self.tau_n[:i + 1], dtype='int') + i + 1):
                user_energy += action[j] * self.Buffer[j]
                prob = self.energy_fuc(self.channel_state[i], action[j], i)
                for k in range(self.Buffer[j]):
                    if prob > random.uniform(0, 1):
                        reward += self.beta[i]
                        self.Buffer[j] -= 1
                if j > np.sum(self.tau_n[:i]) + i:
                    self.Buffer[j - 1] = self.Buffer[j]
                    if j == np.sum(self.tau_n[:i + 1]) + i:
                        self.Buffer[j] = 0
            energy += user_energy
        self.Buffer_mirror = self.Buffer.copy()

        self.cum_energy += energy
        self.throughput += reward

        # packet arrival
        for i in range(self.N):
            self.Buffer[np.sum(self.tau_n[:i + 1]) + i] = self.packet_queue[i, self.t]

        # update channel state
        for i in range(self.N):
            self.channel_state[i] = random.choices(list(range(self.N)), weights=self.transition_matrix[i, :], k=1)[0]

        self.t += 1
        if self.partial_obserbvable:
            state = self.Buffer
        else:
            state = np.concatenate((self.Buffer, self.channel_state), axis=0)

        return state, reward, (self.t >= self.time_step), {'throughput': self.throughput / self.t, 'energy': self.cum_energy / self.t}

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        # agent is represented as a cross, rest as a dot

        print('action  |' + '-' * 5 + 'state after action' + '-' * 20)
        for i in range(self.N):
            print('%3.2f' % self.action[i], end='\t|\t')
            for j in range(np.sum(self.tau_n[:i], dtype='int') + i, np.sum(self.tau_n[:i + 1], dtype='int') + i + 1): \
                    print(self.Buffer_mirror[j], end='\t')
            print()

        print('channel |' + '-' * 5 + 'state after arrival' + '-' * 20)
        for i in range(self.N):
            print(self.channel_state[i], end='\t\t|\t')
            for j in range(np.sum(self.tau_n[:i], dtype='int') + i, np.sum(self.tau_n[:i + 1], dtype='int') + i + 1): \
                    print(self.Buffer[j], end='\t')
            print()
        print('Cumulative Engergy:', self.cum_energy / self.t, 'Weighted Throughput:', self.throughput / self.t)

    def close(self):
        pass
