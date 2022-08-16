import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, num_layers, dropout, hidden_dim):
        super(Actor, self).__init__()

        self.max_action = max_action

        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout)
        self.linear3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, action_dim)

        self.apply(weights_init_)

    def forward(self, state, last_action, hidden_in):
        state = state.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)
        activation = F.relu
        # branch 1
        fc_branch = activation(self.linear1(state))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = activation(self.linear2(lstm_branch))  # lstm_branch: sequential data
        # hidden only for initialization, later on hidden states are passed automatically for sequential data
        self.lstm1.flatten_parameters()
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)
        x = activation(self.linear3(merged_branch))
        x = self.max_action * torch.tanh(self.linear4(x))
        x = x.permute(1, 0, 2)

        return x, lstm_hidden

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, num_layers, dropout, hidden_dim):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers, dropout=dropout)
        self.linear3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.linear4 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action, last_action, hidden_in):
        state = state.permute(1, 0, 2)
        action = action.permute(1, 0, 2)
        last_action = last_action.permute(1, 0, 2)
        activation = F.relu
        # branch 1
        fc_branch = torch.cat([state, action], -1)
        fc_branch = activation(self.linear1(fc_branch))
        # branch 2
        lstm_branch = torch.cat([state, last_action], -1)
        lstm_branch = activation(self.linear2(lstm_branch))  # linear layer for 3d input only applied on the last dim
        self.lstm1.flatten_parameters()
        lstm_branch, lstm_hidden = self.lstm1(lstm_branch, hidden_in)  # no activation after lstm
        # merged
        merged_branch = torch.cat([fc_branch, lstm_branch], -1)

        x = activation(self.linear3(merged_branch))
        x = self.linear4(x)
        x = x.permute(1, 0, 2)  # back to same axes as input
        return x, lstm_hidden  # lstm_hidden is actually tuple: (hidden, cell)

class RTD3_OUT_XIN(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            num_layers,
            device,
            discount=0.99,
            tau=1e-2,
            policy_noise=0.2,
            actor_lr=3e-4,
            critic_lr=3e-4,
            weight_decay=0,
            dropout=0,
            hidden_dim=100,
            policy_freq=10,
    ):
        self.device = device

        self.actor = Actor(state_dim, action_dim, max_action, num_layers, dropout, hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=weight_decay)

        self.critic1 = Critic(state_dim, action_dim, num_layers, dropout, hidden_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr, weight_decay=weight_decay)

        self.critic2 = Critic(state_dim, action_dim, num_layers, dropout, hidden_dim).to(self.device)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr, weight_decay=weight_decay)

        print('Actor Network (1): ', self.actor)
        print('Critic Network (1,2): ', self.critic1)

        para = sum([np.prod(list(p.size())) for p in self.actor.parameters()])
        type_size = 4
        print('Model {} : params: {:4f}M'.format(self.actor._get_name(), para * type_size / 1000 / 1000))

        para = sum([np.prod(list(p.size())) for p in self.critic1.parameters()])
        type_size = 4
        print('Model {} : params: {:4f}M'.format(self.critic1._get_name(), para * type_size / 1000 / 1000))

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state, last_action, hidden_in):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        last_action = torch.FloatTensor(last_action).unsqueeze(0).unsqueeze(0).to(self.device)
        action, hidden_out = self.actor(state, last_action, hidden_in)
        return action.detach().cpu().data.numpy().flatten(), hidden_out

    def train(self, replay_buffer, batch_size=100):
        hidden_in, hidden_out, state, action, last_action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        last_action = torch.FloatTensor(last_action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise * self.max_action).to(self.device)

            next_action, _ = self.actor_target(next_state, action, hidden_out)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            target_Q1, _ = self.critic1_target(next_state, next_action, action, hidden_out)
            target_Q2, _ = self.critic2_target(next_state, next_action, action, hidden_out)

            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.discount * target_Q

        current_Q1, _ = self.critic1(state, action, last_action, hidden_in)
        critic1_loss = F.mse_loss(current_Q1, target_Q.detach())
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        current_Q2, _ = self.critic2(state, action, last_action, hidden_in)
        critic2_loss = F.mse_loss(current_Q2, target_Q.detach())
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            new_action, _ = self.actor(state, last_action, hidden_in)
            q_val, _ = self.critic1(state, new_action, last_action, hidden_in)
            actor_loss = -q_val.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.total_it += 1

    def save(self, filename):
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
        torch.save(self.critic2.state_dict(), filename + "_critic2")
        torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic1.load_state_dict(torch.load(filename + "_critic1", map_location='cuda:0'))
        self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer", map_location='cuda:0'))
        self.critic2.load_state_dict(torch.load(filename + "_critic2", map_location='cuda:0'))
        self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer", map_location='cuda:0'))
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location='cuda:0'))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer", map_location='cuda:0'))