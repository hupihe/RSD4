import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_dim):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, action_dim)

		self.max_action = max_action

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
	def __init__(self, state_dim, action_dim, hidden_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, 1)

	def forward(self, state, action):
		x = torch.cat([state, action], 1)  # the dim 0 is number of samples
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		return x

class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		device,
		discount=0.99,
		tau=1e-2,
		policy_noise=0.2,
		actor_lr=3e-4,
		critic_lr=3e-4,
		weight_decay=0,
		hidden_dim=100,
		policy_freq=10,
	):
		self.device = device

		self.actor = Actor(state_dim, action_dim, max_action, hidden_dim).to(self.device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=weight_decay)

		self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
		self.critic1_target = copy.deepcopy(self.critic1)
		self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr, weight_decay=weight_decay)

		self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
		self.critic2_target = copy.deepcopy(self.critic2)
		self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr, weight_decay=weight_decay)

		print('Actor Network (1): ', self.actor)
		print('Critic Network (1,2): ', self.critic1)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.policy_freq = policy_freq

		self.total_it = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
		action = self.actor(state)
		return action.detach().cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=100):
		state, action, reward, next_state, done = replay_buffer.sample(batch_size)

		state = torch.FloatTensor(state).to(self.device)
		action = torch.FloatTensor(action).to(self.device)
		next_state = torch.FloatTensor(next_state).to(self.device)
		reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
		done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

		noise = (torch.randn_like(action) * self.policy_noise * self.max_action).to(self.device)
		next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

		target_Q1 = self.critic1_target(next_state, next_action)
		target_Q2 = self.critic2_target(next_state, next_action)
		target_Q = torch.min(target_Q1, target_Q2)
		target_Q = reward + (1 - done) * self.discount * target_Q

		current_Q1 = self.critic1(state, action)
		critic1_loss = F.mse_loss(current_Q1, target_Q.detach())
		self.critic1_optimizer.zero_grad()
		critic1_loss.backward()
		self.critic1_optimizer.step()

		current_Q2 = self.critic2(state, action)
		critic2_loss = F.mse_loss(current_Q2, target_Q.detach())
		self.critic2_optimizer.zero_grad()
		critic2_loss.backward()
		self.critic2_optimizer.step()

		if self.total_it % self.policy_freq == 0:
			new_action = self.actor(state)
			q_val = self.critic1(state, new_action)
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
