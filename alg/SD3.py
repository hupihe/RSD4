import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init_(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight, gain=1)
		torch.nn.init.constant_(m.bias, 0)

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action, hidden_dim):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, hidden_dim)
		self.l2 = nn.Linear(hidden_dim, hidden_dim)
		self.l3 = nn.Linear(hidden_dim, action_dim)

		self.max_action = max_action

		#self.apply(weights_init_)

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

		#self.apply(weights_init_)

	def forward(self, state, action):
		x = torch.cat([state, action], -1)  # the dim 0 is number of samples
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.l3(x)
		return x

class SD3(object):
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
		beta=0.001,
		num_noise_samples=50,
		with_importance_sampling=0,
	):
		self.device = device

		self.actor1 = Actor(state_dim, action_dim, max_action, hidden_dim).to(self.device)
		self.actor1_target = copy.deepcopy(self.actor1)
		self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=actor_lr, weight_decay=weight_decay)

		self.actor2 = Actor(state_dim, action_dim, max_action, hidden_dim).to(self.device)
		self.actor2_target = copy.deepcopy(self.actor2)
		self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=actor_lr, weight_decay=weight_decay)

		self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
		self.critic1_target = copy.deepcopy(self.critic1)
		self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr, weight_decay=weight_decay)

		self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
		self.critic2_target = copy.deepcopy(self.critic2)
		self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr, weight_decay=weight_decay)

		print('Actor Network (1,2): ', self.actor1)
		print('Critic Network (1,2): ', self.critic1)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.policy_freq = policy_freq

		self.beta = beta
		self.num_noise_samples = num_noise_samples
		self.with_importance_sampling = with_importance_sampling

		self.total_it = 0

	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)

		action1 = self.actor1(state)
		action2 = self.actor2(state)

		q1 = self.critic1(state, action1)
		q2 = self.critic2(state, action2)

		action = action1 if q1 >= q2 else action2

		return action.detach().cpu().data.numpy().flatten()

	def train(self, replay_buffer, batch_size=100):
		self.train_one_q_and_pi(replay_buffer, update_q1=True, batch_size=batch_size)
		self.train_one_q_and_pi(replay_buffer, update_q1=False, batch_size=batch_size)
		self.total_it += 1

	def softmax_operator(self, q_vals, noise_pdf=None):
		max_q_vals = torch.max(q_vals, 1, keepdim=True).values.to(self.device)
		norm_q_vals = q_vals - max_q_vals
		e_beta_normQ = torch.exp(self.beta * norm_q_vals).to(self.device)
		Q_mult_e = q_vals * e_beta_normQ

		numerators = Q_mult_e
		denominators = e_beta_normQ

		if self.with_importance_sampling:
			numerators /= noise_pdf
			denominators /= noise_pdf

		sum_numerators = torch.sum(numerators, 1).to(self.device)
		sum_denominators = torch.sum(denominators, 1).to(self.device)

		softmax_q_vals = sum_numerators / sum_denominators

		softmax_q_vals = torch.unsqueeze(softmax_q_vals, 1).to(self.device)
		return softmax_q_vals

	def calc_pdf(self, samples, mu=0):
		pdfs = 1 / (self.policy_noise * self.max_action * np.sqrt(2 * np.pi)) * torch.exp(
			- (samples - mu) ** 2 / (2 * (self.policy_noise * self.max_action) ** 2)).to(self.device)
		pdf = torch.prod(pdfs, dim=3).to(self.device)
		return pdf

	def train_one_q_and_pi(self, replay_buffer, update_q1, batch_size=100):
		state, action, reward, next_state, done = replay_buffer.sample(batch_size)

		state = torch.FloatTensor(state).to(self.device)
		action = torch.FloatTensor(action).to(self.device)
		next_state = torch.FloatTensor(next_state).to(self.device)
		reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
		done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

		if update_q1:
			next_action = self.actor1_target(next_state)
		else:
			next_action = self.actor2_target(next_state)

		noise = torch.randn((action.shape[0], self.num_noise_samples, action.shape[1]), dtype=action.dtype, layout=action.layout, device=action.device)
		noise = noise * self.policy_noise * self.max_action
		noise_pdf = self.calc_pdf(noise) if self.with_importance_sampling else None

		next_action = torch.unsqueeze(next_action, 1)
		next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

		next_state = torch.unsqueeze(next_state, 1)
		next_state = next_state.repeat((1, self.num_noise_samples, 1))

		next_Q1 = self.critic1_target(next_state, next_action)
		next_Q2 = self.critic2_target(next_state, next_action)

		next_Q = torch.min(next_Q1, next_Q2)
		next_Q = torch.squeeze(next_Q, 2)

		softmax_next_Q = self.softmax_operator(next_Q, noise_pdf)
		target_Q = reward + (1 - done) * self.discount * softmax_next_Q

		if update_q1:
			current_Q = self.critic1(state, action)
			critic1_loss = F.mse_loss(current_Q, target_Q.detach())

			self.critic1_optimizer.zero_grad()
			critic1_loss.backward()
			self.critic1_optimizer.step()

			if self.total_it % self.policy_freq == 0:
				actor1_loss = -self.critic1(state, self.actor1(state)).mean()

				self.actor1_optimizer.zero_grad()
				actor1_loss.backward()
				self.actor1_optimizer.step()

				for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor1.parameters(), self.actor1_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		else:
			current_Q = self.critic2(state, action)
			critic2_loss = F.mse_loss(current_Q, target_Q.detach())

			self.critic2_optimizer.zero_grad()
			critic2_loss.backward()
			self.critic2_optimizer.step()

			if self.total_it % self.policy_freq == 0:
				actor2_loss = -self.critic2(state, self.actor2(state)).mean()

				self.actor2_optimizer.zero_grad()
				actor2_loss.backward()
				self.actor2_optimizer.step()

				for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

				for param, target_param in zip(self.actor2.parameters(), self.actor2_target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self, filename):
		torch.save(self.critic1.state_dict(), filename + "_critic1")
		torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
		torch.save(self.actor1.state_dict(), filename + "_actor1")
		torch.save(self.actor1_optimizer.state_dict(), filename + "_actor1_optimizer")

		torch.save(self.critic2.state_dict(), filename + "_critic2")
		torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")
		torch.save(self.actor2.state_dict(), filename + "_actor2")
		torch.save(self.actor2_optimizer.state_dict(), filename + "_actor2_optimizer")

	def load(self, filename):
		self.critic1.load_state_dict(torch.load(filename + "_critic1", map_location='cuda:0'))
		self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer", map_location='cuda:0'))
		self.actor1.load_state_dict(torch.load(filename + "_actor1", map_location='cuda:0'))
		self.actor1_optimizer.load_state_dict(torch.load(filename + "_actor1_optimizer", map_location='cuda:0'))

		self.critic2.load_state_dict(torch.load(filename + "_critic2", map_location='cuda:0'))
		self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer", map_location='cuda:0'))
		self.actor2.load_state_dict(torch.load(filename + "_actor2", map_location='cuda:0'))
		self.actor2_optimizer.load_state_dict(torch.load(filename + "_actor2_optimizer", map_location='cuda:0'))