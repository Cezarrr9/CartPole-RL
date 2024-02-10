from collections import namedtuple, deque
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer(object):

    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen = capacity)
    
    def push(self, *args) -> None:
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size: int) -> list:
        return random.sample(self.memory, batch_size)
    
    def __len__(self) -> int:
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_obs, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, obs):
        obs = F.relu(self.layer1(obs))
        obs = F.relu(self.layer2(obs))
        return self.layer3(obs)

class DQNAgent():

    def __init__(self,
                 action_space,
                 n_observations: int, 
                 batch_size: int,
                 discount_factor: float, 
                 eps_start: float,
                 eps_end: float, 
                 eps_decay: int,
                 tau: float, 
                 learning_rate: float):
        
        self.action_space = action_space
        self.n_actions = self.action_space.n
        self.n_observations = n_observations

        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.eps_start = eps_start,
        self.eps_end = eps_end,
        self.eps_decay = eps_decay,
        self.tau = tau
        self.learning_rate = learning_rate

        self.policy_net = DQN(self.n_observations, self.n_actions)
        self.target_net = DQN(self.n_observations, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimzer = optim.AdamW(self.policy_net.parameters(), lr = self.learning_rate, amsgrad = True)
        self.memory = ReplayBuffer(10000)

        self.steps_done = 0

    def select_action(self, obs):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(obs).max(1).indices.view(1, 1)
            
        else:
            return torch.tensor([self.action_space.sample()], dtype = torch.long)
        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

