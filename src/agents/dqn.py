import os
import sys

import random
import numpy as np
from collections import deque, namedtuple, defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_obs, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_obs, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, obs):
        obs = F.relu(self.layer1(obs))
        obs = F.relu(self.layer2(obs))
        return self.layer3(obs)

class DQNAgent:

    def __init__(self, 
                 action_space,
                 n_obs: int,
                 batch_size: int,
                 learning_rate: float, 
                 discount_factor: float, 
                 epsilon_start: float,
                 epsilon_end: float,
                 epsilon_decay: float):

        self.action_space = action_space

        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQN(n_obs, action_space.n)
        self.target_net = DQN(n_obs, action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = learning_rate, amsgrad = True)

        self.memory = ReplayMemory(1000)

    def select_action(self, state) -> int:
        
        if np.random.random() < self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        
        else:
            return torch.tensor([[self.action_space.sample()]], dtype = torch.long)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def update_policy_net(self):

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

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        next_q_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_q_values = reward_batch + self.discount_factor * next_q_values

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def train(self, env, num_episodes, num_timesteps):

        for i in tqdm(range(num_episodes)):
            state, info = env.reset()
            state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)

            for t in range(num_timesteps):
                action = self.select_action(state)
                self.decay_epsilon()
                obs, reward, terminated, truncated, info = env.step(action.item())
                reward = torch.tensor([reward])
                done = terminated or truncated 

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(obs, dtype = torch.float32).unsqueeze(0)

                self.memory.push(state, action, reward, next_state)

                self.update_policy_net()

                if done:
                    break
            
            self.target_net.load_state_dict(self.policy_net.state_dict())

