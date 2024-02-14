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
                 learning_rate: float, 
                 discount_factor: float, 
                 epsilon_start: float,
                 epsilon_end: float,
                 epsilon_decay: float):

        self.action_space = action_space

        self.q_values = defaultdict(lambda: np.zeros(action_space.n))

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

    def select_action(self, obs) -> int:
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        else:
            return int(np.argmax(self.q_values[obs]))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def learn(self, env, num_episodes, num_timesteps):

        for i in tqdm(range(num_episodes)):
            obs, info = env.reset()

            for t in range(num_timesteps):
                pass