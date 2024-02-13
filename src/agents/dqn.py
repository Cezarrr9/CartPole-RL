import os
import sys

import random
import numpy as np
from collections import deque, namedtuple, defaultdict

import torch
import torch.nn as nn

# Define the environment variable name
env_var_name = 'CARTPOLE_RL_PATH'

# Get the path from the environment variable
module_path = os.getenv(env_var_name)

# Check if the environment variable was set
if module_path:
    sys.path.append(module_path)
else:
    print(f"Please set the {env_var_name} environment variable to your 'CartPole-RL' directory path.")
    sys.exit(1)

from src.utils.bucketize import bucketize

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

    def __init__(self):
        pass

    def forward(self, obs):
        pass

class DQNAgent:

    def __init__(self, 
                 action_space,
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

        self.memory = ReplayMemory(1000)

    def select_action(self, obs):
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        else:
            return int(np.argmax(self.q_values[obs]))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def learn(self, env, num_episodes):
        obs_bounds = []
        for i in range(num_episodes):
            # play one episode
            while not done:
                
                action = self.select_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)

                # update the agent
                next_obs = bucketize(next_obs, obs_bounds)
                self.update(obs, action, reward, terminated, next_obs)

                # update if the environment is done and the current obs
                done = terminated or truncated
                obs = next_obs
        pass