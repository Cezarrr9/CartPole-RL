import os
import sys

import numpy as np
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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

from src.utils.plot import plot_rolling_average

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer(object):

    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations: int, n_actions: int):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        return self.layer3(state)

class DQNAgent:

    def __init__(self,
                 n_actions: int,
                 n_obs: int,
                 batch_size: int, 
                 discount_factor: float, 
                 epsilon_start: float,
                 epsilon_end: float, 
                 epsilon_decay: int, 
                 learning_rate: float):
        
        self.n_actions = n_actions
        self.n_obs = n_obs
        self.batch_size = batch_size

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQN(self.n_obs, self.n_actions)
        self.target_net = DQN(self.n_obs, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), 
                                     lr = self.learning_rate, amsgrad=True)
        self.buffer = ReplayBuffer(10000)

        self.steps_done = 0

    def decay_epsilon(self) -> None:
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

    def select_action(self, state: torch.Tensor) -> torch.Tensor:

        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
            
        else:
            return torch.tensor([[random.randint(0, self.n_actions - 1)]], dtype=torch.long)

    def update(self) -> None:

        if len(self.buffer) < self.batch_size:
            return
        
        transitions = self.buffer.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env: gym.wrappers, num_episodes: int) -> list:
        episode_durations = []
        for i_episode in tqdm(range(num_episodes)):

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                
            for t in count():

                self.decay_epsilon()
                action = self.select_action(state)
                
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward])
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

                agent.buffer.push(state, action, reward, next_state)

                state = next_state

                self.update()

                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + \
                                            target_net_state_dict[key]*(1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    break
                    
        return episode_durations

if __name__ == "__main__":

    env = gym.make("CartPole-v1")

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    num_episodes = 300

    agent = DQNAgent(
        n_actions=n_actions,
        n_obs=n_observations,
        batch_size=BATCH_SIZE,
        discount_factor=GAMMA,
        epsilon_start=EPS_START,
        epsilon_decay=EPS_DECAY,
        epsilon_end=EPS_END,
        learning_rate=LR
    )

    episode_durations = agent.train(env=env, num_episodes=num_episodes)
    plot_rolling_average(algorithm="DQN", episode_durations=episode_durations, rolling_length=100)