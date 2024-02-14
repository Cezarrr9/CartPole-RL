import os
import sys
import gymnasium as gym
from itertools import count
from tqdm import tqdm 
import matplotlib.pyplot as plt
import math

import torch

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

from src.agents.dqn import DQNAgent

env = gym.make("CartPole-v1")
num_episodes = 50

obs, info = env.reset()

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size = num_episodes)

agent = DQNAgent(action_space = env.action_space,
                 n_obs = len(obs),
                 batch_size = 128,
                 discount_factor = 0.99,
                 epsilon_start = 0.9,
                 epsilon_end = 0.05,
                 epsilon_decay = 1000, 
                 learning_rate = 1e-4)

agent.train(env = env, num_episodes = num_episodes, num_timesteps = 1000)
