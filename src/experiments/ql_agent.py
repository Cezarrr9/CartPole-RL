import os
import sys

import matplotlib.pyplot as plt 
import numpy as np
import math

import gymnasium as gym
from tqdm import tqdm

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

from src.agents.ql import QLAgent

if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    
    learning_rate = 0.001
    n_episodes = 100_000
    epsilon_start = 1.0
    epsilon_decay = epsilon_start / (n_episodes / 2)  # reduce the exploration over time
    epsilon_end = 0.1

    agent = QLAgent(
        action_space = env.action_space,
        learning_rate = 0.001,
        discount_factor = 0.99,
        epsilon_start = 1.0,
        epsilon_decay = epsilon_decay,
        epsilon_end = 0.1,
    )