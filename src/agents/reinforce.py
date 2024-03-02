import os
import sys

import gymnasium as gym
import math
import random
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

# Import the plotting function
from src.utils.plot import plot_episode_durations

class PolicyNetwork():

    def __init__(self) -> None:
        pass

    def forward(self):
        pass

class ReinforceAgent:

    def __init__(self) -> None:
        pass

    def select_action(self):
        pass

    def update(self):
        pass

    def train(self):
        pass

if __name__ == "__main__":
    
    # Declare the environment
    env = gym.make("CartPole-v1")

    # Set the hyperparameters
    num_episodes = 600

    # Declare the Reinforce agent
    agent = ReinforceAgent()

    # Train the agent
    episode_durations = agent.train(env=env, num_episodes=num_episodes)

    # Plot the episode durations
    plot_episode_durations(algorithm="REINFORCE", episode_durations=episode_durations, num_episodes=num_episodes)

    # Close the environment
    env.close()

