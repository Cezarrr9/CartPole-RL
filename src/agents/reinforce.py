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
from torch.distributions import Categorical

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

class PolicyNetwork(nn.Module):

    def __init__(self, n_observations: int, n_actions: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, n_actions)

    def forward(self, state):
        state = F.relu(self.layer1(state))
        state = self.layer2(state)
        return F.softmax(state, dim = 1)

class ReinforceAgent:

    def __init__(self,
                 n_observations: int, 
                 n_actions: int,
                 learning_rate: float, 
                 discount_factor: float) -> None:

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.policy_net = PolicyNetwork(n_observations, n_actions)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.learning_rate,
                                      amsgrad = True)
        
        self.probs = []
        self.rewards = []

    def select_action(self, state):
        probs = self.policy_net(state)
        c = Categorical(probs)
        action = c.sample()
        self.probs.append(c.log_prob(action))
        return action.item()

    def update(self):
        g = 0
        returns = []
        for r in self.rewards[::-1]:
            g = r + self.discount_factor * g
            returns.insert(0, g)
        
        deltas = torch.tensor(returns)

        loss = 0
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)
        
        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []

    def train(self, env: gym.wrappers, num_episodes: int):

        episode_durations = []
        for i_episode in tqdm(range(num_episodes)):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            for t in count():
                action = self.select_action(state)
                obs, reward, terminated, truncated, info = env.step(action)
                self.rewards.append(reward)

                done = terminated or truncated 

                if done:
                    episode_durations.append(t)
                    break

            self.update()

        return episode_durations

if __name__ == "__main__":
    
    # Declare the environment
    env = gym.make("CartPole-v1")

    learning_rate = 0.01
    discount_factor = 0.99

    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    # Set the hyperparameters
    num_episodes = 600

    # Declare the Reinforce agent
    agent = ReinforceAgent(learning_rate=learning_rate,
                           discount_factor=discount_factor,
                           n_actions=n_actions,
                           n_observations=n_observations)

    # Train the agent
    episode_durations = agent.train(env=env, num_episodes=num_episodes)

    # Plot the episode durations
    plot_episode_durations(algorithm="REINFORCE", episode_durations=episode_durations, num_episodes=num_episodes)

    # Close the environment
    env.close()
