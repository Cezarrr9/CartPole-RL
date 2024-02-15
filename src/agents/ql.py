import os
import sys

import numpy as np
from collections import defaultdict
import gymnasium as gym
import math
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

from src.utils.bucketize import bucketize

class QLAgent:

    def __init__(self,
                 action_space,
                 learning_rate: float, 
                 discount_factor: float, 
                 epsilon_start: float,
                 epsilon_end: float, 
                 epsilon_decay: float) -> None:
        
        self.action_space = action_space

        self.q_values = defaultdict(lambda: np.zeros(action_space.n))

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.training_error = []

    def select_action(self, obs: tuple[float, float, float, float]) -> int:

        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(self,
               obs: tuple[float, float, float, float],
               action: int,
               reward: int,
               terminated: bool,
               next_obs: tuple[float, float, float]) -> None:

        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]

        self.q_values[obs][action] += self.learning_rate * temporal_difference

        self.training_error.append(temporal_difference)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def train(self, env, num_episodes: int):
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size = num_episodes)
        obs_bounds = list(zip(env.observation_space.low, env.observation_space.high))
        obs_bounds[1] = (-0.5, 0.5)
        obs_bounds[3] = (-math.radians(50), math.radians(50))

        for _ in tqdm(range(num_episodes)):
            obs, info = env.reset()
            done = False
            obs = bucketize(obs, obs_bounds)
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

            self.decay_epsilon()

        