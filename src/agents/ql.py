import os
import sys

import math
import random
import numpy as np
from collections import defaultdict
from itertools import count
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

from src.utils.plot import plot_reward

def bucketize(obs: np.ndarray, obs_bounds: list) -> tuple:
    n_buckets = (1, 1, 6, 3)
    bucket_indices = []
    for i in range(len(obs)):
        
        if obs[i] <= obs_bounds[i][0]:
            bucket_index = 0

        elif obs[i] >= obs_bounds[i][1]:
            bucket_index = n_buckets[i] - 1

        else:
            bound_width = obs_bounds[i][1] - obs_bounds[i][0]
            offset = (n_buckets[i] - 1) * obs_bounds[i][0] / bound_width
            scaling = (n_buckets[i] - 1) / bound_width
            bucket_index = int(round(scaling * obs[i] - offset))
 
        bucket_indices.append(bucket_index)
    
    return tuple(bucket_indices)

class QLAgent:

    def __init__(self,
                 n_actions: int,
                 min_learning_rate: float,
                 discount_factor: float,
                 min_epsilon: float) -> None:
        
        self.n_actions = n_actions

        self.q_values = defaultdict(lambda: np.zeros(n_actions))

        self.learning_rate = float
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor
        
        self.epsilon = float
        self.min_epsilon = min_epsilon

    def select_action(self, obs: tuple[float, float, float, float]) -> int:

        if np.random.random() > self.epsilon:
            return int(np.argmax(self.q_values[obs]))
        
        else:
            return random.randint(0, self.n_actions - 1)
        
    def update(self,
               state: tuple[float, float, float, float],
               action: int,
               reward: int,
               terminated: bool,
               next_state: tuple[float, float, float, float]) -> None:

        future_q_value = (not terminated) * np.max(self.q_values[next_state])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[state][action]

        self.q_values[state][action] += self.learning_rate * temporal_difference

    def decay_epsilon(self, step: int) -> None:
        self.epsilon =  max(self.min_epsilon, min(1.0, 1.0 - math.log10((step + 1) / 25)))
    
    def decay_learning_rate(self, step: int) -> None:
        self.learning_rate =  max(self.min_learning_rate, min(1.0, 1.0 - math.log10((step + 1) / 25)))

    def train(self, env: gym.wrappers, num_episodes: int) -> list:
        episode_durations = []
        obs_bounds = list(zip(env.observation_space.low, env.observation_space.high))
        obs_bounds[1] = (-0.5, 0.5)
        obs_bounds[3] = (-math.radians(50), math.radians(50))

        for i_episode in tqdm(range(num_episodes)):
            obs, _ = env.reset()
            state = bucketize(obs, obs_bounds)

            self.decay_epsilon(i_episode)
            self.decay_learning_rate(i_episode)

            # play one episode
            for t in count():
                
                # Select action
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # update the agent
                next_state = bucketize(next_obs, obs_bounds)
                self.update(state, action, reward, terminated, next_state)

                state = next_state

                if done:
                    episode_durations.append(t + 1)
                    break

        return episode_durations

if __name__ == "__main__":

    # Declare the environment
    env = gym.make('CartPole-v1')
    
    # Set the hyper-parameters
    num_episodes = 300
    min_learning_rate = 0.01
    discount_factor = 0.99
    min_epsilon = 0.01
    n_actions = env.action_space.n

    agent = QLAgent(
        n_actions=n_actions,
        min_learning_rate = min_learning_rate,
        discount_factor = discount_factor,
        min_epsilon = min_epsilon
    )

    episode_durations = agent.train(env=env, num_episodes=num_episodes)
    plot_reward(algorithm="QL", episode_durations=episode_durations, num_episodes=num_episodes)
    env.close()