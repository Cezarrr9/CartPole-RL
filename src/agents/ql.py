import os
import sys

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
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
                 learning_rate: float, 
                 discount_factor: float, 
                 epsilon_start: float,
                 epsilon_end: float, 
                 epsilon_decay: float) -> None:
        
        self.n_actions = n_actions

        self.q_values = defaultdict(lambda: np.zeros(n_actions))

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.training_error = []

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

        self.training_error.append(temporal_difference)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def train(self, env: gym.wrappers, num_episodes: int) -> None:
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size = num_episodes)
        obs_bounds = list(zip(env.observation_space.low, env.observation_space.high))
        obs_bounds[1] = (-0.5, 0.5)
        obs_bounds[3] = (-math.radians(50), math.radians(50))

        for i in tqdm(range(num_episodes)):
            obs, _ = env.reset()
            done = False
            state = bucketize(obs, obs_bounds)
            # play one episode
            while not done:
                
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                # update the agent
                next_state = bucketize(next_obs, obs_bounds)
                self.update(state, action, reward, terminated, next_state)

                # update if the environment is done and the current obs
                done = terminated or truncated
                state = next_state

            self.decay_epsilon()
        
        rolling_length = 500
        fig, axs = plt.subplots(ncols=2, figsize=(12, 5))
        axs[0].set_title("Episode rewards")
        # compute and assign a rolling average of the data to provide a smoother graph
        reward_moving_average = (
            np.convolve(
                np.array(env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
        )
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[1].set_title("Episode lengths")
        length_moving_average = (
            np.convolve(
                np.array(env.length_queue).flatten(), np.ones(rolling_length), mode="same"
            )
            / rolling_length
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    # Declare the environment
    env = gym.make('CartPole-v1')
    
    # Set the hyper-parameters
    num_episodes = 10000
    learning_rate = 0.3
    discount_factor = 0.99
    epsilon_start = 1.0
    epsilon_decay = epsilon_start / (num_episodes / 2)  # reduce the exploration over time
    epsilon_end = 0.1
    n_actions = env.action_space.n

    agent = QLAgent(
        n_actions=n_actions,
        learning_rate = learning_rate,
        discount_factor = discount_factor,
        epsilon_start = epsilon_start,
        epsilon_decay = epsilon_decay,
        epsilon_end = epsilon_end,
    )

    agent.train(env=env, num_episodes=num_episodes)
    