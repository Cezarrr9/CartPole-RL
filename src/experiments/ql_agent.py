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
from src.utils.bucketize import bucketize

if __name__ == "__main__":

    env = gym.make('CartPole-v1')
    
    learning_rate = 0.001
    n_episodes = 100_000
    epsilon_start = 1.0
    epsilon_decay = epsilon_start / (n_episodes / 2)  # reduce the exploration over time
    epsilon_end = 0.1

    agent = QLAgent(
        action_space = env.action_space,
        learning_rate = learning_rate,
        discount_factor = 0.99,
        epsilon_start = epsilon_start,
        epsilon_decay = epsilon_decay,
        epsilon_end = epsilon_end,
    )

    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size = n_episodes)
    obs_bounds = list(zip(env.observation_space.low, env.observation_space.high))
    obs_bounds[1] = (-0.5, 0.5)
    obs_bounds[3] = (-math.radians(50), math.radians(50))
    for episode in tqdm(range(n_episodes)):
        obs, info = env.reset()
        done = False
        obs = bucketize(obs, obs_bounds)
        # play one episode
        while not done:
            
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            # update the agent
            next_obs = bucketize(next_obs, obs_bounds)
            agent.update(obs, action, reward, terminated, next_obs)

            # update if the environment is done and the current obs
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

    rolling_length = 500
    fig, axs = plt.subplots(ncols = 3, figsize = (12, 5))
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
    axs[2].set_title("Training Error")
    training_error_moving_average = (
        np.convolve(np.array(agent.training_error), np.ones(rolling_length), mode="same")
        / rolling_length
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    plt.tight_layout()
    plt.show()  
    