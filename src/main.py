import os 
import sys

import gymnasium as gym

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
from src.agents.dqn import DQNAgent

if __name__ == "__main__":
    
    # Declare the environment
    env = gym.make('CartPole-v1')
    
    # Q-learning
    num_episodes = 100_000
    learning_rate = 0.001
    discount_factor = 0.99
    epsilon_start = 1.0
    epsilon_decay = epsilon_start / (num_episodes / 2)  # reduce the exploration over time
    epsilon_end = 0.1

    agent = QLAgent(
        action_space = env.action_space,
        learning_rate = learning_rate,
        discount_factor = discount_factor,
        epsilon_start = epsilon_start,
        epsilon_decay = epsilon_decay,
        epsilon_end = epsilon_end,
    )

    agent.train(env = env, num_episodes = num_episodes)

    # DQN 
    num_episodes = 50
    num_timesteps = 1000
    obs, _ = env.reset()

    learning_rate = 1e-4
    discount_factor = 0.99
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = epsilon_start / (num_episodes / 2)
    batch_size = 128

    agent = DQNAgent(action_space = env.action_space,
                    n_obs = len(obs),
                    learning_rate = learning_rate,
                    batch_size = batch_size,
                    discount_factor = discount_factor,
                    epsilon_start = epsilon_start,
                    epsilon_end = epsilon_end,
                    epsilon_decay = epsilon_decay)

    agent.train(env = env, num_episodes = num_episodes, num_timesteps = num_timesteps)