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

# Import the plotting function
from src.utils.plot import plot_reward

def bucketize(obs: np.ndarray, obs_bounds: list) -> tuple:
    """ Discretizes the continuous state values into a fixed
    number of buckets.

    Parameters:
    - obs (np.ndarray): the continuous observation given by the environment
    - obs_bounds (list[tuple]): the accepted boundaries of the observation space for each type
    of information (check https://gymnasium.farama.org/environments/classic_control/cart_pole/#observation-space)

    Returns:
    - bucket_indices (tuple[int]): the indices of the buckets for each type of information
    
    """
    # Define the number of buckets for each type of observation.
    n_buckets = (1, 1, 6, 3)
    
    # Initialize a list to hold the bucket indices for each observation.
    bucket_indices = []

    for i in range(len(obs)):
        
        # If the value is smaller than or equal to the inferior boundary,
        # then the it is associated with the first bucket
        if obs[i] <= obs_bounds[i][0]:
            bucket_index = 0

        # If the value is bigger than or equal to the superior boundary, 
        # then it is associated with the last bucket
        elif obs[i] >= obs_bounds[i][1]:
            bucket_index = n_buckets[i] - 1

        else:
            # Compute the range of the observation's boundaries
            bound_width = obs_bounds[i][1] - obs_bounds[i][0]

            # Compute the offset based on the inferior
            # boundary and the number of buckets 
            offset = (n_buckets[i] - 1) * obs_bounds[i][0] / bound_width

            # Compute the scaling factor to adjust the observation
            # within the bucket range
            scaling = (n_buckets[i] - 1) / bound_width

            # Apply the scaling and offset to determine
            # the bucket index, rounding to the nearest integer
            bucket_index = int(round(scaling * obs[i] - offset))

        # Store the bucket index 
        bucket_indices.append(bucket_index)
    
    bucket_indices = tuple(bucket_indices)
    return bucket_indices

class QLAgent:
    """Q-learning Agent"""

    def __init__(self,
                 n_actions: int,
                 min_learning_rate: float,
                 discount_factor: float,
                 min_epsilon: float) -> None:
        
        """Initializes an instance of the class.
        
        Parameters:
        - n_action (int): the number of actions available for the agent
        - min_learning_rate (float): the minimum value that the learning rate can take 
        - discount_factor (float): the discount factor used in the q-learning algorithm 
        - min_epsilon (float): the minimum value that the epsilon threshold can take 
        """
        
        self.n_actions = n_actions

        self.q_values = defaultdict(lambda: np.zeros(n_actions))

        self.learning_rate = float
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor
        
        self.epsilon = float
        self.min_epsilon = min_epsilon

    def select_action(self, state: tuple[float, float, float, float]) -> int:
        """Selects an action using the epsilon greedy policy
        
        Parameters:
        - state (tuple[float]): the discretized version of the state

        Returns:
        - int: the index of an action
        """

        # If the random selected number is bigger than epsilon, then select
        # the greedy action. Otherwise, select a random action.
        if np.random.random() > self.epsilon:
            return int(np.argmax(self.q_values[state]))
        
        else:
            return random.randint(0, self.n_actions - 1)
        
    def update(self,
               state: tuple[float, float, float, float],
               action: int,
               reward: int,
               terminated: bool,
               next_state: tuple[float, float, float, float]) -> None:

        """ Updates the state action values.

        Parameters:
        - state (tuple[float]): the discretized version of the current state
        - action (int): the index of the selected action
        - reward (int): the reward obtained after performing the action in the current state
        - terminated (bool): a flag that indicates if the episode was terminated
        - next_state (tuple[float]): the discretized version of the next state
        
        """

        # Applying the Q-learning algorithm
        future_q_value = (not terminated) * np.max(self.q_values[next_state])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[state][action]

        self.q_values[state][action] += self.learning_rate * temporal_difference

    def decay_epsilon(self, step: int) -> None:
        """Decreases the value of the epsilon threshold over time.
        
        Parameters:
        - step (int): the number of steps passed until the current moment
        """
        self.epsilon =  max(self.min_epsilon, min(1.0, 1.0 - math.log10((step + 1) / 25)))
    
    def decay_learning_rate(self, step: int) -> None:
        """Decreases the value of the learning rate over time.
        
        Parameters:
        - step (int): the number of steps passed until the current moment
        """
        self.learning_rate =  max(self.min_learning_rate, min(1.0, 1.0 - math.log10((step + 1) / 25)))

    def train(self, env: gym.wrappers, num_episodes: int) -> list:
        """Train the agent using the Q-learning algorithm.
        
        Paraemeters:
        - env (gym.wrappers): the environment
        - num_episodes (int): the number of episodes for which the agent is trained

        Returns:
        - episode_durations (list[int]): the number of timesteps each episode lasted (the same as
        the reward in the CartPole-v1 environment)
        
        """
        episode_durations = []
        obs_bounds = list(zip(env.observation_space.low, env.observation_space.high))
        obs_bounds[1] = (-0.5, 0.5)
        obs_bounds[3] = (-math.radians(50), math.radians(50))

        for i_episode in tqdm(range(num_episodes)):
            # Get the observation from the environment
            obs, _ = env.reset()

            # Discretize the continuous values corresponding to the current state
            state = bucketize(obs, obs_bounds)

            # Decrease both the epsilon threshold and the learning rate
            self.decay_epsilon(i_episode)
            self.decay_learning_rate(i_episode)

            # Play one episode
            for t in count():
                
                # Select action
                action = self.select_action(state)

                # Execute the selected action
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Discretize the continuous values corresponding to the next state
                next_state = bucketize(next_obs, obs_bounds)

                # Update the agent
                self.update(state, action, reward, terminated, next_state)

                # Move to the next state
                state = next_state

                # If the episode was terminated or truncated
                # store the episode duration 
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

    # Declare the Q-learning agent
    agent = QLAgent(
        n_actions=n_actions,
        min_learning_rate = min_learning_rate,
        discount_factor = discount_factor,
        min_epsilon = min_epsilon
    )

    # Train the agent
    episode_durations = agent.train(env=env, num_episodes=num_episodes)

    # Plot its performance
    plot_reward(algorithm="QL", episode_durations=episode_durations, num_episodes=num_episodes)

    # Close the environment
    env.close()