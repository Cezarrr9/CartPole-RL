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
from src.utils.plot import plot_single_episode, plot_multiple_episodes

# Import a function for discretizing the state space
from src.utils.bucketize import bucketize

class SarsaAgent:
    """SARSA Agent
    
    Attributes:
        n_actions (int): The number of actions available for the agent.
        q_values (defaultdict[np.ndarray]): The current state action values.
        learning_rate (float): The value of the current learning rate.
        min_learning_rate (float): The minimum value that the learning rate can take.
        discount_factor (float): The discount factor used in the SARSA algorithm. 
        epsilon (float): The value of the current epsilon threshold.
        min_epsilon (float): The minimum value that the epsilon threshold can take.
        seed (int): The seed used for resetting the environment. 
    
    Methods:
        select_action(state): Selects an action using the epsilon greedy policy.
        update(state, action, reward, terminated, next_state): Updates the state action values.
        decay_epsilon(): Decreases the value of the epsilon threshold over time.
        decay_learning_rate(): Decreases the value of the learning rate over time.
        train(env, num_episodes): Train the agent using the SARSA algorithm.
    """

    def __init__(self,
                 n_actions: int,
                 min_learning_rate: float,
                 discount_factor: float,
                 min_epsilon: float, 
                 seed: int) -> None:
        
        self.n_actions = n_actions

        self.q_values = defaultdict(lambda: np.zeros(n_actions))

        self.learning_rate = float
        self.min_learning_rate = min_learning_rate
        self.discount_factor = discount_factor
        
        self.epsilon = float
        self.min_epsilon = min_epsilon

        self.seed = seed

    def select_action(self, state: tuple[float, float, float, float]) -> int:
        """Selects an action using the epsilon greedy policy.
        
        Args:
            state (tuple[float]): The discretized version of the state.

        Returns:
            (int): The index of an action.
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
               next_state: tuple[float, float, float, float],
               next_action: int) -> None:

        """ Updates the state action values.

        Args:
            state (tuple[float]): The discretized version of the current state.
            action (int): The index of the selected action.
            reward (int): The reward obtained after performing the action in the current state.
            terminated (bool): A flag that indicates if the episode was terminated.
            next_state (tuple[float]): The discretized version of the next state.
            next_action (int): The index of the next selected action.
        """

        # Applying the SARSA algorithm
        future_q_value = (not terminated) * self.q_values[next_state][next_action]
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[state][action]

        self.q_values[state][action] += self.learning_rate * temporal_difference

    def decay_epsilon(self, step: int) -> None:
        """Decreases the value of the epsilon threshold over time.
        
        Args:
            step (int): The number of steps passed until the current moment.
        """

        self.epsilon =  max(self.min_epsilon, min(1.0, 1.0 - math.log10((step + 1) / 25)))
    
    def decay_learning_rate(self, step: int) -> None:
        """Decreases the value of the learning rate over time.
        
        Args:
            step (int): The number of steps passed until the current moment.
        """

        self.learning_rate =  max(self.min_learning_rate, min(1.0, 1.0 - math.log10((step + 1) / 25)))

    def train(self, env: gym.wrappers, num_episodes: int) -> list:
        """Train the agent using the Q-learning algorithm.
        
        Args:
            env (gym.wrappers): The environment where the agent is trained. 
            num_episodes (int): The number of episodes for which the agent is trained.

        Returns:
            episode_durations (list[int]): The number of timesteps each episode lasted (the same as
        the reward in the CartPole-v1 environment).
        """
        
        episode_durations = []
        obs_bounds = list(zip(env.observation_space.low, env.observation_space.high))
        obs_bounds[1] = (-0.5, 0.5)
        obs_bounds[3] = (-math.radians(50), math.radians(50))

        for i_episode in tqdm(range(num_episodes)):
            # Get the observation from the environment
            obs, _ = env.reset(seed = self.seed)

            # Discretize the continuous values corresponding to the current state
            state = bucketize(obs, obs_bounds)

            # Decrease both the epsilon threshold and the learning rate
            self.decay_epsilon(i_episode)
            self.decay_learning_rate(i_episode)

            # Select an action using the epsilon-greedy policy
            action = self.select_action(state)

            # Play one episode
            for t in count():
                
                # Execute the selected action
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Discretize the continuous values corresponding to the next state
                next_state = bucketize(next_obs, obs_bounds)

                # Select the next action
                next_action = self.select_action(next_state)

                # Update the agent
                self.update(state, action, reward, terminated, next_state, next_action)

                # Move to the next state
                state = next_state

                # Switch to the next action
                action = next_action

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

   # Declare a list to store the performance of the algorithm over seeds
    episode_durations_over_seeds = []

    for seed in [1, 2, 3, 5, 8]: # Fibonacci seeds

        # Set the seed
        random.seed(seed)
        np.random.seed(seed)

        # Declare the SARSA agent
        agent = SarsaAgent(n_actions = n_actions,
                           min_learning_rate = min_learning_rate,
                           discount_factor = discount_factor,
                           min_epsilon = min_epsilon,
                           seed = seed)

        # Train the agent
        episode_durations = agent.train(env = env, num_episodes = num_episodes)

        # Record the performance of the algorithm
        episode_durations_over_seeds.append(episode_durations)

    # Plot the performance recorded over the last seed
    seed_episode_durations = episode_durations_over_seeds[0]
    plot_single_episode(algorithm = "SARSA", episode_durations = seed_episode_durations, num_episodes = num_episodes)

    # Plot the performance of the algorithm over the seeds
    plot_multiple_episodes(algorithm = "SARSA", episode_durations_over_seeds = episode_durations_over_seeds)

    # Close the environment
    env.close()
