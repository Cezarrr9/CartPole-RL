import os
import sys

import gymnasium as gym
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
from src.utils.plot import plot_single_episode, plot_multiple_episodes

class PolicyNetwork(nn.Module):
    """ Parameterized policy network.
    
    Methods:
        forward(state): Defines the forward pass through the network.
    """

    def __init__(self, n_observations: int, n_actions: int) -> None:
        """
        Args:
            n_observations (int): The size of the observation space.
            n_actions (int): The size of the action space.
        """

        super().__init__()
        self.layer = nn.Linear(n_observations, n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Defines the forward pass through the network.

        Args:
            state: The current state of the environment.

        Returns:
            The probability distribution over actions.
        """

        state = self.layer(state)
        return F.softmax(state, dim = 1)

class ReinforceAgent:
    """ An agent that uses the REINFORCE algorithm for training a policy network.

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        discount_factor (float): The discount factor for future rewards.
        policy_net (PolicyNetwork): An instance of the PolicyNetwork.
        optimizer (torch.optim.Optimizer): The optimizer for updating the network weights.
        probs (list): A list to store the log probabilities of the actions taken.
        rewards (list): A list to store the rewards obtained.

    Methods:
        select_action(state): Selects an action based on the policy network's output.
        update(): Updates the policy network based on the collected rewards and log probabilities.
        train(env, num_episodes): Train the agent in the environment for a specified number of episodes.
    """

    def __init__(self,
                 n_observations: int, 
                 n_actions: int,
                 learning_rate: float, 
                 discount_factor: float,
                 seed: int) -> None:
        """
        Args:
            n_observations (int): The size of the observation space.
            n_actions (int): The size of the action space.
            learning_rate (float): The learning rate for the optimizer.
            discount_factor (float): The discount factor for future rewards.
            seed (int): The seed used for resetting the environment. 
        """
        self.seed = seed

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.policy_net = PolicyNetwork(n_observations, n_actions)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.learning_rate)
        
        self.probs = [] 
        self.rewards = []

    def select_action(self, state: tuple[float, float, float, float]) -> int:
        """ Selects an action based on the policy network's output.

        Args:
            state (tuple): The current state of the environment.

        Returns:
            (int): The action selected.
        """
        
        # Convert the state into a tensor to facilitate training
        state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)

        # Get the probability distribution
        probs = self.policy_net(state)

        # Create a Categorical distribution to sample
        # from and then sample an action
        c = Categorical(probs)
        action = c.sample()

        # Store the log probability of the action
        self.probs.append(c.log_prob(action))

        return action.item()

    def update(self) -> None:
        """ Updates the policy network based on the collected rewards and log probabilities."""

        g = 0
        returns = []

        # Compute the return for each time step
        for r in self.rewards[::-1]:
            g = r + self.discount_factor * g
            returns.insert(0, g)
        returns = torch.tensor(returns)

        # Compute the loss
        loss = 0
        for log_prob, delta in zip(self.probs, returns):
            loss += log_prob.mean() * delta * (-1)
        
        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset the lists for the next episodes
        self.probs = []
        self.rewards = []

    def train(self, env: gym.wrappers, num_episodes: int) -> list:
        """ Train the agent in the environment for a specified number of episodes.

        Args:
            env (gym.wrappers): The environment to train in.
            num_episodes (int): The number of episodes to train for.

        Returns:
            (list): A list containing the duration of each episode.
        """

        episode_durations = []

        for i_episode in tqdm(range(num_episodes)):

            # Reset the environment (start new episode) and get the starting state
            state, _ = env.reset(seed = self.seed)

            for t in count():

                # Select an action using a policy network
                action = self.select_action(state)

                # Execute action
                state, reward, terminated, truncated, info = env.step(action)
                self.rewards.append(reward)
                done = terminated or truncated 

                # If episode is done, record the episode duration (associated with the reward)
                if done:
                    episode_durations.append(t)
                    break
            
            # Update the policy network
            self.update()

        return episode_durations

if __name__ == "__main__":
    
    # Declare the environment
    env = gym.make("CartPole-v1")

    # Setting the hyperparameters
    learning_rate = 0.01
    discount_factor = 0.99

    # Get the number of dimensions of the observation space 
    # and the number of available actions
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Set the hyperparameters
    num_episodes = 1000

    # Declare a list to store the performance of the algorithm over seeds
    episode_durations_over_seeds = []

    for seed in [1, 2, 3, 5, 8]: # Fibonacci seeds

        # Set the seed
        torch.manual_seed(seed)

        # Declare the Reinforce agent
        agent = ReinforceAgent(learning_rate = learning_rate,
                            discount_factor = discount_factor,
                            n_actions = n_actions,
                            n_observations = n_observations,
                            seed = seed)

        # Train the agent
        episode_durations = agent.train(env = env, num_episodes = num_episodes)

        # Record the performance of the algorithm
        episode_durations_over_seeds.append(episode_durations)

    # Plot the performance recorded over the last seed
    seed_episode_durations = episode_durations_over_seeds[0]
    plot_single_episode(algorithm = "REINFORCE", episode_durations=seed_episode_durations, num_episodes = num_episodes)

    # Plot the performance of the algorithm over the seeds
    plot_multiple_episodes(algorithm = "REINFORCE", episode_durations_over_seeds = episode_durations_over_seeds)

    # Close the environment
    env.close()
