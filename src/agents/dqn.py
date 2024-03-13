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

# Import the plotting functions
from src.utils.plot import plot_single_episode, plot_multiple_episodes

# Declare a namedtuple to store transitions 
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayBuffer(object):
    """ A buffer that keeps track of the most recent transitions.
    
    Attributes:
        memory (collections.queue): The internal memory where the transitions are stored.

    Methods:
        push (*args): Adds a transition in the memory.
        sample(batch_size): Randomly sample a batch of transitions from the buffer.
        len(): Get how many transitions are stored in the memory.
    
    """

    def __init__(self, capacity: int) -> None:
        """        
        Args:
            capacity (int): The maxium size of the buffer. 
        """

        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """ Adds a transition in the memory.

        Args:
            *args: Components of the transition to be stored (state,
        action, reward and next state)
        """

        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list:
        """ Randomly sample a batch of transitions from the buffer.
        
        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            (list): List of randomly sampled transitions
        """

        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """ Gets the current size of the internal memory.
        
        Returns:
            (int): The number of transitions stored in the buffer.
        """

        return len(self.memory)
    
class DQN(nn.Module):
    """ Implements a Deep Q-Network model.
    
    Methods:
        forward(state):  Defines the forward pass of the DQN model.
    """

    def __init__(self, n_observations: int, n_actions: int):
        """ Initializes the DQN model with two linear layers.

        Args:
            n_observations (int): The number of dimensions of the observation space.
            n_actions (int): The number of possible actions the agent can take.
        """

        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, n_actions)
       
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """ Defines the forward pass of the DQN model.

        Args:
            state (torch.Tensor): The input state tensor for which action values are to be predicted.

        Returns:
            (torch.Tensor): The predicted action values for the input state.
        """

        state = F.relu(self.layer1(state))
        return self.layer2(state)

class DQNAgent:
    """ Implements a an agent that learns using a DQN.
    
    Attributes:
        n_actions (int): Number of possible actions in the environment.
        n_obs (int): Number of observations from the environment.
        batch_size (int): Size of batches to sample from the replay buffer.
        discount_factor (float): Discount factor for future rewards.
        learning_rate (float): Learning rate for the optimizer.
        update_rate (float): Rate at which the target network is updated.
        epsilon (float): Epsilon value for epsilon-greedy action selection.
        epsilon_start (float): Starting value of epsilon.
        epsilon_end (float): Minimum value of epsilon.
        epsilon_decay (int): Rate of decay for epsilon.
        policy_net (DQN): The current policy network.
        target_net (DQN): The target network for stable Q-value estimation.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for learning the policy network's parameters.
        buffer (ReplayBuffer): Replay buffer for storing experiences.
        steps_done (int): Counter for the number of steps taken (for epsilon decay).

    Methods:
        decay_epsilon():  Decays the epsilon value used for epsilon-greedy action selection,
    based on the number of steps taken.
        select_action(state): Selects an action using epsilon-greedy policy based on the current state.
        update(): Updates the policy network based on a batch of experiences sampled from the replay buffer.
        train(env, num_episodes): Trains the agent on the given environment for a specified number of episodes.
    """

    def __init__(self,
                 n_actions: int,
                 n_obs: int,
                 batch_size: int, 
                 discount_factor: float, 
                 epsilon_start: float,
                 epsilon_end: float, 
                 epsilon_decay: int,
                 update_rate: float, 
                 learning_rate: float,
                 seed: int):
        
        """
        Args:
            n_actions (int): Number of possible actions in the environment.
            n_obs (int): Number of observations from the environment.
            batch_size (int): Size of batches to sample from the replay buffer.
            discount_factor (float): Discount factor for future rewards.
            epsilon_start (float): Starting value of epsilon for epsilon-greedy action selection.
            epsilon_end (float): Minimum value of epsilon after decay.
            epsilon_decay (int): Rate of decay for epsilon, affecting how quickly it decreases.
            update_rate (float): Rate at which the target network's weights are updated towards 
        the policy network's weights.
            learning_rate (float): Learning rate for the optimizer.
            seed (int): The seed used for resetting the environment.  
        """
        
        self.n_actions = n_actions
        self.n_obs = n_obs
        self.batch_size = batch_size

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.update_rate = update_rate

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQN(self.n_obs, self.n_actions)
        self.target_net = DQN(self.n_obs, self.n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), 
                                     lr = self.learning_rate, amsgrad = True)
        self.buffer = ReplayBuffer(10000)

        self.steps_done = 0

        self.seed = seed

    def decay_epsilon(self) -> None:
        """
        Decays the epsilon value used for epsilon-greedy action selection,
        based on the number of steps taken.
        """
        
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        self.steps_done += 1

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """ Selects an action using epsilon-greedy policy based on the current state.

        Args:
        - state (torch.Tensor): The current state of the environment.

        Returns:
        - torch.Tensor: The action to be taken.
        """

        # If the random selected number is bigger than epsilon, then select
        # the greedy action. Otherwise, select a random action.
        sample = random.random()
        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
            
        else:
            return torch.tensor([[random.randint(0, self.n_actions - 1)]], dtype = torch.long)

    def update(self) -> None:
        """
        Updates the policy network based on a batch of 
        experiences sampled from the replay buffer.
        """

        # If there are not enough samples available, don't update
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample a batch of experiences from the buffer. 
        transitions = self.buffer.sample(self.batch_size)
        
        # Convert a batch array of Transitions into a Transition of batch arrays
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute the state action values Q(s_t, a) using the policy network and 
        # select the columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute the state values V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute the expected state action values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute the loss between current and expected state action values
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, env: gym.wrappers, num_episodes: int) -> list:
        """ Trains the agent on the given environment for a specified number of episodes.

        Args:
            env (gym.wrappers): The environment to train the agent on.
            num_episodes (int): The number of episodes to train the agent for.

        Returns:
            (list): A list containing the duration of each episode.
        """

        episode_durations = []

        for i_episode in tqdm(range(num_episodes)):
            
            # Reset the environment (start new episode) and get the starting state
            state, _ = env.reset(seed = self.seed)

            # Convert the state into a tensor to facilitate training
            state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
                
            for t in count():
                
                # Decrease the value of epsilon
                self.decay_epsilon()

                # Select an action using the epsilon-greedy policy
                action = self.select_action(state)
                
                # Execute action
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward])
                done = terminated or truncated

                # Mark the next state as None if episode is done
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype = torch.float32).unsqueeze(0)

                # Store the transition into the buffer
                agent.buffer.push(state, action, reward, next_state)

                # Move to the next state
                state = next_state

                # Update the policy network
                self.update()

                # Perform a soft update on the target network's weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.update_rate + \
                                            target_net_state_dict[key] * (1 - self.update_rate)
                self.target_net.load_state_dict(target_net_state_dict)

                # If episode is done, record the episode duration (associated with the reward)
                if done:
                    episode_durations.append(t + 1)
                    break
                    
        return episode_durations

if __name__ == "__main__":

    # Declare the environment
    env = gym.make("CartPole-v1")

    # Set the hyper-parameters
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-3

    # Get the number of dimensions of the observation space 
    # and the number of available actions
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Set the number of episodes
    num_episodes = 450

    # Declare a list to store the performance of the algorithm over seeds
    episode_durations_over_seeds = []

    for seed in [1, 2, 3, 5, 8]: # Fibonacci seeds

        # Set the seed
        torch.manual_seed(seed)
        random.seed(seed)

        # Declare the DQN agent
        agent = DQNAgent(
            n_actions = n_actions,
            n_obs = n_observations,
            batch_size = BATCH_SIZE,
            discount_factor = GAMMA,
            epsilon_start = EPS_START,
            epsilon_decay = EPS_DECAY,
            epsilon_end = EPS_END,
            update_rate = TAU,
            learning_rate = LR,
            seed = seed
        )

        # Train the DQN agent
        episode_durations = agent.train(env = env, num_episodes = num_episodes)

        # Record the performance of the algorithm
        episode_durations_over_seeds.append(episode_durations)

    # Plot the performance recorded over the last seed
    seed_episode_durations = episode_durations_over_seeds[0]
    plot_single_episode(algorithm = "DQN", episode_durations = seed_episode_durations, num_episodes = num_episodes)

    # Plot the performance of the algorithm over the seeds
    plot_multiple_episodes(algorithm = "DQN", episode_durations_over_seeds = episode_durations_over_seeds)

    # Close the environment
    env.close()