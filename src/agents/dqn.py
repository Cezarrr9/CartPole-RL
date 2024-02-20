# import random
# import numpy as np
# import matplotlib.pyplot as plt
# from collections import deque, namedtuple
# from tqdm import tqdm
# import gymnasium as gym

# import math 

# import torch
# import torch.nn as nn
# import torch.nn.functional as F 
# import torch.optim as optim

# Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

# class ReplayMemory(object):

#     def __init__(self, capacity: int):
#         self.memory = deque([], maxlen = capacity)

#     def push(self, *args):
#         self.memory.append(Transition(*args))

#     def sample(self, batch_size: int):
#         return random.sample(self.memory, batch_size)
    
#     def __len__(self):
#         return len(self.memory)
    
# class DQN(nn.Module):

#     def __init__(self, n_obs: int, n_actions: int):
#         super(DQN, self).__init__()
#         self.layer1 = nn.Linear(n_obs, 128)
#         self.layer2 = nn.Linear(128, 128)
#         self.layer3 = nn.Linear(128, n_actions)

#     def forward(self, obs):
#         obs = F.relu(self.layer1(obs))
#         obs = F.relu(self.layer2(obs))
#         return self.layer3(obs)

# class DQNAgent:

#     def __init__(self, 
#                  action_space,
#                  n_obs: int,
#                  batch_size: int,
#                  learning_rate: float, 
#                  discount_factor: float, 
#                  epsilon_start: float,
#                  epsilon_end: float,
#                  epsilon_decay: float):

#         self.action_space = action_space

#         self.batch_size = batch_size

#         self.learning_rate = learning_rate
#         self.discount_factor = discount_factor

#         self.epsilon = epsilon_start
#         self.epsilon_end = epsilon_end
#         self.epsilon_decay = epsilon_decay

#         self.policy_net = DQN(n_obs, action_space.n)
#         self.target_net = DQN(n_obs, action_space.n)
#         self.target_net.load_state_dict(self.policy_net.state_dict())

#         self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = learning_rate, amsgrad = True)

#         self.memory = ReplayMemory(1000)

#     # def select_action(self, state: torch.tensor) -> int:
        
#     #     if np.random.random() < self.epsilon:
#     #         with torch.no_grad():
#     #             return self.policy_net(state).max(1).indices.view(1, 1)
        
#     #     else:
#     #         return torch.tensor([[self.action_space.sample()]], dtype = torch.long)

#     # def decay_epsilon(self) -> None:
#     #     self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
#     def select_action(self, state, t):
#         sample = random.random()
#         eps_threshold = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
#             math.exp(-1. * t / 1000)
#         if sample > eps_threshold:
#             with torch.no_grad():
#                 # t.max(1) will return the largest column value of each row.
#                 # second column on max result is index of where max element was
#                 # found, so we pick action with the larger expected reward.
#                 return self.policy_net(state).max(1).indices.view(1, 1)
#         else:
#             return torch.tensor([[self.action_space.sample()]], dtype=torch.long)

#     def update_policy_net(self):

#         if len(self.memory) < self.batch_size:
#             return 
        
#         transitions = self.memory.sample(self.batch_size)
#         batch = Transition(*zip(*transitions))

#         non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                           batch.next_state)), dtype = torch.bool)
#         non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
#         state_batch = torch.cat(batch.state)
#         action_batch = torch.cat(batch.action)
#         reward_batch = torch.cat(batch.reward)

#         q_values = self.policy_net(state_batch).gather(1, action_batch)

#         next_q_values = torch.zeros(self.batch_size)
#         with torch.no_grad():
#             next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

#         expected_q_values = reward_batch + self.discount_factor * next_q_values

#         # Compute Huber loss
#         criterion = nn.SmoothL1Loss()
#         loss = criterion(q_values, expected_q_values.unsqueeze(1))

#         # Optimize the model
#         self.optimizer.zero_grad()
#         loss.backward()

#         # In-place gradient clipping
#         torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
#         self.optimizer.step()

# if __name__ == "__main__":
#     env = gym.make("CartPole-v1")

#     num_episodes = 300

#     learning_rate = 1e-4
#     discount_factor = 0.99
#     epsilon_start = 0.9
#     epsilon_end = 0.05
#     epsilon_decay = epsilon_start / (num_episodes / 2)
#     batch_size = 128
#     TAU = 0.005

#     obs, _ = env.reset()

#     agent = DQNAgent(action_space = env.action_space,
#                     n_obs = len(obs),
#                     learning_rate = learning_rate,
#                     batch_size = batch_size,
#                     discount_factor = discount_factor,
#                     epsilon_start = epsilon_start,
#                     epsilon_end = epsilon_end,
#                     epsilon_decay = epsilon_decay)

#     rewards = []
#     for episode in tqdm(range(num_episodes)):
#         state, info = env.reset()
#         state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
#         t = 0
#         done = False
#         while not done:
#             # agent.decay_epsilon()
#             action = agent.select_action(state, t)
            
#             obs, reward, terminated, truncated, _ = env.step(action.item())
#             t += 1
#             reward = torch.tensor([reward])
#             done = terminated or truncated 

#             if terminated:
#                 next_state = None
#             else:
#                 next_state = torch.tensor(obs, dtype = torch.float32).unsqueeze(0)

#             agent.memory.push(state, action, reward, next_state)
#             state = next_state

#             agent.update_policy_net()

#             target_net_state_dict = agent.target_net.state_dict()
#             policy_net_state_dict = agent.policy_net.state_dict()
#             for key in policy_net_state_dict:
#                 target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
#             agent.target_net.load_state_dict(target_net_state_dict)

#             if done:
#                 rewards.append(t)
#                 break

#     x = np.arange(num_episodes)
#     plt.plot(x, rewards)  # 'o' creates a circular marker for each point
#     plt.title('Average Rewards Over Time')
#     plt.xlabel('Time or Iteration')
#     plt.ylabel('Average Reward')
#     plt.grid(True)
#     plt.show()
# -*- coding: utf-8 -*-
"""
Reinforcement Learning (DQN) Tutorial
=====================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_
            `Mark Towers <https://github.com/pseudo-rnd-thoughts>`_


This tutorial shows how to use PyTorch to train a Deep Q Learning (DQN) agent
on the CartPole-v1 task from `Gymnasium <https://gymnasium.farama.org>`__.

**Task**

The agent has to decide between two actions - moving the cart left or
right - so that the pole attached to it stays upright. You can find more
information about the environment and other more challenging environments at
`Gymnasium's website <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`__.

.. figure:: /_static/img/cartpole.gif
   :alt: CartPole

   CartPole

As the agent observes the current state of the environment and chooses
an action, the environment *transitions* to a new state, and also
returns a reward that indicates the consequences of the action. In this
task, rewards are +1 for every incremental timestep and the environment
terminates if the pole falls over too far or the cart moves more than 2.4
units away from center. This means better performing scenarios will run
for longer duration, accumulating larger return.

The CartPole task is designed so that the inputs to the agent are 4 real
values representing the environment state (position, velocity, etc.).
We take these 4 inputs without any scaling and pass them through a 
small fully-connected network with 2 outputs, one for each action. 
The network is trained to predict the expected value for each action, 
given the input state. The action with the highest expected value is 
then chosen.


**Packages**


First, let's import needed packages. Firstly, we need
`gymnasium <https://gymnasium.farama.org/>`__ for the environment,
installed by using `pip`. This is a fork of the original OpenAI
Gym project and maintained by the same team since Gym v0.19.
If you are running this in Google Colab, run:

.. code-block:: bash

   %%bash
   pip3 install gymnasium[classic_control]

We'll also use the following from PyTorch:

-  neural networks (``torch.nn``)
-  optimization (``torch.optim``)
-  automatic differentiation (``torch.autograd``)

"""
import numpy as np
import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQNAgent:

    def __init__(self,
                 n_actions: int,
                 n_obs: int,
                 batch_size: int, 
                 discount_factor: float, 
                 epsilon_start: float,
                 epsilon_end: float, 
                 epsilon_decay: int,
                 tau: float, 
                 learning_rate: float):
        
        self.batch_size = batch_size

        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.tau = tau

        self.policy_net = DQN(n_obs, n_actions)
        self.target_net = DQN(n_obs, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def set_target_net(self, target_net_state_dict):
        self.target_net.load_state_dict(target_net_state_dict)

    def get_target_state_dict(self):
        return self.target_net.state_dict()
    
    def get_policy_state_dict(self):
        return self.policy_net.state_dict()

    def push_to_memory(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def select_action(self, state, steps_done):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * steps_done / self.epsilon_decay)
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randint(0, 1)]], dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_factor) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    BATCH_SIZE = 128
    GAMMA = 0.99
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 1000
    TAU = 0.005
    LR = 1e-4

    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    episode_durations = []

    num_episodes = 300

    agent = DQNAgent(
        n_actions=n_actions,
        n_obs=n_observations,
        batch_size=BATCH_SIZE,
        discount_factor=GAMMA,
        epsilon_start=EPS_START,
        epsilon_decay=EPS_DECAY,
        epsilon_end=EPS_END,
        tau=TAU,
        learning_rate=LR
    )

    for i_episode in tqdm(range(num_episodes)):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        for t in count():
            action = agent.select_action(state, t + 1)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward])
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

            # Store the transition in memory
            agent.push_to_memory(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = agent.get_target_state_dict()
            policy_net_state_dict = agent.get_policy_state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            agent.set_target_net(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                break

    x = np.arange(num_episodes)
    plt.plot(x, episode_durations)
    plt.show()

    ######################################################################
    # Here is the diagram that illustrates the overall resulting data flow.
    #
    # .. figure:: /_static/img/reinforcement_learning_diagram.jpg
    #
    # Actions are chosen either randomly or based on a policy, getting the next
    # step sample from the gym environment. We record the results in the
    # replay memory and also run optimization step on every iteration.
    # Optimization picks a random batch from the replay memory to do training of the
    # new policy. The "older" target_net is also used in optimization to compute the
    # expected Q values. A soft update of its weights are performed at every step.
    #