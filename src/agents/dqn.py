import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, namedtuple, defaultdict
from itertools import count
from tqdm import tqdm
import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):

    def __init__(self, n_obs: int, n_actions: int):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_obs, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, obs):
        obs = F.relu(self.layer1(obs))
        obs = F.relu(self.layer2(obs))
        return self.layer3(obs)

class DQNAgent:

    def __init__(self, 
                 action_space,
                 n_obs: int,
                 batch_size: int,
                 learning_rate: float, 
                 discount_factor: float, 
                 epsilon_start: float,
                 epsilon_end: float,
                 epsilon_decay: float):

        self.action_space = action_space

        self.batch_size = batch_size

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.policy_net = DQN(n_obs, action_space.n)
        self.target_net = DQN(n_obs, action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = learning_rate, amsgrad = True)

        self.memory = ReplayMemory(1000)

    def select_action(self, state: torch.tensor) -> int:
        
        if np.random.random() < self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        
        else:
            return torch.tensor([[self.action_space.sample()]], dtype = torch.long)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

    def update_policy_net(self):

        if len(self.memory) < self.batch_size:
            return 
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype = torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        q_values = self.policy_net(state_batch).gather(1, action_batch)

        next_q_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_q_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_q_values = reward_batch + self.discount_factor * next_q_values

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, expected_q_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
    def train(self, env, num_episodes: int, num_timesteps: int):
        env = gym.wrappers.RecordEpisodeStatistics(env, deque_size = num_episodes)
        reward_over_episodes = []
        for episode in tqdm(range(num_episodes)):
            state, _ = env.reset()
            state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)

            for t in range(num_timesteps):
                action = self.select_action(state)
                self.decay_epsilon()
                obs, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward])
                done = terminated or truncated 

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(obs, dtype = torch.float32).unsqueeze(0)

                self.memory.push(state, action, reward, next_state)

                self.update_policy_net()

                if done:
                    break
            
            if episode % 100 == 0:
                avg_reward = int(np.mean(env.return_queue))
                print("Episode:", episode, "Average Reward:", avg_reward)
            
            reward_over_episodes.append(env.return_queue[-1])
            
            self.target_net.load_state_dict(self.policy_net.state_dict())

        rewards_to_plot = [[reward[0] for reward in reward_over_episodes]] 
        df1 = pd.DataFrame(rewards_to_plot).melt()
        df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
        sns.set_theme(style="darkgrid", context="talk", palette="rainbow")
        sns.lineplot(x="episodes", y="reward", data = df1).set(
            title="DQN for CartPole-v1"
        )
        plt.show()

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    num_episodes = 600
    num_timesteps = 1000

    learning_rate = 1e-4
    discount_factor = 0.99
    epsilon_start = 0.9
    epsilon_end = 0.05
    epsilon_decay = epsilon_start / (num_episodes / 2)
    batch_size = 128

    obs, _ = env.reset()

    agent = DQNAgent(action_space = env.action_space,
                    n_obs = len(obs),
                    learning_rate = learning_rate,
                    batch_size = batch_size,
                    discount_factor = discount_factor,
                    epsilon_start = epsilon_start,
                    epsilon_end = epsilon_end,
                    epsilon_decay = epsilon_decay)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size = num_episodes)
    reward_over_episodes = []
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)

        for t in range(num_timesteps):
            action = agent.select_action(state)
            agent.decay_epsilon()
            obs, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward])
            done = terminated or truncated 

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(obs, dtype = torch.float32).unsqueeze(0)

            agent.memory.push(state, action, reward, next_state)

            agent.update_policy_net()

            if done:
                break
        
        if episode % 100 == 0:
            avg_reward = int(np.mean(env.return_queue))
            print("Episode:", episode, "Average Reward:", avg_reward)
        
        reward_over_episodes.append(env.return_queue[-1])
        
        agent.target_net.load_state_dict(agent.policy_net.state_dict())

    rewards_to_plot = [[reward[0] for reward in reward_over_episodes]] 
    df1 = pd.DataFrame(rewards_to_plot).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set_theme(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data = df1).set(
        title="DQN for CartPole-v1"
    )
    plt.show()