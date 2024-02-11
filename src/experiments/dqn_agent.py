import os
import sys
import gymnasium as gym
from itertools import count
from tqdm import tqdm 
import matplotlib.pyplot as plt
import math

import torch

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

from src.agents.dqn import DQNAgent
from src.utils.bucketize import bucketize

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

env = gym.make("CartPole-v1")
num_episodes = 50

obs, info = env.reset()

env = gym.wrappers.RecordEpisodeStatistics(env, deque_size = num_episodes)
obs_bounds = list(zip(env.observation_space.low, env.observation_space.high))
obs_bounds[1] = (-0.5, 0.5)
obs_bounds[3] = (-math.radians(50), math.radians(50))
obs = bucketize(obs, obs_bounds)
agent = DQNAgent(action_space = env.action_space,
                 n_observations = len(obs),
                 batch_size = 128,
                 discount_factor = 0.99,
                 eps_start = 0.9,
                 eps_end = 0.05,
                 eps_decay = 1000, 
                 tau = 0.005,
                 learning_rate = 1e-4)


episode_durations = []

for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = bucketize(state, obs_bounds)
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    for t in count():
        action = agent.select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward])
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            observation = bucketize(observation, obs_bounds)
            next_state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

        # Store the transition in memory
        agent.memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        agent.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = agent.target_net.state_dict()
        policy_net_state_dict = agent.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*agent.tau + target_net_state_dict[key]*(1-agent.tau)
        agent.target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result = True)
plt.ioff()
plt.show()