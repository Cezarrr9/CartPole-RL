import numpy as np
import matplotlib.pyplot as plt

def plot_reward(algorithm: str, episode_durations: list, num_episodes: int):
    plt.plot(range(num_episodes), episode_durations)
    plt.title(f"{algorithm} Performance Over Time")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"data/{algorithm}.png")