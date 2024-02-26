import numpy as np
import matplotlib.pyplot as plt

def plot_episode_durations(algorithm: str, episode_durations: list, num_episodes: int):
    """
    Plots the episode durations over time.

    Parameters:
    - algorithm (str): The name of the reinforcement learning algorithm.
    - episode_durations (list): A list of episode durations obtained by the algorithm (same as rewards).
    - num_episodes (int): The total number of episodes over which the algorithm was run.
    """
    plt.plot(range(num_episodes), episode_durations)
    plt.title(f"{algorithm} Performance Over Time")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"data/{algorithm}.png")