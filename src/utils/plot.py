import numpy as np
import matplotlib.pyplot as plt

def plot_rolling_average(algorithm: str, episode_durations: list, rolling_length: int):
    reward_moving_average = (
        np.convolve(
            np.array(episode_durations), np.ones(rolling_length), mode="valid"
        )
        / rolling_length
    )
    plt.plot(range(len(reward_moving_average)), reward_moving_average)
    plt.title(f"{algorithm} Performance Over Time")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()