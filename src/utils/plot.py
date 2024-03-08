import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_single_episode(algorithm: str, episode_durations: list, num_episodes: int) -> None:
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

def plot_multiple_episodes(algorithm: str, episode_durations_over_seeds: list) -> None:
    df1 = pd.DataFrame(episode_durations_over_seeds).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set_theme(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title=f"{algorithm} Performance Over Time"
    )
    plt.savefig(f"data/{algorithm}_over_seeds.png")