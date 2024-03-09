import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_single_episode(algorithm: str, episode_durations: list, num_episodes: int) -> None:
    """ Plots the raw episode durations over time, the average duration of the last 100 episodes
    in each point and the average duration of the previous episodes in each point. 

    Args:
        algorithm (str): The name of the reinforcement learning algorithm.
        episode_durations (list): A list of episode durations obtained by the algorithm (same as rewards).
        num_episodes (int): The total number of episodes over which the algorithm was run.
    """

    # Plot the raw performance
    plt.plot(range(num_episodes), episode_durations, label = f"{algorithm}")

    # Plot the average of the last 100 episodes in each point
    rolling_averages = []
    for i in range(num_episodes):
        if i >= 100:
            rolling_averages.append(sum(episode_durations[i-100: i]) / 100)
        else:
            rolling_averages.append(sum(episode_durations[:i]) / 100)

    plt.plot(range(num_episodes), rolling_averages, label = "avg100")

    # Plot the average of the previous episodes in each point
    average = []
    for i in range(num_episodes):
        average.append(sum(episode_durations[:i]) / (i + 1))
    plt.plot(range(num_episodes), average, label = "average")

    plt.legend()
    plt.title(f"{algorithm} Performance Over Time")
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(f"data/{algorithm}.png")
    plt.close()

def plot_multiple_episodes(algorithm: str, episode_durations_over_seeds: list) -> None:
    """ Plots the performance of an algorithm over multiple episodes across different seeds.

    Args:
        algorithm (str): The name of the algorithm being evaluated.
        episode_durations_over_seeds (str): A list of lists where each sublist represents
                                      the episode durations for a specific seed.
    """

    df1 = pd.DataFrame(episode_durations_over_seeds).melt()
    df1.rename(columns={"variable": "episodes", "value": "reward"}, inplace=True)
    sns.set_theme(style="darkgrid", context="talk", palette="rainbow")
    sns.lineplot(x="episodes", y="reward", data=df1).set(
        title=f"{algorithm} Performance Over Time"
    )
    plt.savefig(f"data/{algorithm}_over_seeds.png")
    plt.close()