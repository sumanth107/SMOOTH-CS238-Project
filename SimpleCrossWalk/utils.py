import matplotlib.pyplot as plt
import pandas as pd


def plot_test_performance(test_rewards, path):
    plt.figure()
    plt.grid(linestyle='-.')
    returns_smoothed = pd.Series(test_rewards).rolling(10, min_periods=10).mean()
    plt.plot(test_rewards, linewidth=0.5, label='reward per episode')
    plt.plot(returns_smoothed, linewidth=2.0, label='smoothed reward (over window size=10)')
    plt.legend()
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.title("Episode Reward vs Training Episode")
    plt.savefig(path)