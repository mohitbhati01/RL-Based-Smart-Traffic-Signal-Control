import matplotlib.pyplot as plt
import numpy as np

def plot_training(rewards, waitings, filename_prefix="results"):
#rewards and waitings are lists of episode aggregates
    episodes = np.arange(len(rewards))
    plt.figure(figsize=(10,4))
    plt.plot(episodes, rewards)
    plt.title("Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_reward.png")
    plt.close()

    plt.figure(figsize=(10,4))
    plt.plot(episodes, waitings)
    plt.title("Average Total Queue Length per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Sum of Queues")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_queues.png")
    plt.close()
