import numpy as np
from env import TrafficEnv
from dqn import DQNAgent
from q_learning import QLearningAgent
from visualize import plot_training
import os
import argparse
import random
import torch

def train_dqn(episodes=500, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = TrafficEnv(max_time=200, arrival_rate=0.3, pass_rate=2)
    agent = DQNAgent(state_dim=6, action_dim=2, lr=1e-3, batch_size=64)
    rewards = []
    queues = []

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0.0
        total_queue = 0.0
        done = False
        while not done:
            state = obs / 20.0  # simple normalization
            action = agent.act(state)
            next_obs, reward, done, info = env.step(action)
            next_state = next_obs / 20.0
            agent.push(state, action, reward, next_state, float(done))
            loss = agent.update()
            obs = next_obs
            total_reward += reward
            total_queue += np.sum(info["queues"])
        rewards.append(total_reward)
        queues.append(total_queue / env.max_time)  # average queue per timestep
        if (ep+1) % 50 == 0:
            print(f"DQN Ep {ep+1}/{episodes} reward={total_reward:.1f} avg_queue={queues[-1]:.2f} eps={agent.epsilon:.3f}")
            agent.save(f"policy_ep{ep+1}.pth")

    os.makedirs("results", exist_ok=True)
    plot_training(rewards, queues, filename_prefix="results/dqn")
    agent.save("results/dqn_final.pth")
    print("DQN training finished. Models and plots saved in results/")

def train_qlearning(episodes=1000, seed=0):
    np.random.seed(seed)
    random.seed(seed)
    env = TrafficEnv(max_time=100, arrival_rate=0.3, pass_rate=1)
    agent = QLearningAgent(n_bins=6, max_queue=env.max_queue, lr=0.1, eps_decay=0.995)
    rewards = []
    queues = []

    for ep in range(episodes):
        obs = env.reset()
        total_reward = 0
        total_queue = 0
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            total_queue += np.sum(info["queues"])
        rewards.append(total_reward)
        queues.append(total_queue / env.max_time)
        if (ep+1) % 100 == 0:
            print(f"Q Ep {ep+1}/{episodes} reward={total_reward:.1f} avg_queue={queues[-1]:.2f}")
    import os
    os.makedirs("results", exist_ok=True)
    plot_training(rewards, queues, filename_prefix="results/qlearning")
    agent.save("results/qlearning_qtable.pkl")
    print("Q-learning finished. Results saved in results/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['dqn','q'], default='dqn')
    parser.add_argument('--episodes', type=int, default=500)
    args = parser.parse_args()
    if args.method == 'dqn':
        train_dqn(episodes=args.episodes)
    else:
        train_qlearning(episodes=args.episodes)
