import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self,
                 state_dim=6,
                 action_dim=2,
                 lr=1e-3,
                 gamma=0.99,
                 buffer_size=50000,
                 batch_size=64,
                 epsilon_start=1.0,
                 epsilon_final=0.05,
                 epsilon_decay=5000,
                 target_update=200):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = epsilon_start
        self.eps_final = epsilon_final
        self.eps_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.total_steps = 0
        self.target_update = target_update

        self.policy_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayBuffer(capacity=buffer_size)

    def act(self, state):
        self.total_steps += 1
        # linear epsilon decay
        self.epsilon = max(self.eps_final, self.eps_start - (self.eps_start - self.eps_final) * (self.total_steps / self.eps_decay))
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            qvals = self.policy_net(state_t)
        return int(torch.argmax(qvals, dim=1).item())

    def push(self, s,a,r,s2,d):
        self.replay.push(s,a,r,s2,d)

    def update(self):
        if len(self.replay) < self.batch_size:
            return 0.0
        s,a,r,s2,d = self.replay.sample(self.batch_size)
        s = torch.tensor(s, dtype=torch.float32).to(device)
        a = torch.tensor(a, dtype=torch.int64).to(device)
        r = torch.tensor(r, dtype=torch.float32).to(device)
        s2 = torch.tensor(s2, dtype=torch.float32).to(device)
        d = torch.tensor(d, dtype=torch.float32).to(device)

        q_values = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0]
        expected = r + (1 - d) * self.gamma * q_next

        loss = nn.MSELoss()(q_values, expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
