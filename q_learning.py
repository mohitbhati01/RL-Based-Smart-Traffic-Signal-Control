import numpy as np
import pickle

class QLearningAgent:
    """
    Tabular Q-learning agent with discretized state.
    Discretization: queue lengths binned.
    """
    def __init__(self,
                 n_bins=5,
                 max_queue=20,
                 lr=0.1,
                 gamma=0.99,
                 epsilon=1.0,
                 eps_decay=0.995,
                 eps_min=0.05):
        self.n_bins = n_bins
        self.max_queue = max_queue
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        # state: 4 queue bins + phase (2) + time_since_change binned (n_bins)
        state_size = (n_bins,) * 4 + (2,) + (n_bins,)
        self.shape = state_size + (2,)  # 2 actions
        self.q_table = np.zeros(self.shape, dtype=np.float32)

    def _bin(self, value):
        #map 0..max_queue to 0..n_bins-1
        return int(np.clip(value * self.n_bins / (self.max_queue + 1), 0, self.n_bins - 1))

    def state_to_index(self, obs):
        #obs = [q0,q1,q2,q3, phase, time_since_change]
        idx = []
        for i in range(4):
            idx.append(self._bin(obs[i]))
        phase = int(obs[4])
        idx.append(phase)
        idx.append(self._bin(obs[5]))  
        #time_since_change binned
        return tuple(idx)

    def act(self, obs):
        idx = self.state_to_index(obs)
        if np.random.rand() < self.epsilon:
            return np.random.choice([0,1])
        qvals = self.q_table[idx]
        return int(np.argmax(qvals))

    def learn(self, obs, action, reward, next_obs, done):
        idx = self.state_to_index(obs)
        next_idx = self.state_to_index(next_obs)
        q = self.q_table[idx + (action,)]
        q_next = np.max(self.q_table[next_idx])
        target = reward + (0 if done else self.gamma * q_next)
        self.q_table[idx + (action,)] += self.lr * (target - q)
        if done:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load(self, fname):
        import pickle
        with open(fname, 'rb') as f:
            self.q_table = pickle.load(f)
