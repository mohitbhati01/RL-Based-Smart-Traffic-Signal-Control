import numpy as np

class TrafficEnv:
    """
    Simple 4-way intersection environment.
    State: [n_north, n_south, n_east, n_west, current_phase, time_since_change]
    current_phase: 0 => NS green, 1 => EW green
    Action: 0 => keep, 1 => switch
    Each step simulates 1 second.
    Cars arrive as Poisson process.
    """
    def __init__(self,
                 max_queue=20,
                 arrival_rate=0.2,  #mean arrivals per lane per timestep
                 pass_rate=1,       #cars that can pass per green lane per timestep
                 max_time=200,      #max timesteps per episode
                 seed=None):
        self.max_queue = max_queue
        self.arrival_rate = arrival_rate
        self.pass_rate = pass_rate
        self.max_time = max_time
        self.rng = np.random.RandomState(seed)
        self.reset()


#start with small random queues
    def reset(self):
        self.queues = np.zeros(4, dtype=int)
        self.phase = 0  
        self.time = 0
        self.time_since_change = 0
        obs = self._get_obs()
        return obs
    
#return clipped queue lengths and one-hot phase
    def _get_obs(self):
        
        clipped = np.clip(self.queues, 0, self.max_queue)
        return np.concatenate([clipped, np.array([self.phase, self.time_since_change])]).astype(np.float32)

    def step(self, action):
        """
        action: 0 keep, 1 switch
        returns: obs, reward, done, info
        """
        assert action in [0, 1]
        reward = 0.0
        done = False

#apply action
        if action == 1:
            self.phase = 1 - self.phase
            self.time_since_change = 0
        else:
            self.time_since_change += 1

        #cars pass depending on phase
        if self.phase == 0:  

            #north = lane 0, south = lane 1
            for lane in [0, 1]:
                passed = min(self.pass_rate, self.queues[lane])
                self.queues[lane] -= passed
                reward += passed * 2.0  #reward for clearing cars

        else:  #green
            for lane in [2, 3]:
                passed = min(self.pass_rate, self.queues[lane])
                self.queues[lane] -= passed
                reward += passed * 2.0

        # new arrivals
        arrivals = self.rng.poisson(self.arrival_rate, size=4)
        self.queues += arrivals
        overflow = np.maximum(self.queues - self.max_queue, 0)
        self.queues = np.clip(self.queues, 0, self.max_queue)

        #negative reward for waiting and queue length
        total_wait = np.sum(self.queues)
        reward -= total_wait * 0.5  
        reward -= np.sum(arrivals) * 0.2  # small penalty for arrivals (encourages clearing)

        self.time += 1
        done = (self.time >= self.max_time)

        obs = self._get_obs()
        info = {"queues": self.queues.copy(), "time": self.time}
        return obs, reward, done, info

    def render(self):
        print(f"t={self.time:3d} phase={'NS' if self.phase==0 else 'EW'} queues={self.queues}")
