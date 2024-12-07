import numpy as np
from collections import namedtuple, deque ## 

class Experience_replay_buffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer',
                                 field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = np.empty(self.memory_size, dtype=[("priority", np.float32), ("experience", self.Buffer)]) #deque(maxlen=memory_size)
        
        self.priorities = np.array([])
        self.priorities_prob = np.array([])
        self.alpha = 0.5
        self.sampled_priorities = np.array([])
        self._buffer_length = 0 # current number of prioritized experience tuples in buffer

    def sample_batch(self, batch_size=32):

        samples = np.random.choice(np.arange((self.replay_memory[:self._buffer_length]["priority"]).size), batch_size,
                                   replace=True,p = self.compute_probability())
        self.sampled_priorities = samples
        
        experiences = self.replay_memory["experience"][samples]        
        
        return experiences

    def append(self, s_0, a, r, d, s_1):
        priority = 1.0 if self._buffer_length == 0 else self.replay_memory["priority"].max()
        if self._buffer_length==self.memory_size:
            if priority > self.replay_memory["priority"].min():
                idx = self.replay_memory["priority"].argmin()
                self.replay_memory[idx] = (priority, self.Buffer(s_0, a, r, d, s_1))
            else:
                pass # low priority experiences should not be included in buffer
        else:
            self.replay_memory[self._buffer_length]=(priority, self.Buffer(s_0, a, r, d, s_1))
            self._buffer_length += 1

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in

    def capacity(self):
        return len(self.replay_memory) / self.memory_size
    
    def sum_scaled_priorities(self,scaled_priorities):
        return np.sum(scaled_priorities)
    
    def compute_probability(self):
        scaled_priorities = (self.replay_memory[:self._buffer_length]["priority"])
        
        self.priorities_prob = (scaled_priorities**self.alpha)/np.sum(scaled_priorities**self.alpha)
        return self.priorities_prob