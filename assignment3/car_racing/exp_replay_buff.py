import numpy as np
from collections import namedtuple, deque ## 

class Experience_replay_buffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer',
                                 field_names=['state', 'action', 'reward', 'done', 'next_state','priority'])
        self.replay_memory = deque(maxlen=memory_size)
        #self.priorities_sum = 0
        self.priorities = deque(maxlen=memory_size)
        self.priorities_prob = np.array([])
        self.alpha = 0.6
        self.sampled_priorities = np.array([])
    def sample_batch(self, batch_size=32):

        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False,p = self.compute_probability(self.priorities))
        self.sampled_priorities = samples
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, s_0, a, r, d, s_1,p):
        
        self.replay_memory.append(
            self.Buffer(s_0, a, r, d, s_1,p))
        #print("Pre append",self.priorities)
        self.priorities.append(p)
        #print(len(self.priorities))
        # if(len(self.priorities)>=self.memory_size):
        #     np.delete(self.priorities,0)
        #print("AFTER APPEND",len(self.priorities))
        #print(
        #      "\rPriorities size {:d} \t\t".format(len(self.priorities)), end="")

        #print("post_append",self.priorities)

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in

    def capacity(self):
        return len(self.replay_memory) / self.memory_size
    
    def sum_scaled_priorities(self,scaled_priorities):
        return np.sum(scaled_priorities)
    
    def compute_probability(self, priorities):
        scaled_priorities = np.array(priorities)**self.alpha
        self.priorities_prob = (scaled_priorities)/self.sum_scaled_priorities(scaled_priorities)
        #print(
       #       "\rPriorities Probabilities size {:d} \t\t".format(len(self.priorities)), end="")
        return self.priorities_prob