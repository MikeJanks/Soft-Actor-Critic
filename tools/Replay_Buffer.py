import numpy as np
import random

class Replay_Buffer:
    def __init__(self, buffer, capacity):
        self.capacity   = capacity
        self.keys       = buffer.keys()
        self.buffer     = buffer
        self.counter    = 0


    def add(self, *entry):
        i = self.counter % self.capacity
        for k, v in zip(self.keys, entry):
            self.buffer[k][i] = v
        self.counter+=1
    
    
    def sample(self, batch_size):
        N = self.capacity if self.counter > self.capacity else self.counter
        
        indices = np.random.choice(N, batch_size, replace=False)
        samples = []
        for k in self.keys:
            samples.append(self.buffer[k][indices])
        
        return samples

    def is_ready(self, batch_size):
        if self.counter >= batch_size:
            return True
        else:
            return False