import numpy as np
from collections import deque

class Memory():
    def __init__(self,window_size):
        self.buffer=deque(maxlen=window_size)
        
    def add(self,experience):
        self.buffer.append(experience)

    def sample_memories(self,batch_size):
        perm_batch = np.random.permutation(len(self.buffer))[:batch_size]
        mem = np.array(self.buffer)[perm_batch]
        return mem[:,0], mem[:,1], mem[:,2], mem[:,3], mem[:,4]