import numpy as np
import random

class RingBuffer:
    start = 0
    end = 0

    def __init__(self, size):
        self.data = [None] * (size + 1)

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __setitem__(self, idx, v):
        self.data[(self.start + idx) % len(self.data)] = v
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


class ExperienceReplay():
    def __init__(self, size=1000):
        self.buffer = []
        self.size = size
    
    def append(self, experience):
        if len(self.buffer)+1 >= self.size:
            # this feels very Pythonic
            self.buffer[0:1+len(self.buffer)-self.size] = []
        self.buffer.append(experience)
            
    def sample(self, n, lenTrace):
        sampled = random.sample(self.buffer, n)
        sampledTraces = []
        for episode in sampled:
            point = np.random.randint(0,len(episode)+1-lenTrace)
            sampledTraces.append(episode[point:point+lenTrace])
        sampledTraces = np.array(sampledTraces)
        #return np.reshape(sampledTraces, (n*lenTrace,5))
        return sampledTraces