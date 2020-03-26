import random, pickle, os

class RingBuffer:
    def __init__(self, size):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        
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

class ReplayMemory(RingBuffer):
    def __init__(self, size, seed=69, name="replay_memory"):
        RingBuffer.__init__(self, size)
        self.name = name + ".pk"
        random.seed(seed)
    def __del__(self):
        print("Saving replay memory...")
        self.save()

    def sample(self, n):
        validIndices = range(0, RingBuffer.__len__(self))
        indices = random.sample(validIndices, n) 
        return [ RingBuffer.__getitem__(self, inx) for inx in indices ]
    
    def save(self):
        with open(self.name, "wb") as fout:
            state = (self.data, self.start, self.end)
            pickle.dump(state, fout)
    def load(self):
        if os.path.exists(self.name):
            with open(self.name, "rb") as fin:
                state = pickle.load(fin)
                self.data, self.start, self.end = state


class Top100():
    def __init__(self, best=0, seed=69, name="top_100_replay"):
        self.data = RingBuffer(size=100)
        self.name = name + ".pk"
        self.best = 0
        random.seed(seed)

    def rand(self):
        inx = random.randint(0, len(self.data)-1)
        return self.data[inx]

    def append(self, new):
        score = new[0]
        if score > self.best:
            self.data.append(new)
            self.best = score
        else:
            for i in range(len(self.data)):
                if score > self.data[i][0]:
                    self.data[i] = new
                



