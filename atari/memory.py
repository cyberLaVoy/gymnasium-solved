import random
import numpy as np

def oneHotEncode(arr, opt):
    oneHot = np.zeros((arr.size, opt)).astype(np.bool)
    oneHot[np.arange(arr.size), arr] = 1
    return oneHot

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

class ReplayMemory:
    def __init__(self, startState, actionSpace, size=1000000, seed=69):
        """
        Frames to other buffer alignment example:
        [f1, f2, f3, f4, f5, f6]
                    [a1, a2]
        *After capacity reached
        [f2, f3, f4, f5, f6, f7]
                    [a2, a3]
        """
        self.frames = RingBuffer( size + 4 )
        self.actions = RingBuffer(size)
        self.rewards = RingBuffer(size)
        self.isTerminal = RingBuffer(size)
        # frames buffer must start with 4 non-action frames
        for _ in range(4):
            self.frames.append(startState)

        self.actionSpace = actionSpace
        random.seed( seed )

    def getState(self, inx=None):
        # default to grab last state
        if inx is None:
            inx = len(self.frames)-4
        state = [ self.frames[inx+i] for i in range(4) ]
        return np.dstack( state )

    def append(self, mem):
        # append to all memory parts to keep indexing aligned
        self.frames.append(mem[0])
        self.actions.append(mem[1])
        self.rewards.append(mem[2])
        self.isTerminal.append(mem[3])

    def _randomValidIndices(self, n):
        if n > len(self.actions):
           n = len(self.actions) 
        validIndices = range(0, len(self.actions)) 
        return random.sample(validIndices, n)

    def sample(self, n):
        A = {"curr_states": [], "next_states": [], 
                     "actions": [], "rewards":[], "is_terminal": []}
        for inx in self._randomValidIndices(n):
            A["curr_states"].append( self.getState(inx) )
            A["next_states"].append( self.getState(inx+1) )
            A["actions"].append( self.actions[inx] )
            A["rewards"].append( self.rewards[inx] )
            A["is_terminal"].append( self.isTerminal[inx] )
        for key in A:
            A[key] = np.array(A[key])
        A["actions"] = oneHotEncode(A["actions"], self.actionSpace)
        return A["curr_states"], A["next_states"], A["actions"], A["rewards"], A["is_terminal"]

    def __len__(self):
        return len(self.actions)





