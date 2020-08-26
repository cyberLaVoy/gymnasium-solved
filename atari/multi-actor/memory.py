import numpy as np
from custom.memory import RingBuffer, PriorityReplayMemory


class PolicyReplayMemory:
    def __init__(self, expChan, size):
        self.priorities = PriorityReplayMemory(size)
        self.expChan = expChan

    def append(self, exp):
        self.priorities.newLeaf(exp)

    def updatePriorities(self, indices, newPriorities):
        self.priorities.batchUpdate(indices, newPriorities)

    def sample(self, n):
        A = {"curr_states": [], "next_states": [], 
                     "actions": [], "rewards":[], "is_terminal": []}
        experiences, treeIndices, isWeights = self.priorities.sample(n)
        for exp in experiences:
            A["curr_states"].append( exp[0] )
            A["next_states"].append( exp[1] )
            A["actions"].append( exp[2] )
            A["rewards"].append( exp[3] )
            A["is_terminal"].append( exp[4] )
        for key in A:
            A[key] = np.array(A[key])
        batch = (A["curr_states"], A["next_states"], A["actions"], A["rewards"], A["is_terminal"])
        return treeIndices, batch, isWeights

    def load(self):
        for _ in range(self.expChan.qsize()):
            self.append( self.expChan.get() )

    def __len__(self):
        return len(self.priorities)


class ActorReplayMemory:
    def __init__(self, expChan):
        self.expChan = expChan

    def append(self, exp):
        self.expChan.put(exp)
