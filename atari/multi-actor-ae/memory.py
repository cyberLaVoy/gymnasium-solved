import numpy as np
from custom.memory import RingBuffer, PriorityReplayMemory

class PolicyReplayMemory:
    def __init__(self, actorChans, size):
        self.priorities = PriorityReplayMemory(size)
        self.actorChans = actorChans

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
        ready = []
        for chan in self.actorChans:
            while not chan.empty():
                experiences = chan.get()
                ready.append(experiences)
        for experiences in ready:
            for exp in experiences:
                self.append(exp)

    def __len__(self):
        return len(self.priorities)

class AEReplayMemory:
    def __init__(self, actorChans, size):
        self.priorities = PriorityReplayMemory(size)
        self.actorChans = actorChans

    def append(self, exp):
        self.priorities.newLeaf(exp)

    def updatePriorities(self, indices, newPriorities):
        self.priorities.batchUpdate(indices, newPriorities)

    def sample(self, n):
        A = {"curr_states": [], "next_states": [], "actions":[]}
        experiences, treeIndices, isWeights = self.priorities.sample(n)
        for exp in experiences:
            A["curr_states"].append( exp[0] )
            A["next_states"].append( exp[1] )
            A["actions"].append( exp[2] )
        for key in A:
            A[key] = np.array(A[key])
        batch = (A["curr_states"], A["next_states"], A["actions"])
        return treeIndices, batch, isWeights

    def load(self):
        ready = []
        for chan in self.actorChans:
            while not chan.empty():
                experiences = chan.get()
                ready.append(experiences)
        for experiences in ready:
            for exp in experiences:
                self.append(exp)

    def __len__(self):
        return len(self.priorities)


class ActorReplayMemory:
    def __init__(self, learnerChan, thresh):
        self.experiences = []
        self.thresh = thresh
        self.learnerChan = learnerChan

    def append(self, exp):
        self.experiences.append(exp)
        if len(self.experiences) == self.thresh:
            self.dump()

    def dump(self):
        self.learnerChan.put( self.experiences[:] )
        self.experiences.clear()
