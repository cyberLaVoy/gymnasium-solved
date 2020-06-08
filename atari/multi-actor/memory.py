import random
import numpy as np

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

    def max(self):
        return max( [v for v in self] )
    def sum(self):
        return sum( [v for v in self] )

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

class SumTree:
    data_pointer = 0
    n_entries = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity 
    
    def add(self, priority, data):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def update(self, tree_index, priority):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
        
    def get_leaf(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index
        data_index = leaf_index - self.capacity + 1
        return leaf_index, data_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0]

class PriorityReplayMemory:
    e = .001
    alpha = .6
    beta = .4
    betaAnneal = 1/250000
    maxPriority = 1.0 
       
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
      
    def newLeaf(self, exp):
        priority = np.max(self.tree.tree[-self.tree.capacity:])
        if priority == 0:
            priority = self.maxPriority
        self.tree.add(priority, exp) 

    def _annealBeta(self):
        self.beta = min(1.0, self.beta+self.betaAnneal ) 

    def sample(self, n):
        if n > self.tree.n_entries:
            n = self.tree.n_entries
        treeIndices = np.zeros(n).astype(np.uint32)
        priorities = np.zeros(n)
        data = []
        delta_p = self.tree.total_priority / n
        p = 0
        for i in range(n):
            v = np.random.uniform(p, p+delta_p)
            ti, di, priority, exp = self.tree.get_leaf(v)
            data.append(exp)
            treeIndices[i] = ti
            priorities[i] = priority
            p += delta_p

        probabilities = priorities / self.tree.total_priority
        isWeights = np.power(self.tree.n_entries * probabilities, -self.beta)
        isWeights /= isWeights.max()
        self._annealBeta()

        return data, treeIndices, isWeights

    def batchUpdate(self, treeIdx, errors):
        errors += self.e
        np.minimum(errors, self.maxPriority, out=errors)
        np.power(errors, self.alpha, out=errors)
        for ti, priority in zip(treeIdx, errors):
            self.tree.update(ti, priority)

    def __len__(self):
        return self.tree.n_entries

class LearnerReplayMemory:
    def __init__(self, actorChans, size=150000):
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
                #print("Loading")
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
        #print("Dumping")
        self.learnerChan.put( self.experiences[:] )
        self.experiences.clear()



