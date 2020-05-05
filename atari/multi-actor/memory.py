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

class PriorityIndexer:
    e = .001
    alpha = .6
    beta = .4
    betaAnneal = 1/250000
    maxPriority = 1.0 
       
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
      
    def newLeaf(self):
        self.tree.add(self.maxPriority, None) 

    def _annealBeta(self):
        self.beta = min(1.0, self.beta+self.betaAnneal ) 

    def sample(self, n):
        dataIndices = np.zeros(n).astype(np.uint32)
        treeIndices = np.zeros(n).astype(np.uint32)
        priorities = np.zeros(n)
        delta_p = self.tree.total_priority / n
        p = 0
        for i in range(n):
            v = np.random.uniform(p, p+delta_p)
            ti, di, priority, _ = self.tree.get_leaf(v)
            dataIndices[i] = di
            treeIndices[i] = ti
            priorities[i] = priority
            p += delta_p

        probabilities = priorities / self.tree.total_priority
        isWeights = np.power(self.tree.n_entries * probabilities, -self.beta)
        isWeights /= isWeights.max()
        self._annealBeta()

        return dataIndices, treeIndices, isWeights

    def batchUpdate(self, treeIdx, errors):
        errors += self.e
        np.minimum(errors, self.maxPriority, out=errors)
        np.power(errors, self.alpha, out=errors)
        for ti, priority in zip(treeIdx, errors):
            self.tree.update(ti, priority)

class ReplayMemory:
    def __init__(self, startState, actionSpace, size=1000000, seed=69, prioritized=True):
        """
        Frames to info buffer alignment example:
        [f1, f2, f3, f4, f5, f6]
                    [i1, i2]
        *After capacity reached
        [f2, f3, f4, f5, f6, f7]
                    [i2, i3]
        """
        self.frames = RingBuffer( size + 4 )
        # info contains (action, reward, is_terminal) tuples
        self.info = RingBuffer(size)
        # controls logic for prioritized replay experience
        self.prioritized = prioritized
        if self.prioritized:
            self.priorities = PriorityIndexer(size)
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
        self.frames.append( mem[0] )
        self.info.append( (mem[1], mem[2], mem[3]) )
        if self.prioritized:
            self.priorities.newLeaf()

    def _priorityIndices(self, n):
        return self.priorities.sample(n)

    def _randomValidIndices(self, n):
        if n > len(self.info):
           n = len(self.info) 
        validIndices = range(0, len(self.info)) 
        return random.sample(validIndices, n)

    def sample(self, n):
        A = {"curr_states": [], "next_states": [], 
                     "actions": [], "rewards":[], "is_terminal": []}
        if self.prioritized:
            dataIndices, treeIndices, isWeights = self._priorityIndices(n)
        else:
            dataIndices, treeIndices, isWeights = self._randomValidIndices(n), None, np.ones(n)
        for inx in dataIndices:
            A["curr_states"].append( self.getState(inx) )
            A["next_states"].append( self.getState(inx+1) )
            A["actions"].append( self.info[inx][0] )
            A["rewards"].append( self.info[inx][1] )
            A["is_terminal"].append( self.info[inx][2] )
        for key in A:
            A[key] = np.array(A[key])
        batch = (A["curr_states"], A["next_states"], A["actions"], A["rewards"], A["is_terminal"])
        return treeIndices, batch, isWeights

    def updatePriorities(self, indices, newPriorities):
        if self.prioritized:
            self.priorities.batchUpdate(indices, newPriorities)
        else:
            pass

    def __len__(self):
        return len(self.info)





