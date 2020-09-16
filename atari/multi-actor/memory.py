import numpy as np
import threading
from custom.memory import RingBuffer, PriorityReplayMemory
from custom.welford import Welford
import time


class ExperienceReplayMemory:
    def __init__(self, expChan, size):
        self.priorities = PriorityReplayMemory(size)
        self.expChan = expChan
        self.lock = threading.Lock()

    def _append(self, exp):
        # append to PER sum tree
        self.priorities.newLeaf(exp)
    def load(self):
        while True:
            self.lock.acquire()
            self._append( self.expChan.get() )
            self.lock.release()

    def _append_nolock(self, exp, priority):
        # append to PER sum tree
        self.priorities.newPrioritizedLeaf(exp, priority)
    def load_nolock(self):
        while True:
            exp = self.expChan.get()
            priority = exp[5]
            self._append_nolock(exp, priority)

    """
    Update must be called after sample to ensure correct priority updating.
    This is enforced by the locking mechanizm.
    """
    def sample(self, n):
        # acquire sync lock
        self.lock.acquire()
        experiences, treeIndices, isWeights = self.priorities.sample(n)
        batch = self._processExperiences(experiences)
        return treeIndices, batch, isWeights
    def update(self, indices, priorities):
        self.priorities.batchUpdate(indices, priorities)
        # release sync lock
        self.lock.release()

    def sample_nolock(self, n):
        # sample without acquiring lock, with no update of priorities afterward
        experiences, _, isWeights = self.priorities.sample(n)
        batch = self._processExperiences(experiences)
        return batch, isWeights

    def uniform_sample(self, n):
        # no lock is required when uniformly sampling, since no updates take place afterward
        experiences = self.priorities.uniform_sample(n)
        batch = self._processExperiences(experiences)
        return batch

    def _processExperiences(self, experiences):
        A = { "curr_states": [], "next_states": [], "actions": [], "rewards_i":[], "rewards_e":[], "td_error":[], "is_terminal": [] }
        for exp in experiences:
            A["curr_states"].append( exp[0] )
            A["next_states"].append( exp[1] )
            A["actions"].append( exp[2] )
            A["rewards_i"].append( exp[3] )
            A["rewards_e"].append( exp[4] )
            A["td_error"].append( exp[5] )
            A["is_terminal"].append( exp[6] )
        # convert all lists to numpy arrays
        for key in A:
            A[key] = np.array(A[key])
        batch = ( A["curr_states"], A["next_states"], A["actions"], A["rewards_i"], A["rewards_e"], A["td_error"], A["is_terminal"] )
        return batch

    def __len__(self):
        # Returns an estimate of how many experiences have been loaded.
        return len(self.priorities)

