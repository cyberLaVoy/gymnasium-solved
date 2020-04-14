import numpy as np

class AnnealingScheduler:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace
        self.epsilon = 1
        self.decay0 = .9/1000000
        self.decay1 = .1/1000000
    
    def _updateEpsilon(self):
        if self.epsilon > .1:
            self.epsilon -= self.decay0
        else:
            self.epsilon -= self.decay1
        self.epsilon = max(0, self.epsilon)

    def getAction(self, state, agent):
        if np.random.random() < self.epsilon:
            a = np.random.choice( self.actionSpace )
        else:
            a = agent.getAction( state )
        self._updateEpsilon()
        return a

    def getEpsilon(self):
        return self.epsilon
