import numpy as np

class Welford:
    def __init__(self):
        self.k = 0
        self.M = 0
        self.S = 0
    
    def update(self, x):
        self.k += 1
        prevMean = self.M 
        self.M += (x-self.M)/self.k
        self.S += (x-self.M)*(x-prevMean)

    @property
    def mean(self):
        return self.M
    @property
    def std(self):
        if self.k == 1:
            return 0
        return np.sqrt(self.S/(self.k-1))