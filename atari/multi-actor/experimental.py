from tensorflow.keras.models import Model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Dense, Input, Lambda, multiply
from tensorflow.keras.optimizers import Adam
import numpy as np
import os, random
from custom import Noise, loadModel


class ParameterAgent:
    def __init__(self, startParam, low=2, high=64, agentName="parameter_agent", gamma=0.89, modelLoad=None):
        # 0 - no change, 1 - increase, 2 - decrease
        self.actionSpace = 3
        self.changeDefinition = [0, 1, -1]
        self.prevParam = startParam
        self.nextParam = startParam
        self.low = low
        self.high = high
        self.action = 0
        self.gamma = gamma
        self.name = agentName
        self.modelName = self.name + ".h5"
        if modelLoad is None:
            self.model = self._createModel()
        else:
            self.model = loadModel(modelLoad)

    def save(self):
        self.model.save(self.modelName)
    
    def _predict(self, state):
        state = np.array( [state] )
        mask = np.ones( (1, self.actionSpace) )
        prediction = self.model.predict( [state, mask] )
        print(prediction)
        return prediction

    def _getAction(self, state):
        return np.argmax( self._predict(state) )

    def _translateAction(self, action):
        return self.changeDefinition[action]

    def getParam(self):
        return self.nextParam

    def updateParam(self, reward):
        self._learn(reward)
        self.prevParam = self.nextParam
        action = self._getAction( self.prevParam )
        if self.prevParam == self.low and action == 2:
            action = np.random.choice( (0, 1) )
        if self.prevParam == self.high and action == 1:
            action = np.random.choice( (0, 2) )
        self.action = action
        self.nextParam = self.prevParam + self._translateAction( self.action )


    def _learn(self, reward):

        action = np.zeros( self.actionSpace )
        action[self.action] = 1
        action = np.array( [action] ).astype(np.bool)
        nextParam = np.array( [self.nextParam] )
        prevParam = np.array( [self.prevParam] )

        nextQvalues = self.model.predict( [nextParam, np.ones(action.shape)] )
        nextQvalues[action] = reward + self.gamma*np.max(nextQvalues, axis=1)
        self.model.fit([prevParam, action], action*nextQvalues, verbose=0)

    def _createModel(self):
        init = VarianceScaling( scale=2 )
        valueIn = Input((1,), name="value")
        actionsIn = Input((self.actionSpace,), name="mask")

        normalized = Lambda(lambda x: (x-self.low)/self.high)(valueIn)
        hidden1 = Dense(64, activation="relu", kernel_initializer=init)(normalized)
        noise1 = Noise(stddev=1, kernel_initializer=init)(hidden1)
        hidden2 = Dense(64, activation="relu", kernel_initializer=init)(noise1)
        noise2 = Noise(stddev=1, kernel_initializer=init)(hidden2)
        output = Dense(self.actionSpace, kernel_initializer=init)(noise2)

        filteredOut = multiply([output, actionsIn])
        model = Model(inputs=[valueIn, actionsIn], outputs=filteredOut)

        opt = Adam( learning_rate=.0025 )
        model.compile(optimizer=opt, loss="huber_loss")
        return model