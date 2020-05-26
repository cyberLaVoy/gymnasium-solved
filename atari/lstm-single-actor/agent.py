import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Lambda, LSTM, TimeDistributed, LSTMCell, Reshape
from tensorflow.keras.optimizers import Adam

from custom.layers import Noise, loadModel

class Agent:
    def __init__(self, actionSpace, stateful, modelLoad):
        self.actionSpace = actionSpace

        if modelLoad is None:
            self.model = self._createModel(stateful)
        else:
            self.model = loadModel(modelLoad)

        # print summary of model
        self.model.summary()
   
    def _createModel(self, stateful):
        init = VarianceScaling( scale=2 )

        if stateful:
            framesIn = Input(batch_shape=(1,1,84,84,4), name="frames")
        else:
            framesIn = Input((None,84,84,4), name="frames")
        actionsIn = Input((self.actionSpace,), name="mask")

        ### shared convolutional layers
        normalized = TimeDistributed( Lambda(lambda x: x / 255.0, name="prep0") )(framesIn)
        # interactions between pixels
        conv0 = TimeDistributed( Conv2D(32, 8, strides=4, activation="relu", kernel_initializer=init, name="conv0") )(normalized)
        conv1 = TimeDistributed( Conv2D(64, 4, strides=2, activation="relu", kernel_initializer=init, name="conv1") )(conv0)
        # learned summarization
        conv2 = TimeDistributed( Conv2D(64, 3, strides=1, activation="relu", kernel_initializer=init, name="conv2") )(conv1)
        flattened = TimeDistributed( Flatten() )(conv2)

        # LSTM core
        lstm = LSTM(256, stateful=stateful)(flattened)

        ### dual architecture, where we split advantage and value learning
        # advantage split (advantage of taking action a_i in state s)
        split1 = Dense(512, activation="relu", kernel_initializer=init)(lstm)
        noise1 = Noise(stddev=1, kernel_initializer=init)(split1)
        advantage = Dense(self.actionSpace, kernel_initializer=init)(noise1)
        # value split (value of being in state s)
        split2 = Dense(512, activation="relu", kernel_initializer=init)(lstm)
        noise2 = Noise(stddev=1, kernel_initializer=init)(split2)
        value = Dense(1, kernel_initializer=init)(noise2)
        # advantage/value combined
        policy = Lambda( lambda x: x[0]+(x[1]-K.mean(x[1])) )([value, advantage])

        filtered = Lambda( lambda x: x[0]*x[1] )([policy, actionsIn])
        model = Model(inputs=[framesIn, actionsIn], outputs=filtered)

        opt = Adam( learning_rate=.00025 )
        model.compile(opt, loss="huber_loss")
        return model

    def predict(self, state):
        state = np.array( [state] )
        mask = np.ones( (1, self.actionSpace) )
        return self.model.predict( [state, mask] )

    def setWeights(self, weights):
        self.model.set_weights(weights) 
    def getWeights(self):
        return self.model.get_weights()


class ActorAgent(Agent):
    def __init__(self, actionSpace, modelLoad=None):
        super().__init__(actionSpace, True, modelLoad)

class LearnerAgent(Agent):
    def __init__(self, agentName, actionSpace, modelLoad=None, targetUpdateFreq=2500, gamma=0.997, sampleSize=32, traceSize=16):
        super().__init__(actionSpace, False, modelLoad)
        self.name = agentName
        self.modelName = self.name + ".h5"
        self.gamma = gamma
        self.sampleSize = sampleSize
        self.traceSize = traceSize
        self.learnIter = 0
        self.targetUpdateFreq = targetUpdateFreq

        self.targetNet = clone_model(self.model)
        self._updateTarget()

    def __del__(self):
        self.model.save(self.modelName)
    
    def learn(self, memory):
        batch = memory.sample(self.sampleSize, self.traceSize)

        batch = np.reshape(batch,(self.sampleSize*self.traceSize,5))
        startStates = np.vstack( batch[:,0] ).reshape((self.sampleSize, self.traceSize, 84, 84, 4))
        nextStates = np.vstack( batch[:,1] ).reshape((self.sampleSize, self.traceSize, 84, 84, 4))
        actions = np.vstack(batch[:,2]).astype(np.uint8)
        ind = [i for i in range(self.traceSize-1, self.sampleSize*self.traceSize, self.traceSize)]
        actions = actions[ind].reshape(self.sampleSize,)
        rewards = np.vstack(batch[:,3])
        rew = []
        rew.append(rewards[0:ind[0]].sum())
        for i in range(len(ind)-1):
            re = rewards[ind[i]:ind[i+1]].sum()
            rew.append(re)
        rewards = np.array(rew).reshape(self.sampleSize,)
        #rewards = rewards[ind].reshape(self.sampleSize,)
        isTerminal = np.vstack(batch[:,4])
        isTerminal = isTerminal[ind].reshape(self.sampleSize,)
        actions = self._getActionsMask(actions)


        ### double achitecture, where we predict actions with online model
        onlineQvalues = self.model.predict( [nextStates, np.ones(actions.shape)] )
        predActions = np.argmax(onlineQvalues, axis=1)
        actionsMask = self._getActionsMask(predActions)

        # predict Q values from actions mask
        futureQ = self.targetNet.predict( [nextStates, actionsMask] )
        futureQ = np.max(futureQ, axis=1)
        # set what Q values should be, for given actions
        targetQ = np.zeros(actions.shape)
        targetQ[actions] = rewards + (1-isTerminal)*self.gamma*futureQ
        # so if we start in these states, the rewards should look like this
        self.model.fit([startStates, actions], targetQ, verbose=0)
        
        self.learnIter += 1
        if self.learnIter % self.targetUpdateFreq == 0:
            self._updateTarget()

    def _getActionsMask(self, actions):
        mask = np.zeros((actions.size, self.actionSpace)).astype(np.bool)
        mask[np.arange(actions.size), actions] = 1
        return mask

    def _updateTarget(self):
        self.targetNet.set_weights(self.model.get_weights()) 
