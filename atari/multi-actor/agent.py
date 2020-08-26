import numpy as np
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Lambda
from tensorflow.keras.optimizers import Adam

from custom.layers import Noise, loadModel

DEFAULT_ACTION_SPACE = 18

class Agent:
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def _createPolicy(self):
        startStates = Input( (84,84,4) )
        actionsIn = Input( (self.actionSpace,) )

        # normalization
        startStatesNorm = Lambda( lambda x: x / 255.0 )(startStates)

        # shared convolutional layers
        conv = Conv2D(32, 8, strides=4, activation="relu")(startStatesNorm)
        conv = Conv2D(32, 4, strides=2, activation="relu")(conv)
        for _ in range(5):
            conv = Conv2D(16, 2, strides=1, activation="relu")(conv)
        conv = Flatten()(conv)
        # value and advantage layers
        value = Dense(256, activation="relu")( conv )
        value = Dense(256, activation="relu")( value )
        value = Noise()( value )
        value = Dense(1)( value )
        advantage = Dense(256, activation="relu")( conv )
        advantage = Dense(256, activation="relu")( advantage )
        advantage = Noise()( advantage )
        advantage = Dense(self.actionSpace)( advantage )
        policy = Lambda( lambda x: x[0]+(x[1]-K.mean(x[1])) )( [value, advantage] )
        policy = Lambda( lambda x: x[0]*x[1] )( [policy, actionsIn] )

        model = Model(inputs=[startStates, actionsIn], outputs=policy)
        model.compile(Adam( learning_rate=.00025 ), loss="huber_loss")
        return model

class ActorAgent(Agent):
    def __init__(self, game, memPolicy, weightsChan, actorID, totalActors, oracleScore, 
                 actionSpace=DEFAULT_ACTION_SPACE, render=False, enableGPU=False):
        super().__init__(actionSpace)
        if enableGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # actor must create its own model, then load weights from learner
        self.policy = self._createPolicy()
        self.game = game
        self.memPolicy = memPolicy
        self.weightsChan = weightsChan
        self.actorID = actorID
        self.totalActors = totalActors
        self.render = render
        self.oracleScore = oracleScore

    def explore(self):

        bestScore = float("-inf")
        maxEpisodeTime = 30*60 # minutes*seconds
        noiseLevel = .4**(1 + 8*(self.actorID/(self.totalActors-1)))

        while True:

            done = False
            s0 = self.game.reset() 
            episodeStartTime = time.time()
            info = {"life_lost":True}

            while not done:
                if time.time()-episodeStartTime >= maxEpisodeTime:
                    break

                # choose action
                if info["life_lost"]:
                    a = 1
                elif np.random.random() <= noiseLevel:
                    a = np.random.choice( self.actionSpace )
                else:
                    Q = self.predictPolicy( s0 )
                    a = np.argmax( Q )

                # step
                s1, r, done, info = self.game.step(a)

                # append to policy memeory
                self.memPolicy.append( (s0, s1, a, r, info["life_lost"]) )

                # step upkeep
                s0 = s1
                self._updateWeights()

                # render
                if self.render:
                    self.game.render()

            # record game score, if better than best so far
            if self.game.getScore() > bestScore:
                bestScore = self.game.getScore()
                print("Score:", bestScore, "from actor", self.actorID)
                if bestScore >= self.oracleScore.value and bestScore > 0:
                    self.oracleScore.value = bestScore
                    self.game.saveEpisode()

        self.game.close()


    def _updateWeights(self):
        try:
            # retrieve weights (if available) from chan
            weights = self.weightsChan.get_nowait()
            # set policy weights
            self.policy.set_weights( weights )
        except:
            pass

    def predictPolicy(self, state):
        state = np.array( [state] )
        mask = np.ones( (1, self.actionSpace) )
        return self.policy.predict( [state, mask] )



class LearnerAgent(Agent):
    def __init__(self, memory, agentName, weightsChan,
                 load, saveFreq, sampleSize,
                 actionSpace, enableGPU):
        super().__init__(actionSpace)
        if enableGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        self.model = self._createPolicy()
        self.model.summary()
        self.load = load
        if load is not None:
            temp = loadModel(load)
            self.model.set_weights( temp.get_weights() ) 

        self.name = agentName + ".h5"
        self.memory = memory

        self.weightsChan = weightsChan
        self.sampleSize = sampleSize
        self.saveFreq = saveFreq

    def save(self):
        print("Saving model:", self.name)
        self.model.save( self.name )
    def _sendWeights(self):
        # send fresh weights in weights chan
        try:
            self.weightsChan.put_nowait( self.model.get_weights() )
        except:
            pass


class LearnerAgentPolicy(LearnerAgent):
    def __init__(self, memory, agentName, weightsChan,
                 load, saveFreq=2048, sampleSize=64, 
                 actionSpace=DEFAULT_ACTION_SPACE,
                 enableGPU=False, gamma=0.99, targetUpdateFreq=1500, memLoadFreq=4):
        super().__init__(memory, agentName, weightsChan,
                         load, saveFreq, 
                         sampleSize, actionSpace, enableGPU)
        self.gamma = gamma
        self.memLoadFreq = memLoadFreq
        self.targetUpdateFreq = targetUpdateFreq
        self.targetNet = clone_model(self.model)
        self._updateTarget()

    def learn(self):
        learnIter = 0
        while True:
            if learnIter % self.memLoadFreq == 0:
                self.memory.load()
            if len(self.memory) < self.sampleSize:
                continue
            #print("Learn iter:", learnIter)

            self.learnPolicy()
            self._sendWeights()

            learnIter += 1
            if learnIter % self.targetUpdateFreq == 0 or (learnIter < self.targetUpdateFreq and self.load is None):
                self._updateTarget()
            if learnIter % self.saveFreq == 0:
                self.save()


    def learnPolicy(self):
        indices, batch, isWeights = self.memory.sample(self.sampleSize)
        startStates, nextStates, actions, rewards, isTerminal = batch 
        actions = self._getActionsMask(actions)

        ### double achitecture, where we predict actions with online model
        onlineQ = self.model.predict( [nextStates, np.ones(actions.shape)] )
        actionsMask = self._getActionsMask( np.argmax(onlineQ, axis=1) )
        # predict Q from actions mask
        futureQ = self.targetNet.predict( [nextStates, actionsMask] )
        # set what Q should be, for given actions
        targetQ = np.zeros(actions.shape)

        targetQ[actions] = rewards + (1-isTerminal)*self.gamma*np.max(futureQ, axis=1)
        ### soft Q-learning (maximizing entropy) *experiment
        """
        targetQ[actions] = rewards + (1-isTerminal)*self.gamma*np.log(np.trapz(np.exp(futureQ)))
        """

        # so if we start in these states, the rewards should look like this
        self.model.fit([startStates, actions], targetQ, sample_weight=isWeights, verbose=0)

        # update memory priorities with temporal difference error
        tdError = abs( np.max(targetQ, axis=1) - np.max(onlineQ, axis=1) )
        self.memory.updatePriorities(indices, tdError)           

    def _getActionsMask(self, actions):
        mask = np.zeros((actions.size, self.actionSpace)).astype(np.bool)
        mask[np.arange(actions.size), actions] = 1
        return mask

    def _updateTarget(self):
        self.targetNet.set_weights(self.model.get_weights()) 
