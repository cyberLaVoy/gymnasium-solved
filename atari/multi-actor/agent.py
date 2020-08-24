import numpy as np
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Lambda
from tensorflow.keras.optimizers import Adam

from custom.layers import Noise, loadModel


class Agent:
    actionSpace = 18
    def _createPolicy(self):
        startStates = Input( (84,84,4) )
        actionsIn = Input( (self.actionSpace,) )

        # normalization
        startStatesNorm = Lambda( lambda x: x / 255.0 )(startStates)

        # shared convolutional layers
        conv = Conv2D(32, 8, strides=4, activation="relu")(startStatesNorm)
        conv = Conv2D(32, 4, strides=2, activation="relu")(conv)
        conv = Conv2D(64, 3, strides=2, activation="relu")(conv)
        conv = Flatten()(conv)
        # value and advantage layers
        value = Dense(512, activation="relu")( conv )
        value = Noise()( value )
        value = Dense(1)( value )
        advantage = Dense(512, activation="relu")( conv )
        advantage = Noise()( advantage )
        advantage = Dense(self.actionSpace)( advantage )
        policy = Lambda( lambda x: x[0]+(x[1]-K.mean(x[1])) )( [value, advantage] )
        policy = Lambda( lambda x: x[0]*x[1] )( [policy, actionsIn] )

        model = Model(inputs=[startStates, actionsIn], outputs=policy)
        model.compile(Adam( learning_rate=.0001 ), loss="huber_loss")
        return model

class ActorAgent(Agent):
    def __init__(self, game, memPolicy, learnerChan, actorID, oracleScore, render, enableGPU=False):
        if enableGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # actor must create its own model, then load weights from learner
        self.policy = self._createPolicy()
        self.game = game
        self.memPolicy = memPolicy
        self.learnerChan = learnerChan
        self.actorID = actorID
        self.render = render
        self.oracleScore = oracleScore

    def explore(self):

        bestScore = float("-inf")
        noiseLevel = .05
        maxEpisodeTime = 30*60 # minutes*seconds

        while True:

            done = False
            s0 = self.game.reset() 
            noops = np.random.choice( 32 )
            episodeStartTime = time.time()

            while not done:
                if time.time()-episodeStartTime >= maxEpisodeTime:
                    break

                # choose action
                if noops > 0:
                    a = 0
                    noops -= 1
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
                self._updateWeights()
                s0 = s1

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
        update = False
        weightsPolicy = None
        while not self.learnerChan.empty():
            weights = self.learnerChan.get()
            weightsPolicy = weights
        if weightsPolicy is not None:
            self.policy.set_weights( weightsPolicy )
            update = True
        return update

    def predictPolicy(self, state):
        state = np.array( [state] )
        mask = np.ones( (1, self.actionSpace) )
        return self.policy.predict( [state, mask] )



class LearnerAgent(Agent):
    def __init__(self, memory, agentName, actorsChan,
                 load, saveFreq, 
                 actorUpdateFreq, sampleSize,
                 enableGPU):
        if enableGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        self.model = self._createPolicy()
        self.load = load
        if load is not None:
            temp = loadModel(load)
            self.model.set_weights( temp.get_weights() ) 

        self.name = agentName + ".h5"
        self.memory = memory

        self.actorUpdateFreq = actorUpdateFreq
        self.actorsChan = actorsChan
        self.sampleSize = sampleSize
        self.saveFreq = saveFreq

    def save(self):
        print("Saving model:", self.name)
        self.model.save( self.name )
    def _updateActors(self):
        for chan in self.actorsChan:
            chan.put(self.model.get_weights())


class LearnerAgentPolicy(LearnerAgent):
    def __init__(self, memory, agentName, actorsChan,
                 load, saveFreq=2000, actorUpdateFreq=4, sampleSize=64, enableGPU=False,  
                 gamma=0.997, targetUpdateFreq=1500):
        super().__init__(memory, agentName, actorsChan,
                 load, saveFreq, actorUpdateFreq, sampleSize, enableGPU)

        self.gamma = gamma
        self.targetUpdateFreq = targetUpdateFreq
        self.targetNet = clone_model(self.model)
        self._updateTarget()

    def learn(self):
        learnIter = 0
        while True:
            self.memory.load()
            if len(self.memory) < self.sampleSize:
                continue

            self.learnPolicy()

            learnIter += 1
            if learnIter % self.targetUpdateFreq == 0 or (learnIter < self.targetUpdateFreq and self.load is None):
                self._updateTarget()
            if learnIter % self.actorUpdateFreq == 0:
                self._updateActors()
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
        #targetQ[actions] = rewards + (1-isTerminal)*self.gamma*np.max(futureQ, axis=1)
                                                                    # soft Q-learning (maximizing entropy)
        targetQ[actions] = rewards + (1-isTerminal)*self.gamma*np.log(np.trapz(np.exp(futureQ)))
        # so if we start in these states, the rewards should look like this
        self.model.fit([startStates, actions], targetQ, sample_weight=isWeights, verbose=0)

        tdError = abs( np.max(targetQ, axis=1) - np.max(onlineQ, axis=1) )
        self.memory.updatePriorities(indices, tdError)           

    def _getActionsMask(self, actions):
        mask = np.zeros((actions.size, self.actionSpace)).astype(np.bool)
        mask[np.arange(actions.size), actions] = 1
        return mask

    def _updateTarget(self):
        self.targetNet.set_weights(self.model.get_weights()) 
