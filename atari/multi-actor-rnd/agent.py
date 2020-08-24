import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Lambda, Concatenate, GaussianNoise
from tensorflow.keras.optimizers import Adam

from custom.layers import Noise, loadModel
from custom.utils import displayFrames, displayMetric
from memory import RingBuffer


class Agent:
    actionSpace = 18
    beta = .25
    def _createPolicy(self):
        startStates = Input( (84,84,4) )
        actionsIn = Input( (self.actionSpace,) )
        startStatesNorm = Lambda( lambda x: x / 255.0 )(startStates)

        heads = []
        for _ in range(2):
            # shared convolutional layers
            conv = Conv2D(32, 8, strides=4, activation="relu")(startStatesNorm)
            conv = Conv2D(32, 4, strides=2, activation="relu")(conv)
            for _ in range(5):
                conv = Conv2D(32, 2, strides=1, activation="relu")(conv)
            conv = Flatten()(conv)
            # value and advantage layers
            value = Dense(256, activation="relu")( conv )
            value = Dense(256, activation="relu")( value )
            value = Dense(1)( value )
            advantage = Dense(256, activation="relu")( conv )
            advantage = Dense(256, activation="relu")( advantage )
            advantage = Dense(self.actionSpace)( advantage )
            h = Lambda( lambda x: x[0]+(x[1]-K.mean(x[1])) )( [value, advantage] )
            h = Lambda( lambda x: x[0]*x[1] )( [h, actionsIn] )
            heads.append(h)

        model = Model(inputs=[startStates, actionsIn], outputs=heads)
        model.compile(Adam( learning_rate=.0001 ), loss="huber_loss")
        return model

    def _createRND(self):
        states = Input( (84,84,1) )
        norm = Lambda( lambda x: x / 255.0 )(states)
        ### random network distilation
        # fixed network
        fixed = Conv2D(8, 8, strides=4, bias_initializer='random_normal', trainable=False)(norm)
        fixed = Conv2D(16, 4, strides=2, bias_initializer='random_normal', trainable=False)(fixed)
        fixed = Conv2D(32, 3, strides=2, bias_initializer='random_normal', trainable=False)(fixed)
        fixed = Flatten()(fixed)
        fixed = Dense(128, bias_initializer='random_normal', trainable=False)(fixed)
        # distribution network
        dist = Conv2D(8, 8, strides=4)(norm)
        dist = Conv2D(16, 4, strides=2)(dist)
        dist = Conv2D(32, 3, strides=2)(dist)
        dist = Flatten()(dist)
        dist = Dense(128)(dist)

        model = Model(inputs=states, outputs=[fixed, dist])
        model.compile(Adam( learning_rate=.0005 ), loss="huber_loss")
        return model

class ActorAgent(Agent):
    def __init__(self, game, memPolicy, memRND, learnerChan, actorID, oracleScore, render, enableGPU=False):
        if enableGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # actor must create its own model, then load weights from learner
        self.policy = self._createPolicy()
        self.netRND = self._createRND()
        self.game = game
        self.memPolicy = memPolicy
        self.memRND = memRND
        self.learnerChan = learnerChan
        self.actorID = actorID
        self.render = render
        self.oracleScore = oracleScore

    def explore(self):
        if self.render:
            infoStream = RingBuffer(256)

        novMean = 0
        novSquareSum = 0
        novResetItters = 128
        novEngageItters = 256
        bestScore = float("-inf")
        itter = 0
        updates = 0
        noiseLevel = .01
        transitionDecay = .8
        maxEpisodeTime = 30*60 # minutes*seconds

        while True:


            done = False
            s0 = self.game.reset() 
            noops = np.random.choice( 32 )
            episodeStartTime = time.time()
            display = np.zeros( (84,84,1) )

            while not done:
                if time.time()-episodeStartTime >= maxEpisodeTime:
                    break
                itter += 1

                # choose action
                if noops > 0:
                    a = 0
                    noops -= 1
                elif np.random.random() <= noiseLevel:
                    a = np.random.choice( self.actionSpace )
                else:
                    Q_e, Q_i = self.predictPolicy( s0 )
                    a = np.argmax( Q_e + self.beta*Q_i )

                # step
                s1, re, done, info = self.game.step(a)

                # append to rnd memory
                newFrame = self.game.getFrameChange()
                display = np.clip( (display+newFrame), 0, 255)
                display = np.clip( display-transitionDecay*display.mean(), 0, 255)
                self.memRND.append( display )

                # reward engineering (intrinsic)
                fixed, dist = self.predictRND( display )
                novelty = np.sqrt(np.sum((dist-fixed)**2))
                prevMean = novMean
                novMean += (novelty-novMean)/itter
                novSquareSum += (novelty-novMean)*(novelty-prevMean)
                novStdDev = np.sqrt(novSquareSum/itter)
                if updates == novResetItters:
                    novMean = 0
                    novSquareSum = 0
                if updates >= novEngageItters:
                    # how novel is this state?
                    ri = np.tanh( (novelty-novMean)/(novStdDev+.0001) )
                else:
                    ri = 0

                # append to policy memeory
                self.memPolicy.append( (s0, s1, a, re, ri, info["life_lost"]) )

                # display info
                if self.render:
                    self.game.render()
                    displayFrames(display/255.0)
                    infoStream.append( ri+re )
                    if itter % 16 == 0:
                        displayMetric(infoStream.data, "Info Stream")

                # step upkeep
                if self._updateWeights():
                    updates += 1
                s0 = s1

            # record game score, if beter than best so far
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
        weightsRND = None
        while not self.learnerChan.empty():
            weights = self.learnerChan.get()
            if weights[0] == "policy":
                weightsPolicy = weights[1]
            if weights[0] == "rnd":
                weightsRND = weights[1]
        if weightsPolicy is not None:
            self.policy.set_weights( weightsPolicy )
            update = True
        if weightsRND is not None:
            self.netRND.set_weights( weightsRND )
            update = True
        return update

    def predictPolicy(self, state):
        state = np.array( [state] )
        mask = np.ones( (1, self.actionSpace) )
        return self.policy.predict( [state, mask] )

    def predictRND(self, state):
        state = np.array( [state] )
        return self.netRND.predict( state )



class LearnerAgent(Agent):
    def __init__(self, memory, agentName, actorsChan, netType,
                 load, saveFreq, 
                 actorUpdateFreq, sampleSize,
                 enableGPU):
        if enableGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        self.netType = netType
        if netType == "policy":
            self.model = self._createPolicy()
        if netType == "rnd":
            self.model = self._createRND()
        self.load = load
        if load is not None:
            temp = loadModel(load)
            self.model.set_weights(temp.get_weights()) 

        self.name = agentName + '_' + netType + ".h5"
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
            chan.put((self.netType, self.model.get_weights()))


class LearnerAgentPolicy(LearnerAgent):
    netType = "policy"
    def __init__(self, memory, agentName, actorsChan,
                 load, saveFreq=2000, actorUpdateFreq=4, sampleSize=64, enableGPU=False,  
                 gamma_e=0.997, gamma_i=.99, targetUpdateFreq=1500):
        super().__init__(memory, agentName, actorsChan, self.netType,
                 load, saveFreq, actorUpdateFreq, sampleSize, enableGPU)

        self.gamma_e = gamma_e
        self.gamma_i = gamma_i
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
        startStates, nextStates, actions, rewards_e, rewards_i, isTerminal = batch 
        actions = self._getActionsMask(actions)

        ### double achitecture, where we predict actions with online model
        onlineQ_e, onlineQ_i = self.model.predict( [nextStates, np.ones(actions.shape)] )
        onlineQ = onlineQ_e + self.beta*onlineQ_i
        actionsMask = self._getActionsMask( np.argmax(onlineQ, axis=1) )
        # predict Q from actions mask
        futureQ_e, futureQ_i = self.targetNet.predict( [nextStates, actionsMask] )
        # set what Q should be, for given actions
        targetQ_e = np.zeros(actions.shape)
        targetQ_e[actions] = rewards_e + (1-isTerminal)*self.gamma_e*np.max(futureQ_e, axis=1)
                                                                    # soft Q-learning (maximizing entropy)
        #targetQ_e[actions] = rewards_e + (1-isTerminal)*self.gamma_e*np.log(np.trapz(np.exp(futureQ_e)))
        targetQ_i = np.zeros(actions.shape)
        targetQ_i[actions] = rewards_i + (1-isTerminal)*self.gamma_i*np.max(futureQ_i, axis=1)
                                                                    # soft Q-learning (maximizing entropy)
        #targetQ_i[actions] = rewards_i + (1-isTerminal)*self.gamma_i*np.log(np.trapz(np.exp(futureQ_i)))
        targetQ = targetQ_e + self.beta*targetQ_i
        # so if we start in these states, the rewards should look like this
        self.model.fit([startStates, actions], [targetQ_e, targetQ_i], sample_weight=[isWeights, isWeights], verbose=0)

        tdError = abs( np.max(targetQ, axis=1) - np.max(onlineQ, axis=1) )
        self.memory.updatePriorities(indices, tdError)           

    def _getActionsMask(self, actions):
        mask = np.zeros((actions.size, self.actionSpace)).astype(np.bool)
        mask[np.arange(actions.size), actions] = 1
        return mask

    def _updateTarget(self):
        self.targetNet.set_weights(self.model.get_weights()) 


class LearnerAgentRND(LearnerAgent):
    netType = "rnd"
    def __init__(self, memory, agentName, actorsChan, load, 
                 saveFreq=2000, actorUpdateFreq=4, sampleSize=64, enableGPU=False):
        super().__init__(memory, agentName, actorsChan, self.netType,
                 load, saveFreq, actorUpdateFreq, sampleSize, enableGPU)

    def learn(self):
        learnIter = 0
        while True:
            self.memory.load()
            if len(self.memory) < self.sampleSize:
                continue

            self.learnRND()

            learnIter += 1
            if learnIter % self.actorUpdateFreq == 0:
                self._updateActors()
            if learnIter % self.saveFreq == 0:
                self.save()

    def learnRND(self):
        states = self.memory.sample( self.sampleSize )
        states = np.array( states )
        fixed, _ = self.model.predict( states )
        self.model.fit(states, [fixed, fixed], verbose=0)

