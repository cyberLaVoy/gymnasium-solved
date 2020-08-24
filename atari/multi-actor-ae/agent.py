import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Lambda, Conv2DTranspose, Reshape, Add, Multiply, GaussianNoise, Concatenate, ReLU
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.losses import BinaryCrossentropy

from custom.layers import Noise, loadModel, KLDivergenceLayer, Entropy, MutualInformation
from custom.utils import displayFrames, displayMetric
from custom.memory import RingBuffer


class Agent:
    latent_dim = 128
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def _createPolicy(self):
        startStates = Input( (self.latent_dim,) )
        actionsIn = Input( (self.actionSpace,) )

        extracted = Flatten()(startStates)

        # value
        value = Dense(512, activation="relu")( extracted )
        value = Dense(1)( value )
        # advantage
        advantage = Dense(512, activation="relu")( extracted )
        advantage = Dense(self.actionSpace)( advantage )
        policy = Lambda( lambda x: x[0]+(x[1]-K.mean(x[1])) )( [value, advantage] )
        policy = Lambda( lambda x: x[0]*x[1] )( [policy, actionsIn] )

        model = Model(inputs=[startStates, actionsIn], outputs=policy)
        model.compile(Adam( learning_rate=.00025 ), loss="huber_loss")
        model.summary()
        return model

    def _createAEencode(self):
        states = Input( (84, 84, 4) )
        norm = Lambda( lambda x: x / 255.0, name="prep0")(states)

        ### encoding
        encode = Conv2D(8, 8, strides=4, activation="relu", name="encode1")(norm) 
        encode = Conv2D(16, 4, strides=2, activation="relu", name="encode2")(encode) 
        encode = Conv2D(32, 3, strides=2, activation="relu", name="encode3")(encode) 
        encode = Flatten()(encode)
        encode = Dense(self.latent_dim, name="encode4")(encode)
        encode = ReLU(max_value=1.0)(encode)
        # disentanglement factor
        encode = Entropy(scale=.001)(encode)

        # action input
        actions = Input((self.actionSpace,))
        return states, encode, actions

    def _createAEdecode(self, encode, actions):
        ### decoding
        decode = Reshape( (4, 4, self.latent_dim//(4*4)) )(encode)
        decode = Conv2DTranspose(32, 3, strides=2, activation="relu")(decode)
        decode = Conv2DTranspose(16, 4, strides=2, activation="relu")(decode)
        decode = Conv2DTranspose(8, 8, strides=4, activation="relu")(decode)

        decode_dynamics = Conv2D(1, 2, strides=1, padding="same", use_bias=False)(decode) 
        decode_dynamics = ReLU(max_value=1.0)(decode_dynamics)

        # action prediction
        constant = Lambda( lambda x: 1+0*x)(actions)
        action_prediction = Dense(self.actionSpace)(constant)
        action_prediction = ReLU(max_value=1.0)(action_prediction)
        actionProb = Lambda( lambda x: np.max(x[0]*x[1], axis=-1) )( [action_prediction, actions] )

        # empowerment
        conditional = Multiply()( [decode_dynamics, actionProb])
        inverse = Lambda( lambda x: 1 / (x+.0001))(actionProb)
        conditional = Multiply()( [conditional, inverse])
        decode_dynamics, conditional = MutualInformation(scale=.001)([decode_dynamics, conditional]) 

        return decode_dynamics, action_prediction

    def _createAE(self):
        states, encode, actions = self._createAEencode()
        decode_dynamics, action_prediction = self._createAEdecode(encode, actions)
        model = Model(inputs=[states, actions], outputs=[decode_dynamics, action_prediction]) 
        model.compile( optimizer=Adam( learning_rate=.0005 ), loss="mse" )
        model.summary()
        return model

class ActorAgent(Agent):
    def __init__(self, game, actionSpace, memPolicy, memAE, learnerChan, actorID, totalActors, oracleScore, render, enableGPU=False):
        super().__init__(actionSpace)
        if enableGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # actor must create its own model, then load weights from learner
        self.policy = self._createPolicy()
        states, encode, actions = self._createAEencode()
        self.netAEencode = Model(inputs=[states, actions], outputs=encode)
        self.game = game
        self.memPolicy = memPolicy
        self.memAE = memAE
        self.learnerChan = learnerChan
        self.actorID = actorID
        self.totalActors = totalActors
        self.render = render
        self.oracleScore = oracleScore

    def explore(self):
        
        bestScore = float("-inf")
        maxEpisodeTime = 30*60 # minutes*seconds
        noiseLevel = max( .4**(1 + 8*(self.actorID/(self.totalActors-1))), .05)

        while True:

            done = False
            s0 = self.game.reset() 
            noops = np.random.choice( 32 )
            episodeStartTime = time.time()
            s0encoded = self.predictAE(s0)

            while not done:
                if time.time()-episodeStartTime >= maxEpisodeTime:
                    break


                # choose action
                Q = self.predictPolicy( s0encoded )
                if noops > 0:
                    a = 0
                    noops -= 1
                elif np.random.random() <= noiseLevel:
                    a = np.random.choice( self.actionSpace )
                else:
                    a = np.argmax( Q )

                # step
                s1, r, done, info = self.game.step(a)

                # append to ae memory
                #stateChange = self.game.getStateChange()
                self.memAE.append( (s0, s1[:,:,3:], a) )
                s1encoded = self.predictAE(s0)

                # append to policy memeory
                self.memPolicy.append( (s0encoded, s1encoded, a, r, info["life_lost"]) )

                # display info
                if self.render:
                    self.game.render()

                # step upkeep
                self._updateWeights()
                s0 = s1
                s0encoded = s1encoded

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
        weightsAE = None
        while not self.learnerChan.empty():
            weights = self.learnerChan.get()
            if weights[0] == "policy":
                weightsPolicy = weights[1]
            if weights[0] == "ae":
                weightsAE = weights[1]
        if weightsPolicy is not None:
            self.policy.set_weights( weightsPolicy )
            update = True
        if weightsAE is not None:
            self.netAEencode.set_weights( weightsAE )
            update = True
        return update

    def predictPolicy(self, state):
        state = np.array( [state] )
        mask = np.ones( (1, self.actionSpace) )
        return self.policy.predict( [state, mask] )

    def predictAE(self, state):
        state = np.array( [state] )
        action = np.zeros( (1, self.actionSpace) )
        latent = self.netAEencode.predict( [state, action] )
        return latent[0]



class LearnerAgent(Agent):
    def __init__(self, actionSpace, memory, agentName, actorsChan, netType,
                 load, saveFreq, 
                 actorUpdateFreq, sampleSize,
                 enableGPU):
        super().__init__(actionSpace)
        if enableGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        self.netType = netType
        if netType == "policy":
            self.model = self._createPolicy()
        if netType == "ae":
            self.model = self._createAE()
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
            if self.netType == "policy":
                weights = self.model.get_weights()
                chan.put( (self.netType, weights) )
            if self.netType == "ae":
                layers = [layer for layer in self.model.layers if "encode" in layer.name]
                weights = []
                for layer in layers:
                    weights.extend( layer.get_weights() )
                chan.put( (self.netType, weights) )

    def _getActionsMask(self, actions):
        mask = np.zeros((actions.size, self.actionSpace)).astype(np.bool)
        mask[np.arange(actions.size), actions] = 1
        return mask

class LearnerAgentPolicy(LearnerAgent):
    netType = "policy"
    def __init__(self, actionSpace, memory, agentName, actorsChan,
                 load, saveFreq=2048, actorUpdateFreq=4, sampleSize=64, enableGPU=False,  
                 gamma=0.997, targetUpdateFreq=1500):
        super().__init__(actionSpace, memory, agentName, actorsChan, self.netType,
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
                                                              # soft Q-learning (maximizing entropy, until certain)
        targetQ[actions] = rewards + (1-isTerminal)*self.gamma*np.log(np.trapz(np.exp(futureQ)))
        # so if we start in these states, the rewards should look like this
        self.model.fit([startStates, actions], targetQ, sample_weight=isWeights, verbose=0)

        tdError = abs( np.max(targetQ, axis=1) - np.max(onlineQ, axis=1) )
        self.memory.updatePriorities(indices, tdError)           

    def _updateTarget(self):
        self.targetNet.set_weights(self.model.get_weights()) 


class LearnerAgentAE(LearnerAgent):
    netType = "ae"
    def __init__(self, actionSpace, memory, agentName, actorsChan, load, 
                 saveFreq=2048, actorUpdateFreq=4, sampleSize=8, enableGPU=False):
        super().__init__(actionSpace, memory, agentName, actorsChan, self.netType,
                 load, saveFreq, actorUpdateFreq, sampleSize, enableGPU)

    def learn(self):
        learnIter = 0
        while True:
            self.memory.load()
            if len(self.memory) < self.sampleSize:
                continue

            self.learnAE()

            learnIter += 1
            if learnIter % self.actorUpdateFreq == 0:
                self._updateActors()
            if learnIter % self.saveFreq == 0:
                self.save()

    def learnAE(self):
        indices, batch, isWeights = self.memory.sample(self.sampleSize)
        startStates, nextStates, actions = batch
        startStates = np.array( startStates )
        nextStates = np.array( nextStates )
        actionsMask = self._getActionsMask(actions)

        onlineDynamics, _ = self.model.predict( [startStates, actionsMask] )
        targetDynamics = nextStates / 255.0

        display = onlineDynamics[0]
        displayFrames( display )
        #displayFrames( (display-display.min())/(display.max()-display.min()) )

        self.model.fit([startStates, actionsMask], 
                       [targetDynamics, actionsMask], 
                       sample_weight=[isWeights, isWeights], 
                       verbose=0)

        # dynamics
        dynamicsError = (targetDynamics-onlineDynamics)**2
        dynamicsError = dynamicsError.reshape(-1, dynamicsError.shape[-1]).sum(axis=0)
        dynamicsError = np.sqrt( dynamicsError )
        # temporal difference error
        tdError = dynamicsError
        self.memory.updatePriorities(indices, tdError)           



