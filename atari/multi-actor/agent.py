import numpy as np
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Lambda, LeakyReLU, Concatenate
from tensorflow.keras.optimizers import Adam

from custom.layers import Noise, Whiteout, loadModel
from custom.utils import displayFrames, displayMetric 
from custom.memory import RingBuffer
from custom.welford import Welford

DEFAULT_ACTION_SPACE = 18
DISCOUNT_I = .99
DISCOUNT_E = .997

class Agent:
    def __init__(self, actionSpace, discount_i=DISCOUNT_I, discount_e=DISCOUNT_E):
        self.actionSpace = actionSpace
        self.discount_i = discount_i
        self.discount_e = discount_e

    def _createPolicy(self):
        """
        The policy asks the question, "what action should I take, given the state I'm in?".
        """
        statesIn = Input( (84,84,4) )
        actionsIn = Input( (self.actionSpace,) )

        # normalization
        statesNorm = Lambda( lambda x: x / 255.0 )(statesIn)

        # dual headed network for intrinsic and extrinsic reward streams
        heads = []
        for _ in range(2):
            # shared convolutional layers
            conv = Conv2D(32, 8, strides=4, activation="relu")(statesNorm)
            conv = Conv2D(32, 4, strides=2, activation="relu")(conv)
            for _ in range(5):
                conv = Conv2D(16, 2, strides=1, activation="relu")(conv)
            conv = Flatten()(conv)
            # value of being in the current state
            value = Dense(256, activation="relu")( conv )
            value = Dense(256, activation="relu")( value )
            value = Dense(1)( value )
            # advantage of taking an action in the current state
            advantage = Dense(256, activation="relu")( conv )
            advantage = Dense(256, activation="relu")( advantage )
            advantage = Dense(self.actionSpace)( advantage )
            # value and advantage combined 
            policy = Lambda( lambda x: x[0]+(x[1]-K.mean(x[1])) )( [value, advantage] )
            policy = Lambda( lambda x: x[0]*x[1] )( [policy, actionsIn] )
            heads.append(policy)

        model = Model(inputs=[statesIn, actionsIn], outputs=heads)
        model.compile(Adam( learning_rate=.0001 ), loss="huber_loss")
        return model

    def _createRND(self):
        ### random network distilation
        states = Input( (84,84,1) )
        #whiteout = Whiteout(name="whiteout")(states)
        statesNorm = Lambda( lambda x: x / 255.0 )(states)

        # fixed network
        fixed = Conv2D(16, 8, strides=4, activation="relu", trainable=False)(statesNorm)
        fixed = Conv2D(32, 4, strides=2, activation="relu", trainable=False)(fixed)
        fixed = Conv2D(32, 3, strides=1, activation="relu", trainable=False)(fixed)
        fixed = Flatten()(fixed)
        fixed = Dense(128, trainable=False)(fixed)

        # predictor network
        pred = Conv2D(16, 8, strides=4, activation="relu")(statesNorm)
        pred = Conv2D(32, 4, strides=2, activation="relu")(pred)
        pred = Conv2D(32, 3, strides=1, activation="relu")(pred)
        pred = Flatten()(pred)
        pred = Dense(128)(pred)

        model = Model(inputs=states, outputs=[fixed, pred])
        model.compile(Adam( learning_rate=.0005 ), loss="mse")
        return model

    def _createEmbedding(self):
        ### conrollable embedding network
        startStates = Input( (84,84,1) )
        nextStates = Input( (84,84,1) )

        startStatesNorm = Lambda( lambda x: x / 255.0 )(startStates)
        nextStatesNorm = Lambda( lambda x: x / 255.0 )(nextStates)

        streams = []
        for states in (startStatesNorm, nextStatesNorm):
            conv = Conv2D(16, 8, strides=4, activation="relu")(states)
            conv = Conv2D(32, 4, strides=2, activation="relu")(conv)
            conv = Conv2D(32, 3, strides=1, activation="relu")(conv)
            conv = Flatten()(conv)
            stream = Dense(32, activation="relu")(conv)
            streams.append(stream)

        """
        embedding as the start states stream
        this captures the features that predict the action taken to reach the next state
        """
        embedding = streams[0]

        actionPred = Concatenate()( streams )
        actionPred = Dense(128, activation="relu")(actionPred)
        actionPred = Dense(self.actionSpace, activation="softmax")(actionPred)

        model = Model(inputs=[startStates, nextStates], outputs=[embedding, actionPred])
        model.compile(Adam( learning_rate=.0005 ), loss="categorical_crossentropy")
        return model

class ActorAgent(Agent):
    def __init__(self, game, expChan, weightsChan, actorID, totalActors, oracleScore, 
                 actionSpace=DEFAULT_ACTION_SPACE, render=False, enableGPU=False):
        super().__init__(actionSpace)
        if enableGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        # actor must create its own models, then load weights (if available) from learner
        self.netPolicy = self._createPolicy()
        self.netRND = self._createRND()
        self.netEmbedding = self._createEmbedding()
        self.game = game
        self.expChan = expChan
        self.weightsChan = weightsChan
        self.actorID = actorID
        self.totalActors = totalActors
        self.render = render
        self.oracleScore = oracleScore

        # keeps running statisics on embedding distances
        self.embeddingStats = Welford()
        # keeps running statistics of novetly
        self.noveltyStats = Welford() 

        # hyperparameters
        self.kNeighbors = 10
        self.clusterDistance = .008
        self.maximumSimilarity = 8
        self.modulatorMax = 5
        self.intriniscRewardScale = .3

    def explore(self):
        bestScore = float("-inf")
        maxEpisodeTime = 30*60 # minutes*seconds
        noiseLevel = .4**(1 + 8*(self.actorID/(self.totalActors-1)))
        #noiseLevel = .05

        while True:

            done = False
            s0 = self.game.reset() 
            episodicMemory = RingBuffer(2**15)
            episodeStartTime = time.time()
            noops = np.random.randint(32)

            while not done:
                if time.time()-episodeStartTime >= maxEpisodeTime:
                    break

                Q_i, Q_e = self.predictPolicy( s0 )
                # choose action
                if  noops > 0:
                    a = 0
                    noops -= 1
                elif np.random.random() <= noiseLevel:
                    a = np.random.choice( self.actionSpace )
                else:
                    a = np.argmax( Q_i + Q_e )

                # step
                s1, re, done, info = self.game.step( a )

                # intrinsic reward book keeping
                obs = s1[:,:,3:]
                embedding, _ = self.predictEmbedding(obs, obs)
                episodicMemory.append(embedding[0])
                r_episodic = self.calculateEpisodicReward(episodicMemory.getData(), embedding[0])
                ri = self.calculateIntrinsicReward(obs, r_episodic)

                # calculate temporal difference error
                futureQ_i, futureQ_e = self.predictPolicy( s1 )
                targetQ_i = ri + self.discount_i*np.max(futureQ_i)
                targetQ_e = re + self.discount_e*np.max(futureQ_e)
                tdError = abs( np.max(targetQ_i+targetQ_e) - np.max(Q_i+Q_e) )
                
                # append to memeory
                self.expChan.put( (s0, s1, a, ri, re, tdError, info["life_lost"]) )

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
            netType, weights = self.weightsChan.get_nowait()
            # set respective network weights
            if netType == "policy":
                self.netPolicy.set_weights( weights )
            elif netType == "embedding":
                self.netEmbedding.set_weights( weights )
            elif netType == "rnd":
                self.netRND.set_weights( weights )
        except:
            pass

    def predictPolicy(self, state):
        state = np.array( [state] )
        mask = np.ones( (1, self.actionSpace) )
        return self.netPolicy.predict( [state, mask] )
    
    def predictEmbedding(self, startState, nextState):
        startState = np.array( [startState] )
        nextState = np.array( [nextState] )
        return self.netEmbedding.predict( [startState, nextState] )

    def calculateEpisodicReward(self, episodicMemory, embedding):
        if len(episodicMemory) < self.kNeighbors:
            return 0
        distances = np.sum((episodicMemory-embedding)**2, axis=1)
        indices = np.argsort(distances)[0:self.kNeighbors]
        kDistances = distances[indices]
        for d in kDistances:
            self.embeddingStats.update(d)
        normDistances = kDistances/self.embeddingStats.mean
        clusterDistances = np.where((normDistances-self.clusterDistance) < 0, 0, normDistances)
        kernelValues = 1e-4/(clusterDistances+1e-4)
        similarity = np.sqrt(np.sum(kernelValues)) + 1e-3
        if similarity > self.maximumSimilarity:
            return 0
        else:
            return 1/similarity

    def calculateIntrinsicReward(self, state, r_episodic):
        fixed, pred = self.netRND.predict( np.array([state]) )
        novelty = np.sum( (pred-fixed)**2 )
        self.noveltyStats.update(novelty)
        modulator = 1 + (novelty - self.noveltyStats.mean) / (self.noveltyStats.std + 1e-8)
        rewards_i = r_episodic*np.clip(modulator, 1, self.modulatorMax)
        return self.intriniscRewardScale*rewards_i


class LearnerAgent(Agent):
    def __init__(self, netType, memory, agentName, weightsChan, load,
                 saveFreq, sampleSize, actionSpace, enableGPU):
        super().__init__(actionSpace)
        self.netType = netType
        self.memory = memory
        self.agentName = agentName
        self.weightsChan = weightsChan
        self.load = load
        self.saveFreq = saveFreq
        self.enableGPU = enableGPU
        self.sampleSize = sampleSize

    def _createModel(self):
        # enable or disable model to be created on GPU
        if self.enableGPU:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        if self.netType == "policy":
            model = self._createPolicy()
        if self.netType == "rnd":
            model = self._createRND()
        if self.netType == "embedding":
            model = self._createEmbedding()
        # load model weights from file, if provided
        if self.load is not None:
            temp = loadModel(self.load)
            model.set_weights( temp.get_weights() ) 
        # print the summary of the model created
        model.summary()
        return model

    def _save(self, model):
        print("Saving:", self.netType)
        model.save( self.agentName + '_' + self.netType + ".h5" )

    def _sendWeights(self, weights):
        # send fresh weights in weights chan
        try:
            self.weightsChan.put_nowait( (self.netType, weights) )
        except:
            pass

    def _getActionsMask(self, actions):
        mask = np.zeros((actions.size, self.actionSpace))
        mask = mask.astype(np.bool)
        mask[np.arange(actions.size), actions] = 1
        return mask

class LearnerAgentRND(LearnerAgent):
    netType = "rnd"
    def __init__(self, memory, agentName, weightsChan, 
                load=None, saveFreq=2048, weightsSendFreq=4, sampleSize=64, 
                actionSpace=DEFAULT_ACTION_SPACE, enableGPU=False):
        super().__init__(self.netType, memory, agentName, weightsChan, load,
                         saveFreq, sampleSize, actionSpace, enableGPU)
        self.weightsSendFreq = weightsSendFreq

    def learn(self):
        self.model = self._createModel()
        while len(self.memory) < self.sampleSize:
            continue
        learnIter = 0
        while True:
            _, nextStates, _, _, _, _, _ = self.memory.uniform_sample(self.sampleSize) 
            nextStates = nextStates[:,:,:,3:]
            fixed, _ = self.model.predict( nextStates )
            self.model.fit(nextStates, [fixed, fixed], verbose=0)

            learnIter += 1
            if learnIter % self.saveFreq == 0:
                self._save(self.model)
            if learnIter % self.weightsSendFreq == 0:
                self._sendWeights(self.model.get_weights())


class LearnerAgentEmbedding(LearnerAgent):
    netType = "embedding"
    def __init__(self, memory, agentName, weightsChan, 
                 load=None, saveFreq=2048, weightsSendFreq=4, sampleSize=64, 
                 actionSpace=DEFAULT_ACTION_SPACE, enableGPU=False):
        super().__init__(self.netType, memory, agentName, weightsChan, load,
                         saveFreq, sampleSize, actionSpace, enableGPU)
        self.weightsSendFreq = weightsSendFreq

    def learn(self):
        self.model = self._createModel()
        while len(self.memory) < self.sampleSize:
            continue
        learnIter = 0
        while True:
            startStates, nextStates, actions, _, _, _, _ = self.memory.uniform_sample(self.sampleSize)
            startStates = startStates[:,:,:,3:]
            nextStates = nextStates[:,:,:,3:]
            actions = self._getActionsMask(actions)
            embedding, _ = self.model.predict([startStates, nextStates])
            self.model.fit([startStates, nextStates], [embedding, actions], verbose=0)

            learnIter += 1
            if learnIter % self.saveFreq == 0:
                self._save(self.model)
            if learnIter % self.weightsSendFreq == 0:
                self._sendWeights(self.model.get_weights())

class LearnerAgentPolicy(LearnerAgent):
    netType = "policy"
    def __init__(self, memory, agentName, weightsChan, 
                 load=None, saveFreq=2048, weightsSendFreq=1, sampleSize=64, 
                 actionSpace=DEFAULT_ACTION_SPACE, enableGPU=False, targetUpdateFreq=1500):
        super().__init__(self.netType, memory, agentName, weightsChan, load,
                         saveFreq, sampleSize, actionSpace, enableGPU)
        self.weightsSendFreq = weightsSendFreq
        self.targetUpdateFreq = targetUpdateFreq

    def learn(self):
        self.model = self._createModel()
        self.modelTarget = clone_model(self.model)
        self.modelTarget.set_weights( self.model.get_weights() )  

        learnIter = 0
        while len(self.memory) < self.sampleSize:
            continue
        while True:
            print("Policy learn iter:", learnIter)

            batch, isWeights = self.memory.sample_nolock(self.sampleSize)
            startStates, nextStates, actions, rewards_i, rewards_e, _, _ = batch 
            actions = self._getActionsMask(actions)

            # predict actions with online model, and Q from target model, from such actions (double achitecture)
            onlineFutureQ_i, onlineFutureQ_e = self.model.predict( [nextStates, np.ones(actions.shape)] )
            predictedActions = self._getActionsMask(np.argmax(onlineFutureQ_i+onlineFutureQ_e, axis=1))
            futureQ_i, futureQ_e = self.modelTarget.predict( [nextStates, predictedActions] )

            # set what Q should be, for given actions
            targetQ_i = np.zeros(actions.shape)
            targetQ_i[actions] = rewards_i + self.discount_i*np.max(futureQ_i, axis=1)
            targetQ_e = np.zeros(actions.shape)
            targetQ_e[actions] = rewards_e + self.discount_e*np.max(futureQ_e, axis=1)

            # so if we start in these states, the rewards should look like this
            self.model.fit([startStates, actions], [targetQ_i, targetQ_e], sample_weight=[isWeights, isWeights], verbose=0)

            learnIter += 1
            if learnIter % self.targetUpdateFreq == 0:
                self.modelTarget.set_weights( self.model.get_weights() )  
            if learnIter % self.saveFreq == 0:
                self._save(self.model)
            if learnIter % self.weightsSendFreq == 0:
                self._sendWeights(self.model.get_weights())



