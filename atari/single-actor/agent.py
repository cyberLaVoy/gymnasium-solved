import numpy as np
import tensorflow as tf
import os, random
import cv2
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model, clone_model
import tensorflow.keras.initializers as initializers
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Lambda, multiply, Layer, AveragePooling2D
from tensorflow.keras.optimizers import Adam

def loadModel(fileName):
    # any further custom objects must be added to custom_object
    return load_model(fileName, custom_objects={'Noise':Noise})

class Agent:
    def __init__(self, agentName, numActions, modelLoad=None, targetUpdateFreq=2500, gamma=0.99, sampleSize=32, dual=True, double=True, attentionView=True):
        self.dual = dual
        self.double = double
        self.numActions = numActions
        self.gamma = gamma
        self.sampleSize = sampleSize
        self.name = agentName
        self.learnIter = 0
        self.targetUpdateFreq = targetUpdateFreq

        self.modelName = self.name + ".h5"
        if modelLoad is None:
            self.model = self._createModel()
        else:
            self.model = loadModel(modelLoad)

        self.targetNet = clone_model(self.model)
        self._updateTarget()

        self.attentionView = attentionView
        self._createAttention()
        
        # print summary of model
        self.model.summary()

    def __del__(self):
        self.model.save(self.modelName)
    
    def _createModel(self):
        init = VarianceScaling( scale=2 )
        framesIn = Input((84,84,4), name="frames")
        actionsIn = Input((self.numActions,), name="mask")

        ### shared convolutional layers
        normalized = Lambda(lambda x: x / 255.0, name="prep0")(framesIn)
        # interactions between pixels
        conv = Conv2D(32, 8, strides=4, activation="relu", kernel_initializer=init, name="conv0")(normalized)
        conv = Conv2D(32, 4, strides=2, activation="relu", kernel_initializer=init, name="conv1")(conv)
        conv = Conv2D(32, 3, strides=2, activation="relu", kernel_initializer=init, name="conv2")(conv)
        flattened = Flatten()(conv)

        if self.dual:
            ### dual architecture, where we split advantage and value learning
            # advantage split (advantage of taking action a_i in state s)
            split1 = Dense(512, activation="relu", kernel_initializer=init)(flattened)
            noise1 = Noise(stddev=1, kernel_initializer=init)(split1)
            advantage = Dense(self.numActions, kernel_initializer=init)(noise1)
            # value split (value of being in state s)
            split2 = Dense(512, activation="relu", kernel_initializer=init)(flattened)
            noise2 = Noise(stddev=1, kernel_initializer=init)(split2)
            value = Dense(1, kernel_initializer=init)(noise2)
            # advantage/value combined
            policy = Lambda( lambda x: x[0]+(x[1]-K.mean(x[1])) )([value, advantage])
        else:
            hidden = Dense(512, activation="relu", kernel_initializer=init)(flattened)
            noise = Noise(stddev=1, kernel_initializer=init)(hidden)
            policy = Dense(self.numActions, kernel_initializer=init)(noise)

        filtered = Lambda( lambda x: x[0]*x[1] )([policy, actionsIn])
        model = Model(inputs=[framesIn, actionsIn], outputs=filtered)

        #opt = Adam( learning_rate=.00025 )
        opt = Adam( learning_rate=.0001 )
        model.compile(opt, loss="huber_loss")
        return model

    def predict(self, state):
        state = np.array( [state] )
        mask = np.ones( (1, self.numActions) )
        return self.model.predict( [state, mask] )

    def getAction(self, state):
        return np.argmax( self.predict(state) ) 

    def _getActionsMask(self, actions):
        mask = np.zeros((actions.size, self.numActions)).astype(np.bool)
        mask[np.arange(actions.size), actions] = 1
        return mask

    def learn(self, memory):
        indices, batch, isWeights = memory.sample(self.sampleSize) 
        startStates, nextStates, actions, rewards, isTerminal = batch 
        actions = self._getActionsMask(actions)

        # unused if DDQN and PER are both disabled
        onlineQvalues = self.model.predict( [nextStates, np.ones(actions.shape)] )
        if self.double:
            ### double achitecture, where we predict actions with online model
            predActions = np.argmax(onlineQvalues, axis=1)
            actionsMask = self._getActionsMask(predActions)
        else:
            actionsMask = np.ones(actions.shape)

        # predict Q values from actions mask
        futureQ = self.targetNet.predict( [nextStates, actionsMask] )
        futureQ = np.max(futureQ, axis=1)
        # set what Q values should be, for given actions
        targetQ = np.zeros(actions.shape)
        targetQ[actions] = rewards + (1-isTerminal)*self.gamma*futureQ
        # so if we start in these states, the rewards should look like this
        self.model.fit([startStates, actions], targetQ, sample_weight=isWeights, verbose=0)
        
        self.learnIter += 1
        if self.learnIter % self.targetUpdateFreq == 0:
            self._updateTarget()
            self._updateAttention()

        tdError = abs( np.max(targetQ, axis=1) - np.max(onlineQvalues, axis=1) )
        memory.updatePriorities(indices, tdError)

    def _updateTarget(self):
        self.targetNet.set_weights(self.model.get_weights()) 

    def _createAttention(self):
        if self.attentionView:
            outputs = [layer.output for layer in self.model.layers if "conv" in layer.name or "prep" in layer.name]
            self.attention = Model(inputs=self.model.input, outputs=outputs)
        
    def _updateAttention(self):
        if self.attentionView:
            layers = [layer for layer in self.model.layers if "conv" in layer.name]
            weights = []
            for layer in layers:
                weights.extend( layer.get_weights() )
            self.attention.set_weights(weights)

    def viewAttention(self, state):
        if self.attentionView:
            state = np.array( [state] )
            mask = np.ones( (1, self.numActions) )
            activations = self.attention.predict( [state, mask] )

            stack = []
            for activ in activations:
                image = np.mean(activ[0], axis=2)
                image = image/np.max(image) 
                colormap = plt.get_cmap('coolwarm')
                heatmap = colormap(image)
                heatmap = (heatmap * 256).astype(np.uint8)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGBA2BGRA)
                stack.append(heatmap)
            shape = (stack[0].shape[1], stack[0].shape[0])
            for j in range(1, len(stack)):
                stack[j] = cv2.resize(stack[j], shape)
            stack = np.vstack(stack)

            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 600, 1800)
            cv2.imshow('image', stack)
            cv2.waitKey(1)


class Noise(Layer):
    "And the agent may think, 'Hm, what if I try this?'"
    def __init__(self, stddev, kernel_initializer="glorot_uniform", **kwargs):
        super(Noise, self).__init__(**kwargs)
        self.stddev = stddev
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(int(input_shape[1]),),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        super(Noise, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, training=None):
        def noised():
            noise = K.random_normal(shape=(int(inputs.shape[1]),),
                                    mean=0.,
                                    stddev=self.stddev)
            noise = self.kernel*noise 
            output = inputs + noise
            return output
        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'stddev': self.stddev, 'kernel_initializer': self.kernel_initializer}
        base_config = super(Noise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape