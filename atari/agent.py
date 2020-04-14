import numpy as np
import os, random
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.initializers as initializers
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Lambda, multiply, Layer
from tensorflow.keras.optimizers import Adam

def loadModel(fileName):
    # any further custom objects must be added to custom_object
    return load_model(fileName, custom_objects={'Noise':Noise})

class Agent:
    def __init__(self, agentName, numActions, modelLoad=None, targetLoad=None, targetUpdateFreq=2500, gamma=0.99):
        self.numActions = numActions
        self.gamma = gamma
        self.name = agentName
        self.learnIter = 0
        self.targetUpdateFreq = targetUpdateFreq
        self.modelName = self.name + ".h5"
        if modelLoad is None:
            self.model = self._createModel()
        else:
            self.model = loadModel(modelLoad)
        self.targetName = self.name + "_target_temp.h5"
        if targetLoad is None:
            self.targetNet = self.model
        else:
            self.targetNet = loadModel(targetLoad)

    def __del__(self):
        if os.path.exists(self.targetName):
            os.remove(self.targetName)
        self.model.save(self.modelName)
    
    def predict(self, state):
        state = np.array( [state] )
        mask = np.ones( (1, self.numActions) )
        return self.model.predict( [state, mask] )

    def getAction(self, state):
        return np.argmax( self.predict(state) ) 

    def learn(self, batch):
        startStates, nextStates, actions, rewards, isTerminal = batch
        # predict future award
        nextQvalues = self.targetNet.predict( [nextStates, np.ones(actions.shape)] )
        # and we ask, "Was that the actual reward, considering gamma portion of future predicted award?"
        nextQvalues[actions] = rewards + self.gamma*(1-isTerminal)*np.max(nextQvalues, axis=1)
        # so if we start in these states, the rewards should look like this
        self.model.fit([startStates, actions], actions*nextQvalues, batch_size=startStates.size, verbose=0)

        self.learnIter += 1
        if self.learnIter % self.targetUpdateFreq == 0:
            self._updateTarget()
    
    def _updateTarget(self):
        self.model.save(self.targetName)
        self.targetNet = loadModel(self.targetName)

    def _createModel(self):
        init = VarianceScaling( scale=2 )
        framesIn = Input((105,80,4), name="frames")
        actionsIn = Input((self.numActions,), name="mask")

        normalized = Lambda(lambda x: x / 255.0)(framesIn)
        conv1 = Conv2D(16, 8, strides=4 , activation="relu", kernel_initializer=init
        )(normalized)
        conv2 = Conv2D(32, 4, strides=2, activation="relu", kernel_initializer=init
        )(conv1)
        flattened = Flatten()(conv2)
        hidden = Dense(256, activation="relu", kernel_initializer=init)(flattened)
        noise = Noise(stddev=1, kernel_initializer=init)(hidden)
        output = Dense(self.numActions, kernel_initializer=init)(noise)

        filteredOut = multiply([output, actionsIn])
        model = Model(inputs=[framesIn, actionsIn], outputs=filteredOut)

        opt = Adam( learning_rate=.0001 )
        model.compile(opt, loss="huber_loss")
        return model


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