import tensorflow.keras.initializers as initializers
from tensorflow.keras import activations
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


def loadModel(fileName):
    # any further custom objects must be added to custom_object dict
    return load_model(fileName, custom_objects={'Noise':Noise, 'Whiteout':Whiteout})

class Noise(Layer):
    "And the agent may think, 'Hm, what if I try this?'"
    def __init__(self, stddev=1.0, kernel_initializer="glorot_uniform", **kwargs):
        super(Noise, self).__init__(**kwargs)
        self.stddev = stddev
        self.kernel_initializer = initializers.get(kernel_initializer)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(int(input_shape[1]),),
                                      initializer=self.kernel_initializer,
                                      trainable=True)
        super(Noise, self).build(input_shape) 

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

class Whiteout(Layer):
    def __init__(self, std=1.0, mean=0.0, **kwargs):
        super(Whiteout, self).__init__(**kwargs)
        """
        These values are to be changed as program runs.
        Otherwise this layer will simply clip the input between -5 and 5.
        """
        self.std = std
        self.mean = mean

    def build(self, input_shape):
        super(Whiteout, self).build(input_shape)

    def call(self, inputs):
        normalized = (inputs-self.mean)/(self.std+.0001)
        clipped = K.clip(normalized, -5, 5)
        return clipped

    def get_config(self):
        config = {'std': self.std, 'mean': self.mean}
        base_config = super(Whiteout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape