
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras.initializers as initializers
from tensorflow.keras import activations, Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.losses import binary_crossentropy


def loadModel(fileName):
    # any further custom objects must be added to custom_object dict
    return load_model(fileName, custom_objects={'Noise':Noise})

class Noise(Layer):
    "And the agent may think, 'Hm, what if I try this?'"
    def __init__(self, stddev=1.0, kernel_initializer="glorot_uniform", **kwargs):
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


class KLDivergenceLayer(Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """
    def __init__(self, beta=4, *args, **kwargs):
        self.is_placeholder = True
        self.beta = beta
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)
    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = - .5 * K.sum(1 + log_var - K.square(mu) - K.exp(log_var), axis=-1)
        self.add_loss(self.beta*K.mean(kl_batch), inputs=inputs)
        return inputs


class Entropy(Layer):
    """ Identity transform layer that adds Entropy
    to the final model loss.
    """
    def __init__(self, scale=1.0, *args, **kwargs):
        self.is_placeholder = True
        self.scale = scale
        super(Entropy, self).__init__(*args, **kwargs)
    def call(self, inputs):
        entropy_batch = -K.sum(inputs * K.log(inputs+.0001)) 
        # change sign to maximize with gradient descent
        entropy_batch = -entropy_batch
        self.add_loss(self.scale*K.mean(entropy_batch), inputs=inputs)
        return inputs


class MutualInformation(Layer):
    """ Identity transform layer that adds Mutual Information
    to the final model loss.
    """
    def __init__(self, scale=1.0, *args, **kwargs):
        self.is_placeholder = True
        self.scale = scale
        super(MutualInformation, self).__init__(*args, **kwargs)
    def call(self, inputs):
        x, x_given_y = inputs
        x_entropy = -K.sum(x * K.log(x+.0001)) 
        x_given_y_entropy = -K.sum(x_given_y * K.log(x_given_y+.0001))
        mi_batch = x_entropy - x_given_y_entropy
        # change sign to maximize with gradient descent
        mi_batch = -mi_batch
        self.add_loss(self.scale*K.mean(mi_batch), inputs=inputs)
        return inputs
