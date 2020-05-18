import tensorflow.keras.initializers as initializers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


def loadModel(fileName):
    # any further custom objects must be added to custom_object dict
    return load_model(fileName, custom_objects={'Noise':Noise})

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