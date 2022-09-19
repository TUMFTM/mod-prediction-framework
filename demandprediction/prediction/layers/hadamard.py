from tensorflow.keras.layers import Layer


class Hadamard(Layer):
    """
    Keras Layer to perform a trainable Hadamard multiplication.
    """

    def __init__(self, **kwargs):
        super(Hadamard, self).__init__(**kwargs)
        self.kernel = None

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,) + input_shape[1:],
                                      initializer='uniform',
                                      trainable=True)
        super(Hadamard, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * self.kernel

    def compute_output_shape(self, input_shape):
        return input_shape
