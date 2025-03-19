import tensorflow as tf

class InstanceNormalization(tf.keras.layers.Layer):
    """
    Custom Instance Normalization layer
    (replacement for tensorflow_addons.layers.InstanceNormalization)
    """
    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
        self.scale = None
        self.offset = None

    def build(self, input_shape):
        depth = input_shape[-1]
        self.scale = self.add_weight(
            name='scale',
            shape=[depth],
            initializer=tf.random_normal_initializer(1.0, 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=[depth],
            initializer='zeros',
            trainable=True)

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset