import tensorflow as tf
from cyclegan_project.config import CONFIG
from .layers import InstanceNormalization


def downsample(filters, size, apply_norm=True):
    """Downsampling block for generator and discriminator"""
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    return result


def upsample(filters, size, apply_dropout=False):
    """Upsampling block for generator"""
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                               kernel_initializer=initializer, use_bias=False))

    result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(CONFIG['dropout_rate']))

    result.add(tf.keras.layers.ReLU())
    return result


def build_generator():
    """Build U-Net generator model"""
    print("Building generator...")
    inputs = tf.keras.layers.Input(shape=[CONFIG['img_height'], CONFIG['img_width'], 3])

    # Encoder (downsampling)
    down_stack = [
        downsample(64, 4, apply_norm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64, 64, 128)
        downsample(256, 4),  # (batch_size, 32, 32, 256)
        downsample(512, 4),  # (batch_size, 16, 16, 512)
        downsample(512, 4),  # (batch_size, 8, 8, 512)
        downsample(512, 4),  # (batch_size, 4, 4, 512)
        downsample(512, 4),  # (batch_size, 2, 2, 512)
        downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    # Decoder (upsampling)
    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample(512, 4),  # (batch_size, 16, 16, 1024)
        upsample(256, 4),  # (batch_size, 32, 32, 512)
        upsample(128, 4),  # (batch_size, 64, 64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Store the outputs of each encoder block for skip connections
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Add skip connections and upsample
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def build_discriminator():
    """Build PatchGAN discriminator model"""
    print("Building discriminator...")
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[CONFIG['img_height'], CONFIG['img_width'], 3], name='input_image')

    x = downsample(64, 4, False)(inp)  # No normalization in the first layer
    x = downsample(128, 4)(x)
    x = downsample(256, 4)(x)
    x = downsample(512, 4)(x)

    # Don't use sigmoid to avoid vanishing gradients
    x = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(x)

    return tf.keras.Model(inputs=inp, outputs=x)