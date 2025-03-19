import tensorflow as tf
from cyclegan_project.config import CONFIG


def discriminator_loss(real, generated):
    """Discriminator loss function"""
    real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(real), real)
    generated_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.zeros_like(generated), generated)

    total_loss = real_loss + generated_loss
    return total_loss * 0.5


def generator_loss(generated):
    """Generator loss function"""
    return tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(
        tf.ones_like(generated), generated)


def calc_cycle_loss(real_image, cycled_image):
    """Cycle consistency loss function"""
    loss = tf.reduce_mean(tf.abs(real_image - cycled_image))
    return CONFIG['lambda_cycle'] * loss


def identity_loss(real_image, same_image):
    """Identity loss function"""
    loss = tf.reduce_mean(tf.abs(real_image - same_image))
    return CONFIG['lambda_cycle'] * 0.5 * loss