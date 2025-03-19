import tensorflow as tf
from cyclegan_project.config import CONFIG


def decode_image(image):
    """Decode and normalize image data from TFRecord"""
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    image = tf.reshape(image, [CONFIG['img_height'], CONFIG['img_width'], 3])
    return image


def read_tfrecord(example):
    """Parse a single TFRecord example"""
    tfrecord_format = {
        'image_name': tf.io.FixedLenFeature([], tf.string),
        'image': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    return image


def random_jitter(image):
    """Apply data augmentation with random crop and flip"""
    # Resize to larger than target size
    image = tf.image.resize(image, [286, 286], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Randomly crop back to target size
    image = tf.image.random_crop(image, size=[CONFIG['img_height'], CONFIG['img_width'], 3])
    # Random mirroring
    image = tf.image.random_flip_left_right(image)
    return image


def load_dataset(file_pattern):
    """Load TFRecord dataset from file pattern"""
    dataset = tf.data.TFRecordDataset(file_pattern)
    return dataset


def prepare_dataset(ds, augment=True, shuffle=True, batch_size=None):
    """Prepare dataset for training/inference"""
    if batch_size is None:
        batch_size = CONFIG['batch_size']

    # Map the TFRecord parsing function
    ds = ds.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    # Use data augmentation only during training
    if augment:
        ds = ds.map(random_jitter, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(CONFIG['buffer_size'])

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def load_monet_photo_datasets(monet_dir=None, photo_dir=None, augment=True):
    """Load and prepare both Monet and photo datasets"""
    if monet_dir is None:
        monet_dir = CONFIG['monet_tfrecord_pattern']
    if photo_dir is None:
        photo_dir = CONFIG['photo_tfrecord_pattern']

    # Find all TFRecord files
    print("Loading dataset files...")
    monet_files = tf.data.Dataset.list_files(monet_dir)
    photo_files = tf.data.Dataset.list_files(photo_dir)

    # Create datasets from the file patterns
    monet_ds_raw = monet_files.interleave(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    photo_ds_raw = photo_files.interleave(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)

    # Prepare the datasets
    print("Preparing datasets...")
    monet_ds = prepare_dataset(monet_ds_raw, augment=augment)
    photo_ds = prepare_dataset(photo_ds_raw, augment=augment)

    return monet_ds, photo_ds, monet_ds_raw, photo_ds_raw