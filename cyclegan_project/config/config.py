import os

CONFIG = {
    # Data parameters
    'monet_tfrecord_pattern': 'monet_tfrec/*.tfrec',
    'photo_tfrecord_pattern': 'photo_tfrec/*.tfrec',
    'img_height': 256,
    'img_width': 256,
    'buffer_size': 1000,
    'batch_size': 1,

    # Training parameters
    'epochs': 10,  # Set to 40+ for production quality
    'learning_rate': 2e-4,
    'beta1': 0.5,
    'lambda_cycle': 10,
    'save_checkpoint_epochs': 2,

    # Generation parameters
    'num_generated_images': 20,

    # Directories
    'output_dir': 'output',
    'generated_images_dir': 'generated_images',
    'checkpoint_dir': 'training_checkpoints',

    # Model parameters
    'dropout_rate': 0.5,
    'seed': 42
}

# Create directories if they don't exist
for dir_path in [CONFIG['output_dir'], CONFIG['generated_images_dir'], CONFIG['checkpoint_dir']]:
    os.makedirs(dir_path, exist_ok=True)