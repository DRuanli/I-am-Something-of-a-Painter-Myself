import tensorflow as tf
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cyclegan_project.config import CONFIG
from cyclegan_project.data import load_monet_photo_datasets
from cyclegan_project.models import build_generator, build_discriminator
from cyclegan_project.trainers import CycleGANTrainer
from cyclegan_project.utils import show_sample_images


def train_model():
    """Train the CycleGAN model"""
    # Set random seed for reproducibility
    tf.random.set_seed(CONFIG['seed'])

    # Load datasets
    monet_ds, photo_ds, monet_ds_raw, photo_ds_raw = load_monet_photo_datasets()

    # Get sample images for visualization
    sample_monet = next(iter(monet_ds))
    sample_photo = next(iter(photo_ds))

    # Display sample images
    show_sample_images(sample_monet, sample_photo)

    # Build models
    monet_generator = build_generator()
    photo_generator = build_generator()
    monet_discriminator = build_discriminator()
    photo_discriminator = build_discriminator()

    # Create trainer
    trainer = CycleGANTrainer(
        monet_generator,
        photo_generator,
        monet_discriminator,
        photo_discriminator
    )

    # Train the model
    monet_generator, photo_generator = trainer.fit(
        monet_ds, photo_ds,
        monet_ds_raw, photo_ds_raw,
        sample_monet, sample_photo
    )

    print("Training complete!")
    return monet_generator, photo_generator


if __name__ == "__main__":
    train_model()