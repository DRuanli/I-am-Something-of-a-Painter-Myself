import tensorflow as tf
import argparse
import os

from cyclegan_project.config import CONFIG
from cyclegan_project.data import load_monet_photo_datasets
from cyclegan_project.models import build_generator, build_discriminator
from cyclegan_project.trainers import CycleGANTrainer
from cyclegan_project.utils import show_sample_images
from cyclegan_project.scripts.generate import generate_images


def main():
    """Main entry point for the CycleGAN training and image generation"""
    parser = argparse.ArgumentParser(description='Train CycleGAN for Monet-style image generation')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--generate', action='store_true', help='Generate images using trained model')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'], help='Number of training epochs')
    parser.add_argument('--num_images', type=int, default=CONFIG['num_generated_images'],
                        help='Number of images to generate')

    args = parser.parse_args()

    # Set random seed for reproducibility
    tf.random.set_seed(CONFIG['seed'])

    # Create output directories if they don't exist
    for dir_path in [CONFIG['output_dir'], CONFIG['generated_images_dir'], CONFIG['checkpoint_dir']]:
        os.makedirs(dir_path, exist_ok=True)

    # Train or generate or both
    monet_generator = None

    if args.train:
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
        monet_generator, _ = trainer.fit(
            monet_ds, photo_ds,
            monet_ds_raw, photo_ds_raw,
            sample_monet, sample_photo,
            epochs=args.epochs
        )

        print("Training complete!")

    if args.generate:
        generate_images(model=monet_generator, num_images=args.num_images)

    # If no flags provided, print help
    if not (args.train or args.generate):
        parser.print_help()


if __name__ == "__main__":
    main()
