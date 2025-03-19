import tensorflow as tf
import time
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cyclegan_project.config import CONFIG
from cyclegan_project.data import load_monet_photo_datasets, prepare_dataset
from cyclegan_project.models import build_generator
from cyclegan_project.utils import save_generated_image, display_generated_samples


def generate_images(model=None, num_images=None):
    """Generate Monet-style images from photos"""
    if num_images is None:
        num_images = CONFIG['num_generated_images']

    # If no model is provided, load a pretrained model
    if model is None:
        print("Loading pre-trained generator...")
        model = build_generator()

        # Load the checkpoint
        checkpoint = tf.train.Checkpoint(monet_generator=model)
        checkpoint_dir = CONFIG['checkpoint_dir']

        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            print(f"Restored from {latest_checkpoint}")
        else:
            print("No checkpoint found. Using uninitialized model.")

    # Load only the photo dataset (no need for Monet images)
    _, _, _, photo_ds_raw = load_monet_photo_datasets(augment=False)

    print(f"\nGenerating {num_images} Monet-style images...")

    photo_ds_for_gen = prepare_dataset(photo_ds_raw, augment=False, shuffle=True, batch_size=1)
    photo_ds_iter = iter(photo_ds_for_gen)

    for i in range(num_images):
        try:
            # Get next photo
            photo = next(photo_ds_iter)

            # Generate and save the image
            start_time = time.time()
            image_path = f'{CONFIG["generated_images_dir"]}/monet_generated_{i:04d}.jpg'
            save_generated_image(model, photo, image_path)
            time_taken = time.time() - start_time

            print(f'Generated image {i + 1}/{num_images} in {time_taken:.2f}s - Saved to {image_path}')

        except (tf.errors.OutOfRangeError, StopIteration):
            print(f"Ran out of photos after generating {i} images.")
            break

    # Display sample of generated images
    display_generated_samples(min(5, num_images))

    print("Image generation complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Monet-style images from photos')
    parser.add_argument('--num_images', type=int, default=CONFIG['num_generated_images'],
                        help='Number of images to generate')

    args = parser.parse_args()

    generate_images(num_images=args.num_images)