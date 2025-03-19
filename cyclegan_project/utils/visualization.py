import tensorflow as tf
import matplotlib.pyplot as plt
import os
from cyclegan_project.config import CONFIG


def display_progress_bar(progress, total, prefix='', length=30):
    """Display a progress bar in the console"""
    filled_length = int(length * progress // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {progress}/{total}', end='\r')
    if progress == total:
        print()


def generate_images(model, test_input, filename=None):
    """Generate and display images using the model"""
    prediction = model(test_input)

    plt.figure(figsize=(12, 6))

    display_list = [test_input[0], prediction[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        # Denormalize the images from [-1, 1] to [0, 1]
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    if filename:
        plt.savefig(filename)

    plt.show()


def save_generated_image(model, photo, filename):
    """Generate and save a single image"""
    # Generate the Monet-style image
    generated_image = model(photo)

    # Denormalize from [-1, 1] to [0, 1]
    generated_image = (generated_image[0] * 0.5 + 0.5)

    # Convert to uint8 [0, 255]
    generated_image = tf.cast(generated_image * 255, tf.uint8)

    # Save image
    tf.io.write_file(filename, tf.io.encode_jpeg(generated_image))

    return generated_image


def show_sample_images(sample_monet, sample_photo):
    """Display sample images from both datasets"""
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title('Monet')
    plt.imshow(sample_monet[0] * 0.5 + 0.5)  # Denormalize
    plt.subplot(122)
    plt.title('Photo')
    plt.imshow(sample_photo[0] * 0.5 + 0.5)  # Denormalize
    plt.savefig('sample_images.png')
    plt.show()


def display_generated_samples(num_samples=5):
    """Display a grid of generated samples"""
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        try:
            img = tf.io.read_file(f'{CONFIG["generated_images_dir"]}/monet_generated_{i:04d}.jpg')
            img = tf.image.decode_jpeg(img)
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img)
            plt.axis('off')
        except:
            pass
    plt.suptitle("Generated Monet-Style Images", fontsize=16)
    plt.savefig('generated_samples.png')
    plt.show()