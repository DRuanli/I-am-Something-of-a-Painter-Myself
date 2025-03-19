import tensorflow as tf
import numpy as np
import time
import os
from cyclegan_project.config import CONFIG
from cyclegan_project.models import generator_loss, discriminator_loss, calc_cycle_loss, identity_loss
from cyclegan_project.utils.visualization import display_progress_bar, generate_images


class CycleGANTrainer:
    def __init__(self, monet_generator, photo_generator, monet_discriminator, photo_discriminator):
        self.monet_generator = monet_generator
        self.photo_generator = photo_generator
        self.monet_discriminator = monet_discriminator
        self.photo_discriminator = photo_discriminator

        # Initialize optimizers
        self.monet_generator_optimizer = tf.keras.optimizers.Adam(
            CONFIG['learning_rate'], beta_1=CONFIG['beta1'])
        self.photo_generator_optimizer = tf.keras.optimizers.Adam(
            CONFIG['learning_rate'], beta_1=CONFIG['beta1'])
        self.monet_discriminator_optimizer = tf.keras.optimizers.Adam(
            CONFIG['learning_rate'], beta_1=CONFIG['beta1'])
        self.photo_discriminator_optimizer = tf.keras.optimizers.Adam(
            CONFIG['learning_rate'], beta_1=CONFIG['beta1'])

        # Setup checkpoint
        self.checkpoint_prefix = os.path.join(CONFIG['checkpoint_dir'], "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            monet_generator_optimizer=self.monet_generator_optimizer,
            photo_generator_optimizer=self.photo_generator_optimizer,
            monet_discriminator_optimizer=self.monet_discriminator_optimizer,
            photo_discriminator_optimizer=self.photo_discriminator_optimizer,
            monet_generator=self.monet_generator,
            photo_generator=self.photo_generator,
            monet_discriminator=self.monet_discriminator,
            photo_discriminator=self.photo_discriminator
        )

    @tf.function
    def train_step(self, real_monet, real_photo):
        """Execute a single training step for CycleGAN"""
        with tf.GradientTape(persistent=True) as tape:
            # Generate fake images
            fake_photo = self.monet_generator(real_monet, training=True)
            fake_monet = self.photo_generator(real_photo, training=True)

            # Generate cycled images
            cycled_monet = self.photo_generator(fake_photo, training=True)
            cycled_photo = self.monet_generator(fake_monet, training=True)

            # Generate same images (identity mapping)
            same_monet = self.photo_generator(real_monet, training=True)
            same_photo = self.monet_generator(real_photo, training=True)

            # Discriminator outputs
            disc_real_monet = self.monet_discriminator(real_monet, training=True)
            disc_real_photo = self.photo_discriminator(real_photo, training=True)

            disc_fake_monet = self.monet_discriminator(fake_monet, training=True)
            disc_fake_photo = self.photo_discriminator(fake_photo, training=True)

            # Generator losses
            monet_gen_loss = generator_loss(disc_fake_monet)
            photo_gen_loss = generator_loss(disc_fake_photo)

            # Cycle consistency losses
            total_cycle_loss = calc_cycle_loss(real_monet, cycled_monet) + calc_cycle_loss(real_photo, cycled_photo)

            # Identity losses
            total_identity_loss = identity_loss(real_monet, same_monet) + identity_loss(real_photo, same_photo)

            # Total generator losses
            monet_total_gen_loss = tf.reduce_mean(monet_gen_loss) + total_cycle_loss + total_identity_loss
            photo_total_gen_loss = tf.reduce_mean(photo_gen_loss) + total_cycle_loss + total_identity_loss

            # Discriminator losses
            monet_disc_loss = discriminator_loss(disc_real_monet, disc_fake_monet)
            photo_disc_loss = discriminator_loss(disc_real_photo, disc_fake_photo)

            monet_total_disc_loss = tf.reduce_mean(monet_disc_loss)
            photo_total_disc_loss = tf.reduce_mean(photo_disc_loss)

        # Calculate gradients
        monet_generator_gradients = tape.gradient(monet_total_gen_loss, self.monet_generator.trainable_variables)
        photo_generator_gradients = tape.gradient(photo_total_gen_loss, self.photo_generator.trainable_variables)

        monet_discriminator_gradients = tape.gradient(monet_total_disc_loss,
                                                      self.monet_discriminator.trainable_variables)
        photo_discriminator_gradients = tape.gradient(photo_total_disc_loss,
                                                      self.photo_discriminator.trainable_variables)

        # Apply gradients
        self.monet_generator_optimizer.apply_gradients(
            zip(monet_generator_gradients, self.monet_generator.trainable_variables))
        self.photo_generator_optimizer.apply_gradients(
            zip(photo_generator_gradients, self.photo_generator.trainable_variables))

        self.monet_discriminator_optimizer.apply_gradients(
            zip(monet_discriminator_gradients, self.monet_discriminator.trainable_variables))
        self.photo_discriminator_optimizer.apply_gradients(
            zip(photo_discriminator_gradients, self.photo_discriminator.trainable_variables))

        return {
            'monet_gen_loss': monet_total_gen_loss,
            'photo_gen_loss': photo_total_gen_loss,
            'monet_disc_loss': monet_total_disc_loss,
            'photo_disc_loss': photo_total_disc_loss
        }

    def fit(self, monet_ds, photo_ds, monet_ds_raw, photo_ds_raw,
            sample_monet, sample_photo, epochs=None):
        """Train the CycleGAN models for the specified number of epochs"""
        if epochs is None:
            epochs = CONFIG['epochs']

        print("Starting training...")

        # Try to calculate total number of steps per epoch
        try:
            monet_size = sum(1 for _ in monet_ds_raw)
            photo_size = sum(1 for _ in photo_ds_raw)
            steps_per_epoch = min(monet_size, photo_size)
            print(f"Training with {monet_size} Monet images and {photo_size} photos")
        except:
            # If counting fails (e.g., due to large dataset), estimate
            steps_per_epoch = 300  # Approximation based on Monet dataset size
            print(f"Using estimated steps per epoch: {steps_per_epoch}")

        print(f"Steps per epoch: {steps_per_epoch}")

        for epoch in range(epochs):
            start = time.time()
            print(f"\nEpoch {epoch + 1}/{epochs}")

            # Reset datasets for each epoch
            monet_ds_iter = iter(monet_ds)
            photo_ds_iter = iter(photo_ds)

            # Track metrics for this epoch
            epoch_losses = {
                'monet_gen_loss': [],
                'photo_gen_loss': [],
                'monet_disc_loss': [],
                'photo_disc_loss': []
            }

            # Train for one epoch
            for step in range(steps_per_epoch):
                try:
                    image_m = next(monet_ds_iter)
                    image_p = next(photo_ds_iter)

                    losses = self.train_step(image_m, image_p)

                    # Collect losses
                    for key, value in losses.items():
                        epoch_losses[key].append(float(value))

                    # Display progress
                    if step % 10 == 0:
                        # Display current losses
                        loss_str = ', '.join(f"{k}: {v:.4f}" for k, v in losses.items())
                        display_progress_bar(step, steps_per_epoch,
                                             f"Epoch {epoch + 1}/{epochs} - {loss_str} - ",
                                             length=40)

                    # Generate sample image every 100 steps
                    if step % 100 == 0:
                        # Generate and save sample
                        generate_images(self.monet_generator, sample_photo,
                                        filename=f'output/sample_epoch_{epoch + 1}_step_{step}.jpg')

                except (tf.errors.OutOfRangeError, StopIteration):
                    break

            # Display final progress
            display_progress_bar(steps_per_epoch, steps_per_epoch,
                                 f"Epoch {epoch + 1}/{epochs} - Complete - ",
                                 length=40)

            # Display epoch summary
            avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
            print(f"\nEpoch {epoch + 1} Summary:")
            for key, value in avg_losses.items():
                print(f"  {key}: {value:.4f}")

            # Save checkpoint at specified intervals
            if (epoch + 1) % CONFIG['save_checkpoint_epochs'] == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                print(f"Checkpoint saved at epoch {epoch + 1}")

            print(f'Time taken for epoch {epoch + 1}: {time.time() - start:.2f} sec')

            # Generate samples at the end of each epoch
            generate_images(self.monet_generator, sample_photo,
                            filename=f'output/epoch_{epoch + 1}_final.jpg')

        return self.monet_generator, self.photo_generator