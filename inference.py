"""
#####################
Assignment - InstaLOD
#####################
Deep Convolutional Generative Adversarial Network (GANs)
Author: Eduardo Alvarado
Task: In this assignment, I will create a GAN in order to generate novel numbers based on the MNIST dataset.
"""

from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from training import generate_latent_points

# ================ #
LATENT_DIM = 100
SAMPLES = 10
MODEL = "generator_model_015.h5"
# ================ #


def save_fig_inference(image, samples):
    """
    Save generated "fake" image during inference in root directory when project is located.
    :return: returns nothing
    """
    filename = "generated_images_inference.png"
    for i in range(samples * samples):
        plt.subplot(samples, samples, 1 + i)
        plt.axis("off")
        plt.imshow(image[i, :, :, 0], cmap="gray_r")
    plt.savefig(filename)
    plt.close()

print("INFERING")
gan_model = load_model(MODEL)
generator_input = generate_latent_points(LATENT_DIM, SAMPLES)
generated_images = gan_model.predict(generator_input)
save_fig_inference(generated_images, SAMPLES)
