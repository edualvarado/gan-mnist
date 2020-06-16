"""
#####################
Assignment - InstaLOD
#####################
Deep Convolutional Generative Adversarial Network (GANs)
File: inference.py
Author: Eduardo Alvarado
Task: In this assignment, I will create a GAN in order to generate novel numbers based on the MNIST dataset.
"""

from numpy.random import randn
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model


def generate_latent_data(latent_dim, num_samples):
    """
    Prepare latent dimensions for Generator.
    It creates random gaussian values for "latent_dim" dimensions.
    The number of dimensions can be changed.
    :return: random latent data
    """

    x_input_generator = randn(latent_dim * num_samples)
    x_input_generator = x_input_generator.reshape(num_samples, latent_dim)
    return x_input_generator


def save_fig_inference(image, row_num_images=10):
    """
    Save generated "fake" images during inference in root directory when project is located.
    Each time is called, it will save a set of subplots (size: row_num_images ** 2) with grayscale generated images.
    Function used as well for the inference.
    :return: fake dataset X and fake labels Y
    """
    filename = "generated_images_inference.png"
    for i in range(row_num_images * row_num_images):
        plt.subplot(row_num_images, row_num_images, 1 + i)
        plt.axis("off")
        plt.imshow(image[i, :, :, 0], cmap="gray_r")
    plt.savefig(filename)
    plt.close()


# load model
model = load_model('generator_model_015.h5')
latent_points = generate_latent_data(100, 25)
X = model.predict(latent_points)
save_fig_inference(X, 5)