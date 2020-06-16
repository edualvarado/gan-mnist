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

# ================ #
LATENT_DIM = 100
SAMPLES_PER_ROW = 5
# ================ #

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


print("[INFO] Loading pre-trained model...")
gan_model = load_model('generator_model_015.h5')
print("[INFO] Generating latent data...")
x_latent = generate_latent_data(LATENT_DIM, 25)
print("[INFO] Creating and saving prediction...")
generated_image = gan_model.predict(x_latent)
save_fig_inference(generated_image, SAMPLES_PER_ROW)
