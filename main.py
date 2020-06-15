"""
#####################
Assignment - InstaLOD
#####################
Deep Convolutional Generative Adversarial Network (GANs)
Author: Eduardo Alvarado
Task: In this assignment, I will create a GAN in order to generate novel numbers based on the MNIST dataset.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import rand, randint, randn

# MNIST dataset - define sets and print information
from tensorflow.keras.datasets.mnist import load_data
(train_x, train_y), (test_x, test_y) = load_data()
print("Before processing")
print("Train X shape: {} - Train Y shape: {}".format(train_x.shape, train_y.shape))
print("Test X shape: {} - Test Y shape: {}".format(test_x.shape, test_y.shape))

# Data not normalized - Reshape (add grayscale dim) and normalize here
train_x = train_x.reshape(train_x.shape[0], 28, 28, 1).astype("float32")
train_x = train_x / 255.0
test_x = test_x.reshape(test_x.shape[0], 28, 28, 1).astype("float32")
test_x = test_x / 255.0
print("After processing")
print("Train X shape: {} - Train Y shape: {}".format(train_x.shape, train_y.shape))
print("Test X shape: {} - Test Y shape: {}".format(test_x.shape, test_y.shape))

# Plotting numbers for testing
'''
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.axis("off")
    plt.imshow(train_x[i], cmap="gray_r")
plt.show()
'''

# Shuffle batches and randomize data
BUFFER_SIZE = 60000
BATCH_SIZE = 256


# Function to randomize datasets with respective labels
def shuffle_real_set(dataset, num_samples):
    random_positions = randint(0, dataset.shape[0], num_samples)
    random_x = dataset[random_positions]
    random_y = np.ones((num_samples, 1))
    return random_x, random_y


# Discriminator (binary classification)
# (28, 28, 1) -> (14, 14, 64) -> (7,7,64) -> (3136) -> (1)
def discriminator():
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides = (2,2), padding = "same", input_shape = (28,28,1)))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3,3), strides = (2,2), padding = "same"))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation = "sigmoid"))
    opt = Adam(lr = 0.0002, beta_1 = 0.5)
    model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])
    return model


print("\n===== DISCRIMINATOR =====")
discriminator_model = discriminator()
discriminator_model.summary()


# For testing - generate fake data (noise)
def generate_fake_data(num_samples):
    fake_x = rand(28 * 28 * num_samples)
    fake_x = fake_x.reshape((num_samples, 28, 28, 1)).astype("float32")
    fake_y = np.zeros((num_samples, 1))
    return fake_x, fake_y


# Train discriminator (TEST)
def train_discriminator(model, dataset, n_iterations = 100, batch_size = BATCH_SIZE):
    for i in range(n_iterations):
        x_real, y_real = shuffle_real_set(dataset, int(batch_size/2))
        _, real_acc = model.train_on_batch(x_real, y_real)
        x_fake, y_fake = generate_fake_data(int(batch_size/2))
        _, fake_acc = model.train_on_batch(x_fake, y_fake)
        print("i: {} -> real acc = {} - fake acc = {}".format(i, real_acc*100, fake_acc * 100))

# train_discriminator(discriminator_model, train_x)

'''
Until here, we can train standalone discriminator
'''


# Generator (create images with each value in a range of [0,1]
# We start from random low-res noise, then upsample to 14x14 and finally 28x28
# TODO: Check deconvolutions and shapes

def generator(latent_dim):
    model = Sequential()
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim = latent_dim))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = "same"))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = "same"))
    model.add(LeakyReLU(alpha = 0.2))
    model.add(Conv2D(1, (7, 7), activation = "sigmoid", padding = "same"))
    return model


print("\n===== GENERATOR =====")
latent_dim = 100
generator_model = generator(latent_dim)
generator_model.summary()

def generate_latent_points(latent_dim, num_samples):
    x_input_generator = randn(latent_dim * num_samples)
    x_input_generator = x_input_generator.reshape(num_samples, latent_dim)
    return x_input_generator



