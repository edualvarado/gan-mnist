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
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

print(tf.__version__)

import matplotlib.pyplot as plt
import numpy as np
import random

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
def shuffle_real_set(dataset, BUFFER_SIZE, BATCH_SIZE):
    random_positions = random.randint(0, BUFFER_SIZE, BATCH_SIZE)
    random_x = dataset[random_positions]
    random_y = np.ones((BATCH_SIZE, 1))
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

discriminator_model = discriminator()
discriminator_model.summary()

