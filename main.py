"""
#####################
Assignment - InstaLOD
#####################
Deep Convolutional Generative Adversarial Network (GANs)
Author: Eduardo Alvarado
Task: In this assignment, I will create a GAN in order to generate novel numbers based on the MNIST dataset.
"""

import tensorflow as tf
print(tf.__version__)

import matplotlib.pyplot as plt
import numpy as np


# MNIST dataset - define sets and print information
from tensorflow.keras.datasets.mnist import load_data
(train_x, train_y), (test_x, test_y) = load_data()
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