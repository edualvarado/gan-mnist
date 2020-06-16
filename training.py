"""
#####################
Assignment - InstaLOD
#####################
Deep Convolutional Generative Adversarial Network (GANs)
File: training.py
Author: Eduardo Alvarado
Task: In this assignment, I will create a GAN in order to generate novel numbers based on the MNIST dataset.
"""

from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from numpy import vstack
from numpy.random import randint, randn
from tensorflow.keras.datasets.mnist import load_data

# ================ #
BATCH_SIZE = 256
EPOCHS = 100
LATENT_DIM = 100
# ================ #


def plot_example_data(x):
    """
    Subplot example grayscale MNIST data with dimensions (28, 28, 1)
    :return: returns nothing
    """
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.axis("off")
        plt.imshow(np.squeeze(x[i]), cmap="gray_r")
    plt.show()


def load_real_data():
    """
    Load MNIST data.

    Load train and test datasets, reshape
    train-sets (include grayscale channel) and
    normalize between [0,1].
    :return: training data/labels and test data/labels
    """
    (x_train, y_train), (x_test, y_test) = load_data()

    # Data not normalized - Reshape (add grayscale dim) and normalize here
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype("float32")
    x_train = x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype("float32")
    x_test = x_test / 255.0
    print("After processing")
    print("Train X shape: {} - Train Y shape: {}".format(x_train.shape, y_train.shape))
    print("Test X shape: {} - Test Y shape: {}".format(x_test.shape, y_test.shape))
    # Plot example MNIST data
    # plot_example_data(x_train)
    return x_train, y_train, x_test, y_test


def generate_real_data(dataset, num_samples):
    """
    Prepare (shuffle) real data.

    Obtains random (num_samples) indexes of the dataset X
    and catches the respective labels Y
    :return: random dataset X and labels Y
    """
    random_positions = randint(0, dataset.shape[0], num_samples)
    random_x = dataset[random_positions]
    random_y = np.ones((num_samples, 1))
    return random_x, random_y


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


def generate_fake_data_gene(gene_model, latent_dim, num_samples):
    """
    It performs inference for the Generator model.
    Generate fake data using the Generator (initially from gaussian random noise, latent space).
    This data is labelled as "zero" since we know it is fake.
    :return: fake dataset X and fake labels Y
    """
    x_latent = generate_latent_data(latent_dim, num_samples)
    x_fake = gene_model.predict(x_latent)
    y_fake = np.zeros((num_samples, 1))
    return x_fake, y_fake


def save_fig(image, epoch, row_num_images=10):
    """
    Save generated "fake" image in root directory when project is located.
    Each time is called, it will save a set of subplots (size: row_num_images ** 2) with grayscale generated images.
    Function used as well for the inference.
    :return: fake dataset X and fake labels Y
    """
    filename = "generated_images_training/generated_images_epoch_%03d.png" % (epoch + 1)
    for i in range(row_num_images * row_num_images):
        plt.subplot(row_num_images, row_num_images, 1 + i)
        plt.axis("off")
        plt.imshow(image[i, :, :, 0], cmap="gray_r")
    plt.savefig(filename)
    plt.close()


def save_checkpoint(epoch, gene_model, dis_model, dataset, latent_dim, save_model, num_samples=100):
    """
    Each time is called, it prints accuracies and save checkpoint of the network (.h5).
    First, generate both, real data from MNIST dataset and fake data from Generator.
    Then, it performs evaluation for the Discriminator with both, printing its accuracy.
    Finally, saves the model at that point.
    :return: returns nothing
    """
    x_real, y_real = generate_real_data(dataset, num_samples)
    x_fake, y_fake = generate_fake_data_gene(gene_model, latent_dim, num_samples)
    _, acc_real = dis_model.evaluate(x_real, y_real, verbose=0)
    _, acc_fake = dis_model.evaluate(x_fake, y_fake, verbose=0)
    print("Checkpoint =================================================================")
    print("Image saved!")
    print("Epoch {} - Accuracy: on real data: {:.2f}% - on fake data: {:.2f}%".format(epoch, acc_real * 100,
                                                                                      acc_fake * 100))
    save_fig(x_fake, epoch)
    if save_model:
        print("Model (.h5) saved!")
        filename = "keras_models/generator_model_%03d.h5" % (epoch + 1)
        gene_model.save(filename)
    print("============================================================================")


def discriminator():
    """
    Create discriminator model.

    Take an image, which can be real or generated by the Generator (fake).
    Binary Classification (cross-entropy loss)
    (28, 28, 1) -> Conv2D -> (14, 14, 64) -> Conv2D -> (7,7,64) -> Flatten -> (3136) -> Dense -> (1)
    Optimizer: ADAM
    :return: Discriminator model
    """
    model = Sequential()
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same", input_shape=(28, 28, 1)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


print("\n===== Initialize Discriminator =====")
discriminator_model = discriminator()
# discriminator_model.summary()


def generator(latent_dim):
    """
    Create Generator model.
    From latent space (initially from gaussian random noise, latent space).
    We start from random low-res noise, then up-sample to 2D array (14x14 and finally to 28x28).
    We do not need to train the Generator in a standalone way.
    :return: Generator model
    """
    model = Sequential()
    num_nodes = 128 * 7 * 7
    model.add(Dense(num_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7, 7), activation="sigmoid", padding="same"))
    return model


print("\n===== Initialize Generator =====")
generator_model = generator(LATENT_DIM)
# generator_model.summary()


# Define GAN together
def gan(gene_model, dis_model):
    """
    Create GAN model, combining the Discriminator and Generator.
    We freeze discriminator, not not over-train it.
    We only want to update the Generator weights based on the Discriminator error.
    :return: GAN model (Discriminator + Generator)
    """
    dis_model.trainable = False  # freeze discriminator
    model = Sequential()
    model.add(gene_model)
    model.add(dis_model)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt)
    return model


print("\n===== Initialize GAN =====")
gan_model = gan(generator_model, discriminator_model)
gan_model.summary()


def train(gene_model, dis_model, gans_model, dataset, latent_dim, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    It trains the GAN model.

    First, it calculates how many batches per epoch we have based on the size of the dataset and size of the batch.
    We will train with half real data and half fake data, therefore we divide it by two.
    For each of the batches at each epoch:
    1- Generate half-real data from MNIST and half-fake data from current version of Generator model. Merge data.
    2- Update Discriminator weights (on batch) using this data (fake and real) and print accuracy.
    3- We need to prepare input for the Generator in a recursive way (data and labels).
    4- Update Generator weights with the combined model and error caused by the Discriminator.
    5- Save checkpoint each X epochs.
    :return: GAN model (Discriminator + Generator)
    """
    batches_per_epoch = int(dataset.shape[0] / batch_size)
    half_batch = int(batch_size / 2)
    epoch_checkpoint = 10
    for i in range(epochs):
        for j in range(batches_per_epoch):
            x_real, y_real = generate_real_data(dataset, half_batch)
            x_fake, y_fake = generate_fake_data_gene(gene_model, latent_dim, half_batch)
            x_training, y_training = np.vstack((x_real, x_fake)), vstack((y_real, y_fake))
            discriminator_loss, _ = dis_model.train_on_batch(x_training, y_training)
            x_gan = generate_latent_data(latent_dim, batch_size)
            y_gan = np.ones((batch_size, 1))
            loss_gan = gans_model.train_on_batch(x_gan, y_gan)
            print("Epoch: {} - Batch per epoch: {}/{} - Disc. loss: {:.2f}, GAN loss: {:.2f}".format(i + 1,
                                                                                                     j + 1,
                                                                                                     batches_per_epoch,
                                                                                                     discriminator_loss,
                                                                                                     loss_gan))
        if (i + 1) % epoch_checkpoint == 0:
            save_checkpoint(i, gene_model, dis_model, dataset, latent_dim, True)
        else:
            save_checkpoint(i, gene_model, dis_model, dataset, latent_dim, False)

    print("Training has finished =================================================================")
    gene_model.save("generator_model_final")
    print("Final generator model (.pb) saved!")
    print("=======================================================================================")


# Create folder for images
print("[INFO] Create folder for saving images during training...")
Path("generated_images_training").mkdir(parents=True, exist_ok=True)

# Create folder for Keras models
print("[INFO] Create folder for saving Keras models during training...")
Path("keras_models").mkdir(parents=True, exist_ok=True)

# Load real-data from MNIST dataset
print("[INFO] Loading data...")
train_x, _, _, _ = load_real_data()

# Training
print("[INFO] Training...")
train(generator_model, discriminator_model, gan_model, train_x, LATENT_DIM)


"""
For testing:
Generating fake data without Generator (noise).
Training Discriminator as standalone.
"""

"""
def generate_fake_data(num_samples):
    fake_x = rand(28 * 28 * num_samples)
    fake_x = fake_x.reshape((num_samples, 28, 28, 1)).astype("float32")
    fake_y = np.zeros((num_samples, 1))
    return fake_x, fake_y


num_samples = 20
fake_x, fake_y = generate_fake_data(generator_model, latent_dim, num_samples)


for i in range(num_samples):
    plt.subplot (4, 5, i+1)
    plt.axis("off")
    plt.imshow(fake_x[i, :, :, 0], cmap = "gray_r")
plt.show()


# Train discriminator (standalone)
def train_discriminator(model, dataset, n_iterations=100, batch_size=BATCH_SIZE):
    for i in range(n_iterations):
        x_real, y_real = generate_real_data(dataset, int(batch_size / 2))
        _, real_acc = model.train_on_batch(x_real, y_real)
        x_fake, y_fake = generate_fake_data(int(batch_size / 2))
        _, fake_acc = model.train_on_batch(x_fake, y_fake)
        print("i: {} -> real acc = {} - fake acc = {}".format(i, real_acc * 100, fake_acc * 100))


train_discriminator(discriminator_model, train_x)
"""
