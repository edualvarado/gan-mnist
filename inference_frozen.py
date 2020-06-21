"""
#####################
Deep Convolutional Generative Adversarial Network (GANs)
#####################
File: inference.py
Author: Eduardo Alvarado
Task: In this assignment, I will create a GAN in order to generate novel numbers based on the MNIST dataset.
"""

from pathlib import Path
from numpy.random import randn
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

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
    filename = "generated_images_inference/generated_image_inference.png"
    for i in range(row_num_images * row_num_images):
        plt.subplot(row_num_images, row_num_images, 1 + i)
        plt.axis("off")
        plt.imshow(image[i, :, :, 0], cmap="gray_r")
    plt.savefig(filename)
    plt.close()


def wrap_frozen_graph(graph_def, inputs, outputs, print_graph=False):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")

    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    print("-" * 50)
    print("Frozen model layers: ")
    layers = [op.name for op in import_graph.get_operations()]
    if print_graph == True:
        for layer in layers:
            print(layer)
    print("-" * 50)

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

# Create folder for images
print("[INFO] Create folder for saving images during inference...")
Path("generated_images_inference").mkdir(parents=True, exist_ok=True)

# Generate input for Generator
print("[INFO] Generating latent data...")
x_latent = generate_latent_data(LATENT_DIM, 25)

"""
# Load pre-trained Keras model
print("[INFO] Loading pre-trained model...")
gan_model = load_model('generator_model_final')

# Inference
print("[INFO] Creating and saving prediction...")
generated_image = gan_model.predict(x_latent)
save_fig_inference(generated_image, SAMPLES_PER_ROW)
"""

# Load frozen graph using TensorFlow 1.x functions
with tf.io.gfile.GFile("./freeze_graph.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(f.read())

# Wrap frozen graph to ConcreteFunctions
frozen_func = wrap_frozen_graph(graph_def=graph_def,
                                inputs=["serving_default_dense_1_input:0"],
                                outputs=["StatefulPartitionedCall:0"],
                                print_graph=True)