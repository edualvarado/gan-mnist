<h2 align="center"> Assignment - InstaLOD </h2>
<b> Author: </b> Eduardo Alvarado <br> <br> 
In this project, I have design a Generative Adversarial Network (GAN) for digital digits generation (MNIST).
A Discriminator network will be trained in order to differentiate between real digits from the original dataset, and
fake digits, generated artificially by the Generator.
The Generator will be trained in conjunction with the Discriminator (GAN model) to generate more accurate (fake) digits
over time.

Then, the network should be moved to an C++ executable, in which the graph from the network could be loaded to perform inference.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Project Structure](#structure)
- [Development](#development)
- [Feedback](#feedback)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Project Structure

```bash
HRAssignment-Repository
├── generated_images_training
│   ├── generated_images_epoch_001.png
│   ├── generated_images_epoch_002.png
├── generated_images_inference
│   ├── generated_image_inference.png
│   ...
├── keras_model
│   ├── generator_model_010.h5
│   ├── generator_model_020.h5
├── generator_model_final
│   ├── saved_model.pb
│   ├── variables
│	├──...
├── training.py
├── inference.py
├── README.md
└── .gitignore
```


## Development

<h4> 1. Design network with Tensorflow Python API </h4>

<h5> a. Training </h5>

For this part, I will be using Tensorflow and Keras to build our network. 
The file <code>training.py</code> contains the definition for the Discriminator and Generator networks, as well as the GAN model containing both.

First, we load the data from MNIST, shuffle it and create a set with real data. Additionally, we use the Generator to create fake sets from random gaussian noise. These two sets are merged into one single dataset (containing half of each). The resulting dataset is used to train the Discriminator and teach him how to differentiate between real and fake digits.

Second, we update the Generator weights by training the GAN model (containing both, Generator and Discriminator). This leads to a better version of the Generator that takes into account the loss caused by the Discriminator. If the Discriminator loss reduces (it recognizes succesfully between fake and real), the Generator gets updated more; If the loss of the Discriminator is high (meaning
that it does not recognize between fake and real), the Generator weights are more preserved.

During training, generated (fake) digits per epoch are stored in <code>generated_images_training</code>, and a trained model <code>.h5</code> is saved each 10 epochs.

Finally after training, we write in the disk a final graph in <code>generator_model_final</code>.

<h5> b. Inference </h5>

We can now load our graph-model for executing inference in the <code>inference.py</code> file. We load the <code>.pb</code> model (GAN), containing the network's architecture and weights, and generate some random latent space (with certain number of dimensions) to provide it as an input. The GAN model will generate a set of fake digits based on MNIST, and save them in the folder <code>generated_images_inference</code>.

<h4> 2. Deploy model to standalone executable in C++ </h4>

For constructing our final <code>.pb</code> graph, we have used the Tensorflow Python API. However, as the assignment explains, these pre-trained graphs can be loaded to standalone applications across different OS using the Tensorflow C++ API.

To setup the environment, I required cloning the <code>Tensorflow repository</code>, build it from sourcet and using <code>Bazel</code> for compiling the library. However, I had to deal with some problems during the installation, mainly because of incompatibilities between the versions of TensorFlow and Bazel. Finally, I could start the compilation using: <code>Bazel-3.1.0</code> and <code>Tensorflow 2.1 (master branch)</code>.

The steps I have followed to build the C++ executable are:

* Install <code>Bazel-3.1.0</code>. Compatibilities with Tensorflow can be a problem. By looking to <code>configure.py</code> at the root directory of the Tensorflow repository, one can find which are the minimum and maximum supported versions of Bazel. [Link for the installation](https://docs.bazel.build/versions/master/install-ubuntu.html).


* Clone <code>Tensorflow</code> along with all the submodules: <code> git clone --recursive https://github.com/tensorflow/tensorflow </code>. This will clone the correct Tensorflow version, so I can compile it afterwards, to get the C API headers/binaries.

* Build Tensorflow library with Bazel. In the root directory of Tensorflow, I run:

```
bazel test -c opt tensorflow/tools/lib_package:libtensorflow_test
bazel build -c opt tensorflow/tools/lib_package:libtensorflow_test
```

The process of building the library starts. However, I always got an error during the compilation, leading to an unsuccesfully built of the library.













* Then, I have created a folder inside the cloned repository containing my project (in this case, 
<code>tensorflow/tensorflow/my_project</code>). Inside, my goal was to create a file, such as <code>loader.cc</code>, 
in order to read the graph that I exported during training.

* Besides, I had to create a <code>BUILD</code> file to define for <code>Bazel</code> which file to compile.

* Once I have everything at its place, I can build <code>Tensorflow</code> from source, by typing
<code>./configure</code> from the root of the repo.

* Then, inside the project folder I run <code>bazel build :loader</code>

* Error!


## Feedback

By looking to my task, I realize that the most complicated part for me was encapsulating the code to C++ and using the C++ API to build a standalone file. Although I have experience dealing with graphs and pre-trained models using the Python API, I still learn about their deployment using the C/C++ API. 
