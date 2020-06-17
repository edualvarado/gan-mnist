<h2 align="center"> Assignment - InstaLOD </h2>
<b> Author: </b> Eduardo Alvarado <br> <br> 
In this project, I have design a Generative Adversarial Network (GAN) for digital digits generation (MNIST).
A Discriminator network will be trained in order to differentiate between real digits from the original dataset, and
fake digits, generated artificially by the Generator.
The Generator will be trained in conjunction with the Discriminator (GAN model) to generate more accurate (fake) digits
over time. Then, the network should be moved to an C++ executable, in which the graph from the network could be loaded to perform inference.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Project Structure](#structure)
- [Development](#development)
- [Self-Feedback](#self-feedback)

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

For constructing our final <code>.pb</code> graph, we have used the Tensorflow Python API. However, as the assignment 
explains, these pre-trained graphs can be loaded to standalone applications across different OS using the Tensorflow 
C++ API.

To setup the environment, I required cloning the <code>Tensorflow repository</code>, build it from source and using 
<code>Bazel</code> for compiling the library. However, I had to deal with some problems during the installation, 
mainly because of incompatibilities between the versions of TensorFlow and Bazel. Finally, I could finish the compilation
installing Bazel using [Bazelisk](https://github.com/bazelbuild/bazelisk) , which installs the correct version based
on the Tensorflow distribution (in my case, 2.1).

The steps I have followed to build the C++ executable are:

* Install <code>Bazel</code> using <code>Bazelisk</code> for correct compatibility.
[Link for the installation](https://docs.bazel.build/versions/master/install-bazelisk.html).

* Clone <code>Tensorflow</code> along with all the submodules, and build from source using <code>Bazel</code>. I 
followed the following tutorial: [Source](https://www.tensorflow.org/install/source).

* Once Tensorflow is configured (<code>./configure</code>) and built 
(<code>bazel build //tensorflow/tools/pip_package:build_pip_package</code>), it will create an executable (in my case
named <code>build_pip_package</code>). By running 
<code>./bazel-bin/tensorflow/tools/pip_package/build_pip_package --nightly_flag /tmp/tensorflow_pkg</code> I was able
to create finally the <code>.whl</code>.

* After that, my goal was to adapt the already built graph from our network and deploy to C++ (for example following
these instructions). Unfortunately I did not have enough time to finish this part of the assignment.


## Self-Feedback

By looking to my task, I realize that the most complicated part for me was encapsulating the code to C++ and 
using the C++ API to build a standalone file. I required much time researching on how it is done and setting up the 
environment with Bazel and Tensorflow, due to compatibility errors between them during building. 
In the last moment, I realized about <code>Bazelisk</code> and its ability to manage different versions. After installing
<code>Bazelisk</code> in this way, building got succesfully done. It has been very important to learn more about 
these procedures, I am willing to know more, specially after this assignment. In the network part (python) I 
encountered less problems, just few regarding optimization. For the whole assignment, I dedicated around ~7 hours per day.


