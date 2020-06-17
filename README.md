<h2 align="center"> Assignment - InstaLOD </h2>
<b> Author: </b> Eduardo Alvarado <br> <br> 
In this project, I have design a Generative Adversarial Network (GAN) for digital digits generation (MNIST).
A Discriminator network will be trained in order to differentiate between real digits from the original dataset, and
fake digits, generated artificially by the Generator.
The Generator will be trained in conjunction with the Discriminator (GAN model) to generate more accurate (fake) digits
over time.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
## Table of Contents

- [Project Structure](#structure)
- [Introduction](#introduction)
- [Install](#install)
- [Feedback](#feedback)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Project Structure

```bash
HRAssignment-Repository
├── generated_images_training
│   ├── generated_images_epoch_001.png
│   ├── generated_images_epoch_002.png
│   ...
├── keras_model
│   ├── generator_model_010.h5
│   ├── generator_model_020.h5
│   ...
├── training.py
├── inference.py
├── README.md
└── .gitignore
```


## Introduction

<h4> 1. Design network with Python API </h4>

<h5> a. Training </h5>

For this part, I will be using Tensorflow and Keras to build our network. 
The file <code>training.py</code> contains the definition for the Discriminator and Generator networks, as well as
the GAN model containing both.

First, it loads the data from MNIST, shuffles them and creates real sets. Then, it uses the original version of
the Generator to generate fake sets from random gaussian noise. These two sets are merged into one single dataset 
(containing half of each). The dataset is used to train the Discriminator and teach him how to differentiate between
real and fake digits.

Then, we updated the Generator weights by training the GAN model (containing both, Generator and Discriminator). This 
leads to a better version of the Generator that takes into account the loss caused by the Discriminator. 
If the Discriminator loss reduces, the Generator gets updated more; If the loss of the Discriminator is high (meaning
that it does not recognize between fake and real), the Generator is more preserved.

During training, generated (fake) digits per epoch are stored in <code>generated_images_training</code>, and a trained
model <code>.h5</code> is saved each 10 epochs.

<h5> b. Inference </h5>

Finally, we save our model into a graph (<code>.pb</code>), so we can load it after, for example, in the 
<code>inference.py</code> file. In this file, we just need to load the <code>.pb</code> model (GAN), containing the network's
architecture and weights, and generate some random latent space to provide it as an input. 
Finally, the GAN model will generate a set of fake digits based on MNIST, and save them in the folder 
<code>generated_images_inference</code>.

<h4> 2. Porting model to standalone executable in C++ </h4>

For constructing our final <code>.pb</code> graph, we have used our Tensorflow Python API. However, as the assignment 
explains, these pre-trained graphs can be loaded into the C++ API to convert them to standalone applications.

To setup the compilation environment, I required the use of <code>Bazel</code> for compilation, and cloning the
<code>Tensorflow repository</code>, in order to build from source.

The steps I have followed to build the C++ executable are:

* Install <code>Bazel</code>. Here, I had some compatibility problems with its version along with the Tensorflow
version (2.1.0). If the Bazel version was the incorrect, running Bazel would give me an error while compiling.

* Clone <code>Tensorflow</code> along with all the submodules: <code> git clone --recursive https://github.com/tensorflow/tensorflow </code>

* Then, I have created a folder inside the cloned repository containing my project (in this case, 
<code>tensorflow/tensorflow/my_project</code>). Inside, my goal was to create a file, such as <code>loader.cc</code>, 
in order to read the graph that we exported during training.

* Besides, I had to create a <code>BUILD</code> file to define for <code>Bazel</code> which file to compile.

* Once we have everything at its place, we can build <code>Tensorflow</code> from source, by typing
<code>./configure</code> from the root of the repo.

* Then, inside the project folder we run <code>bazel build :loader</code>

* Error!

## Install

## Feedback