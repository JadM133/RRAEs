# Welcome

This repository allows users to train and manipulate Equinox models easily, specifically, Autoencoders.

The library provides trainor classes that allow to train Neural Networks in one line using JAX.

It also provides easy ways to do normalization, and vectorization of matrices during training.

There are also pre-built Autoencoder models, specifically Rank Reduction Autoencoders (RRAEs).

# What are RRAEs?

RRAEs or Rank reduction autoencoders are autoencoders include an SVD in the latent space to regularize the bottleneck.

This library presents all the required classes for creating customized RRAEs and training them (other architectures such as Vanilla AEs, IRMAEs and LoRAEs are also available).

Each script is an example of how to train a different model.

To simply train an MLP (from equinox), try [this](general-MLP.py)

To train an RRAE on curves (1D) using an MLP, refer to [this file](main-MLP.py)
To train an RRAE on curves (1D) using an Convolutions, refer to [this file](main-CNN1D.py)
To train an RRAE on images, refer to [this file](main-CNN.py)
To train a VRRAE on images, refer to [this file](main-var-CNN.py)
To train with an adaptive bottleneck size refer to [this](main-adap-MLP.py) and [this] file(main-adap-CNN.py)

For examples of post-processing and what RRAE trainors can do, refer to [this file](post_process.py)

# General instruction for preparing your own data

In RRAEs.utilities, there's a function called get_data that can import many datasets to test.

If you want to generate your own dataset, you will have to define the following:

x_train: Train input (refer to each script to see the shape)
x_test: Test input (refer to each script to see the shape)
p_train: None (if you don't have any parameters, otherwise, 
            these can be used for interpolation in the latent space)
p_test: Same as p_train
y_train: = x_train for autoencoders
y_test: = x_test for autoencoders
pre_func_inp: lambda x:x (if not needed, this is a function to be applied
                          on batches if memory is not enough to apply
                          over whole dataset)
pre_func_out: lambda x:x (same as above but for output)
kwargs: {} (any other kwargs you might need)

# Installation

pip install RRAEs

# Using the Library in MATLAB

The library is not coded in MATLAB, so we highly recommend that you use the python codes. However, if you would like to simply get predictions using RRAEs in MATLAB, you can run [MATLAB_runner.m](MATLAB_runner.m) and follow the instructions there.

NOTE: The MATLAB code is not regularly maintained so use it carefully.

