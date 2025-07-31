# Welcome

This repository allows users to train and manipulate Equinox models easily

Note: If you're interested in interpolation using MATLAB, refer to [this folder](MATLAB_impl)

# What are RRAEs?

RRAEs or Rank reduction autoencoders are autoencoders include an SVD in the latent space to regularize the bottleneck.

This library presents all the required classes for creating customized RRAEs and training them (other architectures such as IRMAE and LoRAE are also available).

To reproduce the results of the VRRAE paper, please refer to main-var-CNN.py

# Help in installation

This is to help people with no previous experience with github (pre-requisites, have python and pip installed, and have access to the library using an SSH Key).
1. Create a folder locally (on your PC), with any name you want.
2. Using an IDE (e.g. Visual Studio code), or your terminal, change the directory into your new folder (using ``cd`` command), and create a python virtual environment by using the following command ``python3 -m venv .venv``. Note that you might need to write the specific version of python (e.g. ``python3.18 -m ...``). This will create a folder named ``.venv`` inside your new folder, where you should install all your python libraries later on for this project.
3. Activate your virtual environment. This will be different depending on the operating system, but in all cases, you should have a ``(venv)`` sign in your terminal to the left after the following command:
  -  On a Mac or Linux: ``source .venv/bin/activate``.
  -  On Windows: ``.venv/Scripts/activate``.
4. Now when you ``pip install`` something, it is installed in your ``.venv`` folder. You can install the library by doing ``pip install git+https://github.com/JadM133/RRAEs.git`` (note: you have to have access to the library since it is private).
5. To make sure everything went well, run the tests! To do so, start by installing pytest as follows ''pip install -U pytest''. Then execute the following:
  -  On a Mac: ``pytest ./.venv/lib/3.XX/site-packages/RRAEs/tests/``. where XX is your python version.
  -  On Windows: ``pytest .\.venv\Lib\site-packages\RRAEs\tests\``.
If all the tests pass, you're good to go!
6. Give it a try, the best place to start is the [jupyter notebook](tutorial.ipynb).

# Using the Library in MATLAB

The library is not coded in MATLAB, so we highly recommend that you use the python codes. However, if you would like to simply get predictions using RRAEs in MATLAB, you can run [MATLAB_runner.m](MATLAB_runner.m) and follow the instructions there.


----

### Create sparse repo (Load only required libraries)

````
git clone -n --depth=1 --filter=tree:0 https://github.com/JadM133/RRAEs.git

cd RRAEs

git sparse-checkout set --no-cone /RRAEs

git checkout main
````



