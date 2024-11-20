# How to run the RRAE Library using MATLAB (interpolation)

## Objective: Curves characterized by parameters, interpolate to find new curves for new parameter values.
In this readme, we explain how one can use RRAEs to perform interpolation using MATLAB (python running in background).

If the parametric space is small the code will allow you to interpolate linearly the latent space. Otherwise the code will guide you to train an MLP which relates the physical parameters to the latent space (so nonlinear interpolation, since interpolating linearly in N-d space is unfeasible).

## Step1: Run ``RRAE_training.m`` to train RRAEs

If your parametric space is small, you can perform linear interpolation in the same file (instructions inside the file).
Otherwise, continue the steps.

## Step2: Run ``Post_proc_model.m`` to get the SVD coefficients of the training latent space.

## Step3: Run ``MLP_regression.m`` to train an MLP to fit the latent coefficients using your physical parameters as inputs.

## Step4: Run ``final_processing.m`` where you can give a new parameter and get the reconstruction curve.

NOTE: More details are inside every file.

NOTE: For more flexibility we recommend using python.





