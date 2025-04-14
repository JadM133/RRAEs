from RRAEs.AE_classes import (
    Strong_RRAE_CNN,
    VAR_Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Vanilla_AE_CNN,
    IRMAE_CNN,
    LoRAE_CNN,
    VAR_AE_CNN,
)
from RRAEs.training_classes import RRAE_Trainor_class, Trainor_class  # , Trainor_class
import jax.random as jrandom
from RRAEs.utilities import get_data
import numpy as np

# Redirect print statements to logging
class PrintLogger:
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        if message.strip():  # Avoid logging empty messages
            self.logger.info(message.strip())

    def flush(self):
        pass  # This is needed for compatibility with sys.stdout


if __name__ == "__main__":
    # Step 1: Get the data - replace this with your own data of the same shape.
    problem = "2d_gaussian_shift_scale"
    (
        x_train,
        x_test,
        p_train,
        p_test,
        y_train,
        y_test,
        pre_func_inp,
        pre_func_out,
        args,
    ) = get_data(problem, train_size=10, test_size=1)

    print(
        f"Shape of data is {x_train.shape} (C x D0 x D1 x Ntr) and {x_test.shape} (C x D0 x D1 x Nt)"
    )

    # Step 2: Specify the model to use.
    method = "VAR_Strong" # switch this to "VAE" for VAE

    match method:
        case "VAR_Strong":
            model_cls = VAR_Strong_RRAE_CNN
        case "VAE":
            model_cls = VAR_AE_CNN


    loss_type = (
        "VAR_Strong"  # Switch this to "var" for VAE
    )
    match loss_type:
        case "VAR_Strong":
            eps_fn = lambda bs: np.random.normal(0, 1/bs, size=(1, 1, bs, bs))
        case "var":
            eps_fn = lambda bs: np.random.normal(size=(1, 1, latent_size, bs))

    # Step 3: Specify the archietectures' parameters:
    latent_size = 100  # latent space dimension, this L in VRRAEs, and the bottleneck in VAE
    k_max = (
        16  # This is the bottleneck for VRRAEs, useless for VAEs
    )

    # Step 4: Define your trainor, with the model, data, and parameters.
    # Use RRAE_Trainor_class for the Strong RRAEs, and Trainor_class for other architetures.
    trainor = RRAE_Trainor_class(  # switch this to Trainor_class for VAE
        x_train,
        model_cls,
        latent_size=latent_size,
        height=x_train.shape[1],
        width=x_train.shape[2],
        channels=x_train.shape[0],
        k_max=k_max,
        folder=f"{problem}/{method}_{problem}/",
        file=f"{method}_{problem}.pkl",
        norm_in="None",
        norm_out="None",
        out_train=x_train,
        kwargs_enc={
            "width_CNNs": [32, 64],
            "CNNs_num": 2,
            "kernel_conv": 3,
            "stride": 2,
            "padding": 1,
        },
        kwargs_dec={
            "width_CNNs": [256, 128, 32, 8],
            "CNNs_num": 4,
            "kernel_conv": 3,
            "stride": 2,
            "padding": 1,
        },
        key=jrandom.PRNGKey(500),
    )

    # Step 5: Define the kw arguments for training. When using the Strong RRAE formulation,
    # you need to specify training kw arguments (first stage of training with SVD to
    # find the basis), and fine-tuning kw arguments (second stage of training with the
    # basis found in the first stage).
    training_kwargs = {
        "flush": True,
        "step_st": [2],
        "batch_size_st": [300],
        "lr_st": [1e-3, 1e-5, 1e-8],
        "print_every": 1,
        "loss_type": loss_type,
        "loss_kwargs": {"beta": 0.001}, # Only used for VAE
        "eps_fn": eps_fn
    }


    ft_kwargs = {
        "flush": True,
        "step_st": [2],
        "batch_size_st": [300],
        "lr_st": [1e-3, 1e-6, 1e-7, 1e-8],
        "print_every": 1,
        "eps_fn": eps_fn
    }


    # Step 6: Train the model and get the predictions.
    trainor.fit(
        x_train,
        y_train,
        training_key=jrandom.PRNGKey(500),
        pre_func_inp=pre_func_inp,
        pre_func_out=pre_func_out,

        # The following two lines are to be used with RRAE_Trainor_class only.
        training_kwargs=training_kwargs,
        ft_kwargs=ft_kwargs,

        # The following line is to be used with Trainor_class only. (for VAEs)
        # **training_kwargs
    )

    trainor.save_model()
    preds = trainor.evaluate(
        x_train, y_train, x_test, y_test, None, pre_func_inp, pre_func_out
    )


    # Uncomment the following line if you want to hold the session to check your
    # results in the console.
    # pdb.set_trace()
