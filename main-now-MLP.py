from RRAEs.AE_classes import (
    Strong_RRAE_MLP,
    Weak_RRAE_MLP,
    Vanilla_AE_MLP,
    IRMAE_MLP,
    LoRAE_MLP,
    VAR_AE_MLP,
)
from RRAEs.training_classes import RRAE_Trainor_class, Trainor_class  # , Trainor_class
import jax.random as jrandom
import pdb
from RRAEs.utilities import get_data
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import os
from RRAEs.trackers import RRAE_gen_Tracker, RRAE_pars_Tracker, RRAE_Null_Tracker
import jax.nn as jnn


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
    all_errors = []
    all_stds = []
    for data_size in [12]:
        _10_errors = []
        for j in range(1):
            problem = "skf_ft"
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
            ) = get_data(problem, google=data_size, folder="")

            # Split the training data into training and validation sets
            # n = 69  # Number of validation samplesx_val.shape
            # key = jrandom.PRNGKey(10)
            # indices = jrandom.choice(key, x_train.shape[1], shape=(n,), replace=False)
            # x_val = x_train[:, indices]
            # y_val = y_train[:, indices]
            # x_train = jnp.delete(x_train, indices, axis=1)
            # y_train = jnp.delete(y_train, indices, axis=1)

            print(
                f"Shape of data is {x_train.shape} (T x Ntr) and {x_test.shape} (T x Nt)"
            )

            # Step 2: Specify the model to use, Strong_RRAE_MLP is ours (recommended).
            method = "Sparse"

            match method:
                case "Strong":
                    model_cls = Strong_RRAE_MLP
                case "Weak":
                    model_cls = Weak_RRAE_MLP
                case "Vanilla":
                    model_cls = Vanilla_AE_MLP
                case "Sparse":
                    model_cls = Vanilla_AE_MLP
                case "Contractive":
                    model_cls = Vanilla_AE_MLP
                case "IRMAE":
                    model_cls = IRMAE_MLP
                case "LoRAE":
                    model_cls = LoRAE_MLP
                case "VAE":
                    model_cls = VAR_AE_MLP
                case "Long":
                    model_cls = Vanilla_AE_MLP

            loss_type = (
                "Sparse"  # Specify the loss type, according to the model chosen.
            )

            # Step 3: Specify the archietectures' parameters:
            latent_size = 200  # latent space dimension 200
            k_max = 7  # number of features in the latent space (after the truncated SVD).
            
            adap_type = "None"

            log_dir = f"{problem}/{method}_{problem}_{adap_type}"
            
            
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            else:
                with open(
                    f"{problem}/{method}_{problem}_{adap_type}/log_{data_size}.txt", "w"
                ) as log_file:
                    log_file.write("")
            logging.basicConfig(
                filename=f"{problem}/{method}_{problem}_{adap_type}/log_{data_size}.txt",
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )

            # Step 4: Define your trainor, with the model, data, and parameters.
            # Use RRAE_Trainor_class for the Strong RRAEs, and Trainor_class for other architetures.

            trainor = Trainor_class(
                x_train,
                model_cls,
                latent_size=latent_size,
                in_size=x_train.shape[0],
                k_max=k_max,
                folder=f"{problem}/{method}_{problem}_{adap_type}/",
                file=f"{method}_{problem}_{data_size}.pkl",
                norm_in="minmax",
                norm_out="minmax",
                kwargs_enc={
                    "width_size": 300,
                    "depth": 1,
                    "final_activation": lambda x: jnn.sigmoid(x)
                },
                kwargs_dec={
                    "width_size": 300,
                    "depth": 6,
                },
                out_train=x_train,
                key=jrandom.PRNGKey(0),
                adap_type=adap_type,
            )

            # Step 5: Define the kw arguments for training. When using the Strong RRAE formulation,
            # you need to specify training kw arguments (first stage of training with SVD to
            # find the basis), and fine-tuning kw arguments (second stage of training with the
            # basis found in the first stage).
            training_kwargs = {
                "step_st": [60000],  # 7680*data_size/64
                "batch_size_st": [64, 64, 64, 64, 64],
                "lr_st": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                "print_every": 1,
                "loss_type": loss_type,
                "loss_kwargs": {"sparsity":0.01}
                # "tracker": RRAE_Null_Tracker(k_max),
            }

            ft_kwargs = {
                "step_st": [1000],
                "batch_size_st": [64],
                "lr_st": [1e-4, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
                "print_every": 1,
            }

            # Step 6: Train the model and get the predictions.
            trainor.fit(
                x_train,
                y_train,
                training_key=jrandom.PRNGKey(50),
                # training_kwargs=training_kwargs,
                # ft_kwargs=ft_kwargs,
                pre_func_inp=pre_func_inp,
                pre_func_out=pre_func_out,
                **training_kwargs
            )

            preds = trainor.evaluate(
                x_train, y_train, x_test, y_test, None, pre_func_inp, pre_func_out
            )

            # interp_preds = trainor.AE_interpolate(p_train, p_test, x_train, x_test)
            _10_errors.append(preds["error_test_o"])
        all_errors.append(np.mean(_10_errors))
        all_stds.append(np.std(_10_errors))
        trainor.save_model()

    # Copy the script to the logging folder
    script_path = os.path.abspath(__file__)
    destination_path = os.path.join(
        log_dir, os.path.basename(script_path + f"_{data_size}")
    )
    with open(script_path, "r") as src_file:
        with open(destination_path, "w") as dest_file:
            dest_file.write(src_file.read())
    # Uncomment the following line if you want to hold the session to check your
    # results in the console.
    # pdb.set_trace()

    # problem = "gaussian_shift"
    # (
    #     x_train,
    #     x_test,
    #     p_train,
    #     p_test,
    #     y_train,
    #     y_test,
    #     pre_func_inp,
    #     pre_func_out,
    #     args,
    # ) = get_data(problem, google=500)
    # trainor = RRAE_Trainor_class()
    # trainor.load("gaussian_shift/Strong_gaussian_shift_500.pkl")
    # pr = trainor.model(x_test, apply_basis=trainor.basis)
    # # for i in range(pr.shape[-1]):
    # #     plt.plot(pr[:, i])
    # #     plt.show(block=False)
    # #     plt.pause(0.1)
    # #     plt.clf()
    # # print(all_errors)
    # pdb.set_trace()
