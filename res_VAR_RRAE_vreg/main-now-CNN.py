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
import pdb
from RRAEs.utilities import get_data
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import logging
import sys
import os
import jax.nn as jnn
from RRAEs.trackers import RRAE_gen_Tracker, RRAE_pars_Tracker, RRAE_Null_Tracker
import shutil
import jax.sharding as jshard
import jax.experimental.mesh_utils as mesh_utils
import jax
from jax.sharding import PartitionSpec as P
import equinox as eqx

# Redirect print statements to logging
class PrintLogger:
    def __init__(self, logger):
        self.logger = logger

    def write(self, message):
        if message.strip():  # Avoid logging empty messages
            self.logger.info(message.strip())

    def flush(self):
        pass  # This is needed for compatibility with sys.stdout

norm_loss_ = lambda pr, out: jnp.linalg.norm(pr-out)/jnp.linalg.norm(out)*100

@eqx.filter_value_and_grad(has_aux=True)
def loss_fun(diff_model, static_model, input, out, idx, k_max, **kwargs):
    model = eqx.combine(diff_model, static_model)
    pred = model(input, k_max=k_max, inv_norm_out=False)
    coeffs = model.latent(input, k_max=k_max, get_coeffs=True)
    loss_coeff = norm_loss_(coeffs, jnp.repeat(jnp.mean(coeffs, 1, keepdims=True), coeffs.shape[-1], 1))
    aux = {"loss": norm_loss_(pred, out), "loss_c":loss_coeff, "k_max":k_max}
    return norm_loss_(pred, out)+ 0.01*loss_coeff, aux

if __name__ == "__main__":
    num_devices = len(jax.devices())
    devices = mesh_utils.create_device_mesh((1, 1, 1, num_devices))
    sharding = jshard.PositionalSharding(devices)
    replicated = sharding.replicate()
    # Step 1: Get the data - replace this with your own data of the same shape.
    print("test", flush=True)
    all_errors = []
    all_stds = []
    for data_size in [None]:
        _10_errors = []
        for j in range(1):
            problem = "mnist"
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
            ) = get_data(problem, google=data_size, folder="../")

            print(
                f"Shape of data is {x_train.shape} (T x Ntr) and {x_test.shape} (T x Nt)"
            )

            # Step 2: Specify the model to use, Strong_RRAE_MLP is ours (recommended).
            method = "VAR_Strong"

            match method:
                case "VAR_Strong":
                    model_cls = VAR_Strong_RRAE_CNN
                case "Strong":
                    model_cls = Strong_RRAE_CNN
                case "Weak":
                    model_cls = Weak_RRAE_CNN
                case "Vanilla":
                    model_cls = Vanilla_AE_CNN
                case "Sparse":
                    model_cls = Vanilla_AE_CNN
                case "Contractive":
                    model_cls = Vanilla_AE_CNN
                case "IRMAE":
                    model_cls = IRMAE_CNN
                case "LoRAE":
                    model_cls = LoRAE_CNN
                case "VAE":
                    model_cls = VAR_AE_CNN
                case "Long":
                    model_cls = Vanilla_AE_CNN

            loss_type = (
                "Strong"  # Specify the loss type, according to the model chosen.
            )

            # Step 3: Specify the archietectures' parameters:
            latent_size = 100  # latent space dimension 200
            k_max = (
                16  # number of features in the latent space (after the truncated SVD).
            )

            adap_type = "None"

            log_dir = f"{problem}/{method}_{problem}_{adap_type}"
            dir_path = log_dir
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                # Remove the directory and its contents
                shutil.rmtree(dir_path)
                print(f"The directory {dir_path} has been removed.")
            else:
                print(f"The directory {dir_path} does not exist.")

            # Step 4: Define your trainor, with the model, data, and parameters.
            # Use RRAE_Trainor_class for the Strong RRAEs, and Trainor_class for other architetures.
            trainor = RRAE_Trainor_class(
                x_train,
                model_cls,
                latent_size=latent_size,
                height=x_train.shape[1],
                width=x_train.shape[2],
                channels=x_train.shape[0],
                k_max=k_max,
                folder=f"{problem}/{method}_{problem}_{adap_type}/",
                file=f"{method}_{problem}_{data_size}_{adap_type}.pkl",
                norm_in="None",
                norm_out="None",
                out_train=x_train,
                kwargs_enc={
                    "width_CNNs": [32, 64],
                    "CNNs_num": 2,
                    "kernel_conv": 3,
                    "stride": 2,
                    "padding": 1,
                    # "final_activation": lambda x: jnn.sigmoid(x)
                },
                kwargs_dec={
                    "width_CNNs": [256, 128, 32, 8],
                    "CNNs_num": 4,
                    "kernel_conv": 3,
                    "stride": 2,
                    "padding": 1,
                    "final_activation": lambda x: jnn.sigmoid(x), # x of shape (C, D, D)
                },
                key=jrandom.PRNGKey(500),
                adap_type=adap_type,
                # linear_l=8,
            )

            # Step 5: Define the kw arguments for training. When using the Strong RRAE formulation,
            # you need to specify training kw arguments (first stage of training with SVD to
            # find the basis), and fine-tuning kw arguments (second stage of training with the
            # basis found in the first stage).
            training_kwargs = {
                "flush": True,
                "step_st": [20000],  # 7680*data_size/64
                "batch_size_st": [32, 48],
                "lr_st": [1e-4, 1e-5, 1e-8],
                "print_every": 1,
                "loss_type": loss_fun,
                "sharding": sharding,
                "replicated": replicated,
                # "loss_kwargs": {
                #    "sparsity": 0.001,
                #    "beta": 100
                    # "find_layer": lambda model: model.encode.layers[-2].layers[-1].weight,
                #}
                # "loss_kwargs": {"beta": 0.0001, "find_weight": lambda model: model.encode.layers[-2].layers[0].weight},
                "tracker": RRAE_Null_Tracker(k_max), # , perf_loss=42),
            }

            ft_kwargs = {
                "flush": True,
                "step_st": [0],
                "batch_size_st": [32],
                "lr_st": [1e-5, 1e-6, 1e-7, 1e-8],
                "print_every": 1,
                "sharding": sharding,
                "replicated": replicated
            }

            # Step 6: Train the model and get the predictions.
            trainor.fit(
                x_train,
                y_train,
                training_key=jrandom.PRNGKey(500),
                training_kwargs=training_kwargs,
                ft_kwargs=ft_kwargs,
                pre_func_inp=pre_func_inp,
                pre_func_out=pre_func_out,
                # **training_kwargs
            )
            trainor.save_model()
            preds = trainor.evaluate(
                x_train, y_train, x_test, y_test, None, pre_func_inp, pre_func_out
            )
            # interp_preds = trainor.AE_interpolate(p_train, p_test, x_train, x_test)
            # _10_errors.append(preds["error_test_o"])
        # all_errors.append(np.mean(_10_errors))
        # all_stds.append(np.std(_10_errors))
        # trainor.save_model()

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
