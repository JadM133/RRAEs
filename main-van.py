from RRAEs.AE_classes import (
    Strong_RRAE_MLP,
    Weak_RRAE_MLP,
    Vanilla_AE_MLP,
    IRMAE_MLP,
    LoRAE_MLP,
    VAR_AE_MLP,
)
from RRAEs.training_classes import Trainor_class  # , Trainor_class
import jax.random as jrandom
import pdb
from RRAEs.utilities import get_data
import jax.numpy as jnp
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Step 1: Get the data - replace this with your own data of the same shape.
    all_errors = []
    for data_size in []: # [15, 150, 500, 800, 1500, 4000, 10000]:
        problem = "gaussian_shift"
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
        ) = get_data(problem, google=data_size)

        print(f"Shape of data is {x_train.shape} (T x Ntr) and {x_test.shape} (T x Nt)")

        # Step 2: Specify the model to use, Strong_RRAE_MLP is ours (recommended).
        method = "Vanilla"

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

        loss_type = "Strong"  # Specify the loss type, according to the model chosen.

        # Step 3: Specify the archietectures' parameters:
        latent_size = 1  # latent space dimension
        k_max = -1  # number of features in the latent space (after the truncated SVD).

        # Step 4: Define your trainor, with the model, data, and parameters.
        # Use RRAE_Trainor_class for the Strong RRAEs, and Trainor_class for other architetures.
        trainor = Trainor_class(
            x_train,
            model_cls,
            latent_size=latent_size,
            in_size=x_train.shape[0],
            k_max=k_max,
            folder=f"{problem}/",
            file=f"{method}_{problem}_{data_size}.pkl",
            norm_in="minmax",
            norm_out="minmax",
            out_train=x_train,
            key=jrandom.PRNGKey(0),
        )
        # Step 5: Define the kw arguments for training. When using the Strong RRAE formulation,
        # you need to specify training kw arguments (first stage of training with SVD to
        # find the basis), and fine-tuning kw arguments (second stage of training with the
        # basis found in the first stage).
        training_kwargs = {
            "step_st": [int(7680 * data_size / 64), int(4 * 7680 * data_size / 64 / 3)],
            "batch_size_st": [64, 64],
            "lr_st": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            "print_every": 100,
            "loss_type": loss_type,
        }

        # Step 6: Train the model and get the predictions.
        trainor.fit(
            x_train,
            y_train,
            training_key=jrandom.PRNGKey(50),
            pre_func_inp=pre_func_inp,
            pre_func_out=pre_func_out,
            **training_kwargs,
        )

        preds = trainor.evaluate(
            x_train, y_train, x_test, y_test, None, pre_func_inp, pre_func_out
        )
        interp_preds = trainor.AE_interpolate(p_train, p_test, x_train, x_test)
        all_errors.append(preds["error_test"])
        trainor.save()

    problem = "gaussian_shift"
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
    ) = get_data(problem, google=500)
    trainor = Trainor_class()
    trainor.load("gaussian_shift/Vanilla_gaussian_shift_500.pkl")
    pr = trainor.model(x_test)
    for i in range(pr.shape[-1]):
        plt.plot(pr[:, i])
        plt.show(block=False)
        plt.pause(0.1)
        plt.clf()

    print(all_errors)
    pdb.set_trace()
