""" This file is an example of how to train an RRAE with an adaptive latent size dimension.

It is advised to take a look at main-MLP.py first if you haven't already, as some redundant details
are not explained here. """

from RRAEs.AE_classes import (
    Strong_RRAE_MLP,
    Vanilla_AE_MLP,
    IRMAE_MLP,
    LoRAE_MLP,
    VAR_AE_MLP,
)
from RRAEs.training_classes import RRAE_Trainor_class, Trainor_class
import jax.random as jrandom
from RRAEs.utilities import get_data


if __name__ == "__main__":
    # Step 1: Get the data - replace this with your own data of the same shape.
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
    ) = get_data(problem)

    print(f"Shape of data is {x_train.shape} (T x Ntr) and {x_test.shape} (T x Nt)")

    # Step 2: Specify the model to use, Strong_RRAE_MLP is ours (recommended).
    method = "Strong"

    model_cls = Strong_RRAE_MLP

    loss_type = "Strong"  # Specify the loss type, according to the model chosen.

    latent_size = 200  # latent space dimension 200

    # Step 3: Specify the initial truncation value. This is what will be your
    # initial latent space size, before being modified by a tracker.
    # Refer below for tips on how to choose this.
    k_max = 10  # In this case

    # Step 4: Specify your adaptive scheme, choose one of "None", "pars", and "gen".
    # These mean the following:

    # 1- "None": Fixed scheme, i.e. the k_max you specify above will remain fixed. Use
    # this wen you're sure of the dimensionality of the latent space that you want to use.

    # 2- "pars": Parsimonious scheme, starts with the k_max specified above, and tries
    # to decrease the loss as much as possible before incrementing k_max until convergence.
    # In this case, use a small value of k_max, 1 is usually a suitable choice.

    # 3- "gen": Generic scheme, starts with k_max specified above, and removes modes with
    # time to converge to the optimal value of k_max. In this case, use a big value of k_max,
    # usually, k_max=batch_size is a suitbale choice.
    adap_type = "gen"

    # Step 4: Define your trainor, with the model, data, and parameters.
    # Use RRAE_Trainor_class for the Strong RRAEs, and Trainor_class for other architetures.
    trainor = RRAE_Trainor_class(
        x_train,
        model_cls,
        latent_size=latent_size,
        in_size=x_train.shape[0],
        k_max=k_max,
        folder=f"{problem}/{method}_{problem}/",
        file=f"{method}_{problem}.pkl",
        norm_in="minmax",
        norm_out="minmax",
        kwargs_enc={
            "width_size": 300,
            "depth": 1,
        },
        kwargs_dec={
            "width_size": 300,
            "depth": 6,
        },
        out_train=x_train,
        key=jrandom.PRNGKey(0),
        adap_type=adap_type
    )

    # Step 5: Define the kw arguments for training. When using the Strong RRAE formulation,
    # you need to specify training kw arguments (first stage of training with SVD to
    # find the basis), and fine-tuning kw arguments (second stage of training with the
    # basis found in the first stage).
    training_kwargs = {
        "step_st": [2], # Increase those to train better
        "batch_size_st": [64, 64, 64, 64, 64],
        "lr_st": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        "print_every": 1,
        "loss_type": loss_type,
    }

    ft_kwargs = {
        "step_st": [2], # Increase those to train better
        "batch_size_st": [64],
        "lr_st": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        "print_every": 1,
    }

    # Step 6: Train the model and get the predictions.
    trainor.fit(
        x_train,
        y_train,
        training_key=jrandom.PRNGKey(50),
        training_kwargs=training_kwargs,
        ft_kwargs=ft_kwargs,
        pre_func_inp=pre_func_inp,
        pre_func_out=pre_func_out,
    )

    preds = trainor.evaluate(
        x_train, y_train, x_test, y_test, None, pre_func_inp, pre_func_out
    )

    # Uncomment the following line if you want to hold the session to check your
    # results in the console.
    # pdb.set_trace()
