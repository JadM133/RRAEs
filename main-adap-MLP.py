""" This file is an example of how to train an RRAE with an adaptive latent size dimension.

It is advised to take a look at main-MLP.py first if you haven't already, as some redundant details
are not explained here. """
import RRAEs.config
from RRAEs.AE_classes import RRAE_MLP
from RRAEs.training_classes import RRAE_Trainor_class, Trainor_class
from RRAEs.trackers import RRAE_gen_Tracker, RRAE_Null_Tracker, RRAE_pars_Tracker
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
    method = "RRAE"

    model_cls = RRAE_MLP

    loss_type = "RRAE"  # Specify the loss type, according to the model chosen.

    latent_size = 200  # latent space dimension 200

    # Step 3: Specify the initial truncation value. This is what will be your
    # initial latent space size, before being modified by a tracker.
    # Refer below for tips on how to choose this.
    k_max = 64 # In this case, choose k_max to be the min between latent_size and the batch_size

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
        "tracker": RRAE_gen_Tracker(k_init=k_max, patience_init=200)
    }

    # The tracker above will specify the adaptive scheme to be used. Gen means generic and it
    # is the algorithm that starts with the mlargest possible number of modes and starts decreasing
    # until stagnation. For this tracker, consider increasing a lot "step_st" since the algorithm
    # will break training by itself on stagnation.

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
