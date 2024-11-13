print("GOT TO IMPORT")
from RRAEs.AE_classes import (
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Vanilla_AE_CNN,
    IRMAE_CNN,
    LoRAE_CNN,
)
from RRAEs.training_classes import RRAE_Trainor_class
import jax.random as jrandom
import pdb
from RRAEs.utilities import get_data
import jax.nn as jnn

if __name__ == "__main__":
    # Step 1: Get the data - replace this with your own data of the same shape.
    problem = "CelebA"
    (
        x_train,
        x_test,
        p_train,
        p_test,
        y_train,
        y_test,
        pre_func_inp,
        pre_func_out,
        kwargs,
    ) = get_data(problem, folder="../")
    
    # C is channels, D is the dimensions of the image (only same length and width
    # are supported), and Ntr is the number of training samples.
    print(f"Shape of data is {x_train.shape} (C x D x D x Ntr) and {x_test.shape}.")

    # Step 2: Specify the model to use, Strong_RRAE_MLP is ours (recommended).
    method = "Strong"
    match method:
        case "Strong":
            model_cls = Strong_RRAE_CNN
        case "Weak":
            model_cls = Weak_RRAE_CNN
        case "Vanilla":
            model_cls = Vanilla_AE_CNN
        case "IRMAE":
            model_cls = IRMAE_CNN
        case "LoRAE":
            model_cls = LoRAE_CNN

    loss_type = "Strong"  # Specify the loss type, according to the model chosen.

    # Step 3: Specify the archietectures' parameters:
    latent_size = 1000  # latent space dimension
    k_max = 128  # number of features in the latent space (after the truncated SVD).

    # Step 4: Define your trainor, with the model, data, and parameters.
    # Use RRAE_Trainor_class for the Strong RRAEs, and Trainor_class for other architetures.
    trainor = RRAE_Trainor_class(
        x_train,
        model_cls,
        latent_size=latent_size,
        data_size=x_train.shape[1],
        channels=x_train.shape[0],
        pre_func_inp=pre_func_inp,
        pre_func_out=pre_func_out,
        k_max=k_max,
        folder=f"test",
        file=f"{method}_{problem}_test.pkl",
        norm_in="None",
        norm_out="None",
        out_train=x_train,
        kwargs_dec={
            "final_activation": jnn.sigmoid
        },  # this is how you change the final activation
        key=jrandom.PRNGKey(0),
    )

    # Step 5: Define the kw arguments for training. When using the Strong RRAE formulation,
    # you need to specify training kw arguments (first stage of training with SVD to
    # find the basis), and fine-tuning kw arguments (second stage of training with the
    # basis found in the first stage).
    training_kwargs = {
        "step_st": [1,], # aprox 30 epoch (30*202000/256)
        "batch_size_st": [256],
        "lr_st": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        "print_every": 20,
        "save_every": 789,
    }

    ft_kwargs = {
        "step_st": [1,],
        "batch_size_st": [256],
        "lr_st": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        "print_every": 20,
        "save_every": 789,
    }

    # Step 6: Train the model and get the predictions.
    trainor.fit(
        x_train,
        y_train,
        training_key=jrandom.PRNGKey(50),
        training_kwargs=training_kwargs,
        ft_kwargs=ft_kwargs,
    )
    # preds = trainor.evaluate(x_train, y_train, x_test, y_test, p_train, p_test)
    trainor.save(kwargs=kwargs)

    # Uncomment the following line if you want to hold the session to check your
    # results in the console.
    # pdb.set_trace()
