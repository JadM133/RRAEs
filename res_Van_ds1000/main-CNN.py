print("GOT TO IMPORT")
from RRAEs.AE_classes import (
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Vanilla_AE_CNN,
    IRMAE_CNN,
    LoRAE_CNN,
)
from RRAEs.training_classes import Trainor_class
import jax.random as jrandom
import pdb
from RRAEs.utilities import get_data
import jax.nn as jnn
import jax.sharding as jshard
import jax.experimental.mesh_utils as mesh_utils
import jax
from jax.sharding import PartitionSpec as P


if __name__ == "__main__":
    # Step 1: Get the data - replace this with your own data of the same shape.
    num_devices = len(jax.devices())
    devices = mesh_utils.create_device_mesh((1, 1, 1, num_devices))
    sharding = jshard.PositionalSharding(devices)
    replicated = sharding.replicate()
    print("GOT TO MAIN!!!")
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
        kwargs,
    ) = get_data(problem, folder="../")

    # C is channels, D is the dimensions of the image (only same length and width
    # are supported), and Ntr is the number of training samples.
    print(f"Shape of data is {x_train.shape} (C x D x D x Ntr).")
    # Step 2: Specify the model to use, Strong_RRAE_MLP is ours (recommended).
    # x_train = x_train*100
    # y_train = y_train*100
    method = "Vanilla"
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
    latent_size = 8  # latent space dimension
    k_max = -1  # number of features in the latent space (after the truncated SVD).

    # Step 4: Define your trainor, with the model, data, and parameters.
    # Use RRAE_Trainor_class for the Strong RRAEs, and Trainor_class for other architetures.
    trainor = Trainor_class(
        x_train,
        model_cls,
        latent_size=latent_size,
        data_size=x_train.shape[1],
        channels=x_train.shape[0],
        k_max=k_max,
        folder=f"test",
        file=f"{method}_{problem}_test.pkl",
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
            # "final_activation": lambda x: jnn.sigmoid(x),
        },
        key=jrandom.PRNGKey(50),
    )
    # Step 5: Define the kw arguments for training. When using the Strong RRAE formulation,
    # you need to specify training kw arguments (first stage of training with SVD to
    # find the basis), and fine-tuning kw arguments (second stage of training with the
    # basis found in the first stage).
    training_kwargs = {
        "step_st": [int(40000/60), int(40000/60)], # aprox 30 epoch (30*202000/256)
        "batch_size_st": [32, 32],
        "lr_st": [1e-4, 1e-5, 1e-5, 1e-6, 1e-7, 1e-8],
        "print_every": 1,
        # "save_every": 100,
        "sharding": sharding,
        "replicated": replicated,
        "loss_type": loss_type,
    }


    # Step 6: Train the model and get the predictions.
    trainor.fit(
        x_train,
        y_train,
        training_key=jrandom.PRNGKey(500),
        pre_func_inp=pre_func_inp,
        pre_func_out=pre_func_out,
        flush=True,
        **training_kwargs
    )
    # preds = trainor.evaluate(x_train, y_train, x_test, y_test, p_train, p_test)
    trainor.save(kwargs=kwargs)

    # Uncomment the following line if you want to hold the session to check your
    # results in the console.
    # pdb.set_trace()
