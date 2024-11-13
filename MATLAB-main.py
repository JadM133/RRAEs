from RRAEs.AE_classes import *
from RRAEs.training_classes import RRAE_Trainor_class
import jax.random as jrandom
import pdb
from RRAEs.utilities import get_data
import sys
import numpy as np
import collections.abc


def n(arg, none_val=None):
    if arg == "None":
        print("The case")
        category = none_val
    else:
        category = arg
    return category


print(inp)
x_train = np.array(inp["x_train"])
x_test = np.array(inp["x_test"])
p_train = n(inp["p_train"])
p_test = n(inp["p_test"])
y_train = x_train
y_test = x_test

assert inp["run_type"] in ["MLP", "CNN"], "Invalid run type, choose MLP or CNN."

if inp["run_type"] == "MLP":
    print(f"Shape of data is {x_train.shape} (T x Ntr) and {x_test.shape} (T x Nt)")
else:
    print(
        f"Shape of data is {x_train.shape} (C x D x D x Ntr) and {x_test.shape} (C x D x D x Nt)"
    )

# Step 2: Specify the model to use, Strong_RRAE_MLP is ours (recommended).
method = inp["method"]
if inp["run_type"] == "MLP":
    match method:
        case "Strong":
            model_cls = Strong_RRAE_MLP
        case "Weak":
            model_cls = Weak_RRAE_MLP
        case "Vanilla":
            model_cls = Vanilla_AE_MLP
        case "IRMAE":
            model_cls = IRMAE_MLP
        case "LoRAE":
            model_cls = LoRAE_MLP
        case _:
            raise ValueError("Invalid method")
else:
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
        case _:
            raise ValueError("Invalid method")


loss_type = inp["loss_type"]  # Specify the loss type, according to the model chosen.

# Step 3: Specify the archietectures' parameters:
latent_size = int(inp["latent_size"])  # latent space dimension
k_max = int(
    inp["k_max"]
)  # number of features in the latent space (after the truncated SVD).

# Step 4: Define your trainor, with the model, data, and parameters.
# Use RRAE_Trainor_class for the Strong RRAEs, and Trainor_class for other architetures.
trainor = RRAE_Trainor_class(
    x_train,
    model_cls,
    latent_size=latent_size,
    in_size=x_train.shape[0],
    k_max=k_max,
    folder=inp["folder"],
    file=inp["file"],
    norm_in=inp["norm_in"],
    norm_out=inp["norm_in"],
    out_train=x_train,
    key=jrandom.PRNGKey(0),
)

# Step 5: Define the kw arguments for training. When using the Strong RRAE formulation,
# you need to specify training kw arguments (first stage of training with SVD to
# find the basis), and fine-tuning kw arguments (second stage of training with the
# basis found in the first stage).
training_kwargs = inp["training_kwargs"]

ft_kwargs = inp["ft_kwargs"]


# Step 6: Train the model and get the predictions.
def update_val_from_matlab(v):
    if isinstance(v, collections.abc.Sequence):
        v = np.array(v)
        v = np.ndarray.tolist(v)
        v = [int(val) for val in v]
        return v
    elif isinstance(v, float):
        return int(v)
    return v


training_kwargs = {k: update_val_from_matlab(v) for k, v in training_kwargs.items()}
ft_kwargs = {k: update_val_from_matlab(v) for k, v in ft_kwargs.items()}
trainor.fit(
    x_train,
    y_train,
    training_key=jrandom.PRNGKey(50),
    training_kwargs=training_kwargs,
    ft_kwargs=ft_kwargs,
)
if inp["find_preds"]:
    print("Finding preds...")
    preds = trainor.evaluate(x_train, y_train, x_test, y_test, p_train, p_test)
else:
    print("Finding preds is disabled.")
    preds = "None"
trainor.save()

# Uncomment the following line if you want to hold the session to check your
# results in the console.
# pdb.set_trace()
