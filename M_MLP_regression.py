from RRAEs.AE_classes import *
from RRAEs.training_classes import Trainor_class
import jax.random as jrandom
import pdb
import sys
import numpy as np
import scipy.io as sio
from functools import partial
from utilities_for_MATLAB import *
from equinox.nn import MLP


print = partial(print, flush=True)

if __name__ == "__main__":
    inp_file = sys.argv[1]
    mat_contents = sio.loadmat(inp_file)
    inp = mat_contents["mlp"][0, 0]
    x_train = np.array(inp["input_train"])
    y_train = np.array(inp["output_train"])
    x_test = np.array(inp["input_test"])
    y_test = np.array(inp["output_test"])
    loss_type = s(inp["loss_type"])

    mlp_kwargs = from_void_to_dict(inp["mlp_kwargs"])
    mlp_kwargs = {k: update_val_from_matlab(k, v) for k, v in mlp_kwargs.items()}

    trainor = Trainor_class(
        x_train,
        MLP,
        in_size=x_train.shape[0],
        out_size=y_train.shape[0],
        folder=s(inp["folder"]),
        file=s(inp["file"]),
        norm_in=s(inp["norm_in"]),
        norm_out=s(inp["norm_in"]),
        key=jrandom.PRNGKey(0),
        map_axis=-1,
        **mlp_kwargs
    )

    training_kwargs = from_void_to_dict(inp["training_kwargs"])

    training_kwargs = {
        k: update_val_from_matlab(k, v) for k, v in training_kwargs.items()
    }

    training_kwargs["flush"] = True

    trainor.fit(x_train, y_train, training_key=jrandom.PRNGKey(50), **training_kwargs)
    if inp["find_preds"]:
        print("Finding preds...")
        preds = trainor.evaluate(x_train, y_train, x_test, y_test)
    else:
        print("Finding preds is disabled.")
        preds = "None"
    trainor.save()
    print(preds)

    # Uncomment the following line if you want to hold the session to check your
    # results in the console.
    # pdb.set_trace()
