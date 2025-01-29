""" This script contains an example of how to use a Trainor class for any equinox model. Specifically,
The MLP."""

from equinox.nn import MLP
from RRAEs.training_classes import Trainor_class
import jax.random as jrandom
import pdb
from RRAEs.utilities import get_data
import jax.numpy as jnp

if __name__ == "__main__":
    # Step 1: Get the data - in this case dummy data is generated.
    inp = jrandom.normal(jrandom.PRNGKey(0), (10, 80))
    out = jnp.expand_dims(jnp.sum(inp, axis=0) ** 2 / 4 + 3, 0)

    # The shape should be what's expected in your model as inputs and outputs.
    print(f"Shape of data is {inp.shape} (D1 x N) and {out.shape} (D2 x N)")

    # Step 2: Specify the model to use, do not declare an instance of the class.
    # i.e. do not open/close parenthesis.
    model_cls = MLP

    loss_type = "default"  # Specify the loss type, this uses the norm in %.

    # Step 3: Define your trainor, with the model, data, and parameters.
    # Use Trainor_class. It has some slight differences compared to RRAE_Trainor_class.
    trainor = Trainor_class(
        inp,
        model_cls,
        in_size=inp.shape[0],
        out_size=out.shape[0],
        width_size=100,
        depth=2,
        folder="folder_name/",
        file="saved_model.pkl",
        norm_in="None",
        norm_out="None",
        map_axis=-1, # The dimension of your data, to parallelize over.
        key=jrandom.PRNGKey(0),
    )

    # Step 4: Define the kw arguments for training. When using the Strong RRAE formulation,
    # you need to specify training kw arguments (first stage of training with SVD to
    # find the basis), and fine-tuning kw arguments (second stage of training with the
    # basis found in the first stage).
    kwargs = {
        "step_st": [2], # Increase those to train well
        "batch_size_st": [20, 20],
        "lr_st": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        "print_every": 100,
        "loss_type": loss_type,
    }

    # Step 5: Train the model and get the predictions.
    trainor.fit(
        inp,
        out,
        training_key=jrandom.PRNGKey(50),
        **kwargs,
    )
    preds = trainor.evaluate(inp, out)  # could give test as well as inp_test, out_test
    trainor.save()

    # Uncomment the following line if you want to hold the session to check your
    # results in the console.
    # pdb.set_trace()
