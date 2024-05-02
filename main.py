from AE_classes import (
    Strong_RRAE_MLPs,
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Weak_RRAE_MLPs,
    Vanilla_AE_MLP,
    Vanilla_AE_CNN,
)
from training_classes import Trainor_class, Objects_Interpolator_nD
import jax.random as jrandom
import pdb
import equinox as eqx
import jax.numpy as jnp
from utilities import find_weighted_loss, get_data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    problem = "shift"
    (
        ts,
        x_train,
        x_test,
        p_vals,
        p_test,
        inv_func,
        y_train_o,
        y_test_o,
        y_train,
        y_test,
    ) = get_data(problem)

    print(f"Shape of data is {x_train.shape} and {x_test.shape}")
    model_cls = Strong_RRAE_MLPs
    interpolation_cls = Objects_Interpolator_nD
    trainor = Trainor_class(
        model_cls,
        interpolation_cls,
        data=x_train,
        latent_size=500,
        k_max=1,
        key=jrandom.PRNGKey(0),
    )

    kwargs = {
        "step_st": [2000, 2000],
        "depth_enc": 1,
        "width_enc": 20,
        "depth_dec": 6,
        "width_dec": 64,
        "batch_size_st": [-1, -1, -1],
        "lr_st": [1e-3, 1e-4, 1e-5, 1e-6],
        "print_every": 10,
    }

    trainor.fit(
        x_train,
        y_train,
        y_train_o,
        loss_func="Strong",
        training_key=jrandom.PRNGKey(50),
        **kwargs,
    )
    e0, e1, e2, e3 = trainor.post_process(y_test, None, p_vals, p_test)
    pdb.set_trace()
