from AE_classes import (
    Strong_RRAE_MLPs,
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Weak_RRAE_MLPs,
    Vanilla_AE_MLP,
    Vanilla_AE_CNN,
    IRMAE,
    LoRAE,
)
from training_classes import Trainor_class, Objects_Interpolator_nD
import jax.random as jrandom
import pdb
import equinox as eqx
import jax.numpy as jnp
from utilities import find_weighted_loss, get_data
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    problem = "mult_gausses"
    method = "IRMAE"
    loss_func = "Strong"

    latent_size = 4500
    k_max = 6

    (
        ts,
        x_train,
        x_test,
        p_train,
        p_test,
        inv_func,
        y_train_o,
        y_test_o,
        y_train,
        y_test,
    ) = get_data(problem)

    print(f"Shape of data is {x_train.shape} and {x_test.shape}")
    print(f"method is {method}")

    match method:
        case "Strong":
            model_cls = Strong_RRAE_MLPs
        case "Weak":
            model_cls = Weak_RRAE_MLPs
        case "Vanilla":
            model_cls = Vanilla_AE_MLP
        case "IRMAE":
            model_cls = IRMAE
        case "LoRAE":
            model_cls = LoRAE

    interpolation_cls = Objects_Interpolator_nD
    trainor = Trainor_class(
        model_cls,
        interpolation_cls,
        data=x_train,
        latent_size=latent_size, # 4600
        k_max=k_max,
        folder=f"{problem}/{method}_{problem}/",
        file=f"{method}_{problem}",
        key=jrandom.PRNGKey(0),
    )

    kwargs = {
        "step_st": [2000, 2000, 2000, 2000],
        "depth_enc": 1,
        "width_enc": 64,
        "depth_dec": 6,
        "width_dec": 64,
        "batch_size_st": [20, 20, 20, 20],
        "lr_st": [1e-3, 1e-4, 1e-5, 1e-6],
        "print_every": 100,
        "loss_kwargs": {"lambda_nuc":0.001},
        # "mul_lr":[100, 100, 100],
        # "mul_lr_func": lambda tree: (tree.v_vt.vt,)
    }

    trainor.fit(
        x_train,
        y_train,
        None,
        loss_func=loss_func,
        training_key=jrandom.PRNGKey(50),
        **kwargs,
    )
    e0, e1, e2, e3 = trainor.post_process(y_test, None, None, p_train, p_test, modes="all")
    pdb.set_trace()
    trainor.save(ts=ts, p_train=p_train, p_test=p_test)
    pdb.set_trace()
