from RRAEs.AE_classes import (
    Strong_RRAE_MLP,
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Weak_RRAE_MLP,
    Vanilla_AE_MLP,
    Vanilla_AE_CNN,
    IRMAE_MLP,
    LoRAE_MLP,
)
import jax.nn as jnn
from RRAEs.training_classes import Trainor_class, Objects_Interpolator_nD
import jax.random as jrandom
import pdb
import equinox as eqx
import jax.numpy as jnp
from RRAEs.utilities import find_weighted_loss, get_data, plot_welding
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    for prob in ["antenne"]:
        problem = prob
        method = "Strong"
        loss_func = "Strong"

        latent_size = 4000
        k_max = 2

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

        print(f"Shape of data is {x_train.shape} (T x Ntr) and {x_test.shape} (T x Nt)")
        print(f"method is {method}")

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

        interpolation_cls = Objects_Interpolator_nD
        trainor = Trainor_class(
            model_cls,
            interpolation_cls,
            data=x_train,
            latent_size=latent_size,  # 4600
            k_max=k_max,
            folder=f"{problem}_var/{method}_{problem}/",
            file=f"{method}_{problem}",
            variational=False,
            # linear_l=2,
            key=jrandom.PRNGKey(0),
        )
        kwargs = {
            "step_st": [2000, 2000, 2000],
            "batch_size_st": [20, 20, 20],
            "lr_st": [1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
            "print_every": 100,
            "loss_kwargs": {"lambda_nuc": 0.001},
            "kwargs_dec": {"final_activation": jnn.tanh},
            "kwargs_enc": {"depth": 4},
            # "mul_lr":[0.81, 0.81, 0.81, 1],
            # "mul_lr_func": lambda tree: (tree.v_vt.vt,),
        }
        trainor.fit(
            x_train,
            y_train,
            loss_func=loss_func,
            training_key=jrandom.PRNGKey(50),
            **kwargs,
        )
        e0, e1, e2, e3 = trainor.post_process(
            y_train_o, y_test, y_test_o, None, p_train, p_test, inv_func, modes=k_max
        )
        trainor.save(p_train=p_train, p_test=p_test)
        # trainor.plot_results(ts=jnp.arange(0, y_test.shape[0], 1), ts_o=ts)
    pdb.set_trace()
