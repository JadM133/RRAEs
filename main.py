from AE_classes import (
    Strong_RRAE_MLPs,
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Weak_RRAE_MLPs,
    Vanilla_AE_MLP,
    Vanilla_AE_CNN,
    IRMAE_MLP,
    LoRAE_MLP,
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
    for prob in ["mult_gausses"]:
        problem = prob
        method = "LoRAE"
        loss_func = "nuc"

        latent_size = 2800
        k_max = 4

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
            folder=f"{problem}/{method}_{problem}/",
            file=f"{method}_{problem}",
            # linear_l=2,
            post_proc_func=inv_func,
            key=jrandom.PRNGKey(0),
        )
        kwargs = {
            "step_st": [1500, 1500, 1500],
            "batch_size_st": [20, 20, 20, 20],
            "lr_st": [1e-3, 1e-4, 1e-5],
            "print_every": 100,
            "loss_kwargs": {"lambda_nuc": 0.001},
            # "mul_lr":[0.81, 0.81, 0.81, 1],
            # "mul_lr_func": lambda tree: (tree.v_vt.vt,),
        }

        trainor.fit(
            x_train,
            y_train,
            y_train_o,
            loss_func=loss_func,
            training_key=jrandom.PRNGKey(50),
            **kwargs,
        )
        e0, e1, e2, e3 = trainor.post_process(
            y_test, y_test_o, None, p_train, p_test, modes="all"
        )
        
        trainor.save(p_train=p_train, p_test=p_test)
        # trainor.plot_results(ts=jnp.arange(0, y_test.shape[0], 1), ts_o=ts)
    pdb.set_trace()
