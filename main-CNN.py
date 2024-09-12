import equinox as eqx
import jax.numpy as jnp
import jax.nn as jnn
import jax
import jax.random as jrandom
from RRAEs.AE_classes import (
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Vanilla_AE_CNN,
    IRMAE_CNN,
    LoRAE_CNN,
)
from RRAEs.utilities import get_data
from RRAEs.training_classes import AE_Trainor_class, V_AE_Trainor_class

import pdb
from RRAEs.utilities import out_to_pic

from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)

if __name__ == "__main__":
    for i, method in enumerate(["Strong"]):
        problem = "mnist_"
        # method = "Vanilla"
        loss_func = "Strong"

        latent_size = 128
        k_max = 5

        folder = f"{problem}/{method}_{problem}/"
        file = f"{method}_{problem}_var"

        (
            x_train,
            x_test,
            p_train,
            p_test,
            y_train,
            y_test,
            args,
        ) = get_data(problem)

        print(f"Shape of data is {x_train.shape} and {x_test.shape}")
        print(f"method is {method}")

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

        trainor = AE_Trainor_class(
            x_train,
            model_cls,
            in_size=x_train.shape[0],
            latent_size=latent_size,  # 4600
            k_max=k_max,
            folder=folder,
            file=file,
            norm_in="minmax",
            norm_out="minmax",
            out_train=y_train,
            kwargs_dec={
                "final_activation": jnn.sigmoid
            },  # this is how you change the final activation
            key=jrandom.PRNGKey(0),
        )

        train_kwargs = {
            "step_st": [60000, 60000],
            "batch_size_st": [20, 20, 20, 20],
            "lr_st": [1e-4, 1e-5],
            "print_every": 100,
            "loss_kwargs": {"lambda_nuc": 0.001},
        }

        trainor.fit(
            x_train,
            y_train,
            loss_func=loss_func,
            training_key=jrandom.PRNGKey(50),
            **train_kwargs,
        )
        trainor.post_process(x_train, y_train, x_test, y_test, p_train, p_test)
        trainor.save()

    pdb.set_trace()
