import equinox as eqx
import jax.numpy as jnp
import jax.nn as jnn
import jax
import jax.random as jrandom
from RRAEs.utilities import MLP_dropout
from RRAEs.AE_classes import (
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Vanilla_AE_CNN,
    IRMAE_CNN,
    LoRAE_CNN,
    CNN_Autoencoder,
)
from equinox._doc_utils import doc_repr
from RRAEs.utilities import get_data
from RRAEs.training_classes import Trainor_class, Objects_Interpolator_nD

_identity = doc_repr(lambda x, **kwargs: x, "lambda x: x")
import warnings
import pdb
from RRAEs.utilities import out_to_pic


if __name__ == "__main__":
    for i, method in enumerate(["Strong"]):
        problem = "mnist_"
        # method = "Vanilla"
        loss_func = "Strong"

        latent_size = 128
        k_max = 10

        folder = f"{problem}/{method}_{problem}/"
        file = f"{method}_{problem}"

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
            norm_func,
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

        interpolation_cls = Objects_Interpolator_nD

        # old_trainor = Trainor_class()
        # old_trainor.load("antenne_could_be/Strong_antenne/Strong_antenne")

        trainor = Trainor_class(
            model_cls,
            interpolation_cls,
            data=x_train,
            latent_size=latent_size,  # 4600
            k_max=k_max,
            folder=folder,
            file=file,
            post_proc_func=inv_func,
            # kwargs_enc={"CNN_widths": [32,], "CNNs_num": 1},
            # kwargs_dec={
            #     "CNN_widths": [32],
            #     "CNNs_num": 1,
            #     "kwargs_cnn": {"final_activation": lambda x: jnn.tanh(x+0.02)-0.01},
            # },
            key=jrandom.PRNGKey(0),
        )

        # trainor.set_model(old_trainor.model)

        kwargs = {
            "step_st": [100000],
            "batch_size_st": [20, 20, 20, 20],
            "lr_st": [1e-4],
            "print_every": 100,
            "loss_kwargs": {"lambda_nuc": 0.001},
            "kwargs_dec":{"kwargs_cnn":{"final_activation":jnn.tanh}},
            # "mul_lr":[10, 1, 1, 1],
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
        trainor.save()

    pdb.set_trace()
