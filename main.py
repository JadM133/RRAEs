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
    problem = "shift"
    method = "Weak"
    loss_func = "Weak"
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
        latent_size=240,
        k_max=2,
        folder=f"{problem}/{method}_{problem}/",
        file=f"{method}_{problem}",
        key=jrandom.PRNGKey(0),
    )

    kwargs = {
        "step_st": [2000, 2000, 1500],
        "depth_enc": 1,
        "width_enc": 64,
        "depth_dec": 6,
        "width_dec": 64,
        "batch_size_st": [-1, -1, -1],
        "lr_st": [1e-3, 1e-4, 1e-5, 1e-6],
        "print_every": 100,
        "loss_kwargs": {"lambda_nuc":0.001},
        "mul_lr":[100, 100, 100],
        "mul_lr_func": lambda tree: (tree.v_vt.vt,)
    }

    trainor.fit(
        x_train,
        y_train,
        None,
        loss_func=loss_func,
        training_key=jrandom.PRNGKey(50),
        **kwargs,
    )
    e0, e1, e2, e3 = trainor.post_process(y_test, None, None, p_train, p_test)
    trainor.save(ts=ts, p_train=p_train, p_test=p_test)
    new_trainor = Trainor_class()
    new_trainor.load(os.path.join(trainor.folder, trainor.file))
    new_trainor.plot_results()
    pdb.set_trace()
