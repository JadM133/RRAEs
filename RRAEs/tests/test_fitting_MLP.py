import jax.random as jrandom
import pytest
from RRAEs.AE_classes import (
    Strong_RRAE_MLP,
    Vanilla_AE_MLP,
    Weak_RRAE_MLP,
    IRMAE_MLP,
    LoRAE_MLP,
)
import jax.numpy as jnp
import equinox as eqx
from RRAEs.training_classes import RRAE_Trainor_class, Trainor_class


@pytest.mark.parametrize(
    "model_cls, sh, lf",
    [
        (Vanilla_AE_MLP, (500, 10), "default"),
        (Weak_RRAE_MLP, (500, 10), "Weak"),
        (IRMAE_MLP, (500, 10), "default"),
        (LoRAE_MLP, (500, 10), "nuc"),
    ],
)
def test_fitting(model_cls, sh, lf):
    x = jrandom.normal(jrandom.PRNGKey(0), sh)
    trainor = Trainor_class(
        x,
        model_cls,
        in_size=x.shape[0],
        data_size=x.shape[-1],
        samples=x.shape[-1],  # Only for weak
        norm_in="meanstd",
        norm_out="minmax",
        out_train=x,
        latent_size=2000,
        k_max=2,
        key=jrandom.PRNGKey(0),
    )
    kwargs = {
        "step_st": [2],
        "loss_kwargs": {"lambda_nuc": 0.001},
    }
    try:
        trainor.fit(
            x,
            x,
            loss_type=lf,
            verbose=False,
            training_key=jrandom.PRNGKey(50),
            **kwargs,
        )
    except Exception as e:
        assert False, f"Fitting failed with the following exception {repr(e)}"


@pytest.mark.skip(
    reason="modifying lr of specific pytree nodes is currently not possible."
)
def test_weak_mul_lr():
    try:
        model_cls = Weak_RRAE_MLP
        x_train = jrandom.normal(jrandom.PRNGKey(0), (500, 10))
        y_train = jrandom.normal(jrandom.PRNGKey(1), (500, 10))
        latent_size = 2000
        k_max = 2
        trainor_Weak = RRAE_Trainor_class(
            x_train,
            model_cls,
            latent_size=latent_size,  # 4600
            in_size=x_train.shape[0],
            data_size=x_train.shape[-1],  # Only needed for Weak_RRAE_MLP
            k_max=k_max,
            folder=f"test/",
            file=f"testing_Weak",
            norm_in="minmax",  # could choose "meanstd", or "None"
            norm_out="minmax",  # could choose "meanstd", or "None"
            out_train=x_train,
            key=jrandom.PRNGKey(100),
        )
        training_kwargs = {
            "step_st": [1500, 1500, 1500],  # number of batches strategy
            "batch_size_st": [20, 20, 20, 20],  # batch size strategy
            "lr_st": [1e-3, 1e-4, 1e-5],  # learning rate strategy
            "print_every": 100,
            "mul_lr": [0.05, 0.05, 0.05],  # The values of kappa (to multiply lr for A)
            "mul_lr_func": lambda tree: (
                tree.v_vt.vt,
            ),  # Who will be affected by kappa, this means A
        }
        _ = trainor_Weak.fit(
            x_train,
            y_train,
            loss_func="Weak",
            training_key=jrandom.PRNGKey(50),
            **training_kwargs,
        )
    except Exception as e:
        assert False, f"Can not change lr in Weak for the following reason: {repr(e)}"

def test_Strong_fitting():
    sh = (500, 10)
    model_cls = Strong_RRAE_MLP
    x = jrandom.normal(jrandom.PRNGKey(0), sh)
    trainor = RRAE_Trainor_class(
        x,
        model_cls,
        in_size=x.shape[0],
        latent_size=2000,
        k_max=2,
        key=jrandom.PRNGKey(0),
    )
    training_kwargs = {
        "step_st": [2],
    }
    ft_kwargs = {
        "step_st": [2],
    }
    try:
        trainor.fit(
            x,
            x,
            verbose=False,
            training_key=jrandom.PRNGKey(50),
            training_kwargs=training_kwargs,
            ft_kwargs=ft_kwargs,
        )
    except Exception as e:
        assert False, f"Fitting failed with the following exception {repr(e)}"