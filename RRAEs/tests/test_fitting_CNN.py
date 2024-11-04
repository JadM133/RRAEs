import jax.random as jrandom
import pytest
from RRAEs.AE_classes import (
    Strong_RRAE_CNN,
    Vanilla_AE_CNN,
    Weak_RRAE_CNN,
    IRMAE_CNN,
    LoRAE_CNN,
)
import jax.numpy as jnp
import equinox as eqx
from RRAEs.training_classes import RRAE_Trainor_class, Trainor_class


@pytest.mark.parametrize(
    "model_cls, sh, lf",
    [
        (Vanilla_AE_CNN, (2, 2, 10), "default"),
        (Weak_RRAE_CNN, (12, 12, 10), "Weak"),
        (IRMAE_CNN, (5, 5, 5), "default"),
        (LoRAE_CNN, (16, 16, 10), "nuc"),
    ],
)
def test_fitting(model_cls, sh, lf):
    x = jrandom.normal(jrandom.PRNGKey(0), sh)
    trainor = Trainor_class(
        x,
        model_cls,
        in_size=x.shape[0],
        data_size=x.shape[-1],  # only required for the Weak
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


def test_Strong_fitting():
    sh = (20, 20, 10)
    model_cls = Strong_RRAE_CNN
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
