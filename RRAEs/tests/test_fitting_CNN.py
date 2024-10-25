import jax.random as jrandom
import pytest
from RRAEs.AE_classes import (
    Strong_RRAE_CNN,
    Vanilla_AE_CNN,
    Weak_RRAE_CNN,
)
import jax.numpy as jnp
import equinox as eqx
from RRAEs.training_classes import AE_Trainor_class


@pytest.mark.parametrize(
    "model_cls, sh, lf",
    [
        (Strong_RRAE_CNN, (20, 20, 10), None),
        (Vanilla_AE_CNN, (2, 2, 10), None),
        (Weak_RRAE_CNN, (12, 12, 10), "Weak"),
    ],
)
def test_fitting(model_cls, sh, lf):
    x = jrandom.normal(jrandom.PRNGKey(0), sh)
    trainor = AE_Trainor_class(
        x,
        model_cls,
        in_size=x.shape[0],
        data_size=x.shape[-1], # only required for the Weak
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
            loss_func=lf,
            verbose=False,
            training_key=jrandom.PRNGKey(50),
            **kwargs,
        )
    except Exception as e:
        assert False, f"Fitting failed with the following exception {repr(e)}"
