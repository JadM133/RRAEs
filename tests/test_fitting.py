import jax.random as jrandom
import pytest
from AE_classes import (
    Strong_RRAE_MLPs,
    Vanilla_AE_MLP,
    Weak_RRAE_MLPs,
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Vanilla_AE_CNN,
    IRMAE_MLP,
    LoRAE_MLP,
)
import jax.numpy as jnp
import equinox as eqx
from training_classes import Trainor_class, Objects_Interpolator_nD
from utilities import find_weighted_loss


@pytest.mark.parametrize(
    "model_cls, sh, lf",
    [
        (Strong_RRAE_MLPs, (500, 10), None),
        (Vanilla_AE_MLP, (500, 10), None),
        (Weak_RRAE_MLPs, (500, 10), "Weak"),
        (IRMAE_MLP, (500, 10), None),
        (LoRAE_MLP, (500, 10), "nuc"),
        (Strong_RRAE_CNN, (500, 10, 10), None),
        (Vanilla_AE_CNN, (500, 10, 10), None),
        (Weak_RRAE_CNN, (500, 10, 10), "Weak"),
    ],
)
def test_fitting(model_cls, sh, lf):
    x = jrandom.normal(jrandom.PRNGKey(0), sh)
    interpolation_cls = Objects_Interpolator_nD
    trainor = Trainor_class(
        model_cls,
        interpolation_cls,
        data=x,
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
