import jax.random as jrandom
import pytest
from RRAEs.AE_classes import Strong_RRAE_CNN
import jax.numpy as jnp
import equinox as eqx
from RRAEs.training_classes import RRAE_Trainor_class
from RRAEs.utilities import find_weighted_loss
import jax.tree_util as jtu
import jax

def test_save(): # Only to test if saving/loading is causing a problem
    data = jrandom.normal(jrandom.key(0), (1, 28, 28, 1))
    model_cls = Strong_RRAE_CNN

    trainor = RRAE_Trainor_class(
        data,
        model_cls,
        latent_size=100,
        data_size=data.shape[1],
        channels=data.shape[0],
        k_max=2,
        key=jrandom.PRNGKey(0),
    )

    trainor.save("test_")
    new_trainor = RRAE_Trainor_class()
    new_trainor.load("test_", erase=True)
    try:
        pr = new_trainor(data[..., 0:1])
    except:
        raise ValueError("Save FAILED")

