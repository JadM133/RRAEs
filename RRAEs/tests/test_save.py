import jax.random as jrandom
import pytest
from RRAEs.AE_classes import Strong_RRAE_CNN
import jax.numpy as jnp
import equinox as eqx
from RRAEs.training_classes import RRAE_Trainor_class
from RRAEs.utilities import find_weighted_loss
import jax.tree_util as jtu
import jax

def test_fitting(): # Only to test if saving/loading is causing a problem
    data = jrandom.normal(jrandom.key(0), (28, 28, 1))
    model_cls = Strong_RRAE_CNN

    trainor = RRAE_Trainor_class(
        data,
        model_cls,
        in_size=data.shape[0],
        latent_size=200,
        k_max=2,
        norm_type="minmax",
        key=jrandom.PRNGKey(0),
    )

    trainor.save("test_")
    new_trainor = RRAE_Trainor_class()
    new_trainor.load("test_", erase=True)
