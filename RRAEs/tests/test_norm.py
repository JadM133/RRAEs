import jax.random as jrandom
import pytest
from RRAEs.AE_classes import Strong_RRAE_CNN
import jax.numpy as jnp
import equinox as eqx
from RRAEs.training_classes import Trainor_class, Objects_Interpolator_nD
from RRAEs.utilities import find_weighted_loss


@pytest.mark.parametrize(
    "norm_type",
    ["minmax", "meanstd", "None"],
)
def test_fitting(norm_type):

    data = jrandom.normal(jrandom.key(0), (28, 28, 1))
    model_cls = Strong_RRAE_CNN
    interpolation_cls = Objects_Interpolator_nD

    trainor = Trainor_class(
        model_cls,
        interpolation_cls,
        data=data,
        latent_size=200,
        k_max=2,
        norm_type=norm_type,
        key=jrandom.PRNGKey(0),
    )

    if norm_type == "None":
        assert (trainor.model.norm(data) == data).all()
    elif norm_type == "minmax":
        assert (trainor.model.norm(data) <= 1).all()
        assert (trainor.model.norm(data) >= 0).all()
        assert trainor.model.norm_cls.params.keys() == {"min", "max"}
    elif norm_type == "meanstd":
        assert jnp.abs(jnp.mean(trainor.model.norm(data))) <= 1e-3
        assert jnp.abs(jnp.std(trainor.model.norm(data)) - 1) <= 1e-3
        assert trainor.model.norm_cls.params.keys() == {"mean", "std"}
