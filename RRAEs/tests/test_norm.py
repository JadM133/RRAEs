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
        data,
        model_cls,
        interpolation_cls,
        in_size=data.shape[0],
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
    elif norm_type == "meanstd":
        assert jnp.abs(jnp.mean(trainor.model.norm(data))) <= 1e-3
        assert jnp.abs(jnp.std(trainor.model.norm(data)) - 1) <= 1e-3
    assert jnp.allclose(trainor.model.inv_norm(trainor.model.norm(data)), data, rtol=1e-2)

