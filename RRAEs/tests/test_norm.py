import jax.random as jrandom
import pytest
from RRAEs.AE_classes import Strong_RRAE_CNN
import jax.numpy as jnp
import equinox as eqx
from RRAEs.training_classes import RRAE_Trainor_class
from RRAEs.utilities import find_weighted_loss


@pytest.mark.parametrize("norm_in", ["minmax", "meanstd", "None"])
@pytest.mark.parametrize("norm_out", ["minmax", "meanstd"])
def test_fitting(norm_in, norm_out):

    data = jrandom.normal(jrandom.key(0), (28, 28, 1))
    model_cls = Strong_RRAE_CNN

    trainor = RRAE_Trainor_class(
        data,
        model_cls,
        in_size=data.shape[0],
        latent_size=200,
        k_max=2,
        norm_in=norm_in,
        norm_out=norm_out,
        out_train=data,
        key=jrandom.PRNGKey(0),
    )

    if norm_in == "None":
        assert (trainor.model.norm_in(data) == data).all()
    elif norm_in == "minmax":
        assert (trainor.model.norm_in(data) <= 1).all()
        assert (trainor.model.norm_in(data) >= 0).all()
    elif norm_in == "meanstd":
        assert jnp.abs(jnp.mean(trainor.model.norm_in(data))) <= 1e-3
        assert jnp.abs(jnp.std(trainor.model.norm_in(data)) - 1) <= 1e-3

    if norm_out == "minmax":
        assert (trainor.model.norm_out(data) <= 1).all()
        assert (trainor.model.norm_out(data) >= 0).all()
    elif norm_out == "meanstd":
        assert jnp.abs(jnp.mean(trainor.model.norm_out(data))) <= 1e-3
        assert jnp.abs(jnp.std(trainor.model.norm_out(data)) - 1) <= 1e-3
    assert jnp.allclose(trainor.model.inv_norm_in(trainor.model.norm_in(data)), data, rtol=1e-2)
    assert jnp.allclose(trainor.model.inv_norm_out(trainor.model.norm_out(data)), data, rtol=1e-2)

