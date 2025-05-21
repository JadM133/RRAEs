import jax.random as jrandom
import pytest
from RRAEs.AE_classes import Strong_RRAE_CNN, Test_AE_for_Norm
import jax.numpy as jnp
import equinox as eqx
from RRAEs.training_classes import RRAE_Trainor_class, Trainor_class
from RRAEs.utilities import find_weighted_loss


@pytest.mark.parametrize("norm_in", ["minmax", "meanstd", "None"])
@pytest.mark.parametrize("norm_out", ["minmax", "meanstd"])
def test_fitting(norm_in, norm_out):

    data = jrandom.normal(jrandom.key(0), (3, 28, 28, 1))
    model_cls = Strong_RRAE_CNN

    trainor = RRAE_Trainor_class(
        data,
        model_cls,
        latent_size=100,
        channels=data.shape[0],
        width=data.shape[1],
        height=data.shape[2],
        k_max=2,
        norm_in=norm_in,
        norm_out=norm_out,
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


@pytest.mark.parametrize("norm_in", ["minmax", "meanstd", "None"])
@pytest.mark.parametrize("norm_out", ["minmax", "meanstd"])
def test_norm(norm_in, norm_out):

    data = jrandom.normal(jrandom.key(0), (500, 200))
    model_cls = Test_AE_for_Norm

    trainor = RRAE_Trainor_class(
        data,
        model_cls,
        latent_size=100,
        in_size=data.shape[0],
        norm_in=norm_in,
        norm_out=norm_out,
        key=jrandom.PRNGKey(0),
    )

    if norm_in == "None":
        assert (trainor.model.encode(data) == data).all()
        assert (trainor.model.latent(data) == data).all()
        
    elif norm_in == "minmax":
        assert (trainor.model.encode(data) <= 1).all()
        assert (trainor.model.encode(data) >= 0).all()
        assert (trainor.model.latent(data) <= 1).all()
        assert (trainor.model.encode(data) >= 0).all()

    elif norm_in == "meanstd":
        assert jnp.abs(jnp.mean(trainor.model.encode(data))) <= 1e-3
        assert jnp.abs(jnp.std(trainor.model.encode(data)) - 1) <= 1e-3

        assert jnp.abs(jnp.mean(trainor.model.latent(data))) <= 1e-3
        assert jnp.abs(jnp.std(trainor.model.latent(data)) - 1) <= 1e-3

    if norm_out == norm_in:
        assert jnp.allclose(trainor.model(data), data, rtol=1e-2)
