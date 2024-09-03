import jax.random as jrandom
import pytest
from RRAEs.AE_classes import (
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Vanilla_AE_CNN,
)
import jax.numpy as jnp


@pytest.mark.parametrize("dim_D", (10, 17, 149))
@pytest.mark.parametrize("latent", (200,))
@pytest.mark.parametrize("num_modes", (1,))
@pytest.mark.parametrize("num_samples", (10, 100))
class Test_AEs_shapes:
    def test_Strong_CNN(self, latent, num_modes, dim_D, num_samples):
        x = jrandom.normal(jrandom.PRNGKey(0), (dim_D, dim_D, num_samples))
        kwargs = {"kwargs_dec": {"stride": 2}}
        model = Strong_RRAE_CNN(x, latent, num_modes, key=jrandom.PRNGKey(0), **kwargs)
        y = model.encode(x)
        assert y.shape == (latent, num_samples)
        y = model.latent(x)
        _, sing_vals, _ = jnp.linalg.svd(y, full_matrices=False)
        assert sing_vals[num_modes + 1] < 1e-5
        assert y.shape == (latent, num_samples)
        assert model.decode(y).shape == (dim_D, dim_D, num_samples)

    def test_Vanilla_CNN(self, latent, num_modes, dim_D, num_samples):
        x = jrandom.normal(jrandom.PRNGKey(0), (dim_D, dim_D, num_samples))
        kwargs = {"kwargs_dec": {"stride": 2}}
        model = Vanilla_AE_CNN(x, latent, key=jrandom.PRNGKey(0), **kwargs)
        y = model.encode(x)
        assert y.shape == (latent, num_samples)
        x = model.decode(y)
        assert x.shape == (dim_D, dim_D, num_samples)

    def test_Weak_CNN(self, latent, num_modes, dim_D, num_samples):
        x = jrandom.normal(jrandom.PRNGKey(0), (dim_D, dim_D, num_samples))
        kwargs = {"kwargs_dec": {"stride": 2}}
        model = Weak_RRAE_CNN(x, latent, num_modes, key=jrandom.PRNGKey(0), **kwargs)
        y = model.encode(x)
        assert y.shape == (latent, num_samples)
        x = model.decode(y)
        assert x.shape == (dim_D, dim_D, num_samples)
        assert model.v_vt.v.shape == (latent, num_modes)
        assert model.v_vt.vt.shape == (num_modes, num_samples)
