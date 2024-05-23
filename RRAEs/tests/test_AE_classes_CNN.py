import jax.random as jrandom
import pytest
from RRAEs.AE_classes import (
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Vanilla_AE_CNN,
)
import jax.numpy as jnp


@pytest.mark.parametrize("dim_D", (10, 15, 50))
@pytest.mark.parametrize("latent", (200, 400, 800))
@pytest.mark.parametrize("num_modes", (1, 2, 6))
class Test_AEs_shapes:  
    def test_Strong_CNN(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, 500, dim_D))
        model = Strong_RRAE_CNN(x, latent, num_modes, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        y = model.perform_in_latent(y)
        _, sing_vals, _ = jnp.linalg.svd(y, full_matrices=False)
        assert sing_vals[num_modes + 1] < 1e-5
        assert y.shape == (latent, dim_D)
        assert model.decode(y).shape == (500, 500, dim_D)

    def test_Vanilla_CNN(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, 500, dim_D))
        model = Vanilla_AE_CNN(x, latent, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        x = model.decode(y)
        assert x.shape == (500, 500, dim_D)

    def test_Weak_CNN(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, 500, dim_D))
        model = Weak_RRAE_CNN(x, latent, num_modes, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        x = model.decode(y)
        assert x.shape == (500, 500, dim_D)
        assert model.v_vt.v.shape == (latent, num_modes)
        assert model.v_vt.vt.shape == (num_modes, dim_D)
