import jax.random as jrandom
import pytest
from RRAEs.AE_classes import (
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Vanilla_AE_CNN,
    IRMAE_CNN,
    LoRAE_CNN,
)
import jax.numpy as jnp


@pytest.mark.parametrize("width", (10, 17, 149))
@pytest.mark.parametrize("height", (20,))
@pytest.mark.parametrize("latent", (200,))
@pytest.mark.parametrize("num_modes", (1,))
@pytest.mark.parametrize("channels", (1, 3, 5))
@pytest.mark.parametrize("num_samples", (10, 100))
class Test_AEs_shapes:
    def test_Strong_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = jrandom.normal(jrandom.PRNGKey(0), (channels, width, height, num_samples))
        kwargs = {"kwargs_dec": {"stride": 2}}
        model = Strong_RRAE_CNN(
            x.shape[0], x.shape[1], x.shape[2], latent, num_modes, key=jrandom.PRNGKey(0), **kwargs
        )
        y = model.encode(x)
        assert y.shape == (latent, num_samples)
        y = model.latent(x)
        _, sing_vals, _ = jnp.linalg.svd(y, full_matrices=False)
        assert sing_vals[num_modes + 1] < 1e-5
        assert y.shape == (latent, num_samples)
        assert model.decode(y).shape == (channels, width, height, num_samples)

    def test_Vanilla_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = jrandom.normal(jrandom.PRNGKey(0), (channels, width, height, num_samples))
        kwargs = {"kwargs_dec": {"stride": 2}}
        model = Vanilla_AE_CNN(
            x.shape[0], x.shape[1], x.shape[2], latent, key=jrandom.PRNGKey(0), **kwargs
        )
        y = model.encode(x)
        assert y.shape == (latent, num_samples)
        x = model.decode(y)
        assert x.shape == (channels, width, height, num_samples)

    def test_Weak_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = jrandom.normal(jrandom.PRNGKey(0), (channels, width, height, num_samples))
        kwargs = {"kwargs_dec": {"stride": 2}}
        model = Weak_RRAE_CNN(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            latent,
            num_modes,
            x.shape[-1],
            key=jrandom.PRNGKey(0),
            **kwargs
        )
        y = model.encode(x)
        assert y.shape == (latent, num_samples)
        x = model.decode(y)
        assert x.shape == (channels, width, height, num_samples)
        assert model.v_vt.v.shape == (latent, num_modes)
        assert model.v_vt.vt.shape == (num_modes, num_samples)

    def test_IRMAE_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = jrandom.normal(jrandom.PRNGKey(0), (channels, width, height, num_samples))
        model = IRMAE_CNN(
            x.shape[0], x.shape[1], x.shape[2], latent, key=jrandom.PRNGKey(0), linear_l=2
        )
        y = model.encode(x)
        assert y.shape == (latent, num_samples)
        assert len(model.encode.layers[-2].layers_l) == 2
        x = model.decode(y)
        assert x.shape == (channels, width, height, num_samples)

    def test_LoRAE_CNN(self, latent, num_modes, width, height, channels, num_samples):
        x = jrandom.normal(jrandom.PRNGKey(0), (channels, width, height, num_samples))
        model = LoRAE_CNN(x.shape[0], x.shape[1], x.shape[2], latent, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, num_samples)
        assert len(model.encode.layers[-2].layers_l) == 1
        x = model.decode(y)
        assert x.shape == (channels, width, height, num_samples)
