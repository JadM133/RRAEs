import jax.random as jrandom
import pytest
from RRAEs.AE_classes import (
    RRAE_MLP,
    Vanilla_AE_MLP,
    IRMAE_MLP,
    LoRAE_MLP,
)
import jax.numpy as jnp
from RRAEs.wrappers import vmap_wrap

methods = ["encode", "decode"]

v_RRAE_MLP = vmap_wrap(RRAE_MLP, -1, 1, methods)
v_Vanilla_AE_MLP = vmap_wrap(Vanilla_AE_MLP, -1, 1, methods)
v_IRMAE_MLP = vmap_wrap(IRMAE_MLP, -1, 1, methods)
v_LoRAE_MLP = vmap_wrap(LoRAE_MLP, -1, 1, methods)

@pytest.mark.parametrize("dim_D", (10, 15, 50))
@pytest.mark.parametrize("latent", (200, 400, 800))
@pytest.mark.parametrize("num_modes", (1, 2, 6))
class Test_AEs_shapes:
    def test_RRAE_MLP(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
        model = v_RRAE_MLP(x.shape[0], latent, num_modes, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        y = model.perform_in_latent(y, k_max=num_modes)
        _, sing_vals, _ = jnp.linalg.svd(y, full_matrices=False)
        assert sing_vals[num_modes + 1] < 1e-5
        assert y.shape == (latent, dim_D)
        assert model.decode(y).shape == (500, dim_D)

    def test_Vanilla_MLP(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
        model = v_Vanilla_AE_MLP(x.shape[0], latent, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        x = model.decode(y)
        assert x.shape == (500, dim_D)

    def test_IRMAE_MLP(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
        model = v_IRMAE_MLP(x.shape[0], latent, linear_l=2, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        assert len(model._encode.layers_l) == 2
        x = model.decode(y)
        assert x.shape == (500, dim_D)

    def test_LoRAE_MLP(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
        model = v_LoRAE_MLP(x.shape[0], latent, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        assert len(model._encode.layers_l) == 1
        x = model.decode(y)
        assert x.shape == (500, dim_D)

def test_getting_SVD_coeffs():
    data = jrandom.uniform(jrandom.key(0), (500, 15))
    model_s = v_RRAE_MLP(data.shape[0], 200, 3, key=jrandom.PRNGKey(0))
    basis, coeffs = model_s.latent(data, k_max=3, get_basis_coeffs=True)
    assert basis.shape == (200, 3)
    assert coeffs.shape == (3, 15)

