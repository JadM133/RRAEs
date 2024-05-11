import jax.random as jrandom
import pytest
from AE_classes import (
    Strong_RRAE_MLPs,
    Vanilla_AE_MLP,
    Weak_RRAE_MLPs,
    Strong_RRAE_CNN,
    Weak_RRAE_CNN,
    Vanilla_AE_CNN,
    IRMAE_MLP,
    LoRAE_MLP,
)
import jax.numpy as jnp


@pytest.mark.parametrize("dim_D", (10, 15, 50))
@pytest.mark.parametrize("latent", (200, 400, 800))
@pytest.mark.parametrize("num_modes", (1, 2, 6))
class Test_AEs_shapes:
    def test_Strong_MLP(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
        model = Strong_RRAE_MLPs(x, latent, num_modes, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        y = model.perform_in_latent(y)
        _, sing_vals, _ = jnp.linalg.svd(y, full_matrices=False)
        assert sing_vals[num_modes + 1] < 1e-5
        assert y.shape == (latent, dim_D)
        assert model.decode(y).shape == (500, dim_D)

    def test_Vanilla_MLP(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
        model = Vanilla_AE_MLP(x, latent, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        x = model.decode(y)
        assert x.shape == (500, dim_D)

    def test_Weak_MLP(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
        model = Weak_RRAE_MLPs(x, latent, num_modes, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        x = model.decode(y)
        assert x.shape == (500, dim_D)
        assert model.v_vt.v.shape == (latent, num_modes)
        assert model.v_vt.vt.shape == (num_modes, dim_D)

    def test_IRMAE_MLP(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
        model = IRMAE_MLP(x, latent, linear_l=2, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        assert len(model._encode.mlp.layers_l) == 2
        x = model.decode(y)
        assert x.shape == (500, dim_D)

    def test_LoRAE_MLP(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
        model = LoRAE_MLP(x, latent, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        assert len(model._encode.mlp.layers_l) == 1
        x = model.decode(y)
        assert x.shape == (500, dim_D)

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


@pytest.mark.parametrize("width,depth", [(64, 2), ([2, 4, 6], 3)])
class Test_MLPs_width:
    def test_MLP(self, width, depth):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, 15))
        true_width = width if isinstance(width, list) else [width] * depth

        model = Strong_RRAE_MLPs(
            x, 200, 3, width_enc=width, depth_enc=depth, key=jrandom.PRNGKey(0)
        )
        try:
            model(x)
        except:
            raise ValueError("Width is not correctly specified")
        assert all(
            [
                model._encode.mlp.layers[i].weight.shape[0] == true_width[i]
                for i in range(depth)
            ]
        )

        model = Weak_RRAE_MLPs(
            x, 200, 3, width_dec=width, depth_dec=depth, key=jrandom.PRNGKey(0)
        )
        model(x)
        assert all(
            [
                model._decode.mlp.layers[i].weight.shape[0] == true_width[i]
                for i in range(depth)
            ]
        )
