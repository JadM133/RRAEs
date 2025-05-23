import jax.random as jrandom
import pytest
from RRAEs.AE_classes import (
    Strong_RRAE_MLP,
    Vanilla_AE_MLP,
    Weak_RRAE_MLP,
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
        model = Strong_RRAE_MLP(x.shape[0], latent, num_modes, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        y = model.perform_in_latent(y, k_max=num_modes)
        _, sing_vals, _ = jnp.linalg.svd(y, full_matrices=False)
        assert sing_vals[num_modes + 1] < 1e-5
        assert y.shape == (latent, dim_D)
        assert model.decode(y).shape == (500, dim_D)

    def test_Vanilla_MLP(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
        model = Vanilla_AE_MLP(x.shape[0], latent, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        x = model.decode(y)
        assert x.shape == (500, dim_D)

    # def test_Weak_MLP(self, latent, num_modes, dim_D):
    #     x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
    #     model = Weak_RRAE_MLP(x.shape[0], latent, num_modes, x.shape[-1], key=jrandom.PRNGKey(0))
    #     y = model.encode(x)
    #     assert y.shape == (latent, dim_D)
    #     x = model.decode(y)
    #     assert x.shape == (500, dim_D)
    #     assert model.v_vt.v.shape == (latent, num_modes)
    #     assert model.v_vt.vt.shape == (num_modes, dim_D)

    def test_IRMAE_MLP(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
        model = IRMAE_MLP(x.shape[0], latent, linear_l=2, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        assert len(model._encode.layers_l) == 2
        x = model.decode(y)
        assert x.shape == (500, dim_D)

    def test_LoRAE_MLP(self, latent, num_modes, dim_D):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, dim_D))
        model = LoRAE_MLP(x.shape[0], latent, key=jrandom.PRNGKey(0))
        y = model.encode(x)
        assert y.shape == (latent, dim_D)
        assert len(model._encode.layers_l) == 1
        x = model.decode(y)
        assert x.shape == (500, dim_D)

@pytest.mark.parametrize("width,depth", [(64, 2), ([2, 4, 6], 3)])
class Test_width:
    def test_MLP(self, width, depth):
        x = jrandom.normal(jrandom.PRNGKey(0), (500, 15))
        true_width = width if isinstance(width, list) else [width] * depth
        kwargs_enc={"width_size":width, "depth":depth}
        k_max = 3
        model = Strong_RRAE_MLP(
            x.shape[0], 200, k_max, key=jrandom.PRNGKey(0), kwargs_enc=kwargs_enc
        )
        try:
            model(x, k_max=k_max)
        except:
            raise ValueError("Width is not correctly specified")
        assert all(
            [
                model._encode.layers[i].weight.shape[0] == true_width[i]
                for i in range(depth)
            ]
        )

        # kwargs_dec={"width_size":width, "depth":depth}
        # model = Weak_RRAE_MLP(
        #     x.shape[0], 200, 3, x.shape[-1], key=jrandom.PRNGKey(0), kwargs_dec=kwargs_dec
        # )
        # model(x)
        # assert all(
        #     [
        #         model.decode.layers[i].weight.shape[0] == true_width[i]
        #         for i in range(depth)
        #     ]
        # )

def test_getting_SVD_coeffs():
    data = jrandom.uniform(jrandom.key(0), (500, 15))
    model_s = Strong_RRAE_MLP(data.shape[0], 200, 3, key=jrandom.PRNGKey(0))
    basis, coeffs = model_s.latent(data, get_basis_coeffs=True)

