import jax
import jax.numpy as jnp
from RRAEs.utilities import (
    Sample,
    CNNs_with_MLP,
    MLP_with_CNNs_trans,
    CNN3D_with_MLP,
    MLP_with_CNN3D_trans,
    MLP_with_linear,
    stable_SVD,
)   
from RRAEs.wrappers import vmap_wrap
import equinox as eqx
import jax.random as jrandom
import warnings
from equinox.nn._linear import Linear
import equinox as eqx

def set_autoencoder_base(cls):
    global _AutoencoderBase
    _AutoencoderBase = cls

def get_autoencoder_base():
    global _AutoencoderBase
    return _AutoencoderBase or _default_autoencoder

def _default_autoencoder():
    class Autoencoder(eqx.Module):
        _encode: MLP_with_linear
        _decode: MLP_with_linear
        _perform_in_latent: callable
        _perform_in_latent: callable
        map_latent: bool
        norm_funcs: list
        inv_norm_funcs: list
        count: int

        def __init__(
            self,
            in_size,
            latent_size,
            latent_size_after=None,
            _encode=None,
            _decode=None,
            map_latent=True,
            *,
            key,
            count=1,
            kwargs_enc={},
            kwargs_dec={},
            **kwargs,
        ):
            key_e, key_d = jrandom.split(key)
            if latent_size_after is None:
                latent_size_after = latent_size

            if _encode is None:
                if "width_size" not in kwargs_enc.keys():
                    kwargs_enc["width_size"] = 64

                if "depth" not in kwargs_enc.keys():
                    kwargs_enc["depth"] = 1

                self._encode = MLP_with_linear(
                    in_size=in_size,
                    out_size=latent_size,
                    key=key_e,
                    **kwargs_enc,
                )

            else:
                self._encode = _encode

            if not hasattr(self, "_perform_in_latent"):
                self._perform_in_latent = lambda x, *args, **kwargs: x 

            if _decode is None:
                if "width_size" not in kwargs_dec.keys():
                    kwargs_dec["width_size"] = 64
                if "depth" not in kwargs_dec.keys():
                    kwargs_dec["depth"] = 6

                self._decode = MLP_with_linear(
                    in_size=latent_size_after,
                    out_size=in_size,
                    key=key_d,
                    **kwargs_dec,
                )
            else:
                self._decode = _decode

            self.count = count
            self.map_latent = map_latent
            self.inv_norm_funcs = ["decode"]
            self.norm_funcs = ["encode", "latent"]

        def encode(self, x, *args, **kwargs):
            return self._encode(x, *args, **kwargs)
        
        def decode(self, x, *args, **kwargs):
            return self._decode(x, *args, **kwargs)

        def perform_in_latent(self, y, *args, **kwargs):
            if self.map_latent:
                new_perform_in_latent = lambda x: self._perform_in_latent(
                    x, *args, **kwargs
                )
                for _ in range(self.count):
                    new_perform_in_latent = jax.vmap(new_perform_in_latent, in_axes=-1, out_axes=-1) 
                return new_perform_in_latent(y)
            return self._perform_in_latent(y, *args, **kwargs)

        def __call__(self, x, *args, **kwargs):
            return self.decode(self.perform_in_latent(self.encode(x, no_map=True), *args, **kwargs), no_map=True)

        def latent(self, x, *args, **kwargs):
            return self.perform_in_latent(self.encode(x, no_map=True), *args, **kwargs)

    return Autoencoder
