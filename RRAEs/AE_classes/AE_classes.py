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

_identity = lambda x, *args, **kwargs: x

def latent_func_strong_RRAE(
    self,
    y,
    k_max=None,
    apply_basis=None,
    get_basis_coeffs=False,
    get_coeffs=False,
    get_right_sing=False,
    ret=False,
    *args,
    **kwargs,
):
    """Performing the truncated SVD in the latent space.

    Parameters
    ----------
    y : jnp.array
        The latent space.
    k_max : int
        The maximum number of modes to keep. If this is -1,
        the function will return y (i.e. all the modes).

    Returns
    -------
    y_approx : jnp.array
        The latent space after the truncation.
    """
    if apply_basis is not None:
        if get_basis_coeffs:
            return apply_basis, apply_basis.T @ y
        if get_coeffs:
            if get_right_sing:
                raise ValueError("Can not find right singular vector when projecting on basis")
            if get_right_sing:
                raise ValueError("Can not find right singular vector when projecting on basis")
            return apply_basis.T @ y
        return apply_basis @ apply_basis.T @ y

    k_max = -1 if k_max is None else k_max

    if get_basis_coeffs or get_coeffs:
        u, s, v = stable_SVD(y)

        if isinstance(k_max, int):
            k_max = [k_max]

        u_now = [u[:, :k] for k in k_max]
        coeffs = [jnp.multiply(v[:k, :], jnp.expand_dims(s[:k], -1)) for k in k_max]

        if len(k_max) == 1:
            u_now = u_now[0]
            coeffs = coeffs[0]

        if get_coeffs:
            if get_right_sing:
                return v[:k_max, :]
            if get_right_sing:
                return v[:k_max, :]
            return coeffs
        return u_now, coeffs

    if k_max != -1:
        u, s, v = stable_SVD(y)

        if k_max is None:
            raise ValueError("k_max was not given when truncation is required.")

        if isinstance(k_max, int):
            k_max = [k_max]

        y_approx = [(u[..., :k] * s[:k]) @ v[:k] for k in k_max]

        if len(k_max) == 1:
            y_approx = y_approx[0]

    else:
        y_approx = y
        u_now = None
        coeffs = None
        sigs = None
    if ret:
        return u_now, coeffs, sigs
    return y_approx

def latent_func_var_strong_RRAE(self, y, k_max=None, epsilon=None, return_dist=False, return_lat_dist=False, **kwargs):
    apply_basis = kwargs.get("apply_basis")
    
    if "apply_basis" in kwargs:
        kwargs.pop("apply_basis")
        
    if kwargs.get("get_coeffs") or kwargs.get("get_basis_coeffs"):
        if return_dist or return_lat_dist:
            raise ValueError
        return latent_func_strong_RRAE(self, y, k_max, apply_basis=apply_basis, **kwargs)

    basis, coeffs = latent_func_strong_RRAE(self, y, k_max=k_max, get_basis_coeffs=True, apply_basis=apply_basis)
    if self.typ == "eye":
        mean = coeffs
    elif self.typ == "trainable":
        mean = self.lin_mean(coeffs)
    else:
        raise ValueError("typ must be either 'eye' or 'trainable'")

    logvar = self.lin_logvar(coeffs)

    if return_dist:
        return mean, logvar

    std = jnp.exp(0.5 * logvar)
    if epsilon is not None:
        if len(epsilon.shape) == 4:
            epsilon = epsilon[0, 0] # to allow tpu sharding
        z = mean + epsilon * std
    else:
        z = mean

    if return_lat_dist:
        return basis @ z, mean, logvar
    return basis @ z


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
        return self.decode(self.perform_in_latent(self.encode(x), *args, **kwargs))

    def latent(self, x, *args, **kwargs):
        return self.perform_in_latent(self.encode(x), *args, **kwargs)


class RRAE_MLP(Autoencoder):
    """Subclass of RRAEs with the strong formulation when the input
    is of dimension (data_size, batch_size).

    Attributes
    ----------
    encode : MLP_with_linear
        An MLP as the encoding function.
    decode : MLP_with_linear
        An MLP as the decoding function.
    perform_in_latent : function
        The function that performs operations in the latent space.
    k_max : int
        The maximum number of modes to keep in the latent space.
    """

    def __init__(
        self,
        in_size,
        latent_size,
        k_max,
        post_proc_func=None,
        *,
        key,
        kwargs_enc={},
        kwargs_dec={},
        **kwargs,
    ):

        if "linear_l" in kwargs.keys():
            warnings.warn("linear_l can not be specified for Strong")
            kwargs.pop("linear_l")

        super().__init__(
            in_size,
            latent_size,
            map_latent=False,
            post_proc_func=post_proc_func,
            key=key,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
            **kwargs,
        )

    def _perform_in_latent(self, y, *args, **kwargs):
        return latent_func_strong_RRAE(self, y, *args, **kwargs)
    

class Vanilla_AE_MLP(Autoencoder):
    """Vanilla Autoencoder.

    Subclass for the Vanilla AE, basically the strong RRAE with
    k_max = -1, hence returning all the modes with no truncation.
    """

    def __init__(
        self,
        in_size,
        latent_size,
        *,
        key,
        kwargs_enc={},
        kwargs_dec={},
        **kwargs,
    ):
        if "k_max" in kwargs.keys():
            if kwargs["k_max"] != -1:
                warnings.warn(
                    "k_max can not be specified for Vanilla_AE_MLP, switching to -1 (all modes)"
                )
            kwargs.pop("k_max")

        latent_size_after = latent_size

        super().__init__(
            in_size,
            latent_size,
            latent_size_after,
            key=key,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
            **kwargs,
        )


def sample(y, sample_cls, k_max=None, epsilon=None, *args, **kwargs):
    if epsilon is None:
        new_perform_sample = lambda m, lv: sample_cls(m, lv, *args, **kwargs)
        return jax.vmap(new_perform_sample, in_axes=[-1, -1], out_axes=-1)(*y)
    else:
        new_perform_sample = lambda m, lv, s: sample_cls(m, lv, s, *args, **kwargs)
        return jax.vmap(new_perform_sample, in_axes=[-1, -1, -1], out_axes=-1)(
            *y, epsilon
        )


class VAE_MLP(Autoencoder):
    _sample: Sample
    lin_mean: Linear
    lin_logvar: Linear
    latent_size: int

    def __init__(
        self,
        in_size,
        latent_size,
        *,
        key,
        kwargs_enc={},
        kwargs_dec={},
        **kwargs,
    ):
        key, key_m, key_s = jrandom.split(key, 3)
        self.latent_size = latent_size
        self._sample = Sample(sample_dim=latent_size)
        self.lin_mean = Linear(latent_size, latent_size, key=key_m)
        self.lin_logvar = Linear(latent_size, latent_size, key=key_s)

        super().__init__(
            in_size,
            latent_size,
            map_latent=False,
            key=key,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
        )

    def _perform_in_latent(self, y, *args, return_dist=False, **kwargs):
        y = jax.vmap(self.lin_mean, in_axes=-1, out_axes=-1)(y), jax.vmap(
            self.lin_logvar, in_axes=-1, out_axes=-1
        )(y)
        if return_dist:
            return y[0], y[1]
        return sample(y, self._sample, *args, **kwargs)


class IRMAE_MLP(Autoencoder):
    def __init__(
        self,
        in_size,
        latent_size,
        linear_l=None,
        *,
        key,
        kwargs_enc={},
        kwargs_dec={},
        **kwargs,
    ):

        assert linear_l is not None, "linear_l must be specified for IRMAE_MLP"

        if "k_max" in kwargs.keys():
            if kwargs["k_max"] != -1:
                warnings.warn(
                    "k_max can not be specified for the model proposed, switching to -1 (all modes)"
                )
            kwargs.pop("k_max")

        if "linear_l" in kwargs.keys():
            raise ValueError("Specify linear_l in the constructor, not in kwargs")

        kwargs_enc = {**kwargs_enc, "linear_l": linear_l}

        super().__init__(
            in_size,
            latent_size,
            key=key,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
            **kwargs,
        )


class LoRAE_MLP(IRMAE_MLP):
    def __init__(
        self, in_size, latent_size, *, key, kwargs_enc={}, kwargs_dec={}, **kwargs
    ):
        if "linear_l" in kwargs.keys():
            if kwargs["linear_l"] != 1:
                raise ValueError("linear_l can not be specified for LoRAE_CNN")

        super().__init__(
            in_size,
            latent_size,
            linear_l=1,
            key=key,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
            **kwargs,
        )



class CNN_Autoencoder(Autoencoder):
    def __init__(
        self,
        channels,
        height,
        width,
        latent_size,
        latent_size_after=None,
        *,
        key,
        count=1,
        dimension=2,
        kwargs_enc={},
        kwargs_dec={},
        **kwargs,
    ):
        latent_size_after = (
            latent_size if latent_size_after is None else latent_size_after
        )
        key1, key2, key3 = jrandom.split(key, 3)

        _encode = CNNs_with_MLP(
            width=width,
            height=height,
            channels=channels,
            out=latent_size,
            key=key1,
            dimension=dimension,
            **kwargs_enc,
        )

        _decode = MLP_with_CNNs_trans(
            width=width,
            height=height,
            inp=latent_size_after,
            channels=channels,
            key=key2,
            dimension=dimension,
            **kwargs_dec,
        )

        super().__init__(
            None,
            latent_size,
            _encode=_encode,
            map_latent=False,
            _decode=_decode,
            key=key3,
            count=count,
            **kwargs,
        )

class CNN3D_Autoencoder(Autoencoder):
    def __init__(
        self,
        depth,   # I add the depth
        width,
        height,
        channels,
        latent_size,
        k_max=-1,
        latent_size_after=None,
        _perform_in_latent=_identity,
        *,
        key,
        kwargs_enc={},
        kwargs_dec={},
        **kwargs,
    ):
        latent_size_after = (
            latent_size if latent_size_after is None else latent_size_after
        )
        key1, key2, key3 = jrandom.split(key, 3)

        _encode = CNN3D_with_MLP(
            depth=depth,
            width=width,
            height=height,
            channels=channels,
            out=latent_size,
            key=key1,
            **kwargs_enc,
        )

        _decode = MLP_with_CNN3D_trans(
            depth=depth,
            width=width,
            height=height,
            inp=latent_size_after,
            channels=channels,
            key=key2,
            **kwargs_dec,
        )

        super().__init__(
            None,
            latent_size,
            k_max=k_max,
            _encode=_encode,
            _perform_in_latent=_perform_in_latent,
            map_latent=False,
            _decode=_decode,
            key=key3,
            **kwargs,
        )
        
        
class VAE_CNN(CNN_Autoencoder):
    lin_mean: Linear
    lin_logvar: Linear
    latent_size: int

    def __init__(self, channels, height, width, latent_size, *, key, count=1, **kwargs):
        key, key_m, key_s = jrandom.split(key, 3)
        v_Linear = vmap_wrap(Linear, -1, count=count)
        self.lin_mean = v_Linear(latent_size, latent_size, key=key_m)
        self.lin_logvar = v_Linear(latent_size, latent_size, key=key_s)
        self.latent_size = latent_size
        super().__init__(
            channels,
            height,
            width,
            latent_size,
            key=key,
            count=count,
            **kwargs,
        )

    def _perform_in_latent(self, y, *args, epsilon=None, return_dist=False, return_lat_dist=False, **kwargs):
        mean = self.lin_mean(y)
        logvar = self.lin_logvar(y)

        if return_dist:
            return mean, logvar

        std = jnp.exp(0.5 * logvar)
        if epsilon is not None:
            if len(epsilon.shape) == 4:
                epsilon = epsilon[0, 0] # to allow tpu sharding
            z = mean + epsilon * std
        else:
            z = mean

        if return_lat_dist:
            return z, mean, logvar
        return z

class RRAE_CNN(CNN_Autoencoder):
    """Subclass of RRAEs with the strong formulation for inputs of
    dimension (channels, width, height).
    """

    def __init__(self, channels, height, width, latent_size, k_max, *, key, **kwargs):

        super().__init__(
            channels,
            height,
            width,
            latent_size,
            key=key,
            **kwargs,
        )

    def _perform_in_latent(self, y, *args, **kwargs):
        return latent_func_strong_RRAE(self, y, *args, **kwargs)


class RRAE_CNN3D(CNN3D_Autoencoder):

    def __init__(self, depth, width, height, channels, latent_size, k_max, *, key, **kwargs):

        super().__init__(
            depth,
            width,
            height,
            channels,
            latent_size,
            k_max,
            _perform_in_latent=latent_func_strong_RRAE,
            key=key,
            **kwargs,
        )
    
    def _perform_in_latent(self, y, *args, **kwargs):
        return latent_func_strong_RRAE(self, y, *args, **kwargs)
        
        
class VRRAE_CNN(CNN_Autoencoder):
    lin_mean: Linear
    lin_logvar: Linear
    typ: int

    def __init__(self, channels, height, width, latent_size, k_max, typ="eye", *, key, count=1, **kwargs):
        key, key_m, key_s = jrandom.split(key, 3)
        v_Linear = vmap_wrap(Linear, -1, count=count)
        self.lin_mean = v_Linear(k_max, k_max, key=key_m)
        self.lin_logvar = v_Linear(k_max, k_max, key=key_s)
        self.typ = typ
        super().__init__(
            channels,
            height,
            width,
            latent_size,
            key=key,
            count=count,
            **kwargs,
        )

    def _perform_in_latent(self, y, *args, k_max=None, epsilon=None, return_dist=False, return_lat_dist=False, **kwargs):
        return latent_func_var_strong_RRAE(self, y, k_max, epsilon, return_dist, return_lat_dist, **kwargs)
    

class Vanilla_AE_CNN(CNN_Autoencoder):
    """Vanilla Autoencoder.

    Subclass for the Vanilla AE, basically the strong RRAE with
    k_max = -1, hence returning all the modes with no truncation.
    """

    def __init__(self, channels, height, width, latent_size, *, key, **kwargs):
        if "linear_l" in kwargs.keys():
            warnings.warn("linear_l can not be specified for Vanilla_CNN")
            kwargs.pop("linear_l")

        super().__init__(channels, height, width, latent_size, key=key, **kwargs)


class IRMAE_CNN(CNN_Autoencoder):
    def __init__(
        self, channels, height, width, latent_size, linear_l=None, *, key, **kwargs
    ):

        assert linear_l is not None, "linear_l must be specified for IRMAE_CNN"

        if "kwargs_enc" in kwargs:
            kwargs_enc = kwargs["kwargs_enc"]
            kwargs_enc["kwargs_mlp"] = {"linear_l": linear_l}
            kwargs["kwargs_enc"] = kwargs_enc
        else:
            kwargs["kwargs_enc"] = {"kwargs_mlp": {"linear_l": linear_l}}
        super().__init__(
            channels,
            height,
            width,
            latent_size,
            key=key,
            **kwargs,
        )


class LoRAE_CNN(IRMAE_CNN):
    def __init__(self, channels, height, width, latent_size, *, key, **kwargs):

        if "linear_l" in kwargs.keys():
            if kwargs["linear_l"] != 1:
                raise ValueError("linear_l can not be specified for LoRAE_CNN")

        super().__init__(
            channels, height, width, latent_size, linear_l=1, key=key, **kwargs
        )


class CNN1D_Autoencoder(CNN_Autoencoder):
    def __init__(
        self,
        channels,
        input_dim,
        latent_size,
        latent_size_after=None,
        *,
        key,
        count=1,
        kwargs_enc={},
        kwargs_dec={},
        **kwargs,
    ):
        super().__init__(channels,
            input_dim,
            None,
            latent_size,
            latent_size_after=latent_size_after,
            key=key,
            count=count,
            dimension=1,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
        )

class RRAE_CNN1D(CNN1D_Autoencoder):
    """Subclass of RRAEs with the strong formulation for inputs of
    dimension (channels, width, height).
    """

    def __init__(self, channels, input_dim, latent_size, k_max, *, key, **kwargs):

        super().__init__(
            channels,
            input_dim,
            latent_size,
            key=key,
            **kwargs,
        )

    def _perform_in_latent(self, y, *args, **kwargs):
        return latent_func_strong_RRAE(self, y, *args, **kwargs)


class VRRAE_CNN1D(CNN1D_Autoencoder):
    lin_mean: Linear
    lin_logvar: Linear
    typ: int

    def __init__(self, channels, input_dim, latent_size, k_max, typ="eye", *, key, count=1, **kwargs):
        key, key_m, key_s = jrandom.split(key, 3)
        v_Linear = vmap_wrap(Linear, -1, count=count)
        self.lin_mean = v_Linear(k_max, k_max, key=key_m)
        self.lin_logvar = v_Linear(k_max, k_max, key=key_s)
        self.typ = typ
        super().__init__(
            channels,
            input_dim,
            latent_size,
            key=key,
            count=count,
            **kwargs,
        )

    def _perform_in_latent(self, y, *args, k_max=None, epsilon=None, return_dist=False, return_lat_dist=False, **kwargs):
        return latent_func_var_strong_RRAE(self, y, k_max, epsilon, return_dist, return_lat_dist, **kwargs)

    def get_basis_coeffs(self, x, *args, **kwargs):
        return self.perform_in_latent(self.encode(x), *args, get_basis_coeffs=True, **kwargs)

    def decode_coeffs(self, c, basis, *args, **kwargs):
        return self.decode(basis @ c, *args, **kwargs)
