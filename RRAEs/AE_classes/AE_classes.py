import jax
import jax.numpy as jnp
import pdb
from RRAEs.utilities import (
    Sample,
    v_vt_class,
    CNNs_with_MLP,
    MLP_with_CNNs_trans,
    dataloader,
    MLP_with_linear,
    stable_SVD,
)
import itertools
import equinox as eqx
import jax.random as jrandom
from equinox._doc_utils import doc_repr
import warnings
from tqdm import tqdm
import numpy as np
from equinox.nn._linear import Linear


_identity = doc_repr(lambda x, *args, **kwargs: x, "lambda x: x")


class BaseClass(eqx.Module):
    map_axis: int
    model: eqx.Module
    count: int

    def __init__(self, model, map_axis=None, *args, count=1, **kwargs):
        self.map_axis = map_axis
        self.model = model
        self.count = count

    def __call__(self, x, *args, **kwargs):
        if self.map_axis is None:
            return self.model(x, *args, **kwargs)
        fn = lambda x: self.model(x, *args, **kwargs)
        for _ in range(self.count):
            fn = jax.vmap(fn, in_axes=(self.map_axis,), out_axes=self.map_axis)
        out = fn(x)
        return out

    def eval_with_batches(
        self,
        x,
        batch_size,
        call_func,
        end_type="concat_and_resort",
        str=None,
        *args,
        key_idx,
        **kwargs,
    ):
        """Works for array input with data as final dim."""
        idxs = []
        all_preds = []

        if str is not None:
            print(str)
            fn = lambda x, *args, **kwargs: tqdm(x, *args, **kwargs)
        else:
            fn = lambda x, *args, **kwargs: x

        if not (isinstance(x, tuple) or isinstance(x, list)):
            x = [x]
        x = [el.T for el in x]

        for _, inputs in fn(
            zip(
                itertools.count(start=0),
                dataloader(
                    [*x, jnp.arange(0, x[0].shape[0], 1)],
                    batch_size,
                    key_idx=key_idx,
                    once=True,
                ),
            ),
            total=int(x[0].shape[-1] / batch_size),
        ):
            input_b = inputs[:-1]
            idx = inputs[-1]

            input_b = [el.T for el in input_b]

            pred = call_func(*input_b, *args, **kwargs)
            idxs.append(idx)
            all_preds.append(pred)
            if end_type == "first":
                break
        idxs = jnp.concatenate(idxs)
        match end_type:
            case "concat_and_resort":
                final_pred = jnp.concatenate(all_preds, -1)[..., jnp.argsort(idxs)]
            case "concat":
                final_pred = jnp.concatenate(all_preds, -1)
            case "mean":
                final_pred = sum(all_preds) / len(all_preds)
            case "sum":
                final_pred = sum(all_preds)
            case "first":
                final_pred = all_preds[0]
            case _:
                final_pred = all_preds
        return final_pred

    def __getattr__(self, name: str):
        return getattr(self.model, name)


class Autoencoder(eqx.Module):
    encode: MLP_with_linear
    decode: MLP_with_linear
    _perform_in_latent: callable
    _perform_in_latent: callable
    map_latent: bool

    """Abstract base class for all Autoencoders.

    You can create your own Autoencoder as a subclass of this one.
    This class is subclass of eqx.Module so it inherits its behavior.
    First, it is an ABC so abstract methods, etc. can be used. Also,
    the class itself, as well as its atributes are PyTrees.

    Attributes
    ----------
    encode : function
        The encoding function.
    decode : function
        The decoding function.
    perform_in_latent : function
        The function that performs operations in the latent space.

    Note
    -----
    In general, the methods are expecting to
    be the following maps:

    - encode: (..., batch_size) -> (latent_size, batch_size)
    - perform_in_latent: (latent_size, batch_size) -> (latent_size, batch_size)
    - decode: (latent_size, batch_size) -> (..., batch_size)
    """

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

            model_cls = MLP_with_linear(
                in_size=in_size,
                out_size=latent_size,
                key=key_e,
                **kwargs_enc,
            )
            self.encode = BaseClass(model_cls, -1, count=count)

        else:
            self.encode = _encode

        if not hasattr(self, "_perform_in_latent"):
            self._perform_in_latent = _identity

        if map_latent:
            self.perform_in_latent = BaseClass(self._perform_in_latent)
        else:
            self.perform_in_latent = self._perform_in_latent
            
        if _decode is None:
            if "width_size" not in kwargs_dec.keys():
                kwargs_dec["width_size"] = 64
            if "depth" not in kwargs_dec.keys():
                kwargs_dec["depth"] = 6

            model_cls = MLP_with_linear(
                in_size=latent_size_after,
                out_size=in_size,
                key=key_d,
                **kwargs_dec,
            )
            self.decode = BaseClass(model_cls, -1, count=count)
        else:
            self.decode = _decode

        self.map_latent = map_latent

    def __call__(self, x, *args, **kwargs):
        return self.decode(self.perform_in_latent(self.encode(x), *args, **kwargs))

    def latent(self, x, *args, **kwargs):
        return self.perform_in_latent(self.encode(x), *args, **kwargs)


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


def latent_func_var_strong_RRAE(
    self,
    y,
    k_max=None,
    apply_basis=None,
    get_basis_coeffs=False,
    get_right_sing=False,
    get_coeffs=False,
    get_sings=False,
    ret=False,
    epsilon=None,
    sigma=1,
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
    batch_size = y.shape[-1]
    I = jnp.eye(batch_size) # *perc_imp     

    if apply_basis is not None:
        if get_basis_coeffs or get_right_sing or get_sings:
            raise ValueError("Can not get SVD and apply basis at the same time.")
        if get_coeffs:
            return apply_basis.T @ y
        alpha = apply_basis.T @ y
      
        if epsilon is not None:
            epsilon = epsilon if len(epsilon.shape) == 2 else epsilon[0, 0]
            epsilon = epsilon*sigma/y.shape[-1]
            G = jnp.multiply(epsilon, jnp.expand_dims(s[:k_max], -1))
            alpha = alpha + G
        return apply_basis @ alpha

    if get_basis_coeffs or get_coeffs:
        u, s, v = stable_SVD(y)
        u_now = u[:, :k_max]
        coeffs = jnp.multiply(v[:k_max, :], jnp.expand_dims(s[:k_max], -1))
        if get_sings:
            return s[:k_max]
        if get_right_sing:
            return v[:k_max, :]
        if get_sings:
            return s[:k_max]
        if get_right_sing:
            return v[:k_max, :]
        if get_coeffs:
            return coeffs
        return u_now, coeffs

    if k_max != -1:
        u, s, v = stable_SVD(y)
        y_approx = (u[..., :k_max] * s[:k_max]) @ v[:k_max]
        alpha = jnp.multiply(v[:k_max, :], jnp.expand_dims(s[:k_max], -1))
      
        if epsilon is not None:
            epsilon = epsilon if len(epsilon.shape) == 2 else epsilon[0, 0]
            epsilon = epsilon*sigma/y.shape[-1]
            G = jnp.multiply(epsilon, jnp.expand_dims(s[:k_max], -1))
            alpha = alpha + G
        y_approx = u[..., :k_max] @ alpha
    else:
        y_approx = y
        u_now = None
        coeffs = None
        sigs = None
    if ret:
        return G
    return y_approx


class Strong_RRAE_MLP(Autoencoder):
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

class VAR_Strong_v3_RRAE_MLP(Autoencoder):
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
        return latent_func_var_strong_RRAE(self, y, *args, **kwargs)

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


class Strong_Dynamics_RRAE_MLP(Autoencoder):
    DMD_W: Linear

    """Vanilla Autoencoder.

    Subclass for the Vanilla AE, basically the strong RRAE with
    k_max = -1, hence returning all the modes with no truncation.
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
        raise NotImplementedError
        if "linear_l" in kwargs.keys():
            warnings.warn("linear_l can not be specified for Strong")
            kwargs.pop("linear_l")

        key1, key2 = jrandom.split(key, 2)

        latent_func = latent_func_strong_RRAE
        self.DMD_W = Linear(k_max, k_max, use_bias=False, key=key1)

        super().__init__(
            in_size,
            latent_size,
            k_max,
            _perform_in_latent=latent_func,
            map_latent=False,
            post_proc_func=post_proc_func,
            key=key2,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
            **kwargs,
        )


class Weak_RRAE_MLP(Autoencoder):
    v_vt: v_vt_class

    """Weak Rank Reduction Autoencoder. We define it as a
    subclass of the strong RRAE with k_max = -1, and an
    additional attribute v_vt.

    Additional Attributes:
    -----------------------
    v_vt : v_vt_class
        A class the include two trainable matrices v, and vt
        that can approximate the latent space in the weak 
        formulation.
    """

    def __init__(
        self,
        in_size,
        latent_size,
        k_max,
        samples,
        *,
        key,
        kwargs_enc={},
        kwargs_dec={},
        **kwargs,
    ):
        raise NotImplementedError
        super().__init__(
            in_size,
            latent_size,
            -1,
            key=key,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
            **kwargs,
        )

        if k_max == -1:
            k_max = samples

        self.v_vt = v_vt_class(latent_size, samples, k_max, key=key)


def sample(y, sample_cls, k_max=None, epsilon=None, *args, **kwargs):
    if epsilon is None:
        new_perform_sample = lambda m, lv: sample_cls(m, lv, *args, **kwargs)
        return jax.vmap(new_perform_sample, in_axes=[-1, -1], out_axes=-1)(*y)
    else:
        new_perform_sample = lambda m, lv, s: sample_cls(m, lv, s, *args, **kwargs)
        return jax.vmap(new_perform_sample, in_axes=[-1, -1, -1], out_axes=-1)(
            *y, epsilon
        )


class VAR_AE_MLP(Autoencoder):
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
        kwargs_enc={},
        kwargs_dec={},
        **kwargs,
    ):
        latent_size_after = (
            latent_size if latent_size_after is None else latent_size_after
        )
        key1, key2, key3 = jrandom.split(key, 3)

        encode = CNNs_with_MLP(
            width=width,
            height=height,
            channels=channels,
            out=latent_size,
            key=key1,
            **kwargs_enc,
        )
        _encode = BaseClass(encode, -1, count=count)

        decode = MLP_with_CNNs_trans(
            width=width,
            height=height,
            inp=latent_size_after,
            channels=channels,
            key=key2,
            **kwargs_dec,
        )
        _decode = BaseClass(decode, -1, count=count)

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


class VAR_AE_CNN(CNN_Autoencoder):
    lin_mean: Linear
    lin_logvar: Linear
    latent_size: int

    def __init__(self, channels, height, width, latent_size, *, key, count=1, **kwargs):
        key, key_m, key_s = jrandom.split(key, 3)
        self.lin_mean = BaseClass(Linear(latent_size, latent_size, key=key_m), -1, count=count)
        self.lin_logvar = BaseClass(Linear(latent_size, latent_size, key=key_s), -1, count=count)
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

class Strong_RRAE_CNN(CNN_Autoencoder):
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


class VAR_Strong_RRAE_CNN(CNN_Autoencoder):
    lin_mean: Linear
    lin_logvar: Linear
    typ: int

    def __init__(self, channels, height, width, latent_size, k_max, typ="eye", *, key, count=1, **kwargs):
        key, key_m, key_s = jrandom.split(key, 3)

        self.lin_mean = BaseClass(Linear(k_max, k_max, key=key_m), -1, count=count)
        self.lin_logvar = BaseClass(Linear(k_max, k_max, key=key_s), -1, count=count)
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
        
        apply_basis = kwargs.get("apply_basis")
        
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

class VAR_Strong_v3_RRAE_CNN(CNN_Autoencoder):
    """Subclass of RRAEs with the strong formulation for inputs of
    dimension (channels, width, height).
    """

    def __init__(self, channels, height, width, latent_size, k_max, *, key, basis=None, **kwargs):

        super().__init__(
            channels,
            height,
            width,
            latent_size,
            basis=basis,
            key=key,
            **kwargs,
        )
    
    def _perform_in_latent(self, y, *args, **kwargs):
        return latent_func_var_strong_RRAE(self, y, *args, **kwargs)


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


class Weak_RRAE_CNN(CNN_Autoencoder):
    v_vt: v_vt_class

    """Weak Rank Reduction Autoencoder. We define it as a
    subclass of the strong RRAE (CNN) with k_max = -1, and an
    additional attribute v_vt.

    Additional Attributes:
    -----------------------
    v_vt : v_vt_class
        A class the include two trainable matrices v, and vt
        that can approximate the latent space in the weak 
        formulation.
    """

    def __init__(
        self, channels, height, width, latent_size, k_max, samples, *, key, **kwargs
    ):
        raise NotImplementedError
        if "linear_l" in kwargs.keys():
            warnings.warn("linear_l can not be specified for Vanilla_CNN")
            kwargs.pop("linear_l")

        super().__init__(channels, height, width, latent_size, key=key, **kwargs)
        if k_max == -1:
            k_max = samples

        self.v_vt = v_vt_class(latent_size, samples, k_max, key=key)


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
