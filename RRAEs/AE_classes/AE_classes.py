import jax
import jax.numpy as jnp
import pdb
from RRAEs.utilities import (
    Sample,
    v_vt_class,
    CNNs_with_linear,
    Linear_with_CNNs_trans,
    dataloader,
    MLP_with_linear,
)
import itertools
import equinox as eqx
import jax.random as jrandom
from equinox._doc_utils import doc_repr
import warnings

_identity = doc_repr(lambda x, *args, **kwargs: x, "lambda x: x")


class BaseClass(eqx.Module):
    map_axis: int
    model: eqx.Module
    # norm_funcs: list

    def __init__(self, model, map_axis=None, *args, **kwargs):
        self.map_axis = map_axis
        self.model = model

    def __call__(self, x):
        if self.map_axis is None:
            return self.model(x)
        return jax.vmap(self.model, in_axes=[self.map_axis], out_axes=self.map_axis)(x)

    def eval_with_batches(
        self,
        x,
        batch_size,
        call_func,
        end_type="concat_and_resort",
        *args,
        key,
        **kwargs,
    ):
        idxs = []
        all_preds = []

        for _, (input_b, idx) in zip(
            itertools.count(start=0),
            dataloader(
                [x.T, jnp.arange(0, x.shape[-1], 1)],
                batch_size,
                key=key,
                once=True,
            ),
        ):
            pred = call_func(input_b.T)
            idxs.append(idx)
            all_preds.append(pred)
            if end_type == "first":
                break

        idxs = jnp.concatenate(idxs)
        match end_type:
            case "concat_and_resort":
                final_pred = jnp.concatenate(all_preds, -1)[..., jnp.argsort(idxs)]
            case "mean":
                final_pred = sum(all_preds) / len(all_preds)
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
    _perform_in_latent: None
    k_max: int
    map_latent: bool
    norm_funcs: list
    inv_norm_funcs: list

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
        k_max=-1,
        latent_size_after=None,
        _encode=None,
        _perform_in_latent=None,
        _decode=None,
        map_latent=True,
        *,
        key,
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
            self.encode = BaseClass(model_cls, -1)

        else:
            self.encode = _encode

        if _perform_in_latent is None:
            self._perform_in_latent = _identity
        else:
            self._perform_in_latent = _perform_in_latent

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
            self.decode = BaseClass(model_cls, -1)
        else:
            self.decode = _decode

        self.k_max = k_max
        self.map_latent = map_latent
        self.inv_norm_funcs = ["decode"]
        self.norm_funcs = ["encode", "latent"]

    def perform_in_latent(self, y, *args, **kwargs):
        if self.map_latent:
            new_perform_in_latent = lambda x: self._perform_in_latent(
                x, self.k_max, *args, **kwargs
            )
            return jax.vmap(new_perform_in_latent, in_axes=[-1], out_axes=-1)(y)
        return self._perform_in_latent(y, self.k_max, *args, **kwargs)

    def __call__(self, x, *args, **kwargs):
        return self.decode(self.perform_in_latent(self.encode(x), *args, **kwargs))

    def latent(self, x, *args, **kwargs):
        return self.perform_in_latent(self.encode(x), *args, **kwargs)


def latent_func_strong_RRAE(y, k_max, basis=None, ret=False, *args, **kwargs):
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
    if basis is not None:
        return basis @ basis.T @ y
      
    if k_max != -1:
        u, s, v = jnp.linalg.svd(y, full_matrices=False)
        sigs = s[:k_max]
        v_now = v[:k_max, :]
        u_now = u[:, :k_max]

        coeffs = jnp.multiply(v_now, jnp.expand_dims(sigs, -1))
        y_approx = jnp.sum(
            jax.vmap(lambda u, v: jnp.outer(u, v), in_axes=[-1, 0], out_axes=-1)(
                u_now, coeffs
            ),
            axis=-1,
        )
    else:
        y_approx = y
        u_now = None
        v_now = None
        sigs = None
    if ret:
        return u_now, coeffs, sigs
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

        latent_func = latent_func_strong_RRAE

        super().__init__(
            in_size,
            latent_size,
            k_max,
            _perform_in_latent=latent_func,
            map_latent=False,
            post_proc_func=post_proc_func,
            key=key,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
            **kwargs,
        )


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

        latent_func = None
        latent_size_after = latent_size

        super().__init__(
            in_size,
            latent_size,
            -1,
            latent_size_after,
            key=key,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
            _perform_in_latent=latent_func,
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
        data_size,
        *,
        key,
        kwargs_enc={},
        kwargs_dec={},
        **kwargs,
    ):

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
            k_max = data_size

        self.v_vt = v_vt_class(latent_size, data_size, k_max, key=key)


def sample(y, sample_cls, epsilon=None, *args, **kwargs):
    if epsilon is None:
        new_perform_sample = lambda x: sample_cls(x, *args, **kwargs)
        return jax.vmap(new_perform_sample, in_axes=[-1], out_axes=-1)(y)
    else:
        new_perform_sample = lambda x, s: sample_cls(x, s, *args, **kwargs)
        return jax.vmap(new_perform_sample, in_axes=[-1, -1], out_axes=-1)(y, epsilon)


class VAR_AE_MLP(Autoencoder):
    _sample: Sample

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

        self._sample = Sample(sample_dim=latent_size)

        super().__init__(
            in_size,
            latent_size=latent_size * 2,
            latent_size_after=latent_size,
            _perform_in_latent=lambda y, *args, **kwargs: sample(
                y, self._sample, *args, **kwargs
            ),  # Note: can not define sample as calss method to maintain tree structure
            map_latent=False,
            key=key,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
        )


class IRMAE_MLP(Autoencoder):
    def __init__(
        self,
        in_size,
        latent_size,
        linear_l=2,
        *,
        key,
        kwargs_enc={},
        kwargs_dec={},
        **kwargs,
    ):

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
            -1,
            key=key,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
            **kwargs,
        )


class LoRAE_MLP(IRMAE_MLP):
    def __init__(
        self, in_size, latent_size, *, key, kwargs_enc={}, kwargs_dec={}, **kwargs
    ):
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
        in_size,
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

        encode = CNNs_with_linear(
            data_dim0=in_size,
            out=latent_size,
            key=key1,
            **kwargs_enc,
        )
        _encode = BaseClass(encode, -1)

        decode = Linear_with_CNNs_trans(
            data_dim0=in_size,
            inp=latent_size_after,
            key=key2,
            **kwargs_dec,
        )
        _decode = BaseClass(decode, -1)

        super().__init__(
            in_size,
            latent_size,
            k_max=k_max,
            _encode=_encode,
            _perform_in_latent=_perform_in_latent,
            map_latent=False,
            _decode=_decode,
            key=key3,
            **kwargs,
        )


class VAR_AE_CNN(CNN_Autoencoder):
    _sample: Sample

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

        self._sample = Sample(sample_dim=latent_size)

        super().__init__(
            in_size,
            latent_size=latent_size * 2,
            latent_size_after=latent_size,
            _perform_in_latent=lambda y, *args, **kwargs: sample(
                y, self._sample, *args, **kwargs
            ),  # Note: can not define sample as calss method to maintain tree structure
            key=key,
            kwargs_enc=kwargs_enc,
            kwargs_dec=kwargs_dec,
        )


class Strong_RRAE_CNN(CNN_Autoencoder):
    """Subclass of RRAEs with the strong formulation for inputs of
    dimension (data_size_1 x data_size_2 x batch_size).
    """

    def __init__(self, in_size, latent_size, k_max, *, key, **kwargs):

        super().__init__(
            in_size,
            latent_size,
            k_max,
            _perform_in_latent=latent_func_strong_RRAE,
            key=key,
            **kwargs,
        )


class Vanilla_AE_CNN(CNN_Autoencoder):
    """Vanilla Autoencoder.

    Subclass for the Vanilla AE, basically the strong RRAE with
    k_max = -1, hence returning all the modes with no truncation.
    """

    def __init__(self, in_size, latent_size, *, key, **kwargs):
        if "k_max" in kwargs.keys():
            if kwargs["k_max"] != -1:
                warnings.warn(
                    "k_max can not be specified for Vanilla_AE_CNN, switching to -1 (all modes)"
                )
            kwargs.pop("k_max")

        if "linear_l" in kwargs.keys():
            warnings.warn("linear_l can not be specified for Vanilla_CNN")
            kwargs.pop("linear_l")

        super().__init__(in_size, latent_size, key=key, **kwargs)


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

    def __init__(self, in_size, latent_size, k_max, data_size, *, key, **kwargs):
        if "linear_l" in kwargs.keys():
            warnings.warn("linear_l can not be specified for Vanilla_CNN")
            kwargs.pop("linear_l")

        super().__init__(in_size, latent_size, key=key, **kwargs)
        if k_max == -1:
            k_max = data_size

        self.v_vt = v_vt_class(latent_size, data_size, k_max, key=key)


class IRMAE_CNN(CNN_Autoencoder):
    def __init__(self, in_size, latent_size, linear_l=2, *, key, **kwargs):

        if "k_max" in kwargs.keys():
            if kwargs["k_max"] != -1:
                warnings.warn(
                    "k_max can not be specified for the model proposed, switching to -1 (all modes)"
                )
            kwargs.pop("k_max")

        warnings.warn(
            "IRMAEs and LoRAEs are not tested for CNNs, be careful when using them..."
        )
        super().__init__(
            in_size,
            latent_size,
            -1,
            kwargs_mlp_enc={"linear_l": linear_l},
            key=key,
            **kwargs,
        )


class LoRAE_CNN(IRMAE_CNN):
    def __init__(self, data, latent_size, *, key, **kwargs):
        super().__init__(data, latent_size, linear_l=1, key=key, **kwargs)
