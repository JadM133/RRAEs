import jax
import jax.numpy as jnp
from abc import abstractmethod
import pdb
from utilities import Func, v_vt_class, CNN, CNN_trans
import equinox as eqx
import jax.random as jrandom
import jax.nn as jnn
from equinox._doc_utils import doc_repr
import warnings

_identity = doc_repr(lambda x: x, "lambda x: x")


def None_ag_kg(*args, **kwargs):
    return [None] * len(args) + [None] * len(kwargs)


class Autoencoder(eqx.Module):
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

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def decode(self, *args, **kwargs):
        pass

    def perform_in_latent(self, x, *args, **kwargs):
        return _identity(x)

    def __call__(self, x, *args, **kwargs):
        return self.decode(
            self.perform_in_latent(self.encode(x, *args, **kwargs), *args, **kwargs),
            *args,
            **kwargs
        )
    
    def latent(self, x, *args, **kwargs):
        return self.perform_in_latent(self.encode(x, *args, **kwargs), *args, **kwargs)


def latent_func_strong_RRAE(y, k_max, *args, **kwargs):
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
    if k_max != -1:
        u, s, v = jnp.linalg.svd(y, full_matrices=False)
        sigs = s[:k_max]
        v_now = v[:k_max, :]
        u_now = u[:, :k_max]
        y_approx = jnp.sum(
            jax.vmap(
                lambda u, s, v: s * jnp.outer(u, v), in_axes=[-1, 0, 0], out_axes=-1
            )(u_now, sigs, v_now),
            axis=-1,
        )
    else:
        y_approx = y
        u_now = None
        v_now = None
        sigs = None
    return y_approx


class Strong_RRAE_MLPs(Autoencoder):
    _encode: Func
    _decode: Func
    _perform_in_latent: None
    k_max: int
    params: dict

    """Subclass of RRAEs with the strong formulation when the input
    is of dimension (data_size, batch_size).

    Attributes
    ----------
    encode : Func
        An MLP as the encoding function.
    decode : Func
        An MLP as the decoding function.
    perform_in_latent : function
        The function that performs operations in the latent space.
    k_max : int
        The maximum number of modes to keep in the latent space.
    """

    def __init__(
        self,
        data,
        latent_size,
        k_max,
        width_enc=64,
        depth_enc=1,
        width_dec=64,
        depth_dec=6,
        post_proc_func=_identity,
        _encode=None,
        _perform_in_latent=None,
        _decode=None,
        *,
        key,
        **kwargs
    ):
        key_e, key_d = jrandom.split(key)
        if _encode is None:
            self._encode = Func(
                data.shape[0],
                width_enc,
                depth_enc,
                out_size=latent_size,
                key=key_e,
            )
        if _perform_in_latent is None:
            self._perform_in_latent = latent_func_strong_RRAE
        if _decode is None:
            self._decode = Func(
                latent_size,
                width_dec,
                depth_dec,
                out_size=data.shape[0],
                post_proc_func=post_proc_func,
                key=key_d,
            )
        self.k_max = k_max
        self.params = {
            "data": data,
            "post_proc_func": post_proc_func,
            "latent_size": latent_size,
            "k_max": k_max,
            "width_enc": width_enc,
            "depth_enc": depth_enc,
            "width_dec": width_dec,
            "depth_dec": depth_dec,
            "key": key,
        }

    def encode(self, x, *argss, **kwargs):
        new_encode =  lambda x: self._encode(x, *argss, **kwargs)
        return jax.vmap(new_encode, in_axes=[-1], out_axes=-1)(x)

    def perform_in_latent(self, y, *args, **kwargs):
        return self._perform_in_latent(y, self.k_max, *args, **kwargs)

    def decode(self, y, *args, **kwargs):
        new_decode = lambda x: self._decode(x, *args, **kwargs)
        return jax.vmap(new_decode, in_axes=[-1], out_axes=-1)(y)

class Vanilla_AE_MLP(Strong_RRAE_MLPs):
    """Vanilla Autoencoder.

    Subclass for the Vanilla AE, basically the strong RRAE with
    k_max = -1, hence returning all the modes with no truncation.
    """

    def __init__(
        self,
        data,
        latent_size,
        width_enc=64,
        depth_enc=1,
        width_dec=64,
        depth_dec=6,
        *,
        key,
        **kwargs
    ):
        if "k_max" in kwargs.keys():
            if kwargs["k_max"] != -1:
                warnings.warn("k_max can not be specified for Vanilla_AE_MLP, switching to -1 (all modes)")
            kwargs.pop("k_max")
        super().__init__(
            data,
            latent_size,
            -1,
            width_enc,
            depth_enc,
            width_dec,
            depth_dec,
            key=key,
            **kwargs
        )


class Weak_RRAE_MLPs(Strong_RRAE_MLPs):
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
        data,
        latent_size,
        k_max,
        width_enc=64,
        depth_enc=1,
        width_dec=64,
        depth_dec=6,
        *,
        key,
        **kwargs
    ):
        super().__init__(
            data,
            latent_size,
            -1,
            width_enc,
            depth_enc,
            width_dec,
            depth_dec,
            key=key,
            **kwargs
        )
        self.v_vt = v_vt_class(latent_size, data.shape[-1], k_max, key=key)


class IRMAE(Strong_RRAE_MLPs):
    def __init__(
        self,
        data,
        latent_size,
        width_enc=64,
        depth_enc=1,
        width_dec=64,
        depth_dec=6,
        linear_l=2,
        linear_width=None,
        key_linear=None,
        *,
        key,
        **kwargs
    ):

        if "k_max" in kwargs.keys():
            if kwargs["k_max"] != -1:
                warnings.warn("k_max can not be specified for the model proposed, switching to -1 (all modes)")
            kwargs.pop("k_max")
        super().__init__(
            data,
            latent_size,
            -1,
            width_enc,
            depth_enc,
            width_dec,
            depth_dec,
            key=key,
            **kwargs
        )

        linear_width = latent_size if linear_width is None else linear_width
        key_linear = jrandom.PRNGKey(0) if key_linear is None else key_linear

        self._perform_in_latent = Func(
            latent_size,
            linear_width,
            linear_l-1,
            out_size=latent_size,
            inside_activation=_identity,
            post_proc_func=_identity,
            use_bias=False,
            key=key_linear,
        )

    def perform_in_latent(self, y, *args, **kwargs):
        new_perform_in_latent = lambda x: self._perform_in_latent(x, *args, **kwargs)
        return jax.vmap(new_perform_in_latent, in_axes=[-1,], out_axes=-1)(y)

class LoRAE(IRMAE):
    def __init__(
        self,
        data,
        latent_size,
        width_enc=64,
        depth_enc=1,
        width_dec=64,
        depth_dec=6,
        *,
        key,
        **kwargs
    ):
        super().__init__(
            data,
            latent_size,
            width_enc=width_enc,
            depth_enc=depth_enc,
            width_dec=width_dec,
            depth_dec=depth_dec,
            linear_l=1,
            linear_width=None,
            key_linear=None,
            key=key,
            **kwargs
        )

    def perform_in_latent(self, y, *args, **kwargs):
        return jnn.relu(super().perform_in_latent(y, *args, **kwargs))

class Strong_RRAE_CNN(Autoencoder):
    _encode: Func
    _decode: Func
    _perform_in_latent: None
    k_max: int
    params: dict
    """Subclass of RRAEs with the strong formulation for inputs of
    dimension (data_size_1 x data_size_2 x batch_size).
    """

    def __init__(
        self,
        data,
        latent_size,
        k_max,
        width_enc=64,
        depth_enc=1,
        width_dec=64,
        depth_dec=6,
        kernel_conv=3,
        stride=1,
        padding=2,
        out_conv=3,
        *,
        key,
        **kwargs
    ):
        self._encode = CNN(
            data_dim0=data.shape[0],
            data_dim1=data.shape[1],
            width=width_enc,
            depth=depth_enc,
            out=latent_size,
            kernel_conv=kernel_conv,
            stride=stride,
            padding=padding,
            out_conv=out_conv,
            key=key,
        )
        self._perform_in_latent = latent_func_strong_RRAE
        self._decode = CNN_trans(
            data_dim0=data.shape[0],
            data_dim1=data.shape[1],
            width=width_dec,
            depth=depth_dec,
            kernel_conv=kernel_conv,
            stride=stride,
            padding=padding,
            out_conv=out_conv,
            out=latent_size,
            key=key,
        )
        self.k_max = k_max
        self.params = {
            "data": data,
            "latent_size": latent_size,
            "k_max": k_max,
            "width_enc": width_enc,
            "depth_enc": depth_enc,
            "width_dec": width_dec,
            "depth_dec": depth_dec,
            "kernel_conv": kernel_conv,
            "stride": stride,
            "padding": padding,
            "out_conv": out_conv,
            "key": key,
        }

    def encode(self, x, *args, **kwargs):
        new_encode = lambda x: self._encode(x, *args, **kwargs)
        return jax.vmap(new_encode, in_axes=[-1], out_axes=-1)(x)

    def perform_in_latent(self, y, *args, **kwargs):
        return self._perform_in_latent(y, self.k_max, *args, **kwargs)

    def decode(self, y, *args, **kwargs):
        new_decode = lambda x: self._decode(x, *args, **kwargs)
        return jax.vmap(new_decode, in_axes=[-1], out_axes=-1)(y)


class Vanilla_AE_CNN(Strong_RRAE_CNN):
    """Vanilla Autoencoder.

    Subclass for the Vanilla AE, basically the strong RRAE with
    k_max = -1, hence returning all the modes with no truncation.
    """

    def __init__(
        self,
        data,
        latent_size,
        width_enc=64,
        depth_enc=1,
        width_dec=64,
        depth_dec=6,
        *,
        key,
        **kwargs
    ):
        if "k_max" in kwargs.keys():
            if kwargs["k_max"] != -1:
                warnings.warn("k_max can not be specified for Vanilla_AE_MLP, switching to -1 (all modes)")
            kwargs.pop("k_max")
        super().__init__(
            data,
            latent_size,
            -1,
            width_enc,
            depth_enc,
            width_dec,
            depth_dec,
            key=key,
            **kwargs
        )


class Weak_RRAE_CNN(Strong_RRAE_CNN):
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
        self,
        data,
        latent_size,
        k_max,
        width_enc=64,
        depth_enc=1,
        width_dec=64,
        depth_dec=6,
        *,
        key,
        **kwargs
    ):
        super().__init__(
            data,
            latent_size,
            -1,
            width_enc,
            depth_enc,
            width_dec,
            depth_dec,
            key=key,
            **kwargs
        )
        self.v_vt = v_vt_class(latent_size, data.shape[-1], k_max, key=key)
