import shutil
from equinox.nn import LayerNorm
from equinox.nn._linear import Linear
from equinox._vmap_pmap import filter_vmap
from equinox._module import field, Module
from equinox._filters import is_array
from equinox._doc_utils import doc_repr
import math
from jaxtyping import Array, PRNGKeyArray
import jax.random as jrandom
from typing import (
    Literal,
    Optional,
    Union,
)
from collections.abc import Callable
import jax.numpy as jnp
import jax
import pdb
import matplotlib.pyplot as plt
import equinox as eqx
import jax.nn as jnn
import jax.random as jr
import optax
from operator import itemgetter
import time
import pickle
import numpy as np
import json
import os
import dill
from jax import config
import jax.tree_util as jtu

config.update("jax_enable_x64", True)


_identity = doc_repr(lambda x: x, "lambda x: x")
_relu = doc_repr(jnn.relu, "<function relu>")


def finite_diff(y, t):
    return jnp.hstack([(y[1] - y[0]) / (t[1] - t[0]), jnp.diff(y) / jnp.diff(t)])


def cumtrapz(y, t):
    return jnp.hstack([0, jnp.cumsum((y[:-1] + y[1:]) * (t[1:] - t[:-1]) / 2.0)])


def into_continuous(y_vals, ts, slope=1):
    jumps = jnp.concatenate([jnp.abs(jnp.diff(y_vals)), jnp.expand_dims(0, 0)])
    to_the_rgiht = jnp.concatenate([jnp.expand_dims(0, 0), (jumps == 1)[:-1]])
    jumps = jumps + to_the_rgiht
    if (jumps == 2).any():
        idx1 = list(jnp.where(jumps == 1)[0])
        idx2 = jnp.where(jumps == 2)[0]
        new_idx1 = list(idx1)
        for i2 in idx2:
            new_idx1.insert(jnp.searchsorted(jnp.array(idx1), i2), i2)
            new_idx1.insert(jnp.searchsorted(jnp.array(idx1), i2), i2)
        new_idx1 = jnp.array(new_idx1)
    else:
        new_idx1 = jnp.where(jumps == 1)[0]
    t_jumps = jnp.concatenate(
        [jnp.expand_dims(ts[0], 0), ts[new_idx1], jnp.expand_dims(ts[-1], 0)]
    )
    t_half = jnp.concatenate(
        [
            jnp.expand_dims(ts[0], 0),
            jnp.sum(jnp.reshape(t_jumps, (-1, 2)), axis=1) / 2,
            jnp.expand_dims(ts[-1], 0),
        ]
    )

    slopes = slope * jnp.array(
        [-1 if i % 2 == 0 else 1 for i in range(t_half.shape[0] - 1)]
    )
    if y_vals[0] == 0:
        slopes = slopes * -1

    def to_scan(state, inp):
        bn = (inp[1] - inp[0]) * inp[2] + state[0]
        new_inc = bn
        return ((new_inc, state[1].at[state[2]].set(bn), state[2] + 1), None)

    b0 = -slopes[0] * t_half[0]
    biases = jnp.concatenate(
        [
            jnp.expand_dims(b0, 0),
            jax.lax.scan(
                to_scan,
                (b0, jnp.zeros((slopes.shape[0] - 1,)), 0),
                (slopes[1:], slopes[:-1], t_half[1:-1]),
            )[0][1],
        ]
    )
    idx_t = jnp.searchsorted(t_half, ts, "right")
    idx_t = idx_t.at[-1].set(idx_t[-2]) - 1
    cont_sol = slopes[idx_t] * ts + biases[idx_t]
    return cont_sol


class WX(Module, strict=True):
    layers: tuple[Linear, ...]
    dropout: eqx.nn.Dropout
    use_bias: bool = field(static=True)
    use_final_bias: bool = field(static=True)
    in_size: Union[int, Literal["scalar"]] = field(static=True)
    out_size: Union[int, Literal["scalar"]] = field(static=True)
    width_size: int = field(static=True)
    depth: int = field(static=True)

    def __init__(
        self,
        in_size,
        out_size,
        width_size,
        depth,
        dropout,
        use_bias=True,
        use_final_bias=True,
        *,
        key,
    ):

        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(Linear(in_size, out_size, use_final_bias, key=keys[0]))
        else:
            layers.append(Linear(in_size, width_size, use_bias, key=keys[0]))
            for i in range(depth - 1):
                layers.append(Linear(width_size, width_size, use_bias, key=keys[i + 1]))
            layers.append(Linear(width_size, out_size, use_final_bias, key=keys[-1]))
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.dropout = dropout
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @jax.named_scope("eqx.nn.MLP")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.dropout(x, key=key)
        x = self.layers[-1](x)
        return x


class WX_Func(eqx.Module):
    WX_: WX
    mat_end: jnp.array
    post_proc_func: Callable

    def __init__(
        self,
        data_size,
        width_size,
        depth,
        out_size=None,
        dropout=0,
        *,
        key,
        mat_end=None,
        post_proc_func=_identity,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if out_size is None:
            out_size = data_size

        self.WX_ = WX(
            in_size=data_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            dropout=eqx.nn.Dropout(dropout),
            key=key,
        )
        self.mat_end = mat_end
        self.post_proc_func = post_proc_func

    def __call__(self, y, k=None, train=False):
        if not train and self.mat_end is not None:
            return self.post_proc_func(self.mat_end @ self.WX_(y, key=k))
        if not train:
            return self.post_proc_func(self.WX_(y, key=k))
        else:
            return self.WX_(y, key=k)


class Linear(Module, strict=True):
    """Performs a linear transformation."""

    weight: Array
    bias: Optional[Array]
    in_features: Union[int, Literal["scalar"]] = field(static=True)
    out_features: Union[int, Literal["scalar"]] = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `in_features`: The input size. The input to the layer should be a vector of
            shape `(in_features,)`
        - `out_features`: The output size. The output from the layer will be a vector
            of shape `(out_features,)`.
        - `use_bias`: Whether to add on a bias as well.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_features` also supports the string `"scalar"` as a special value.
        In this case the input to the layer should be of shape `()`.

        Likewise `out_features` can also be a string `"scalar"`, in which case the
        output from the layer will have shape `()`.
        """
        wkey, bkey = jrandom.split(key, 2)
        in_features_ = 1 if in_features == "scalar" else in_features
        out_features_ = 1 if out_features == "scalar" else out_features
        lim = 1 / math.sqrt(in_features_)
        self.weight = jrandom.uniform(
            wkey, (out_features_, in_features_), minval=-lim, maxval=lim
        )
        if use_bias:
            self.bias = jrandom.uniform(bkey, (out_features_,), minval=-lim, maxval=lim)
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

    @jax.named_scope("eqx.nn.Linear")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_features,)`. (Or shape
            `()` if `in_features="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        !!! info

            If you want to use higher order tensors as inputs (for example featuring "
            "batch dimensions) then use `jax.vmap`. For example, for an input `x` of "
            "shape `(batch, in_features)`, using
            ```python
            linear = equinox.nn.Linear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.

        **Returns:**

        A JAX array of shape `(out_features,)`. (Or shape `()` if
        `out_features="scalar"`.)
        """

        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = self.weight @ x
        if self.bias is not None:
            x = x + jnp.expand_dims(self.bias, -1)
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x


class MLP_dropout(Module, strict=True):
    layers: tuple[Linear, ...]
    activation: Callable
    dropout: eqx.nn.Dropout
    use_bias: bool = field(static=True)
    use_final_bias: bool = field(static=True)
    in_size: Union[int, Literal["scalar"]] = field(static=True)
    out_size: Union[int, Literal["scalar"]] = field(static=True)
    width_size: int = field(static=True)
    depth: int = field(static=True)

    def __init__(
        self,
        in_size,
        out_size,
        width_size,
        depth,
        dropout,
        activation=_relu,
        use_bias=True,
        use_final_bias=True,
        *,
        key,
        **kwargs,
    ):

        keys = jrandom.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(Linear(in_size, out_size, use_final_bias, key=keys[0]))
        else:
            if not isinstance(width_size, list):
                width_size = [width_size] * depth
            layers.append(Linear(in_size, width_size[0], use_bias, key=keys[0]))
            for i in range(depth - 1):
                layers.append(
                    Linear(width_size[i], width_size[i + 1], use_bias, key=keys[i + 1])
                )
            layers.append(
                Linear(width_size[depth - 1], out_size, use_final_bias, key=keys[-1])
            )
        self.layers = tuple(layers)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.dropout = dropout
        self.activation = [
            filter_vmap(filter_vmap(lambda: activation, axis_size=w), axis_size=depth)()
            for w in width_size
        ]
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @jax.named_scope("eqx.nn.MLP")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        for i, (layer, act) in enumerate(zip(self.layers[:-1], self.activation)):
            x = layer(x)
            x = self.dropout(x, key=key)
            layer_activation = jtu.tree_map(lambda x: x[i] if is_array(x) else x, act)
            x = filter_vmap(lambda a, b: a(b))(layer_activation, x)

        x = self.layers[-1](x)
        # if self.out_size == "scalar":
        x = self.activation[-1](x)
        # else:
        #     x = filter_vmap(lambda a, b: a(b))(self.final_activation, x)
        return x


class Func(eqx.Module):
    mlp: MLP_dropout
    mat_end: jnp.array
    post_proc_func: Callable

    def __init__(
        self,
        data_size,
        width_size,
        depth,
        out_size=None,
        dropout=0,
        *,
        key,
        mat_end=None,
        inside_activation=None,
        activation=None,
        post_proc_func=_identity,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if out_size is None:
            out_size = data_size

        def my_f(x):
            return jnp.exp(-(x**2))

        activation = _identity if activation is None else activation
        inside_activation = (
            jnn.softplus if inside_activation is None else inside_activation
        )

        self.mlp = MLP_dropout(
            in_size=data_size,
            out_size=out_size,
            width_size=width_size,
            depth=depth,
            activation=inside_activation,
            dropout=eqx.nn.Dropout(dropout),
            final_activation=activation,
            key=key,
        )
        self.mat_end = mat_end
        self.post_proc_func = post_proc_func

    def call_per_input(self, y, k=None, train=False):
        if not train and self.mat_end is not None:
            return self.post_proc_func(self.mat_end @ self.mlp(y, key=k))
        if not train:
            return self.post_proc_func(self.mlp(y, key=k))
        else:
            return self.mlp(y, key=k)

    def __call__(self, y, k=None, train=False):
        return jnp.squeeze(
            jax.vmap(
                lambda x, kv, tr: self.call_per_input(jnp.expand_dims(x, -1), kv, tr),
                in_axes=[0, None, None],
            )(y, k, train)
        ).T


class v_vt_class(eqx.Module):
    v: jnp.array
    vt: jnp.array

    def __init__(self, dim0, dim1, num_nodes=1, **kwargs):
        super().__init__(**kwargs)
        self.v = jrandom.uniform(
            jrandom.PRNGKey(1200), (dim0, num_nodes), minval=-1, maxval=1
        )
        self.v = jax.vmap(lambda x: x / jnp.linalg.norm(x), in_axes=[-1])(self.v).T
        self.vt = jrandom.uniform(
            jrandom.PRNGKey(0), (dim1, num_nodes), minval=-1, maxval=1
        )  # no T
        self.vt = jax.vmap(lambda x: x / jnp.linalg.norm(x), in_axes=[-1])(self.vt)

    def __call__(self):
        normalized_v = jax.vmap(lambda x: x / jnp.linalg.norm(x), in_axes=[-1])(
            self.v
        ).T
        return jnp.dot(self.vt.T, normalized_v.T).T


class simple_koop_operator(eqx.Module):
    func_encode: Func
    func_decode: Func
    func_interp: Func
    v_vt1: v_vt_class

    def __init__(
        self, func_enc, func_dec, func_interp=None, v_vt1=None, **kwargs
    ) -> None:
        self.func_encode = func_enc
        self.func_decode = func_dec
        self.func_interp = func_interp
        self.v_vt1 = v_vt1

    def __call__(
        self, ys, n_mode=1, key=jrandom.PRNGKey(0), train=False, pv=None, v1_now=None
    ):
        x = self.func_encode(ys)
        if n_mode != -1:
            u, s, v = jnp.linalg.svd(x, full_matrices=False)
            sigs = s[:n_mode]
            v_now = v[:n_mode, :]
            u_now = u[:, :n_mode]
            xs_m = jnp.sum(
                jax.vmap(lambda u, s, v: s * jnp.outer(u, v), in_axes=[-1, 0, 0])(
                    u_now, sigs, v_now
                ).T,
                axis=-1,
            )
        else:
            xs_m = x
            u_now = None
            v_now = None
            sigs = None
        y = self.func_decode(xs_m, key, train).T
        return x, y, xs_m, (u_now, v_now, sigs)


def find_weighted_loss(terms, weight_vals=None):
    terms = jnp.asarray(terms, dtype=jnp.float32)
    total = jnp.sum(jnp.abs(terms))
    if weight_vals is None:
        weights = jnp.asarray([jnp.abs(term) / total for term in terms])
    else:
        weights = weight_vals
    res = jnp.multiply(terms, weights)
    return sum(res)


def dataloader(arrays, batch_size, p_vals=None, *, key):
    dataset_size = arrays[0].shape[0]
    # assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    kk = 0

    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            arrs = tuple(
                itemgetter(*batch_perm)(array) for array in arrays
            )  # Works for lists and arrays
            if batch_size != 1:
                yield [arr if None in arr else jnp.array(arr) for arr in arrs]
            else:
                yield [
                    [arr] if arr is None else jnp.expand_dims(jnp.array(arr), axis=0)
                    for arr in arrs
                ]
            start = end
            end = start + batch_size
        kk += 1


def make_model(
    key,
    data_size,
    data_size_end,
    mul_latent,
    dropout,
    WX_,
    v_vt,
    num_modes_vvt,
    width_enc,
    depth_enc,
    width_dec,
    depth_dec,
    activation_enc,
    activation_dec,
    post_proc_func=_identity,
    interp=True,
    width_i=64,
    depth_i=2,
    p_vals=None,
    modes=None,
    out_size=None,
    **kwargs,
):
    func_encode = Func(
        data_size,
        width_enc,
        depth_enc,
        out_size=int(data_size * mul_latent),
        activation=activation_enc,
        key=key,
    )
    if WX_:
        func_encode = WX(data_size, int(data_size * mul_latent))
    if out_size is None:
        out_size = data_size
    func_decode = Func(
        int(data_size * mul_latent),
        width_dec,
        depth_dec,
        out_size=out_size,
        dropout=dropout,
        activation=activation_dec,
        key=key,
        post_proc_func=post_proc_func,
    )
    if interp:
        func_interp = Func(
            p_vals.shape[-1], width_i, depth_i, out_size=modes, dropout=dropout, key=key
        )
    else:
        func_interp = None
    print(
        f"Dimension of latent space is {int(data_size*mul_latent)} and data size is {data_size}"
    )
    if v_vt:
        vvt = v_vt_class(
            int(data_size * mul_latent), data_size_end, num_nodes=num_modes_vvt
        )

    model = (
        simple_koop_operator(func_encode, func_decode, func_interp, None)
        if not v_vt
        else simple_koop_operator(func_encode, func_decode, func_interp, vvt)
    )
    return model


class CNN(eqx.Module):
    layers: list

    def __init__(
        self,
        kernel_conv,
        stride,
        padding,
        kernel_pool,
        out_conv,
        data_dim0,
        data_dim1,
        out,
        width,
        depth,
        dropout,
        *,
        key,
        **kwargs,
    ):
        key1, key2 = jax.random.split(key, 2)
        # Standard CNN setup: convolutional layer, followed by flattening,
        # with a small MLP on top.

        self.layers = [
            eqx.nn.Conv2d(
                1,
                int(out_conv / 2),
                stride=stride,
                padding=padding,
                kernel_size=kernel_conv,
                key=key1,
            ),
            jnn.softplus,
            eqx.nn.Conv2d(
                int(out_conv / 2),
                out_conv,
                stride=stride,
                padding=padding,
                kernel_size=kernel_conv,
                key=key1,
            ),
            jnn.softplus,
            lambda x: jnp.expand_dims(jnp.ravel(x), -1),
            MLP_dropout(
                out_conv * (data_dim0 + 2 * padding) * (data_dim1 + 2 * padding),
                out,
                width,
                depth,
                eqx.nn.Dropout(dropout),
                key=key2,
            ),
        ]

    def call_per_input(self, x):
        for layer in self.layers:
            x = layer(x)
        return jnp.squeeze(x).T

    def __call__(self, x, *args, **kwargs):
        return jnp.squeeze(jax.vmap(self.call_per_input)(x)).T


class CNN_trans(eqx.Module):
    layers: list

    def __init__(
        self,
        kernel_conv,
        stride,
        padding,
        kernel_pool,
        out_conv,
        data_dim0,
        data_dim1,
        out,
        width,
        depth,
        dropout,
        *,
        key,
        **kwargs,
    ):
        key1, key2, key3, key4 = jax.random.split(key, 4)

        width = jnp.flip(np.array(width)).tolist()

        self.layers = [
            lambda x: jnp.expand_dims(jnp.ravel(x), -1),
            MLP_dropout(
                out,
                out_conv * (data_dim0 + 2 * padding) * (data_dim1 + 2 * padding),
                width,
                depth,
                eqx.nn.Dropout(dropout),
                key=key2,
            ),
            lambda x: jnp.reshape(
                x, (out_conv, data_dim0 + 2 * padding, data_dim1 + 2 * padding)
            ),
            jnn.softplus,
            eqx.nn.ConvTranspose2d(
                out_conv,
                int(out_conv / 2),
                stride=stride,
                padding=padding,
                kernel_size=kernel_conv,
                key=key1,
            ),
            jnn.softplus,
            eqx.nn.ConvTranspose2d(
                int(out_conv / 2),
                1,
                stride=stride,
                padding=padding,
                kernel_size=kernel_conv,
                key=key1,
            ),
        ]

    def call_per_batch(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x, *args, **kwds):
        return jax.vmap(self.call_per_batch, in_axes=[-1])(x)


def make_conv_model(
    key,
    data_shape,
    data_size,
    data_size_end,
    mul_latent,
    dropout,
    WX_,
    v_vt,
    num_modes_vvt,
    width_enc,
    depth_enc,
    width_dec,
    depth_dec,
    activation_enc,
    activation_dec,
    kernel_conv,
    stride,
    padding,
    kernel_pool,
    out_conv,
    post_proc_func=_identity,
    interp=True,
    width_i=64,
    depth_i=2,
    p_vals=None,
    modes=None,
    out_size=None,
    **kwargs,
):
    key1, key2, key3 = jr.split(key, 3)
    data_size = data_shape[-1] * data_shape[-2]
    func_encode = CNN(
        kernel_conv,
        stride,
        padding,
        kernel_pool,
        out_conv,
        data_shape[2],
        data_shape[3],
        int(data_size * mul_latent),
        width_enc,
        depth_enc,
        dropout,
        key=key1,
    )
    if WX_:
        func_encode = WX(data_size, int(data_size * mul_latent))
    if out_size is None:
        out_size = data_size

    func_decode = CNN_trans(
        kernel_conv,
        stride,
        padding,
        kernel_pool,
        out_conv,
        data_shape[2],
        data_shape[3],
        int(data_size * mul_latent),
        width_dec,
        depth_dec,
        dropout,
        key=key2,
    )

    if interp:
        func_interp = Func(
            p_vals.shape[-1], width_i, depth_i, out_size=modes, dropout=dropout, key=key
        )
    else:
        func_interp = None

    print(
        f"Dimension of latent space is {int(data_size*mul_latent)} and data size is {data_size}"
    )

    if v_vt:
        vvt = v_vt_class(
            int(data_size * mul_latent), data_size_end, num_nodes=num_modes_vvt
        )

    model = (
        simple_koop_operator(func_encode, func_decode, func_interp, None)
        if not v_vt
        else simple_koop_operator(func_encode, func_decode, func_interp, vvt)
    )
    return model


def train_loop_RRAE(
    ys,
    output,
    step_st=[3000, 3000],  # 000, 8000],
    lr_st=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    width_enc=64,
    depth_enc=2,
    width_dec=64,
    depth_dec=6,
    activation_enc=None,
    activation_dec=None,
    loss_func=None,
    post_proc_func=_identity,
    seed=5678,
    print_every=20,
    stagn_every=100,  # 80
    reg=False,
    batch_size_st=[16, 16, 16, 16, 32],
    n_mode=-1,
    num_modes_vvt=None,
    v_vt=False,
    mul_latent=0.05,  # 0.05
    dropout=0,
    WX_=True,
    p_vals=None,
    mat_end=None,
    widgetf=None,
    method=None,
    mul_lr=None,
    add_interp=0,
    width_i=64,
    depth_i=2,
    interp=False,
    training_intr=[False, False, False, False, False],
    kernel_conv=None,
    stride=None,
    padding=None,
    kernel_pool=None,
    out_conv=None,
    regression=False,
    **kwargs,
):

    key = jr.PRNGKey(seed)
    loader_key, dropout_key = jr.split(key, 2)
    seed = 0
    model_key = jr.PRNGKey(seed)
    parameters = {
        "kernel_conv": kernel_conv,
        "stride": stride,
        "padding": padding,
        "kernel_pool": kernel_pool,
        "out_conv": out_conv,
        "data_shape": ys.shape,
        "width_i": width_i,
        "out_size": int(output.shape[-1]),
        "depth_i": depth_i,
        "modes": num_modes,
        "interp": interp,
        "p_vals": p_vals,
        "data_size": int(ys.shape[-1]),
        "activation_enc": activation_enc,
        "post_proc_func": post_proc_func,
        "activation_dec": activation_dec,
        "data_size_end": int(ys.shape[-1]),
        "mul_latent": mul_latent,
        "dropout": dropout,
        "WX_": WX_,
        "v_vt": v_vt,
        "num_modes_vvt": num_modes_vvt,
        "width_enc": width_enc,
        "depth_enc": depth_enc,
        "width_dec": width_dec,
        "depth_dec": depth_dec,
        "seed": seed,
    }
    model_func = make_model if len(ys.shape) == 2 else make_conv_model
    model = model_func(key=model_key, **parameters)
    # UP TO HERE IT IS WORKING

    if loss_func is None:

        def loss_func(x1, x2):
            return jnp.linalg.norm(x1 - x2) / jnp.linalg.norm(x2) * 100

    @eqx.filter_value_and_grad(has_aux=True)
    def grad_loss(model, input, out, idx, key, pv, y_interp, u_now):
        _, y, _, svd = model(input, n_mode, key, True, pv, u_now)
        wv = jnp.array([1.0,])
        return find_weighted_loss([loss_func(y, out)], weight_vals=wv), (y)

    @eqx.filter_value_and_grad
    def grad_loss_v_vt(model, input, idx, key, pv, y_interp, u_now):
        x, y, _, svd, _, _ = model(input, n_mode, key, True)
        # if y_interp is not None:
        #     v1 = jax.vmap(lambda x: x / jnp.linalg.norm(x), in_axes=[-1])(model.v_vt1.v).T
        #     v_vec = model.v_vt1.vt
        #     _, y_pred_interp, _, _ = p_of_dim_1(v1, v_vec, pv, p_interp_now, model, all_ids)
        #     wv = jnp.array([1., 1., 1.]) # 4 for mult_freqs
        #     return find_weighted_loss([loss_func(y, input), jnp.linalg.norm(x-model.v_vt1()[:, idx])/jnp.linalg.norm(x)*100, loss_func(y_pred_interp, y_interp)], weight_vals=wv) #
        wv = jnp.array([1.0, 1.0])
        return find_weighted_loss(
            [
                loss_func(y, input),
                jnp.linalg.norm(x - model.v_vt1()[:, idx]) / jnp.linalg.norm(x) * 100,
            ],
            weight_vals=wv,
        )

    @eqx.filter_jit
    def make_step(input, out, model, opt_state, bs, idx, key, pv, y_interp_now, v1_now):
        loss_func = grad_loss_v_vt if v_vt else grad_loss
        (loss, aux), grads = loss_func(
            model, input, out, idx, key, pv, y_interp_now, v1_now
        )
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state, aux

    print("Training the RRAE...")
    filter_spec = jtu.tree_map(lambda _: False, model)
    if v_vt:
        is_acc = eqx.tree_at(
            lambda tree: (tree.v_vt1.v, tree.v_vt1.vt),
            filter_spec,
            replace=(True, True),
        )
        is_not_acc = jtu.tree_map(lambda x: not x, is_acc)
    else:
        is_acc = eqx.tree_at(
            lambda tree: jtu.tree_leaves(tree.func_interp),
            filter_spec,
            replace=(True,) * len(jtu.tree_leaves(filter_spec.func_interp)),
        )
        is_not_acc = jtu.tree_map(lambda x: not x, is_acc)

    for steps, lr, batch_size, mul_l, bool_val in zip(
        step_st, lr_st, batch_size_st, mul_lr, training_intr
    ):
        stagn_num = 0
        loss_old = jnp.inf
        t_t = 0

        if v_vt:
            optim = optax.chain(
                optax.masked(optax.adabelief(lr), is_not_acc),
                optax.masked(optax.adabelief(mul_l * lr), is_acc),
            )
        else:
            #     if not bool_val:
            #         optim = optax.chain(optax.masked(optax.adabelief(lr), is_not_acc), optax.masked(optax.set_to_zero(), is_acc))
            #         v1_now = None
            #     else:
            #         optim = optax.chain(optax.masked(optax.set_to_zero(), is_not_acc), optax.masked(optax.adabelief(lr), is_acc))
            #         sv = jnp.expand_dims(svd[2], 0)
            #         u_vec = svd[0]
            #         v1_now = jnp.multiply(sv, u_vec)
            #
            # if bool_val:
            #     optim = optax.chain(optax.masked(optax.adabelief(lr), is_not_acc), optax.masked(optax.adabelief(mul_l*lr), is_acc))
            # else:
            optim = optax.adabelief(lr)

        filtered_model = eqx.filter(model, eqx.is_inexact_array)
        opt_state = optim.init(filtered_model)

        keys = jr.split(dropout_key, steps)

        if (batch_size > ys.shape[-1]) or batch_size == -1:
            batch_size = ys.shape[-1]

        for step, (yb, pv, out, idx, key) in zip(
            range(steps),
            dataloader(
                [ys, p_vals, output, jnp.arange(0, ys.shape[-1], 1), keys],
                batch_size,
                key=loader_key,
            ),
        ):
            start = time.time()
            loss, model, opt_state, aux = make_step(
                yb, out, model, opt_state, batch_size, idx, key[0], pv, None, None
            )
            end = time.time()
            t_t += end - start
            if (step % stagn_every) == 0:
                if jnp.abs(loss_old - loss) / jnp.abs(loss_old) * 100 < 1:
                    stagn_num += 1
                    if stagn_num > 10:
                        print("Stagnated....")
                        break
                loss_old = loss

            if (step % print_every) == 0 or step == steps - 1:
                if regression:
                    pred = np.zeros_like(aux)
                    pred[aux.argmax(0), np.arange(aux.shape[-1])] = 1
                    print(
                        f"Step: {step}, Loss: {loss}, Computation time: {t_t}, Accuracy: {jnp.sum((pred == out.T))/pred.size*100}"
                    )
                else:
                    print(f"Step: {step}, Loss: {loss}, Computation time: {t_t}")
                if widgetf is not None:
                    widgetf.setText(f"Loss: {loss}")
                t_t = 0

        model = eqx.nn.inference_mode(model)
        x, y, x_m, svd = model(ys, n_mode, 0, True)
        model = eqx.nn.inference_mode(model)
    model = eqx.nn.inference_mode(model)
    _, y_o, _, _ = model(ys, n_mode, 0, False)
    if method == "weak":
        u_vec, sv, v_vec = determine_u_now(x_m, eps=1, ret=True, full_matrices=False)
        v1 = jax.vmap(lambda x: x / jnp.linalg.norm(x), in_axes=[-1])(model.v_vt1.v).T
        v_vec = model.v_vt1.vt
    else:
        sv = jnp.expand_dims(svd[2], 0)
        u_vec = svd[0]
        v1 = jnp.multiply(sv, u_vec)
        v_vec = svd[1]
        print(f"First {sv.shape[0]} singular values are {sv}")

    return y, x_m, model, v1, v_vec, parameters, y_o, ys, p_vals


def my_vmap(func, to_array=True):
    def map_func(arrays, args=None):
        sols = []
        for elems in arrays:
            if args is None:
                sols.append(func(elems))
            else:
                sols.append(func(elems, *args))
        try:
            return [jnp.squeeze(jnp.stack(sol, axis=0)) for sol in sols]

        except (TypeError, ValueError) as e:
            if isinstance(e, ValueError):
                return sols
            if to_array:
                return jnp.array(sols)
            else:
                return sols

    return map_func


def my_vmap_multiple_arrays(func, to_array=True):
    def map_func(*arrays, args=None, kwargs=None):
        sols = []
        for elems in zip(*arrays):
            if (args is None) and (kwargs is None):
                sols.append(func(*elems))
            elif (args is not None) and (kwargs is not None):
                sols.append(func(*elems, *args, **kwargs))
            elif args is not None:
                print("A")
                sols.append(func(*elems, *args))
            else:
                sols.append(func(*elems, **kwargs))
        try:
            if isinstance(sols[0], list) or isinstance(sols[0], tuple):
                final_sols = []
                for i in range(len(sols[0])):
                    final_sols.append(jnp.array([sol[i] for sol in sols]))
                return final_sols
            return jnp.array([jnp.squeeze(jnp.stack(sol, axis=0)) for sol in sols])
        except:
            pdb.set_trace()
            if to_array:
                return jnp.array(sols)
            else:
                return sols

    return map_func


def plot_confusion_matrix(
    cm, target_names, title="Confusion matrix", cmap=None, normalize=True
):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap("Blues")

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(
                j,
                i,
                "{:0.4f}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        else:
            plt.text(
                j,
                i,
                "{:,}".format(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel(
        "Predicted label\naccuracy={:0.4f}; misclass={:0.4f}".format(accuracy, misclass)
    )


def post_process_class(
    p_vals,
    p_test,
    problem,
    method,
    x_train,
    y_pred_train,
    v_train,
    vt_train,
    v_test,
    y_pred_test,
    ys,
    ys_test,
    y_original=None,
    y_pred_train_o=None,
    y_test_o=None,
    y_pred_test_o=None,
    changed=False,
    x_test_modes=None,
    test=True,
    file=None,
    pp=True,
    out_train=None,
    out_test=None,
    **kwargs,
):
    from sklearn.metrics import confusion_matrix

    pred_train = np.zeros_like(y_pred_train)
    pred_train[y_pred_train.argmax(0), np.arange(y_pred_train.shape[-1])] = 1

    pred_test = np.zeros_like(y_pred_test)
    pred_test[y_pred_test.argmax(0), np.arange(y_pred_test.shape[-1])] = 1

    error_train = jnp.sum(pred_train != out_train) / out_train.size * 100
    error_test = jnp.sum(pred_test != out_test) / out_test.size * 100
    accuracy_train = jnp.sum(pred_train == out_train) / out_train.size * 100
    accuracy_test = jnp.sum(pred_test == out_test) / out_test.size * 100

    fig1 = plt.figure(1)
    fig1.set_size_inches(18.5, 10.5)
    fig1 = fig1.add_subplot(1, 2, 1)
    plt.subplots_adjust(hspace=0.8)
    plt.subplot(1, 2, 1)
    cm_train = confusion_matrix(np.argmax(out_train, 0), np.argmax(pred_train, 0))

    names = []
    for i in range(out_train.shape[0]):
        if i == 0:
            names.append(f"{i+1} Step")
        else:
            names.append(f"{i+1} Steps")

    plot_confusion_matrix(
        cm_train, names, title="Confusion matrix for train", normalize=True
    )
    plt.subplot(1, 2, 2)
    cm_test = confusion_matrix(np.argmax(out_test, 0), np.argmax(pred_test, 0))
    plot_confusion_matrix(
        cm_test, names, title="Confusion matrix for test", normalize=True
    )
    plt.show()

    print(f"Accuracy for train is {accuracy_train} and for test is {accuracy_test}")

    return error_train, error_test, None, None


def process_class(y_train, y_test, out_train, out_test, model, n_mode, **kwargs):
    pred_test = model(y_test, n_mode, 0, True)[1]
    pred_test_org = model(y_train, n_mode, 0, False)[1]
    return None, pred_test, pred_test_org, None


def post_process_interp(
    p_vals,
    p_test,
    problem,
    method,
    x_train,
    y_pred_train,
    v_train,
    vt_train,
    v_test,
    y_pred_test,
    ys,
    ys_test,
    y_original=None,
    y_pred_train_o=None,
    y_test_o=None,
    y_pred_test_o=None,
    changed=False,
    x_test_modes=None,
    test=True,
    file=None,
    pp=True,
    out_train=None,
    out_test=None,
    **kwargs,
):
    import matplotlib

    matplotlib.rc("xtick", labelsize=15)
    matplotlib.rc("ytick", labelsize=15)
    plt.rcParams["figure.figsize"] = (10, 12)

    error_train = jnp.linalg.norm(y_pred_train - ys) / jnp.linalg.norm(ys) * 100
    print(f"Error for train is {error_train}")
    if pp:
        print("Plotting decoder results for train")
        for i, (yt, yr) in enumerate(zip(ys.T, y_pred_train.T)):
            lab = r"$\tilde{x}$-train" if i == 0 else "_no_legend_"
            lab_ = r"$x$-train" if i == 0 else "_no_legend_"
            plt.plot(yt, color="blue", label=lab_)
            plt.plot(yr, color="red", linestyle="--", label=lab)

        plt.xlabel(r"$t$", fontsize=20)
        plt.ylabel(r"$x, \tilde{x}$", fontsize=20)
        plt.legend()
        plt.title(f"Decoder result over train set", fontsize=20)
        if file is not None:
            plt.savefig(os.path.join(file, f"decoder_train_{problem}_{method}.pdf"))
        plt.show()
    if y_original is not None:
        error_orig = (
            jnp.linalg.norm(y_original - y_pred_train_o)
            / jnp.linalg.norm(y_original)
            * 100
        )
        print(f"Error for original train is {error_orig}")
        if pp:
            print("Plotting decoder results for original train")
            for i, (yt, yr) in enumerate(zip(y_pred_train_o.T, y_original.T)):
                lab = r"$\tilde{x}$-train" if i == 0 else "_no_legend_"
                lab_ = r"$x$-train" if i == 0 else "_no_legend_"
                plt.plot(yr, color="blue", label=lab_)
                plt.plot(yt, color="red", linestyle="--", label=lab)
            plt.xlabel(r"$t$", fontsize=20)
            plt.ylabel(r"$x, \tilde{x}$", fontsize=20)
            plt.legend()
            plt.title(f"Decoder result over original train set", fontsize=20)
            if file is not None:
                plt.savefig(os.path.join(file, f"decoder_orig_{problem}_{method}.pdf"))
            plt.show()
        error_train_o = (
            jnp.linalg.norm(y_pred_train_o - y_original)
            / jnp.linalg.norm(y_original)
            * 100
        )
    else:
        error_train_o = None
    x_train_modes = jax.vmap(lambda o1, o2: jnp.outer(o1, o2), in_axes=[-1, 0])(
        v_train, vt_train
    )
    if test:
        error_test = (
            jnp.linalg.norm(y_pred_test - ys_test) / jnp.linalg.norm(ys_test) * 100
        )
        print(f"Error for test is {error_test}")
        print("Plotting latent space for test")
        if x_test_modes is None:
            x_test_modes = jax.vmap(lambda o1, o2: jnp.outer(o1, o2), in_axes=[-1, 0])(
                v_train, v_test
            )
        if x_test_modes.shape[1] < 1500:
            if pp:
                for i, (x_train, x_test) in enumerate(zip(x_train_modes, x_test_modes)):
                    for j, x_tr in enumerate(x_train.T):
                        lab = f"Train-mode{i}" if j == 0 else "_no_legend_"
                        plt.plot(x_tr, color="blue", label=lab)
                    for j, xt in enumerate(x_test.T):
                        lab_ = f"Test-mode{i}" if j == 0 else "_no_legend_"
                        plt.plot(xt, color="red", label=lab_)
                    plt.ylabel(r"$Y$", fontsize=20)
                    plt.xlabel(r"$t$", fontsize=20)
                    plt.legend()
                    plt.title(f"Latent space for mode {i}", fontsize=20)
                    if file is not None:
                        plt.savefig(os.path.join(file, f"latent_mode_{i}.pdf"))
                    plt.clf()
        if v_test is not None:
            if pp:
                for i, (v_tr, v_t) in enumerate(
                    zip(vt_train, v_test)
                ):  # , p_vals.T, p_test.T
                    if p_vals.shape[-1] == 1:
                        plt.scatter(
                            p_vals, v_tr, label=f"Train-coeffs{i}", color="blue"
                        )
                        plt.scatter(p_test, v_t, label=f"Test-coeffs{i}", color="red")
                    else:
                        plt.scatter(
                            jnp.arange(0, v_tr.shape[0], 1),
                            v_tr,
                            color="blue",
                            label=f"Train-coeffs{i}",
                        )
                        # plt.hlines(v_t, color="red", label=f"Test-coeffs{i}")
                    plt.ylabel(r"$coeffs$", fontsize=20)
                    plt.xlabel(r"$param$", fontsize=20)
                    plt.legend()
                    plt.title(f"Latent space coeffs", fontsize=20)
                    if file is not None:
                        plt.savefig(os.path.join(file, f"coeffs-mode{i}.pdf"))
                    plt.show()
        if pp:
            print("Plotting decoder results for test")
            for i, (yt, yr) in enumerate(zip(ys_test.T, y_pred_test.T)):
                lab = r"$\tilde{x}$-test" if i == 0 else "_no_legend_"
                lab_ = r"$x$-test" if i == 0 else "_no_legend_"
                plt.plot(yr, color="blue", label=lab)
                plt.plot(yt, color="red", linestyle="--", label=lab_)

            plt.ylabel(r"$x, \tilde{x}$", fontsize=20)
            plt.xlabel(r"$t$", fontsize=20)
            plt.legend()
            plt.title(f"Decoder result over test set", fontsize=20)
            if file is not None:
                plt.savefig(os.path.join(file, f"decoder_test_{problem}_{method}.pdf"))
            plt.show()
        if y_test_o is not None:
            error_orig_test = (
                jnp.linalg.norm(y_test_o - y_pred_test_o)
                / jnp.linalg.norm(y_test_o)
                * 100
            )
            print(f"Error for original test is {error_orig_test}")
            if pp:
                print("Plotting decoder results for original test")
                for i, (yt, yr) in enumerate(zip(y_pred_test_o.T, y_test_o.T)):
                    lab = r"$\tilde{x}$-test" if i == 0 else "_no_legend_"
                    lab_ = r"$x$-test" if i == 0 else "_no_legend_"
                    plt.plot(yr, color="blue", label=lab_)
                    plt.plot(yt, color="red", linestyle="--", label=lab)
                plt.xlabel(r"$t$", fontsize=20)
                plt.ylabel(r"$x, \tilde{x}$", fontsize=20)
                plt.legend()
                plt.title(f"Decoder result over original test set", fontsize=20)
                if file is not None:
                    plt.savefig(
                        os.path.join(file, f"decoder_orig_test_{problem}_{method}.pdf")
                    )
                plt.show()
            error_test_o = (
                jnp.linalg.norm(y_pred_test_o - y_test_o)
                / jnp.linalg.norm(y_test_o)
                * 100
            )
        else:
            error_test_o = None
    else:
        print("Plotting latent space for train")
        for i, x_train in enumerate(x_train_modes):
            plt.plot(x_train, color="blue")
            plt.title(f"Latent space for mode {i}")
            plt.clf()
        error_test = None
        error_test_o = None

    return error_train, error_test, error_train_o, error_test_o


def is_test_inside(p_vals, p_test):
    def check_func(p_v, p_t):
        return (jnp.min(p_v, axis=0) < p_t).all() & (p_t < jnp.max(p_v, axis=0)).all()

    if p_vals.shape[-1] == 1:
        return check_func(p_vals, p_test)
    elif p_vals.shape[-1] == 2:

        def is_in_rect(vals):
            return (
                (jnp.sum(vals, axis=1) == 2).any()
                and (jnp.sum(vals, axis=1) == 0).any()
                and ((jnp.sum(vals, axis=1) == 1) & (vals[:, 0] == 1)).any()
                and ((jnp.sum(vals, axis=1) == 1) & (vals[:, 0] == 0)).any()
            )

        def check_func_2(p_t):
            return is_in_rect(p_vals > p_t)

        return my_vmap(check_func_2)(p_test).all()
    else:
        return True


def determine_u_now(ys, eps=1, ret=False, full_matrices=True):
    u, sv, v = jnp.linalg.svd(ys, full_matrices=full_matrices)

    def to_scan(state, inp):
        u_n = u[:, inp]
        s_n = sv[inp]
        v_n = v[inp]
        pred = s_n * jnp.outer(u_n, v_n)
        return ((state[0].at[state[1]].set(pred), state[1] + 1), None)

    truncs = jnp.cumsum(
        jax.lax.scan(
            to_scan,
            (jnp.zeros((int(0.1 * sv.shape[0]), ys.shape[0], ys.shape[1])), 0),
            jnp.arange(0, int(0.1 * sv.shape[0]), 1),
        )[0][0],
        axis=0,
    )
    errors = jax.vmap(
        lambda app: jnp.linalg.norm(ys - app) / jnp.linalg.norm(ys) * 100
    )(truncs)
    n_mode = jnp.argmax(errors < eps)
    n_mode = n_mode if n_mode != 0 else 1
    print(f"Number of modes for initial V is {n_mode}")
    u_now = u[:, :n_mode]
    if ret:
        return u_now, sv[:n_mode], v[:n_mode, :]
    return u_now


def do_all_the_u_now_stuff(y_shift, y_test, eps=None):
    u_now = determine_u_now(y_shift, eps=eps)
    coeffs = u_now.T @ y_shift

    def to_norm(x):
        return (x - jnp.mean(x)) / jnp.std(x), jnp.mean(x), jnp.std(x)

    def to_norm_test(x, mean, std):
        return (x - mean) / std

    def inv_to_norm(x, mean, std):
        return x * std + mean

    norm_coeffs, means, stds = jax.vmap(to_norm)(coeffs)
    norm_coeffs_test = jax.vmap(to_norm_test)(u_now.T @ y_test, means, stds)

    def inv_func(x):
        return u_now @ jax.vmap(inv_to_norm)(x, means, stds)

    return norm_coeffs, norm_coeffs_test, inv_func


def plot_surfaces(ts, solutions_pred, solutions_true, idx_):
    from matplotlib import cm
    import numpy as np

    fig, axe = plt.subplots(nrows=2, ncols=2, subplot_kw={"projection": "3d"})
    axes = axe.flat

    X_vec, Y_vec = ts
    idx = np.argmax(jnp.diff(X_vec[:, 0]) < 0)
    X_vec = X_vec[: idx + 1]
    Y_vec = np.reshape(Y_vec, (idx + 1, -1))[:, 0]

    Z_n_true = np.reshape(solutions_true[idx_], (X_vec.shape[0], Y_vec.shape[0]))
    axes[0].plot_surface(
        X_vec, Y_vec, Z_n_true, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    axes[0].view_init(elev=0, azim=0, roll=0)

    Z_n_true = np.reshape(solutions_true[idx_], (X_vec.shape[0], Y_vec.shape[0]))
    surf = axes[1].plot_surface(
        X_vec, Y_vec, Z_n_true, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    axes[1].view_init(elev=90, azim=-90, roll=0)

    Z_n_pred = jnp.reshape(solutions_pred[idx_], (X_vec.shape[0], Y_vec.shape[0]))
    axes[2].plot_surface(
        X_vec, Y_vec, Z_n_pred, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    axes[2].view_init(elev=0, azim=0, roll=0)

    Z_n_pred = jnp.reshape(solutions_pred[idx_], (X_vec.shape[0], Y_vec.shape[0]))
    axes[3].plot_surface(
        X_vec, Y_vec, Z_n_pred, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    axes[3].view_init(elev=90, azim=-90, roll=0)

    fig.colorbar(surf, ax=axe.ravel().tolist())
    plt.show()


def norm_divide_return(
    ts,
    y_all,
    p_all,
    prop_train=0.8,
    pod=False,
    test_end=0,
    eps=1,
    output=None,
    conv=False,
):
    def norm_vec(x):
        return (x - jnp.mean(x)) / jnp.std(x)

    p_all = jnp.stack(my_vmap(lambda y: norm_vec(y))(p_all.T)).T

    if test_end == 0:
        res = jnp.stack(
            my_vmap(lambda p: (p > jnp.min(p)) & (p < jnp.max(p)))(p_all.T)
        ).T
        idx = jnp.linspace(0, res.shape[0] - 1, res.shape[0], dtype=int)
        cbt_idx = idx[jnp.sum(res, 1) == res.shape[1]]  # could be test
        permut_idx = jrandom.permutation(jrandom.PRNGKey(200), cbt_idx.shape[0])
        idx_test = cbt_idx[permut_idx][: int(res.shape[0] * (1 - prop_train))]

        p_test = p_all[idx_test]
        p_vals = jnp.delete(p_all, idx_test, 0)
        y_test_old = y_all[:, idx_test]
        y_shift_old = jnp.delete(y_all, idx_test, 1)
    else:
        p_test = p_all[-test_end:]
        p_vals = p_all[: len(p_all) - test_end]
        y_test_old = y_all[:, -test_end:]
        y_shift_old = y_all[:, : y_all.shape[-1] - test_end]

    y_shift = (y_shift_old - jnp.mean(y_shift_old)) / jnp.std(y_shift_old)
    y_test = (y_test_old - jnp.mean(y_shift_old)) / jnp.std(y_shift_old)

    y_shift = y_shift.T
    y_test = y_test.T

    if conv == True:
        X_vec = ts[0]
        idx = np.argmax(jnp.diff(X_vec[:, 0]) < 0)
        y_shift = jnp.expand_dims(
            jax.vmap(lambda y: jnp.reshape(y, (idx + 1, idx + 1)), in_axes=[-1])(
                y_shift
            ),
            1,
        )[:, :, 1:-1, 1:-1]
        y_test = jnp.expand_dims(
            jax.vmap(lambda y: jnp.reshape(y, (idx + 1, idx + 1)), in_axes=[-1])(
                y_test
            ),
            1,
        )[:, :, 1:-1, 1:-1]

    if output is None:
        output_shift = y_shift
        output_test = y_test
    else:
        output_test = output[idx_test]
        output_shift = jnp.delete(output, idx_test, 0)

    def inv_func(xx):
        return xx * jnp.std(y_shift_old) + jnp.mean(y_shift_old)

    if not pod:
        return (
            ts,
            y_shift,
            y_test,
            p_vals,
            p_test,
            inv_func,
            y_shift_old,
            y_test_old,
            output_shift,
            output_test,
        )

    u_now = determine_u_now(y_shift.T, eps=eps)
    coeffs_shift = u_now.T @ y_shift.T
    mean_vals = jnp.mean(coeffs_shift, axis=1)
    std_vals = jnp.std(coeffs_shift, axis=1)
    coeffs_shift = jax.vmap(lambda x, m, s: (x - m) / s)(
        coeffs_shift, mean_vals, std_vals
    )
    coeffs_test = u_now.T @ y_test.T
    coeffs_test = jax.vmap(lambda x, m, s: (x - m) / s)(
        coeffs_test, mean_vals, std_vals
    )

    def inv_func(xx):
        return u_now @ jax.vmap(lambda x, m, s: x * s + m)(xx, mean_vals, std_vals)

    return (
        ts,
        coeffs_shift.T,
        coeffs_test.T,
        p_vals,
        p_test,
        inv_func,
        y_shift,
        y_test,
        output_shift,
        output_test,
    )


def get_data(problem, **kwargs):
    match problem:
        case "accelerate":
            ts = jnp.linspace(0, 2 * jnp.pi, 200)

            def func(f, x):
                return jnp.sin(f * jnp.pi * x)

            p_vals = jnp.linspace(1 / 3, 1, 150)[:-1]
            y_shift = jax.vmap(func, in_axes=[0, None])(p_vals, ts).T
            p_test = jrandom.uniform(
                jrandom.PRNGKey(0), (200,), minval=1 / 3 + 0.01, maxval=1 - 0.01
            )
            y_test = jax.vmap(func, in_axes=[0, None])(p_test, ts).T
            return (
                ts,
                y_shift,
                y_test,
                jnp.expand_dims(p_vals, axis=-1),
                jnp.expand_dims(p_test, axis=-1),
                None,
                None,
                None,
                None,
                None,
            )
        case "shift":
            ts = jnp.linspace(0, 2 * jnp.pi, 200)

            def sf_func(s, x):
                return jnp.sin(x - s * jnp.pi)

            p_vals = jnp.array([-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])
            p_vals = jnp.linspace(0, 1.5, 15)[:-1]
            y_shift = jax.vmap(sf_func, in_axes=[0, None])(p_vals, ts).T
            p_test = jnp.array([-0.15, 0.28, 0.41])
            p_test = jrandom.uniform(
                jrandom.PRNGKey(0), (20,), minval=0 + 0.01, maxval=p_vals[-1] * 0.99
            )
            y_test = jax.vmap(sf_func, in_axes=[0, None])(p_test, ts).T
            return (
                ts,
                y_shift,
                y_test,
                jnp.expand_dims(p_vals, axis=-1),
                jnp.expand_dims(p_test, axis=-1),
                None,
                None,
                None,
                y_shift,
                y_test,
            )
        case "stairs":
            Tend = 3.5  # [s]
            NT = 500
            nt = NT + 1
            times = jnp.linspace(0, Tend, nt)
            freq = 1  # [Hz] # 3
            wrad = 2 * jnp.pi * freq
            nAmp = 100  # 60
            yo = 2.3
            Amp = jnp.arange(1, 5, 0.1)
            phases = jnp.linspace(1 / 4 * Tend, 3 / 4 * Tend, nAmp)
            p_vals = Amp

            def find_ph(amp):
                return phases[0] + (amp - Amp[0]) / (Amp[1] - Amp[0]) * (
                    phases[1] - phases[0]
                )

            def create_escal(amp):
                return jnp.cumsum(
                    (
                        (
                            jnp.abs(
                                (
                                    amp
                                    * jnp.sqrt(times)
                                    * jnp.sin(wrad * (times - find_ph(amp)))
                                )
                                - yo
                            )
                            + (
                                (
                                    amp
                                    * jnp.sqrt(times)
                                    * jnp.sin(wrad * (times - find_ph(amp)))
                                )
                                - yo
                            )
                        )
                        / 2
                    )
                    ** 5
                )

            y_shift_old = jax.vmap(create_escal)(p_vals).T
            y_shift = jax.vmap(
                lambda y: (y - jnp.mean(y_shift_old)) / jnp.std(y_shift_old),
                in_axes=[-1],
            )(
                y_shift_old
            ).T  # lambda y: (y-jnp.mean(y))/jnp.std(y)
            y_shift = y_shift[:, ~jnp.isnan(y_shift).any(axis=0)]

            p_test = jrandom.uniform(
                jrandom.PRNGKey(0),
                (100,),
                minval=jnp.min(p_vals) + 0.01,
                maxval=jnp.max(p_vals) - 0.01,
            )
            y_test = jax.vmap(
                lambda y: (y - jnp.mean(y_shift_old)) / jnp.std(y_shift_old)
            )(jax.vmap(create_escal)(p_test)).T

            ts = jnp.arange(0, y_shift.shape[0], 1)
            # y_pod_shift, y_pod_test, inv_func = do_all_the_u_now_stuff(y_shift)
            # inv_func, y_shift, y_test
            return (
                ts,
                y_shift,
                y_test,
                jnp.expand_dims(p_vals, axis=-1),
                jnp.expand_dims(p_test, axis=-1),
                None,
                None,
                None,
                None,
                None,
            )

        case "mult_freqs":
            p_vals_0 = jnp.repeat(jnp.linspace(0.8 * jnp.pi, jnp.pi, 25), 25)
            p_vals_1 = jnp.tile(jnp.linspace(0.3 * jnp.pi, 0.5 * jnp.pi, 25), 25)
            p_vals = jnp.stack([p_vals_0, p_vals_1], axis=-1)
            ts = jnp.arange(0, 5 * jnp.pi, 0.01)
            y_shift = jax.vmap(lambda p: jnp.sin(p[0] * ts) + jnp.sin(p[1] * ts))(
                p_vals
            ).T

            p_vals_0 = jrandom.uniform(
                jrandom.PRNGKey(140),
                (100,),
                minval=0.8 * jnp.pi + 0.01,
                maxval=jnp.pi - 0.01,
            )
            p_vals_1 = jrandom.uniform(
                jrandom.PRNGKey(8),
                (100,),
                minval=0.3 * jnp.pi + 0.01,
                maxval=jnp.pi / 2 - 0.01,
            )
            p_test = jnp.stack([p_vals_0, p_vals_1], axis=-1)
            y_test = jax.vmap(lambda p: jnp.sin(p[0] * ts) + jnp.sin(p[1] * ts))(
                p_test
            ).T
            return ts, y_shift, y_test, p_vals, p_test, None, None, None

        case "angelo":
            import os
            import pandas as pd

            filename = os.path.join(os.getcwd(), "data_angelo/")

            def f(n):
                return os.path.join(filename, n)

            scaledDesign = pd.read_csv(f("data.csv"))
            ts = jnp.array(scaledDesign["freqEng"][0].strip("[]").split(), "float")

            def read_series(y):
                return np.stack(
                    y.apply(
                        lambda x: np.array(x.strip("[]").split(), dtype=np.float32)
                    ).to_numpy()
                )

            p_all = jnp.asarray(scaledDesign[scaledDesign.columns[:14]])
            y_all = read_series(scaledDesign["S22dB"]).T
            return norm_divide_return(ts, y_all, p_all, prop_train=0.9)

        case "angelo_new":
            import os
            import pandas as pd

            filename = os.path.join(os.getcwd(), "data_angelo_new/")

            def f(n):
                return os.path.join(filename, n)

            scaledDesign = pd.read_csv(f("data_400.csv"))
            ts = jnp.array(scaledDesign["freqEng"][0].strip("[]").split(), "float")

            def read_series(y):
                return np.stack(
                    y.apply(
                        lambda x: np.array(x.strip("[]").split(), dtype=np.float32)
                    ).to_numpy()
                )

            p_all_0 = jnp.asarray(scaledDesign[scaledDesign.columns[:14]])
            y_all_0 = read_series(scaledDesign["S22dB"]).T  # S21dB, S22dB, Nf

            scaledDesign = pd.read_csv(f("data_800.csv"))
            ts = jnp.array(scaledDesign["freqEng"][0].strip("[]").split(), "float")

            def read_series(y):
                return np.stack(
                    y.apply(
                        lambda x: np.array(x.strip("[]").split(), dtype=np.float32)
                    ).to_numpy()
                )

            p_all_1 = jnp.asarray(scaledDesign[scaledDesign.columns[:14]])
            y_all_1 = read_series(scaledDesign["S22dB"]).T

            y_all = jnp.concatenate([y_all_0, y_all_1], axis=-1)
            p_all = jnp.concatenate([p_all_0, p_all_1], axis=0)

            def func(p):
                return jnp.concatenate((np.argsort(p)[:10], np.argsort(p)[-10:]))

            idxs = jnp.concatenate(jax.vmap(func, in_axes=[-1])(p_all))

            p_all = jnp.delete(p_all, idxs, 0)
            y_all = jnp.delete(y_all, idxs, 1)
            return norm_divide_return(ts, y_all, p_all, prop_train=0.95)

        case "mult_gausses":

            p_vals_0 = jnp.repeat(jnp.linspace(1, 3, 25), 25)
            p_vals_1 = jnp.tile(jnp.linspace(4, 6, 25), 25)
            p_vals = jnp.stack([p_vals_0, p_vals_1], axis=-1)
            p_test_0 = jrandom.uniform(
                jrandom.PRNGKey(100), (100,), minval=1.1, maxval=2.9
            )
            p_test_1 = jrandom.uniform(
                jrandom.PRNGKey(0), (100,), minval=4.1, maxval=5.9
            )
            p_test = jnp.stack([p_test_0, p_test_1], axis=-1)

            ts = jnp.arange(0, 6, 0.005)

            def gauss(a, b, c, t):
                return a * jnp.exp(-((t - b) ** 2) / (2 * c**2))

            a = 1.3
            c = 0.2
            y_shift = jax.vmap(
                lambda p, t: gauss(a, p[0], c, t) + gauss(-a, p[1], c, t),
                in_axes=[0, None],
            )(p_vals, ts).T
            y_test = jax.vmap(
                lambda p, t: gauss(a, p[0], c, t) + gauss(-a, p[1], c, t),
                in_axes=[0, None],
            )(p_test, ts).T
            # u_now = determine_u_now(y_shift, eps=0.0001)
            return ts, y_shift, y_test, p_vals, p_test, None, None, None, None, None

        case "avrami_noise":
            n = 4
            N = jnp.repeat(jnp.linspace(1, 3, 20), 20)
            G = jnp.tile(jnp.linspace(1, 3, 20), 20)
            p_vals = jnp.stack([N, G], axis=-1)
            p_test_1 = jrandom.uniform(
                jrandom.PRNGKey(100), (150,), minval=N[0] + 0.01, maxval=N[-1] - 0.01
            )
            p_test_2 = jrandom.uniform(
                jrandom.PRNGKey(0), (150,), minval=G[0] + 0.01, maxval=G[-1] - 0.01
            )
            p_test = jnp.stack([p_test_1, p_test_2], axis=-1)

            ts = jnp.arange(0, 1.5, 0.005)
            y_shift = jax.vmap(
                lambda p, t: 1 - jnp.exp(-jnp.pi * p[0] * p[1] ** 3 / 3 * t**n),
                in_axes=[0, None],
            )(p_vals, ts).T
            y_test = jax.vmap(
                lambda p, t: 1 - jnp.exp(-jnp.pi * p[0] * p[1] ** 3 / 3 * t**n),
                in_axes=[0, None],
            )(p_test, ts).T

            noise_keys_train = jrandom.split(jrandom.PRNGKey(0), y_shift.shape[-1])
            noise_keys_test = jrandom.split(jrandom.PRNGKey(50), y_test.shape[-1])
            y_shift = jax.vmap(
                lambda y, k: y + jrandom.normal(k, y.shape) * 0.01, in_axes=[-1, 0]
            )(y_shift, noise_keys_train).T
            y_test = jax.vmap(
                lambda y, k: y + jrandom.normal(k, y.shape) * 0.01, in_axes=[-1, 0]
            )(y_test, noise_keys_test).T
            u_now = determine_u_now(y_shift, 1.3)
            return (
                ts,
                u_now.T @ y_shift,
                u_now.T @ y_test,
                p_vals,
                p_test,
                lambda x: u_now @ x,
                y_shift,
                y_test,
                None,
                None,
            )

        case "avrami":
            n = 4
            N = jnp.repeat(jnp.linspace(1.5, 3, 20), 20)
            G = jnp.tile(jnp.linspace(1.5, 3, 20), 20)
            p_vals = jnp.stack([N, G], axis=-1)
            p_test_1 = jrandom.uniform(
                jrandom.PRNGKey(100), (150,), minval=N[0] + 0.01, maxval=N[-1] - 0.01
            )
            p_test_2 = jrandom.uniform(
                jrandom.PRNGKey(0), (150,), minval=G[0] + 0.01, maxval=G[-1] - 0.01
            )
            p_test = jnp.stack([p_test_1, p_test_2], axis=-1)

            ts = jnp.arange(0, 1.5, 0.01)
            y_shift = jax.vmap(
                lambda p, t: 1 - jnp.exp(-jnp.pi * p[0] * p[1] ** 3 / 3 * t**n),
                in_axes=[0, None],
            )(p_vals, ts).T
            mean, std = jnp.mean(y_shift), jnp.std(y_shift)
            y_shift = (y_shift - mean) / std
            y_test = jax.vmap(
                lambda p, t: 1 - jnp.exp(-jnp.pi * p[0] * p[1] ** 3 / 3 * t**n),
                in_axes=[0, None],
            )(p_test, ts).T
            y_test = (y_test - mean) / std
            return ts, y_shift, y_test, p_vals, p_test, None, None, None, None, None

        case "welding":
            import os
            import h5py

            filename = os.path.join(os.getcwd(), "data-chady/")

            def f(n):
                return os.path.join(filename, n)

            Data_1 = h5py.File(f("dataset_1_grid_1000_point.mat"), "r")
            y_all_train = jnp.array(Data_1["Solution"]).T
            location = jnp.array(Data_1["location"]).T
            radius = jnp.array(Data_1["radius"]).T
            X = jnp.array(Data_1["X"]).T
            Y = jnp.array(Data_1["Y"]).T
            location = jnp.array(Data_1["location"]).T
            radius = jnp.array(Data_1["radius"]).T
            Data_1.close()
            p_all_train = jnp.concatenate([location, radius], -1)

            Data_2 = h5py.File(f("dataset_1.mat"), "r")
            y_all_test = jnp.array(Data_2["Solution"]).T
            location = jnp.array(Data_2["location"]).T
            radius = jnp.array(Data_2["radius"]).T
            X = jnp.array(Data_2["X"]).T
            Y = jnp.array(Data_2["Y"]).T
            location = jnp.array(Data_2["location"]).T
            radius = jnp.array(Data_2["radius"]).T
            Data_2.close()
            p_all_test = jnp.concatenate([location, radius], -1)
            to_remove = np.bitwise_or.reduce(
                (p_all_test >= jnp.max(p_all_train, 0))
                | (p_all_test <= jnp.min(p_all_train, 0)),
                1,
            )
            p_all_test = jnp.delete(p_all_test, to_remove, 0)
            y_all_test = jnp.delete(y_all_test, to_remove, 1)
            y_all_test = jnp.delete(
                y_all_test,
                np.bitwise_or.reduce(
                    (p_all_test >= jnp.max(p_all_train, 0))
                    | (p_all_test <= jnp.min(p_all_train, 0)),
                    1,
                ),
                1,
            )

            p_all = jnp.concatenate([p_all_train, p_all_test], 0)
            y_all = jnp.concatenate([y_all_train, y_all_test], -1)
            return norm_divide_return(
                (X, Y), y_all, p_all, prop_train=0.8, test_end=p_all_test.shape[0]
            )

        case "multiple_steps":

            def create_steps(p, w, t):
                return np.bitwise_or.reduce(
                    jnp.stack(
                        my_vmap(
                            lambda pp, w, t: ~((t < (pp - w / 2)) | (t > (pp + w / 2)))
                        )(p, args=(w, t))
                    ),
                    0,
                )

            t = jnp.linspace(0, 20, 500)

            w1 = 0.5
            p_1 = jrandom.uniform(
                jrandom.PRNGKey(0), (100,), minval=t[0] + w1 / 2, maxval=t[-1] - w1 / 2
            )
            steps_1 = jnp.stack(
                my_vmap(create_steps)(jnp.expand_dims(p_1, 0).T, args=(w1, t))
            ).T

            w2 = 0.5
            p_2_0 = jrandom.uniform(
                jrandom.PRNGKey(30),
                (100,),
                minval=t[0] + 3 * w2 / 2,
                maxval=t[-1] - 3 * w2 / 2,
            )
            p_2_1 = jax.vmap(lambda p: p + 2 * w2)(p_2_0)
            p_2 = jnp.stack([p_2_0, p_2_1], 0)
            steps_2 = jnp.stack(my_vmap(create_steps)(p_2.T, args=(w2, t))).T

            w3 = 0.5
            p_3_0 = jrandom.uniform(
                jrandom.PRNGKey(20),
                (100,),
                minval=t[0] + 5 * w3 / 2,
                maxval=t[-1] - 5 * w3 / 2,
            )
            p_3_1 = jax.vmap(lambda p: p + 2 * w3)(p_3_0)
            p_3_2 = jax.vmap(lambda p: p + 2 * w3)(p_3_1)
            p_3 = jnp.stack([p_3_0, p_3_1, p_3_2], 0)
            steps_3 = jnp.stack(my_vmap(create_steps)(p_3.T, args=(w3, t))).T

            w4 = 0.4
            p_4_0 = jrandom.uniform(
                jrandom.PRNGKey(500),
                (100,),
                minval=t[0] + 7 * w4 / 2,
                maxval=t[-1] - 7 * w4 / 2,
            )
            p_4_1 = jax.vmap(lambda p: p + 2 * w4)(p_4_0)
            p_4_2 = jax.vmap(lambda p: p + 2 * w4)(p_4_1)
            p_4_3 = jax.vmap(lambda p: p + 2 * w4)(p_4_2)
            p_4 = jnp.stack([p_4_0, p_4_1, p_4_2, p_4_3], 0)
            steps_4 = jnp.stack(my_vmap(create_steps)(p_4.T, args=(w3, t))).T

            y_all = jnp.concatenate([steps_1, steps_2, steps_3, steps_4], -1)
            output_all = jnp.zeros((y_all.shape[-1], 4))
            output_all = output_all.at[: steps_1.shape[-1], 0].set(1)
            output_all = output_all.at[
                steps_1.shape[-1] : steps_1.shape[-1] + steps_2.shape[-1], 1
            ].set(1)
            output_all = output_all.at[
                steps_1.shape[-1]
                + steps_2.shape[-1] : steps_1.shape[-1]
                + steps_2.shape[-1]
                + steps_3.shape[-1],
                2,
            ].set(1)
            output_all = output_all.at[
                steps_1.shape[-1] + steps_2.shape[-1] + steps_3.shape[-1] :, 3
            ].set(1)
            output_all = output_all.T
            p_all = jnp.concatenate([p_1, jnp.sum(p_2, 0) / 2, p_3[1]], -1)

            permutation = jrandom.permutation(jrandom.PRNGKey(0), y_all.shape[1])
            y_all = y_all[:, permutation]
            output_all = output_all[:, permutation]
            return norm_divide_return(
                t, y_all, jnp.expand_dims(p_all, -1), prop_train=0.8, output=output_all
            )

        case _:
            raise ValueError(f"Problem {problem} not recognized")


def p_of_dim_n_equi_grid(v_train, vt_train, p_vals, p_test, model, **kwargs):
    dims = p_vals.shape[-1]

    def to_map_over_test(p0):
        all_idxs = np.linspace(0, p_vals.shape[0] - 1, p_vals.shape[0], dtype=int)
        neg_mat = p_vals < p0
        pos_mat = p_vals > p0
        mats = [np.stack([neg_mat[:, i], pos_mat[:, i]], axis=-1) for i in range(dims)]

        def max_min(arr, bools):
            return np.stack(
                [
                    (1 - b) * (v == np.min(v)) + b * (v == np.max(v))
                    for v, b in zip(arr.T, bools)
                ],
                axis=-1,
            ).astype(bool)

        def switch_1_0(a):
            return np.where((a == 0) | (a == 1), a ^ 1, a)

        def nested_loops(d, all_is, mats):
            interp_idxs = []
            if d == 0:
                bool_mat = np.stack([m[:, i] for m, i in zip(mats, all_is)], axis=-1)
                p_now = p_vals[np.bitwise_and.reduce(bool_mat, 1)]
                idxs = all_idxs[np.bitwise_and.reduce(bool_mat, 1)]
                id = np.array([int(i) for i in all_is])
                return idxs[np.bitwise_and.reduce(max_min(p_now, switch_1_0(id)), 1)]

            for i in range(2):
                interp_idxs.append(nested_loops(d - 1, all_is + [i], mats))
            return np.array(interp_idxs).flatten()

        return nested_loops(dims, [], mats)

    interp_ps = my_vmap_multiple_arrays(to_map_over_test)(p_test)

    def interpolate(coords, coord0, ps):
        ds = np.abs(coords - coord0)
        vols = np.prod(ds, axis=-1)
        vols = vols / np.sum(vols)
        sorted = np.argsort(vols)

        def func_per_mode(p):
            ps_sorted = p[sorted]
            return np.sum(np.flip(np.sort(vols)) * ps_sorted)

        return jax.vmap(func_per_mode, in_axes=[-1])(ps)

    vt_test = my_vmap_multiple_arrays(interpolate)(
        p_vals[interp_ps], p_test, vt_train[:, interp_ps.T].T
    ).T

    x_test = jnp.sum(
        jax.vmap(lambda o1, o2: jnp.outer(o1, o2), in_axes=[-1, 0])(v_train, vt_test),
        axis=0,
    ).T
    y_test = model.func_decode(x_test, train=True).T
    y_test_o = model.func_decode(x_test, train=False).T
    return x_test, y_test, y_test_o, vt_test


def p_of_dim_1(v_train, vt_train, p_vals, p_test, model, ids=None, **kwargs):
    def evaluate_at_parameter(idx, p_val, v):
        v_1 = v[idx]
        p_1 = p_vals[idx]
        v_2 = v[idx + 1]
        p_2 = p_vals[idx + 1]
        return v_1 + (p_val - p_1) * (v_2 - v_1) / (p_2 - p_1)

    if ids is None:
        ids = jnp.array(
            my_vmap(lambda p: jnp.where((p_vals[:-1] <= p) & (p_vals[1:] >= p))[0])(
                p_test
            )
        )
    vt_test = jax.vmap(
        jax.vmap(evaluate_at_parameter, in_axes=[0, 0, None]), in_axes=[None, None, 0]
    )(ids, p_test[:, 0], vt_train)
    x_test = jnp.sum(
        jax.vmap(lambda o1, o2: jnp.outer(o1, o2), in_axes=[-1, 0])(v_train, vt_test),
        axis=0,
    )
    y_test = model.func_decode(x_test, train=True)
    y_test_o = model.func_decode(x_test, train=False)
    return x_test, y_test, y_test_o, vt_test


def interpolate_pimm_data(R, T, ts):  # for T decroissant
    def to_map_over_ts(t):
        if t < np.min(T):
            return R[-1]
        if t > np.max(T):
            return R[0]
        idx = np.where((T[1:] <= t) & (T[:-1] >= t))[0][0]
        t1, t2 = T[idx], T[idx + 1]
        r1, r2 = R[idx], R[idx + 1]
        interpolated_r = r1 + (r2 - r1) * (t - t1) / (t2 - t1)
        return interpolated_r

    return my_vmap(to_map_over_ts)(ts)


def interpolate_quad(p00, p11, p22, p33, v00, v11, v22, v33, pt):
    p0 = p22
    p1 = p33
    p2 = p00
    p3 = p11
    v0 = v22
    v1 = v33
    v2 = v00
    v3 = v11

    x0, y0 = p0[0], p0[1]
    x1, y1 = p1[0], p1[1]
    x2, y2 = p2[0], p2[1]
    x3, y3 = p3[0], p3[1]
    xs, ys = pt[0], pt[1]
    d0 = jnp.sqrt((x0 - xs) ** 2 + (y0 - ys) ** 2)
    d1 = jnp.sqrt((x1 - xs) ** 2 + (y1 - ys) ** 2)
    d2 = jnp.sqrt((x2 - xs) ** 2 + (y2 - ys) ** 2)
    d3 = jnp.sqrt((x3 - xs) ** 2 + (y3 - ys) ** 2)
    sorted_ds = jnp.sort(jnp.array([d0, d1, d2, d3]))
    sorted_idx = jnp.argsort(jnp.array([d0, d1, d2, d3]))

    return jnp.sum(
        jnp.multiply(
            sorted_ds / jnp.sum(sorted_ds), jnp.array([v0, v1, v2, v3])[sorted_idx]
        )
    )
    # aa = x0
    # ap = y0
    # bb = -x0 + x1
    # bp = -y0 + y1
    # cc = -x0 + x3
    # cp = -y0 + y3
    # dd = x0 - x1 + x2 - x3
    # dp = y0 - y1 + y2 - y3

    # A = -cc*dp
    # B = ap*dd+bp*cc+cp+dp*xs-dp*aa-ys*dd
    # C = ap*bb+bp*xs-bp*aa-ys*bb
    # coeff = jnp.array([A, B, C])
    # nu = jnp.roots(coeff)
    # xi = (xs-aa-cc*nu)/(bb+dd*nu)
    # pdb.set_trace()
    # idx = jnp.where((xi >= 0) & (xi <= 1) & (nu >= 0) & (nu <= 1))[0][0]
    # nu = jnp.real(nu[idx])
    # xi = jnp.real(xi[idx])

    # psi1 = lambda xi, nu: (1-xi)*(1-nu)
    # psi2 = lambda xi, nu: xi*(1-nu)
    # psi3 = lambda xi, nu: xi*nu
    # psi4 = lambda xi, nu: (1-xi)*nu
    # return psi1(xi, nu)*v0 + psi2(xi, nu)*v1 + psi3(xi, nu)*v2 + psi4(xi, nu)*v3


def get_v_test_multi_p(p_train, vt_train, p_test, graph=False):

    p_train = np.array(p_train)
    p_test = np.array(p_test)
    vt_train = np.array(vt_train)

    def to_map_over_p_test(pt):
        indices = np.argsort(jnp.sum(jnp.square(p_train - pt), axis=1))
        p_now = p_train[indices]
        v_now = vt_train[indices]
        rect_bool = np.stack([p_now[:, 0] < pt[0], p_now[:, 1] < pt[1]], axis=-1)

        p0 = p_now[np.where((rect_bool[:, 0] == 0) & (rect_bool[:, 1] == 0))[0][0]]
        v0 = v_now[np.where((rect_bool[:, 0] == 0) & (rect_bool[:, 1] == 0))[0][0]]
        p1 = p_now[np.where((rect_bool[:, 0] == 1) & (rect_bool[:, 1] == 0))[0][0]]
        v1 = v_now[np.where((rect_bool[:, 0] == 1) & (rect_bool[:, 1] == 0))[0][0]]
        p2 = p_now[np.where((rect_bool[:, 0] == 1) & (rect_bool[:, 1] == 1))[0][0]]
        v2 = v_now[np.where((rect_bool[:, 0] == 1) & (rect_bool[:, 1] == 1))[0][0]]
        p3 = p_now[np.where((rect_bool[:, 0] == 0) & (rect_bool[:, 1] == 1))[0][0]]
        v3 = v_now[np.where((rect_bool[:, 0] == 0) & (rect_bool[:, 1] == 1))[0][0]]
        if graph:
            return p0, p1, p2, p3
        return interpolate_quad(p0, p1, p2, p3, v0, v1, v2, v3, pt)

    return my_vmap(to_map_over_p_test)(p_test)


def p_of_dim_2(v_train, vt_train, p_vals, p_test, model, **kwargs):
    to_conc = []
    for j in range(vt_train.shape[0]):
        vt_trai = vt_train[j]
        vt_test_ = get_v_test_multi_p(p_vals, vt_trai, p_test, graph=False)
        to_conc.append(vt_test_)
        x_test_ = jnp.outer(v_train[:, j], vt_test_)
        x_test = x_test_ if j == 0 else x_test + x_test_
    vt_test = jnp.stack(to_conc)
    y_test_ = model.func_decode(x_test, train=True)
    y_test_o = model.func_decode(x_test, train=False)
    return x_test, y_test_, y_test_o, vt_test


def normalize(x_tr, x_t, axis=0):
    means = jnp.mean(x_tr, axis=axis)
    std = jnp.std(x_tr, axis=axis)
    return (x_tr - means) / std, (x_t - means) / std


def main_RRAE(
    method,
    prob_name,
    data_func,
    train_func,
    post_process_func,
    process_func=None,
    post_process_bool=False,
    train_nn=True,
    pp=True,
    **kwargs,
):

    if method == "weak":
        num_modes = -1
        num_modes_vvt = kwargs["num_modes"]
        v_vt = True
        Wx_ = False
        num_modes_true = num_modes_vvt
    elif method == "strong":
        num_modes = kwargs["num_modes"]
        num_modes_vvt = None
        v_vt = False
        Wx_ = False
        num_modes_true = num_modes

    (
        ts,
        y_shift,
        y_test,
        p_vals,
        p_test,
        lambda_post,
        y_original,
        y_test_original,
        output_shift,
        output_test,
    ) = data_func(**kwargs)

    lambda_post = _identity if lambda_post is None else lambda_post
    assert is_test_inside(p_vals, p_test)

    print(f"Shape of y_train is {y_shift.shape}, (N x T)")
    print(f"Shape of p_vals is {p_vals.shape}, (N x P)")
    print(f"Shape of output_train is {output_shift.shape}, (N x O)")
    (
        y_pred_train,
        x_m,
        model,
        v_train,
        vt_train,
        parameters,
        y_pred_train_o,
        y_shift,
        p_vals,
    ) = train_func(
        y_shift,
        output_shift,
        n_mode=num_modes,
        num_modes_vvt=num_modes_vvt,
        dropout=0,
        v_vt=v_vt,
        WX_=Wx_,
        p_vals=p_vals,
        post_proc_func=lambda_post,
        method=method,
        **kwargs,
    )
    # if (not train_nn) and  (p_vals.shape[-1] != 1) and (p_vals.shape[-1] != 2):
    #     print("Only P = 1 or P = 2 are supported without training a Neural Network , switching to training mode")
    #     train_nn = True

    folder = f"{prob_name}/{prob_name}_{method}"
    folder_name = f"{folder}/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename = os.path.join(folder_name, f"{method}_{prob_name}")

    if not train_nn:

        x_test, y_pred_test, y_pred_test_o, vt_test = process_func(
            n_mode=num_modes,
            v_train=v_train,
            vt_train=vt_train,
            p_vals=p_vals,
            p_test=p_test,
            model=model,
            y_train=y_shift,
            y_test=y_test,
            out_train=output_shift,
            out_test=output_test,
        )

        kwargs = {
            k: kwargs[k]
            for k in set(list(kwargs.keys()))
            - set({"activation_enc", "activation_dec", "loss_func", "post_proc_func"})
        }
        if post_process_bool:
            pdb.set_trace()
            if not os.path.exists(folder):
                os.makedirs(folder)
            shutil.rmtree(folder)
            os.makedirs(folder)
            error_train, error_test, error_train_o, error_test_o = post_process_func(
                p_vals,
                p_test,
                problem,
                method,
                x_m,
                y_pred_train,
                v_train,
                vt_train,
                vt_test,
                y_pred_test,
                y_shift,
                y_test,
                y_original,
                y_pred_train_o,
                y_test_original,
                y_pred_test_o,
                out_test=output_test,
                out_train=output_shift,
                file=folder_name,
                pp=pp,
            )

            with open(f"{filename}_.pkl", "wb") as f:
                dill.dump(
                    [
                        v_train,
                        vt_train,
                        vt_test,
                        x_m,
                        y_pred_train,
                        x_test,
                        y_pred_test,
                        y_shift,
                        y_test,
                        y_original,
                        y_pred_train_o,
                        y_test_original,
                        y_pred_test_o,
                        ts,
                        error_train,
                        error_test,
                        error_train_o,
                        error_test_o,
                        p_vals,
                        p_test,
                        kwargs,
                        [],
                    ],
                    f,
                )
        else:
            error_train = (
                jnp.linalg.norm(y_pred_train - y_shift) / jnp.linalg.norm(y_shift) * 100
            )
            with open(f"{filename}_.pkl", "wb") as f:
                dill.dump([v_train, vt_train, error_train], f)

    else:
        x_test, y_pred_test, y_pred_test_o, vt_test = None, None, None, None
        if post_process_bool:
            error_train, error_test, error_train_o, error_test_o = post_process_func(
                p_vals,
                p_test,
                problem,
                method,
                x_m,
                y_pred_train,
                v_train,
                vt_train,
                vt_test,
                y_pred_test,
                y_shift,
                y_test,
                y_original,
                y_pred_train_o,
                y_test_original,
                y_pred_test_o,
                out_test=output_test,
                out_shift=output_shift,
                test=False,
                file=folder_name,
            )

            with open(f"{filename}_.pkl", "wb") as f:
                dill.dump(
                    [
                        v_train,
                        vt_train,
                        vt_test,
                        x_m,
                        y_pred_train,
                        x_test,
                        y_pred_test,
                        y_shift,
                        y_test,
                        y_original,
                        y_pred_train_o,
                        y_test_original,
                        y_pred_test_o,
                        ts,
                        error_train,
                        error_test,
                        error_train_o,
                        error_test_o,
                        p_vals,
                        p_test,
                        kwargs,
                        [],
                    ],
                    f,
                )
        else:
            error_train = (
                jnp.linalg.norm(y_pred_train - y_shift) / jnp.linalg.norm(y_shift) * 100
            )
            with open(f"{filename}_.pkl", "wb") as f:
                dill.dump([v_train, vt_train, error_train], f)

    def save(filename, hyperparams, model):
        with open(filename, "wb") as f:
            dill.dump(hyperparams, f)
            eqx.tree_serialise_leaves(f, model)

    save(f"{filename}_nn.pkl", parameters, model)

    print(f"Folder name is {folder_name}")
    print(f"Base filename is {filename}")
    if train_nn:
        print("Use them in train_alpha_nn.py to train a NN to get the coeffs.")
    return (
        p_vals,
        p_test,
        model,
        x_m,
        y_pred_train,
        v_train,
        vt_train,
        vt_test,
        y_pred_test,
        y_shift,
        y_test,
        num_modes,
        y_original,
        y_pred_train_o,
        y_test_original,
        y_pred_test_o,
        folder_name,
        error_train,
    )


if __name__ == "__main__":
    method = "strong"
    problem = "shift"  # "shift", "accelerate", "stairs", "mult_freqs", "pimm_curves", "angelo", "mult_gausses", "avrami", "avrami_noise", "mass_spring"
    train_nn = False  # 12
    num_modes = 1

    if num_modes == 1:
        process_interp = p_of_dim_1
    elif num_modes == 2:
        process_interp = p_of_dim_2
    else:
        process_interp = p_of_dim_n_equi_grid

    kwargs = {
        "interp": True,
        "width_i": 64,
        "depth_i": 4,
        "num_modes": num_modes,
        "problem": problem,
        "step_st": [
            1000, 1000
        ],
        "lr_st": [1e-3, 1e-4, 1e-5],
        "width_enc": 64,
        "depth_enc": 1,
        "width_dec": 64,
        "depth_dec": 6,
        "mul_latent": 1.2,
        "batch_size_st": [-1, -1, -1, 32, 32, 32, 32, 32, 32, 32, 32],
        "mul_lr": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        "training_intr": [
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
            True,
            True,
            False,
            False,
            False,
        ],
        # "loss_func": lambda x, y: (-1*jnp.sum(y*jnp.log(x)))/x.shape[-1],
        # "activation_dec": lambda x: jax.nn.softmax(x.T).T,
        "kernel_conv": 3,
        "stride": 1,
        "padding": 2,
        "kernel_pool": 2,
        "out_conv": 3,
    }
    res = main_RRAE(
        method,
        problem,
        get_data,
        train_loop_RRAE,
        process_func=process_interp,
        post_process_bool=True,
        train_nn=train_nn,
        post_process_func=post_process_interp,
        pp=False,
        **kwargs,
    )
    (
        p_vals,
        p_test,
        model,
        x_m,
        y_pred_train,
        v_train,
        vt_train,
        vt_test,
        y_pred_test,
        y_shift,
        y_test,
        num_modes,
        y_original,
        y_pred_train_o,
        y_test_original,
        y_pred_test_o,
        folder_name,
        error_train,
    ) = res
    pdb.set_trace()
