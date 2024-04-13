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
from train_RRAE import get_data, interpolate_quad
import shutil
config.update("jax_enable_x64", True)

from collections.abc import Callable
from typing import (
    Literal,
    Optional,
    Union,
)

import jax
import jax.nn as jnn
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray

from equinox._doc_utils import doc_repr
from equinox._filters import is_array
from equinox._module import field, Module
from equinox._vmap_pmap import filter_vmap
from equinox.nn._linear import Linear
from equinox.nn import LayerNorm

_identity = doc_repr(lambda x: x, "lambda x: x")
_relu = doc_repr(jnn.relu, "<function relu>")
class WX(Module):
    w : jnp.array
    def __init__(self, dim0, dim1, *, key=jrandom.PRNGKey(1200), **kwargs):
        super().__init__(**kwargs)
        self.w = jrandom.uniform(key, (dim1, dim0), minval=-1, maxval=1)

    def __call__(self, x, *args, **kwargs):
        return jnp.dot(self.w, x)
    
class MLP_dropout(Module, strict=True):
    layers: tuple[Linear, ...]
    activation: Callable
    final_activation: Callable
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
        final_activation=_identity,
        use_bias=True,
        use_final_bias=True,
        *,
        key
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
        self.activation = filter_vmap(
            filter_vmap(lambda: activation, axis_size=width_size), axis_size=depth
        )()
        if out_size == "scalar":
            self.final_activation = final_activation
        else:
            self.final_activation = filter_vmap(
                lambda: final_activation, axis_size=out_size
            )()
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @jax.named_scope("eqx.nn.MLP")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        for i, layer in enumerate(self.layers[:-1]):
            if i == 0:
                x = self.dropout(x, key=key)
            x = layer(x)
            layer_activation = jtu.tree_map(
                lambda x: x[i] if is_array(x) else x, self.activation
            )
            x = filter_vmap(lambda a, b: a(b))(layer_activation, x)
            
        x = self.layers[-1](x)
        if self.out_size == "scalar":
            x = self.final_activation(x)
        else:
            x = filter_vmap(lambda a, b: a(b))(self.final_activation, x)
        return x

class Func(eqx.Module):
    mlp: MLP_dropout
    mat_end: jnp.array
    post_proc_func: Callable

    def __init__(self, data_size, width_size, depth, out_size=None, dropout=0, *, key, mat_end=None, inside_activation=None, activation=None, post_proc_func=_identity, **kwargs):
        super().__init__(**kwargs)

        if out_size is None:
            out_size = data_size

        my_f = lambda x: jnp.exp(-x**2)

        activation = _identity if activation is None else activation
        inside_activation = jnn.softplus if inside_activation is None else inside_activation

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

    def __call__(self, y, k=None, train=False):
        if not train and self.mat_end is not None:
            return self.post_proc_func(self.mat_end @ self.mlp(y, key=k))
        if not train:
            return self.post_proc_func(self.mlp(y, key=k))
        else:
            return self.mlp(y, key=k)

class v_vt_class(eqx.Module):
    v: jnp.array
    vt: jnp.array

    def __init__(self, dim0, dim1, num_nodes=1, **kwargs):
        super().__init__(**kwargs)
        self.v = jrandom.uniform(jrandom.PRNGKey(1200), (dim0, num_nodes), minval=-1, maxval=1)
        self.v = jax.vmap(lambda x: x / jnp.linalg.norm(x), in_axes=[-1])(self.v).T
        self.vt = jrandom.uniform(jrandom.PRNGKey(1200), (dim1, num_nodes), minval=-1, maxval=1)
        self.vt = jax.vmap(lambda x: x / jnp.linalg.norm(x), in_axes=[-1])(self.vt)
        
    
    def __call__(self, idx):
        return jnp.sum(jax.vmap(lambda o1, o2: jnp.outer(o1, o2), in_axes=[-1, 0])(self.v, self.vt[:, idx]), axis=0)
    
class simple_AE_operator(eqx.Module):
    func_encode: Func
    func_decode: Func
    v_vt1: v_vt_class

    def __init__(self, func_enc, func_dec, v_vt1=None, **kwargs) -> None:
        self.func_encode = func_enc
        self.func_decode = func_dec
        self.v_vt1 = v_vt1
    
    def __call__(self, ys, n_mode=1, key=jrandom.PRNGKey(0), train=False):
        x = self.func_encode(ys)
        y = self.func_decode(x, key, train)
        return x, y
    
def find_weighted_loss(terms, weight_vals=None):
        terms = jnp.asarray(terms, dtype=jnp.float32)
        total = jnp.sum(jnp.abs(terms))
        if weight_vals is None:
            weights = jnp.asarray([jnp.abs(term)/total for term in terms])
        else:
            weights = weight_vals
        res = jnp.multiply(terms, weights)
        return sum(res)

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    # assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            arrs = tuple(itemgetter(*batch_perm)(array) for array in arrays) # Works for lists and arrays
            if batch_size != 1:
                yield [arr if None in arr else jnp.array(arr) for arr in arrs]
            else:
                yield [[arr] if arr is None else jnp.expand_dims(jnp.array(arr), axis=0) for arr in arrs]
            start = end
            end = start + batch_size

def make_model(key, data_size, data_size_end, latent, dropout, WX_, v_vt, num_modes_vvt, width_enc, depth_enc, width_dec, depth_dec, activation_enc, activation_dec, post_proc_func=_identity, **kwargs):
    func_encode = Func(data_size, width_enc, depth_enc, out_size=latent, activation=activation_enc, key=key)
    if WX_:
        func_encode = WX(data_size, latent)
    func_decode = Func(latent, width_dec, depth_dec, out_size=data_size, dropout=dropout, activation=activation_dec, key=key, post_proc_func=post_proc_func)
    print(f"Dimension of latent space is {latent} and data size is {data_size}")
    if v_vt:
        vvt = v_vt_class(latent, data_size_end, num_nodes=num_modes_vvt)

    model = simple_AE_operator(func_encode, func_decode, None) if not v_vt else simple_AE_operator(func_encode, func_decode, vvt)
    return model

def train_loop_AE(
    ys,
    step_st=[3000, 3000], #000, 8000],
    lr_st=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
    width_enc=64,
    depth_enc=2,
    width_dec=64,
    depth_dec=6,
    activation_enc = None,
    activation_dec = None,
    loss_func=None,
    post_proc_func=_identity,
    seed=5678,
    print_every=100,
    stagn_every=40,
    reg=False,
    batch_size_st=[32, 32, 32, 32, 32],
    n_mode=-1,
    num_modes_vvt=None,
    v_vt=False,
    latent=1, # 0.05
    dropout=0, 
    WX_=True,
    p_vals=None,
    mat_end=None,
    **kwargs
):
    key = jr.PRNGKey(seed)
    loader_key, dropout_key = jr.split(key, 2)
    seed = 0
    model_key = jr.PRNGKey(seed)
    parameters = {"data_size": int(ys.shape[0]), "activation_enc":activation_enc, "post_proc_func":post_proc_func , "activation_dec":activation_dec, "data_size_end": int(ys.shape[-1]), "latent": latent, "dropout": dropout, "WX_": WX_, "v_vt": v_vt, "num_modes_vvt": num_modes_vvt, "width_enc": width_enc, "depth_enc": depth_enc, "width_dec": width_dec, "depth_dec": depth_dec, "seed": seed}
    model = make_model(key=model_key, **parameters)
    loss_func = lambda x1, x2 : jnp.linalg.norm(x1-x2)/jnp.linalg.norm(x2)*100
    @eqx.filter_value_and_grad
    def grad_loss(model, input, bs, idx, key, pv):
        _, y  = model(input, n_mode, key, True)
        wv = jnp.array([1.,])
        return find_weighted_loss([loss_func(y, input)], weight_vals=wv) 

    @eqx.filter_jit
    def make_step(input, model, opt_state, bs, idx, key, pv):
        loss_func = grad_loss
        loss, grads = loss_func(model, input, bs, idx, key, pv)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    print("Training the Vanilla AE...")

    for steps, lr, batch_size in zip(step_st, lr_st, batch_size_st):

        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        stagn_num = 0
        loss_old = jnp.inf
        t_t = 0

        if (batch_size > ys.shape[-1]) or batch_size == -1:
            batch_size = ys.shape[-1]

        keys = jr.split(dropout_key, steps)
        for step, (yb, idx, key, pv) in zip(range(steps), dataloader([ys.T, jnp.arange(0, ys.shape[-1], 1), keys, p_vals], batch_size, key=loader_key)):
            start = time.time()
            loss, model, opt_state = make_step(yb.T, model, opt_state, batch_size, idx, key[0], pv)
            end = time.time()
            t_t += end - start
            if (step % stagn_every) == 0:
                if jnp.abs(loss_old - loss)/jnp.abs(loss_old)*100 < 1:
                    stagn_num += 1
                    if stagn_num > 10:
                        print("Stagnated....")
                        break
                loss_old = loss

            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {t_t}")
                t_t = 0
    model = eqx.nn.inference_mode(model)
    x, y = model(ys, n_mode, 0, True)
    _, y_o = model(ys, n_mode, 0, False)

    return y, x, model, parameters, y_o


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
        except TypeError:
            if to_array:
                return jnp.array(sols)
            else:
                return sols
    return map_func

def post_process_AE(p_vals, p_test, problem, method, x_train, y_pred_train, y_pred_test, ys, ys_test, y_original=None, y_pred_train_o=None, y_test_o=None, y_pred_test_o=None, changed=False, x_test_modes=None, test=True, file=None):
    import matplotlib
    matplotlib.rc('xtick', labelsize=15) 
    matplotlib.rc('ytick', labelsize=15)
    plt.rcParams["figure.figsize"] = (10, 12)

    error_train = jnp.linalg.norm(y_pred_train-ys)/jnp.linalg.norm(ys)*100
    print(f"Error for train is {error_train}")
    
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
        error_orig = jnp.linalg.norm(y_original-y_pred_train_o)/jnp.linalg.norm(y_original)*100
        print(f"Error for original train is {error_orig}")
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
        error_train_o = jnp.linalg.norm(y_pred_train_o-y_original)/jnp.linalg.norm(y_original)*100
        if file is not None:
            plt.savefig(os.path.join(file, f"decoder_orig_{problem}_{method}.pdf"))
        plt.show()
    else:
        error_train_o = None
    x_train_modes = x_train
    if test:
        error_test = jnp.linalg.norm(y_pred_test-ys_test)/jnp.linalg.norm(ys_test)*100
        print(f"Error for test is {error_test}")
        print("Plotting latent space for test")
        if x_test_modes is None:
            raise NotImplementedError
        # if x_train.shape[0] >1:
        #     for i, (x_train, x_test) in enumerate(zip(x_train_modes, x_test_modes)):
        #             lab = f"Train-mode{i}" if j == 0 else "_no_legend_"
        #             plt.plot(x_train, color="blue", label=lab)
        #         for j, xt in enumerate(x_test.T):
        #             lab_ = f"Test-mode{i}" if j == 0 else "_no_legend_"
        #             plt.plot(xt, color="red", label=lab_)
        #         plt.ylabel(r"$Y$", fontsize=20)
        #         plt.xlabel(r"$t$", fontsize=20)
        #         plt.legend()
        #         plt.title(f"Latent space for mode {i}", fontsize=20)
        #         if file is not None:
        #             plt.savefig(os.path.join(file, f"latent_mode_{i}.pdf"))
        #         plt.show()
        for i, (x_tr, x_t) in enumerate(zip(x_train_modes, x_test_modes)): # , p_vals.T, p_test.T

            if p_vals.shape[-1] == 1:
                plt.scatter(p_vals,  x_tr, label=f"Train-coeffs{i}", color="blue")
                plt.scatter(p_test, x_t, label=f"Test-coeffs{i}", color="red")
            else:
                plt.scatter(jnp.arange(0, x_tr.shape[0], 1), x_tr, color="blue", label=f"Train-coeffs{i}")

            plt.ylabel(r"$Y$", fontsize=20)
            plt.xlabel(r"$t$", fontsize=20)
            plt.legend()
            plt.title(f"Latent space - mode {i}", fontsize=20)
            if file is not None:
                plt.savefig(os.path.join(file, f"latent-mode{i}.pdf"))
            plt.show()

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
            error_orig_test = jnp.linalg.norm(y_test_o-y_pred_test_o)/jnp.linalg.norm(y_test_o)*100
            print(f"Error for original test is {error_orig_test}")
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
            error_test_o = jnp.linalg.norm(y_pred_test_o-y_test_o)/jnp.linalg.norm(y_test_o)*100
            if file is not None:
                plt.savefig(os.path.join(file, f"decoder_orig_test_{problem}_{method}.pdf"))
            plt.show()
        else:
            error_test_o = None

    return error_train, error_test, error_train_o, error_test_o

def is_test_inside(p_vals, p_test):
    check_func = lambda p_v, p_t: (jnp.min(p_v, axis=0) < p_t).all() & (p_t < jnp.max(p_v, axis=0)).all()
    if p_vals.shape[-1] == 1:
        return check_func(p_vals, p_test)
    else:
        is_in_rect = lambda vals: (jnp.sum(vals, axis=1) == 2).any()  and (jnp.sum(vals, axis=1) == 0).any() and ((jnp.sum(vals, axis=1) == 1) & (vals[:, 0] == 1)).any() and ((jnp.sum(vals, axis=1) == 1) & (vals[:, 0] == 0)).any() 
        check_func_2 = lambda p_t: is_in_rect(p_vals > p_t)
        return my_vmap(check_func_2)(p_test).all()

def determine_u_now(ys, eps=1, ret=False, full_matrices=True):
    u, sv, v = jnp.linalg.svd(ys, full_matrices=full_matrices)
    def to_scan(state, inp):
        u_n = u[:, inp]
        s_n = sv[inp]
        v_n = v[inp]
        pred = s_n*jnp.outer(u_n, v_n)
        return ((state[0].at[state[1]].set(pred), state[1]+1), None)
    truncs = jnp.cumsum(jax.lax.scan(to_scan, (jnp.zeros((sv.shape[0], ys.shape[0], ys.shape[1])), 0), jnp.arange(0, sv.shape[0], 1))[0][0], axis=0)
    errors = jax.vmap(lambda app: jnp.linalg.norm(ys-app)/jnp.linalg.norm(ys)*100)(truncs)
    n_mode = jnp.argmax(errors < eps)
    n_mode = n_mode if n_mode != 0 else 1
    print(f"Number of modes for initial V is {n_mode}")
    u_now = u[:, :n_mode]
    if ret:
        return u_now, sv[:n_mode], v[:n_mode, :]
    return u_now

def p_of_dim_1_AE(x_m, p_vals, p_test, model):
    def evaluate_at_parameter(idx, p_val, v):
        v_1 = v[idx]
        p_1 = p_vals[idx]
        v_2 = v[idx+1]
        p_2 = p_vals[idx+1]
        return v_1 + (p_val-p_1)*(v_2-v_1)/(p_2-p_1)
    assert (jnp.diff(p_vals) > 0).all() or (jnp.diff(p_vals) < 0).all()
    ids = jnp.array(my_vmap(lambda p: jnp.where((p_vals[:-1] <= p) & (p_vals[1:] >= p))[0])(p_test))
    x_test = jax.vmap(jax.vmap(evaluate_at_parameter, in_axes=[0, 0, None]), in_axes=[None, None, 0])(ids, p_test[:, 0], x_m)[..., 0]
    y_test = model.func_decode(x_test, train=True)
    y_test_o = model.func_decode(x_test, train=False)
    return x_test, y_test, y_test_o

def interpolate_pimm_data(R, T, ts): # for T decroissant
    def to_map_over_ts(t):
        if t < np.min(T):
            return R[-1]
        if t > np.max(T):
            return R[0]
        idx = np.where((T[1:] <= t) & (T[:-1] >= t))[0][0]
        t1, t2 = T[idx], T[idx+1]
        r1, r2 = R[idx], R[idx+1]
        interpolated_r = r1 + (r2 - r1) * (t - t1) / (t2 - t1)
        return interpolated_r
    return my_vmap(to_map_over_ts)(ts)

def get_v_test_multi_p(p_train, vt_train, p_test, graph=False):

    p_train = np.array(p_train)
    p_test = np.array(p_test)
    vt_train = np.array(vt_train)

    def to_map_over_p_test(pt):
        indices = np.argsort(jnp.sum(jnp.square(p_train-pt), axis=1))
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

def p_of_dim_2_AE(x_m, p_vals, p_test, model):
    res = []
    for i, x_now in enumerate(x_m):
        res_ = get_v_test_multi_p(p_vals, x_now, p_test, graph=True)
        res.append(get_v_test_multi_p(p_vals, x_now, p_test, graph=False))
        
    x_test = jnp.stack(res, axis=0)
    
    plt.scatter(p_vals[:, 0], p_vals[:, 1], color="blue", label = "Train")
    for i, (p_now, re) in enumerate(zip(p_test, res_)):
        legends = ["Test", "Interp"] if i == 0 else ["_no_legend", "_no_legend"]
        plt.scatter(p_now[0], p_now[1], color="green", label = legends[0])
        plt.scatter(re[:, 0], re[:, 1], color="red", label = legends[1])
        plt.plot(jnp.concatenate((re[:, 0], jnp.expand_dims(re[0, 0], 0))), jnp.concatenate((re[:, 1], jnp.expand_dims(re[0, 1], 0))), color="red")
    plt.legend()
    plt.show()
    y_test_ = model.func_decode(x_test, train=True)
    y_test_o = model.func_decode(x_test, train=False)
    return x_test, y_test_, y_test_o,

def normalize(x_tr, x_t, axis=0):
    means = jnp.mean(x_tr, axis=axis)
    std = jnp.std(x_tr, axis=axis)
    return (x_tr-means)/std, (x_t-means)/std

def main_RRAE(method, prob_name, data_func, train_func, train_nn=True,**kwargs):


    folder = f"{prob_name}/{prob_name}_{method}"
    folder_name = f"{folder}/"
    filename = os.path.join(folder_name, f"{method}_{prob_name}")
    ts, y_shift, y_test, p_vals, p_test, lambda_post, y_original, y_test_original = data_func(**kwargs)
    lambda_post = _identity if lambda_post is None else lambda_post
    assert is_test_inside(p_vals, p_test)

    print(f"Shape of y_train is {y_shift.shape}, (T x N)")
    print(f"Shape of p_vals is {p_vals.shape}, (N x P)")
    
    y_pred_train, x_m, model, parameters, y_pred_train_o = train_func(y_shift, dropout=0, p_vals=p_vals, post_proc_func=lambda_post, **kwargs)

    pdb.set_trace()
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.rmtree(folder)
    os.makedirs(folder)
    if (not train_nn) and  (p_vals.shape[-1] != 1) and (p_vals.shape[-1] != 2):
        print("Only P = 1 or P = 2 are supported without training a Neural Network , switching to training mode")
        train_nn = True
        
    process_func = p_of_dim_2_AE if p_vals.shape[-1] == 2 else p_of_dim_1_AE
    x_test, y_pred_test, y_pred_test_o = process_func(x_m, p_vals, p_test, model)

    kwargs = {k: kwargs[k] for k in set(list(kwargs.keys())) - set({"activation_enc", "activation_dec", "loss_func", "post_proc_func"})}

    error_train, error_test, error_train_o, error_test_o = post_process_AE(p_vals, p_test, problem, method, x_m, y_pred_train, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, x_test_modes=x_test, file=folder_name)

    with open(f"{filename}_.pkl", 'wb') as f:
        dill.dump([None, None, None, x_m, y_pred_train, x_test, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, ts, error_train, error_test, error_train_o, error_test_o, p_vals, p_test, kwargs, []], f)


    print(f"Folder name is {folder_name}")
    print(f"Base filename is {filename}")
    if train_nn:
        print("Use them in train_alpha_nn.py to train a NN to get the coeffs.")
    return x_m, y_pred_train, y_pred_test, y_shift, y_test, y_original, y_pred_train_o, y_test_original, y_pred_test_o, folder_name

if __name__ == "__main__":

    method = "AE"
    problem = "avrami" # "shift", "accelerate", "stairs", "mult_freqs", "pimm_curves", "angelo", "mult_gausses", "avrami", "avrami_noise"
    train_nn = False
    kwargs = {"latent": 2, "problem": problem, "step_st": [2000, 2000, 2000], "lr_st": [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9], "width_enc": 64, "depth_enc": 1, "width_dec": 64, "depth_dec": 6, "batch_size_st":[-1, -1, -1,],}
    results = main_RRAE(method, problem, get_data, train_loop_AE, train_nn, **kwargs)
    pdb.set_trace()
    