from collections.abc import Callable
from typing import (
    Literal,
    Optional,
    Union,
)
from equinox.nn._linear import Linear
from equinox.nn import Conv2d, ConvTranspose2d
from equinox._module import field, Module
import equinox as eqx
import jax.random as jrandom
import jax.numpy as jnp
from equinox._vmap_pmap import filter_vmap
from jaxtyping import Array, PRNGKeyArray
import jax.nn as jnn
from equinox._doc_utils import doc_repr
import jax
import jax.tree_util as jtu
from equinox._filters import is_array
import pdb
from operator import itemgetter
import numpy as np
import warnings
from itertools import cycle
import dill
from tqdm import tqdm
import jax.lax as lax
from jax._src.lax import lax as lax_internal
from jax import custom_jvp
import jax.tree_util as jtu

_identity = doc_repr(lambda x: x, "lambda x: x")
_relu = doc_repr(jnn.relu, "<function relu>")


def get_diff_func(x_1, x_2, V_1, V_2, type="tanh"):
    match type:
        case "tanh":
            eps = 0.00001
            a = (V_2 - V_1) / 2
            c = (V_2 + V_1) / 2
            sn = jnp.sign(V_2 - V_1)
            F_1 = jnp.arctanh((V_1 + sn * eps - c) / a)
            F_2 = jnp.arctanh((V_2 - sn * eps - c) / a)
            d = (F_2 - F_1) / (x_2 - x_1)
            b = F_2 - d * x_2
            inter_func = lambda x: a * jnp.tanh(d * x + b) + c
                
        case "line":
            inter_func = lambda x: (V_2 - V_1) / (x_2 - x_1) * (x - x_1) + V_1
            
    def func(x):
        bl = ((x > x_2) + (x < x_1)) > 0
        other = (x > x_2) * V_2 + (x < x_1) * V_1
        inter = inter_func(x) 
        return (1 - bl) * inter + bl * other
    
    return func


def tree_map(f, tree, *rest, is_leaf=None):
    def filtered_f(leaf):
        if (
            callable(leaf)
            or (isinstance(leaf, int) and not isinstance(leaf, bool))
            or (leaf is None)
        ):
            return leaf
        else:
            return f(leaf)

    return jtu.tree_map(filtered_f, tree, *rest, is_leaf=is_leaf)


def _extract_diagonal(s: Array) -> Array:
    """Extract the diagonal from a batched matrix"""
    i = lax.iota("int32", min(s.shape[-2], s.shape[-1]))
    return s[..., i, i]


def _construct_diagonal(s: Array) -> Array:
    """Construct a (batched) diagonal matrix"""
    i = lax.iota("int32", s.shape[-1])
    return lax.full((*s.shape, s.shape[-1]), 0, s.dtype).at[..., i, i].set(s)


def _H(x: Array) -> Array:
    return _T(x).conj()


def _T(x: Array) -> Array:
    return lax.transpose(x, (*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))


@custom_jvp
def stable_SVD(x):
    return jnp.linalg.svd(x, full_matrices=False)


@stable_SVD.defjvp
def _svd_jvp_rule(primals, tangents):
    (A,) = primals
    (dA,) = tangents
    U, s, Vt = jnp.linalg.svd(A, full_matrices=False)

    s = (s / s[0] * 100 >= 1e-9) * s

    Ut, V = _H(U), _H(Vt)
    s_dim = s[..., None, :]
    dS = Ut @ dA @ V
    ds = _extract_diagonal(dS.real)

    s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
    s_diffs_zeros = (s_diffs == 0).astype(s_diffs.dtype)
    F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
    dSS = s_dim.astype(A.dtype) * dS  # dS.dot(jnp.diag(s))
    SdS = _T(s_dim.astype(A.dtype)) * dS  # jnp.diag(s).dot(dS)

    s_zeros = (s == 0).astype(s.dtype)
    s_inv = 1 / (s + s_zeros) - s_zeros
    s_inv_mat = _construct_diagonal(s_inv)
    dUdV_diag = 0.5 * (dS - _H(dS)) * s_inv_mat.astype(A.dtype)
    dU = U @ (F.astype(A.dtype) * (dSS + _H(dSS)) + dUdV_diag)
    dV = V @ (F.astype(A.dtype) * (SdS + _H(SdS)))

    m, n = A.shape[-2:]
    if m > n:
        dAV = dA @ V
        dU = dU + (dAV - U @ (Ut @ dAV)) * s_inv.astype(A.dtype)
    if n > m:
        dAHU = _H(dA) @ U
        dV = dV + (dAHU - V @ (Vt @ dAHU)) * s_inv.astype(A.dtype)

    return (U, s, Vt), (dU, ds, _H(dV))


def loss_generator(which=None, norm_loss_=None):
    if norm_loss_ is None:
        norm_loss_ = lambda x1, x2: jnp.linalg.norm(x1 - x2) / jnp.linalg.norm(x2) * 100

    if (which == "default") or (which == "Vanilla"):

        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(diff_model, static_model, input, out, idx, **kwargs):
            model = eqx.combine(diff_model, static_model)
            pred = model(input, inv_norm_out=False)
            wv = jnp.array([1.0])
            aux = {"loss": norm_loss_(pred, out)}
            return find_weighted_loss([norm_loss_(pred, out)], weight_vals=wv), aux
        
    elif which == "Strong":
        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(diff_model, static_model, input, out, idx, k_max, **kwargs):
            model = eqx.combine(diff_model, static_model)
            pred = model(input, k_max=k_max, inv_norm_out=False)
            wv = jnp.array([1.0])
            aux = {"loss": norm_loss_(pred, out), "k_max": k_max}
            return find_weighted_loss([norm_loss_(pred, out)], weight_vals=wv), aux
    
    elif which == "Sparse":

        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(
            diff_model, static_model, input, out, idx, sparsity=0.05, beta=1.0, **kwargs
        ):
            model = eqx.combine(diff_model, static_model)
            pred = model(input, inv_norm_out=False)
            lat = model.latent(input)
            wv = jnp.array([1.0, beta])
            sparse_term = sparsity * jnp.log(sparsity / (jnp.mean(lat) + 1e-8)) + (
                1 - sparsity
            ) * jnp.log((1 - sparsity) / (1 - jnp.mean(lat) + 1e-8))

            aux = {"loss rec": norm_loss_(pred, out), "loss sparse": sparse_term}
            return find_weighted_loss(
                [
                    norm_loss_(pred, out),
                    sparse_term
                ],
                weight_vals=wv,
            ), aux

    elif which == "Weak":
        raise NotImplementedError
        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(
            diff_model, static_model, input, out, idx, norm_loss=None, **kwargs
        ):
            model = eqx.combine(diff_model, static_model)
            if norm_loss is None:
                norm_loss = norm_loss_
            pred = model(input)
            x = model.latent(input)
            wv = jnp.array([1.0, 1.0])
            return find_weighted_loss(
                [
                    norm_loss(pred, out),
                    jnp.linalg.norm(x - model.v_vt()[:, idx])
                    / jnp.linalg.norm(x)
                    * 100,
                ],
                weight_vals=wv,
            ), (
                norm_loss(pred, out),
                jnp.linalg.norm(x - model.v_vt()[:, idx]) / jnp.linalg.norm(x) * 100,
            )

    elif which == "nuc":

        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(
            diff_model,
            static_model,
            input,
            out,
            idx,
            lambda_nuc=0.001,
            norm_loss=None,
            find_layer=None,
            **kwargs,
        ):
            model = eqx.combine(diff_model, static_model)
            if norm_loss is None:
                norm_loss = norm_loss_
            pred = model(input)
            wv = jnp.array([1.0, lambda_nuc])

            if find_layer is None:
                raise ValueError(
                    "To use LoRAE, you should specify how to find the layer for "
                    "which we add the nuclear norm in the loss. To do so, give the path "
                    "to the layer as loss kwargs to the trainor: "
                    'e.g.: \n"loss_kwargs": {"find_layer": lambda model: model.encode.layers[-2].layers_l[-1].weight} (for predefined CNN AE) \n'
                    '"loss_kwargs": {"find_layer": lambda model: model.encode.layers_l[-1].weight} (for predefined MLP AE).'
                )
            else:
                weight = find_layer(model)

            aux = {"loss rec": norm_loss_(pred, out), "loss nuc": jnp.linalg.norm(weight, "nuc")}
            return (
                find_weighted_loss(
                    [
                        norm_loss(pred, out),
                        jnp.linalg.norm(weight, "nuc"),
                    ],
                    weight_vals=wv,
                ),
                aux
            )

    elif which == "var":

        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(diff_model, static_model, input, out, idx, epsilon, beta=1.0, **kwargs):
            model = eqx.combine(diff_model, static_model)
            lat, means, logvars = model.latent(input, epsilon=epsilon, return_lat_dist=True)
            pred = model.decode(lat)
            wv = jnp.array([1.0, beta])
            kl_loss = jnp.sum(
                -0.5 * (1 + logvars - jnp.square(means) - jnp.exp(logvars))
            )

            aux = {
                "loss rec": norm_loss_(pred, out),
                "loss kl": kl_loss,
            }

            return find_weighted_loss(
                [norm_loss_(pred, out), kl_loss], weight_vals=wv
            ), aux
        
    elif "Contractive":

        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(diff_model, static_model, input, out, idx, beta=1.0, find_weight=None, **kwargs):
            assert find_weight is not None
            model = eqx.combine(diff_model, static_model)
            lat = model.latent(input)
            pred = model(input, inv_norm_out=False)
            W = find_weight(model)
            dh = lat * (1 - lat)
            dh = dh.T
            loss_contr = jnp.sum(jnp.matmul(dh**2, jnp.square(W)))
            wv = jnp.array([1.0, beta])
            aux = {"loss": norm_loss_(pred, out), "cont": loss_contr}
            return find_weighted_loss([norm_loss_(pred, out), loss_contr], weight_vals=wv), aux
    else:
        raise ValueError(f"{which} is an Unknown loss type")

    return loss_fun


def out_to_pic(out):
    return jnp.reshape(
        out,
        (
            int(jnp.sqrt(out.shape[0])),
            int(jnp.sqrt(out.shape[0])),
            out.shape[-1],
        ),
    ).T


def remove_keys_from_dict(d, keys):
    return {k: v for k, v in d.items() if k not in keys}


def merge_dicts(d1, d2):
    return {**d1, **d2}


def v_print(s, v, f=False):
    if v:
        print(s, flush=f)


def countList(lst1, lst2):
    return [sub[item] for item in range(len(lst2)) for sub in [lst1, lst2]]


def eval_model_fixed_coeffs(model, input, batch_size=None, v_ref=None, *, key):
    batch_size = input.shape[-1] if batch_size is None else batch_size
    all_vts = []
    all_preds = []
    idxs = []
    for step, (input_b, idx) in enumerate(
        zip(
            dataloader(
                [input.T, jnp.arange(0, input.shape[-1], 1)],
                batch_size,
                key=key,
                once=True,
            )
        ),
    ):
        pred = model(input_b.T)

        if v_ref is None:
            v_ref, vt_now = model.latent(input_b.T, ret=True)
        else:
            vt_now = v_ref.T @ model.latent(input_b.T)

        # all_vs.append(v)
        all_vts.append(vt_now)
        all_preds.append(pred)
        idxs.append(idx)

    idxs = jnp.concatenate(idxs)
    y_pred_train = jnp.concatenate(all_preds, -1)[..., jnp.argsort(idxs)]
    vt_train = jnp.concatenate(all_vts, -1)[..., jnp.argsort(idxs)]
    v = v_ref
    return y_pred_train, vt_train, v


def divide_return(
    inp_all,
    p_all=None,
    output=None,
    prop_train=0.8,
    pod=False,
    test_end=0,
    eps=1,
    pre_func_in=lambda x: x,
    pre_func_out=lambda x: x,
    args=(),
):
    """p_all of shape (P x N) and y_all of shape (T x N).
    The function divides into train/test according to the parameters
    to allow the test set to be interpolated linearly from the training set
    (if possible). If test_end is specified this is overwridden to only take
    the lest test_end values for testing.

    NOTE: pre_func_in and pre_func_out are functions you want to apply over
    the input and output but you can not do it on all the data since it is
    too big (e.g. conversion to float). These functions will be applied on
    batches during training/evaluation of the Network."""

    if test_end == 0:
        if p_all is not None:
            res = jnp.stack(
                my_vmap(lambda p: (p > jnp.min(p)) & (p < jnp.max(p)))(p_all.T)
            ).T
            idx = jnp.linspace(0, res.shape[0] - 1, res.shape[0], dtype=int)
            cbt_idx = idx[jnp.sum(res, 1) == res.shape[1]]  # could be test
            permut_idx = jrandom.permutation(jrandom.PRNGKey(200), cbt_idx)
            idx_test = permut_idx[: int(res.shape[0] * (1 - prop_train))]
            if cbt_idx.shape[0] < res.shape[0] * (1 - prop_train):
                raise ValueError("Not enough data to got an interpolable test")
            p_test = p_all[idx_test]
            p_train = jrandom.permutation(
                jrandom.PRNGKey(200), jnp.delete(p_all, idx_test, 0)
            )

        else:
            idx_test = jrandom.permutation(jrandom.PRNGKey(200), inp_all.shape[-1])[
                : int(inp_all.shape[-1] * (1 - prop_train))
            ]

        x_test = inp_all[..., idx_test]
        x_train = jrandom.permutation(
            jrandom.PRNGKey(200), jnp.delete(inp_all, idx_test, -1), -1
        )
    else:
        if p_all is not None:
            p_test = p_all[-test_end:]
            p_train = p_all[: len(p_all) - test_end]
        x_test = inp_all[..., -test_end:]
        x_train = inp_all[..., : inp_all.shape[-1] - test_end]

    if output is None:
        output_train = x_train
        output_test = x_test
    else:
        output_test = output[idx_test]
        output_train = jnp.delete(output, idx_test, 0)

    if p_all is not None:
        p_train = jnp.expand_dims(p_train, -1) if len(p_train.shape) == 1 else p_train
        p_test = jnp.expand_dims(p_test, -1) if len(p_test.shape) == 1 else p_test
    else:
        p_train = None
        p_test = None

    if not pod:
        return (
            x_train,
            x_test,
            p_train,
            p_test,
            output_train,
            output_test,
            pre_func_in,
            pre_func_out,
            args,
        )

    u_now, _, _ = adaptive_TSVD(x_train, eps=eps, verbose=True)
    coeffs_train = u_now.T @ x_train
    mean_vals = jnp.mean(coeffs_train, axis=1)
    std_vals = jnp.std(coeffs_train, axis=1)
    coeffs_train = jax.vmap(lambda x, m, s: (x - m) / s)(
        coeffs_train, mean_vals, std_vals
    )

    coeffs_test = u_now.T @ x_test
    coeffs_test = jax.vmap(lambda x, m, s: (x - m) / s)(
        coeffs_test, mean_vals, std_vals
    )

    if output is None:
        output_train = coeffs_train
        output_test = coeffs_test
    else:
        output_test = output[idx_test]
        output_train = jnp.delete(output, idx_test, 0)

    return (
        coeffs_train,
        coeffs_test,
        p_train,
        p_test,
        output_train,
        output_test,
        pre_func_in,
        pre_func_out,
        args,
    )


def get_data(problem, folder=None, google=True, **kwargs):
    """Function that generates the examples presented in the paper."""

    match problem:
        case "2d_gaussian_shift_scale":
            D = 64  # Dimension of the domain
            Ntr = google  # Number of training samples
            Nte = 10000  # Number of test samples
            sigma = 0.2

            def gaussian_2d(x, y, x0, y0, sigma):
                return jnp.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

            x = jnp.linspace(-1, 1, D)
            y = jnp.linspace(-1, 1, D)
            X, Y = jnp.meshgrid(x, y)

            # Create training data
            train_data = []
            x0_vals = jnp.linspace(-0.5, 0.5, int(jnp.sqrt(Ntr)))
            y0_vals = jnp.linspace(-0.5, 0.5, int(jnp.sqrt(Ntr)))
            x0_mesh, y0_mesh = jnp.meshgrid(x0_vals, y0_vals)
            x0_mesh = x0_mesh.flatten()
            y0_mesh = y0_mesh.flatten()

            for i in range(Ntr):
                train_data.append(gaussian_2d(X, Y, x0_mesh[i], y0_mesh[i], sigma))
            train_data = jnp.stack(train_data, axis=-1)

            # Create test data
            key = jrandom.PRNGKey(0)
            x0_vals_test = jrandom.uniform(key, (Nte,), minval=-0.5, maxval=0.5)
            y0_vals_test = jrandom.uniform(key, (Nte,), minval=-0.5, maxval=0.5)
            x0_mesh_test = x0_vals_test
            y0_mesh_test = y0_vals_test

            test_data = []
            for i in range(Nte):
                test_data.append(gaussian_2d(X, Y, x0_mesh_test[i], y0_mesh_test[i], sigma))
            test_data = jnp.stack(test_data, axis=-1)

            # Normalize the data
            train_data = (train_data - jnp.mean(train_data)) / jnp.std(train_data)
            test_data = (test_data - jnp.mean(test_data)) / jnp.std(test_data)

            # Split the data into training and test sets
            x_train = jnp.expand_dims(train_data, 0)
            x_test = jnp.expand_dims(test_data, 0)
            y_train = jnp.expand_dims(train_data, 0)
            y_test = jnp.expand_dims(test_data, 0)
            p_train = jnp.stack([x0_mesh, y0_mesh], axis=-1)
            p_test = jnp.stack([x0_mesh_test, y0_mesh_test], axis=-1)
            return x_train, x_test, p_train, p_test, y_train, y_test, lambda x: x, lambda x: x, ()

        case "Angelo":
            import os

            def f(n):
                return os.path.join(folder, n)

            import h5py

            data = h5py.File(f("Data.mat"), "r")
            data = jnp.array(data["data"]).T
            param = h5py.File(f("Param.mat"), "r")
            param = jnp.array(param["param"]).T

            data = jrandom.permutation(jrandom.key(0), data, axis=-1)
            data = jnp.log(data[::10])
            perc = 0.8
            x_train = data[:, : int(perc * data.shape[-1])]
            x_test = data[:, int(perc * data.shape[-1]) :]
            y_train = x_train
            y_test = x_test
            p_train = param[..., : int(perc * data.shape[-1])]
            p_test = param[..., int(perc * data.shape[-1]) :]
            return (
                x_train,
                x_test,
                None,
                None,
                y_train,
                y_test,
                lambda x: x,
                lambda x: x,
                (),
            )

        case "CIFAR-10":
            import pickle
            import os
            
            def load_cifar10_batch(cifar10_dataset_folder_path, batch_id):
                with open(os.path.join(cifar10_dataset_folder_path, 'data_batch_' + str(batch_id)), mode='rb') as file:
                    batch = pickle.load(file, encoding='latin1')
                features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
                labels = batch['labels']
                return features, labels

            def load_cifar10(cifar10_dataset_folder_path):
                x_train = []
                y_train = []
                for batch_id in range(1, 6):
                    features, labels = load_cifar10_batch(cifar10_dataset_folder_path, batch_id)
                    x_train.extend(features)
                    y_train.extend(labels)
                x_train = jnp.array(x_train)
                y_train = jnp.array(y_train)
                with open(os.path.join(cifar10_dataset_folder_path, 'test_batch'), mode='rb') as file:
                    batch = pickle.load(file, encoding='latin1')
                x_test = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
                y_test = jnp.array(batch['labels'])
                return x_train, x_test, y_train, y_test

            cifar10_dataset_folder_path = folder
            x_train, x_test, y_train, y_test = load_cifar10(cifar10_dataset_folder_path)
            pre_func_in = lambda x: jnp.array(x, dtype=jnp.float32) / 255.0
            pre_func_out = lambda x: jnp.array(x, dtype=jnp.float32) / 255.0
            x_train = jnp.swapaxes(x_train, 0, -1)
            x_test = jnp.swapaxes(x_test, 0, -1)
            return x_train, x_test, None, None, x_train, x_test, pre_func_in, pre_func_out, ()
        
        case "CelebA":
            data_res = 160
            import os
            from PIL import Image
            import numpy as np
            from skimage.transform import resize

            if os.path.exists(f"{folder}/celeba_data_{data_res}.npy"):
                print("Loading data from file")
                data = np.load(f"{folder}/celeba_data_{data_res}.npy")
            else:
                print("Loading data and processing...")
                data = np.load(f"{folder}/celeba_data.npy")
                celeb_transform = lambda im: np.astype(
                    resize(im, (data_res, data_res, 3), order=1, anti_aliasing=True)
                    * 255.0,
                    np.uint8,
                )
                all_data = []
                for i in tqdm(range(data.shape[0])):
                    all_data.append(celeb_transform(data[i]))

                data = np.stack(all_data, axis=0)
                data = jnp.swapaxes(data, 0, 3)
                np.save(f"{folder}/celeba_data_{data_res}.npy", data)

            print("Data shape: ", data.shape)
            x_train = data[..., :162770]
            x_test = data[..., 182638:]
            y_train = x_train
            y_test = x_test
            pre_func_in = lambda x: jnp.astype(x, jnp.float32) / 255.0
            pre_func_out = lambda x: jnp.astype(x, jnp.float32) / 255.0
            return (
                x_train,
                x_test,
                None,
                None,
                y_train,
                y_test,
                pre_func_in,
                pre_func_out,
                (),
            )

        case "Circle":
            import pickle as pkl
            import os

            if os.path.exists(f"{folder}/Inputs_One_Circle_Regular.pickle"):
                print("Loading data from file")
                with open(f"{folder}/Inputs_One_Circle_Regular.pickle", "rb") as f:
                    data = pkl.load(f)
            else:
                raise ValueError("data is missing")
            data = jnp.expand_dims(data, 0)
            data = jrandom.permutation(jrandom.key(100), data, axis=-1)
            perc = 0.8
            x_train = data[..., : int(perc * data.shape[-1])]
            x_test = data[..., int(perc * data.shape[-1]) :]
            return (
                x_train,
                x_test,
                None,
                None,
                x_train,
                x_test,
                lambda x: x,
                lambda x: x,
                (),
            )

        case "skf_ft":
            import os
            import scipy.io

            folder = "skf_transfer_func/"

            def f(n):
                return os.path.join(folder, n)

            y_all = jnp.log(
                jnp.abs(
                    scipy.io.loadmat(f("data_model_sensor1_lhsu.mat"))[
                        "data_model_sensor1_lhsu"
                    ][..., 0]
                )
            )
            p_all = scipy.io.loadmat(f("parameters_set.mat"))["s"]
            return divide_return(y_all, p_all, None, 0.8)

        case "hypersonic":
            import meshio
            import os

            folder = "hypersonic/Hypersonic_snapshots/"

            mashes = []
            datas = []
            i = 0
            for name in os.listdir(folder):
                try:
                    integ = int(name[-3])
                    dec = int(name[-1])
                    try:
                        ten = int(name[-4])
                    except ValueError:
                        ten = 0
                    # if i != 0:
                    mashes.append(ten * 10 + integ + dec / 10)
                except ValueError:
                    print(f"Name {name} skipped.")
                    continue
                mesh = meshio.read(f"{folder}{name}/solution.exo")
                if (ten * 10 + integ + dec / 10) == 8.2:
                    data_ref = mesh.point_data["Mach"]
                datas.append(
                    mesh.point_data["Mach"]
                )  # Temperature, "Pressure", "Density"
                # datas.append(
                #     mesh.point_data["Temperature"]
                # )  # Temperature, "Pressure", "Density"
                i += 1

            p_all = np.expand_dims(np.array(mashes), -1)
            p_all = jnp.delete(p_all, jnp.array([30, 38]), 0)
            y_all = np.stack(datas, -1)
            y_all = jnp.delete(y_all, jnp.array([30, 38]), 1)

            p_all = jnp.sort(p_all, 0)
            y_all = y_all[:, jnp.argsort(p_all, 0).flatten()]

            import matplotlib.pyplot as plt
            from RRAEs.training_classes import Trainor_class
            import scipy

            idx = jnp.argmax(p_all == 9.7)
            y_all = jnp.concatenate(
                (y_all[:, :idx], y_all[:, idx + 1 :], y_all[:, idx : idx + 1]), 1
            )
            p_all = jnp.concatenate(
                (p_all[:idx], p_all[idx + 1 :], p_all[idx : idx + 1]), 0
            )

            def plot_scatter_wing(
                idx,
                y_all,
                xlow=None,
                xhigh=None,
                ylow=None,
                yhigh=None,
                vmax=3,
                typ="interp",
                cmap="seismic",
                **kwargs,
            ):
                def get_plotting_pic(idx, xlow, xhigh, ylow, yhigh, y_all):
                    xyz = mesh.points[:65535]
                    data1 = y_all[:65535, idx]  # plane z=0
                    xs = []
                    ys = []
                    datas = []

                    if xlow is None or xhigh is None or ylow is None or yhigh is None:
                        xlow = min(xyz[:, 0])
                        xhigh = max(xyz[:, 0])
                        ylow = min(xyz[:, 1])
                        yhigh = max(xyz[:, 1])

                    for i in range(xyz.shape[0]):
                        if (
                            xyz[i][0] > xlow
                            and xyz[i][0] < xhigh
                            and xyz[i][1] > ylow
                            and xyz[i][1] < yhigh
                        ):
                            xs.append(xyz[i][0])
                            ys.append(xyz[i][1])
                            datas.append(data1[i])

                    xs0 = jnp.array(xs)
                    ys0 = jnp.array(ys)
                    datas = jnp.array(datas)
                    N = 300j
                    extent = (min(xs0), max(xs0), min(ys0), max(ys0))
                    xs, ys = np.mgrid[
                        extent[0] : extent[1] : N, extent[2] : extent[3] : N
                    ]

                    return (
                        scipy.interpolate.griddata((xs0, ys0), datas, (xs, ys)),
                        extent,
                    )

                true_pic, extent = get_plotting_pic(
                    idx, xlow, xhigh, ylow, yhigh, y_all
                )
                pic_before, _ = get_plotting_pic(
                    idx - 1, xlow, xhigh, ylow, yhigh, y_all
                )
                pic_after, _ = get_plotting_pic(
                    idx + 1, xlow, xhigh, ylow, yhigh, y_all
                )

                fig = plt.figure()
                ax = fig.add_subplot(1, 2, 1)
                im = ax.imshow(true_pic, vmin=0, vmax=vmax, extent=extent, cmap=cmap)
                # fig.colorbar(im, ax=ax, **kwargs)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(
                    "Error between True (9.7) and Linear interpolation (9.6, 9.8)"
                )

                ax = fig.add_subplot(1, 2, 2)
                im = ax.imshow(
                    (pic_after + pic_before) / 2,
                    vmin=0,
                    vmax=vmax,
                    extent=extent,
                    cmap=cmap,
                )
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title("Linear Interpolation")
                # fig.colorbar(im, ax=ax, **kwargs)
                plt.show()

            # plot_scatter_wing(jnp.argmax(p_all==9.7), y_all, ylow=0.04, yhigh=0.1, xlow=0.05, xhigh=0.12)

            return divide_return(
                y_all, p_all, None, 0.8, test_end=1, args=(data_ref, mesh.points)
            )

        case "NVH":
            import pandas as pd
            from scipy.interpolate import interp1d
            import os

            curves_folder = "data-NVH/NVH_CSVs/"
            file_name = os.path.join(curves_folder, "DoE_new_mod.csv")
            mat = pd.read_csv(file_name, header=None)
            p_all = jnp.array(mat[3:], dtype=float)

            outputs_raw_y = []
            N_samples = 98
            N = 9991

            for simidx in range(N_samples):
                df_curve = np.genfromtxt(
                    curves_folder + "Sim_" + str(simidx + 3) + "_mean_all_points.csv",
                    delimiter=",",
                    skip_header=1,
                    skip_footer=0,
                )

            f_min = df_curve[0, 0].T
            f_max = df_curve[-1, 0].T

            # re-discretize in less number of points
            N_max = 1000
            outputs_raw_y = np.zeros((N_samples, int(N_max)))

            for simidx in range(N_samples):
                df_curve = np.genfromtxt(
                    curves_folder + "Sim_" + str(simidx + 3) + "_mean_all_points.csv",
                    delimiter=",",
                    skip_header=1,
                    skip_footer=0,
                )

                f_aux = np.linspace(f_min, f_max, N_max)
                outputs_raw_y[simidx] = interp1d(
                    df_curve[:, 0].T, np.log10(df_curve[:, 1].T)
                )(f_aux)

            y_all = outputs_raw_y
            return divide_return(y_all.T[:400], p_all, None, prop_train=0.9)

        case "antenne":
            with open("antenne_data/second_data_all.pkl", "rb") as f:
                y_all, _ = dill.load(f)
            # y_all = jnp.reshape(y_all, (1600, 100, 100)) # only for border
            y_all = y_all[:, 200:300, 200:300]  # only for not border
            y_all = jnp.concatenate((y_all[0:1600], y_all[1800:-1]))
            y_all = jnp.concatenate(
                (y_all[0:400], y_all[600:800], y_all[1000:1600], y_all[1800:2200])
            )
            y_all = (y_all + 1) / 2
            return divide_return(y_all.T, None, None, 1)

        case "accelerate":
            ts = jnp.linspace(0, 2 * jnp.pi, 200)

            def func(f, x):
                return jnp.sin(f * jnp.pi * x)

            p_vals = jnp.linspace(1 / 3, 1, 15)[:-1]
            y_shift = jax.vmap(func, in_axes=[0, None])(p_vals, ts).T
            p_test = jnp.linspace(1 / 3, 1, 500)
            y_test = jax.vmap(func, in_axes=[0, None])(p_test, ts).T
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return divide_return(y_all, p_all, test_end=y_test.shape[-1])

        case "shift":
            ts = jnp.linspace(0, 2 * jnp.pi, 200)

            def sf_func(s, x):
                return jnp.sin(x - s * jnp.pi)

            p_vals = jnp.linspace(0, 1.8, 5)[:-1]  # 18
            y_shift = jax.vmap(sf_func, in_axes=[0, None])(p_vals, ts).T
            p_test = jnp.linspace(0, jnp.max(p_vals), 500)[1:-1]
            y_test = jax.vmap(sf_func, in_axes=[0, None])(p_test, ts).T
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return divide_return(y_all, p_all, test_end=y_test.shape[-1])
        
        case "test_data_CNN":
            y_all = jrandom.uniform(jrandom.key(0), (1, 100, 120, 200))
            p_all = None
            return divide_return(y_all, p_all, None, 0.8)
         
        case "gaussian_shift":
            ts = jnp.linspace(0, 2 * jnp.pi, 200)
            def gauss_shift(s, x):
                return jnp.exp(-((x - s) ** 2) / 0.1)  # Smaller width
            p_vals = jnp.linspace(1, 2 * jnp.pi +1, 20)
            ts = jnp.linspace(0, 2 * jnp.pi + 2, 500)
            y_shift = jax.vmap(gauss_shift, in_axes=[0, None])(p_vals, ts).T
            p_test = jnp.linspace(jnp.min(p_vals), jnp.max(p_vals), 500)[1:-1]
            y_test = jax.vmap(gauss_shift, in_axes=[0, None])(p_test, ts).T
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return divide_return(y_all, p_all, test_end=y_test.shape[-1])
        
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
            )(y_shift_old).T
            y_shift = y_shift[:, ~jnp.isnan(y_shift).any(axis=0)]

            p_test = jrandom.uniform(
                jrandom.PRNGKey(0),
                (300,),
                minval=jnp.min(p_vals) * 1.00001,
                maxval=jnp.max(p_vals) * 0.99999,
            )
            y_test = jax.vmap(
                lambda y: (y - jnp.mean(y_shift_old)) / jnp.std(y_shift_old)
            )(jax.vmap(create_escal)(p_test)).T
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            ts = jnp.arange(0, y_shift.shape[0], 1)
            return divide_return(y_all, p_all, test_end=y_test.shape[-1])

        case "mult_freqs":
            p_vals_0 = jnp.repeat(jnp.linspace(0.5 * jnp.pi, jnp.pi, 15), 15)
            p_vals_1 = jnp.tile(jnp.linspace(0.3 * jnp.pi, 0.8 * jnp.pi, 15), 15)
            p_vals = jnp.stack([p_vals_0, p_vals_1], axis=-1)
            ts = jnp.arange(0, 5 * jnp.pi, 0.01)
            y_shift = jax.vmap(lambda p: jnp.sin(p[0] * ts) + jnp.sin(p[1] * ts))(
                p_vals
            ).T

            p_vals_0 = jrandom.uniform(
                jrandom.PRNGKey(140),
                (1000,),
                minval=p_vals_0[0] * 1.001,
                maxval=p_vals_0[-1] * 0.999,
            )
            p_vals_1 = jrandom.uniform(
                jrandom.PRNGKey(8),
                (1000,),
                minval=p_vals_1[0] * 1.001,
                maxval=p_vals_1[-1] * 0.999,
            )
            p_test = jnp.stack([p_vals_0, p_vals_1], axis=-1)
            y_test = jax.vmap(lambda p: jnp.sin(p[0] * ts) + jnp.sin(p[1] * ts))(
                p_test
            ).T
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return divide_return(y_all, p_all, test_end=y_test.shape[-1])

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
            return divide_return(y_all, p_all, test_end=y_all_1.shape[-1])

        case "angelo_newest":
            import os
            import pandas as pd

            filename = os.path.join(os.getcwd(), "data_ang_newest/doe_anova_boost/")

            def f(n):
                return os.path.join(filename, n)

            y_all = pd.read_csv(f("db_s22_matrix.csv"), sep=" ").to_numpy()
            p_all = pd.read_csv(f("inputs.csv"), sep=",").to_numpy()
            ts = pd.read_csv(f("freq_red.csv"), sep=" ").to_numpy()

            return divide_return(y_all, p_all)

        case "angelo_final":
            import os
            import pandas as pd

            filename = os.path.join(
                os.getcwd(), "2024_06_21_NVH_transfer/6th_study_multi_anova_transfer/"
            )

            def f(n):
                return os.path.join(filename, n)

            ts = pd.read_csv(f(f"Sim_1_mean_all_points.csv")).to_numpy()[:, 0]
            p_all = pd.read_csv(f("DOE_new_mod.csv")).to_numpy()
            y_all = []
            for i in range(p_all.shape[0]):
                y_all.append(
                    jnp.log(
                        pd.read_csv(f(f"Sim_{i+1}_mean_all_points.csv")).to_numpy()[
                            :, 1
                        ]
                    )
                )
            y_all = jnp.stack(y_all, -1)
            return divide_return(y_all, p_all, prop_train=0.9)

        case "mult_gausses":

            p_vals_0 = jnp.repeat(jnp.linspace(1, 3, 10), 10)
            p_vals_1 = jnp.tile(jnp.linspace(4, 6, 10), 10)
            p_vals = jnp.stack([p_vals_0, p_vals_1], axis=-1)
            p_test_0 = jrandom.uniform(
                jrandom.PRNGKey(1000),
                (1000,),
                minval=p_vals_0[0] * 1.001,
                maxval=p_vals_0[-1] * 0.999,
            )
            p_test_1 = jrandom.uniform(
                jrandom.PRNGKey(50),
                (1000,),
                minval=p_vals_1[0] * 1.001,
                maxval=p_vals_1[-1] * 0.999,
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
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return divide_return(y_all, p_all, test_end=y_test.shape[-1])

        case "avrami_noise":
            n = jnp.repeat(jnp.repeat(jnp.linspace(2, 3.5, 10), 10), 10)
            N = jnp.tile(jnp.repeat(jnp.linspace(1.5, 3, 10), 10), 10)
            G = jnp.tile(jnp.tile(jnp.linspace(1.5, 3, 10), 10), 10)
            p_vals = jnp.stack([n, N, G], axis=-1)
            p_test_0 = jrandom.uniform(
                jrandom.PRNGKey(10),
                (5000,),
                minval=n[0] * 1.0001,
                maxval=n[-1] * 0.99999,
            )
            p_test_1 = jrandom.uniform(
                jrandom.PRNGKey(100),
                (5000,),
                minval=N[0] * 1.00001,
                maxval=N[-1] * 0.99999,
            )
            p_test_2 = jrandom.uniform(
                jrandom.PRNGKey(0),
                (5000,),
                minval=G[0] * 1.00001,
                maxval=G[-1] * 0.999999,
            )
            p_test = jnp.stack([p_test_0, p_test_1, p_test_2], axis=-1)

            ts = jnp.arange(0, 1.5, 0.01)
            y_shift = jax.vmap(
                lambda p, t: 1 - jnp.exp(-jnp.pi * p[1] * p[2] ** 3 / 3 * t ** p[0]),
                in_axes=[0, None],
            )(p_vals, ts).T
            y_test = jax.vmap(
                lambda p, t: 1 - jnp.exp(-jnp.pi * p[1] * p[2] ** 3 / 3 * t ** p[0]),
                in_axes=[0, None],
            )(p_test, ts).T

            to_remove = np.bitwise_or.reduce(
                (p_test >= jnp.max(p_vals, 0)) | (p_test <= jnp.min(p_vals, 0)),
                1,
            )
            p_test = jnp.delete(p_test, to_remove, 0)
            y_test = jnp.delete(y_test, to_remove, 1)
            y_test = jnp.delete(
                y_test,
                np.bitwise_or.reduce((p_test >= jnp.maxp_vals, 0))
                | (p_test <= jnp.min(p_vals, 0)),
                1,
            )
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)

            noise_keys = jrandom.split(jrandom.PRNGKey(0), y_all.shape[-1])
            y_all = jax.vmap(
                lambda y, k: y + jrandom.normal(k, y.shape) * 0.01, in_axes=[-1, 0]
            )(y_all, noise_keys).T
            return divide_return(
                y_all, p_all, eps=1.4, pod=True, test_end=y_test.shape[-1]
            )

        case "avrami":
            n = jnp.repeat(jnp.repeat(jnp.linspace(2.5, 3.5, 3), 3), 3)
            N = jnp.tile(jnp.repeat(jnp.linspace(1.2, 3, 3), 3), 3)
            G = jnp.tile(jnp.tile(jnp.linspace(1.2, 3, 3), 3), 3)
            p_vals = jnp.stack([n, N, G], axis=-1)
            p_test_0 = jrandom.uniform(
                jrandom.PRNGKey(10),
                (2000,),
                minval=n[0] * 1.0001,
                maxval=n[-1] * 0.99999,
            )
            p_test_1 = jrandom.uniform(
                jrandom.PRNGKey(100),
                (2000,),
                minval=N[0] * 1.00001,
                maxval=N[-1] * 0.99999,
            )
            p_test_2 = jrandom.uniform(
                jrandom.PRNGKey(0),
                (2000,),
                minval=G[0] * 1.00001,
                maxval=G[-1] * 0.999999,
            )
            p_test = jnp.stack([p_test_0, p_test_1, p_test_2], axis=-1)

            ts = jnp.arange(0, 1.5, 0.01)
            y_shift = jax.vmap(
                lambda p, t: 1 - jnp.exp(-jnp.pi * p[1] * p[2] ** 3 / 3 * t ** p[0]),
                in_axes=[0, None],
            )(p_vals, ts).T
            y_test = jax.vmap(
                lambda p, t: 1 - jnp.exp(-jnp.pi * p[1] * p[2] ** 3 / 3 * t ** p[0]),
                in_axes=[0, None],
            )(p_test, ts).T

            to_remove = np.bitwise_or.reduce(
                (p_test >= jnp.max(p_vals, 0)) | (p_test <= jnp.min(p_vals, 0)),
                1,
            )
            p_test = jnp.delete(p_test, to_remove, 0)
            y_test = jnp.delete(y_test, to_remove, 1)
            y_test = jnp.delete(
                y_test,
                np.bitwise_or.reduce(
                    (p_test >= jnp.max(p_vals, 0)) | (p_test <= jnp.min(p_vals, 0)),
                    1,
                ),
                1,
            )
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return divide_return(y_all, p_all, test_end=y_test.shape[-1])

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

            p_all = jnp.concatenate([p_all_train, p_all_test], 0)
            y_all = jnp.concatenate([y_all_train, y_all_test], -1)
            return divide_return(y_all, p_all, test_end=p_all_test.shape[0])

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
            return divide_return(
                y_all, jnp.expand_dims(p_all, -1), prop_train=0.8, output=output_all
            )

        case "fashion_mnist":
            import pandas
            x_train = pandas.read_csv("fashin_mnist/fashion-mnist_train.csv").to_numpy().T[1:]
            x_test = pandas.read_csv("fashin_mnist/fashion-mnist_test.csv").to_numpy().T[1:]
            y_all = jnp.concatenate([x_train, x_test], axis=-1)
            y_all = jnp.reshape(y_all, (1, 28, 28, -1))
            pre_func_in = lambda x: jnp.astype(x, jnp.float32) / 255
            return divide_return(y_all, None, test_end=x_test.shape[-1], pre_func_in=pre_func_in, pre_func_out=pre_func_in)
            
        case "mnist":
            import os
            import gzip
            import numpy as np
            import pickle as pkl

            if os.path.exists(f"{folder}/mnist_data.npy"):
                print("Loading data from file")
                with open(f"{folder}/mnist_data.npy", "rb") as f:
                    train_images, train_labels, test_images, test_labels = pkl.load(f)
            else:
                print("Loading data and processing...")

                def load_mnist_images(filename):
                    with gzip.open(filename, 'rb') as f:
                        data = np.frombuffer(f.read(), np.uint8, offset=16)
                        data = data.reshape(-1, 28, 28)
                    return data

                def load_mnist_labels(filename):
                    with gzip.open(filename, 'rb') as f:
                        data = np.frombuffer(f.read(), np.uint8, offset=8)
                    return data

                def load_mnist(path):
                    train_images = load_mnist_images(os.path.join(path, 'train-images-idx3-ubyte.gz'))
                    train_labels = load_mnist_labels(os.path.join(path, 'train-labels-idx1-ubyte.gz'))
                    test_images = load_mnist_images(os.path.join(path, 't10k-images-idx3-ubyte.gz'))
                    test_labels = load_mnist_labels(os.path.join(path, 't10k-labels-idx1-ubyte.gz'))
                    return (train_images, train_labels), (test_images, test_labels)

                def preprocess_mnist(images):
                    images = images.astype(np.float32) / 255.0
                    images = np.expand_dims(images, axis=1)  # Add channel dimension
                    return images

                def get_mnist_data(path):
                    (train_images, train_labels), (test_images, test_labels) = load_mnist(path)
                    train_images = preprocess_mnist(train_images)
                    test_images = preprocess_mnist(test_images)
                    return train_images, train_labels, test_images, test_labels

                train_images, train_labels, test_images, test_labels = get_mnist_data(folder)
                train_images = jnp.swapaxes(jnp.moveaxis(train_images, 1, -1), 0, -1)
                test_images = jnp.swapaxes(jnp.moveaxis(test_images, 1, -1), 0, -1)
                with open(f"{folder}/mnist_data.npy", "wb") as f:
                    pkl.dump((train_images, train_labels, test_images, test_labels), f)

            return (
                train_images,
                test_images,
                None,
                None,
                train_images,
                test_images,
                lambda x: x,
                lambda x: x,
                (),
            )

        case _:
            raise ValueError(f"Problem {problem} not recognized")


def adaptive_TSVD(
    ys, eps=0.2, prop=0.1, full_matrices=True, verbose=True, modes=None, **kwargs
):
    """Adaptive truncated SVD for a given matrix ys.

    Parameters
    ----------
    eps: float
        The tolerance for which we accept a truncated SVD (in %).
    prop: float
        The proportion of the singular values to consider initially.
        This parameter is mainly to avoid comuting a lot of SVDs if
        it is not necessary. The function starts with the given
        value of prop and then multiplies it by 2 until it finds the
        desired tolerance eps.
    full_matrices: bool
        Whether to return the full matrices or not with the SVD.
    verbose: bool
        Whether to print the number of modes found or not.
    """
    if ys.shape[0] == 1 or len(ys.shape) == 1:
        if len(ys.shape) == 1:
            ys = jnp.expand_dims(ys, 0)
        u = jnp.ones((1, ys.shape[0]))
        return u, jnp.array([1.0]), ys

    u, sv, v = jnp.linalg.svd(ys, full_matrices=full_matrices)

    if modes is not None:
        if modes == "all":
            return u, sv, v
        return u[:, :modes], sv[:modes], v[:modes, :]

    def to_scan(state, inp):
        u_n = u[:, inp]
        s_n = sv[inp]
        v_n = v[inp]
        pred = s_n * jnp.outer(u_n, v_n)
        return ((state[0].at[state[1]].set(pred), state[1] + 1), None)

    while True:
        if int(prop * sv.shape[0]) == 0:
            prop = 1
        truncs = jnp.cumsum(
            jax.lax.scan(
                to_scan,
                (jnp.zeros((int(prop * sv.shape[0]), ys.shape[0], ys.shape[1])), 0),
                jnp.arange(0, int(prop * sv.shape[0]), 1),
            )[0][0],
            axis=0,
        )
        errors = jax.vmap(
            lambda app: jnp.linalg.norm(ys - app) / jnp.linalg.norm(ys) * 100
        )(truncs)
        if (errors > eps).all():
            prop *= 2
            continue
        break

    n_mode = jnp.argmax(errors < eps)
    n_mode = n_mode if n_mode != 0 else 1
    v_print(f"Number of modes for initial V is {n_mode}", verbose)
    u_now = u[:, :n_mode]
    return u_now, sv[:n_mode], v[:n_mode, :]


def find_weighted_loss(terms, weight_vals=None):
    terms = jnp.asarray(terms, dtype=jnp.float32)
    total = jnp.sum(jnp.abs(terms))
    if weight_vals is None:
        weights = jnp.asarray([jnp.abs(term) / total for term in terms])
    else:
        weights = weight_vals
    res = jnp.multiply(terms, weights)
    return sum(res)


def v_dataloader(arrays, batch_size, p_vals=None, once=False, *, key, idx_changer=0):
    dataset_size = arrays[0].shape[0]
    arrays = [array if array is not None else [None] * dataset_size for array in arrays]
    indices = jnp.arange(dataset_size)
    kk = idx_changer
    start = 0
    all_zeros = jnp.zeros((arrays[0].shape[0], batch_size))
    idx_batch = jnp.arange(batch_size)

    while True:
        perm = jrandom.permutation(key, indices)
        i = kk
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            I = all_zeros.at[batch_perm, idx_batch].set(1)
            i += 1
            G = jrandom.normal(jrandom.key(i), (arrays[0].shape[0], batch_size)) / 1000
            G = jnp.zeros_like(G)
            X = I + G
            arrs = [(array.T @ X).T for array in arrays]
            start = end
            end = start + batch_size
            yield arrs
        if once:
            if dataset_size % batch_size != 0:
                batch_perm = perm[-(dataset_size % batch_size) :]
                arrs = tuple(
                    itemgetter(*batch_perm)(array) for array in arrays
                )  # Works for lists and arrays
                if dataset_size % batch_size == 1:
                    yield [
                        [arr] if arr is None else jnp.expand_dims(jnp.array(arr), 0)
                        for arr in arrs
                    ]
                else:
                    yield [[arr] if arr is None else jnp.array(arr) for arr in arrs]
            break
        kk += 1


def dataloader(arrays, batch_size, p_vals=None, once=False, *, key_idx):
    dataset_size = arrays[0].shape[0]
    arrays = [array if array is not None else [None] * dataset_size for array in arrays]
    # assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    kk = 0

    while True:
        key = jrandom.key(key_idx + kk)
        perm = jrandom.permutation(key, indices)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            arrs = tuple(
                itemgetter(*batch_perm)(array) for array in arrays
            )  # Works for lists and arrays
            if batch_size != 1:
                yield [jnp.array(arr) for arr in arrs]
            else:
                yield [
                    [arr] if arr is None else jnp.expand_dims(jnp.array(arr), axis=0)
                    for arr in arrs
                ]
            start = end
            end = start + batch_size
        if once:
            if dataset_size % batch_size != 0:
                batch_perm = perm[-(dataset_size % batch_size) :]
                arrs = tuple(
                    itemgetter(*batch_perm)(array) for array in arrays
                )  # Works for lists and arrays
                if dataset_size % batch_size == 1:
                    yield [
                        [arr] if arr is None else jnp.expand_dims(jnp.array(arr), 0)
                        for arr in arrs
                    ]
                else:
                    yield [[arr] if arr is None else jnp.array(arr) for arr in arrs]
            break
        kk += 1


def my_vmap(func, to_array=True):
    def map_func(*arrays, args=None, kwargs=None):
        sols = []
        for elems in zip(*arrays):
            if (args is None) and (kwargs is None):
                sols.append(func(*elems))
            elif (args is not None) and (kwargs is not None):
                sols.append(func(*elems, *args, **kwargs))
            elif args is not None:
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
            if to_array:
                return jnp.array(sols)
            else:
                return sols

    return map_func


class v_vt_class(eqx.Module):
    v: jnp.array
    vt: jnp.array

    def __init__(self, latent_size, data_size, num_modes=1, *, key, **kwargs):
        super().__init__(**kwargs)
        k1, k2 = jrandom.split(key, 2)
        self.v = jrandom.uniform(k1, (latent_size, num_modes), minval=-1, maxval=1)
        self.v = jax.vmap(lambda x: x / jnp.linalg.norm(x), in_axes=[-1], out_axes=-1)(
            self.v
        )
        self.vt = jrandom.uniform(k2, (num_modes, data_size), minval=-1, maxval=1)
        self.vt = jax.vmap(lambda x: x / jnp.linalg.norm(x))(self.vt)

    def __call__(self):
        norm_f = lambda x: x / jnp.linalg.norm(x)
        U_mat = jax.vmap(norm_f, in_axes=[-1], out_axes=-1)(self.v)
        return U_mat @ self.vt


class MLP_with_linear(Module, strict=True):
    layers: tuple[Linear, ...]
    layers_l: tuple[Linear, ...]
    activation: Callable
    use_bias: bool = field(static=True)
    use_final_bias: bool = field(static=True)
    in_size: Union[int, Literal["scalar"]] = field(static=True)
    out_size: Union[int, Literal["scalar"]] = field(static=True)
    width_size: tuple[int, ...]
    depth: tuple[int, ...]
    final_activation: Callable
    linear_l: int

    def __init__(
        self,
        in_size,
        out_size,
        width_size,
        depth,
        activation=_relu,
        use_bias=True,
        use_final_bias=True,
        final_activation=_identity,
        linear_l=0,
        *,
        key,
        **kwargs,
    ):

        keys = jrandom.split(key, depth + 2)
        layers = []
        layers_l = []
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
                Linear(width_size[depth - 1], out_size, use_final_bias, key=keys[-2])
            )

        if linear_l != 0:
            for i in range(linear_l):
                layers_l.append(
                    Linear(out_size, out_size, use_bias=False, key=keys[-2])
                )

        self.layers = tuple(layers)
        self.layers_l = tuple(layers_l)
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth

        if depth != 0:
            self.activation = [
                filter_vmap(
                    filter_vmap(lambda: activation, axis_size=w), axis_size=depth
                )()
                for w in width_size
            ]
        else:
            self.activation = None
        self.final_activation = final_activation
        self.linear_l = linear_l
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @jax.named_scope("eqx.nn.MLP")
    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None, **kwargs
    ) -> Array:
        if self.depth != 0:
            for i, (layer, act) in enumerate(zip(self.layers[:-1], self.activation)):
                x = layer(x)
                layer_activation = jtu.tree_map(
                    lambda x: x[i] if is_array(x) else x, act
                )
                x = filter_vmap(lambda a, b: a(b))(layer_activation, x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        if self.linear_l != 0:
            for i, layer in enumerate(self.layers_l[:-1]):
                x = layer(x)
        return x


def plot_welding(trainor, idx):
    import matplotlib.pyplot as plt

    x = trainor.ts[0]
    dim = jnp.argmax(jnp.diff(x, axis=0) < 0) + 1
    y = trainor.ts[1]
    X, Y = jnp.meshgrid(x[:dim, 0], jnp.reshape(y, (dim, -1))[:, 0])
    m1 = trainor.y_test_o[:, idx : idx + 1]
    m2 = trainor.y_pred_mlp_test_o[:, idx : idx + 1]
    m3 = trainor.y_pred_mlp_test_o[:, idx : idx + 1]

    M1 = m1.reshape(X.shape[0], X.shape[0])
    M2 = m2.reshape(X.shape[0], X.shape[0])
    M3 = m3.reshape(X.shape[0], X.shape[0])

    fig = plt.figure()
    # Plot for M1
    ax1 = fig.add_subplot(231, projection="3d")
    ax1.plot_surface(X, Y, M1, cmap="viridis")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("true")

    # Plot for M2
    ax2 = fig.add_subplot(232, projection="3d")
    ax2.plot_surface(X, Y, M2, cmap="viridis")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("pred")

    ax3 = fig.add_subplot(233, projection="3d")
    ax3.plot_surface(X, Y, M3, cmap="viridis")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("pred-mlp")

    # Set the same color scale for both subplots
    vmin = min(M1.min(), M2.min())
    vmax = max(M1.max(), M2.max())
    ax1.set_zlim(vmin, vmax)
    ax2.set_zlim(vmin, vmax)
    ax3.set_zlim(vmin, vmax)
    ax1.view_init(elev=0, azim=0)
    ax2.view_init(elev=0, azim=0)
    ax3.view_init(elev=0, azim=0)
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax1.set_zticks([])
    ax2.set_zticks([])
    ax3.set_xticks([])
    ax3.set_zticks([])

    ax1 = fig.add_subplot(234, projection="3d")
    ax1.plot_surface(X, Y, M1, cmap="viridis")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Plot for M2
    ax2 = fig.add_subplot(235, projection="3d")
    ax2.plot_surface(X, Y, M2, cmap="viridis")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    ax3 = fig.add_subplot(236, projection="3d")
    ax3.plot_surface(X, Y, M3, cmap="viridis")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")

    # Set the same color scale for both subplots
    vmin = min(M1.min(), M2.min())
    vmax = max(M1.max(), M2.max())
    ax1.set_zlim(vmin, vmax)
    ax2.set_zlim(vmin, vmax)
    ax3.set_zlim(vmin, vmax)
    ax1.view_init(elev=90, azim=0)
    ax2.view_init(elev=90, azim=0)
    ax3.view_init(elev=90, azim=0)
    ax1.set_zticks([])
    ax2.set_zticks([])
    ax1.set_xticks([])
    ax2.set_xticks([])
    ax3.set_zticks([])
    ax3.set_xticks([])
    plt.show()


class Conv2d_(Conv2d):
    def __init__(self, *args, **kwargs):
        if "output_padding" in kwargs:
            kwargs.pop("output_padding")
        super().__init__(*args, **kwargs)


class MLCNN(Module, strict=True):
    layers: tuple[eqx.nn.Conv, ...]
    activation: Callable
    final_activation: Callable

    def __init__(
        self,
        start_dim,
        out_dim,
        stride,
        padding,
        kernel_conv,
        dilation,
        CNN_widths,
        CNNs_num,
        activation=_relu,
        final_activation=_identity,
        transpose=False,
        output_padding=None,
        *,
        key,
        kwargs_cnn={},
        **kwargs,
    ):
        """Note: if provided as lists, activations should be one less than widths.
        The last activation is specified by "final activation"."""

        CNN_widths = (
            [CNN_widths] * (CNNs_num - 1) + [out_dim]
            if not isinstance(CNN_widths, list)
            else CNN_widths
        )

        CNN_widths_b = [start_dim] + CNN_widths[:-1]
        CNN_keys = jrandom.split(key, CNNs_num)
        layers = []
        fn = Conv2d_ if not transpose else ConvTranspose2d
        for i in range(len(CNN_widths)):
            layers.append(
                fn(
                    CNN_widths_b[i],
                    CNN_widths[i],
                    kernel_size=kernel_conv,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    output_padding=output_padding,
                    key=CNN_keys[i],
                    **kwargs_cnn,
                )
            )

        self.layers = tuple(layers)

        self.activation = [
            filter_vmap(
                filter_vmap(lambda: activation, axis_size=w), axis_size=CNNs_num
            )()
            for w in CNN_widths
        ]

        self.final_activation = final_activation

    @jax.named_scope("eqx.nn.MLCNN")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        for i, (layer, act) in enumerate(zip(self.layers[:-1], self.activation[:-1])):
            x = layer(x)
            layer_activation = jtu.tree_map(lambda x: x[i] if is_array(x) else x, act)
            x = filter_vmap(lambda a, b: a(b))(layer_activation, x)
        x = self.final_activation(self.layers[-1](x))
        return x


class CNNs_with_MLP(eqx.Module, strict=True):
    """Class mainly for creating encoders with CNNs.
    The encoder is composed of multiple CNNs followed by an MLP.
    """

    layers: tuple[MLCNN, Linear]

    def __init__(
        self,
        width,
        height,
        out,
        channels=1,
        CNNs_num=4,
        width_CNNs=[32, 64, 128, 256],
        kernel_conv=3,
        stride=2,
        padding=1,
        dilation=1,
        mlp_width=None,
        mlp_depth=0,
        final_activation=_identity,
        *,
        key,
        kwargs_cnn={},
        kwargs_mlp={},
    ):
        super().__init__()

        if mlp_depth != 0:
            if mlp_width is not None:
                assert (
                    mlp_width >= out
                ), "Choose a bigger (or equal) MLP width than the latent space in the encoder."
            else:
                mlp_width = out

        assert CNNs_num == len(width_CNNs)
        key1, key2 = jax.random.split(key, 2)

        try:
            last_width = width_CNNs[-1]
        except:
            last_width = width_CNNs

        mlcnn = MLCNN(
            channels,
            last_width,
            stride,
            padding,
            kernel_conv,
            dilation,
            width_CNNs,
            CNNs_num,
            key=key1,
            final_activation=_relu,
            **kwargs_cnn,
        )
        final_Ds = mlcnn(jnp.zeros((channels, height, width))).shape[-2:]
        mlp = MLP_with_linear(
            final_Ds[0] * final_Ds[1] * last_width,
            out,
            mlp_width,
            mlp_depth,
            key=key2,
            **kwargs_mlp,
        )
        act = lambda x: final_activation(x)
        self.layers = tuple([mlcnn, mlp, act])

    def __call__(self, x, *args, **kwargs):
        x = self.layers[0](x)
        x = jnp.expand_dims(jnp.ravel(x), -1)
        x = self.layers[1](jnp.squeeze(x))
        x = self.layers[2](x)
        return x


def int_to_lst(x, len=1):
    if isinstance(x, int):
        return [x]*len
    return x


def prev_D_CNN_trans(D0, D1, pad, ker, st, dil, outpad, num, all_D0s=[], all_D1s=[]):
    pad = int_to_lst(pad, 2)
    ker = int_to_lst(ker, 2)
    st = int_to_lst(st, 2)
    dil = int_to_lst(dil, 2)
    outpad = int_to_lst(outpad, 2)

    if num == 0:
        return all_D0s, all_D1s
    
    all_D0s.append(int(jnp.ceil(D0)))
    all_D1s.append(int(jnp.ceil(D1)))

    return prev_D_CNN_trans(
        (D0 + 2 * pad[0] - dil[0] * (ker[0] - 1) - 1 - outpad[0]) / st[0] + 1,
        (D1 + 2 * pad[1] - dil[1] * (ker[1] - 1) - 1 - outpad[1]) / st[1] + 1,
        pad,
        ker,
        st,
        dil,
        outpad,
        num - 1,
    )


def find_padding_convT(D, data_dim0, ker, st, dil, outpad):

    return D


def next_CNN_trans(O0, O1, pad, ker, st, dil, outpad, num, all_D0s=[], all_D1s=[]):
    pad = int_to_lst(pad, 2)
    ker = int_to_lst(ker, 2)
    st = int_to_lst(st, 2)
    dil = int_to_lst(dil, 2)
    outpad = int_to_lst(outpad, 2)

    if num == 0:
        return all_D0s, all_D1s
    
    all_D0s.append(int(O0))
    all_D1s.append(int(O1))

    return next_CNN_trans(
        (O0 - 1) * st[0] + dil[0] * (ker[0] - 1) - 2 * pad[0] + 1 + outpad[0],
        (O1 - 1) * st[1] + dil[1] * (ker[1] - 1) - 2 * pad[1] + 1 + outpad[1],
        pad,
        ker,
        st,
        dil,
        outpad,
        num - 1,
    )


def is_convT_valid(D0, D1, data_dim0, data_dim1, pad, ker, st, dil, outpad, nums):
    all_D0s, all_D1s = next_CNN_trans(D0, D1, pad, ker, st, dil, outpad, nums, all_D0s=[], all_D1s=[])
    final_D0 = all_D0s[-1]
    final_D1 = all_D1s[-1]
    return final_D0 == data_dim0, final_D1 == data_dim1, final_D0, final_D1


class MLP_with_CNNs_trans(eqx.Module, strict=True):
    """Class mainly for creating encoders with CNNs.
    The encoder is composed of multiple CNNs followed by an MLP.
    """

    layers: tuple[MLCNN, Linear]
    start_dim: int
    first_D0: int
    first_D1: int
    out_after_mlp: int
    final_act: Callable

    def __init__(
        self,
        width,
        height,
        inp,
        channels,
        out_after_mlp=32,
        CNNs_num=2,
        width_CNNs=[64, 32],
        kernel_conv=3,
        stride=2,
        padding=1,
        dilation=1,
        output_padding=1,
        final_activation=_identity,
        mlp_width=None,
        mlp_depth=0,
        *,
        key,
        kwargs_cnn={},
        kwargs_mlp={},
    ):
        super().__init__()
        assert CNNs_num == len(width_CNNs)
        D0s, D1s = prev_D_CNN_trans(
            height,
            width,
            padding,
            kernel_conv,
            stride,
            dilation,
            output_padding,
            CNNs_num + 1,
            all_D0s=[],
            all_D1s=[],
        )

        first_D0 = D0s[-1]
        first_D1 = D1s[-1]

        _, _, final_D0, final_D1 = is_convT_valid(
            first_D0,
            first_D1,
            height,
            width,
            padding,
            kernel_conv,
            stride,
            dilation,
            output_padding,
            CNNs_num + 1,
        )


        key1, key2 = jax.random.split(key, 2)
        mlcnn = MLCNN(
            out_after_mlp,
            width_CNNs[-1],
            stride,
            padding,
            kernel_conv,
            dilation,
            width_CNNs,
            CNNs_num,
            transpose=True,
            output_padding=output_padding,
            final_activation=_relu,
            key=key1,
            **kwargs_cnn,
        )

        if mlp_depth != 0:
            if mlp_width is not None:
                assert (
                    mlp_width >= inp
                ), "Choose a bigger (or equal) MLP width than the latent space in decoder."
            else:
                mlp_width = inp

        mlp = MLP_with_linear(
            inp,
            out_after_mlp * first_D0 * first_D1,
            mlp_width,
            mlp_depth,
            key=key2,
            **kwargs_mlp,
        )

        final_conv = Conv2d(
            width_CNNs[-1],
            channels,
            kernel_size=(1 + (final_D0 - height), 1 + (final_D1 - width)),
            stride=1,
            padding=0,
            dilation=1,
            key=key2,
        )

        self.start_dim = inp
        self.first_D0 = first_D0
        self.first_D1 = first_D1
        self.final_act = doc_repr(
            lambda x: final_activation(x), "lambda x: final_activation(x)"
        )
        self.layers = tuple([mlp, mlcnn, final_conv, self.final_act])
        self.out_after_mlp = out_after_mlp

    def __call__(self, x, *args, **kwargs):
        x = self.layers[0](x)
        x = jnp.reshape(x, (self.out_after_mlp, self.first_D0, self.first_D1))
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        return x

class Sample(eqx.Module):
    sample_dim: int

    def __init__(self, sample_dim):
        self.sample_dim = sample_dim

    def __call__(self, mean, logvar, epsilon=None, ret=False, *args, **kwargs):
        epsilon = 0 if epsilon is None else epsilon
        if ret:
            return mean + jnp.exp(0.5 * logvar) * epsilon, mean, logvar
        return mean + jnp.exp(0.5 * logvar) * epsilon

    def create_epsilon(self, seed, shape):
        return jrandom.normal(jrandom.key(seed), shape)
