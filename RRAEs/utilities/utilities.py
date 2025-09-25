from collections.abc import Callable
from typing import (
    Literal,
    Optional,
    Union,
)
from equinox.nn._linear import Linear
from equinox.nn import Conv2d, ConvTranspose2d, Conv1d, ConvTranspose1d, Conv3d, ConvTranspose3d
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
from operator import itemgetter
import numpy as np
import dill
from tqdm import tqdm
import jax.lax as lax
from jax import custom_jvp
import jax.tree_util as jtu
import itertools
from equinox.nn import MLP

##################### ML Training/Evaluation functions ##########################

def eval_with_batches(
        x,
        batch_size,
        call_func,
        end_type="concat_and_resort",
        str=None,
        *args,
        key_idx,
        **kwargs,
    ):
        """Function to evaluate a model on a dataset using batches.

        Parameters
        ----------
        x : jnp.array
            Data to which the model will be applied.
        batch_size : int
            The batch size
        call_func : callable
            The function that calls the model, usually model.__call__
        end_type : str
            What to do with the output, can use first, sum, mean, concat,
            stack, and concat_and_resort. Note that the data is shuffled
            when batched. The last option (and defaut value) resorts the 
            outputs to give back the same order as the input.
        key_idx : int
            Seed for random key to shuffle the data before batching.

        Returns
        -------
        final_pred : jnp.array
            The predictions over the batches.
        """
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
            case "stack":
                final_pred = jnp.stack(all_preds, -1)
            case "mean":
                final_pred = sum(all_preds) / len(all_preds)
            case "sum":
                final_pred = sum(all_preds)
            case "first":
                final_pred = all_preds[0]
            case _:
                final_pred = all_preds
        return final_pred


def loss_generator(which=None, norm_loss_=None):
    """ Allows users to use different loss functions by only providing strings. """
    if norm_loss_ is None:
        norm_loss_ = lambda x1, x2: jnp.linalg.norm(x1 - x2) / jnp.linalg.norm(x2) * 100

    if (which == "default") or (which == "Vanilla"):
        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(diff_model, static_model, input, out, **kwargs):
            model = eqx.combine(diff_model, static_model)
            pred = model(input, keep_normalized=True)
            aux = {"loss": norm_loss_(pred, out)}
            return norm_loss_(pred, out), (aux, {})
        
    elif which == "RRAE":
        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(diff_model, static_model, input, out, *, k_max, **kwargs):
            model = eqx.combine(diff_model, static_model)
            pred = model(input, k_max=k_max, keep_normalized=True)
            aux = {"loss": norm_loss_(pred, out), "k_max": k_max}
            return norm_loss_(pred, out), (aux, {})
    
    elif which == "Sparse":
        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(
            diff_model, static_model, input, out, *, sparsity=0.05, beta=1.0, **kwargs
        ):
            model = eqx.combine(diff_model, static_model)
            pred = model(input, keep_normalized=True)
            lat = model.latent(input)
            sparse_term = sparsity * jnp.log(sparsity / (jnp.mean(lat) + 1e-8)) + (
                1 - sparsity
            ) * jnp.log((1 - sparsity) / (1 - jnp.mean(lat) + 1e-8))

            aux = {"loss rec": norm_loss_(pred, out), "loss sparse": sparse_term}
            return norm_loss_(pred, out) + beta * sparse_term, (aux, {})

    elif which == "nuc":
        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(
            diff_model,
            static_model,
            input,
            out,
            *,
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
            return norm_loss(pred, out) + lambda_nuc * jnp.linalg.norm(weight, "nuc"), (aux, {})

    elif which == "VRRAE":
        norm_loss_ = lambda pr, out: jnp.linalg.norm(pr-out)/jnp.linalg.norm(out)*100

        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(diff_model, static_model, input, out, idx, epsilon, k_max, beta=None, **kwargs):
            model = eqx.combine(diff_model, static_model)
            lat, means, logvars = model.latent(input, epsilon=epsilon, k_max=k_max, return_lat_dist=True)
            pred = model.decode(lat)
            wv = jnp.array([1.0, beta])
            kl_loss = jnp.sum(
                -0.5 * (1 + logvars - jnp.square(means) - jnp.exp(logvars))
            )
            loss_rec = norm_loss_(pred, out)
            aux = {
                "loss rec": loss_rec,
                "loss kl": kl_loss,
            }
            if beta is None:
                beta = lambda_fn(loss_rec, kl_loss)
            aux["beta"] = beta
            return loss_rec + beta*kl_loss, (aux, {})

    elif which == "VAE":
        norm_loss_ = lambda pr, out: jnp.linalg.norm(pr-out)/jnp.linalg.norm(out)*100

        def lambda_fn(loss, loss_c):
            return loss_c*jnp.exp(-0.1382*loss)

        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(diff_model, static_model, input, out, idx, epsilon, beta=None, **kwargs):
            model = eqx.combine(diff_model, static_model)
            lat, means, logvars = model.latent(input, epsilon=epsilon, length=input.shape[3], return_lat_dist=True)
            pred = model.decode(lat)
            kl_loss = jnp.sum(
                -0.5 * (1 + logvars - jnp.square(means) - jnp.exp(logvars))
            )
            loss_rec = norm_loss_(pred, out)
            aux = {
                "loss rec": loss_rec,
                "loss kl": kl_loss,
            }
            if beta is None:
                beta = lambda_fn(loss_rec, kl_loss)
            aux["beta"] = beta
            return loss_rec + beta*kl_loss, (aux, {})
        
    elif "Contractive":
        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(diff_model, static_model, input, out, *, beta=1.0, find_weight=None, **kwargs):
            assert find_weight is not None
            model = eqx.combine(diff_model, static_model)
            lat = model.latent(input)
            pred = model(input, keep_normalized=True)
            W = find_weight(model)
            W = find_weight(model)
            dh = lat * (1 - lat)
            dh = dh.T
            loss_contr = jnp.sum(jnp.matmul(dh**2, jnp.square(W)))
            wv = jnp.array([1.0, beta])
            aux = {"loss": norm_loss_(pred, out), "cont": loss_contr}
            aux = {"loss": norm_loss_(pred, out), "cont": loss_contr}
            return norm_loss_(pred, out) + beta * loss_contr, (aux, {})
    else:
        raise ValueError(f"{which} is an Unknown loss type")
    return loss_fun

def dataloader(arrays, batch_size, p_vals=None, once=False, *, key_idx):
    """ JAX copatible dataloader to batch data randomly and differently
    between epochs. """
    dataset_size = arrays[0].shape[0]
    arrays = [array if array is not None else [None] * dataset_size for array in arrays]
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

##################### ML synthetic data generation ########################

def divide_return(
    inp_all,
    p_all=None,
    output=None,
    prop_train=0.8,
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


def get_data(problem, folder=None, train_size=1000, test_size=10000, **kwargs):
    """Function that generates the examples presented in the paper."""

    match problem:
        case "2d_gaussian_shift_scale":
            D = 64  # Dimension of the domain
            Ntr = train_size  # Number of training samples
            Nte = test_size  # Number of test samples
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


        case "shift":
            ts = jnp.linspace(0, 2 * jnp.pi, 200)

            def sf_func(s, x):
                return jnp.sin(x - s * jnp.pi)

            p_vals = jnp.linspace(0, 1.8, 18)[:-1]  # 18
            y_shift = jax.vmap(sf_func, in_axes=[0, None])(p_vals, ts).T
            p_test = jnp.linspace(0, jnp.max(p_vals), 500)[1:-1]
            y_test = jax.vmap(sf_func, in_axes=[0, None])(p_test, ts).T
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return divide_return(y_all, p_all, test_end=y_test.shape[-1])
         
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

########################### SVD functions #################################

def _extract_diagonal(s):
    """Extract the diagonal from a batched matrix"""
    i = lax.iota("int32", min(s.shape[-2], s.shape[-1]))
    return s[..., i, i]

def _construct_diagonal(s):
    """Construct a (batched) diagonal matrix"""
    i = lax.iota("int32", s.shape[-1])
    return lax.full((*s.shape, s.shape[-1]), 0, s.dtype).at[..., i, i].set(s)

def _H(x):
    return _T(x).conj()

def _T(x):
    return lax.transpose(x, (*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))

@custom_jvp
def stable_SVD(x):
    """ An SVD wth stable gradients."""
    return jnp.linalg.svd(x, full_matrices=False)

@stable_SVD.defjvp
def _svd_jvp_rule(primals, tangents):
    """ Slight modification in the JVP rule to make gradients stable. """
    (A,) = primals
    (dA,) = tangents
    U, s, Vt = jnp.linalg.svd(A, full_matrices=False)

    s = (s / s[0] * 100 >= 1e-9) * s # Changed here

    Ut, V = _H(U), _H(Vt)
    s_dim = s[..., None, :]
    dS = Ut @ dA @ V
    ds = _extract_diagonal(dS.real)

    s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
    s_diffs_zeros = (s_diffs == 0).astype(s_diffs.dtype)
    F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
    dSS = s_dim.astype(A.dtype) * dS
    SdS = _T(s_dim.astype(A.dtype)) * dS

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
    
######################## Typical Neural Network Architecturs ##############################

class MLP_with_linear(MLP):
    layers_l: tuple[Linear, ...]
    """ Similar to an Eauinox MLP but with additional linear multiplications by matrices. 
    
    The only new_paraeter is linear_l and it is the number of matrices to use after the MLP.
    e.g. if linear_l = 2, the model wil be constituted of an MLP and two matrix multiplications.
    
    This is especially useful for IRMAE and LoRAE that use matrix multiplications in their latent
    space.
    """
    def __init__(
        self,
        *,
        in_size,
        out_size,
        width_size,
        depth,
        linear_l=0,
        
        key,
        **kwargs,
    ):
        new_key, _ = jrandom.split(key)

        layers_l = []
        if linear_l != 0:
            for _ in range(linear_l):
                layers_l.append(
                    Linear(out_size, out_size, use_bias=False, key=new_key)
                )

        self.layers_l = tuple(layers_l)

        super().__init__(in_size, out_size, width_size, depth, key=key, **kwargs)

    def __call__(self, x, *args, **kwargs):
        x = super().__call__(x, *args, **kwargs)
        if len(self.layers_l) != 0:
            for layer in self.layers_l[:-1]:
                x = layer(x)
        return x

class Conv2d_(Conv2d):
    """ For compatibility, accepting output_padding as kwarg"""
    def __init__(self, *args, **kwargs):
        if "output_padding" in kwargs:
            kwargs.pop("output_padding")
        super().__init__(*args, **kwargs)

class Conv1d_(Conv1d):
    """ For compatibility, accepting output_padding as kwarg"""
    def __init__(self, *args, **kwargs):
        if "output_padding" in kwargs:
            kwargs.pop("output_padding")
        super().__init__(*args, **kwargs)


class MLCNN(Module, strict=True):
    layers: tuple[eqx.nn.Conv, ...]
    activation: Callable
    final_activation: Callable
    """ A Multilayer CNN model. It consists of multiple conv layers.
    
    """
    def __init__(
        self,
        start_dim,
        out_dim,
        stride,
        padding,
        kernel_conv,
        dilation,
        CNN_widths,
        activation=jnn.relu,
        final_activation=lambda x: x,
        transpose=False,
        output_padding=None,
        dimension=2,
        *,
        key,
        kwargs_cnn={},
        **kwargs,
    ):
        """Note: if provided as lists, activations should be one less than widths.
        The last activation is specified by "final activation".
        """

        if isinstance(CNN_widths, list):
            CNNs_num = len(CNN_widths)
        else: 
            CNN_widths = [CNN_widths] * (CNNs_num - 1) + [out_dim]

        CNN_widths_b = [start_dim] + CNN_widths[:-1]
        CNN_keys = jrandom.split(key, CNNs_num)
        layers = []
        if dimension == 2:
            fn = Conv2d_ if not transpose else ConvTranspose2d
        elif dimension == 1:
            fn = Conv1d_ if not transpose else ConvTranspose1d

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

    def __call__(self, x):
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

    layers: tuple[MLCNN, MLP]

    def __init__(
        self,
        width,
        height,
        out,
        channels=1,
        width_CNNs=[32, 64, 128, 256],
        kernel_conv=3,
        stride=2,
        padding=1,
        dilation=1,
        mlp_width=None,
        mlp_depth=0,
        dimension=2,
        final_activation=lambda x: x,
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
            key=key1,
            dimension=dimension,
            final_activation=jnn.relu,
            **kwargs_cnn,
        )

        if dimension == 2:
            final_Ds = mlcnn(jnp.zeros((channels, height, width))).shape[-2:]
            fD = final_Ds[0] * final_Ds[1]
        elif dimension == 1:
            fD = mlcnn(jnp.zeros((channels, height))).shape[-1]

        mlp = MLP_with_linear(
            in_size=fD * last_width,
            out_size=out,
            width_size=mlp_width,
            depth=mlp_depth,
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
    d_shape: int
    out_after_mlp: int
    final_act: Callable

    def __init__(
        self,
        width,
        height,
        inp,
        channels,
        out_after_mlp=32,
        width_CNNs=[64, 32],
        kernel_conv=3,
        stride=2,
        padding=1,
        dilation=1,
        output_padding=1,
        final_activation=lambda x: x,
        mlp_width=None,
        mlp_depth=0,
        dimension=2,
        *,
        key,
        kwargs_cnn={},
        kwargs_mlp={},
    ):
        super().__init__()

        width = 10 if width is None else width #  a default value to avoid errors

        D0s, D1s = prev_D_CNN_trans(
            height,
            width,
            padding,
            kernel_conv,
            stride,
            dilation,
            output_padding,
            len(width_CNNs) + 1,
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
            len(width_CNNs) + 1,
        )
        
        if dimension == 2:
            self.d_shape = [first_D0, first_D1]
        elif dimension == 1:
            self.d_shape = [first_D0]

        key1, key2 = jax.random.split(key, 2)
        mlcnn = MLCNN(
            out_after_mlp,
            width_CNNs[-1],
            stride,
            padding,
            kernel_conv,
            dilation,
            width_CNNs,
            dimension=dimension,
            transpose=True,
            output_padding=output_padding,
            final_activation=jnn.relu,
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
            in_size=inp,
            out_size=out_after_mlp * int(np.prod(self.d_shape)),
            width_size=mlp_width,
            depth=mlp_depth,
            key=key2,
            **kwargs_mlp,
        )

        if dimension == 2:
            final_conv = Conv2d(
                width_CNNs[-1],
                channels,
                kernel_size=(1 + (final_D0 - height), 1 + (final_D1 - width)),
                stride=1,
                padding=0,
                dilation=1,
                key=key2,
            )
        elif dimension == 1:
            final_conv = Conv1d(
                width_CNNs[-1],
                channels,
                kernel_size=(1 + (final_D0 - height)),
                stride=1,
                padding=0,
                dilation=1,
                key=key2,
            )

        self.start_dim = inp
        
        self.final_act = doc_repr(
            lambda x: final_activation(x), "lambda x: final_activation(x)"
        )
        self.layers = tuple([mlp, mlcnn, final_conv, self.final_act])
        self.out_after_mlp = out_after_mlp

    def __call__(self, x, *args, **kwargs):
        x = self.layers[0](x)
        x = jnp.reshape(x, (self.out_after_mlp, *self.d_shape))
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        return x

class Conv3d_(Conv3d):
    def __init__(self, *args, **kwargs):
        if "output_padding" in kwargs:
            kwargs.pop("output_padding")
        super().__init__(*args, **kwargs)
   

class MLCNN3D(Module, strict=True):
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
        activation=jnn.relu,
        final_activation=lambda x: x,
        transpose=False,
        output_padding=None,
        *,
        key,
        kwargs_cnn={},
        **kwargs,
    ):
        """Note: if provided as lists, activations should be one less than widths.
        The last activation is specified by "final activation"."""

        if isinstance(CNN_widths, list):
            CNNs_num = len(CNN_widths)
        else:
            CNN_widths = [CNN_widths] * (CNNs_num - 1) + [out_dim]

        CNN_widths_b = [start_dim] + CNN_widths[:-1]
        CNN_keys = jrandom.split(key, CNNs_num)
        layers = []
        fn = Conv3d_ if not transpose else ConvTranspose3d
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

    @jax.named_scope("eqx.nn.MLCNN3D")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        for i, (layer, act) in enumerate(zip(self.layers[:-1], self.activation[:-1])):
            x = layer(x)
            layer_activation = jtu.tree_map(lambda x: x[i] if is_array(x) else x, act)
            x = filter_vmap(lambda a, b: a(b))(layer_activation, x)
        x = self.final_activation(self.layers[-1](x))
        return x


class CNN3D_with_MLP(eqx.Module, strict=True):
    """Class mainly for creating encoders with CNNs.
    The encoder is composed of multiple CNNs followed by an MLP.
    """

    layers: tuple[MLCNN3D, Linear]

    def __init__(
        self,
        depth,
        height,                
        width,
        out,
        channels=1,
        width_CNNs=[32, 64, 128, 256],
        kernel_conv=3,
        stride=2,
        padding=1,
        dilation=1,
        mlp_width=None,
        mlp_depth=0,
        final_activation=lambda x: x,
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

        key1, key2 = jax.random.split(key, 2)

        try:
            last_width = width_CNNs[-1]
        except:
            last_width = width_CNNs

        mlcnn3d = MLCNN3D(
            channels,
            last_width,
            stride,
            padding,
            kernel_conv,
            dilation,
            width_CNNs,
            key=key1,
            final_activation=jnn.relu,
            **kwargs_cnn,
        )
        final_Ds = mlcnn3d(jnp.zeros((channels, depth, height, width))).shape[-3:]
        mlp = MLP_with_linear(
            in_size=final_Ds[0] * final_Ds[1] * final_Ds[2] *last_width,
            out_size=out,
            width_size=mlp_width,
            depth=mlp_depth,
            key=key2,
            **kwargs_mlp,
        )
        act = lambda x: final_activation(x)
        self.layers = tuple([mlcnn3d, mlp, act])

    def __call__(self, x, *args, **kwargs):
        x = self.layers[0](x)
        x = jnp.expand_dims(jnp.ravel(x), -1)
        x = self.layers[1](jnp.squeeze(x))
        x = self.layers[2](x)
        return x

def prev_D_CNN3D_trans(D0, D1, D2, pad, ker, st, dil, outpad, num, all_D0s=[], all_D1s=[], all_D2s=[]):
    pad = int_to_lst(pad, 3)
    ker = int_to_lst(ker, 3)
    st = int_to_lst(st, 3)
    dil = int_to_lst(dil, 3)
    outpad = int_to_lst(outpad, 3)

    if num == 0:
        return all_D0s, all_D1s , all_D2s
    
    all_D0s.append(int(jnp.ceil(D0)))
    all_D1s.append(int(jnp.ceil(D1)))
    all_D2s.append(int(jnp.ceil(D2)))    

    return prev_D_CNN3D_trans(
        (D0 + 2 * pad[0] - dil[0] * (ker[0] - 1) - 1 - outpad[0]) / st[0] + 1,
        (D1 + 2 * pad[1] - dil[1] * (ker[1] - 1) - 1 - outpad[1]) / st[1] + 1,
        (D2 + 2 * pad[2] - dil[2] * (ker[2] - 1) - 1 - outpad[2]) / st[2] + 1,
        pad,
        ker,
        st,
        dil,
        outpad,
        num - 1,
    )

def find_padding_conv3dT(D, data_dim0, ker, st, dil, outpad):
    return D

def next_CNN3D_trans(O0, O1, O2, pad, ker, st, dil, outpad, num, all_D0s=[], all_D1s=[], all_D2s=[]):
    pad = int_to_lst(pad, 3)
    ker = int_to_lst(ker, 3)
    st = int_to_lst(st, 3)
    dil = int_to_lst(dil, 3)
    outpad = int_to_lst(outpad, 3)

    if num == 0:
        return all_D0s, all_D1s, all_D2s
    
    all_D0s.append(int(O0))
    all_D1s.append(int(O1))
    all_D2s.append(int(O2))

    return next_CNN3D_trans(
        (O0 - 1) * st[0] + dil[0] * (ker[0] - 1) - 2 * pad[0] + 1 + outpad[0],
        (O1 - 1) * st[1] + dil[1] * (ker[1] - 1) - 2 * pad[1] + 1 + outpad[1],
        (O2 - 1) * st[2] + dil[2] * (ker[2] - 1) - 2 * pad[2] + 1 + outpad[2],
        pad,
        ker,
        st,
        dil,
        outpad,
        num - 1,
    )

def is_conv3dT_valid(D0, D1, D2, data_dim0, data_dim1, data_dim2, pad, ker, st, dil, outpad, nums):
    all_D0s, all_D1s, all_D2s = next_CNN3D_trans(D0, D1, D2, pad, ker, st, dil, outpad, nums, all_D0s=[], all_D1s=[], all_D2s=[])
    final_D0 = all_D0s[-1]
    final_D1 = all_D1s[-1]
    final_D2 = all_D2s[-1]
    return final_D0 == data_dim0, final_D1 == data_dim1, final_D2 ==data_dim2, final_D0, final_D1, final_D2

class MLP_with_CNN3D_trans(eqx.Module, strict=True):
    """Class mainly for creating encoders with CNNs.
    The encoder is composed of multiple CNNs followed by an MLP.
    """

    layers: tuple[MLCNN3D, Linear]
    start_dim: int
    first_D0: int
    first_D1: int
    first_D2: int
    out_after_mlp: int
    final_act: Callable

    def __init__(
        self,
        depth, 
        height,               
        width,
        inp,
        channels,
        out_after_mlp=32,
        width_CNNs=[64, 32],
        kernel_conv=3,
        stride=2,
        padding=1,
        dilation=1,
        output_padding=1,
        final_activation=lambda x: x,
        mlp_width=None,
        mlp_depth=0,
        *,
        key,
        kwargs_cnn={},
        kwargs_mlp={},
    ):
		
        super().__init__()

        D0s, D1s, D2s = prev_D_CNN3D_trans(
            depth,
            height,
            width,
            padding,
            kernel_conv,
            stride,
            dilation,
            output_padding,
            len(width_CNNs) + 1,
            all_D0s=[],
            all_D1s=[],
            all_D2s=[],            
        )

        first_D0 = D0s[-1]
        first_D1 = D1s[-1]
        first_D2 = D2s[-1]

        _, _, _, final_D0, final_D1, final_D2 = is_conv3dT_valid(
            first_D0,
            first_D1,
            first_D2,
            depth,            
            height,
            width,
            padding,
            kernel_conv,
            stride,
            dilation,
            output_padding,
            len(width_CNNs) + 1,
        )


        key1, key2 = jax.random.split(key, 2)
        mlcnn3d = MLCNN3D(
            out_after_mlp,
            width_CNNs[-1],
            stride,
            padding,
            kernel_conv,
            dilation,
            width_CNNs,
            len(width_CNNs),
            transpose=True,
            output_padding=output_padding,
            final_activation=jnn.relu,
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
            in_size=inp,
            out_size=out_after_mlp * first_D0 * first_D1 * first_D2,
            width_size=mlp_width,
            depth=mlp_depth,
            key=key2,
            **kwargs_mlp,
        )

        final_conv = Conv3d(
            width_CNNs[-1],
            channels,
            kernel_size=(1 + (final_D0 - depth), 1 + (final_D1 - height), 1 + (final_D2 - width)),
            stride=1,
            padding=0,
            dilation=1,
            key=key2,
        )

        self.start_dim = inp
        self.first_D0 = first_D0
        self.first_D1 = first_D1
        self.first_D2 = first_D2
        self.final_act = doc_repr(
            lambda x: final_activation(x), "lambda x: final_activation(x)"
        )
        self.layers = tuple([mlp, mlcnn3d, final_conv, self.final_act])
        self.out_after_mlp = out_after_mlp

    def __call__(self, x, *args, **kwargs):
        print(x.shape)
        x = self.layers[0](x)
        x = jnp.reshape(x, (self.out_after_mlp, self.first_D0, self.first_D1, self.first_D2))
        x = self.layers[1](x)
        x = self.layers[2](x)
        x = self.layers[3](x)
        return x

################# Other Useful functions ################################

def int_to_lst(x, len=1):
    """ Integer to list """
    if isinstance(x, int):
        return [x]*len
    return x

def tree_map(f, tree, *rest, is_leaf=None):
    """ Map over tree after filtering equinox style. """
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
    
def remove_keys_from_dict(d, keys):
    return {k: v for k, v in d.items() if k not in keys}

def merge_dicts(d1, d2):
    return {**d1, **d2}

def v_print(s, v, f=False):
    if v:
        print(s, flush=f)

def countList(lst1, lst2):
    return [sub[item] for item in range(len(lst2)) for sub in [lst1, lst2]]

def np_vmap(func, to_array=True):
    """ Similar to JAX's vmap but in numpy. 
    Slow but useful if we want to use the same syntax."""
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

class Sample(eqx.Module):
    sample_dim: int

    """ Class used to allow random sampling using JAX.
    
    Funny trick to allow seed to change since otherwise we end up
    with the same noise epsilon for every forward pass.
    """
    def __init__(self, sample_dim):
        self.sample_dim = sample_dim

    def __call__(self, mean, logvar, epsilon=None, ret=False, *args, **kwargs):
        epsilon = 0 if epsilon is None else epsilon
        if ret:
            return mean + jnp.exp(0.5 * logvar) * epsilon, mean, logvar
        return mean + jnp.exp(0.5 * logvar) * epsilon

    def create_epsilon(self, seed, shape):
        return jrandom.normal(jrandom.key(seed), shape)