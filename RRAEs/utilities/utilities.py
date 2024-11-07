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


_identity = doc_repr(lambda x: x, "lambda x: x")
_relu = doc_repr(jnn.relu, "<function relu>")


def loss_generator(which=None, norm_loss_=None):
    if norm_loss_ is None:
        norm_loss_ = lambda x1, x2: jnp.linalg.norm(x1 - x2) / jnp.linalg.norm(x2) * 100

    if (which == "Strong") or (which == "default") or (which == "Vanilla"):

        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(model, input, out, idx, **kwargs):
            pred = model(input, inv_norm_out=False)
            wv = jnp.array([1.0])
            return find_weighted_loss([norm_loss_(pred, out)], weight_vals=wv), (pred,)

    elif which == "Weak":

        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(model, input, out, idx, norm_loss=None, **kwargs):
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
            model,
            input,
            out,
            idx,
            lambda_nuc,
            norm_loss=None,
            find_layer=None,
            **kwargs,
        ):
            if norm_loss is None:
                norm_loss = norm_loss_
            pred = model(input)
            wv = jnp.array([1.0, lambda_nuc])

            if find_layer is None:
                weight = model.encode.layers_l[0].weight  # 1 for CNN
            else:
                weight = find_layer(model)

            return find_weighted_loss(
                [
                    norm_loss(pred, out),
                    jnp.linalg.norm(weight, "nuc"),
                ],
                weight_vals=wv,
            ), (pred,)

    elif which == "var":

        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(model, input, out, idx, epsilon, **kwargs):
            pred = model(input, epsilon=epsilon)
            _, means, logvars = model.latent(input, epsilon=epsilon, ret=True)
            wv = jnp.array([1.0, 0.1])
            kl_loss = jnp.sum(
                -0.5 * (1 + logvars - jnp.square(means) - jnp.exp(logvars))
            )
            return find_weighted_loss(
                [norm_loss_(pred, out), kl_loss], weight_vals=wv
            ), (
                norm_loss_(pred, out),
                kl_loss,
            )

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


def v_print(s, v):
    if v:
        print(s)


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
    post_func_out=lambda x: x,
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
            post_func_out,
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
        post_func_out,
        args,
    )


def get_data(problem, folder=None, google=True, **kwargs):
    """Function that generates the examples presented in the paper."""

    match problem:
        case "CelebA":
            print("GOT INSIDE DATA")
            data_res = 160
            import os
            from PIL import Image
            import numpy as np
            
            if os.path.exists(f"celeba_data_{data_res}.npy"):
                print("Loading data from file")
                data = np.load(f"celeba_data_{data_res}.npy")
            else:
                print("Loading data and processing...")
                data = np.load("celeba_data.npy")
                pdb.set_trace()
                celeb_transform = lambda im: jnp.astype(jax.image.resize(
                            jnp.array(im, dtype=jnp.uint8), (data_res, data_res, 3), method="bilinear"), dtype=jnp.uint8
                        )
                all_data = []
                for i in tqdm(range(data.shape[0] // 100 + 1)):
                    if i == data.shape[0] // 100:
                        all_data.append(celeb_transform(data[i * 100 :]))
                    else:
                        all_data.append(
                            celeb_transform(data[i * 100 : (i + 1) * 100])
                        )

                final_data = jnp.array((np.concatenate(all_data, axis=0)))
                np.save(final_data, f"celeba_data_{data_res}.npy")

            data = jnp.moveaxis(data, 2, 0)
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
            p_test = jrandom.uniform(
                jrandom.PRNGKey(0),
                (300,),
                minval=p_vals[0] * 1.00001,
                maxval=p_vals[-1] * 0.99999,
            )
            y_test = jax.vmap(func, in_axes=[0, None])(p_test, ts).T
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return divide_return(y_all, p_all, test_end=y_test.shape[-1])

        case "shift":
            ts = jnp.linspace(0, 2 * jnp.pi, 200)

            def sf_func(s, x):
                return jnp.sin(x - s * jnp.pi)

            p_vals = jnp.linspace(0, 1.8, 18)[:-1]  # 18
            y_shift = jax.vmap(sf_func, in_axes=[0, None])(p_vals, ts).T
            p_test = jrandom.uniform(
                jrandom.PRNGKey(0),
                (80,),
                minval=p_vals[0] * 1.00001,
                maxval=p_vals[-1] * 0.99999,
            )
            y_test = jax.vmap(sf_func, in_axes=[0, None])(p_test, ts).T
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
            print(y_test.shape)
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

        case "mnist_":
            import torchvision

            normalise_data = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5,), (0.5,)),
                ]
            )
            train_dataset = torchvision.datasets.MNIST(
                "MNIST",
                train=True,
                download=True,
                transform=normalise_data,
            )
            test_dataset = torchvision.datasets.MNIST(
                "MNIST",
                train=False,
                download=True,
                transform=normalise_data,
            )
            x_train = my_vmap(lambda x: x[0])(train_dataset).T
            x_test = my_vmap(lambda x: x[0])(test_dataset).T
            x_all = jnp.squeeze(jnp.concatenate([x_train, x_test], axis=-1))

            if "mlp" in kwargs.keys():
                if kwargs["mlp"]:
                    y_train = my_vmap(lambda x: x[1])(train_dataset)
                    y_test = my_vmap(lambda x: x[1])(test_dataset)
                    idx = np.arange(0, y_train.shape[0], 1)
                    y_now_tr = np.zeros((y_train.shape[0], jnp.max(y_train) + 1))
                    all_idx = np.stack([idx, y_train])
                    y_now_tr[all_idx[0, :], all_idx[1, :]] = 1

                    idx = np.arange(0, y_test.shape[0], 1)
                    y_now_t = np.zeros((y_test.shape[0], jnp.max(y_test) + 1))
                    all_idx = np.stack([idx, y_test])
                    y_now_t[all_idx[0, :], all_idx[1, :]] = 1

                    return (
                        jnp.array(y_now_tr),
                        jnp.squeeze(x_test),
                        jnp.array(y_now_t),
                    )  # [45000:]
            x_all = jnp.expand_dims(x_all, 0)  # [..., 45000:]
            return divide_return(x_all, None, test_end=x_test.shape[-1])
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


def dataloader(arrays, batch_size, p_vals=None, once=False, *, key):
    dataset_size = arrays[0].shape[0]
    arrays = [array if array is not None else [None] * dataset_size for array in arrays]
    # assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    kk = 0

    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 0
        end = batch_size
        while end <= dataset_size:
            batch_perm = perm[start:end]
            arrs = tuple(
                itemgetter(*batch_perm)(array) for array in arrays
            )  # Works for lists and arrays
            if batch_size != 1:
                yield [None if None in arr else jnp.array(arr) for arr in arrs]
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
    activation_l: Callable
    use_bias: bool = field(static=True)
    use_final_bias: bool = field(static=True)
    in_size: Union[int, Literal["scalar"]] = field(static=True)
    out_size: Union[int, Literal["scalar"]] = field(static=True)
    width_size: tuple[int, ...]
    depth: tuple[int, ...]
    final_activation: Callable
    final_activation_l: Callable

    def __init__(
        self,
        in_size,
        out_size,
        width_size,
        depth,
        activation=_relu,
        activation_l=_identity,
        use_bias=True,
        use_final_bias=True,
        final_activation=_identity,
        final_activation_l=_identity,
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
            width_size_l = [width_size[-1]] * (linear_l - 1)
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
            self.activation_l = [
                filter_vmap(
                    filter_vmap(lambda: _identity, axis_size=w), axis_size=depth
                )()
                for w in width_size_l
            ]
        else:
            self.activation = None
        self.final_activation = final_activation
        self.final_activation_l = final_activation_l
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
        if self.depth != 0:
            for i, (layer, act) in enumerate(
                zip(self.layers_l[:-1], self.activation_l)
            ):
                x = layer(x)
                layer_activation = jtu.tree_map(
                    lambda x: x[i] if is_array(x) else x, act
                )
                x = filter_vmap(lambda a, b: a(b))(layer_activation, x)
            x = self.final_activation_l(x)
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


class CNNs_with_linear(eqx.Module, strict=True):
    """Class mainly for creating encoders with CNNs.
    The encoder is composed of multiple CNNs followed by an MLP.
    """

    layers: tuple[MLCNN, MLP_with_linear]

    def __init__(
        self,
        data_dim0,
        out,
        channels=1,
        CNNs_num=4,
        CNN_widths=[32, 64, 128],
        kernel_conv=3,
        stride=2,
        padding=1,
        dilation=1,
        final_activation=_identity,
        *,
        key,
        kwargs_cnn={},
        **kwargs,
    ):
        super().__init__()

        key1, key2 = jax.random.split(key, 2)

        try:
            last_width = CNN_widths[-1]
        except:
            last_width = CNN_widths

        mlcnn = MLCNN(
            channels,
            last_width,
            stride,
            padding,
            kernel_conv,
            dilation,
            CNN_widths,
            CNNs_num,
            key=key1,
            final_activation=_relu,
            **kwargs_cnn,
        )
        final_D = mlcnn(jnp.zeros((channels, data_dim0, data_dim0))).shape[-1]
        linear = Linear((final_D) ** 2 * last_width, out, key=key2)
        act = lambda x: final_activation(x)
        self.layers = tuple([mlcnn, linear, act])

    def __call__(self, x, *args, **kwargs):
        x = self.layers[0](x)
        x = jnp.expand_dims(jnp.ravel(x), -1)
        x = self.layers[1](jnp.squeeze(x))
        x = self.layers[2](x)
        return x


# def next_D_CNN(D, pad, ker, st, dil, num, all_Ds=[]):
#     if num == 0:
#         return all_Ds
#     all_Ds.append(int(jnp.floor(D)))
#     return next_D_CNN((D + 2 * pad - dil*(ker-1)) / st + 1, pad, ker, st, dil, num - 1)


def prev_D_CNN_trans(D, pad, ker, st, dil, outpad, num, all_Ds=[]):
    if num == 0:
        return all_Ds
    all_Ds.append(int(jnp.ceil(D)))
    return prev_D_CNN_trans(
        (D + 2 * pad - dil * (ker - 1) - 1 - outpad) / st + 1,
        pad,
        ker,
        st,
        dil,
        outpad,
        num - 1,
    )


def find_padding_convT(D, data_dim0, ker, st, dil, outpad):

    return D


def next_CNN_trans(O, pad, ker, st, dil, outpad, num, all_Ds=[]):
    if num == 0:
        return all_Ds
    all_Ds.append(int(O))
    return next_CNN_trans(
        (O - 1) * st + dil * (ker - 1) - 2 * pad + 1 + outpad,
        pad,
        ker,
        st,
        dil,
        outpad,
        num - 1,
    )


def is_convT_valid(D, data_dim0, pad, ker, st, dil, outpad, nums):
    final_D = next_CNN_trans(D, pad, ker, st, dil, outpad, nums, all_Ds=[])[-1]
    return final_D == data_dim0, final_D


class Linear_with_CNNs_trans(eqx.Module, strict=True):
    """Class mainly for creating encoders with CNNs.
    The encoder is composed of multiple CNNs followed by an MLP.
    """

    layers_: tuple[MLCNN, MLP_with_linear]
    start_dim: int
    first_D: int
    out_after_mlp: int

    def __init__(
        self,
        data_dim0,
        inp,
        out_after_mlp=32,
        CNNs_num=3,
        width_CNNs=[128, 64, 32],
        kernel_conv=3,
        stride=2,
        padding=1,
        dilation=1,
        output_padding=1,
        final_activation=_identity,
        *,
        key,
        kwargs_cnn={},
        **kwargs,
    ):
        super().__init__()

        first_D = prev_D_CNN_trans(
            data_dim0,
            padding,
            kernel_conv,
            stride,
            dilation,
            output_padding,
            CNNs_num + 1,
            all_Ds=[],
        )[-1]

        valid, final_D = is_convT_valid(
            first_D,
            data_dim0,
            padding,
            kernel_conv,
            stride,
            dilation,
            output_padding,
            CNNs_num + 1,
        )

        key1, key2 = jax.random.split(key, 2)
        mlcnn_ = MLCNN(
            out_after_mlp,
            1,
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

        linear = Linear(inp, out_after_mlp * first_D**2, key=key2)
        final_conv = Conv2d(
            width_CNNs[-1],
            1,
            kernel_size=1 + (final_D - data_dim0),
            stride=1,
            padding=0,
            dilation=1,
            key=key2,
        )

        self.start_dim = inp
        self.first_D = first_D
        act = lambda x: final_activation(x)
        self.layers_ = tuple([linear, mlcnn_, final_conv, act])
        self.out_after_mlp = out_after_mlp

    def __call__(self, x, *args, **kwargs):
        x = self.layers_[0](x)
        x = jnp.reshape(x, (self.out_after_mlp, self.first_D, self.first_D))
        x = self.layers_[1](x)
        x = self.layers_[2](x)
        x = self.layers_[3](x)
        return x


class Sample(eqx.Module):
    sample_dim: int

    def __init__(self, sample_dim):
        self.sample_dim = sample_dim

    def __call__(self, x, epsilon=None, ret=False, *args, **kwargs):
        mean = x[..., : self.sample_dim]
        logvar = x[..., self.sample_dim :]
        epsilon = 0 if epsilon is None else epsilon
        if ret:
            return mean + jnp.exp(0.5 * logvar) * epsilon, mean, logvar
        return mean + jnp.exp(0.5 * logvar) * epsilon

    def create_epsilon(self, seed, batch_size):
        return jrandom.normal(jrandom.key(seed), (self.sample_dim, batch_size))
