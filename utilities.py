from collections.abc import Callable
from typing import (
    Literal,
    Optional,
    Union,
)
from equinox.nn._linear import Linear
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

_identity = doc_repr(lambda x: x, "lambda x: x")
_relu = doc_repr(jnn.relu, "<function relu>")


def remove_keys_from_dict(d, keys):
    return {k: v for k, v in d.items() if k not in keys}


def merge_dicts(d1, d2):
    return {**d1, **d2}


def v_print(s, v):
    if v:
        print(s)


def norm_divide_return(
    ts,
    y_all,
    p_all,
    prop_train=0.8,
    pod=False,
    test_end=0,
    eps=1,
    output=None,
    norm_p=False,
    norm_data=False,
    conv=False,
):
    def norm_vec(x):
        return (x - jnp.mean(x)) / jnp.std(x)

    if norm_p:
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

    if norm_data:
        y_shift = (y_shift_old - jnp.mean(y_shift_old)) / jnp.std(y_shift_old)
        y_test = (y_test_old - jnp.mean(y_shift_old)) / jnp.std(y_shift_old)
    else:
        y_shift = y_shift_old
        y_test = y_test_old

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

    if norm_data:

        def inv_func(xx):
            return xx * jnp.std(y_shift_old) + jnp.mean(y_shift_old)

    else:

        def inv_func(xx):
            return xx

    p_vals = jnp.expand_dims(p_vals, -1) if len(p_vals.shape) == 1 else p_vals
    p_test = jnp.expand_dims(p_test, -1) if len(p_test.shape) == 1 else p_test

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

    u_now, _, _ = adaptive_TSVD(y_shift.T, eps=eps)
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
    """Function that generates the examples presented in the paper."""

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
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return norm_divide_return(ts, y_all, p_all, test_end=y_test.shape[-1])

        case "shift":
            ts = jnp.linspace(0, 2 * jnp.pi, 200)

            def sf_func(s, x):
                return jnp.sin(x - s * jnp.pi)

            p_vals = jnp.linspace(0, 1.8, 19)[:-1]
            y_shift = jax.vmap(sf_func, in_axes=[0, None])(p_vals, ts).T
            p_test = jrandom.uniform(
                jrandom.PRNGKey(0), (80,), minval=0 + 0.01, maxval=p_vals[-1] * 0.99
            )
            y_test = jax.vmap(sf_func, in_axes=[0, None])(p_test, ts).T
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return norm_divide_return(ts, y_all, p_all, test_end=y_test.shape[-1])

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
                (100,),
                minval=jnp.min(p_vals) + 0.01,
                maxval=jnp.max(p_vals) - 0.01,
            )
            y_test = jax.vmap(
                lambda y: (y - jnp.mean(y_shift_old)) / jnp.std(y_shift_old)
            )(jax.vmap(create_escal)(p_test)).T

            ts = jnp.arange(0, y_shift.shape[0], 1)
            return norm_divide_return(ts, y_shift, p_vals, test_end=y_test.shape[-1])

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
                (1000,),
                minval=0.8 * jnp.pi + 0.01,
                maxval=jnp.pi - 0.01,
            )
            p_vals_1 = jrandom.uniform(
                jrandom.PRNGKey(8),
                (1000,),
                minval=0.3 * jnp.pi + 0.01,
                maxval=jnp.pi / 2 - 0.01,
            )
            p_test = jnp.stack([p_vals_0, p_vals_1], axis=-1)
            y_test = jax.vmap(lambda p: jnp.sin(p[0] * ts) + jnp.sin(p[1] * ts))(
                p_test
            ).T
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return norm_divide_return(ts, y_all, p_all, test_end=y_test.shape[-1])

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
            return norm_divide_return(ts, y_all, p_all, test_end=y_all_1.shape[-1])

        case "mult_gausses":

            p_vals_0 = jnp.repeat(jnp.linspace(1, 3, 25), 25)
            p_vals_1 = jnp.tile(jnp.linspace(4, 6, 25), 25)
            p_vals = jnp.stack([p_vals_0, p_vals_1], axis=-1)
            p_test_0 = jrandom.uniform(
                jrandom.PRNGKey(100), (1000,), minval=1.1, maxval=2.9
            )
            p_test_1 = jrandom.uniform(
                jrandom.PRNGKey(0), (1000,), minval=4.1, maxval=5.9
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
            return norm_divide_return(ts, y_all, p_all, test_end=y_test.shape[-1])

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
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return norm_divide_return(ts, y_all, p_all, test_end=y_test.shape[-1])

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
            y_test = jax.vmap(
                lambda p, t: 1 - jnp.exp(-jnp.pi * p[0] * p[1] ** 3 / 3 * t**n),
                in_axes=[0, None],
            )(p_test, ts).T
            y_all = jnp.concatenate([y_shift, y_test], axis=-1)
            p_all = jnp.concatenate([p_vals, p_test], axis=0)
            return norm_divide_return(ts, y_all, p_all, test_end=y_test.shape[-1])

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


def adaptive_TSVD(ys, eps=0.2, prop=0.1, full_matrices=True, verbose=True, modes=None, **kwargs):
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


def dataloader(arrays, batch_size, p_vals=None, *, key):
    dataset_size = arrays[0].shape[0]
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
                yield [arr if None in arr else jnp.array(arr) for arr in arrs]
            else:
                yield [
                    [arr] if arr is None else jnp.expand_dims(jnp.array(arr), axis=0)
                    for arr in arrs
                ]
            start = end
            end = start + batch_size
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
            pdb.set_trace()
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


class MLP_dropout(Module, strict=True):
    layers: tuple[Linear, ...]
    activation: Callable
    dropout: eqx.nn.Dropout
    use_bias: bool = field(static=True)
    use_final_bias: bool = field(static=True)
    in_size: Union[int, Literal["scalar"]] = field(static=True)
    out_size: Union[int, Literal["scalar"]] = field(static=True)
    width_size: tuple[int, ...]
    depth: tuple[int, ...]
    final_activation: Callable

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
        final_activation=_identity,
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
        self.use_bias = use_bias
        self.use_final_bias = use_final_bias

    @jax.named_scope("eqx.nn.MLP")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        if self.depth != 0:
            for i, (layer, act) in enumerate(zip(self.layers[:-1], self.activation)):
                x = layer(x)
                # x = self.dropout(x, key=key)
                layer_activation = jtu.tree_map(
                    lambda x: x[i] if is_array(x) else x, act
                )
                x = filter_vmap(lambda a, b: a(b))(layer_activation, x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x


class Func(eqx.Module):
    mlp: MLP_dropout
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
        inside_activation=None,
        final_activation=None,
        post_proc_func=_identity,
        use_bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if out_size is None:
            out_size = data_size

        final_activation = (
            _identity if final_activation is None else final_activation
        )  # not used
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
            final_activation=final_activation,
            use_bias=use_bias,
            key=key,
        )
        self.post_proc_func = post_proc_func

    def __call__(self, y, k=None, train=True):
        if not train:
            return self.post_proc_func(self.mlp(y, key=k))
        else:
            return self.mlp(y, key=k)


class CNN(eqx.Module):
    layers: list

    def __init__(
        self,
        data_dim0,
        data_dim1,
        out,
        width=64,
        depth=3,
        kernel_conv=3,
        stride=1,
        padding=2,
        out_conv=3,
        dropout=0,
        *,
        key,
        **kwargs,
    ):
        key1, key2 = jax.random.split(key, 2)

        self.layers = [
            eqx.nn.Conv2d(
                1,
                out_conv,
                stride=stride,
                padding=padding,
                kernel_size=kernel_conv,
                key=key1,
            ),
            jnn.softplus,
            lambda x: jnp.expand_dims(jnp.ravel(x), -1),
            lambda x: MLP_dropout(
                out_conv * (data_dim0 + padding) * (data_dim1 + padding),
                out,
                width,
                depth,
                eqx.nn.Dropout(dropout),
                key=key2,
            )(jnp.squeeze(x)),
        ]

    def __call__(self, x, *args, **kwargs):
        x = jnp.expand_dims(x, 0)
        for layer in self.layers:
            x = layer(x)
        return x


class CNN_trans(eqx.Module):
    layers: list

    def __init__(
        self,
        data_dim0,
        data_dim1,
        out,
        width=64,
        depth=3,
        kernel_conv=3,
        stride=1,
        padding=2,
        out_conv=3,
        dropout=0,
        *,
        key,
        **kwargs,
    ):
        key1, key2 = jax.random.split(key, 2)

        width = jnp.flip(jnp.array(width)).tolist()

        self.layers = [
            lambda x: jnp.expand_dims(jnp.ravel(x), -1),
            lambda x: MLP_dropout(
                out,
                out_conv * (data_dim0 + padding) * (data_dim1 + padding),
                width,
                depth,
                eqx.nn.Dropout(dropout),
                key=key2,
            )(jnp.squeeze(x)),
            lambda x: jnp.reshape(
                x, (out_conv, data_dim0 + padding, data_dim1 + padding)
            ),
            jnn.softplus,
            eqx.nn.ConvTranspose2d(
                out_conv,
                1,
                stride=stride,
                padding=padding,
                kernel_size=kernel_conv,
                key=key1,
            ),
        ]

    def __call__(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x)
        return jnp.squeeze(x)
