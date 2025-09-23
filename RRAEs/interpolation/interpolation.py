import numpy as np
import jax
from RRAEs.utilities import np_vmap

def p_of_dim_n_equi_grid(y_train, x_train, x_test):
    """Performs interpolation over an n-dimensional space with equidistant grid points."""
    dims = x_train.shape[-1]

    def to_map_over_test(p0):
        all_idxs = np.linspace(0, x_train.shape[0] - 1, x_train.shape[0], dtype=int)
        neg_mat = x_train < p0
        pos_mat = x_train > p0
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

        def nested_loops(d_, all_is, mats):
            interp_idxs = []
            if d_ == 0:
                bool_mat = np.stack([m[:, i] for m, i in zip(mats, all_is)], axis=-1)
                p_now = x_train[np.bitwise_and.reduce(bool_mat, 1)]
                idxs = all_idxs[np.bitwise_and.reduce(bool_mat, 1)]
                id = np.array([int(i) for i in all_is])
                try:
                    return idxs[
                        np.bitwise_and.reduce(max_min(p_now, switch_1_0(id)), 1)
                    ]
                except:
                    raise ValueError(
                        "Test value does not have values arround it in every direction and this function doesn't"
                        "perform extrapolation. Either change the test or use another interpolation strategy."
                    )

            for i in range(2):
                interp_idxs.append(nested_loops(d_ - 1, all_is + [i], mats))
            return np.array(interp_idxs).flatten()

        return nested_loops(dims, [], mats)

    interp_ps = np_vmap(to_map_over_test)(x_test)

    def interpolate(coords, coord0, ps):
        ds = np.abs(coords - coord0)
        vols = np.prod(ds, axis=-1)
        vols = vols / np.sum(vols)
        sorted = np.argsort(vols)

        def func_per_mode(p):
            ps_sorted = p[sorted]
            return np.sum(np.flip(np.sort(vols)) * ps_sorted)

        return jax.vmap(func_per_mode, in_axes=[-1])(ps)

    vt_test = np_vmap(interpolate)(
        x_train[interp_ps], x_test, y_train[:, interp_ps.T].T
    ).T
    if len(vt_test.shape) == 1:
        vt_test = np.expand_dims(vt_test, 0)
    return vt_test


class Objects_Interpolator_nD:
    """Class that interpolates over an n-dimensional space with equidistant grid points.

    The data to be interpolated must be on an increasing grid (in 1D) and equdistent
    grids in every other dimension. The arrays are expected to ahve the following shapes:

    x_train: (n_samples, n_dims)
    y_train: (n_modes, n_samples) # n_modes is the number of values at every point,
                                  # these are interpolated seperately.
    x_test: (n_test_samples, n_dims)

    when called on the test data, the function will return the interpolated values of shape,
    y_test: (n_modes, n_test_samples)

    """

    def __init__(self, **kwargs):
        self.model = None
        self.fitted = False

    def fit(self, x_train, y_train):
        self.model = lambda x_test: p_of_dim_n_equi_grid(y_train, x_train, x_test)
        self.fitted = True

    def __call__(self, x_new, x_train=None, y_train=None, *args, **kwargs):
        if not self.fitted:
            if (x_train is None) or (y_train is None):
                raise ValueError(
                    "Interpolation Model is not available. You should either fit the interpolation class first,"
                    "or provide the training data when calling the interpolation class."
                )
            self.fit(x_train, y_train)
        return self.model(x_new)
