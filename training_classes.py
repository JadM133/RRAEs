from abc import ABC, abstractmethod
import numpy as np
import jax
from utilities import my_vmap
import pdb
import jax.random as jr
import jax.numpy as jnp
from utilities import _identity
import equinox as eqx
import jax.tree_util as jtu
import optax
from utilities import dataloader, find_weighted_loss, adaptive_TSVD, v_print
import time

class Interpolator(ABC):

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def __call__(self, x_new, *args, **kwargs):
        pass


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

    interp_ps = my_vmap(to_map_over_test)(x_test)

    def interpolate(coords, coord0, ps):
        ds = np.abs(coords - coord0)
        vols = np.prod(ds, axis=-1)
        vols = vols / np.sum(vols)
        sorted = np.argsort(vols)

        def func_per_mode(p):
            ps_sorted = p[sorted]
            return np.sum(np.flip(np.sort(vols)) * ps_sorted)

        return jax.vmap(func_per_mode, in_axes=[-1])(ps)

    vt_test = my_vmap(interpolate)(
        x_train[interp_ps], x_test, y_train[:, interp_ps.T].T
    ).T
    return vt_test


class Objects_Interpolator_nD(Interpolator):
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

    def fit(self, x_train, y_train):
        self.model = lambda x_test: p_of_dim_n_equi_grid(y_train, x_train, x_test)

    def __call__(self, x_new, *args, **kwargs):
        return self.model(x_new)

class Trainor_class():
    def __init__(self, model_cls, interpolation_cls, **kwargs):
        self.model = model_cls(**kwargs)
        self.model_params = self.model.params
        self.interpolation = interpolation_cls(**kwargs)
        self.kwargs = kwargs
        self.fitted = False

    def fit(self, 
        input,
        output,
        output_o=None,
        loss_func=None,
        step_st=[3000, 3000],  # 000, 8000],
        lr_st=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        print_every=20,
        stagn_every=100, 
        batch_size_st=[16, 16, 16, 16, 32],
        mul_lr=None,
        mul_lr_func=None, #  lambda tree: (tree.v_vt1.v, tree.v_vt1.vt)
        regression=False,
        verbose=True,
        *,
        training_key,
        **kwargs):
        
        self.x_train = input
        self.y_train = output
        self.y_train_o = output_o

        if mul_lr is None:
            mul_lr = [1,]*len(lr_st)

        model = self.model

        if (loss_func == "Strong") or loss_func is None:
            norm_loss = lambda x1, x2: jnp.linalg.norm(x1-x2)/jnp.linalg.norm(x2)*100
            @eqx.filter_value_and_grad(has_aux=True)
            def loss_func(model, input, out, idx, key):
                pred = model(input)
                wv = jnp.array([1.0,])
                return find_weighted_loss([norm_loss(pred, out)], weight_vals=wv), (pred,)
            
        elif loss_func == "Weak":
            norm_loss = lambda x1, x2: jnp.linalg.norm(x1-x2)/jnp.linalg.norm(x2)*100
            @eqx.filter_value_and_grad(has_aux=True)
            def loss_func(model, input, out, idx, key):
                pred = model(input)
                x = model.encode(input)
                wv = jnp.array([1.0, 1.0])
                return find_weighted_loss(
                    [
                        norm_loss(pred, out),
                        jnp.linalg.norm(x - model.v_vt()[:, idx]) / jnp.linalg.norm(x) * 100,
                    ],
                    weight_vals=wv,
                ), (pred,)
    
        @eqx.filter_jit
        def make_step(input, out, model, opt_state, idx, key):
            (loss, aux), grads = loss_func(model, input, out, idx, key)
            updates, opt_state = optim.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state, aux

        v_print("Training the RRAE...", verbose)
        filter_spec = jtu.tree_map(lambda _: False, model)
        if mul_lr_func is not None:
            is_acc = eqx.tree_at(
                mul_lr_func,
                filter_spec,
                replace=(True, True),
            )
            is_not_acc = jtu.tree_map(lambda x: not x, is_acc)

        for steps, lr, batch_size, mul_l in zip(step_st, lr_st, batch_size_st, mul_lr):
            stagn_num = 0
            loss_old = jnp.inf
            t_t = 0

            if mul_lr_func is not None:
                optim = optax.chain(
                    optax.masked(optax.adabelief(lr), is_not_acc),
                    optax.masked(optax.adabelief(mul_l * lr), is_acc),
                )
            else:
                optim = optax.adabelief(lr)

            filtered_model = eqx.filter(model, eqx.is_inexact_array)
            opt_state = optim.init(filtered_model)

            keys = jr.split(training_key, steps+1)

            if (batch_size > input.shape[-1]) or batch_size == -1:
                batch_size = input.shape[-1]

            for step, (input_b, out, idx, key) in zip(
                range(steps),
                dataloader(
                    [input.T, output.T, jnp.arange(0, input.shape[-1], 1), keys[1:]],
                    batch_size,
                    key=keys[0],
                ),
            ):
                start = time.time()
                loss, model, opt_state, aux = make_step(input_b.T, out.T, model, opt_state, idx, key[0])
                end = time.time()
                t_t += end - start
                if (step % stagn_every) == 0:
                    if jnp.abs(loss_old - loss) / jnp.abs(loss_old) * 100 < 1:
                        stagn_num += 1
                        if stagn_num > 10:
                            v_print("Stagnated....", verbose)
                            break
                    loss_old = loss

                if (step % print_every) == 0 or step == steps - 1:
                    if regression:
                        pred = np.zeros_like(aux)
                        pred[aux.argmax(0), np.arange(aux.shape[-1])] = 1
                        v_print(
                            f"Step: {step}, Loss: {loss}, Computation time: {t_t}, Accuracy: {jnp.sum((pred == out.T))/pred.size*100}", verbose
                        )
                    else:
                        v_print(f"Step: {step}, Loss: {loss}, Computation time: {t_t}", verbose)
                    t_t = 0

        model = eqx.nn.inference_mode(model)
        self.model = model
        
        return model
    
    def post_process(self, y_test=None, x_test=None, p_train=None, p_test=None,  save=False, **kwargs):
        """Performs post-processing to find the relative error of the RRAE model.
        
        Parameters:
        -----------
        y_test: jnp.array
            The test data to be used for the error calculation.
        x_test: jnp.array
            The test input. If this is provided the error_test will be computed by sipmly giving
            x_test to the model. 
        p_train: jnp.array
            The training data to be used for the interpolation. If this is provided along with p_test (next),
            the error_test will be computed by interpolating the latent space of the model and then decoding it.
        p_test: jnp.array
            The test parameters for which to interpolate.
        save: bool
            If anything other than False, the model as well as the results will be saved in f"{save}".pkl
        """
        y_pred_train = self.model(self.x_train)
        error_train = jnp.linalg.norm(y_pred_train - self.y_train)/jnp.linalg.norm(self.y_train)*100
        print("Train error: ", error_train)

        if self.y_train_o is not None:
            y_pred_train_o = self.model(self.x_train, train=False)
            error_train_o = jnp.linalg.norm(y_pred_train_o - self.y_train_o)/jnp.linalg.norm(self.y_train_o)*100
            print("Train error_o: ", error_train_o)
        else:
            error_train_o = None

        if x_test is not None:
            y_pred_test = self.model(x_test)
            error_test = jnp.linalg.norm(y_pred_test - y_test)/jnp.linalg.norm(y_test)*100
            print("Test error: ", error_test)

            if self.y_train_o is not None:
                y_pred_test_o = self.model(x_test, train=False)
                error_test_o = jnp.linalg.norm(y_pred_test_o - y_test)/jnp.linalg.norm(y_test)*100
                print("Test error_o: ", error_test_o)
            else:
                error_test_o = None

        elif (p_train is not None) and (p_test is not None):
            u_vec, sing, vt = adaptive_TSVD(self.model.encode(self.x_train), full_matrices=False, verbose=False, **kwargs)
            sv = jnp.expand_dims(sing, 0)
            v = jnp.multiply(sv, u_vec)
            self.v = v
            self.vt_train = vt
            self.vt_test = self.interpolate(p_test, p_train, vt, save=save)
            latent_test = jnp.sum(jax.vmap(lambda o1, o2 : jnp.outer(o1, o2), in_axes=[-1, 0])(self.v, self.vt_test), 0)
            y_pred_test = self.model.decode(latent_test)
            error_test = jnp.linalg.norm(y_pred_test - y_test)/jnp.linalg.norm(y_test)*100
            print("Test error: ", error_test)

            if self.y_train_o is not None:
                y_pred_test_o = self.model.decode(latent_test, train=False)
                error_test_o = jnp.linalg.norm(y_pred_test_o - y_test)/jnp.linalg.norm(y_test)*100
                print("Test error_o: ", error_test_o)
            else:
                error_test_o = None
        else:
            error_test = None
            error_test_o = None
        
        return error_train, error_test, error_train_o, error_test_o
    
    def interpolate(self, x_new, x_interp, y_interp, save=False):
        if self.fitted:
            return self.interpolation(x_new)
        self.interpolation.fit(x_interp, y_interp)
        self.fitted = True
        return self.interpolation(x_new)
