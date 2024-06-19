from abc import ABC, abstractmethod
import numpy as np
import jax
from RRAEs.utilities import my_vmap
import pdb
import jax.random as jrandom
import jax.numpy as jnp
import equinox as eqx
import jax.tree_util as jtu
import optax
from RRAEs.utilities import (
    dataloader,
    find_weighted_loss,
    adaptive_TSVD,
    v_print,
    remove_keys_from_dict,
    merge_dicts,
    MLP_dropout,
)
import os
import time
import dill
import shutil
import matplotlib.pyplot as plt


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
    if len(vt_test.shape) == 1:
        vt_test = np.expand_dims(vt_test, 0)
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


class Trainor_class:
    def __init__(
        self,
        model_cls=None,
        interpolation_cls=None,
        folder=None,
        file=None,
        ts=None,
        **kwargs,
    ):
        if model_cls is not None:
            self.model = model_cls(**kwargs)
        if interpolation_cls is not None:
            self.interpolation = interpolation_cls(**kwargs)
        self.all_kwargs = {
            **kwargs,
            "model_cls": model_cls,
            "interpolation_cls": interpolation_cls,
        }

        self.folder = folder
        if folder is not None:
            if not os.path.exists(folder):
                os.makedirs(folder)
        self.ts = ts
        self.file = file
        self.fitted = False

    def set_model(self, model):
        self.model = model

    def set_interpolation(self, interpolation):
        self.interpolation = interpolation
        self.all_kwargs["interpolation_cls"] = interpolation

    def fit(
        self,
        input,
        output,
        loss_func=None,
        step_st=[3000, 3000],  # 000, 8000],
        lr_st=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        print_every=20,
        stagn_every=None,
        batch_size_st=[16, 16, 16, 16, 32],
        mul_lr=None,
        mul_lr_func=None,  #  lambda tree: (tree.v_vt1.v, tree.v_vt1.vt)
        regression=False,
        verbose=True,
        loss_kwargs={},
        *,
        training_key,
        **kwargs,
    ):

        if (self.all_kwargs["model_cls"] is None) or (
            self.all_kwargs["interpolation_cls"] is None
        ):
            raise ValueError(
                "Model and Interpolation classes must be provided for fitting."
            )

        training_params = {
            "loss_func": loss_func,
            "step_st": step_st,
            "lr_st": lr_st,
            "print_every": print_every,
            "stagn_every": stagn_every,
            "batch_size_st": batch_size_st,
            "mul_lr": mul_lr,
            "mul_lr_func": mul_lr_func,
            "regression": regression,
            "verbose": verbose,
            "loss_kwargs": loss_kwargs,
            "training_key": training_key,
            "kwargs": kwargs,
        }
        self.all_kwargs = {**self.all_kwargs, **training_params, **kwargs}
        self.x_train = input
        self.y_train = output

        model = self.model

        if mul_lr is None:
            mul_lr = [
                1,
            ] * len(lr_st)

        norm_loss_ = lambda x1, x2: jnp.linalg.norm(x1 - x2) / jnp.linalg.norm(x2) * 100


        if (loss_func == "Strong") or (loss_func is None) or (loss_func == "Vanilla"):

            @eqx.filter_value_and_grad(has_aux=True)
            def loss_fun(model, input, out, idx, **kwargs):
                pred = model(input)
                wv = jnp.array([1.0])
                return find_weighted_loss([norm_loss_(pred, out)], weight_vals=wv), (
                    pred,
                )

        elif loss_func == "Weak":

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
                    jnp.linalg.norm(x - model.v_vt()[:, idx])
                    / jnp.linalg.norm(x)
                    * 100,
                )

        elif loss_func == "nuc":

            @eqx.filter_value_and_grad(has_aux=True)
            def loss_fun(model, input, out, idx, lambda_nuc, norm_loss=None, **kwargs):
                if norm_loss is None:
                    norm_loss = norm_loss_
                pred = model(input)
                wv = jnp.array([1.0, lambda_nuc])
                if "mlp" in model._encode.__dict__.keys():
                    weight = model._encode.mlp.layers_l[0].weight
                else:
                    idx = [isinstance(val, MLP_dropout) for val in model._encode.layers]
                    idx = idx.index(max(idx))
                    weight = model._encode.layers[idx].layers_l[0].weight
                return find_weighted_loss(
                    [
                        norm_loss(pred, out),
                        jnp.linalg.norm(weight, "nuc"),
                    ],
                    weight_vals=wv,
                ), (pred,)

        elif loss_func == "var":

            @eqx.filter_value_and_grad(has_aux=True)
            def loss_fun(model, input, out, idx, rd, **kwargs):
                pred = model(input, eps_rand=rd)
                y = model.encode(input)
                means = y[: int(y.shape[0] / 2)]
                stds = y[int(y.shape[0] / 2) :]
                wv = jnp.array([1.0, 0.01])
                kl_loss = -0.5 * (1 + stds - jnp.square(means) - jnp.exp(stds))
                kl_loss = jnp.mean(jnp.sum(kl_loss, axis=1))
                return find_weighted_loss(
                    [norm_loss_(pred, out), kl_loss], weight_vals=wv
                ), (norm_loss_(pred, out), kl_loss)

        elif loss_func == "bce":

            @eqx.filter_value_and_grad(has_aux=True)
            def loss_fun(model, input, out, idx, rd, **kwargs):
                # my_lf = lambda x1, x2: jnp.mean(
                #     (1 - x2) * jnp.square(x1 + 1 * jnp.ones_like(x1)) / 2
                #     + (1 + x2) * jnp.square(x1 - jnp.ones_like(x1)) / 2
                # )
                lf = lambda x1, x2:  jnp.mean(jnp.sum(-x2*jnp.log(x1) - (1-x2)*jnp.log(1-x1), 0))
                pred = model(input)
                wv = jnp.array(
                    [
                        1.0,
                    ]
                )
                return find_weighted_loss([lf(pred, out)], weight_vals=wv), (pred,)

        @eqx.filter_jit
        def make_step(model, input, out, opt_state, idx, **loss_kwargs):
            (loss, aux), grads = loss_fun(model, input, out, idx, **loss_kwargs)
            updates, opt_state = optim.update(grads, opt_state)
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state, aux

        v_print("Training the RRAE...", verbose)

        if mul_lr_func is not None:
            filter_spec = jtu.tree_map(lambda _: False, model)
            is_acc = eqx.tree_at(
                mul_lr_func,
                filter_spec,
                replace=(True,) * len(mul_lr_func(model)),
            )
            is_not_acc = jtu.tree_map(lambda x: not x, is_acc)

        t_all = 0

        try:
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
                try:
                    opt_state = optim.init(filtered_model)
                except ValueError:
                    raise ValueError(
                        "Optax has a bug! Send a message to Jad so he can fix it to you..."
                    )

                if (batch_size > input.shape[-1]) or batch_size == -1:
                    batch_size = input.shape[-1]

                if loss_func == "var":
                    all_rds = jrandom.normal(
                        jrandom.PRNGKey(1253),
                        shape=(steps, int(model.latent(input).shape[0]), input.shape[-1]),
                    )
                else:
                    all_rds = [None] * steps

                for step, (input_b, out, idx) in zip(
                    range(steps),
                    dataloader(
                        [input.T, output.T, jnp.arange(0, input.shape[-1], 1)],
                        batch_size,
                        key=training_key,
                    ),
                ):
                    start = time.time()
                    loss_kwargs["rd"] = all_rds[step]
                    loss, model, opt_state, aux = make_step(
                        model, input_b.T, out.T, opt_state, idx, **loss_kwargs
                    )
                    end = time.time()
                    t_t += end - start

                    if stagn_every is not None:
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
                                f"Step: {step}, Loss: {loss}, Computation time: {t_t}, Accuracy: {jnp.sum((pred == out.T))/pred.size*100}",
                                verbose,
                            )
                        else:
                            # to_acc =  (aux[0] > 0.5)*1 # (aux[0] > 0.5) * 2 - 1  # 
                            # accuracy = jnp.sum(to_acc == out.T) / to_acc.size * 100
                            # print("Accuracy: ", accuracy)

                            if len(aux) == 2:
                                v_print(
                                    f"Step: {step}, Loss: {loss}, Computation time: {t_t}, loss1: {aux[0]}, loss2: {aux[1]}",
                                    verbose,
                                )
                            else:
                                v_print(
                                    f"Step: {step}, Loss: {loss}, Computation time: {t_t}",
                                    verbose,
                                )
                        t_all += t_t
                        t_t = 0
                    if jnp.isnan(loss):
                        print("Loss is nan, stopping training...")
                        break
        except:
            pass
            pdb.set_trace()

        self.y_pred_train = self.model(input, batch=batch_size_st[0])
        final_acc = jnp.sum((self.y_pred_train > 0.5)*1 == output) / self.y_pred_train.size * 100
        print(f"Final accuracy: {final_acc}")
        model = eqx.nn.inference_mode(model)
        self.model = model
        self.t_all = t_all
        return model

    def plot_results(self, ts=None, ts_o=None, filename=None):
        if filename is None:
            filename = os.path.join(self.folder, self.file)
        if not hasattr(self, "error_train"):
            raise ValueError(
                "Model must be post_processed before plotting."
                "Try running post_process() first."
            )

        if ts is None and not hasattr(self, "ts"):
            raise ValueError("Time steps must be provided for plotting.")
        if ts is None:
            ts = self.ts
        else:
            self.ts = ts
        if ts_o is None:
            ts_o = self.ts_o
        else:
            self.ts_o = ts_o

        plt.plot(ts, self.y_train, color="blue", label=r"$X$")
        plt.plot(ts, self.y_pred_train, color="red", label=r"$\tilde{X}$")
        plt.title("Predictions over Train")
        plt.savefig(f"{filename}_train.pdf")
        plt.clf()

        if self.y_train_o is not None:
            plt.plot(ts_o, self.y_train_o, color="blue", label=r"$X_o$")
            plt.plot(ts_o, self.y_pred_train_o, color="red", label=r"$\tilde{X}_o$")
            plt.title("Predictions over original Train")
            plt.savefig(f"{filename}_train_original.pdf")
            plt.clf()

        if hasattr(self, "y_pred_test"):
            plt.plot(ts, self.y_test, color="blue", label=r"$X$")
            plt.plot(ts, self.y_pred_test, color="red", label=r"$\tilde{X}$")
            plt.title("Predictions over Test")
            plt.savefig(f"{filename}_test.pdf")
            plt.clf()

            if self.y_test_o is not None:
                plt.plot(ts_o, self.y_test_o, color="blue", label=r"$X_o$")
                plt.plot(ts_o, self.y_pred_test_o, color="red", label=r"$\tilde{X}_o$")
                plt.title("Predictions over original Test")
                plt.savefig(f"{filename}_test_original.pdf")
                plt.clf()

        if (
            (self.p_train is not None)
            and (self.p_test is not None)
            and (self.vt_test is not None)
        ):
            for i, (tr, te) in enumerate(zip(self.vt_train, self.vt_test)):
                if self.vt_train.shape[0] == 1:
                    plt.scatter(self.p_test, te, color="red", label="Test")
                    plt.scatter(self.p_train, tr, color="blue", label="Train")
                    plt.title("Interpolated cofficients")
                    plt.savefig(f"{filename}_coeffs_mode_{i}.pdf")
                    plt.clf()
                else:
                    pass

    def post_process(
        self,
        y_train_o=None,
        y_test=None,
        y_test_o=None,
        x_test=None,
        p_train=None,
        p_test=None,
        inv_func=None,
        save=False,
        modes="all",
        interp=None,
        verbose=False,
        batch=False,
        **kwargs,
    ):
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
        if batch:
            y_pred_train = self.y_pred_train
        else:
            y_pred_train = self.model(self.x_train)
            self.y_pred_train = y_pred_train
        error_train = (
            jnp.linalg.norm(y_pred_train - self.y_train)
            / jnp.linalg.norm(self.y_train)
            * 100
        )
        print("Train error: ", error_train)
        self.error_train = error_train
        self.p_train = p_train
        self.p_test = p_test
        self.y_train_o = y_train_o
        self.y_test_o = y_test_o
        self.y_test = y_test

        if (y_train_o is not None) and (inv_func is not None):
            y_pred_train_o = inv_func(y_pred_train)
            self.inv_func = inv_func
            error_train_o = (
                jnp.linalg.norm(y_pred_train_o - self.y_train_o)
                / jnp.linalg.norm(self.y_train_o)
                * 100
            )
            self.y_pred_train_o = y_pred_train_o
            print("Train error_o: ", error_train_o)
        else:
            self.inv_func = lambda x: x
            error_train_o = None

        self.error_train_o = error_train_o
        latent_train = self.model.latent(self.x_train)
        u_vec, sing, vt = adaptive_TSVD(
            latent_train, full_matrices=False, verbose=verbose, modes=modes, **kwargs
        )
        sv = jnp.expand_dims(sing, 0)
        v = jnp.multiply(sv, u_vec)
        self.v = v
        self.vt_train = vt
        # plt.scatter(p_train, vt)
        # plt.show()
        if x_test is not None:
            y_pred_test = self.model(x_test)
            error_test = (
                jnp.linalg.norm(y_pred_test - y_test) / jnp.linalg.norm(y_test) * 100
            )
            print("Test error: ", error_test)

            if self.y_train_o is not None:
                y_pred_test_o = self.model(x_test, train=False)
                error_test_o = (
                    jnp.linalg.norm(y_pred_test_o - y_test_o)
                    / jnp.linalg.norm(y_test_o)
                    * 100
                )
                print("Test error_o: ", error_test_o)
            else:
                error_test_o = None

        elif interp:

            self.latent_train = latent_train

            if modes == "all":
                latent_test = self.interpolate(p_test, p_train, latent_train, save=save)
                self.vt_test = None
            else:
                self.vt_test = self.interpolate(p_test, p_train, vt, save=save)
                latent_test = jnp.sum(
                    jax.vmap(lambda o1, o2: jnp.outer(o1, o2), in_axes=[-1, 0])(
                        self.v, self.vt_test
                    ),
                    0,
                )
            self.latent_test = latent_test

            y_pred_test = self.model.decode(latent_test)
            error_test = (
                jnp.linalg.norm(y_pred_test - y_test) / jnp.linalg.norm(y_test) * 100
            )
            print("Test error: ", error_test)
            if self.y_test_o is not None:
                y_pred_test_o = inv_func(self.model.decode(latent_test))
                self.y_pred_test_o = y_pred_test_o
                error_test_o = (
                    jnp.linalg.norm(y_pred_test_o - y_test_o)
                    / jnp.linalg.norm(y_test)
                    * 100
                )
                print("Test error_o: ", error_test_o)
            else:
                y_pred_test_o = None
                error_test_o = None
        else:
            error_test = None
            error_test_o = None
            y_pred_test = None
            y_pred_test_o = None

        self.y_pred_test = y_pred_test
        self.y_pred_test_o = y_pred_test_o
        self.error_test = error_test
        self.error_test_o = error_test_o
        print("Total computation time: ", self.t_all)
        return error_train, error_test, error_train_o, error_test_o

    def sing_vals(self):
        svs = jnp.linalg.svd(self.model.latent(self.x_train), full_matrices=False)[1]
        plt.plot(svs[:40] / svs[0], marker="o")
        plt.ylim(0, 0.4)
        plt.show()
        return svs

    def interpolate(self, x_new, x_interp, y_interp, save=False):
        if self.fitted:
            return self.interpolation(x_new)
        self.interpolation.fit(x_interp, y_interp)
        self.fitted = True
        return self.interpolation(x_new)

    def save(self, filename=None, erase=True, **kwargs):
        """Saves the"""
        if self.all_kwargs["model_cls"] is None:
            raise ValueError("Model class must be provided for saving.")

        if self.all_kwargs["interpolation_cls"] is None:
            raise ValueError("Interpolation class must be provided for saving.")

        if filename is None:
            filename = os.path.join(self.folder, self.file)
            if erase:
                shutil.rmtree(self.folder)
                os.makedirs(self.folder)

        with open(f"{filename}.pkl", "wb") as f:
            dill.dump(self.all_kwargs, f)
            eqx.tree_serialise_leaves(f, self.model)
            dill.dump(
                merge_dicts(
                    remove_keys_from_dict(
                        self.__dict__, ("model", "interpolation", "all_kwargs")
                    ),
                    kwargs,
                ),
                f,
            )

    def load(self, filename):
        with open(f"{filename}.pkl", "rb") as f:
            self.all_kwargs = dill.load(f)
            self.model_cls = self.all_kwargs["model_cls"]
            self.interpolation_cls = self.all_kwargs["interpolation_cls"]
            self.model = self.model_cls(**self.all_kwargs)
            self.model = eqx.tree_deserialise_leaves(f, self.model)
            self.interpolation = self.interpolation_cls(**self.all_kwargs)
            attributes = dill.load(f)
            for key in attributes:
                setattr(self, key, attributes[key])
            self.fitted = True
