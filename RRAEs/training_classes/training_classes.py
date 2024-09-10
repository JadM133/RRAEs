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
    v_print,
    remove_keys_from_dict,
    merge_dicts,
)
from RRAEs.interpolation import Objects_Interpolator_nD
from RRAEs.AE_classes import BaseClass, CNN_Autoencoder
from RRAEs.norm import Norm
import os
import time
import dill
import shutil


class Trainor_class:
    def __init__(
        self,
        in_train=None,
        model_cls=None,
        map_axis=None,
        folder=None,
        file=None,
        ts=None,
        out_train=None,
        norm_in="None",
        norm_out="None",
        **kwargs,
    ):
        if model_cls is not None:
            self.model = Norm(
                BaseClass(model_cls(**kwargs), map_axis=map_axis),
                in_train=in_train,
                out_train=out_train,
                norm_in=norm_in,
                norm_out=norm_out,
            )

        self.all_kwargs = {
            "kwargs": kwargs,
            "in_train": in_train,
            "out_train": out_train,
            "norm_in": norm_in,
            "norm_out": norm_out,
            "map_axis": map_axis,
            "model_cls": model_cls,
        }

        self.folder = folder
        if folder is not None:
            if not os.path.exists(folder):
                os.makedirs(folder)
        self.ts = ts
        self.file = file
        self.fitted = False

    def fit(
        self,
        input,
        output,
        loss_func=None,
        step_st=[3000, 3000],  # 000, 8000],
        lr_st=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        print_every=20,
        batch_size_st=[16, 16, 16, 16, 32],
        mul_lr=None,
        mul_lr_func=None,  #  lambda tree: (tree.v_vt1.v, tree.v_vt1.vt)
        regression=False,
        verbose=True,
        loss_kwargs={},
        *,
        training_key,
    ):
        self.x_train = input
        self.y_train = output
        output = self.model.norm_out(output)

        training_params = {
            "loss_func": loss_func,
            "step_st": step_st,
            "lr_st": lr_st,
            "print_every": print_every,
            "batch_size_st": batch_size_st,
            "mul_lr": mul_lr,
            "mul_lr_func": mul_lr_func,
            "regression": regression,
            "verbose": verbose,
            "loss_kwargs": loss_kwargs,
            "training_key": training_key,
        }
        self.all_kwargs = {**self.all_kwargs, **training_params}
        model = self.model

        if mul_lr is None:
            mul_lr = [1] * len(lr_st)

        norm_loss_ = lambda x1, x2: jnp.linalg.norm(x1 - x2) / jnp.linalg.norm(x2) * 100

        if (loss_func == "Strong") or (loss_func is None) or (loss_func == "Vanilla"):

            @eqx.filter_value_and_grad(has_aux=True)
            def loss_fun(model, input, out, idx, **kwargs):
                pred = model(input, inv_norm_out=False)
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
                if isinstance(model, CNN_Autoencoder):
                    weight = model.encode.layers[1].weight
                else:
                    weight = model.encode.layers_l[0].weight
                return find_weighted_loss(
                    [
                        norm_loss(pred, out),
                        jnp.linalg.norm(weight, "nuc"),
                    ],
                    weight_vals=wv,
                ), (pred,)

        elif loss_func == "var":

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

        # try:
        counter = 0
        for steps, lr, batch_size, mul_l in zip(step_st, lr_st, batch_size_st, mul_lr):
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

            for step, (input_b, out, idx) in zip(
                range(steps),
                dataloader(
                    [input.T, output.T, jnp.arange(0, input.shape[-1], 1)],
                    batch_size,
                    key=training_key,
                ),
            ):
                start = time.perf_counter()

                if loss_func == "var":
                    epsilon = model._sample.create_epsilon(counter, input_b.shape[0])
                    loss_kwargs["epsilon"] = epsilon
                loss, model, opt_state, aux = make_step(
                    model,
                    input_b.T,
                    out.T,
                    opt_state,
                    idx,
                    **loss_kwargs,
                )
                counter += 1
                end = time.perf_counter()
                t_t += end - start

                if (step % print_every) == 0 or step == steps - 1:
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
        # except (Exception, KeyboardInterrupt) as e:
        #     print(e)
        #     pass

        model = eqx.nn.inference_mode(model)
        self.model = model
        self.batch_size = batch_size
        self.y_pred_train_o = self.model.eval_with_batches(
            input, batch_size, call_func=self.model, key=training_key
        )
        self.t_all = t_all
        return model

    def post_process(
        self,
        x_train_o,
        y_train_o,
        x_test_o=None,
        y_test_o=None,
        batch_size=None,
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
        batch_size = self.batch_size if batch_size is None else batch_size
        self.latent_train = self.model.eval_with_batches(
            x_train_o, batch_size, call_func=self.model.latent, key=jrandom.key(0)
        )

        self.x_train_o = x_train_o
        self.y_train_o = y_train_o
        self.error_train_o = (
            jnp.linalg.norm(self.y_pred_train_o - self.y_train_o)
            / jnp.linalg.norm(self.y_train_o)
            * 100
        )
        print("Train error on original output: ", self.error_train_o)

        self.y_pred_train = self.model.norm_out(self.y_pred_train_o)
        self.y_train = self.model.norm_out(self.y_train_o)
        self.error_train = (
            jnp.linalg.norm(self.y_pred_train - self.y_train)
            / jnp.linalg.norm(self.y_train)
            * 100
        )
        print("Train error on normalized output: ", self.error_train)

        if x_test_o is not None:
            self.y_pred_test_o = self.model.eval_with_batches(
                x_test_o, batch_size, call_func=self.model, key=jrandom.key(0)
            )
            self.y_test_o = y_test_o
            self.error_test_o = (
                jnp.linalg.norm(self.y_pred_test_o - self.y_test_o)
                / jnp.linalg.norm(self.y_test_o)
                * 100
            )
            print("Test error on original output: ", self.error_test_o)

            self.y_test = self.model.norm_out(self.y_test_o)
            self.y_pred_test = self.model.norm_out(self.y_pred_test_o)
            self.error_test = (
                jnp.linalg.norm(self.y_pred_test - self.y_test)
                / jnp.linalg.norm(self.y_test)
                * 100
            )
            print("Test error on normalized output: ", self.error_test)

            self.latent_test = self.model.eval_with_batches(
                x_test_o, batch_size, call_func=self.model.latent, key=jrandom.key(0)
            )
        else:
            self.error_test = None
            self.error_test_o = None
            self.y_pred_test = None
            self.y_pred_test_o = None

        print("Total training time: ", self.t_all)
        return self.error_train, self.error_test, self.error_train_o, self.error_test_o

    def AE_interpolate(self, p_train, p_test, y_test_o=None, batch_size=None):
        """Interpolates the latent space of the model and then decodes it to find the output."""

        self.y_test_o = y_test_o if y_test_o is not None else self.y_test_o
        self.y_test = self.model.norm_out(self.y_test_o)

        try:
            self.p_train = p_train
            self.p_test = p_test

            batch_size = self.batch_size if batch_size is None else batch_size

            interpolation = Objects_Interpolator_nD()
            self.latent_test_interp = interpolation(
                self.p_test, self.p_train, self.latent_train
            )

            self.y_pred_interp_test_o = self.model.decode(self.latent_test_interp)
            self.error_interp_test_o = (
                jnp.linalg.norm(self.y_pred_interp_test_o - self.y_test_o)
                / jnp.linalg.norm(self.y_test_o)
                * 100
            )
            print(
                "Test (interpolation) error over original output: ",
                self.error_interp_test_o,
            )

            self.y_pred_interp_test = self.model.norm_out(self.y_pred_interp_test_o)
            self.error_interp_test = (
                jnp.linalg.norm(self.y_pred_interp_test - self.y_test)
                / jnp.linalg.norm(self.y_test)
                * 100
            )
            print(
                "Test (interpolation) error over normalized output: ",
                self.error_interp_test,
            )
            return self.error_interp_test, self.error_interp_test_o
        except AttributeError:
            raise AttributeError(
                "You should first call the post_process method before calling the AE_interpolate method."
            )

    def save(self, filename=None, erase=True, **kwargs):
        """Saves the trainor class."""
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
                    remove_keys_from_dict(self.__dict__, ("model", "all_kwargs")),
                    kwargs,
                ),
                f,
            )
        print(f"Model saved in {filename}.pkl")

    def load(self, filename, erase=False):
        with open(f"{filename}.pkl", "rb") as f:
            self.all_kwargs = dill.load(f)
            self.model_cls = self.all_kwargs["model_cls"]
            kwargs = self.all_kwargs["kwargs"]
            self.map_axis = self.all_kwargs["map_axis"]
            self.in_train = self.all_kwargs["in_train"]
            self.out_train = self.all_kwargs["out_train"]
            self.norm_in = self.all_kwargs["norm_in"]
            self.norm_out = self.all_kwargs["norm_out"]
            self.model = Norm(
                BaseClass(self.model_cls(**kwargs), map_axis=self.map_axis),
                in_train=self.in_train,
                norm_in=self.norm_in,
                out_train=self.out_train,
                norm_out=self.norm_out,
            )
            self.model = eqx.tree_deserialise_leaves(f, self.model)
            attributes = dill.load(f)
            for key in attributes:
                setattr(self, key, attributes[key])
            self.fitted = True
        if erase:
            os.remove(f"{filename}.pkl")
