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
    remove_keys_from_dict,
    merge_dicts,
    loss_generator,
    tree_map,
)
import warnings
from RRAEs.utilities import v_print
from RRAEs.interpolation import Objects_Interpolator_nD
from RRAEs.AE_classes import BaseClass
from RRAEs.norm import Norm
import os
import time
import dill
import shutil
import copy
from functools import partial
from RRAEs.trackers import (
    Null_Tracker,
    RRAE_Null_Tracker,
    RRAE_pars_Tracker,
    RRAE_gen_Tracker,
)


class Circular_list:
    """
        Creates a list of fixed size.
        Adds elements in a circular manner
    """
    def __init__(self, size):
        self.size = size
        self.buffer = [0.0] * size  
        self.index = 0  

    def add(self, value):
        self.buffer[self.index] = value  
        self.index = (self.index + 1) % self.size  
        
    def __iter__(self):
        for value in self.buffer:
            yield value



class Trainor_class:
    def __init__(
        self,
        in_train=None,
        model_cls=None,
        map_axis=None,
        folder=None,
        file=None,
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
            params_in = self.model.params_in
            params_out = self.model.params_out
        else:
            params_in = None
            params_out = None

        self.all_kwargs = {
            "kwargs": kwargs,
            "params_in": params_in,
            "params_out": params_out,
            "norm_in": norm_in,
            "norm_out": norm_out,
            "map_axis": map_axis,
            "model_cls": model_cls,
        }

        self.folder = folder
        if folder is not None:
            if not os.path.exists(folder):
                os.makedirs(folder)
        self.file = file

    def train(self):
        pass

    def eval(self):
        pass

    def fit(
        self,
        input,
        output,
        loss_type="default",  # should be string to use pre defined functions
        loss=None,  # a function loss(pred, true) to differentiate in the model
        step_st=[3000, 3000],  # 000, 8000],
        lr_st=[1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
        print_every=jnp.nan,
        save_every=jnp.nan,
        batch_size_st=[16, 16, 16, 16, 32],
        regression=False,
        verbose=True,
        loss_kwargs={},
        flush=False,
        pre_func_inp=lambda x: x,
        pre_func_out=lambda x: x,
        fix_comp=lambda _: (),
        tracker=Null_Tracker(),
        stagn_window=20,
        optimizer=optax.adabelief,
        *,
        training_key,
    ):
        from RRAEs.utilities import v_print

        if flush:
            v_print = partial(v_print, f=True)
        else:
            v_print = partial(v_print, f=False)

        training_params = {
            "loss": loss,
            "step_st": step_st,
            "lr_st": lr_st,
            "print_every": print_every,
            "batch_size_st": batch_size_st,
            "regression": regression,
            "verbose": verbose,
            "loss_kwargs": loss_kwargs,
            "training_key": training_key,
        }

        self.all_kwargs = merge_dicts(self.all_kwargs, training_params) # Append dicts
        
        model = self.model    # Create alias for model

        fn = lambda x: x if fn is None else fn

        # Process loss function
        if callable(loss_type):
            loss_fun = loss_type
        else:
            loss_fun = loss_generator(loss_type, loss)

        # Make step funciton
        @eqx.filter_jit
        def make_step(model, input, out, opt_state, idx, **loss_kwargs):
            diff_model, static_model = eqx.partition(model, filter_spec) # Split model into differential and static portions
            
            # Compute (loss, auxiliar vars) and gradient
            (loss, aux), grads = loss_fun(
                                              diff_model, 
                                              static_model, 
                                              input, 
                                              out, 
                                              idx, 
                                              **loss_kwargs
                                          )
            
            # Perform back-propagation
            updates, opt_state = optim.update(grads, opt_state)
            diff_model = eqx.apply_updates(diff_model, updates)
            
            # Recombine differential and static portions
            model = eqx.combine(diff_model, static_model)
            
            return loss, model, opt_state, aux

        # Create filter for splitting the model into differential and static portions
        filtered_filter_spec = tree_map(lambda _: True, model)
        new_filt = jtu.tree_map(lambda _: False, fix_comp(filtered_filter_spec))
        filter_spec = jtu.tree_map(lambda _: True, model)
        filter_spec = eqx.tree_at(fix_comp, filter_spec, replace=new_filt)

        diff_model, _ = eqx.partition(model, filter_spec)
        filtered_model = eqx.filter(diff_model, eqx.is_inexact_array)

        # Loop variables
        t_all = 0.0   # Total time
        avg_loss = jnp.inf
        training_num = jrandom.randint(training_key, (1,), 0, 1000)[0]       
        
        # Window to store averages
        store_window = min(stagn_window, sum(step_st))
        prev_losses = Circular_list(store_window)       
        
        # Initialize tracker
        track_params = tracker.init()
        
        
        # Outler Loop
        for steps, lr, batch_size in zip(step_st, lr_st, batch_size_st):
            try:
                t_t = 0.0                                       # Zero time
                optim = optimizer(lr)                           # Create optimizer
                opt_state = optim.init(filtered_model)          # Initialize optimizer

                if (batch_size > input.shape[-1]) or batch_size == -1:
                    print(f"Setting batch size to: {input.shape[-1]}")
                    batch_size = input.shape[-1]

                # Inner loop (batch)
                for step, (input_b, out_b, idx_b) in \
                zip(
                    range(steps),
                    dataloader(
                                [
                                    input.T, 
                                    output.T, 
                                    jnp.arange(0, input.shape[-1], 1)
                                ],
                                batch_size,
                                key_idx=training_num,
                                ),
                   ):
                    start_time = time.perf_counter()             # Start time
                    
                    out_b = self.model.norm_out(pre_func_out(out_b))    # Pre-process batch out values
                    input_b = pre_func_inp(input_b)              # Pre-process batch input values 

                    step_kwargs = merge_dicts(loss_kwargs, track_params)

                    # Compute loss
                    loss, model, opt_state, aux = make_step(
                                                                model,
                                                                input_b.T,
                                                                out_b.T,
                                                                opt_state,
                                                                idx_b,
                                                                **step_kwargs,
                                                            )

                    # Add loss to list (maybe store info for plotting?)
                    prev_losses.add(loss.item())
                    
                    if step > stagn_window:
                        avg_loss = sum(prev_losses) / stagn_window
                        
                    track_params = tracker(loss, avg_loss, track_params)

                    dt = time.perf_counter() - start_time  # Execution time
                    t_t += dt                   # Batch execution time
                    t_all += dt                 # Total execution time  

                    if (step % print_every) == 0 or step == steps - 1:
                        t_t = 0.0               # Reset Batch execution time
                        
                        message = ", ".join([f"{k}: {v}" for k, v in aux.items()])
                        
                        print(
                                f"Step: {step}, "
                                + message
                                + ", "
                                + f"Step time: {round(dt, 4)}, "
                                + f"Total elapsed time: {round(t_all, 4)}"
                             ) 
                        
                        
                    if ((step % save_every) == 0) or jnp.isnan(loss):
                        if jnp.isnan(loss):
                            raise ValueError("Loss is nan, stopping training...")
                            
                        self.model = model
                        
                        orig = (
                                    f"checkpoint_{step}"
                                    if not jnp.isnan(loss)
                                    else "checkpoint_bf_nan"
                               )
                        
                        checkpoint_filename = f"{orig}_0.pkl"
                        
                        if os.path.exists(checkpoint_filename):
                            i = 1
                            new_filename = f"{orig}_{i}.pkl"
                            while os.path.exists(new_filename):
                                i += 1
                                new_filename = f"{orig}_{i}.pkl"
                            checkpoint_filename = new_filename
                        self.save(checkpoint_filename)
            except KeyboardInterrupt:
                pass

        model = eqx.nn.inference_mode(model)
        
        self.model = model
        self.batch_size = batch_size
        self.t_all = t_all
        return model, track_params

    def evaluate_no_preds(
        self,
        x_train_o,
        y_train_o,
        x_test_o=None,
        y_test_o=None,
        batch_size=None,
        call_func=None,
        **kwargs,
    ):
        """Performs post-processing to find the relative error of the RRAE model."""
        # TODO: Same as evaluate but without finding preds for large data
        raise NotImplementedError("This method is not implemented yet.")

    def evaluate(
        self,
        x_train_o=None,
        y_train_o=None,
        x_test_o=None,
        y_test_o=None,
        batch_size=None,
        pre_func_inp=lambda x: x,
        pre_func_out=lambda x: x,
        call_func=None,
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
        call_func = (
            (lambda x: self.model(pre_func_inp(x))) if call_func is None else call_func
        )
        if x_train_o is not None:
            y_train_o = pre_func_out(y_train_o)
            assert (
                hasattr(self, "batch_size") or batch_size is not None
            ), "You should either provide a batch_size or fit the model first."

            batch_size = self.batch_size if batch_size is None else batch_size
            y_pred_train_o = self.model.eval_with_batches(
                x_train_o,
                batch_size,
                call_func=call_func,
                str="Finding train predictions...",
                key_idx=0,
            )

            self.error_train_o = (
                jnp.linalg.norm(y_pred_train_o - y_train_o)
                / jnp.linalg.norm(y_train_o)
                * 100
            )
            print("Train error on original output: ", self.error_train_o)

            y_pred_train = self.model.norm_out(y_pred_train_o)
            y_train = self.model.norm_out(y_train_o)
            self.error_train = (
                jnp.linalg.norm(y_pred_train - y_train) / jnp.linalg.norm(y_train) * 100
            )
            print("Train error on normalized output: ", self.error_train)

        if x_test_o is not None:
            y_test_o = pre_func_out(y_test_o)
            y_pred_test_o = self.model.eval_with_batches(
                x_test_o,
                batch_size,
                call_func=call_func,
                key_idx=0,
            )
            self.error_test_o = (
                jnp.linalg.norm(y_pred_test_o - y_test_o)
                / jnp.linalg.norm(y_test_o)
                * 100
            )
            print("Test error on original output: ", self.error_test_o)

            y_test = self.model.norm_out(y_test_o)
            y_pred_test = self.model.norm_out(y_pred_test_o)
            self.error_test = (
                jnp.linalg.norm(y_pred_test - y_test) / jnp.linalg.norm(y_test) * 100
            )
            print("Test error on normalized output: ", self.error_test)

        else:
            self.error_test = None
            self.error_test_o = None
            y_pred_test_o = None
            y_pred_test = None

        print("Total training time: ", self.t_all)
        return {
            "error_train": self.error_train,
            "error_test": self.error_test,
            "error_train_o": self.error_train_o,
            "error_test_o": self.error_test_o,
            "y_pred_train_o": y_pred_train_o,
            "y_pred_test_o": y_pred_test_o,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
        }

    def AE_interpolate(
        self,
        p_train,
        p_test,
        x_train_o,
        y_test_o,
        batch_size=None,
        latent_func=None,
        decode_func=None,
        norm_out_func=None,
    ):
        """Interpolates the latent space of the model and then decodes it to find the output."""
        batch_size = self.batch_size if batch_size is None else batch_size

        if latent_func is None:
            call_func = lambda x: self.model.latent(x)
        else:
            call_func = latent_func

        latent_train = self.model.eval_with_batches(
            x_train_o,
            batch_size,
            call_func=call_func,
            str="Finding train latent space used for interpolation...",
            key_idx=0,
        )

        interpolation = Objects_Interpolator_nD()
        latent_test_interp = interpolation(p_test, p_train, latent_train)

        if decode_func is None:
            call_func = lambda x: self.model.decode(x)
        else:
            call_func = decode_func

        y_pred_interp_test_o = self.model.eval_with_batches(
            latent_test_interp,
            batch_size,
            call_func=call_func,
            str="Decoding interpolated latent space ...",
            key_idx=0,
        )

        self.error_interp_test_o = (
            jnp.linalg.norm(y_pred_interp_test_o - y_test_o)
            / jnp.linalg.norm(y_test_o)
            * 100
        )
        print(
            "Test (interpolation) error over original output: ",
            self.error_interp_test_o,
        )

        if norm_out_func is None:
            call_func = lambda x: self.model.norm_out(x)
        else:
            call_func = norm_out_func

        y_pred_interp_test = self.model.eval_with_batches(
            y_pred_interp_test_o,
            batch_size,
            call_func=call_func,
            str="Finding Normalized pred of interpolated latent space ...",
            key_idx=0,
        )

        y_test = self.model.eval_with_batches(
            y_test_o,
            batch_size,
            call_func=call_func,
            str="Finding Normalized output of interpolated latent space ...",
            key_idx=0,
        )
        self.error_interp_test = (
            jnp.linalg.norm(y_pred_interp_test - y_test) / jnp.linalg.norm(y_test) * 100
        )
        print(
            "Test (interpolation) error over normalized output: ",
            self.error_interp_test,
        )
        return {
            "error_interp_test": self.error_interp_test,
            "error_interp_test_o": self.error_interp_test_o,
            "y_pred_interp_test_o": y_pred_interp_test_o,
            "y_pred_interp_test": y_pred_interp_test,
        }

    def save(self, filename=None, erase=False, **kwargs):
        """Saves the trainor class."""
        if filename is None:
            if (self.folder is None) or (self.file is None):
                raise ValueError("You should provide a filename to save")
            filename = os.path.join(self.folder, self.file)
            if erase:
                shutil.rmtree(self.folder)
                os.makedirs(self.folder)
        else:
            if not os.path.exists(filename):
                with open(filename, "a") as temp_file:
                    pass
                os.utime(filename, None)
        attr = merge_dicts(
            remove_keys_from_dict(self.__dict__, ("model", "all_kwargs")),
            kwargs,
        )
        with open(filename, "wb") as f:
            dill.dump(self.all_kwargs, f)
            eqx.tree_serialise_leaves(f, self.model)
            dill.dump(attr, f)
        print(f"Model saved in {filename}")

    def load(self, filename, erase=False, **fn_kwargs):
        """NOTE: fn_kwargs defines the functions of the model
        (e.g. final_activation, inner activation), if
        needed to be saved/loaded on different devices/OS."""
        with open(filename, "rb") as f:
            self.all_kwargs = dill.load(f)
            self.model_cls = self.all_kwargs["model_cls"]
            kwargs = self.all_kwargs["kwargs"]
            self.map_axis = self.all_kwargs["map_axis"]
            self.params_in = self.all_kwargs["params_in"]
            self.params_out = self.all_kwargs["params_out"]
            self.norm_in = self.all_kwargs["norm_in"]
            self.norm_out = self.all_kwargs["norm_out"]
            kwargs.update(fn_kwargs)
            self.model = Norm(
                BaseClass(self.model_cls(**kwargs), map_axis=self.map_axis),
                norm_in=self.norm_in,
                norm_out=self.norm_out,
                params_in=self.params_in,
                params_out=self.params_out,
            )
            self.model = eqx.tree_deserialise_leaves(f, self.model)
            attributes = dill.load(f)
            for key in attributes:
                setattr(self, key, attributes[key])
        if erase:
            os.remove(filename)


class RRAE_Trainor_class(Trainor_class):
    def __init__(self, *args, adapt=False, k_max=None, adap_type="None", **kwargs):
        self.k_init = k_max
        self.adap_type = adap_type
        kwargs["k_max"] = k_max
        super().__init__(*args, **kwargs)
        self.adapt = adapt

    def fit(self, *args, training_key, **kwargs):
        if self.adap_type == "pars":
            default_tracker = RRAE_pars_Tracker(k_init=self.k_init)
        elif self.adap_type == "gen":
            if self.k_init is None:
                warnings.warn(
                    "k_max can not be None when using gen adaptive scheme, choose a big initial k_max to start with."
                )
            default_tracker = RRAE_gen_Tracker(k_init=self.k_init)
        elif self.adap_type == "None":
            if self.k_init is None:
                warnings.warn(
                    "k_max can not be None when using fixed scheme, choose a fixed k_max to use."
                )
            default_tracker = RRAE_Null_Tracker(k_init=self.k_init)

        print("Training RRAEs...")

        if "training_kwargs" in kwargs:
            training_kwargs = kwargs["training_kwargs"]
            kwargs.pop("training_kwargs")
        else:
            training_kwargs = {}

        if "ft_kwargs" in kwargs:
            ft_kwargs = kwargs["ft_kwargs"]
            kwargs.pop("ft_kwargs")
        else:
            ft_kwargs = {}

        if "pre_func_inp" not in kwargs:
            self.pre_func_inp = lambda x: x
        else:
            self.pre_func_inp = kwargs["pre_func_inp"]

        if "tracker" not in training_kwargs:
            training_kwargs["tracker"] = default_tracker

        key0, key1 = jrandom.split(training_key)

        training_kwargs = merge_dicts(kwargs, training_kwargs)

        model, track_params = super().fit(*args, training_key=key0, **training_kwargs)  # train model

        self.track_params = track_params    # Save track parameters in class?

        if "batch_size_st" in training_kwargs:
            self.batch_size = training_kwargs["batch_size_st"][-1]
        else:
            self.batch_size = 16  # default value

        ft_kwargs = merge_dicts(kwargs, ft_kwargs)

        self.fine_tune_basis(None, args=args, kwargs=ft_kwargs, key=key1)       # fine tune basis


    def fine_tune_basis(self, basis = None, *, args, kwargs, key):

        if "loss" in kwargs:
            norm_loss_ = kwargs["loss"]
        else:
            print("Defaulting to L2 norm")    
            norm_loss_ = lambda x1, x2: 100*(jnp.linalg.norm(x1 - x2) / jnp.linalg.norm(x2))


        if basis is None:
            inp = args[0] if len(args) > 0 else kwargs["input"]

            all_bases = self.model.eval_with_batches(
                inp,
                self.batch_size,
                call_func=lambda x: self.model.latent(
                    self.pre_func_inp(x), get_basis_coeffs=True, **self.track_params
                )[0],
                str="Finding train latent space...",
                end_type="concat",
                key_idx=0,
            )
            basis = jnp.linalg.svd(all_bases, full_matrices=False)[0]
            self.basis = basis[:, : self.track_params["k_max"]]
        else:
            self.basis = basis
        

        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(diff_model, static_model, input, out, idx, basis):
            model = eqx.combine(diff_model, static_model)
            pred = model(input, apply_basis=basis, inv_norm_out=False)  # Note: here was self.basis, thus basis arg was not being used
            aux = {"loss": norm_loss_(pred, out)}
            return norm_loss_(pred, out), aux

        if "loss_type" in kwargs:
            raise ValueError(
                "You should not provide loss_type in ft_kwargs since it is predefined to apply the basis."
            )

        kwargs["loss_type"] = loss_fun
        kwargs["loss_kwargs"] = {"basis": self.basis}
        fix_comp = lambda model: model.encode.model
        print("Fine tuning the basis ...")
        super().fit(*args, training_key=key, fix_comp=fix_comp, **kwargs)

    def evaluate(
        self,
        x_train_o=None,
        y_train_o=None,
        x_test_o=None,
        y_test_o=None,
        batch_size=None,
        pre_func_inp=lambda x: x,
        pre_func_out=lambda x: x,
        call_func=None,
    ):

        call_func = lambda x: self.model(pre_func_inp(x), apply_basis=self.basis)
        res = super().evaluate(
            x_train_o,
            y_train_o,
            x_test_o,
            y_test_o,
            batch_size,
            call_func=call_func,
            pre_func_inp=pre_func_inp,
            pre_func_out=pre_func_out,
        )
        return res

    def AE_interpolate(
        self,
        p_train,
        p_test,
        x_train_o,
        y_test_o,
        batch_size=None,
        latent_func=None,
        decode_func=None,
        norm_out_func=None,
    ):
        call_func = lambda x: (
            self.model.latent(x, apply_basis=self.basis)
            if latent_func is None
            else latent_func
        )
        return super().AE_interpolate(
            p_train,
            p_test,
            x_train_o,
            y_test_o,
            batch_size,
            latent_func=call_func,
            decode_func=decode_func,
            norm_out_func=norm_out_func,
        )


class V_AE_Trainor_class(RRAE_Trainor_class):
    """ " Trainor class for variational batching."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("This class is not implemented yet.")

    def fit(
        self,
        input,
        output,
        loss_type=None,
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

        output = self.model.norm_out(output)

        training_params = {
            "loss_type": loss_type,
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

        @eqx.filter_value_and_grad(has_aux=True)
        def loss_fun(model, input, out, rand_mat, **kwargs):
            all_lat = model.latent(input)
            lat_b = all_lat @ rand_mat
            pred = model.norm_out(model.decode(lat_b))
            wv = jnp.array([1.0])
            return find_weighted_loss([norm_loss_(pred, out)], weight_vals=wv), (pred,)

        @eqx.filter_jit
        def make_step(model, input, out, opt_state, rand_mat, **loss_kwargs):
            (loss, aux), grads = loss_fun(model, input, out, rand_mat, **loss_kwargs)
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
        training_num = jrandom.randint(training_key, (1,), 0, 1000)[0]

        try:
            inc = 0
            for steps, lr, batch_size, mul_l in zip(
                step_st, lr_st, batch_size_st, mul_lr
            ):
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

                idx_batch = jnp.arange(batch_size)

                for step, (inp, out, batch_perm) in zip(
                    range(steps),
                    dataloader(
                        [input.T, output.T, jnp.arange(0, input.shape[-1], 1)],
                        batch_size,
                        key_idx=training_num,
                    ),
                ):

                    start = time.perf_counter()

                    I = jnp.eye(batch_size)
                    G = jrandom.uniform(
                        jrandom.key(inc),
                        (batch_size, batch_size),
                        minval=-0.1,
                        maxval=0.1,
                    )
                    G = G.at[idx_batch, idx_batch].set(0)

                    loss, model, opt_state, aux = make_step(
                        model,
                        inp.T,
                        out.T,
                        opt_state,
                        I + G,
                        **loss_kwargs,
                    )
                    inc += 1
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
        except (Exception, KeyboardInterrupt) as e:
            print(e)
            pdb.set_trace()
            pass

        model = eqx.nn.inference_mode(model)
        self.model = model
        self.batch_size = batch_size
        self.t_all = t_all
        return model
