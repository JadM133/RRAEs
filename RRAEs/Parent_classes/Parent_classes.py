import equinox as eqx
import jax
from RRAEs.utilities import dataloader, MLP_with_linear
import jax.numpy as jnp
import jax.random as jrandom
import itertools
from tqdm import tqdm

_identity = lambda x, *args, **kwargs: x 

class BaseClass(eqx.Module):
    map_axis: int
    model: eqx.Module
    count: int

    def __init__(self, model, map_axis=None, *args, count=1, **kwargs):
        self.map_axis = map_axis
        self.model = model
        self.count = count

    def __call__(self, x, *args, **kwargs):
        if self.map_axis is None:
            return self.model(x, *args, **kwargs)
        fn = lambda x: self.model(x, *args, **kwargs)
        for _ in range(self.count):
            fn = jax.vmap(fn, in_axes=(self.map_axis,), out_axes=self.map_axis)
        out = fn(x)
        return out

    def eval_with_batches(
        self,
        x,
        batch_size,
        call_func,
        end_type="concat_and_resort",
        str=None,
        *args,
        key_idx,
        **kwargs,
    ):
        """Works for array input with data as final dim."""
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

    def __getattr__(self, name: str):
        return getattr(self.model, name)


class Autoencoder(eqx.Module):
    _encode: MLP_with_linear
    _decode: MLP_with_linear
    _perform_in_latent: callable
    _perform_in_latent: callable
    map_latent: bool
    norm_funcs: list
    inv_norm_funcs: list
    count: int

    def __init__(
        self,
        in_size,
        latent_size,
        latent_size_after=None,
        _encode=None,
        _decode=None,
        map_latent=True,
        *,
        key,
        count=1,
        kwargs_enc={},
        kwargs_dec={},
        **kwargs,
    ):
        key_e, key_d = jrandom.split(key)
        if latent_size_after is None:
            latent_size_after = latent_size

        if _encode is None:
            if "width_size" not in kwargs_enc.keys():
                kwargs_enc["width_size"] = 64

            if "depth" not in kwargs_enc.keys():
                kwargs_enc["depth"] = 1

            model_cls = MLP_with_linear(
                in_size=in_size,
                out_size=latent_size,
                key=key_e,
                **kwargs_enc,
            )
            self._encode = BaseClass(model_cls, -1, count=count)

        else:
            self._encode = _encode

        if not hasattr(self, "_perform_in_latent"):
            self._perform_in_latent = _identity

        if _decode is None:
            if "width_size" not in kwargs_dec.keys():
                kwargs_dec["width_size"] = 64
            if "depth" not in kwargs_dec.keys():
                kwargs_dec["depth"] = 6

            model_cls = MLP_with_linear(
                in_size=latent_size_after,
                out_size=in_size,
                key=key_d,
                **kwargs_dec,
            )
            self._decode = BaseClass(model_cls, -1, count=count)
        else:
            self._decode = _decode

        self.count = count
        self.map_latent = map_latent
        self.inv_norm_funcs = ["decode"]
        self.norm_funcs = ["encode", "latent"]

    def encode(self, x, *args, **kwargs):
        return self._encode(x, *args, **kwargs)
    
    def decode(self, x, *args, **kwargs):
        return self._decode(x, *args, **kwargs)

    def perform_in_latent(self, y, *args, **kwargs):
        if self.map_latent:
            new_perform_in_latent = lambda x: self._perform_in_latent(
                x, *args, **kwargs
            )
            for _ in range(self.count):
                new_perform_in_latent = jax.vmap(new_perform_in_latent, in_axes=-1, out_axes=-1) 
            return new_perform_in_latent(y)
        return self._perform_in_latent(y, *args, **kwargs)

    def __call__(self, x, *args, **kwargs):
        return self.decode(self.perform_in_latent(self.encode(x), *args, **kwargs))

    def latent(self, x, *args, **kwargs):
        return self.perform_in_latent(self.encode(x), *args, **kwargs)
    
    def get_basis_coeffs(self, x, *args, **kwargs):
        return None, self.latent(x, *args, **kwargs)

    def decode_coeffs(self, c, *args, **kwargs):
        return self.decode(x, *args, **kwargs)
