import pdb
import jax.numpy as jnp
import equinox as eqx


class Attribute_Class(eqx.Module):
    attr: eqx.Module
    call_func: callable

    def __init__(self, attr, call_func):
        self.attr = attr
        self.call_func = call_func

    def __repr__(self):
        return repr(self.attr)

    def __getattr__(self, name):
        if hasattr(self.attr, name):
            return getattr(self.attr, name)
        else:
            raise AttributeError(f"Attribute {name} not found in {self.attr}")

    def __str__(self):
        return str(self.attr)

    def __call__(self, *args, **kwargs):
        return self.call_func(*args, **kwargs)


class Norm(eqx.Module):
    model: eqx.Module
    norm_in: str
    norm_out: str
    params_in: dict
    params_out: dict
    norm_in: callable
    inv_norm_in: callable
    norm_out: callable
    inv_norm_out: callable
    norm_and_inv_func: callable
    pre_func_inp: callable
    pre_func_out: callable

    def __init__(
        self,
        model,
        in_train=None,
        out_train=None,
        norm_in="None",
        norm_out="None",
        params_in=None,
        params_out=None,
        pre_func_inp=lambda x: x,
        pre_func_out=lambda x: x,
    ):

        assert (params_in is not None) or (
            in_train is not None
        ), "Either params or in_train must be provided to set norm parameters"

        assert not (
            params_in is not None and in_train is not None
        ), "Only one of params or in_train must be provided to set norm parameters"

        self.norm_in = norm_in
        self.norm_out = norm_out

        self.model = model

        if out_train is None:
            out_train = in_train

        match norm_in:
            case "minmax":
                if params_in is None:
                    self.params_in = {
                        "min": jnp.min(in_train),
                        "max": jnp.max(in_train),
                    }
                else:
                    self.params_in = params_in
                self.norm_in = lambda x: (x - self.params_in["min"]) / (
                    self.params_in["max"] - self.params_in["min"]
                )
                self.inv_norm_in = (
                    lambda x: x * (self.params_in["max"] - self.params_in["min"])
                    + self.params_in["min"]
                )
            case "meanstd":
                if params_in is None:
                    self.params_in = {
                        "mean": jnp.mean(in_train),
                        "std": jnp.std(in_train),
                    }
                else:
                    self.params_in = params_in
                self.norm_in = (
                    lambda x: (x - self.params_in["mean"]) / self.params_in["std"]
                )
                self.inv_norm_in = (
                    lambda x: x * self.params_in["std"] + self.params_in["mean"]
                )
            case "None":
                if params_in is None:
                    self.params_in = {}
                else:
                    self.params_in = params_in
                self.norm_in = lambda x: x
                self.inv_norm_in = lambda x: x
            case _:
                raise ValueError(f"Unknown input norm type: {norm_in}")

        match norm_out:
            case "minmax":
                if params_out is None:
                    self.params_out = {
                        "min": jnp.min(out_train),
                        "max": jnp.max(out_train),
                    }
                else:
                    self.params_out = params_out
                self.norm_out = lambda x: (x - self.params_out["min"]) / (
                    self.params_out["max"] - self.params_out["min"]
                )
                self.inv_norm_out = (
                    lambda x: x * (self.params_out["max"] - self.params_out["min"])
                    + self.params_out["min"]
                )
            case "meanstd":
                if params_out is None:
                    self.params_out = {
                        "mean": jnp.mean(out_train),
                        "std": jnp.std(out_train),
                    }
                else:
                    self.params_out = params_out
                self.norm_out = (
                    lambda x: (x - self.params_out["mean"]) / self.params_out["std"]
                )
                self.inv_norm_out = (
                    lambda x: x * self.params_out["std"] + self.params_out["mean"]
                )
            case "None":
                if params_out is None:
                    self.params_out = {}
                else:
                    self.params_out = params_out
                self.norm_out = lambda x: x
                self.inv_norm_out = lambda x: x
        self.pre_func_inp = pre_func_inp
        self.pre_func_out = pre_func_out

        def norm_and_inv_func(
            func, norm_bool=True, inv_bool=True
        ):  # functions to be norm/inv_norl have to accept x as first arg
            if norm_bool and inv_bool:
                return lambda x, *args, **kwargs: self.inv_norm_out(
                    func(self.norm_in(self.pre_func_inp(x)), *args, **kwargs)
                )
            if norm_bool:
                return lambda x, *args, **kwargs: func(
                    self.norm_in(self.pre_func_inp(x)), *args, **kwargs
                )
            if inv_bool:
                return lambda x, *args, **kwargs: self.inv_norm_out(
                    func(x), *args, **kwargs
                )
            raise ValueError("Either norm_bool or inv_bool must be True")

        self.norm_and_inv_func = norm_and_inv_func

    def __call__(self, x, norm_in=True, inv_norm_out=True, *args, **kwargs):
        return self.norm_and_inv_func(self.model, norm_in, inv_norm_out)(
            x, *args, **kwargs
        )

    def __getattr__(self, name: str):
        if hasattr(self.model, "norm_funcs"):
            norm_bool = name in self.model.norm_funcs
        else:
            norm_bool = False
        if hasattr(self.model, "inv_norm_funcs"):
            inv_bool = name in self.model.inv_norm_funcs
        else:
            inv_bool = False

        attr = getattr(self.model, name)

        if attr is None:
            return None
        
        if not (norm_bool or inv_bool):
            call_func = lambda *args, **kwargs: attr(*args, **kwargs)
        else:
            call_func = self.norm_and_inv_func(attr, norm_bool, inv_bool)

        return Attribute_Class(attr, call_func)
