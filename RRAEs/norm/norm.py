import pdb
import jax.numpy as jnp
import equinox as eqx


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

    def __init__(
        self, model, in_train, out_train=None, norm_in="None", norm_out="None"
    ):

        self.norm_in = norm_in
        self.norm_out = norm_out

        self.model = model

        if out_train is None:
            out_train = in_train

        match norm_in:
            case "minmax":
                self.params_in = {"min": jnp.min(in_train), "max": jnp.max(in_train)}
                self.norm_in = lambda x: (x - self.params_in["min"]) / (
                    self.params_in["max"] - self.params_in["min"]
                )
                self.inv_norm_in = (
                    lambda x: x * (self.params_in["max"] - self.params_in["min"])
                    + self.params_in["min"]
                )
            case "meanstd":
                self.params_in = {"mean": jnp.mean(in_train), "std": jnp.std(in_train)}
                self.norm_in = (
                    lambda x: (x - self.params_in["mean"]) / self.params_in["std"]
                )
                self.inv_norm_in = (
                    lambda x: x * self.params_in["std"] + self.params_in["mean"]
                )
            case "None":
                self.params_in = {}
                self.norm_in = lambda x: x
                self.inv_norm_in = lambda x: x
            case _:
                raise ValueError(f"Unknown input norm type: {norm_in}")

        match norm_out:
            case "minmax":
                self.params_out = {"min": jnp.min(out_train), "max": jnp.max(out_train)}
                self.norm_out = lambda x: (x - self.params_out["min"]) / (
                    self.params_out["max"] - self.params_out["min"]
                )
                self.inv_norm_out = (
                    lambda x: x * (self.params_out["max"] - self.params_out["min"])
                    + self.params_out["min"]
                )
            case "meanstd":
                self.params_out = {
                    "mean": jnp.mean(out_train),
                    "std": jnp.std(out_train),
                }
                self.norm_out = lambda x: (x - self.params_out["mean"]) / self.params_out["std"]
                self.inv_norm_out = (
                    lambda x: x * self.params_out["std"] + self.params_out["mean"]
                )
            case "None":
                self.params_out = {}
                self.norm_out = lambda x: x
                self.inv_norm_out = lambda x: x

    def with_no_inv(self, x):
        return self.model(self.norm_in(x))
    
    def __call__(self, x):
        return self.inv_norm_out(self.model(self.norm_in(x)))

    def __getattr__(self, name: str):
        return getattr(self.model, name)
