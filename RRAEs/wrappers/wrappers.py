import jax
import jax.numpy as jnp
import dataclasses
from dataclasses import dataclass

@dataclass(frozen=True)
class NormParams:
    pass

@dataclass(frozen=True)
class MeanStdParams(NormParams):
    mean: float
    std: float

@dataclass(frozen=True)
class MinMaxParams(NormParams):
    min: float
    max: float

def find_norm_funcs(
    array=None,
    norm_typ="None",
    params=None,
    ):
    """ Function that finds norm and inv_norm functions

    The functions can be defined either based on an array (in_train),
    In this case, the required parameters (e.g., mean, min, etc.) are
    found using in_train. Otherwise, the parameters can be given
    explicitly using params_in (check below for some examples).

    Parameters
    ----------
    in_train : input array that is to be normalized
    norm_in: Type of normalization, "minmax" and "meanstd" supported
    params_in: Normalization parameters if they are to be given manually
               instead of being computed from in_train.
               These for example can be:
               For mean/std normalization, params_out = {"mean": 0.12, "std": 0.23}
               For min/max normalization, params_out = {"min": -1, "max": 3,}
    Returns
    -------
        A new subclass of base_cls with methods that normalize the input when called.
    """
    if norm_typ != "None":
        assert (params is not None) or (
            array is not None
        ), "Either params or in_train must be provided to set norm parameters"

        assert not (
            params is not None and array is not None
        ), "Only one of params or in_train must be provided to set norm parameters"


    match norm_typ:
        case "minmax":
            if params is None:
                params = MinMaxParams(min=jnp.min(array), max=jnp.max(array))
            else:
                params = params
            norm_fn = lambda self, x: (x - params.min) / (params.max - params.min)
            inv_norm_fn = lambda self, x: x * (params.max - params.min) + params.min
        case "meanstd":
            if params is None:
                params = MeanStdParams(mean=jnp.mean(array), std=jnp.std(array))
            else:
                params = params
            norm_fn = lambda self, x: (x - params.mean) / params.std
            inv_norm_fn = lambda self, x: x * params.std + params.mean
        case "None":
            if params is None:
                params = NormParams()
            else:
                params = params
            norm_fn = lambda self, x: x
            inv_norm_fn = lambda self, x: x
        case _:
            raise NotImplementedError(f"norm_in specified {norm_typ} is not implemented.")

    return norm_fn, inv_norm_fn, params

def norm_in_wrap(base_cls, array=None, norm_typ="None", params=None, methods_to_wrap=["__call__"]):
    """ Wrapper that normalizes the input of a function of a subclass of eqx.Module

    The parameters of normalization can be either computed based on an array, or given
    by the user.

    Parameters
    ----------
    base_cls : Base class, subclass of eqx.Module of which functions will
               be modified.
    methods_to_wrap: Name of the methods in base_cls to be wrapped
    array : The array from which norm parameters are to be found
    norm_typ: Type of normalization, "meanstd" and "minmax" are supported
    params: Parameters of normalization if these are to be given manually
                These for example can be:
                For mean/std normalization, params_out = {"mean": 0.12, "std": 0.23}
                For min/max normalization, params_out = {"min": -1, "max": 3,}
    Returns
    -------
        A new subclass of base_cls with methods that normalize the input when called.
    """
    norm_in, inv_norm_in, params_in = find_norm_funcs(array, norm_typ, params)
    def norm_in_decorator(fn):
        def wrapped(self, x, *args, **kwargs):
            result = fn(self, norm_in(self, x), *args, **kwargs)
            return result
        return wrapped, {"norm_in": (callable, norm_in), "inv_norm_in": (callable, inv_norm_in), "params_in": (dict, params_in)}
    return make_wrapped(base_cls, norm_in_decorator, methods_to_wrap)

def inv_norm_out_wrap(base_cls, array=None, norm_typ="None", params=None, methods_to_wrap=["__call__"]):
    """ Wrapper that de-normalizes the output of a function of a subclass of eqx.Module

    The parameters of normalization can be either computed based on an array, or given
    by the user.

    Parameters
    ----------
    base_cls : Base class, subclass of eqx.Module of which functions will
               be modified.
    methods_to_wrap: Name of the methods in base_cls to be wrapped
    array : The array from which norm parameters are to be found
    norm_typ: Type of normalization, "meanstd" and "minmax" are supported
    params: Parameters of normalization if these are to be given manually
                These for example can be:
                For mean/std normalization, params_out = {"mean": 0.12, "std": 0.23}
                For min/max normalization, params_out = {"min": -1, "max": 3,}
    Returns
    -------
        A new subclass of base_cls with de-normalized methods.
        The given methods accept an additional keyword argument "keep_normalized"
        If this argument is passed as True, functions retrieve original behavior
    """
    norm_out, inv_norm_out, params_out = find_norm_funcs(array, norm_typ, params)

    def norm_out_decorator(fn):
        def wrapped(self, x, *args, keep_normalized=False, **kwargs):
            if keep_normalized:
                result = fn(self, x, *args, **kwargs)
            else:
                result = inv_norm_out(self, fn(self, x, *args, **kwargs))
            return result
        return wrapped, {"norm_out": (callable, norm_out), "inv_norm_out": (callable, inv_norm_out), "params_out": (dict, params_out)}
    return make_wrapped(base_cls, norm_out_decorator, methods_to_wrap)

def norm_wrap(base_cls, array_in=None, norm_typ_in="None", params_in=None, array_out=None, norm_typ_out="None", params_out=None, methods_to_wrap_in=["__call__"], methods_to_wrap_out=["__call__"]):
    """ Wrapper that normalizes functions of a subclass of eqx.Module

    Parameters
    ----------
    base_cls : Base class, subclass of eqx.Module of which functions will
               be modified.
    methods_to_wrap: Name of the methods in base_cls to be wrapped
    ... : Other parameters are explained in norm_in_wrap and norm_out_wrap

    Returns
    -------
        A new subclass of base_cls with normalized methods.
    """
    after_in = norm_in_wrap(base_cls, array_in, norm_typ_in, params_in, methods_to_wrap_in)
    after_out = inv_norm_out_wrap(after_in, array_out, norm_typ_out, params_out, methods_to_wrap_out)
    return after_out

def vmap_wrap(base_cls, map_axis, count=1, methods_to_wrap=["__call__"]):
    """ Wrapper that vectorizes functions of a subclass of eqx.Module

    Parameters
    ----------
    base_cls : Base class, subclass of eqx.Module of which functions will
               be modified.
    map_axis : Axis along which to vectorize the functions
    methods_to_wrap: Name of the methods in base_cls to be wrapped
    count: How many times to vectorize the functions

    Returns
    -------
        A new subclass of base_cls with vectorized methods.
        The given methods accept an additional keyword argument "no_map"
        If this argument is passed as True, functions retrieve original behavior
    """
    def vmap_decorator(fn):
        def wrapped(self, x, *args, no_map=False, **kwargs):
            if (map_axis is None) or no_map:
                return fn(self, x, *args, **kwargs)
            f = lambda x: fn(self, x, *args, **kwargs)
            for _ in range(count):
                f = jax.vmap(f, in_axes=(map_axis,), out_axes=map_axis)
            out = f(x)
            return out
        return wrapped, {}
    return make_wrapped(base_cls, vmap_decorator, methods_to_wrap)



def make_wrapped(base_cls, decorator, methods_to_wrap=["__call__"]):
    """
    Create a subclass of base_cls with specified methods wrapped by decorator.

    Parameters
    ----------
        base_cls: Original class to wrap.
        methods_to_wrap: List of method names (strings) to decorate.
        decorator: The wanted modification to the methods given above

    Returns
    -------
        A new subclass of base_cls with decorated methods.
    """
    attrs = {}
    annotations = {}
    seen_fields = set()

    for method_name in methods_to_wrap:
        if not hasattr(base_cls, method_name):
            raise AttributeError(f"Method {method_name} not found in {base_cls}")
        
        original = getattr(base_cls, method_name)
        wrapped_method, extra_fields = decorator(original)
        attrs[method_name] = wrapped_method

        # Sort field names to avoid ordering issues
        for field_name in extra_fields:  
            if field_name in seen_fields:
                continue

            field_type, default_value = extra_fields[field_name]
            annotations[field_name] = field_type

            attrs[field_name] = dataclasses.field(default=default_value)

            seen_fields.add(field_name)

    if annotations:
        attrs["__annotations__"] = annotations

    Wrapped = type(base_cls.__name__, (base_cls,), attrs)
    return Wrapped

