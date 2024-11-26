import numpy as np
import jax.numpy as jnp


def match_k(k):
    if (k == "step_st") or k == ("batch_size_st") or (k == "lr_st"):
        return "list"
    elif k == "print_every" or k == "width_size" or k == "depth":
        return "int"
    elif k == "get_svd" or k == "get_coeffs":
        return "bool"
    else:
        raise NotImplementedError(
            "Define match_k first if you want to add a new argument to fitting."
        )


def update_val_from_matlab(k, v):
    match match_k(k):
        case "int":
            return int(v[0][0][0][0])
        case "list":
            return (v[0][0][0]).tolist()
        case "bool":
            return bool(v[0][0][0][0])


def from_void_to_dict(v):
    return {name: v[name].tolist() for name in v.dtype.names}


def s(st_from_mat):
    """To use o string inputs from MATLAB."""
    return str(st_from_mat[0][0][0])


def n(arg, none_val=None):
    if arg == "None":
        category = none_val
    else:
        category = np.array(arg)
    return category


def prep_struct(st):
    st = from_void_to_dict(st)
    return {k: update_val_from_matlab(k, v) for k, v in st.items()}


def dict_to_double(d):
    for k, v in d.items():
        if isinstance(v, jnp.array):
            d[k] = jnp.astype(v, jnp.double)
