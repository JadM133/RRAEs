import jax.lax as lax
from jax._src.lax import lax as lax_internal
from jax import custom_jvp
import jax.numpy as jnp
import jax.random as jrandom
import jax


def _extract_diagonal(s):
    i = lax.iota("int32", min(s.shape[-2], s.shape[-1]))
    return s[..., i, i]


def _construct_diagonal(s):
    i = lax.iota("int32", s.shape[-1])
    return lax.full((*s.shape, s.shape[-1]), 0, s.dtype).at[..., i, i].set(s)


def _H(x):
    return _T(x).conj()


def _T(x):
    return lax.transpose(x, (*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))


@custom_jvp
def my_SVD(x):
    return jnp.linalg.svd(x, full_matrices=False)


@my_SVD.defjvp
def _svd_jvp_rule(primals, tangents):
    (A,) = primals
    (dA,) = tangents
    U, s, Vt = jnp.linalg.svd(A, full_matrices=False)

    Ut, V = _H(U), _H(Vt)
    s_dim = s[..., None, :]
    dS = Ut @ dA @ V
    ds = _extract_diagonal(dS.real)

    s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
    s_diffs_zeros = lax_internal._eye(s.dtype, (s.shape[-1], s.shape[-1]))
    s_diffs_zeros = lax.expand_dims(s_diffs_zeros, range(s_diffs.ndim - 2))
    F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
    dSS = s_dim.astype(A.dtype) * dS
    SdS = _T(s_dim.astype(A.dtype)) * dS

    s_zeros = (s == 0).astype(s.dtype)
    s_inv = 1 / (s + s_zeros) - s_zeros
    s_inv_mat = _construct_diagonal(s_inv)
    dUdV_diag = 0.5 * (dS - _H(dS)) * s_inv_mat.astype(A.dtype)
    dU = U @ (F.astype(A.dtype) * (dSS + _H(dSS)) + dUdV_diag)
    dV = V @ (F.astype(A.dtype) * (SdS + _H(SdS)))

    m, n = A.shape[-2:]
    if m > n:
        dAV = dA @ V
        dU = dU + (dAV - U @ (Ut @ dAV)) * s_inv.astype(A.dtype)
    if n > m:
        dAHU = _H(dA) @ U
        dV = dV + (dAHU - V @ (Vt @ dAHU)) * s_inv.astype(A.dtype)

    return (U, s, Vt), (dU, ds, _H(dV))


def new_SVD_to_scalar(A):
    U, s, Vt = my_SVD(A)
    # Case 1:
    # return jnp.linalg.norm((U*s) @ Vt - A)
    # Case 2:
    # return jnp.linalg.norm((U*s) @ Vt)


def normal_SVD_to_scalar(A):
    U, s, Vt = jnp.linalg.svd(A, full_matrices=False)
    # Case 1:
    # return jnp.linalg.norm((U*s) @ Vt - A)
    # Case 2:
    # return jnp.linalg.norm((U*s) @ Vt)


def test_random_normal(length, width):
    A = jrandom.uniform(jrandom.PRNGKey(0), (length, width))
    new_res = jax.value_and_grad(new_SVD_to_scalar)(A)
    normal_res = jax.value_and_grad(normal_SVD_to_scalar)(A)
    assert jnp.allclose(new_res[0], normal_res[0])
    assert jnp.allclose(new_res[1], normal_res[1])  # Returns False in Case 1


if __name__ == "__main__":
    test_random_normal(100, 10)
