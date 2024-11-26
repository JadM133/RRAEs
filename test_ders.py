from jax import vjp, jvp
import jax.random as random
import jax.numpy as jnp
import pdb
import jax
from jax import custom_jvp
import dill
from jax._src.lax import lax as lax_internal
import jax.lax as lax
from jax import custom_jvp


def _extract_diagonal(s):
    """Extract the diagonal from a batched matrix"""
    i = lax.iota("int32", min(s.shape[-2], s.shape[-1]))
    return s[..., i, i]


def _construct_diagonal(s):
    """Construct a (batched) diagonal matrix"""
    i = lax.iota("int32", s.shape[-1])
    return lax.full((*s.shape, s.shape[-1]), 0, s.dtype).at[..., i, i].set(s)


def _H(x):
    return _T(x).conj()


def _T(x):
    return lax.transpose(x, (*range(x.ndim - 2), x.ndim - 1, x.ndim - 2))


@custom_jvp
def stable_SVD(x):
    return jnp.linalg.svd(x, full_matrices=False)


@stable_SVD.defjvp
def stable_SVD_jvp(primals, tangents):
    (A,) = primals
    (dA,) = tangents
    U, s, Vt = jnp.linalg.svd(A, full_matrices=False)
    Ut, V = _H(U), _H(Vt)
    s_dim = s[..., None, :]
    dS = Ut @ dA @ V
    ds = _extract_diagonal(dS.real)
    s_diffs = (s_dim + _T(s_dim)) * (s_dim - _T(s_dim))
    s_diffs_zeros = lax_internal._eye(s.dtype, (s.shape[-1], s.shape[-1]), 0)
    s_diffs_zeros = lax.expand_dims(s_diffs_zeros, range(s_diffs.ndim - 2))
    F = 1 / (s_diffs + s_diffs_zeros) - s_diffs_zeros
    dSS = s_dim.astype(A.dtype) * dS
    SdS = _T(s_dim.astype(A.dtype)) * dS
    s_zeros = (s < 1e-5).astype(s.dtype)
    s_inv = 1 / (s + s_zeros) - s_zeros
    s_inv_mat = _construct_diagonal(s_inv)
    dUdV_diag = 0.5 * (dS - _H(dS)) * s_inv_mat.astype(A.dtype)
    dU = U @ (F.astype(A.dtype) * (dSS + _H(dSS)) + dUdV_diag)
    dV = V @ (F.astype(A.dtype) * (SdS + _H(SdS)))
    m, n = A.shape[-2:]
    s_inv = jnp.expand_dims(s_inv, 0)
    if m > n:
        dAV = dA @ V
        dU = dU + (dAV - U @ (Ut @ dAV)) / s_dim.astype(A.dtype)
    if n > m:
        dAHU = _H(dA) @ U
        dV = dV + (dAHU - V @ (Vt @ dAHU)) / s_dim.astype(A.dtype)
    return (U, s, Vt), (dU, ds, _H(dV))


if __name__ == "__main__":
    key = random.key(0)

    def f(x):
        U, s, Vt = stable_SVD(x)
        return jnp.linalg.norm(U[:]*s[:] @ Vt[:] - x)

    with open("lat_of_nan.pkl", "rb") as ff:
        A = dill.load(ff)

    pdb.set_trace()
    # A = jnp.diag(jnp.asarray([1, 1, 1, 2, 3], dtype=float))
    # y, u = jvp(predict, (A,), (jnp.array([2.0]),))
    # y, vjp_fun = vjp(stable_SVD, A)

    # print(vjp_fun(jnp.linalg.svd(A)[0]))
    grad_fun = jax.value_and_grad(f)
    print(grad_fun(A))
    # print(u)
    # print(grad_fun(1.0))
    pdb.set_trace()
