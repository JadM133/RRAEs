from RRAEs.utilities import stable_SVD
from jax.numpy.linalg import svd as normal_svd
import jax.random as jrandom
import pytest
import jax
import jax.numpy as jnp


def stable_SVD_to_scalar(A):
    U, s, Vt = stable_SVD(A)
    return jnp.linalg.norm((U * s) @ Vt)  # Any scalar depending on U, s, and Vt.

def normal_svd_to_scalar(A):
    U, s, Vt = normal_svd(A, full_matrices=False)
    return jnp.linalg.norm((U * s) @ Vt)  # Any scalar depending on U, s, and Vt.

@pytest.mark.parametrize(
    "length, width",
    [(10, 10), (100, 10), (10, 100), (50000, 100), (1000, 1000), (100, 50000)],
)
def test_random_normal(length, width):
    A = jrandom.uniform(jrandom.PRNGKey(0), (length, width))
    stable_res = jax.value_and_grad(stable_SVD_to_scalar)(A)
    normal_res = jax.value_and_grad(normal_svd_to_scalar)(A)
    assert jnp.allclose(stable_res[0], normal_res[0])
    assert jnp.allclose(stable_res[1], normal_res[1])

