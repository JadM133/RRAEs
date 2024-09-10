import jax.random as jr
import pytest
import jax.numpy as jnp
import numpy as np
from RRAEs.utilities import my_vmap
from RRAEs.interpolation import Objects_Interpolator_nD


def test_nD_interp_on_1D():
    x_train = jnp.expand_dims(jnp.linspace(0, 100, 99), -1)
    y_train = jr.normal(jr.PRNGKey(0), (5, 99))
    x_test = jr.uniform(jr.PRNGKey(0), (25, 1), minval=0, maxval=99)

    obj1 = Objects_Interpolator_nD()
    obj1.fit(x_train, y_train)
    res1 = obj1(x_test)

    def interp_1D(y_train, x_train, x_test):
        res = []
        my_interp = lambda x, y, z: np.interp(y, z, x)
        for test in x_test:
            res.append(my_vmap(my_interp)(y_train, args=(test, x_train[:, 0])))
        return np.stack(res, axis=1)

    res2 = interp_1D(y_train, x_train, x_test)
    assert jnp.allclose(res1, res2) 

def test_nD_interp_on_2D():
    """ We test the D-dimensional interpolation on a 2D grid."""
    
    grid_size = 9
    x_vals_0 = np.linspace(0, 100, grid_size)
    x_vals_1 = np.linspace(0, 100, grid_size)
    l = x_vals_1[1]-x_vals_0[0]
    x_0 = np.repeat(x_vals_0, grid_size)
    x_1 = np.tile(x_vals_1, grid_size)
    x_train = jnp.stack([x_0, x_1], axis=-1)
    y_train = jr.normal(jr.PRNGKey(0), (5, grid_size**2))
    idx = [2, 3]
    x1, x2, x3, x4 = x_train[grid_size*idx[0]+idx[1]], x_train[grid_size*idx[0]+idx[1]+1], x_train[grid_size*(idx[0]+1)+idx[1]], x_train[grid_size*(idx[0]+1)+idx[1]+1]
    y1, y2, y3, y4 = y_train[:, grid_size*idx[0]+idx[1]], y_train[:, grid_size*idx[0]+idx[1]+1], y_train[:, grid_size*(idx[0]+1)+idx[1]], y_train[:, grid_size*(idx[0]+1)+idx[1]+1]
    
    # Testing a point in the center of a grid square.
    xx = (x1+x2+x3+x4)/4
    y_true = jnp.expand_dims((y1+y2+y3+y4)/4, -1)
    x_test = jnp.array([[xx[0], xx[1]],])
    obj = Objects_Interpolator_nD()
    obj.fit(x_train, y_train)
    res = obj(x_test)

    # assert jnp.allclose(res, y_true) 

    # Testing a point at a quarter distance (surface wise).
    xx = (x1+3*x4)/4
    coeffs = [(l/4)**2, (3*l/4)**2, (l/4)*(3*l/4), (3*l/4)*(l/4)]
    coeffs = coeffs/sum(coeffs)
    y_true = jnp.expand_dims(coeffs[0]*y1 + coeffs[1]*y4 + coeffs[2]*y3 + coeffs[3]*y2, -1)
    x_test = jnp.array([[xx[0], xx[1]],])
    obj = Objects_Interpolator_nD()
    obj.fit(x_train, y_train)
    res = obj(x_test)
    assert res.shape == y_true.shape
    assert jnp.allclose(res, y_true) 

def test_nD_interp_shape_3D():
    x_vals = np.linspace(0, 100, 9)
    y_vals = np.linspace(0, 100, 9)
    X_m = np.tile(np.repeat(x_vals, 9), 9)
    Y_m = np.tile(np.tile(y_vals, 9), 9)
    Z_m = np.repeat(np.linspace(0, 100, 9), 9 * 9)
    x_train = np.stack([X_m, Y_m, Z_m], axis=-1)
    y_train = np.random.rand(2, 1500)
    x_test = np.random.rand(8, 3) * 100
    obj = Objects_Interpolator_nD()
    obj.fit(x_train, y_train)
    res = obj(x_test)
    assert res.shape == (2, 8)

