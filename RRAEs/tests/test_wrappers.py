from RRAEs.wrappers import vmap_wrap, norm_wrap
from equinox.nn import MLP
import jax.random as jrandom
import pytest
import numpy as np
import math

def test_vmap_wrapper():
    # Usually MLP only accepts a vector, here we give
    # a tensor and vectorize over the last axis twice
    data = jrandom.normal(jrandom.key(0), (50, 60, 600))
    model_cls = vmap_wrap(MLP, -1, 2)
    model = model_cls(50, 100, 64, 2, key=jrandom.key(0))
    try:
        model(data)
    except ValueError:
        pytest.fail("Vmap wrapper is not working properly.")

def test_norm_wrapper():
    # Testing the keep_normalized kwarg
    data = jrandom.normal(jrandom.key(0), (50,))
    model_cls = norm_wrap(MLP, data, "minmax", None, data, "minmax", None)
    model = model_cls(50, 100, 64, 2, key=jrandom.key(0))
    try:
        assert not np.allclose(model(data), model(data, keep_normalized=True))
    except AssertionError:
        pytest.fail("The keep_normalized kwarg for norm wrapper is not behaving as expected.")

    # Testing minmax with knwon mins and maxs
    data = np.linspace(-1, 1, 100)
    model_cls = norm_wrap(MLP, data, "minmax", None, data, "minmax", None)
    model = model_cls(50, 100, 64, 2, key=jrandom.key(0))
    try:
        assert 0.55 == model.norm_in(0.1)
        assert -0.8 == model.inv_norm_out(0.1)
    except AssertionError:
        pytest.fail("Something wrong with minmax wrapper.")

    # Testing meanstd with knwon mean and std
    data = jrandom.normal(jrandom.key(0), (50,))
    data = (data-np.mean(data))/np.std(data)
    data = data*2.0 + 1.0  # mean of 1 and std of 2

    model_cls = norm_wrap(MLP, data, "meanstd", None, data, "meanstd", None)
    model = model_cls(50, 100, 64, 2, key=jrandom.key(0))
    try:
        assert math.isclose(2, model.norm_in(5), rel_tol=1e-4, abs_tol=1e-4)
        assert math.isclose(7, model.inv_norm_out(3), rel_tol=1e-4, abs_tol=1e-4)
    except AssertionError:
        pytest.fail("Something wrong with norm wrapper.")