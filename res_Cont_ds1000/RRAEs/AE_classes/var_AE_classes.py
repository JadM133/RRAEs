import jax
import jax.numpy as jnp
import pdb
from RRAEs.utilities import Func, Linear
import equinox as eqx
import jax.random as jrandom
from equinox._doc_utils import doc_repr
import warnings

_identity = doc_repr(lambda x, **kwargs: x, "lambda x: x")

