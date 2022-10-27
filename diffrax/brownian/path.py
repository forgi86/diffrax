from typing import Tuple

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
import jax.random as jrandom

from ..custom_types import Array, Scalar
from ..misc import force_bitcast_convert_type
from .base import AbstractBrownianPath


class UnsafeBrownianPath(AbstractBrownianPath):
    """Brownian simulation that is only suitable for certain cases.

    This is a very quick way to simulate Brownian motion, but can only be used when all
    of the following are true:

    1. You are using a fixed step size controller. (Not an adaptive one.)

    2. You do not need to backpropagate through the differential equation.

    3. You do not need deterministic solutions with respect to `key`. (This
       implementation will produce different results based on fluctuations in
       floating-point arithmetic.)

    Internally this operates by just sampling a fresh normal random variable over every
    interval, ignoring the correlation between samples exhibited in true Brownian
    motion. Hence the restrictions above. (They describe the general case for which the
    correlation structure isn't needed.)
    """

    shape: Tuple[int] = eqx.static_field()
    # Handled as a string because PRNGKey is actually a function, not a class, which
    # makes it appearly badly in autogenerated documentation.
    key: "jax.random.PRNGKey"  # noqa: F821

    @property
    def t0(self):
        return None

    @property
    def t1(self):
        return None

    @eqx.filter_jit
    def evaluate(self, t0: Scalar, t1: Scalar, left: bool = True) -> Array:
        del left
        t0 = eqxi.nondifferentiable(t0)
        t1 = eqxi.nondifferentiable(t1)
        t0_ = force_bitcast_convert_type(t0, jnp.int32)
        t1_ = force_bitcast_convert_type(t1, jnp.int32)
        key = jrandom.fold_in(self.key, t0_)
        key = jrandom.fold_in(key, t1_)
        return jrandom.normal(key, self.shape) * jnp.sqrt(t1 - t0)


UnsafeBrownianPath.__init__.__doc__ = """
**Arguments:**

- `shape`: What shape each individual Brownian sample should be.
- `key`: A random key.
"""
