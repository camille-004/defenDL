from typing import Protocol

import jax.numpy as jnp

from defenDL.common.types import Array


class Model(Protocol):
    weights: jnp.ndarray

    def apply(self, params: Array, x: Array) -> Array: ...
