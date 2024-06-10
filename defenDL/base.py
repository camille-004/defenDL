from typing import Protocol

import jax.numpy as jnp


class Model(Protocol):
    weights: jnp.ndarray

    def apply(self, params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray: ...
