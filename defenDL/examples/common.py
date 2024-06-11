from dataclasses import dataclass, field

import jax.numpy as jnp

from defenDL.base.model import Model
from defenDL.common.types import Array


@dataclass
class SimpleModel(Model):
    weights: jnp.ndarray = field(
        default_factory=lambda: jnp.array([[0.1, 0.2], [0.3, 0.4]])
    )

    def apply(self, params: Array, x: Array) -> Array:
        return jnp.dot(x, params)

    def __call__(self, x: Array) -> Array:
        return self.apply(self.weights, x)
