import jax.numpy as jnp
import numpy as np

from defenDL.common.types import Array


def validate_array(array: Array, name: str) -> Array:
    if array is None:
        raise ValueError(f"{name} cannot be None")

    if not isinstance(array, (np.ndarray, jnp.ndarray)):
        raise TypeError(f"Unsupposed type for {name}")

    return array


def array_shape_flatten(array: Array) -> tuple[tuple[int, ...], np.ndarray]:
    array = validate_array(array, "array")
    return array.shape, np.array(array.flatten())
