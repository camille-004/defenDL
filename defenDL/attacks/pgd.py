from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defenDL.base.model import Model
from defenDL.common.types import Array
from defenDL.utils import validate_array

from .base import BaseAttack


class PGD(BaseAttack):
    """Implement the Projected Gradient Descent (PGD) attack."""

    def __init__(self, model: Model, eps: float, alpha: float, num_iter: int):
        """Initialize the PGD attack.

        Parameters
        ----------
        model : Model
            The model to attack.
        eps : float
            The perturbation magnitude.
        alpha : float
            The step size.
        num_iter : int
            Number of iterations.
        """
        self._model = model
        self._eps = eps
        self._alpha = alpha
        self._num_iter = num_iter

    def generate(self, x: Array, y: Array) -> np.ndarray:
        x = validate_array(x, "x")
        y = validate_array(y, "y")

        if x.size == 0 or y.size == 0:
            return np.array([])

        x_adv = x

        for _ in range(self._num_iter):
            gradient = self._gradient(x_adv, y)
            x_adv = x_adv + self._alpha * jnp.sign(gradient)
            x_adv = jnp.clip(x_adv, x - self._eps, x + self._eps)
            x_adv = jnp.clip(x_adv, 0, 1)

        return np.array(x_adv)

    def _gradient(self, x: Array, y: Array) -> Array:
        def loss_fn(x):
            logits = self._model(x)
            return -jnp.mean(
                jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), y]
            )

        gradient_fn = jax.grad(loss_fn)
        return gradient_fn(x)
