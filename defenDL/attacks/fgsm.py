from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defenDL.base.model import Model
from defenDL.common.types import Array
from defenDL.utils import validate_array

from .base import BaseAttack


class FGSM(BaseAttack):
    """Implement the Fast Gradient Sign Method (FGSM)."""

    def __init__(self, model: Model, eps: float):
        """Initialize the FGSM attack.

        Parameters
        ----------
        model : Model
            The model to attack.
        eps : float
            The perturbation magnitude.
        """
        self._model = model
        self._eps = eps

    def generate(self, x: Array, y: Array) -> np.ndarray:
        x = validate_array(x, "x")
        y = validate_array(y, "y")

        if x.size == 0 or y.size == 0:
            return np.array([])

        # Compute gradients of the loss w.r.t. input.
        gradient = self._gradient(x, y)

        # Add the sign of the gradient to generate examples.
        x_adv = x + self._eps * jnp.sign(gradient)

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
