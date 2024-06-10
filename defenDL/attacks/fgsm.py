from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .base import Attack


class FGSM(Attack):
    """Implement the Fast Gradient Sign Method (FGSM)."""

    def __init__(
        self, model: Callable[[jnp.ndarray], jnp.ndarray], eps: float
    ):
        """Initialize the FGSM attack.

        Parameters
        ----------
        model : Callable[[np.ndarray], np.ndarray]
            The model to attack.
        eps : float
            The perturbation magnitude.
        """
        self._model = model
        self._eps = eps

    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.size == 0 or y.size == 0:
            return np.array([])

        x_jax = jnp.array(x)
        y_jax = jnp.array(y)

        # Compute gradients of the loss w.r.t. input.
        gradient = self._gradient(x_jax, y_jax)

        # Add the sign of the gradient to generate examples.
        x_adv = x_jax + self._eps * jnp.sign(gradient)

        x_adv = jnp.clip(x_adv, 0, 1)

        return np.array(x_adv)

    def _gradient(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        def loss_fn(x):
            logits = self._model(x)
            return -jnp.mean(
                jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), y]
            )

        gradient_fn = jax.grad(loss_fn)
        return gradient_fn(x)
