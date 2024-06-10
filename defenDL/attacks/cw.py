from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from defenDL.base import Model

from .base import BaseAttack


class CW(BaseAttack):
    """Implement the Carlini-Wagner (CW) attack."""

    def __init__(
        self,
        model: Model,
        confidence: float,
        learning_rate: float,
        binary_search_steps: int,
        max_iter: int,
        initial_const: float,
    ):
        """Initialize the CW attack.

        Parameters
        ----------
        model : Model
            The model to attack.
        confidence : float
            Confidence of adversarial examples.
        learning_rate : float
            Learning rate.
        binary_search_steps : int
            Number of binary search steps to find the best constant.
        max_iter : int
            Maximum number of iterations for optimization.
        initial_const : float
            Initial value of the constant for optimization.
        """
        super().__init__(model)
        self._confidence = confidence
        self._learning_rate = learning_rate
        self._binary_search_steps = binary_search_steps
        self._max_iter = max_iter
        self._initial_const = initial_const

    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.size == 0 or y.size == 0:
            return np.array([])

        x_jax = jnp.array(x)
        y_jax = jnp.array(y)

        def loss_fn(delta: jnp.ndarray, c: float):
            adv_x = x_jax + delta
            logits = self._model(adv_x)
            real = logits[jnp.arange(logits.shape[0]), y_jax]
            other = jnp.max(
                logits * (1 - jax.nn.one_hot(y_jax, logits.shape[1])), axis=1
            )
            return c * jnp.sum(
                jnp.maximum(0, other - real + self._confidence)
            ) + jnp.sum(jnp.square(delta))

        def optimize(delta: jnp.ndarray, c: float) -> jnp.ndarray:
            optimizer = optax.sgd(self._learning_rate)
            opt_state = optimizer.init(delta)

            def step(i, opt_state, delta):
                loss, grads = jax.value_and_grad(loss_fn)(delta, c)
                updates, opt_state = optimizer.update(grads, opt_state)
                delta = optax.apply_updates(delta, updates)
                return opt_state, delta

            for i in range(self._max_iter):
                opt_state, delta = step(i, opt_state, delta)

            return delta

        # Binary search over `c`
        lower_bound = 0.0
        upper_bound = 1e10
        best_delta = jnp.zeros_like(x_jax)
        best_loss = float("inf")

        for _ in range(self._binary_search_steps):
            c = (lower_bound + upper_bound) / 2
            delta = jnp.zeros_like(x_jax)
            delta = optimize(delta, c)
            loss = loss_fn(delta, c)

            if loss < best_loss:
                best_loss = loss
                best_delta = delta

            if jnp.any(delta != 0):
                upper_bound = c
            else:
                lower_bound = c

        x_adv = x_jax + best_delta
        x_adv = jnp.clip(x_adv, 0, 1)

        return np.array(x_adv)

    def _gradient(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError("CW attack does not use _gradient method.")
