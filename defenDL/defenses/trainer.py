from __future__ import annotations

from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax

from defenDL.attacks import BaseAttack


class Model(Protocol):
    weights: jnp.ndarray

    def apply(self, params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray: ...


class Trainer:
    """Implement Adversarial Training."""

    def __init__(
        self,
        model: Model,
        optimizer: optax.GradientTransformation,
        attack: BaseAttack,
        rng_key: Any,
    ):
        """Initialize the Adversarial Training.

        Parameters
        ----------
        model : Model
            The model to train.
        optimizer : optax.GradientTransformation
            The optimizer to use for training.
        attack : BaseAttack
            The attack to generate adversarial examples.
        rng_key : Any
            The random number generator key.
        """
        self._model = model
        self._optimizer = optimizer
        self._optimizer_state = self._optimizer.init(model.weights)
        self._attack = attack
        self._rng_key = rng_key

    def train_step(
        self, x: jnp.ndarray, y: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Perform one step of the training.

        Parameters
        ----------
        x : jnp.ndarray
            The input data.
        y : jnp.ndarray
            The true labels for the input data.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            The updated model parameters and the loss.
        """
        x_np = np.array(x)
        y_np = np.array(y)

        examples = self._attack.generate(x_np, y_np)

        def loss_fn(params):
            logits = self._model.apply(params, examples)
            return -jnp.mean(
                jax.nn.log_softmax(logits)[jnp.arange(logits.shape[0]), y]
            )

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(self._model.weights)
        updates, self._optimizer_state = self._optimizer.update(
            grads, self._optimizer_state
        )
        self._model.weights = optax.apply_updates(self._model.weights, updates)

        return self._model.weights, loss

    def train(self, dataset: Any, epochs: int) -> None:
        """Traing the model with adversarial training.

        Parameters
        ----------
        dataset : Any
            The training dataset.
        epochs : int
            The number of epochs to train for.
        """
        for epoch in range(epochs):
            for x, y in dataset:
                params, loss = self.train_step(x, y)
                print(f"Epoch {epoch}, Loss: {loss}")
