from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from defenDL.attacks import BaseAttack
from defenDL.base.model import Model
from defenDL.common.types import Array
from defenDL.utils import validate_array


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
        self, x: Array, y: Array
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Perform one step of the training.

        Parameters
        ----------
        x : Array
            The input data.
        y : Array
            The true labels for the input data.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            The updated model parameters and the loss.
        """
        x = validate_array(x, "x")
        y = validate_array(y, "y")

        examples = self._attack.generate(x, y)

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

    def train(self, dataset: tuple[Array, Array], epochs: int) -> None:
        """Traing the model with adversarial training.

        Parameters
        ----------
        dataset : tuple[Array, Array]
            The training dataset.
        epochs : int
            The number of epochs to train for.
        """
        inputs, labels = dataset

        inputs = validate_array(inputs, "inputs")
        labels = validate_array(labels, "labels")

        for epoch in range(epochs):
            params, loss = self.train_step(inputs, labels)
            print(f"Epoch {epoch}, Loss: {loss}")
