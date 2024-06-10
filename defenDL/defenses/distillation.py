from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from defenDL.base import Model


class Distillation:
    """Implement Defensive Distillation."""

    def __init__(
        self,
        student_model: Model,
        teacher_model: Model,
        optimizer: optax.GradientTransformation,
        temperature: float,
        rng_key: Any,
    ):
        """Initialize the Defensive Distillation.

        Parameters
        ----------
        student_model : Model
            The student model to train.
        teacher_model : Model
            The pre-trained teacher model.
        optimizer : optax.GradientTransformation
            The optimizer to use for training.
        temperature : float
            The temperature used for distillation.
        rng_key : Any
            The random number generator key.
        """
        self._student_model = student_model
        self._teacher_model = teacher_model
        self._optimizer = optimizer
        self._optimizer_state = self._optimizer.init(student_model.weights)
        self._temperature = temperature
        self._rng_key = rng_key

    def distillation_loss(
        self,
        student_logits: jnp.ndarray,
        teacher_logits: jnp.ndarray,
        temperature: float,
    ) -> jnp.ndarray:
        """Compute the distillation loss.

        Parameters
        ----------
        student_logits : jnp.ndarray
            Logits from the student model.
        teacher_logits : jnp.ndarray
            Logits from the teacher model.
        temperature : float
            The temperature used for distillation.

        Returns
        -------
        jnp.ndarray
            The distillation loss.
        """
        student_probs = jax.nn.log_softmax(student_logits / temperature)
        teacher_probs = jax.nn.softmax(teacher_logits / temperature)
        return -jnp.mean(jnp.sum(teacher_probs * student_probs, axis=1))

    def train_step(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Perform one step of the training.

        Parameters
        ----------
        x : jnp.ndarray
            The input data.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            The updated model parameters and the loss.
        """
        teacher_logits = self._teacher_model.apply(
            self._teacher_model.weights, x
        )

        def loss_fn(params):
            student_logits = self._student_model.apply(params, x)
            return self.distillation_loss(
                student_logits, teacher_logits, self._temperature
            )

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(self._student_model.weights)
        updates, self._optimizer_state = self._optimizer.update(
            grads, self._optimizer_state
        )
        self._student_model.weights = optax.apply_updates(
            self._student_model.weights, updates
        )

        return self._student_model.weights, loss

    def train(self, dataset: Any, epochs: int) -> None:
        """Train the student model using defensive distillation.

        Parameters
        ----------
        dataset : Any
            The training dataset.
        epochs : int
            The number of epochs to train for.
        """
        for epoch in range(epochs):
            for x in dataset:
                params, loss = self.train_step(x)
                print(f"Epoch {epoch}, Loss: {loss}")
