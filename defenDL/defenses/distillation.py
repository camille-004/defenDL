from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import optax

from defenDL.base.model import Model
from defenDL.common.types import Array
from defenDL.utils import validate_array


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
        student_logits: Array,
        teacher_logits: Array,
        temperature: float,
    ) -> jnp.ndarray:
        """Compute the distillation loss.

        Parameters
        ----------
        student_logits : Array
            Logits from the student model.
        teacher_logits : Array
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
        loss = -jnp.mean(jnp.sum(teacher_probs * student_probs, axis=1))

        print(
            f"Distillation loss computation: student_probs shape: {student_probs.shape}, teacher_probs shape: {teacher_probs.shape}, loss: {loss.item()}"  # noqa
        )
        return loss

    def train_step(self, x: Array) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Perform one step of the training.

        Parameters
        ----------
        x : Array
            The input data.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            The updated model parameters and the loss.
        """
        teacher_logits = self._teacher_model.apply(
            self._teacher_model.weights, x
        )
        print(
            f"Teacher logits shape: {teacher_logits.shape}, first elements: {teacher_logits.flatten()[:4]}"  # noqa
        )

        def loss_fn(params):
            student_logits = self._student_model.apply(params, x)
            loss = self.distillation_loss(
                student_logits, teacher_logits, self._temperature
            )
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(self._student_model.weights)
        updates, self._optimizer_state = self._optimizer.update(
            grads, self._optimizer_state
        )
        self._student_model.weights = optax.apply_updates(
            self._student_model.weights, updates
        )

        student_logits = self._student_model.apply(
            self._student_model.weights, x
        )
        print(
            f"Student logits shape: {student_logits.shape}, first elements: {jax.device_get(student_logits.flatten()[:4])}"  # noqa
        )
        print(f"Loss: {loss}")
        print(f"Gradients: {jax.tree.map(jax.device_get, grads)}")
        print(f"Updates: {jax.tree.map(jax.device_get, updates)}")
        print(
            f"Updated weights: {jax.device_get(self._student_model.weights)}"
        )

        return self._student_model.weights, loss

    def train(self, dataset: tuple[Array, Array], epochs: int) -> None:
        """Train the student model using defensive distillation.

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
            params, loss = self.train_step(inputs)
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            print(f"Updated parameters: {params[:4]}")
