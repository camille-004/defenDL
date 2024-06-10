from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import optax
import pytest
from jax import random

from defenDL.base import Model
from defenDL.defenses import Distillation


@dataclass
class DummyModel(Model):
    weights: jnp.ndarray = field(
        default_factory=lambda: jnp.array([[0.1, 0.2], [0.3, 0.4]])
    )

    def apply(self, params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, params)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.apply(self.weights, x)


class TestDistillation:
    @pytest.fixture
    def student_model(self) -> DummyModel:
        return DummyModel()

    @pytest.fixture
    def teacher_model(self) -> DummyModel:
        return DummyModel()

    @pytest.fixture
    def optimizer(self) -> optax.GradientTransformation:
        return optax.sgd(learning_rate=0.01)

    @pytest.fixture
    def rng_key(self) -> Any:
        return random.PRNGKey(0)

    @pytest.fixture
    def distillation(
        self,
        student_model: Model,
        teacher_model: Model,
        optimizer: optax.GradientTransformation,
        rng_key: Any,
    ) -> Distillation:
        temperature = 1.0
        return Distillation(
            student_model, teacher_model, optimizer, temperature, rng_key
        )

    def test_distillation_loss(self, distillation: Distillation):
        student_logits = jnp.array([[2.0, 1.0], [0.5, 0.5]])
        teacher_logits = jnp.array([[1.5, 1.5], [1.0, 0.0]])
        temperature = 1.0

        loss = distillation.distillation_loss(
            student_logits, teacher_logits, temperature
        )

        assert loss > 0, "Distillation loss should be greater than 0."

    def test_train_step(self, distillation: Distillation) -> None:
        x = jnp.array([[0.5, 0.5], [0.1, 0.9]])

        params, loss = distillation.train_step(x)

        assert (
            params is not None
        ), "Parameters should not be None after training step."
        assert loss is not None, "Loss should not be None after training step."

    def test_train(self, distillation: Distillation) -> None:
        dataset = [jnp.array([[0.5, 0.5], [0.1, 0.9]])]

        distillation.train(dataset, epochs=1)
