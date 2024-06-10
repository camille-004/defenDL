from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import pytest

from defenDL.attacks import FGSM, BaseAttack
from defenDL.base import Model
from defenDL.metrics import accuracy, eval_model, eval_robustness


@dataclass
class DummyModel(Model):
    weights: jnp.ndarray = field(
        default_factory=lambda: jnp.array([[0.1, 0.2], [0.3, 0.4]])
    )

    def apply(self, params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, params)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.apply(self.weights, x)


class TestMetric:
    @pytest.fixture
    def model(self) -> Model:
        return DummyModel()

    @pytest.fixture
    def attack(self, model: Model) -> BaseAttack:
        return FGSM(model, eps=0.1)

    def test_accuracy(self) -> None:
        predictions = jnp.array([1, 0, 1, 1, 0])
        labels = jnp.array([1, 0, 0, 1, 0])

        acc = accuracy(predictions, labels)

        assert acc == pytest.approx(
            0.8, rel=1e-5
        ), f"Expected accuracy to be 0.8, but got {acc}"

    def test_eval_model(self, model: Model) -> None:
        dataset = (jnp.array([[0.5, 0.5], [0.1, 0.9]]), jnp.array([1, 0]))

        acc = eval_model(model, dataset)

        assert acc == 0.5, f"Expected accuracy to be 0.5, but got {acc}"

    def test_eval_robustness(self, model: Model, attack: BaseAttack) -> None:
        dataset = (jnp.array([[0.5, 0.5], [0.1, 0.9]]), jnp.array([1, 0]))

        acc = eval_robustness(model, attack, dataset)

        assert acc == 0.5, f"Expected accuracy to be 0.5, but got {acc}"
