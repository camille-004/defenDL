from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp
import numpy as np
import pytest

from defenDL.attacks import CW
from defenDL.base import Model


@dataclass
class DummyModel(Model):
    weights: jnp.ndarray = field(
        default_factory=lambda: jnp.array([[0.1, 0.2], [0.3, 0.4]])
    )

    def apply(self, params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        return jnp.dot(x, params)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.apply(self.weights, x)


class TestFGSM:
    @pytest.fixture
    def model(self) -> DummyModel:
        return DummyModel()

    @pytest.fixture
    def cw(self, model: Model) -> CW:
        confidence = 0.0
        learning_rate = 0.01
        binary_search_steps = 9
        max_iter = 1000
        initial_const = 0.01
        return CW(
            model,
            confidence,
            learning_rate,
            binary_search_steps,
            max_iter,
            initial_const,
        )

    def test_fgsm_generate(self, cw: CW) -> None:
        x = np.array([[0.5, 0.5], [0.1, 0.9]])
        y = np.array([1, 0])

        x_adv = cw.generate(x, y)

        assert (
            x_adv.shape == x.shape
        ), "The shape of the adversarial examples should match the input shape."  # noqa
        assert (x_adv >= 0).all() and (
            x_adv <= 1
        ).all(), "Adversarial examples should be within the data range [0, 1]."  # noqa

    def test_fgsm_empty_input(self, cw: CW) -> None:
        x = np.array([])
        y = np.array([])

        x_adv = cw.generate(x, y)

        assert (
            x_adv.size == 0
        ), "Adversarial examples generated from empty input should also be empty."  # noqa

    def test_fgsm_single_data_point(self, cw: CW) -> None:
        x = np.array([[0.5, 0.5]])
        y = np.array([1])

        x_adv = cw.generate(x, y)

        assert (
            x_adv.shape == x.shape
        ), "The shape of adversarial inputs should match the input shape."
        assert (x_adv >= 0).all() and (
            x_adv <= 1
        ).all(), "Adversarial examples should be within the data range [0, 1]."

    @pytest.mark.parametrize(
        "confidence, learning_rate, binary_search_steps, max_iter, initial_const",  # noqa
        [
            (0.0, 0.01, 9, 1000, 0.01),
            (0.5, 0.02, 10, 500, 0.1),
            (1.0, 0.05, 5, 200, 0.2),
        ],
    )
    def test_cw_different_params(
        self,
        model: Model,
        confidence: float,
        learning_rate: float,
        binary_search_steps: int,
        max_iter: int,
        initial_const: float,
    ) -> None:
        cw = CW(
            model,
            confidence,
            learning_rate,
            binary_search_steps,
            max_iter,
            initial_const,
        )

        x = np.array([[0.5, 0.5], [0.1, 0.9]])
        y = np.array([1, 0])

        x_adv = cw.generate(x, y)

        assert (
            x_adv.shape == x.shape
        ), "The shape of adversarial examples should match the input shape."
        assert (x_adv >= 0).all() and (
            x_adv <= 1
        ).all(), "Adversarial examples should be within the data range [0, 1]."
