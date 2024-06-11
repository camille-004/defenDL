from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from defenDL.attacks import FGSM
from defenDL.base.model import Model
from tests.common import DummyModel


class TestFGSM:
    @pytest.fixture
    def model(self) -> DummyModel:
        return DummyModel()

    @pytest.fixture
    def fgsm(self, model: Model) -> FGSM:
        eps = 0.1
        return FGSM(model, eps)

    def test_fgsm_generate(self, fgsm: FGSM) -> None:
        x = np.array([[0.5, 0.5], [0.1, 0.9]])
        y = np.array([1, 0])

        x_adv = fgsm.generate(x, y)

        assert (
            x_adv.shape == x.shape
        ), "The shape of the adversarial examples should match the input shape."  # noqa
        assert (x_adv >= 0).all() and (
            x_adv <= 1
        ).all(), "Adversarial examples should be within the data range [0, 1]."  # noqa

    def test_fgsm_gradient(self, fgsm: FGSM) -> None:
        x = jnp.array([[0.5, 0.5], [0.1, 0.9]])
        y = jnp.array([1, 0])

        gradient = fgsm._gradient(x, y)

        assert (
            gradient.shape == x.shape
        ), "The shape of the gradient should match the input shape."

    def test_fgsm_empty_input(self, fgsm: FGSM) -> None:
        x = np.array([])
        y = np.array([])

        x_adv = fgsm.generate(x, y)

        assert (
            x_adv.size == 0
        ), "Adversarial examples generated from empty input should also be empty."  # noqa

    def test_fgsm_single_data_point(self, fgsm: FGSM) -> None:
        x = np.array([[0.5, 0.5]])
        y = np.array([1])

        x_adv = fgsm.generate(x, y)

        assert (
            x_adv.shape == x.shape
        ), "The shape of adversarial inputs should match the input shape."
        assert (x_adv >= 0).all() and (
            x_adv <= 1
        ).all(), "Adversarial examples should be within the data range [0, 1]."

    @pytest.mark.parametrize("eps", [0.0, 0.05, 0.1, 0.2])
    def test_fgsm_different_eps(self, model: Model, eps: float) -> None:
        fgsm = FGSM(model, eps)

        x = np.array([[0.5, 0.5], [0.1, 0.9]])
        y = np.array([1, 0])

        x_adv = fgsm.generate(x, y)

        assert (
            x_adv.shape == x.shape
        ), "The shape of adversarial examples should match the input shape."
        assert (x_adv >= 0).all() and (
            x_adv <= 1
        ).all(), "Adversarial examples should be within the data range [0, 1]."
