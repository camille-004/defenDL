from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from defenDL.attacks import PGD
from defenDL.base.model import Model
from tests.common import DummyModel


class TestPGD:
    @pytest.fixture
    def model(self) -> Model:
        return DummyModel()

    @pytest.fixture
    def pgd(self, model: Model) -> PGD:
        eps = 0.1
        alpha = 0.01
        num_iter = 40
        return PGD(model, eps, alpha, num_iter)

    def test_pgd_generate(self, pgd: PGD) -> None:
        x = np.array([[0.5, 0.5], [0.1, 0.9]])
        y = np.array([1, 0])

        x_adv = pgd.generate(x, y)

        assert (
            x_adv.shape == x.shape
        ), "The shape of the adversarial examples should match the input shape."  # noqa
        assert (x_adv >= 0).all() and (
            x_adv <= 1
        ).all(), "Adversarial examples should be within the data range [0, 1]."

    def test_pgd_gradient(self, pgd: PGD) -> None:
        x = jnp.array([[0.5, 0.5], [0.1, 0.9]])
        y = jnp.array([1, 0])

        gradient = pgd._gradient(x, y)

        assert (
            gradient.shape == x.shape
        ), "The shape of the gradient should match the input shape."

    def test_pgd_empty_input(self, pgd: PGD) -> None:
        x = np.array([])
        y = np.array([])

        x_adv = pgd.generate(x, y)

        assert (
            x_adv.size == 0
        ), "Adversarial example generated from empty input should also be empty."  # noqa

    def test_pgd_single_data_point(self, pgd: PGD) -> None:
        x = np.array([[0.5, 0.5]])
        y = np.array([1])

        x_adv = pgd.generate(x, y)

        assert (
            x_adv.shape == x.shape
        ), "The shape of adversarial inputs should match the input shape."
        assert (x_adv >= 0).all() and (
            x_adv <= 1
        ).all(), "Adversarial examples should be within the data range [0, 1]."

    @pytest.mark.parametrize(
        "eps, alpha, num_iter",
        [(0.1, 0.01, 40), (0.2, 0.02, 30), (0.3, 0.03, 20)],
    )
    def test_pgd_different_params(
        self, model: Model, eps: float, alpha: float, num_iter: int
    ) -> None:
        pgd = PGD(model, eps, alpha, num_iter)

        x = np.array([[0.5, 0.5], [0.1, 0.9]])
        y = np.array([1, 0])

        x_adv = pgd.generate(x, y)

        assert (
            x_adv.shape == x.shape
        ), "The shape of adversarial examples should match the input shape."
        assert (x_adv >= 0).all() and (
            x_adv <= 1
        ).all(), "Adversarial examples should be within the data range [0, 1]."
