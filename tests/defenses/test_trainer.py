from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import optax
import pytest
from jax import random

from defenDL.attacks import CW, FGSM, PGD, BaseAttack
from defenDL.defenses import Trainer
from tests.common import DummyModel


class TestTrainer:
    @pytest.fixture
    def model(self) -> DummyModel:
        return DummyModel()

    @pytest.fixture
    def optimizer(self) -> optax.GradientTransformation:
        return optax.sgd(learning_rate=0.01)

    @pytest.fixture
    def attack(self, model: DummyModel) -> FGSM:
        return FGSM(model, eps=0.1)

    @pytest.fixture
    def rng_key(self) -> Any:
        return random.PRNGKey(0)

    @pytest.fixture
    def trainer(
        self,
        model: DummyModel,
        optimizer: optax.GradientTransformation,
        attack: FGSM,
        rng_key: Any,
    ) -> Trainer:
        return Trainer(model, optimizer, attack, rng_key)

    @pytest.mark.parametrize(
        "attack_class, attack_params",
        [
            (FGSM, {"eps": 0.1}),
            (PGD, {"eps": 0.1, "alpha": 0.01, "num_iter": 40}),
            (
                CW,
                {
                    "confidence": 0.0,
                    "learning_rate": 0.01,
                    "binary_search_steps": 9,
                    "max_iter": 1000,
                    "initial_const": 0.01,
                },
            ),
        ],
    )
    def test_training_step(
        self,
        trainer: Trainer,
        attack_class: type[BaseAttack],
        attack_params: dict[str, Any],
    ) -> None:
        model = DummyModel()
        attack = attack_class(model, **attack_params)
        trainer = Trainer(
            model, optax.sgd(learning_rate=0.01), attack, random.PRNGKey(0)
        )

        x = jnp.array([[0.5, 0.5], [0.1, 0.9]])
        y = jnp.array([1, 0])

        params, loss = trainer.train_step(x, y)

        assert (
            params is not None
        ), "Parameters should not be None after training step."
        assert loss is not None, "Loss should not be None after training step."

    @pytest.mark.parametrize(
        "attack_class, attack_params",
        [
            (FGSM, {"eps": 0.1}),
            (PGD, {"eps": 0.1, "alpha": 0.01, "num_iter": 40}),
            (
                CW,
                {
                    "confidence": 0.0,
                    "learning_rate": 0.01,
                    "binary_search_steps": 9,
                    "max_iter": 1000,
                    "initial_const": 0.01,
                },
            ),
        ],
    )
    def test_training(
        self, attack_class: type[BaseAttack], attack_params: dict[str, Any]
    ) -> None:
        model = DummyModel()
        attack = attack_class(model, **attack_params)
        trainer = Trainer(
            model, optax.sgd(learning_rate=0.01), attack, random.PRNGKey(0)
        )

        dataset = (jnp.array([[0.5, 0.5], [0.1, 0.9]]), jnp.array([1, 0]))

        trainer.train(dataset, epochs=1)
