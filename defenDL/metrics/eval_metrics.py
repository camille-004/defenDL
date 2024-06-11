from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defenDL.attacks import BaseAttack
from defenDL.base.model import Model
from defenDL.common.types import Array


def accuracy(predictions: Array, labels: Array) -> float:
    """Compute accuracy.

    Parameters
    ----------
    predictions : Array
        The predicted labels.
    labels : Array
        The true labels.

    Returns
    -------
    float
        The accuracy score.
    """
    return jnp.mean(predictions == labels).item()


def eval_model(model: Model, dataset: tuple[Array, Array]) -> float:
    """Evaluate the model on a dataset.

    Parameters
    ----------
    model : Model
        The model to evaluate.
    dataset : tuple[Array, Array]
        The dataset to evaluate on (inputs, labels).

    Returns
    -------
    float
        The accuracy score.
    """
    inputs, labels = dataset
    predictions = model.apply(model.weights, inputs)
    predicted_labels = jnp.argmax(predictions, axis=1)
    return accuracy(predicted_labels, labels)


def eval_robustness(
    model: Model, attack: BaseAttack, dataset: tuple[Array, Array]
) -> float:
    """Evaluate the robustness of a model against an attack.

    Parameters
    ----------
    model : Model
        The model to evaluate.
    attack : BaseAttack
        The adversarial attack to evaluate against.
    dataset : tuple[Array, Array]
        The dataset to evaluate on (inputs, labels).

    Returns
    -------
    float
        The accuracy score on adversarial examples.
    """
    inputs, labels = dataset
    examples = attack.generate(np.array(inputs), np.array(labels))
    adv_predictions = model.apply(model.weights, jnp.array(examples))
    adv_predicted_labels = jnp.argmax(adv_predictions, axis=1)
    return accuracy(adv_predicted_labels, labels)
