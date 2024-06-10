from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defenDL.attacks import BaseAttack
from defenDL.base import Model


def accuracy(predictions: jnp.ndarray, labels: jnp.ndarray) -> float:
    """Compute accuracy.

    Parameters
    ----------
    predictions : jnp.ndarray
        The predicted labels.
    labels : jnp.ndarray
        The true labels.

    Returns
    -------
    float
        The accuracy score.
    """
    return jnp.mean(predictions == labels).item()


def eval_model(
    model: Model, dataset: tuple[jnp.ndarray, jnp.ndarray]
) -> float:
    """Evaluate the model on a dataset.

    Parameters
    ----------
    model : Model
        The model to evaluate.
    dataset : tuple[jnp.ndarray, jnp.ndarray]
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
    model: Model, attack: BaseAttack, dataset: tuple[jnp.ndarray, jnp.ndarray]
) -> float:
    """Evaluate the robustness of a model against an attack.

    Parameters
    ----------
    model : Model
        The model to evaluate.
    attack : BaseAttack
        The adversarial attack to evaluate against.
    dataset : tuple[jnp.ndarray, jnp.ndarray]
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
