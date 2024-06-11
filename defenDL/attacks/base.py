from abc import ABC, abstractmethod

from defenDL.base.model import Model
from defenDL.common.types import Array


class BaseAttack(ABC):
    """Abstract base class for adversarial attacks."""

    def __init__(self, model: Model):
        self._model = model

    @abstractmethod
    def generate(self, x: Array, y: Array) -> Array:
        """Generate adversarial examples.

        Parameters
        ----------
        x : Array
            The input data.
        y : Array
            The true labels for the input data.

        Returns
        -------
        Array
            The adversarial examples.
        """
        pass

    @abstractmethod
    def _gradient(self, x: Array, y: Array) -> Array:
        """Computes the gradient of the loss w.r.t. input data.

        Parameters
        ----------
        x : Array
            The input data.
        y : Array
            The true labels for the input data.

        Returns
        -------
        Array
            The computed gradients.
        """
        pass
