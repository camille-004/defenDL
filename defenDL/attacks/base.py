from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np


class BaseAttack(ABC):
    """Abstract base class for adversarial attacks."""

    @abstractmethod
    def generate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate adversarial examples.

        Parameters
        ----------
        x : np.ndarray
            The input data.
        y : np.ndarray
            The true labels for the input data.

        Returns
        -------
        np.ndarray
            The adversarial examples.
        """
        pass

    @abstractmethod
    def _gradient(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Computes the gradient of the loss w.r.t. input data.

        Parameters
        ----------
        x : jnp.ndarray
            The input data.
        y : jnp.ndarray
            The true labels for the input data.

        Returns
        -------
        jnp.ndarray
            The computed gradients.
        """
        pass
