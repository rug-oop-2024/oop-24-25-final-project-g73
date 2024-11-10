from abc import ABC, abstractmethod
# from autoop.core.ml.artifact import Artifact
import numpy as np
# from copy import deepcopy
# from typing import Literal


class Model(ABC):
    """Base model from which all other models will inherit."""

    def __init__(self):
        """Initialize the base model."""
        self._parameters: dict = {}

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Trains the model based on observations and ground thruth.

        Parameters
        ----------
            observations : np.ndarray
                data used to train the model
            ground_truth : np.ndarray
                labels or values of each of the observations

        Returns
        -------
            None
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Do predictions based on observations.

        Parameters
        ----------
            observations : np.ndarray
                data used to make a prediction

        Returns
        -------
            labels or values predicted for each of the observations
        """
        pass
