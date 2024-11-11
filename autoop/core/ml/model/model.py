"""
model.py

This module defines the abstract `Model` base class
for machine learning models, providing a common interface
for training and prediction that other models inherit.

Classes
-------
Model
    Abstract base class for all models,
    defining the methods `fit` and `predict`.
"""
from abc import ABC, abstractmethod
# from autoop.core.ml.artifact import Artifact
import numpy as np
# from copy import deepcopy
# from typing import Literal


class Model(ABC):
    """Base model from which all other models will inherit."""

    def __init__(self) -> None:
        """Initialize the base model."""
        self._parameters: dict = {}

    @abstractmethod
    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Trains the model based on observations and ground thruth.

        Parameters
        ----------
        observations : np.ndarray
            Data used to train the model.
        ground_truth : np.ndarray
            Labels or values of each of the observations.

        Returns
        -------
            None
        """
        pass

    @abstractmethod
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions based on observations.

        Parameters
        ----------
        observations : np.ndarray
            Data used to make predictions.

        Returns
        -------
        np.ndarray
            Predicted labels or values for each of the observations.
        """
        pass
