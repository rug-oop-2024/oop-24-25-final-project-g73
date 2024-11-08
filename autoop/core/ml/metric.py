from abc import ABC, abstractmethod
from typing import Any
import numpy as np


METRICS = [
    "mean_squared_error",
    "accuracy",
] # add the names (in strings) of the metrics you implement


def get_metric(name: str):
    """
    Function to choose a metric

    Parameters
    ----------
    name : str
        Name of the metric

    Returns
    -------
    Metric
        The Metric class

    Raises
    ------
    ValueError
        If the metric name is not recognized
    """
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    else:
        raise ValueError(f"Metric '{name}' is not implemented.")


class Metric(ABC):
    """
    Abstract Base class for all metrics
    """

    @abstractmethod
    def __call__(self, x_truth: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the metric value

        Parameters
        ----------
        y_truth : array-like
            Ground truth values
        y_pred : array-like
            Predicted values

        Returns
        -------
        float
            The calculated metric value
        """
        pass


class MeanSquaredError(Metric):
    """
    Mean Squared Error metric,
    measuring the average squared difference between
    predictions and actual values
    """

    def __call__(self, y_truth: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates the Mean Squared Error

        Parameters
        ----------
        y_truth : array-like
            Ground truth values
        y_pred : array-like
            Predicted values

        Returns
        -------
        float
            Mean squared error value
        """
        y_truth = np.ndarray(y_truth)
        y_pred = np.ndarray(y_pred)
        return np.mean((y_truth - y_pred) ^ 2)


class Accuracy(Metric):
    """
    Accuracy metric,
    calculating the proportion of correct predictions
    """

    def __call__(self, y_truth: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates accuracy

        Parameters
        ----------
        y_truth : array-like
            Ground truth values
        y_pred : array-like
            Predicted values

        Returns
        -------
        float
            Accuracy as a proportion of correct predictions
        """
        y_truth = np.ndarray(y_truth)
        y_pred = np.ndarray(y_pred)
        return np.mean(y_truth == y_pred)
