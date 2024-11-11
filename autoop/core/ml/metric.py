"""
metric.py

Provides the `Metric` abstract base class for
evaluation metrics in machine learning.
It includes multiple metric implementations.

Classes
-------
Metric
    Abstract base class for evaluation metrics.
MeanSquaredError
    Calculates Mean Squared Error.
RootMeanSquaredError
    Calculates Root Mean Squared Error.
RSquared
    Calculates R-squared score.
Accuracy
    Calculates accuracy.
Precision
    Calculates precision.
Recall
    Calculates recall.
"""
from abc import ABC, abstractmethod
import numpy as np


"""List of metrics."""
METRICS = [
    "mean_squared_error",
    "root_mean_squared_error",
    "r_squared",
    "accuracy",
    "precision",
    "recall",
]


def get_metric(name: str) -> "Metric":
    """
    Function to choose a metric.

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
    elif name == "root_mean_squared_error":
        return RootMeanSquaredError()
    elif name == "r_squared":
        return RSquared()
    elif name == "accuracy":
        return Accuracy()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    else:
        raise ValueError(f"Metric '{name}' is not implemented.")


class Metric(ABC):
    """Abstract Base class for all metrics."""

    @abstractmethod
    def __call__(self, x_truth: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the metric value.

        Parameters
        ----------
        y_truth : np.ndarray
            Ground truth values
        y_pred : np.ndarray
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
    predictions and actual values.
    """

    def __call__(self, y_truth: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the Mean Squared Error.

        Parameters
        ----------
        y_truth : np.ndarray
            Ground truth values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        float
            Mean squared error value
        """
        return np.mean(pow((y_truth - y_pred), 2))


class RootMeanSquaredError(Metric):
    """
    Root Mean Squared Error (RMSE) metric, calculating the square root of the
    average of squared differences between predictions and actual values.
    """

    def __call__(self, y_truth: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Error.

        Parameters
        ----------
        y_truth : np.ndarray
            Ground truth values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        float
            Root Mean Squared Error value
        """
        return np.sqrt(np.mean((y_truth - y_pred) ** 2))


class RSquared(Metric):
    """
    R-squared metric measures how much of the variation
    in the dependent variable can be attributed to the model.
    """

    def __call__(self, y_truth: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R-squared.

        Parameters
        ----------
        y_truth : np.ndarray
            Ground truth values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        float
            R-squared value, between 0 and 1,
            greater value suggesting a more precise match
        """
        ss_total = np.sum((y_truth - np.mean(y_truth)) ** 2)
        ss_residual = np.sum((y_truth - y_pred) ** 2)
        if ss_total != 0:
            return 1 - (ss_residual / ss_total)
        else:
            return 0.0


class Accuracy(Metric):
    """
    Accuracy metric,
    calculating the proportion of correct predictions
    """

    def __call__(self, y_truth: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate accuracy

        Parameters
        ----------
        y_truth : np.ndarray
            Ground truth values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        float
            Accuracy as a proportion of correct predictions
        """
        return np.mean(y_truth == y_pred)


class Precision(Metric):
    """
    Precision metric,
    calculating the proportion of true positives out of
    all predicted positives.
    """

    def __call__(self, y_truth: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate precision.

        Parameters
        ----------
        y_truth : np.ndarray
            Ground truth values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        float
            Precision as a proportion of
            true positives over predicted positives
        """
        true_positive = np.sum((y_truth == 1) & (y_pred == 1))
        predicted_positive = np.sum(y_pred == 1)
        if predicted_positive != 0:
            return true_positive / predicted_positive
        else:
            return 0.0


class Recall(Metric):
    """
    Recall metric, calculating the proportion
    of true positives out of all positives.
    """

    def __call__(self, y_truth: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate recall.

        Parameters
        ----------
        y_truth : np.ndarray
            Ground truth values
        y_pred : np.ndarray
            Predicted values

        Returns
        -------
        float
            Recall as a proportion of true positives over actual positives
        """
        true_positive = np.sum((y_truth == 1) & (y_pred == 1))
        actual_positive = np.sum(y_truth == 1)
        if actual_positive != 0:
            return true_positive / actual_positive
        else:
            return 0.0
