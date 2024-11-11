"""
classification_models.py

This module provides classification models,
implementing the `fit` and `predict` methods.

Classes
-------
LogisticRegressionModel
    Logistic Regression model using scikit-learn's `LogisticRegression`.
DecisionTreeClassifierModel
    Decision Tree Classifier model using
    scikit-learn's `DecisionTreeClassifier`.
SVMClassifierModel
    Support Vector Classifier model using scikit-learn's `SVC`.
"""
from autoop.core.ml.model import Model
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


class LogisticRegressionModel(Model):
    """
    Logistic Regression model for binary or multi-class classification tasks.

    This model uses scikit-learn's `LogisticRegression` to train
    on labeled data and make predictions based on learned relationships.
    """

    def __init__(self) -> None:
        """Initialize Logistic Regression model."""
        self.model = LogisticRegression()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the Logistic Regression model.

        Parameters
        ----------
        observations : np.ndarray
            Feature data used for training.
        ground_truth : np.ndarray
            Target data for training.

        Returns
        -------
        None
        """
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the Logistic Regression model.

        Parameters
        ----------
        observations : np.ndarray
            Feature data used for making predictions.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        return self.model.predict(observations)


class DecisionTreeClassifierModel(Model):
    """
    Decision Tree Classifier for binary or multi-class classification tasks.

    This model uses scikit-learn's `DecisionTreeClassifier`, which applies
    a tree structure to classify data by learning decision rules
    from labeled training data.
    """

    def __init__(self) -> None:
        """Initialize Decision Tree Classifier model."""
        self.model = DecisionTreeClassifier()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the Decision Tree Classifier model.

        Parameters
        ----------
        observations : np.ndarray
            Feature data used for training.
        ground_truth : np.ndarray
            Target data for training.

        Returns
        -------
        None
        """
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the Decision Tree Classifier model.

        Parameters
        ----------
        observations : np.ndarray
            Feature data used for making predictions.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        return self.model.predict(observations)


class SVMClassifierModel(Model):
    """
    Support Vector Classifier for binary or multi-class classification tasks.

    This model uses scikit-learn's `SVC` to find the optimal hyperplane
    that maximizes the margin between classes,
    making it suitable for both linear and non-linear classification problems.
    """

    def __init__(self) -> None:
        """Initialize SVC model."""
        self.model = SVC()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the Support Vector Classifier model.

        Parameters
        ----------
        observations : np.ndarray
            Feature data used for training.
        ground_truth : np.ndarray
            Target data for training.

        Returns
        -------
        None
        """
        self.model.fit(observations, ground_truth)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the Support Vector Classifier model.

        Parameters
        ----------
        observations : np.ndarray
            Feature data used for making predictions.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        return self.model.predict(observations)
