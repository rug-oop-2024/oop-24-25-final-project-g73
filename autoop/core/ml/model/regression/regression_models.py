"""
regression_models.py

This module provides regression models,
implementing the `fit` and `predict` methods.

Classes
-------
LinearRegressionModel
    Linear Regression model using scikit-learn's `LinearRegression`.
DecisionTreeRegressorModel
    Decision Tree Regressor model using scikit-learn's `DecisionTreeRegressor`.
SVMRegressorModel
    Support Vector Regressor model using scikit-learn's `SVR`.
"""
from autoop.core.ml.model import Model
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


class LinearRegressionModel(Model):
    """
    Linear Regression model for predicting continuous target variables.

    This model uses scikit-learn's `LinearRegression` to fit a
    linear relationship between input features and a continuous target,
    making it suitable for simple or multiple regression tasks.
    """

    def __init__(self) -> None:
        """Initialize Linear Regression model."""
        self.model = LinearRegression()
        self.parameters = None

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the Linear Regression model.

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
        self.parameters = self.model.coef_

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Make predictions using the Linear Regression model.

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


class DecisionTreeRegressorModel(Model):
    """
    Decision Tree Regressor model for predicting continuous target variables.

    This model uses scikit-learn's `DecisionTreeRegressor` to predict
    a continuous target by learning decision rules that segment the data
    into partitions, each with a distinct value or range for the target.
    """

    def __init__(self) -> None:
        """Initialize Decision Tree Regressor model."""
        self.model = DecisionTreeRegressor()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the Decision Tree Regressor model.

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
        Make predictions using the Decision Tree Regressor model.

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


class SVMRegressorModel(Model):
    """
    Support Vector Regressor for predicting continuous target variables.

    This model uses scikit-learn's `SVR` to fit a hyperplane that
    best approximates the relationship between input features and a
    continuous target, providing support for both linear and
    non-linear regression tasks.
    """

    def __init__(self) -> None:
        """Initialize SVR model."""
        self.model = SVR()

    def fit(self, observations: np.ndarray, ground_truth: np.ndarray) -> None:
        """
        Train the Support Vector Regressor model.

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
        Make predictions using the Support Vector Regressor model.

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
