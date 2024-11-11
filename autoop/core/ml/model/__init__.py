from autoop.core.ml.model.model import Model
from autoop.core.ml.model.classification.classification_models import LogisticRegressionModel, DecisionTreeClassifierModel, SVMClassifierModel
from autoop.core.ml.model.regression.regression_models import LinearRegressionModel, DecisionTreeRegressorModel, SVMRegressorModel

CLASSIFICATION_MODELS = [
    "logistic_regression_model",
    "decision_tree_classifier_model",
    "svm_classifier_model"
]

REGRESSION_MODELS = [
    "linear_regression_model",
    "decision_tree_regressor_model",
    "svm_regressor_model"
]


def get_model(model_name: str) -> Model:
    """Factory function which gets a model by name."""
    if model_name == "logistic_regression_model":
        return LogisticRegressionModel()
    elif model_name == "decision_tree_classifier_model":
        return DecisionTreeClassifierModel()
    elif model_name == "svm_classifier_model":
        return SVMClassifierModel()
    elif model_name == "linear_regression_model":
        return LinearRegressionModel()
    elif model_name == "decision_tree_regressor_model":
        return DecisionTreeRegressorModel()
    elif model_name == "svm_regressor_model":
        return SVMRegressorModel()
    else:
        raise NotImplementedError("To be implemented.")