"""
pipeline.py

This module defines the `Pipeline` class, which operates
a machine learning workflow.
The pipeline includes steps for data preprocessing, model training,
and evaluation of model performance using specified metrics.

Classes
-------
Pipeline
    A class for managing and executing machine learning workflows,
    including preprocessing, training, and evaluation
    with a model and metrics.
"""
from typing import List, Dict, Any
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    """
    A machine learning pipeline that
    preprocesses data, trains a model, and evaluates metrics.

    Attributes
    ----------
    _dataset : Dataset
        Dataset containing the features and target for training and testing.
    _model : Model
        Model used for training and evaluation.
    _input_features : List[Feature]
        List of features used as inputs for the model.
    _target_feature : Feature
        The target feature for the model.
    _metrics : List[Metric]
        List of metrics to evaluate model performance.
    _artifacts : dict
        Dictionary storing artifacts generated during the pipeline execution.
    _split : float
        Proportion of data to be used for training.
    """

    def __init__(self,
                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ) -> None:
        """
        Initializes the pipeline with a dataset,
        model, and evaluation metrics.

        Parameters
        ----------
        metrics : List[Metric]
            List of metrics to evaluate the model.
        dataset : Dataset
            The dataset to be used in the pipeline.
        model : Model
            The model to be used in the pipeline.
        input_features : List[Feature]
            List of features used as inputs for the model.
        target_feature : Feature
            The target feature for the model.
        split : float
            Proportion of data to be used for training, by default 0.8.

        Raises
        ------
        ValueError
            If the target feature type does not match the model type.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if target_feature.type == "categorical" and model.type != "classification":
            raise ValueError(
                "Model type must be classification for categorical target feature")
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature")

    def __str__(self) -> str:
        """
        Returns a string representation of the pipeline.

        Returns
        -------
        str
            A string describing the pipeline configuration.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Gets the model used in the pipeline.

        Returns
        -------
        Model
            The model object.
        """
        return self._model

    @property
    def artifacts(self) -> List[Artifact]:
        """
        Retrieves artifacts generated during pipeline execution.

        Returns
        -------
        List[Artifact]
            List of artifacts for the
            pipeline configuration, model, and encoders.
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(name=name, data=data))
        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
        }
        artifacts.append(Artifact(name="pipeline_config",
                                  data=pickle.dumps(pipeline_data)))
        artifacts.append(self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}"))
        return artifacts

    def _register_artifact(self, name: str, artifact: Any) -> None:
        """
        Registers an artifact with a specified name.

        Parameters
        ----------
        name : str
            Name of the artifact.
        artifact : Any
            The artifact that will be register.
        """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """
        Preprocesses input and target features, and registers
        any transformation artifacts.

        Returns
        -------
        None
        """
        (target_feature_name, target_data, artifact) = preprocess_features([self._target_feature], self._dataset)[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset)
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [data for (feature_name, data, artifact) in input_results]

    def _split_data(self) -> None:
        """
        Splits data into training and testing sets based on the specified split ratio.

        Returns
        -------
        None
        """
        split = self._split
        self._train_X = [vector[:int(split * len(vector))] for vector in self._input_vectors]
        self._test_X = [vector[int(split * len(vector)):] for vector in self._input_vectors]
        self._train_y = self._output_vector[:int(split * len(self._output_vector))]
        self._test_y = self._output_vector[int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """
        Compacts a list of numpy arrays into a single array.

        Parameters
        ----------
        vectors : List[np.array]
            A list of numpy arrays.

        Returns
        -------
        np.array
            Compacted numpy array.
        """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """
        Trains the model using the training data
        and calculates training metrics.

        Returns
        -------
        None
        """
        X = np.array(self._compact_vectors(self._train_X))
        Y = np.array(self._train_y)
        self._model.fit(X, Y)
        self._metrics_train = []
        for metric in self._metrics:
            result = metric(X, Y)
            self._metrics_train.append((metric.__class__.__name__, result))

    def _evaluate(self) -> None:
        """
        Evaluates the model on the test data
        and calculates evaluation metrics.

        Returns
        -------
        None
        """
        X = np.array(self._compact_vectors(self._test_X))
        Y = np.array(self._test_y)
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric(predictions, Y)
            self._metrics_results.append((metric.__class__.__name__, result))
        self._predictions = predictions

    def execute(self) -> Dict[str, Any]:
        """
        Executes the pipeline, including preprocessing,
        training, and evaluation.

        Returns
        -------
        Dict[str, Any]
            Results of training and evaluation,
            including metrics and predictions.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()
        return {
            "training metrics": self._metrics_train,
            "testing metrics": self._metrics_results,
            "predictions": self._predictions,
        }
