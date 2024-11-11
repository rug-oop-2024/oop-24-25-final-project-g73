"""
feature.py

This module detects and categorizes feature types in a dataset as either
numerical or categorical.
The primary function, `detect_feature_types`, examines each column
of the dataset and assigns a type based on the data's characteristics.

Functions
---------
detect_feature_types(dataset: Dataset) -> List[Feature]
    Analyzes a dataset and assigns a type to each feature.
"""
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Takes features from a dataset and categorize them
    as either numerical or categorical.

    Assumption:
        -only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    features = []
    data = dataset.read()

    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
            feature_type = "numerical"
        else:
            feature_type = "categorical"

        feature = Feature(name=column, type=feature_type)
        features.append(feature)

    return features
