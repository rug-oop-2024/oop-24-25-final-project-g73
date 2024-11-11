"""
dataset.py

This module provides the `Dataset` class, inheriting from `Artifact`.
It represents a dataset artifact in a machine learning pipeline.

Classes
-------
Dataset
    Represents an ML dataset as an artifact.
"""
from autoop.core.ml.artifact import Artifact
import pandas as pd


class Dataset(Artifact):
    """A class to represent an ML dataset.

    Methods
    -------
    from_dataframe(data: pd.DataFrame, name: str,
    asset_path: str, version: str = "1.0.0") -> "Dataset"
        Creates a Dataset instance from a DataFrame.
    """

    def __init__(self, *args: tuple, **kwargs: dict) -> None:
        """
        Initializes a Dataset.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to the superclass
        **kwargs : dict
            Keyword arguments passed to the superclass
        """
        super().__init__(type="dataset", *args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str, version: str = "1.0.0") -> "Dataset":
        """
        Create a dataset from a pandas dataframe.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame that will be
            saved as CSV bytes in the Dataset
        name : str
            The name of the dataset
        asset_path : str
            The path where the dataset will be stored
        version : str
            The version of the dataset, by default "1.0.0"

        Returns
        -------
        Dataset
            The Dataset class with CSV data encoded as bytes
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )
