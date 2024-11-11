"""
artifact.py

This module defines the `Artifact` class,
representing machine learning artifacts such as datasets or models.
The class provides methods for saving and reading data.

Classes
-------
Artifact
    Represents an ML artifact, such as a dataset or model.
"""
from pydantic import BaseModel, Field
import base64
import pandas as pd
import io


class Artifact(BaseModel):
    """
    Represents a ML artifact, such as a dataset or model.

    Attributes
    ----------
    asset_path : str
        Path to the asset file
    version : str
        Version of the artifact
    data : bytes
        Binary state data of the artifact
    metadata : dict
        Metadata for the artifact
    type : str
        Type of the artifact
    tags : list
        Tags for categorizing the artifact
    """

    asset_path: str = Field(title="Path to the asset file")
    version: str = Field(title="Version")
    data: bytes = Field(title="Binary state data")
    metadata: dict = Field(title="Metadata for the artifact",
                           default_factory=dict)
    type: str = Field(title="Type of the artifact")
    tags: list = Field(title="Tags for categorizing the artifact",
                       default_factory=list)

    @property
    def id(self) -> str:
        """
        Generates a unique ID for the artifact
        based on the asset_path and the version.

        Returns
        -------
        str
            An unique ID
        """
        encoded_path = base64.b64encode(
            self.asset_path.encode()).decode()
        return f"{encoded_path}.{self.version}"

    def read(self) -> pd.DataFrame:
        """
        Read the dataset from the given encoded CSV data.

        Returns
        -------
        pd.DataFrame
            The dataset as a pandas DataFrame.
        """
        csv_string = self.data.decode()
        return pd.read_csv(io.StringIO(csv_string))

    def save(self, data: pd.DataFrame) -> None:
        """
        Encodes and saves the given DataFrame as
        CSV bytes into `self.data`.

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to be saved as CSV bytes.

        Returns
        -------
            None
        """
        self.data = data.to_csv(index=False).encode()
