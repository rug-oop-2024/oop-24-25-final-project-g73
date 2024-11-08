from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    """
    Represents a ML artifact, such as a dataset or model

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
    metadata: dict = Field(title="Metadata for the artifact")
    type: str = Field(title="Type of the artifact")
    tags: list = Field(title="Tags for categorizing the artifact")

    @property
    def id(self) -> str:
        """
        Generates a unique ID for the artifact
        based on the asset_path and the version

        Returns
        -------
        str
            An unique ID
        """
        encoded_path = base64.b64encode(
            self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def encode_data(self) -> str:
        """
        Encode binary data to a base64 string for easy serialization

        Returns
        -------
        str
            Encoded string of data
        """
        if self.data:
            return base64.b64encode(self.data).decode()
        return None

    def decode_data(self, encoded_data: str) -> None:
        """
        Decode base64 string data back into binary format

        Parameters
        ----------
        encoded_data: str
            Base64 string of encoded data

        Returns
        -------
            None
        """
        if not encoded_data:
            raise ValueError("No encoded data provded")
        self.data = base64.b64decode(encoded_data)

    def load_metadata(self, metadata_dict: dict) -> None:
        """
        Load metadata from a dictionary

        Parameters
        ----------
        metadata_dict: dict
            Dictionary containing metadata info

        Returns
        -------
            None
        """
        if not isinstance(metadata_dict, dict):
            raise ValueError("Metadata must be a dict")
        self.metadata = metadata_dict

    def save_metadata(self) -> dict:
        """
        Return metadata as a dictionary

        Returns
        -------
        dict
            The metadata dict
        """
        return self.metadata or {}

    def load(self, path: str) -> None:
        """
        Load binary data from a file and encode it in the artifact

        Parameters
        ----------
        path : str
            Path to the file from where we load the data

        Returns
        -------
            None
        """
        if not path:
            raise ValueError("Path could not be found")
        with open(path, 'rb') as file:
            self.data = file.read()

    def save(self, path: str) -> None:
        """
        Save the binary data of the artifact to a file

        Parameters
        ----------
        path : str
            Destination path on where the
            content of the file will be saved

        Returns
        -------
            None
        """
        if self.data is None:
            raise ValueError("No data to save")
        with open(path, 'wb') as file:
            file.write(self.data)
