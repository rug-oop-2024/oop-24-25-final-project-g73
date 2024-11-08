from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    """
    Represents a ML artifact, such as a dataset or model
    """

    asset_path = Field(..., description="Path to the asset file")
    version = Field("1.0.0", description="Version")
    data = Field(None, description="Binary state data")
    metadata = Field(None, description="Metadata for the artifact")
    type = Field(..., description="Type of the artifact")
    tags = Field(None, description="Tags for categorizing the artifact")

    @property
    def id(self):
        """
        Generates a unique ID for the artifact
        based on the asset_path and the version
        """
        encoded_path = base64.urlsafe_b64encode(
            self.asset_path.encode()).decode('utf-8')
        return f"{encoded_path}:{self.version}"

    def encode_data(self):
        """
        Encode binary data to a base64 string for easy serialization
        """
        if self.data:
            return base64.b64encode(self.data).decode('utf-8')
        return None

    def decode_data(self, encoded_data):
        """
        Decode base64 string data back into binary format
        """
        self.data = base64.b64decode(encoded_data)

    def load_metadata(self, metadata_dict):
        """
        Load metadata from a dictionary
        """
        self.metadata = metadata_dict

    def save_metadata(self):
        """
        Return metadata as a dictionary
        """
        return self.metadata

    def load(self, path):
        """
        Load binary data from a file and encode it in the artifact
        """
        with open(path, 'rb') as file:
            self.data = file.read()

    def save(self, path):
        """
        Save the binary data of the artifact to a file
        """
        if self.data is None:
            raise ValueError("No data to save")
        with open(path, 'wb') as file:
            file.write(self.data)
