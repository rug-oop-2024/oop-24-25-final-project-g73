
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np


class Feature(BaseModel):
    """
    Represents a feature in a dataset

    Attributes
    ----------
    name : str
        The name of the feature
    type : Literal["numerical", "categorical"]
        The type of the feature, either categorical or numerical
    """
    name: str = Field(title="Name of feature", default=None)
    type: Literal["numerical", "categorical"] = Field(title="Type of feature",
                                                      default=None)

    def __str__(self) -> str:
        return f"Feature name={self.name}, type={self.type}"
