"""
storage.py

This module defines an abstract base class Storage for storage and
implements a LocalStorage.
These classes provide a standardized interface for
saving, loading, deleting, and listing data.

Classes
-------
Storage
    Abstract base class defining the interface for storage.
LocalStorage
    Implement Storage using the local filesystem for persistence.
NotFoundError
    Exception raised when a specified path is not found.
"""

from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """Exception raised when a specified path is not found."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """Abstract base class defining the interface for a storage."""

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path.

        Parameters
        ----------
        data : bytes
            The data to be saved
        path : str
            The path where data will be saved

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path.

        Parameters
        ----------
        path : str
            The path from where data will be loaded

        Returns
        -------
        bytes
            The loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path.

        Parameters
        ----------
        path : str
            The path of the data that will be deleted

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given directory path.

        Parameters
        ----------
        path : str
            The path from which data will be listed

        Returns
        -------
        List[str]
            A list of paths under the specified path
        """
        pass


class LocalStorage(Storage):
    """
    A local file system-based storage
    implementing the Storage Interface.

    Attributes
    ----------
    _base_path : str
        Base directory for storage
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes LocalStorage with a base directory path.

        Parameters
        ----------
        base_path : str
            The base path for storage, default is "./assets"
        """
        self._base_path = os.path.normpath(base_path)
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Save data to a file at the specified key path.

        Parameters
        ----------
        data : bytes
            The data that will be saved
        key : str
            Path corresponding to base path where data will be stored

        Returns
        -------
        None
        """
        path = self._join_path(key)
        # Ensure parent directories are created
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Load data from a file at the specified key path.

        Parameters
        ----------
        key : str
            Path corresponding to base path where data is stored

        Returns
        -------
        bytes
            The loaded data
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, 'rb') as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Delete a file at the specified key path.

        Parameters
        ----------
        key : str
            Path corresponding to base path that will be deleted

        Returns
        -------
        None
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        os.remove(path)

    def list(self, prefix: str = "/") -> List[str]:
        """
        List all file paths under a given prefix.

        Parameters
        ----------
        prefix : str
            Path prefix under which files will be listed, default is "/"

        Returns
        -------
        List[str]
            List of related paths under the specified prefix
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        # Use os.path.join for compatibility across platforms
        keys = glob(os.path.join(path, "**", "*"), recursive=True)
        return [os.path.relpath(p, self._base_path) for p in keys
                if os.path.isfile(p)]

    def _assert_path_exists(self, path: str) -> None:
        """
        Checks if a specified path exists and raise NotFoundError if not found.

        Parameters
        ----------
        path : str
            The path to check

        Returns
        -------
        None
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Join a given path with the base path to form a full path.

        Parameters
        ----------
        path : str
            The path to join with the base path

        Returns
        -------
        str
            The full path as an OS string
        """
        # Ensure paths are OS-agnostic
        return os.path.normpath(os.path.join(self._base_path, path))
