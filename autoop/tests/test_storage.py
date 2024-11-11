
import unittest

from autoop.core.storage import LocalStorage, NotFoundError
import random
import tempfile
import os

class TestStorage(unittest.TestCase):
    """
    Unit tests for the LocalStorage class,
    focusing on file storage and retrieval.
    """

    def setUp(self):
        """Set up a temporary storage instance for testing."""
        temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(temp_dir)

    def test_init(self):
        """Test the initialization of the LocalStorage instance."""
        self.assertIsInstance(self.storage, LocalStorage)

    def test_store(self):
        """Test storing and loading data in LocalStorage."""
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = f"test{os.sep}path"
        self.storage.save(test_bytes, key)
        self.assertEqual(self.storage.load(key), test_bytes)
        otherkey = f"test{os.sep}otherpath"
        # should not be the same
        try:
            self.storage.load(otherkey)
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_delete(self):
        """Test deleting data from LocalStorage."""
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = f"test{os.sep}path"
        self.storage.save(test_bytes, key)
        self.storage.delete(key)
        try:
            self.assertIsNone(self.storage.load(key))
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_list(self):
        """Test listing all saved data in LocalStorage."""
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        random_keys = [f"test{os.sep}{random.randint(0, 100)}" for _ in range(10)]
        for key in random_keys:
            self.storage.save(test_bytes, key)
        keys = self.storage.list("test")
        keys = [f"{os.sep}".join(key.split(f"{os.sep}")[-2:]) for key in keys]
        self.assertEqual(set(keys), set(random_keys))
