import unittest
from unittest.mock import MagicMock
from app.services.ingestion.manager import IngestionManager

class TestIngestionManager(unittest.TestCase):
    def setUp(self):
        self.manager = IngestionManager()

    def test_get_provider(self):
        provider = self.manager.get_provider("arctic_shift")
        self.assertIsNotNone(provider)

    def test_execute_operation_unsupported(self):
        with self.assertRaises(Exception):
            self.manager.execute_operation("scrape_url", url="http://invalid.url")

if __name__ == "__main__":
    unittest.main()
