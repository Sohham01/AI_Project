import unittest
from unittest.mock import patch, MagicMock
from app.services.ingestion.providers.hermes import HermesProvider

class TestHermesProvider(unittest.TestCase):
    def setUp(self):
        self.provider = HermesProvider()
        self.provider.api_url = "http://test.com"
        self.provider.api_key = "test_key"

    @patch("requests.post")
    def test_scrape_url(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "1", "title": "Test"}
        mock_post.return_value = mock_response
        
        post = self.provider.scrape_url("http://test.com/post")
        self.assertEqual(post.title, "Test")

if __name__ == "__main__":
    unittest.main()
