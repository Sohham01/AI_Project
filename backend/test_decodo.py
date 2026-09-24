import unittest
from unittest.mock import patch, MagicMock
from app.services.ingestion.providers.decodo import DecodoProvider

class TestDecodoProvider(unittest.TestCase):
    def setUp(self):
        self.provider = DecodoProvider()
        self.provider.api_token = "test_token"

    @patch("requests.post")
    def test_scrape_url(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "1", "title": "Test"}
        mock_post.return_value = mock_response
        
        post = self.provider.scrape_url("http://test.com/post")
        self.assertEqual(post.title, "Test")

if __name__ == "__main__":
    unittest.main()
