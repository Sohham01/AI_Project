import unittest
from unittest.mock import patch, MagicMock
from app.services.ingestion.providers.rss import RSSProvider

class TestRSSProvider(unittest.TestCase):
    def setUp(self):
        self.provider = RSSProvider()

    @patch("requests.get")
    def test_search_posts(self, mock_get):
        mock_response = MagicMock()
        mock_response.content = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
    <item>
        <title>Test Post</title>
        <link>http://test.com/post</link>
        <description>Test body</description>
    </item>
</channel></rss>"""
        mock_get.return_value = mock_response
        
        posts = self.provider.search_posts(url="http://rss.test")
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].title, "Test Post")

if __name__ == "__main__":
    unittest.main()
