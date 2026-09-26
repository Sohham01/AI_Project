"""
URL and mention normalization stage.

Rules:
- URLs are replaced with the token <URL>
- Reddit-style mentions (u/username, /u/username) are replaced with <MENTION>
- Subreddit references (r/name, /r/name) are preserved as-is because they
  are topical information, not personal identifiers.
- This reduces noise and avoids leaking identifying information downstream.
"""

import re

# Match http/https URLs and bare www. URLs
_URL_RE = re.compile(
    r"https?://[^\s<>\"']+|www\.[^\s<>\"']+",
    re.IGNORECASE,
)

# Match Reddit user mentions: u/username or /u/username
_MENTION_RE = re.compile(r"/?u/[A-Za-z0-9_-]+")


def normalize_urls_mentions(text: str) -> str:
    """Replace URLs with <URL> and user mentions with <MENTION>."""
    text = _URL_RE.sub("<URL>", text)
    text = _MENTION_RE.sub("<MENTION>", text)
    return text
