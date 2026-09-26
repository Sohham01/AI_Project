"""
Whitespace normalization stage.

Rules:
- Collapse multiple consecutive spaces into one
- Replace tabs with a single space
- Replace multiple newlines with a single newline
- Strip leading/trailing whitespace
- Preserve single newlines (paragraph boundaries may matter)
"""

import re

_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving word boundaries."""
    text = text.replace("\t", " ")
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    text = text.strip()
    return text
