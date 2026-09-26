"""
Unicode normalization stage.

Applies NFC normalization and removes zero-width characters.
Does NOT remove meaningful Unicode characters (e.g. Devanagari, Bengali, Odia).

Rules:
- NFC normalization to compose equivalent representations
- Remove zero-width joiners/non-joiners (U+200B, U+200C, U+200D, U+FEFF)
- Preserve all alphabetic characters in any script
"""

import re
import unicodedata

# Zero-width and invisible formatting characters that carry no semantic value
_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")


def normalize_unicode(text: str) -> str:
    """Apply NFC normalization and strip zero-width characters."""
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH_RE.sub("", text)
    return text
