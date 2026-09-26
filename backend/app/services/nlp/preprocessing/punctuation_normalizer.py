"""
Punctuation normalization stage.

Conservative approach — punctuation often carries sentiment/emotion:
  "really??"  signals emphasis / disbelief
  "what!!!"   signals shock / frustration

Rules:
- Runs of 3+ identical punctuation marks are already reduced to 2 by
  the repeated-character stage.  This stage does NOT further reduce them.
- Ellipsis normalization: multiple dots (.. / .... / ......) -> "..."
- Curly/smart quotes are normalized to straight quotes.
- Em-dash / en-dash variants are normalized to " - ".
- Negation-carrying punctuation ("not", "n't") is NEVER removed.
"""

import re


# Normalize multiple dots to a standard ellipsis (three dots)
_MULTI_DOT_RE = re.compile(r"\.{2,}")

# Smart / curly quotes -> straight
_SMART_QUOTES = str.maketrans({
    "\u2018": "'",   # left single
    "\u2019": "'",   # right single
    "\u201c": '"',   # left double
    "\u201d": '"',   # right double
    "\u2032": "'",   # prime
    "\u2033": '"',   # double prime
})

# Dash variants -> standard hyphen-minus with spaces
_DASH_RE = re.compile(r"[\u2013\u2014\u2015]+")  # en-dash, em-dash, horizontal bar


def normalize_punctuation(text: str) -> str:
    """Conservatively normalize punctuation while preserving sentiment cues."""
    text = _MULTI_DOT_RE.sub("...", text)
    text = text.translate(_SMART_QUOTES)
    text = _DASH_RE.sub(" - ", text)
    return text
