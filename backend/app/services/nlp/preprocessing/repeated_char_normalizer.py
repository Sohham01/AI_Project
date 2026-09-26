"""
Repeated character normalization stage.

Strategy (conservative):
- If a character repeats 3+ times consecutively, reduce to 2 occurrences.
- This preserves informal emphasis ("soo good") while removing extreme
  repetition ("soooooo good" -> "soo good").
- Two repetitions are kept because they signal emphasis without being noise.

Examples:
  "soooo good"    -> "soo good"
  "broooo"        -> "broo"
  "goooood"       -> "good"    (already has natural double 'o')
  "really??"      -> "really??" (only 2, no change)
  "what!!!!!!"    -> "what!!"

Rule: any run of 3+ identical characters -> 2 of that character.
This applies to letters AND punctuation uniformly; punctuation-specific
rules are in punctuation_normalizer.py but this stage runs first as a
general safety net.
"""

import re

# Match any character repeated 3 or more times
_REPEATED_RE = re.compile(r"(.)\1{2,}")


def normalize_repeated_chars(text: str) -> str:
    """Reduce runs of 3+ identical characters to 2."""
    return _REPEATED_RE.sub(r"\1\1", text)
