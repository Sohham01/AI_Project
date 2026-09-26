"""
Slang / informal text normalization stage.

Maintains a small versioned lexicon of common slang and abbreviations
found in Indian social-media / Reddit discourse.

Design decisions:
- Slang is annotated, not aggressively replaced.
- Each slang term maps to a normalized form that preserves the original
  semantic intent.  "bro" stays "bro" (it's widely understood); "idk"
  expands to "i don't know" because abbreviation loss hurts downstream NLP.
- Hindi slang like "bhai" is preserved as-is — it is a real Hindi word.
- The lexicon is intentionally small and conservative.  It's versioned so
  that expansions can be tracked across preprocessing versions.
- Only whole-word matches are replaced (word boundaries enforced).

How to add slang:
  Add entries to SLANG_LEXICON below.  Key = lowercase slang form,
  value = normalized replacement.  Bump SLANG_LEXICON_VERSION.
"""

import re
from typing import Dict

SLANG_LEXICON_VERSION = "1.0.0"

# key: lowercase slang -> value: normalized replacement
# Words that are already meaningful on their own (bro, bhai, lol)
# are kept as-is or given a minimal expansion.
SLANG_LEXICON: Dict[str, str] = {
    # English internet abbreviations
    "idk": "i don't know",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "tbh": "to be honest",
    "tbf": "to be fair",
    "wtf": "what the fuck",
    "wth": "what the hell",
    "omg": "oh my god",
    "smh": "shaking my head",
    "fwiw": "for what it's worth",
    "afaik": "as far as i know",
    "iirc": "if i recall correctly",
    "tl;dr": "too long didn't read",
    "tldr": "too long didn't read",
    "ngl": "not gonna lie",
    "ikr": "i know right",
    "ftw": "for the win",
    "rn": "right now",
    "nvm": "never mind",
    "dm": "direct message",
    "ama": "ask me anything",
    "eli5": "explain like i'm five",
    "imo": "in my opinion",
    "btw": "by the way",
    "fyi": "for your information",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "ty": "thank you",
    "np": "no problem",
    "brb": "be right back",
    "gtg": "got to go",

    # Expressions kept as-is (they're widely understood or carry tone)
    # "lol" -> kept, "lmao" -> kept, "bro" -> kept, "bhai" -> kept
    # These are NOT in the lexicon so they pass through unchanged.
}


def normalize_slang(text: str) -> str:
    """Replace known slang abbreviations with expanded forms (whole-word)."""
    words = text.split()
    result = []
    for word in words:
        # Strip surrounding punctuation for lookup, preserve it
        stripped = word.strip(".,!?;:\"'()[]{}").lower()
        if stripped in SLANG_LEXICON:
            # Preserve leading/trailing punctuation from original word
            prefix = ""
            suffix = ""
            i = 0
            while i < len(word) and not word[i].isalnum():
                prefix += word[i]
                i += 1
            j = len(word) - 1
            while j >= 0 and not word[j].isalnum():
                suffix = word[j] + suffix
                j -= 1
            result.append(prefix + SLANG_LEXICON[stripped] + suffix)
        else:
            result.append(word)
    return " ".join(result)
