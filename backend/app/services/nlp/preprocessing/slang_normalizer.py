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
from typing import Dict, List

SLANG_LEXICON_VERSION = "1.1.0"

# key: lowercase slang -> value: normalized replacement
# Words that are already meaningful on their own (bro, bhai, lol)
# are kept as-is or given a minimal expansion.
SLANG_LEXICON: Dict[str, str] = {
    # English internet / Gen-Z abbreviations
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
    "fr": "for real",
    "ikr": "i know right",
    "ftw": "for the win",
    "rn": "right now",
    "nvm": "never mind",
    "dm": "direct message",
    "ama": "ask me anything",
    "eli5": "explain like i'm five",
    "btw": "by the way",
    "fyi": "for your information",
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "ty": "thank you",
    "np": "no problem",
    "brb": "be right back",
    "gtg": "got to go",
    "lmk": "let me know",
    "cfbr": "commenting for better reach",
    "irl": "in real life",
    "fomo": "fear of missing out",

    # Expressions kept as-is (they're widely understood or carry tone)
    # "lol" -> kept, "lmao" -> kept, "bro" -> kept, "bhai" -> kept
    # Context-sensitive Gen-Z terms ("mid", "based", "cooked") are preserved
    # rather than blindly rewritten to avoid corrupting "mid sem", "based on",
    # or literal food mentions, and are surfaced via detect_slang_terms().
}

PRESERVED_TONE_SLANG: Dict[str, str] = {
    "bro": "informal peer address (tone preserved)",
    "bhai": "Hindi/Hinglish peer address (tone preserved)",
    "yaar": "Hindi/Hinglish peer address (tone preserved)",
    "lol": "amusement / softening tone (preserved)",
    "lmao": "strong amusement (preserved)",
}

CONTEXT_SENSITIVE_SLANG: Dict[str, str] = {
    "mid": "Gen-Z 'mediocre' (preserved to avoid corrupting 'mid-tier' / 'mid sem' / 'mid 30s')",
    "based": "Gen-Z 'boldly agreeable' (preserved to avoid corrupting 'based on')",
    "cooked": "Gen-Z 'overwhelmed / doomed' (preserved to avoid corrupting literal food mentions)",
}


def detect_slang_terms(text: str) -> List[Dict[str, str]]:
    """Inspect all recognized slang / informal Reddit expressions in ``text``."""
    found: List[Dict[str, str]] = []
    seen: set[str] = set()
    for word in str(text or "").split():
        stripped = word.strip(".,!?;:\"'()[]{}").lower()
        if not stripped or stripped in seen:
            continue
        if stripped in SLANG_LEXICON:
            seen.add(stripped)
            found.append({
                "term": stripped,
                "action": "expanded",
                "meaning": SLANG_LEXICON[stripped],
            })
        elif stripped in PRESERVED_TONE_SLANG:
            seen.add(stripped)
            found.append({
                "term": stripped,
                "action": "preserved_tone",
                "meaning": PRESERVED_TONE_SLANG[stripped],
            })
        elif stripped in CONTEXT_SENSITIVE_SLANG:
            seen.add(stripped)
            found.append({
                "term": stripped,
                "action": "context_preserved",
                "meaning": CONTEXT_SENSITIVE_SLANG[stripped],
            })
    return found


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
