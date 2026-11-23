"""Context-aware Gen-Z slang, Indian internet expression, and emoji normalization layer (Phase 3).

Pure Python stdlib (zero third-party dependencies) so it works identically in
offline 5-task ablation experiments, Docker containers, and Vercel serverless endpoints.

Supports the four Phase 3 ablation modes:
- ``original``: untouched text (baseline).
- ``slang_normalized``: context-aware slang & Indian internet expression normalization
  (disambiguating Gen-Z ``mid``, ``based``, ``cooked`` from literal uses like
  ``mid sem`` / ``based on`` / ``home cooked``; preserving tone terms like ``bhai``,
  ``bro``, ``yaar``, ``lol`` with structured metadata).
- ``emoji_aware``: affect-preserving semantic emoji verbalization (never drops emojis!).
- ``slang_emoji_semantic``: combined context-aware slang + affect-preserving emoji semantics.

Always preserves ``original_text`` and returns structured ``transformations`` metadata.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from app.services.nlp.preprocessing.emoji_normalizer import (
    extract_emojis,
    normalize_emojis_semantic_affect,
)
from app.services.nlp.preprocessing.slang_normalizer import (
    PRESERVED_TONE_SLANG,
    SLANG_LEXICON,
)

# Additional Indian Reddit / internet slang expressions (Phase 3)
INDIAN_INTERNET_SLANG: Dict[str, str] = {
    "jugaad": "resourceful workaround",
    "timepass": "casual distraction",
    "funda": "fundamental concept",
    "fundae": "fundamental concepts",
    "chill": "relaxed",
    "scene": "situation",
}

# Context patterns for Gen-Z ambiguous terms ("mid", "based", "cooked")
_MID_LITERAL_RE = re.compile(
    r"\bmid[-\s]+(sem|semester|term|tier|dec|december|jan|january|feb|march|april|may|june|july|aug|sept|oct|nov|"
    r"20s|30s|40s|year|way|range|career|level|sized|size|point|week|day|night|morning|afternoon|summer|winter|august)\b",
    re.IGNORECASE,
)
_MID_SLANG_RE = re.compile(
    r"(?P<prefix>\b(?:is|are|was|were|looks|feels|seems|literally|pretty|kinda|kind of|very|so|super|toh|quite|really|actually)\s+)"
    r"(?P<term>\bmid\b)"
    r"|\b(?P<term2>mid)\b(?=\s+(?:hai|hain|af|bro|bhai|yaar|tbh|ngl|fr|imo)\b|[.!?\s]*$)",
    re.IGNORECASE,
)

_BASED_LITERAL_RE = re.compile(
    r"\b(?:based\s+(?:on|in|out|upon|around|at|near)|"
    r"(?:skill|merit|case|need|score|profile|cat|percentile|data|fact|research|evidence|value|rule|cloud|web|home|campus)[-\s]+based)\b",
    re.IGNORECASE,
)
_BASED_SLANG_RE = re.compile(
    r"(?P<prefix>\b(?:so|very|pretty|super|unironically|really|literally|actually|extremely)\s+)?(?P<term>\bbased\b)"
    r"(?=\s+(?:take|opinion|comment|reply|advice|bro|bhai|yaar|af|fr|ngl|tbh)\b|[.!?\s]*$)",
    re.IGNORECASE,
)

_COOKED_LITERAL_RE = re.compile(
    r"\b(?:(?:home|well|over|under|freshly|properly|badly|poorly|nicely)[-\s]+cooked|"
    r"cooked\s+(?:food|meal|meals|rice|dal|roti|chicken|paneer|veg|properly|well|fresh))\b",
    re.IGNORECASE,
)
_COOKED_SLANG_RE = re.compile(
    r"(?P<prefix>\b(?:i['’]?m|im|am|we['’]?re|were|we\s+are|you['’]?re|you\s+are|he['’]?s|bro\s+is|bhai\s+is|"
    r"beyond|totally|completely|literally|actually|so|royally|properly|lowkey|deadass)\s+)(?P<term>\bcooked\b)",
    re.IGNORECASE,
)

_SCENE_SLANG_RE = re.compile(
    r"(?P<prefix>\b(?:kya|placement|internship|hostel|mess|campus|sip|job|college|loan)\s+)(?P<term>\bscene\b)",
    re.IGNORECASE,
)

_SPACE_RE = re.compile(r"\s+")


def normalize_context_aware_slang(text: str) -> Dict[str, Any]:
    """Apply context-aware Gen-Z & Indian Reddit slang normalization without mutating originals."""
    original = str(text or "")
    working = original
    transformations: List[Dict[str, Any]] = []

    # 1. Context-aware disambiguation for "mid"
    if re.search(r"\bmid\b", working, re.IGNORECASE):
        if _MID_LITERAL_RE.search(working):
            transformations.append({
                "term": "mid",
                "action": "literal_preserved",
                "replacement": "mid",
                "reason": "Literal temporal/tier compound detected (e.g., mid-sem / mid-tier / mid-Dec).",
            })
        else:
            def _rep_mid(m: re.Match[str]) -> str:
                transformations.append({
                    "term": "mid",
                    "action": "context_expanded",
                    "replacement": "mediocre",
                    "reason": "Evaluative Gen-Z predicate context detected.",
                })
                if m.group("term"):
                    return f"{m.group('prefix')}mediocre"
                return "mediocre"

            working = _MID_SLANG_RE.sub(_rep_mid, working)

    # 2. Context-aware disambiguation for "based"
    if re.search(r"\bbased\b", working, re.IGNORECASE):
        if _BASED_LITERAL_RE.search(working):
            transformations.append({
                "term": "based",
                "action": "literal_preserved",
                "replacement": "based",
                "reason": "Literal prepositional/compound use detected (e.g., based on / profile-based).",
            })
        else:
            def _rep_based(m: re.Match[str]) -> str:
                transformations.append({
                    "term": "based",
                    "action": "context_expanded",
                    "replacement": "boldly agreeable",
                    "reason": "Evaluative Gen-Z stance context detected.",
                })
                pref = m.group("prefix") or ""
                return f"{pref}boldly agreeable"

            working = _BASED_SLANG_RE.sub(_rep_based, working)

    # 3. Context-aware disambiguation for "cooked"
    if re.search(r"\bcooked\b", working, re.IGNORECASE):
        if _COOKED_LITERAL_RE.search(working):
            transformations.append({
                "term": "cooked",
                "action": "literal_preserved",
                "replacement": "cooked",
                "reason": "Literal culinary context detected (e.g., home-cooked / cooked food).",
            })
        else:
            def _rep_cooked(m: re.Match[str]) -> str:
                transformations.append({
                    "term": "cooked",
                    "action": "context_expanded",
                    "replacement": "overwhelmed doomed",
                    "reason": "Gen-Z distress/doom predicate context detected.",
                })
                return f"{m.group('prefix')}overwhelmed doomed"

            working = _COOKED_SLANG_RE.sub(_rep_cooked, working)

    # 4. Indian English slang "scene" in campus/placement context
    if _SCENE_SLANG_RE.search(working):
        def _rep_scene(m: re.Match[str]) -> str:
            transformations.append({
                "term": "scene",
                "action": "context_expanded",
                "replacement": "situation",
                "reason": "Indian student slang ('placement/kya scene' -> situation).",
            })
            return f"{m.group('prefix')}situation"

        working = _SCENE_SLANG_RE.sub(_rep_scene, working)

    # 5. Token-level safe abbreviation expansion & preserved-tone tracking
    words = working.split()
    out_words: List[str] = []
    for word in words:
        stripped = word.strip(".,!?;:\"'()[]{}").lower()
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

        if stripped in SLANG_LEXICON:
            repl = SLANG_LEXICON[stripped]
            out_words.append(prefix + repl + suffix)
            transformations.append({
                "term": stripped,
                "action": "expanded",
                "replacement": repl,
                "reason": "Standard internet/Gen-Z abbreviation lexicon.",
            })
        elif stripped in ("jugaad", "timepass", "funda", "fundae"):
            repl = INDIAN_INTERNET_SLANG[stripped]
            out_words.append(prefix + repl + suffix)
            transformations.append({
                "term": stripped,
                "action": "indian_slang_expanded",
                "replacement": repl,
                "reason": "Indian student/internet colloquialism.",
            })
        else:
            out_words.append(word)
            if stripped in PRESERVED_TONE_SLANG:
                transformations.append({
                    "term": stripped,
                    "action": "tone_preserved",
                    "replacement": stripped,
                    "reason": PRESERVED_TONE_SLANG[stripped],
                })

    normalized = _SPACE_RE.sub(" ", " ".join(out_words)).strip()
    return {
        "original_text": original,
        "normalized_text": normalized,
        "changed": normalized != original,
        "transformations": transformations,
    }


def apply_informal_normalization(text: str, mode: str = "slang_emoji_semantic") -> Dict[str, Any]:
    """Run one of the four Phase 3 normalization regimes with full provenance metadata.

    Supported modes:
    - ``original`` (A): unaltered text
    - ``slang_normalized`` (B): context-aware Gen-Z + Indian internet slang normalization
    - ``emoji_aware`` (C): affect-preserving emoji semantic verbalization
    - ``slang_emoji_semantic`` (D): context-aware slang + emoji semantic verbalization
    """
    original = str(text or "")
    emojis = extract_emojis(original)
    transformations: List[Dict[str, Any]] = []

    if mode == "original":
        norm = original
    elif mode == "slang_normalized":
        s_res = normalize_context_aware_slang(original)
        norm = s_res["normalized_text"]
        transformations.extend(s_res["transformations"])
    elif mode == "emoji_aware":
        norm = normalize_emojis_semantic_affect(original)
        for em in emojis:
            transformations.append({
                "term": em["emoji"],
                "action": "emoji_semantic_affect",
                "replacement": em["semantic_affect"],
                "reason": f"Preserved emotional signal from {em['token']}.",
            })
    elif mode == "slang_emoji_semantic":
        s_res = normalize_context_aware_slang(original)
        transformations.extend(s_res["transformations"])
        norm = normalize_emojis_semantic_affect(s_res["normalized_text"])
        for em in emojis:
            transformations.append({
                "term": em["emoji"],
                "action": "emoji_semantic_affect",
                "replacement": em["semantic_affect"],
                "reason": f"Preserved emotional signal from {em['token']}.",
            })
    else:
        raise ValueError(
            f"Unknown mode={mode!r}. Expected one of: original, slang_normalized, emoji_aware, slang_emoji_semantic"
        )

    return {
        "original_text": original,
        "normalized_text": norm,
        "mode": mode,
        "changed": norm != original,
        "transformations": transformations,
        "emojis_detected": emojis,
    }
