"""
Script detection module.

Detects the writing system(s) used in a text based on Unicode code-point ranges.
Script detection is independent of language detection: Roman/Latin script does not
imply English, and Devanagari does not imply Hindi.

Supported scripts:
- Latin / Roman
- Devanagari (Hindi, Marathi, Sanskrit, etc.)
- Bengali (Bengali, Assamese)
- Odia
- Other (CJK, Arabic, Cyrillic, etc.)
- Unknown (no alphabetic content)

Design decisions:
- Uses Unicode block ranges, which are deterministic and require no ML model.
- Returns the dominant script plus a breakdown of all detected scripts.
- Mixed-script texts report the dominant script with a lower confidence.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from pydantic import BaseModel


# Unicode ranges for script identification
_SCRIPT_RANGES: List[tuple] = [
    # (start, end, script_name)
    (0x0000, 0x024F, "latin"),       # Basic Latin + Latin Extended-A/B
    (0x0900, 0x097F, "devanagari"),  # Devanagari
    (0x0980, 0x09FF, "bengali"),     # Bengali / Assamese
    (0x0B00, 0x0B7F, "odia"),        # Odia (Oriya)
    (0x0A00, 0x0A7F, "gurmukhi"),    # Gurmukhi (Punjabi)
    (0x0A80, 0x0AFF, "gujarati"),    # Gujarati
    (0x0B80, 0x0BFF, "tamil"),       # Tamil
    (0x0C00, 0x0C7F, "telugu"),      # Telugu
    (0x0C80, 0x0CFF, "kannada"),     # Kannada
    (0x0D00, 0x0D7F, "malayalam"),   # Malayalam
    (0x0600, 0x06FF, "arabic"),      # Arabic
    (0x4E00, 0x9FFF, "cjk"),         # CJK Unified Ideographs
    (0x0400, 0x04FF, "cyrillic"),    # Cyrillic
]


def _classify_char(c: str) -> str:
    """Classify a single character into a script category."""
    cp = ord(c)
    if not c.isalpha():
        return "_non_alpha"
    for start, end, script in _SCRIPT_RANGES:
        if start <= cp <= end:
            return script
    return "other"


class ScriptResult(BaseModel):
    """Result of script detection for a single text."""

    script: str
    confidence: float
    script_counts: Dict[str, int]
    detector_name: str
    detector_version: str


class ScriptDetector(ABC):
    """Abstract interface for script detection."""

    @abstractmethod
    def detect(self, text: str) -> ScriptResult:
        """Detect the dominant script in the given text."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        ...


class SimpleScriptDetector(ScriptDetector):
    """
    Baseline script detector using Unicode code-point ranges.

    Returns the dominant script by character count.
    Confidence is the proportion of alphabetic characters
    belonging to the dominant script.
    """

    _NAME = "unicode_range"
    _VERSION = "1.0.0"

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def version(self) -> str:
        return self._VERSION

    def detect(self, text: str) -> ScriptResult:
        if not text or not text.strip():
            return ScriptResult(
                script="unknown",
                confidence=0.0,
                script_counts={},
                detector_name=self._NAME,
                detector_version=self._VERSION,
            )

        counts: Dict[str, int] = {}
        for ch in text:
            script = _classify_char(ch)
            if script == "_non_alpha":
                continue
            counts[script] = counts.get(script, 0) + 1

        if not counts:
            return ScriptResult(
                script="unknown",
                confidence=0.0,
                script_counts={},
                detector_name=self._NAME,
                detector_version=self._VERSION,
            )

        total = sum(counts.values())
        dominant = max(counts, key=counts.get)  # type: ignore[arg-type]
        confidence = round(counts[dominant] / total, 4) if total > 0 else 0.0

        return ScriptResult(
            script=dominant,
            confidence=confidence,
            script_counts=counts,
            detector_name=self._NAME,
            detector_version=self._VERSION,
        )
