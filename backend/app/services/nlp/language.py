"""
Language detection module.

Provides a modular LanguageDetector interface and a baseline
SimpleLanguageDetector implementation using the langdetect library.

Design decisions:
- Language detection is separated from script detection and code-mix detection.
- The interface returns language + confidence so detectors can be swapped.
- langdetect is a lightweight baseline; it struggles with Romanized Hindi/Bengali
  and very short texts. These limitations are documented, not hidden.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

from pydantic import BaseModel
from langdetect import detect_langs, DetectorFactory

# Ensure reproducible results across runs
DetectorFactory.seed = 42


class LanguageResult(BaseModel):
    """Result of language detection for a single text."""

    language: str
    confidence: float
    all_detected: List[Tuple[str, float]]
    detector_name: str
    detector_version: str


class LanguageDetector(ABC):
    """Abstract interface for language detection."""

    @abstractmethod
    def detect(self, text: str) -> LanguageResult:
        """Detect the primary language of the given text."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        ...


class SimpleLanguageDetector(LanguageDetector):
    """
    Baseline language detector using the langdetect library.

    Known limitations:
    - Romanized Hindi (e.g. "jaana worth it hai kya") is often misclassified
      as a European language because langdetect relies on character n-grams
      and Latin-script Hindi looks similar to Romance languages.
    - Very short texts (< 20 chars) produce unreliable results.
    - Bengali and Odia in native script are generally detected well.
    - Code-mixed text may return the dominant language only.
    """

    _NAME = "langdetect"
    _VERSION = "1.0.9"

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def version(self) -> str:
        return self._VERSION

    def detect(self, text: str) -> LanguageResult:
        if not text or not text.strip():
            return LanguageResult(
                language="unknown",
                confidence=0.0,
                all_detected=[],
                detector_name=self._NAME,
                detector_version=self._VERSION,
            )

        try:
            results = detect_langs(text)
            all_detected = [(r.lang, round(r.prob, 4)) for r in results]
            primary_lang = results[0].lang
            primary_conf = round(results[0].prob, 4)
        except Exception:
            primary_lang = "unknown"
            primary_conf = 0.0
            all_detected = []

        return LanguageResult(
            language=primary_lang,
            confidence=primary_conf,
            all_detected=all_detected,
            detector_name=self._NAME,
            detector_version=self._VERSION,
        )
