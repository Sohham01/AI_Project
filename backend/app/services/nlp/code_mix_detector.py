"""
Code-mix detection module.

Determines whether a text contains meaningful multilingual / code-mixed content.
Code-mixing is common in Indian online discourse where users freely switch between
English and Hindi/Bengali/Odia, often within the same sentence.

Key distinction:
- Roman script != English.  "IIM jaana worth it hai kya?" is Roman-script
  Hindi-English code-mix, NOT monolingual English.

Design decisions:
- Uses a word-level heuristic approach as a baseline.
- Maintains a small lexicon of common Hindi/Bengali/Odia words written in
  Roman script to distinguish them from English words.
- Script mixing (e.g. Latin + Devanagari) is strong evidence of code-mixing.
- Pure lexicon matching is supplemented by langdetect multi-language output.
- This is a baseline; a trained classifier can replace it later.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import re

from pydantic import BaseModel
from langdetect import detect_langs, DetectorFactory

DetectorFactory.seed = 42


# Common Romanized Hindi words / particles frequently seen in Indian English
# code-mixed text.  This is intentionally conservative — false positives
# (English words that happen to be here) are worse than false negatives.
_HINDI_ROMAN_WORDS = frozenset({
    # function words / particles
    "hai", "hain", "ho", "tha", "thi", "the", "toh", "bhi",
    "ka", "ki", "ke", "ko", "se", "me", "mein", "par",
    "yeh", "ye", "woh", "wo", "kya", "kyu", "kyun", "kaise",
    "nahi", "nhi", "na", "mat", "haan", "ji",
    "aur", "ya", "lekin", "par", "agar", "toh",
    # common verbs / adjectives
    "accha", "achha", "acha", "theek", "thik",
    "karo", "karna", "karke", "karega", "karenge",
    "jao", "jaana", "jaao", "gaya", "gayi", "gaye",
    "aao", "aana", "aata", "aati", "aate",
    "dekho", "dekhna", "dekha", "dekhi",
    "bolo", "bolna", "bola", "boli",
    "padho", "padhna", "padha", "padhai",
    "milta", "milti", "milte", "mila", "mile",
    "chahiye", "chahte", "chahti",
    # common nouns / expressions
    "bhai", "bhaiya", "didi", "yaar", "dost",
    "paisa", "paise", "rupya", "rupaye",
    "sahi", "galat", "bahut", "bohot", "zyada", "kam",
    "abhi", "baad", "pehle", "phir",
    "sab", "kuch", "koi", "kaun",
    "wala", "wali", "wale",
    "kaam", "ghar", "log", "banda", "bande",
    "scene", "jugaad",
})

# Common Romanized Bengali words
_BENGALI_ROMAN_WORDS = frozenset({
    "ki", "keno", "kothay", "kokhon", "kemon",
    "hobe", "holo", "hoye", "hoyeche",
    "ache", "achhe", "nei", "nai",
    "ami", "tumi", "apni", "se", "ora", "tara",
    "bhalo", "bhaloi", "kharap",
    "jani", "jano", "jane",
    "bolo", "bola", "bolcho",
    "koro", "korcho", "korbe",
    "jai", "jao", "jabe", "gelo",
    "dada", "didi", "bhai",
    "ta", "to", "na", "r",
    "ekhane", "okhane", "ekhon",
})

# Common Romanized Odia words
_ODIA_ROMAN_WORDS = frozenset({
    "ki", "kana", "kemiti", "kebe", "kouthi",
    "achi", "achhi", "nahi", "nahin",
    "mu", "tume", "se", "amme", "apana",
    "bhala", "kharap",
    "kara", "kariba", "kale",
    "jiba", "jao", "gala",
    "bhai", "didi", "apa",
    "ta", "to", "na",
    "etha", "setha", "ebe",
    "hau", "na",
})


class CodeMixResult(BaseModel):
    """Result of code-mix detection for a single text."""

    is_code_mixed: bool
    confidence: float
    detected_languages: List[str]
    evidence: Dict[str, float]
    detector_name: str
    detector_version: str


class CodeMixDetector(ABC):
    """Abstract interface for code-mix detection."""

    @abstractmethod
    def detect(self, text: str, script: Optional[str] = None) -> CodeMixResult:
        """Determine whether the text is code-mixed."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        ...


class SimpleCodeMixDetector(CodeMixDetector):
    """
    Baseline code-mix detector using:
    1. Script mixing (Latin + Devanagari etc.) — strong signal.
    2. Romanized-Indic lexicon matching — moderate signal.
    3. langdetect multi-language probabilities — weak supplementary signal.

    Known limitations:
    - Lexicon coverage is intentionally small; recall for rare words is low.
    - Short texts (< 5 words) produce unreliable results.
    - Ambiguous words shared by English and Hindi (e.g. "par") are conservatively
      handled by requiring multiple Indic-word hits.
    """

    _NAME = "lexicon_script_heuristic"
    _VERSION = "1.0.0"

    # Minimum fraction of words matching an Indic lexicon to consider
    # code-mixing when the script is purely Latin.
    _MIN_INDIC_WORD_RATIO = 0.15
    _MIN_INDIC_WORD_COUNT = 2

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def version(self) -> str:
        return self._VERSION

    def detect(self, text: str, script: Optional[str] = None) -> CodeMixResult:
        if not text or not text.strip():
            return CodeMixResult(
                is_code_mixed=False,
                confidence=0.0,
                detected_languages=[],
                evidence={},
                detector_name=self._NAME,
                detector_version=self._VERSION,
            )

        words = self._tokenize(text)
        if not words:
            return CodeMixResult(
                is_code_mixed=False,
                confidence=0.0,
                detected_languages=[],
                evidence={},
                detector_name=self._NAME,
                detector_version=self._VERSION,
            )

        evidence: Dict[str, float] = {}
        detected_langs: List[str] = []

        # --- Signal 1: Script mixing ---
        script_mixed = self._check_script_mixing(text)
        if script_mixed:
            evidence["script_mixing"] = 1.0

        # --- Signal 2: Romanized Indic word matching ---
        hindi_ratio = self._lexicon_ratio(words, _HINDI_ROMAN_WORDS)
        bengali_ratio = self._lexicon_ratio(words, _BENGALI_ROMAN_WORDS)
        odia_ratio = self._lexicon_ratio(words, _ODIA_ROMAN_WORDS)

        hindi_count = self._lexicon_count(words, _HINDI_ROMAN_WORDS)
        bengali_count = self._lexicon_count(words, _BENGALI_ROMAN_WORDS)
        odia_count = self._lexicon_count(words, _ODIA_ROMAN_WORDS)

        evidence["hindi_roman_ratio"] = round(hindi_ratio, 4)
        evidence["bengali_roman_ratio"] = round(bengali_ratio, 4)
        evidence["odia_roman_ratio"] = round(odia_ratio, 4)

        # Check if we have English words too (simple heuristic: words NOT in
        # any Indic lexicon and containing only ASCII alpha)
        all_indic = _HINDI_ROMAN_WORDS | _BENGALI_ROMAN_WORDS | _ODIA_ROMAN_WORDS
        english_words = [
            w for w in words
            if w not in all_indic and w.isascii() and w.isalpha()
        ]
        english_ratio = len(english_words) / len(words) if words else 0.0
        evidence["english_word_ratio"] = round(english_ratio, 4)

        # --- Signal 3: langdetect multi-language ---
        try:
            ld_results = detect_langs(text)
            if len(ld_results) > 1:
                top_two_gap = ld_results[0].prob - ld_results[1].prob
                evidence["langdetect_ambiguity"] = round(1.0 - top_two_gap, 4)
            else:
                evidence["langdetect_ambiguity"] = 0.0
        except Exception:
            evidence["langdetect_ambiguity"] = 0.0

        # --- Decision logic ---
        is_code_mixed = False
        confidence = 0.0

        # Case 1: Script mixing is definitive evidence
        if script_mixed:
            is_code_mixed = True
            confidence = 0.95
            detected_langs = self._infer_languages_from_scripts(text)

        # Case 2: Significant Romanized Indic words + English words
        elif (
            hindi_count >= self._MIN_INDIC_WORD_COUNT
            and hindi_ratio >= self._MIN_INDIC_WORD_RATIO
            and english_ratio >= self._MIN_INDIC_WORD_RATIO
        ):
            is_code_mixed = True
            confidence = min(0.5 + hindi_ratio + english_ratio * 0.5, 0.9)
            detected_langs = ["en", "hi"]

        elif (
            bengali_count >= self._MIN_INDIC_WORD_COUNT
            and bengali_ratio >= self._MIN_INDIC_WORD_RATIO
            and english_ratio >= self._MIN_INDIC_WORD_RATIO
        ):
            is_code_mixed = True
            confidence = min(0.5 + bengali_ratio + english_ratio * 0.5, 0.9)
            detected_langs = ["en", "bn"]

        elif (
            odia_count >= self._MIN_INDIC_WORD_COUNT
            and odia_ratio >= self._MIN_INDIC_WORD_RATIO
            and english_ratio >= self._MIN_INDIC_WORD_RATIO
        ):
            is_code_mixed = True
            confidence = min(0.5 + odia_ratio + english_ratio * 0.5, 0.9)
            detected_langs = ["en", "or"]

        else:
            # Not code-mixed (or insufficient evidence)
            is_code_mixed = False
            confidence = max(0.5, 1.0 - max(hindi_ratio, bengali_ratio, odia_ratio))
            # Determine monolingual label
            if hindi_ratio > 0.4:
                detected_langs = ["hi"]
            elif bengali_ratio > 0.4:
                detected_langs = ["bn"]
            elif odia_ratio > 0.4:
                detected_langs = ["or"]
            elif english_ratio > 0.5:
                detected_langs = ["en"]
            else:
                detected_langs = ["unknown"]

        return CodeMixResult(
            is_code_mixed=is_code_mixed,
            confidence=round(confidence, 4),
            detected_languages=detected_langs,
            evidence=evidence,
            detector_name=self._NAME,
            detector_version=self._VERSION,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer, lowercased."""
        tokens = re.findall(r"[a-zA-Z\u0900-\u0D7F]+", text.lower())
        return tokens

    @staticmethod
    def _lexicon_ratio(words: List[str], lexicon: frozenset) -> float:
        if not words:
            return 0.0
        hits = sum(1 for w in words if w in lexicon)
        return hits / len(words)

    @staticmethod
    def _lexicon_count(words: List[str], lexicon: frozenset) -> int:
        return sum(1 for w in words if w in lexicon)

    @staticmethod
    def _check_script_mixing(text: str) -> bool:
        """Return True if text contains alphabetic characters from more than
        one major script family (Latin + Devanagari, Latin + Bengali, etc.)."""
        has_latin = False
        has_indic = False
        for ch in text:
            cp = ord(ch)
            if not ch.isalpha():
                continue
            if cp <= 0x024F:
                has_latin = True
            elif 0x0900 <= cp <= 0x0D7F:
                has_indic = True
            if has_latin and has_indic:
                return True
        return False

    @staticmethod
    def _infer_languages_from_scripts(text: str) -> List[str]:
        """Given script-mixed text, guess which languages are present."""
        langs = []
        has_latin = any(ch.isalpha() and ord(ch) <= 0x024F for ch in text)
        has_devanagari = any(0x0900 <= ord(ch) <= 0x097F for ch in text)
        has_bengali = any(0x0980 <= ord(ch) <= 0x09FF for ch in text)
        has_odia = any(0x0B00 <= ord(ch) <= 0x0B7F for ch in text)

        if has_latin:
            langs.append("en")
        if has_devanagari:
            langs.append("hi")
        if has_bengali:
            langs.append("bn")
        if has_odia:
            langs.append("or")
        return langs if langs else ["unknown"]
