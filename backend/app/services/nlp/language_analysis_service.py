"""
Combined language analysis service.

Orchestrates:
  language detection -> script detection -> code-mix detection

Each sub-detector is injected via its abstract interface so implementations
can be swapped without changing the orchestration logic.
"""

from typing import List, Optional

from pydantic import BaseModel

from app.services.nlp.language import LanguageDetector, LanguageResult
from app.services.nlp.script_detector import ScriptDetector, ScriptResult
from app.services.nlp.code_mix_detector import CodeMixDetector, CodeMixResult


class FullLanguageAnalysis(BaseModel):
    """Combined result of all language-related analysis."""

    language: str
    language_confidence: float
    script: str
    script_confidence: float
    is_code_mixed: bool
    detected_languages: List[str]
    detector_name: str
    detector_version: str

    # Sub-results for detailed inspection
    language_detail: LanguageResult
    script_detail: ScriptResult
    code_mix_detail: CodeMixResult


class LanguageAnalysisService:
    """
    Orchestrates language, script, and code-mix detection.

    Usage:
        from app.services.nlp.language import SimpleLanguageDetector
        from app.services.nlp.script_detector import SimpleScriptDetector
        from app.services.nlp.code_mix_detector import SimpleCodeMixDetector

        service = LanguageAnalysisService(
            language_detector=SimpleLanguageDetector(),
            script_detector=SimpleScriptDetector(),
            code_mix_detector=SimpleCodeMixDetector(),
        )
        result = service.analyze("IIM jaana worth it hai kya?")
    """

    _VERSION = "1.0.0"

    def __init__(
        self,
        language_detector: LanguageDetector,
        script_detector: ScriptDetector,
        code_mix_detector: CodeMixDetector,
    ) -> None:
        self._lang = language_detector
        self._script = script_detector
        self._codemix = code_mix_detector

    def analyze(self, text: str) -> FullLanguageAnalysis:
        """Run all three detectors and return a combined result."""
        lang_result = self._lang.detect(text)
        script_result = self._script.detect(text)
        codemix_result = self._codemix.detect(text, script=script_result.script)

        # Reconcile: if code-mix detector found languages, use those;
        # otherwise fall back to the language detector's output.
        if codemix_result.detected_languages and codemix_result.detected_languages != ["unknown"]:
            detected_languages = codemix_result.detected_languages
        else:
            detected_languages = [lang_result.language]

        # Primary language: prefer code-mix detected if code-mixed,
        # else use language detector result
        if codemix_result.is_code_mixed and len(detected_languages) > 0:
            primary_language = detected_languages[0]
        else:
            primary_language = lang_result.language

        combined_detector = (
            f"{self._lang.name}+{self._script.name}+{self._codemix.name}"
        )
        combined_version = (
            f"{self._lang.version}+{self._script.version}+{self._codemix.version}"
        )

        return FullLanguageAnalysis(
            language=primary_language,
            language_confidence=lang_result.confidence,
            script=script_result.script,
            script_confidence=script_result.confidence,
            is_code_mixed=codemix_result.is_code_mixed,
            detected_languages=detected_languages,
            detector_name=combined_detector,
            detector_version=combined_version,
            language_detail=lang_result,
            script_detail=script_result,
            code_mix_detail=codemix_result,
        )
