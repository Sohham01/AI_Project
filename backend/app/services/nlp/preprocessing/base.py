"""
Base interfaces and result model for the preprocessing pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel


class PreprocessingResult(BaseModel):
    """Result of preprocessing a single text."""

    original_text: str
    normalized_text: str
    preprocessing_version: str
    stages_applied: List[str]


class TextPreprocessor(ABC):
    """Abstract interface for text preprocessing."""

    @abstractmethod
    def preprocess(
        self,
        text: str,
        language: Optional[str] = None,
        script: Optional[str] = None,
        is_code_mixed: bool = False,
        detected_languages: Optional[List[str]] = None,
    ) -> PreprocessingResult:
        """Preprocess text and return a structured result."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the version string for this preprocessor."""
        ...
