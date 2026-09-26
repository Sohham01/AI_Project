"""
Multilingual text preprocessing pipeline.

Stages (applied in order):
1. Unicode normalization
2. URL / mention handling
3. Whitespace normalization
4. Repeated character normalization
5. Emoji normalization
6. Slang normalization
7. Punctuation normalization

Design principles:
- original_text is NEVER modified
- normalized_text is deterministic
- code-mixed text is preserved, not translated
- emojis are converted to semantic tokens, not removed
- negation is preserved
- preprocessing version is tracked for reproducibility
"""

from app.services.nlp.preprocessing.base import (
    PreprocessingResult,
    TextPreprocessor,
)
from app.services.nlp.preprocessing.multilingual_preprocessor import (
    MultilingualTextPreprocessor,
)

__all__ = [
    "PreprocessingResult",
    "TextPreprocessor",
    "MultilingualTextPreprocessor",
]
