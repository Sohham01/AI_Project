"""
Multilingual text preprocessor.

Composes all normalization stages into a single deterministic pipeline:

1. Unicode normalization
2. URL / mention normalization
3. Whitespace normalization
4. Repeated character normalization
5. Emoji normalization
6. Slang normalization
7. Punctuation normalization
8. Final whitespace cleanup

The pipeline accepts language analysis metadata so that code-mixed and
non-English texts are handled correctly — no automatic translation,
no destruction of Romanized Hindi/Bengali/Odia content.

The original_text is NEVER modified; only normalized_text is produced.
"""

from typing import List, Optional

from app.services.nlp.preprocessing.base import PreprocessingResult, TextPreprocessor
from app.services.nlp.preprocessing.unicode_normalizer import normalize_unicode
from app.services.nlp.preprocessing.url_mention_normalizer import normalize_urls_mentions
from app.services.nlp.preprocessing.whitespace_normalizer import normalize_whitespace
from app.services.nlp.preprocessing.repeated_char_normalizer import normalize_repeated_chars
from app.services.nlp.preprocessing.emoji_normalizer import normalize_emojis
from app.services.nlp.preprocessing.slang_normalizer import normalize_slang, SLANG_LEXICON_VERSION
from app.services.nlp.preprocessing.punctuation_normalizer import normalize_punctuation


class MultilingualTextPreprocessor(TextPreprocessor):
    """
    Production preprocessor that runs all normalization stages in order.

    Usage:
        preprocessor = MultilingualTextPreprocessor()
        result = preprocessor.preprocess(
            text="bhaiiii 😭 placement ka kya scene hai???",
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
        )
        print(result.normalized_text)
    """

    _VERSION = "v1"

    @property
    def version(self) -> str:
        return self._VERSION

    def preprocess(
        self,
        text: str,
        language: Optional[str] = None,
        script: Optional[str] = None,
        is_code_mixed: bool = False,
        detected_languages: Optional[List[str]] = None,
    ) -> PreprocessingResult:
        if text is None:
            text = ""

        original = text
        stages: List[str] = []

        # 1. Unicode normalization
        text = normalize_unicode(text)
        stages.append("unicode")

        # 2. URL / mention normalization
        text = normalize_urls_mentions(text)
        stages.append("url_mention")

        # 3. Whitespace normalization
        text = normalize_whitespace(text)
        stages.append("whitespace")

        # 4. Repeated character normalization
        text = normalize_repeated_chars(text)
        stages.append("repeated_char")

        # 5. Emoji normalization (emojis -> semantic tokens)
        text = normalize_emojis(text)
        stages.append("emoji")

        # 6. Slang normalization
        text = normalize_slang(text)
        stages.append("slang")

        # 7. Punctuation normalization
        text = normalize_punctuation(text)
        stages.append("punctuation")

        # 8. Final whitespace cleanup (stages above may introduce extra spaces)
        text = normalize_whitespace(text)
        stages.append("final_whitespace")

        return PreprocessingResult(
            original_text=original,
            normalized_text=text,
            preprocessing_version=self._VERSION,
            stages_applied=stages,
        )
