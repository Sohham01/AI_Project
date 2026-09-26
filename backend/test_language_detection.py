"""
Comprehensive tests for the multilingual NLP detection layer.

Tests cover:
- Language detection (SimpleLanguageDetector)
- Script detection (SimpleScriptDetector)
- Code-mix detection (SimpleCodeMixDetector)
- Combined LanguageAnalysisService
- Edge cases: empty, short, emoji, punctuation, ambiguous text
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app.services.nlp.language import SimpleLanguageDetector
from app.services.nlp.script_detector import SimpleScriptDetector
from app.services.nlp.code_mix_detector import SimpleCodeMixDetector
from app.services.nlp.language_analysis_service import LanguageAnalysisService


class TestSimpleLanguageDetector(unittest.TestCase):

    def setUp(self):
        self.detector = SimpleLanguageDetector()

    def test_english(self):
        result = self.detector.detect("Is IIM Sambalpur worth it for MBA?")
        self.assertEqual(result.language, "en")
        self.assertGreater(result.confidence, 0.5)

    def test_hindi_devanagari(self):
        result = self.detector.detect("क्या IIM संबलपुर अच्छा है?")
        self.assertIn(result.language, ["hi", "mr"])
        self.assertGreater(result.confidence, 0.3)

    def test_bengali(self):
        result = self.detector.detect("আইআইএম সম্বলপুর কেমন?")
        self.assertEqual(result.language, "bn")
        self.assertGreater(result.confidence, 0.3)

    def test_empty_text(self):
        result = self.detector.detect("")
        self.assertEqual(result.language, "unknown")
        self.assertEqual(result.confidence, 0.0)

    def test_whitespace_only(self):
        result = self.detector.detect("   ")
        self.assertEqual(result.language, "unknown")
        self.assertEqual(result.confidence, 0.0)

    def test_detector_metadata(self):
        result = self.detector.detect("Hello world")
        self.assertEqual(result.detector_name, "langdetect")
        self.assertEqual(result.detector_version, "1.0.9")

    def test_all_detected_populated(self):
        result = self.detector.detect("This is a longer English sentence for testing purposes.")
        self.assertIsInstance(result.all_detected, list)
        self.assertGreater(len(result.all_detected), 0)


class TestSimpleScriptDetector(unittest.TestCase):

    def setUp(self):
        self.detector = SimpleScriptDetector()

    def test_latin(self):
        result = self.detector.detect("IIM Sambalpur is good")
        self.assertEqual(result.script, "latin")
        self.assertGreater(result.confidence, 0.9)

    def test_devanagari(self):
        result = self.detector.detect("आईआईएम संबलपुर अच्छा है")
        self.assertEqual(result.script, "devanagari")
        self.assertGreater(result.confidence, 0.9)

    def test_bengali(self):
        result = self.detector.detect("আইআইএম সম্বলপুর কেমন")
        self.assertEqual(result.script, "bengali")
        self.assertGreater(result.confidence, 0.9)

    def test_odia(self):
        result = self.detector.detect("ଆଇଆଇଏମ ସମ୍ବଲପୁର")
        self.assertEqual(result.script, "odia")
        self.assertGreater(result.confidence, 0.9)

    def test_mixed_latin_devanagari(self):
        result = self.detector.detect("IIM Sambalpur अच्छा है क्या?")
        # Should detect both scripts; dominant depends on char count
        self.assertIn("latin", result.script_counts)
        self.assertIn("devanagari", result.script_counts)
        # Confidence should be < 1.0 for mixed
        self.assertLess(result.confidence, 1.0)

    def test_empty_text(self):
        result = self.detector.detect("")
        self.assertEqual(result.script, "unknown")
        self.assertEqual(result.confidence, 0.0)

    def test_punctuation_only(self):
        result = self.detector.detect("!!! ???  ...")
        self.assertEqual(result.script, "unknown")

    def test_emoji_only(self):
        result = self.detector.detect("😭😂🔥💯")
        self.assertEqual(result.script, "unknown")

    def test_detector_metadata(self):
        result = self.detector.detect("test")
        self.assertEqual(result.detector_name, "unicode_range")
        self.assertEqual(result.detector_version, "1.0.0")


class TestSimpleCodeMixDetector(unittest.TestCase):

    def setUp(self):
        self.detector = SimpleCodeMixDetector()

    def test_english_monolingual(self):
        result = self.detector.detect("IIM is good for placements and MBA")
        self.assertFalse(result.is_code_mixed)

    def test_hindi_english_codemix_roman(self):
        result = self.detector.detect("IIM jaana worth it hai kya bhai?")
        self.assertTrue(result.is_code_mixed)
        self.assertIn("hi", result.detected_languages)
        self.assertIn("en", result.detected_languages)

    def test_script_mixing_latin_devanagari(self):
        result = self.detector.detect("IIM Sambalpur अच्छा है क्या?")
        self.assertTrue(result.is_code_mixed)
        self.assertGreater(result.confidence, 0.8)

    def test_pure_devanagari_not_codemixed(self):
        result = self.detector.detect("आईआईएम संबलपुर अच्छा है")
        self.assertFalse(result.is_code_mixed)

    def test_empty_text(self):
        result = self.detector.detect("")
        self.assertFalse(result.is_code_mixed)
        self.assertEqual(result.detected_languages, [])

    def test_short_text(self):
        result = self.detector.detect("ok")
        self.assertFalse(result.is_code_mixed)

    def test_emoji_heavy(self):
        result = self.detector.detect("😭😭😭 placement kya scene hai bhai 🔥")
        # Has Hindi words mixed with English
        self.assertTrue(result.is_code_mixed)

    def test_detector_metadata(self):
        result = self.detector.detect("test text")
        self.assertEqual(result.detector_name, "lexicon_script_heuristic")
        self.assertEqual(result.detector_version, "1.0.0")


class TestLanguageAnalysisService(unittest.TestCase):

    def setUp(self):
        self.service = LanguageAnalysisService(
            language_detector=SimpleLanguageDetector(),
            script_detector=SimpleScriptDetector(),
            code_mix_detector=SimpleCodeMixDetector(),
        )

    def test_english(self):
        result = self.service.analyze("Is IIM Sambalpur worth it for MBA?")
        self.assertEqual(result.script, "latin")
        self.assertFalse(result.is_code_mixed)

    def test_hindi_english_codemix(self):
        result = self.service.analyze("IIM jaana worth it hai kya bhai?")
        self.assertEqual(result.script, "latin")
        self.assertTrue(result.is_code_mixed)
        self.assertIn("hi", result.detected_languages)
        self.assertIn("en", result.detected_languages)

    def test_devanagari_hindi(self):
        result = self.service.analyze("आईआईएम संबलपुर अच्छा है")
        self.assertEqual(result.script, "devanagari")
        self.assertFalse(result.is_code_mixed)

    def test_mixed_script(self):
        result = self.service.analyze("IIM Sambalpur अच्छा है क्या?")
        self.assertTrue(result.is_code_mixed)

    def test_bengali(self):
        result = self.service.analyze("আইআইএম সম্বলপুর কেমন?")
        self.assertEqual(result.script, "bengali")

    def test_odia(self):
        result = self.service.analyze("ଆଇଆଇଏମ ସମ୍ବଲପୁର")
        self.assertEqual(result.script, "odia")

    def test_empty(self):
        result = self.service.analyze("")
        self.assertEqual(result.language, "unknown")
        self.assertEqual(result.script, "unknown")
        self.assertFalse(result.is_code_mixed)

    def test_combined_detector_name(self):
        result = self.service.analyze("Hello")
        self.assertIn("+", result.detector_name)
        self.assertIn("+", result.detector_version)

    def test_sub_results_present(self):
        result = self.service.analyze("Test text for analysis")
        self.assertIsNotNone(result.language_detail)
        self.assertIsNotNone(result.script_detail)
        self.assertIsNotNone(result.code_mix_detail)


if __name__ == "__main__":
    unittest.main()
