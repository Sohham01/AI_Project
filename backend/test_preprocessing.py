"""
Comprehensive tests for the multilingual preprocessing pipeline.

Tests verify:
1. original_text is never modified
2. normalized_text is deterministic
3. Romanized Hindi/Bengali/Odia is NOT turned into English
4. Emojis are converted to semantic tokens, not removed
5. Negation is preserved
6. Code-mixed meaning is retained
"""

import unittest

from app.services.nlp.preprocessing import MultilingualTextPreprocessor
from app.services.nlp.preprocessing.unicode_normalizer import normalize_unicode
from app.services.nlp.preprocessing.whitespace_normalizer import normalize_whitespace
from app.services.nlp.preprocessing.repeated_char_normalizer import normalize_repeated_chars
from app.services.nlp.preprocessing.emoji_normalizer import normalize_emojis
from app.services.nlp.preprocessing.slang_normalizer import normalize_slang
from app.services.nlp.preprocessing.punctuation_normalizer import normalize_punctuation
from app.services.nlp.preprocessing.url_mention_normalizer import normalize_urls_mentions


class TestUnicodeNormalizer(unittest.TestCase):
    def test_nfc_normalization(self):
        # Combining character e + combining acute -> precomposed e-acute
        text = "caf\u0065\u0301"
        result = normalize_unicode(text)
        self.assertEqual(result, "caf\u00e9")

    def test_zero_width_removal(self):
        text = "hello\u200bworld"
        result = normalize_unicode(text)
        self.assertEqual(result, "helloworld")

    def test_preserves_devanagari(self):
        text = "\u092f\u0939 \u0915\u0949\u0932\u0947\u091c"
        result = normalize_unicode(text)
        self.assertIn("\u092f", result)

    def test_preserves_bengali(self):
        text = "\u098f\u0987 \u0995\u09b2\u09c7\u099c"
        result = normalize_unicode(text)
        self.assertIn("\u098f", result)


class TestWhitespaceNormalizer(unittest.TestCase):
    def test_multiple_spaces(self):
        self.assertEqual(normalize_whitespace("this    has   spaces"), "this has spaces")

    def test_tabs(self):
        self.assertEqual(normalize_whitespace("hello\tworld"), "hello world")

    def test_multiple_newlines(self):
        result = normalize_whitespace("a\n\n\n\nb")
        self.assertEqual(result, "a\n\nb")

    def test_strip(self):
        self.assertEqual(normalize_whitespace("  hello  "), "hello")


class TestRepeatedCharNormalizer(unittest.TestCase):
    def test_soooo(self):
        self.assertEqual(normalize_repeated_chars("soooo good"), "soo good")

    def test_broooo(self):
        self.assertEqual(normalize_repeated_chars("broooo"), "broo")

    def test_two_chars_unchanged(self):
        self.assertEqual(normalize_repeated_chars("really??"), "really??")

    def test_many_exclamation(self):
        self.assertEqual(normalize_repeated_chars("what!!!!!!"), "what!!")

    def test_normal_word(self):
        self.assertEqual(normalize_repeated_chars("good"), "good")

    def test_goooood(self):
        # "goooood" -> "good" (3+ o's -> 2, then 'oo' is natural)
        self.assertEqual(normalize_repeated_chars("goooood"), "good")


class TestEmojiNormalizer(unittest.TestCase):
    def test_crying_emoji(self):
        result = normalize_emojis("sad \U0001f62d")
        self.assertIn("<EMOJI_loudly_crying_face>", result)
        self.assertIn("sad", result)

    def test_fire_emoji(self):
        result = normalize_emojis("amazing \U0001f525")
        self.assertIn("<EMOJI_fire>", result)

    def test_multiple_emojis(self):
        result = normalize_emojis("\U0001f62d\U0001f525")
        self.assertIn("<EMOJI_loudly_crying_face>", result)
        self.assertIn("<EMOJI_fire>", result)

    def test_no_emoji(self):
        text = "just regular text"
        self.assertEqual(normalize_emojis(text), text)

    def test_emoji_not_removed(self):
        result = normalize_emojis("wow \U0001f602")
        # Emoji should be replaced with token, NOT removed
        self.assertTrue(len(result) > len("wow "))

    # --- Regression: ASCII punctuation must never be classified as emoji ---

    def test_question_marks_not_emoji(self):
        self.assertEqual(normalize_emojis("really??"), "really??")

    def test_exclamation_marks_not_emoji(self):
        self.assertEqual(normalize_emojis("what!!!"), "what!!!")

    def test_ascii_punctuation_not_emoji(self):
        text = "wow! really? ok... yes, no: fine; done"
        self.assertEqual(normalize_emojis(text), text)

    def test_ascii_punctuation_alongside_emoji(self):
        result = normalize_emojis("\U0001f525 what?!")
        self.assertIn("<EMOJI_fire>", result)
        self.assertTrue(result.endswith("what?!"))

    def test_variation_selector_sequence(self):
        # Red heart + VS16 must yield a single token, no stray VS16
        result = normalize_emojis("love \u2764\ufe0f you")
        self.assertEqual(result, "love <EMOJI_red_heart> you")

    def test_zwj_sequence_single_token(self):
        result = normalize_emojis("ship \U0001f9d1\u200d\U0001f4bb it")
        self.assertIn("<EMOJI_", result)
        self.assertNotIn("\u200d", result)
        self.assertNotIn("\ufe0f", result)


class TestSlangNormalizer(unittest.TestCase):
    def test_idk(self):
        self.assertEqual(normalize_slang("idk what to do"), "i don't know what to do")

    def test_tbh(self):
        self.assertEqual(normalize_slang("tbh this is good"), "to be honest this is good")

    def test_bro_preserved(self):
        # "bro" is NOT in the lexicon, should pass through
        self.assertEqual(normalize_slang("bro this is crazy"), "bro this is crazy")

    def test_bhai_preserved(self):
        # "bhai" is NOT in the lexicon, should pass through
        self.assertEqual(normalize_slang("bhai kya scene hai"), "bhai kya scene hai")

    def test_lol_preserved(self):
        self.assertEqual(normalize_slang("lol same"), "lol same")

    def test_with_punctuation(self):
        result = normalize_slang("idk!")
        self.assertIn("i don't know", result)


class TestPunctuationNormalizer(unittest.TestCase):
    def test_ellipsis(self):
        self.assertEqual(normalize_punctuation("wait...."), "wait...")

    def test_smart_quotes(self):
        self.assertEqual(normalize_punctuation("\u201chello\u201d"), '"hello"')

    def test_em_dash(self):
        result = normalize_punctuation("word\u2014word")
        self.assertIn(" - ", result)

    def test_question_marks_preserved(self):
        # Double ?? should be preserved (sentiment cue)
        self.assertEqual(normalize_punctuation("really??"), "really??")

    def test_exclamation_preserved(self):
        self.assertEqual(normalize_punctuation("what!!"), "what!!")


class TestUrlMentionNormalizer(unittest.TestCase):
    def test_url(self):
        result = normalize_urls_mentions("check https://example.com please")
        self.assertEqual(result, "check <URL> please")

    def test_www_url(self):
        result = normalize_urls_mentions("visit www.example.com")
        self.assertEqual(result, "visit <URL>")

    def test_mention(self):
        result = normalize_urls_mentions("ask u/someone")
        self.assertEqual(result, "ask <MENTION>")

    def test_mention_with_slash(self):
        result = normalize_urls_mentions("ask /u/someone")
        self.assertEqual(result, "ask <MENTION>")

    def test_subreddit_preserved(self):
        result = normalize_urls_mentions("check r/IIM")
        self.assertEqual(result, "check r/IIM")


class TestMultilingualPreprocessor(unittest.TestCase):
    def setUp(self):
        self.pp = MultilingualTextPreprocessor()

    def test_version(self):
        self.assertEqual(self.pp.version, "v1")

    # --- original_text preservation ---

    def test_original_text_unchanged(self):
        original = "broooo \U0001f62d IIM ka placement kya scene hai???"
        result = self.pp.preprocess(original)
        self.assertEqual(result.original_text, original)

    def test_original_text_unchanged_hindi(self):
        original = "\u092f\u0939 \u0915\u0949\u0932\u0947\u091c \u0915\u0948\u0938\u093e \u0939\u0948"
        result = self.pp.preprocess(original, language="hi", script="devanagari")
        self.assertEqual(result.original_text, original)

    # --- deterministic ---

    def test_deterministic(self):
        text = "soooo good \U0001f525 idk bro"
        r1 = self.pp.preprocess(text)
        r2 = self.pp.preprocess(text)
        self.assertEqual(r1.normalized_text, r2.normalized_text)

    # --- English ---

    def test_english_basic(self):
        result = self.pp.preprocess("this placement process is bad", language="en")
        self.assertEqual(result.normalized_text, "this placement process is bad")

    # --- Hindi Devanagari ---

    def test_hindi_devanagari(self):
        text = "\u092f\u0939 \u0915\u0949\u0932\u0947\u091c \u0915\u0948\u0938\u093e \u0939\u0948"
        result = self.pp.preprocess(text, language="hi", script="devanagari")
        # Hindi text should be preserved, not translated
        self.assertIn("\u092f\u0939", result.normalized_text)
        self.assertIn("\u0915\u0949\u0932\u0947\u091c", result.normalized_text)

    # --- Romanized Hindi ---

    def test_romanized_hindi(self):
        text = "IIM jaana worth it hai kya?"
        result = self.pp.preprocess(text, language="hi", script="latin", is_code_mixed=True)
        # Must NOT translate; Romanized Hindi words should remain
        self.assertIn("jaana", result.normalized_text)
        self.assertIn("hai", result.normalized_text)
        self.assertIn("kya", result.normalized_text)

    # --- Code-mixed Hindi + English ---

    def test_code_mixed(self):
        text = "placement ka scene confusing hai"
        result = self.pp.preprocess(
            text, language="hi", script="latin",
            is_code_mixed=True, detected_languages=["hi", "en"],
        )
        # Both English and Hindi words should survive
        self.assertIn("placement", result.normalized_text)
        self.assertIn("confusing", result.normalized_text)
        self.assertIn("hai", result.normalized_text)

    # --- Bengali ---

    def test_bengali(self):
        text = "\u098f\u0987 \u0995\u09b2\u09c7\u099c \u0995\u09c7\u09ae\u09a8?"
        result = self.pp.preprocess(text, language="bn", script="bengali")
        self.assertIn("\u0995\u09b2\u09c7\u099c", result.normalized_text)

    # --- Odia ---

    def test_odia(self):
        text = "\u0b0f\u0b39\u0b3f \u0b15\u0b32\u0b47\u0b1c \u0b15\u0b47\u0b2e\u0b3f\u0b24\u0b3f?"
        result = self.pp.preprocess(text, language="or", script="odia")
        self.assertIn("\u0b15\u0b32\u0b47\u0b1c", result.normalized_text)

    # --- Emoji ---

    def test_emoji_preserved_as_token(self):
        text = "This is amazing \U0001f62d\U0001f525"
        result = self.pp.preprocess(text)
        self.assertIn("<EMOJI_loudly_crying_face>", result.normalized_text)
        self.assertIn("<EMOJI_fire>", result.normalized_text)
        self.assertIn("amazing", result.normalized_text)

    # --- Repeated characters ---

    def test_repeated_chars(self):
        result = self.pp.preprocess("soooo good")
        self.assertEqual(result.normalized_text, "soo good")

    # --- Slang ---

    def test_slang_expansion(self):
        result = self.pp.preprocess("bro this is crazy lol")
        # "bro" and "lol" should stay as-is (not in lexicon)
        self.assertIn("bro", result.normalized_text)
        self.assertIn("lol", result.normalized_text)
        self.assertIn("crazy", result.normalized_text)

    # --- Negation preserved ---

    def test_negation_preserved(self):
        result = self.pp.preprocess("this is not good")
        self.assertIn("not", result.normalized_text)
        self.assertIn("good", result.normalized_text)

    def test_negation_contraction(self):
        result = self.pp.preprocess("I don't think this is right")
        self.assertIn("don't", result.normalized_text)

    # --- Punctuation ---

    def test_question_marks(self):
        result = self.pp.preprocess("really??")
        self.assertIn("??", result.normalized_text)

    def test_exclamation(self):
        result = self.pp.preprocess("what!!!")
        # 3+ -> 2 by repeated char normalizer
        self.assertIn("!!", result.normalized_text)
        self.assertNotIn("<EMOJI_question_mark>", result.normalized_text)
        self.assertNotIn("<EMOJI_exclamation_mark>", result.normalized_text)

    def test_question_marks_not_emoji_token(self):
        result = self.pp.preprocess("really??")
        self.assertEqual(result.normalized_text, "really??")
        self.assertNotIn("<EMOJI_", result.normalized_text)

    # --- URLs ---

    def test_url_replaced(self):
        result = self.pp.preprocess("check https://example.com")
        self.assertIn("<URL>", result.normalized_text)
        self.assertNotIn("https://", result.normalized_text)

    # --- Whitespace ---

    def test_whitespace(self):
        result = self.pp.preprocess("this    has   spaces")
        self.assertEqual(result.normalized_text, "this has spaces")

    # --- Mixed complex example ---

    def test_mixed_complex(self):
        text = "bhaiiii \U0001f62d placement ka kya scene hai???"
        result = self.pp.preprocess(
            text, language="hi", script="latin",
            is_code_mixed=True, detected_languages=["hi", "en"],
        )
        # "bhaiiii" -> "bhaii" (repeated char)
        self.assertIn("bhaii", result.normalized_text)
        # emoji preserved as token
        self.assertIn("<EMOJI_", result.normalized_text)
        # Hindi words preserved
        self.assertIn("kya", result.normalized_text)
        self.assertIn("scene", result.normalized_text)
        self.assertIn("hai", result.normalized_text)
        # Excessive ? reduced
        self.assertIn("??", result.normalized_text)
        # original unchanged
        self.assertEqual(result.original_text, text)

    # --- Empty / None ---

    def test_empty_string(self):
        result = self.pp.preprocess("")
        self.assertEqual(result.normalized_text, "")
        self.assertEqual(result.original_text, "")

    def test_none_input(self):
        result = self.pp.preprocess(None)
        self.assertEqual(result.normalized_text, "")

    # --- Preprocessing version ---

    def test_preprocessing_version(self):
        result = self.pp.preprocess("test")
        self.assertEqual(result.preprocessing_version, "v1")

    # --- Stages applied ---

    def test_stages_applied(self):
        result = self.pp.preprocess("test text")
        self.assertIn("unicode", result.stages_applied)
        self.assertIn("emoji", result.stages_applied)
        self.assertIn("slang", result.stages_applied)
        self.assertIn("whitespace", result.stages_applied)


if __name__ == "__main__":
    unittest.main()
