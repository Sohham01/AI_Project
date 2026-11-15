"""
Comprehensive tests for the sentiment and emotion analysis stage.

Tests cover:
- The separation of the two dimensions: emotion is never inferred from
  sentiment, and neither label set is hard-coded
- Sentiment: positive / negative / neutral, probability distribution,
  confidence, abstention threshold, configuration, determinism
- Emotion: single emotion, multi-emotion, independent sigmoids (not softmax),
  configurable threshold, unseen labels, determinism
- Annotation models and both annotation CSV schemas
- Training refusals: empty corpus, missing labels, too few examples,
  insufficient label variation, unusable TF-IDF vocabulary
- Evaluation: single-label metrics + confusion matrix, multi-label metrics,
  and the meaningful-data guards that withhold metrics instead of inventing them
- Multilingual and code-mixed text: English, Hindi Devanagari, Romanized Hindi,
  Bengali, Odia, Hinglish, emoji semantic tokens
- Negation survival, placeholder handling, input immutability, serialization
- Architectural isolation: no preprocessing, no database, no coupling to
  earlier stages
"""

import ast
import inspect
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(__file__))

from app.services.nlp.sentiment_emotion import (
    DEFAULT_EMOTION_CSV,
    DEFAULT_SENTIMENT_CSV,
    EMOTION_CSV_COLUMNS,
    SENTIMENT_CSV_COLUMNS,
    EmotionAnalysis,
    EmotionAnnotation,
    EmotionAnalysisResult,
    EmotionCorpus,
    EmotionEvaluationReport,
    EmotionModelConfig,
    EmotionPrediction,
    EmotionTrainingReport,
    SentimentAnalysis,
    SentimentAnalysisResult,
    SentimentAnnotation,
    SentimentCorpus,
    SentimentEmotionDocument,
    SentimentEvaluationReport,
    SentimentModelConfig,
    SentimentPrediction,
    SentimentTrainingReport,
    TfidfLogisticEmotionClassifier,
    TfidfLogisticSentimentClassifier,
    evaluate_emotions,
    evaluate_sentiment,
)

PACKAGE_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "app",
    "services",
    "nlp",
    "sentiment_emotion",
)


# ---------------------------------------------------------------------------
# Synthetic annotation fixtures.
#
# SYNTHETIC, illustrative data written to exercise the architecture. It was not
# collected from Reddit, it was not annotated by real annotators, and no metric
# in this file is project performance.
#
# The class / label values below are simply what a fictional annotation
# guideline happened to use. Nothing in the package knows them in advance.
# ---------------------------------------------------------------------------

# (document_id, preprocessed text, sentiment, language, script, is_code_mixed)
SENTIMENT_FIXTURE = [
    # --- positive ---
    ("s01", "this placement season is amazing", "positive", "en", "latin", False),
    ("s02", "what a wonderful and memorable experience", "positive", "en", "latin", False),
    ("s03", "the campus fest was great and so much fun", "positive", "en", "latin", False),
    ("s04", "teachers here are brilliant and very helpful", "positive", "en", "latin", False),
    ("s05", "the campus is beautiful and the library is lovely", "positive", "en", "latin", False),
    ("s06", "maine bahut achha experience paaya yahan", "positive", "hi", "latin", True),
    ("s07", "placement mila aur khushi hui bahut", "positive", "hi", "latin", True),
    ("s08", "what a great fest <EMOJI_tada>", "positive", "en", "latin", False),
    ("s09", "i am not worried, everything is really great", "positive", "en", "latin", False),
    ("s10", "আমাদের প্রার্থীদের উজ্জ্বল ভবিষ্যৎ দেখে সত্যিই ভালো লাগল", "positive", "bn", "bengali", True),
    ("s11", "আমি খুব খুশি এবং কৃতজ্ঞ", "positive", "bn", "bengali", False),
    ("s12", "আমার খুব সুন্দর অভিজ্ঞতা হয়েছে", "positive", "bn", "bengali", False),
    ("s13", "ମୁଁ ବହୁତ ଖୁସି", "positive", "or", "oriya", False),
    # --- negative ---
    ("n01", "this placement process is horrible", "negative", "en", "latin", False),
    ("n02", "the worst mess food i have ever had", "negative", "en", "latin", False),
    ("n03", "fees are ridiculously bad and totally unfair", "negative", "en", "latin", False),
    ("n04", "hostel maintenance is terrible and filthy", "negative", "en", "latin", False),
    ("n05", "exam pressure ne mujhe puri tarah tod diya", "negative", "hi", "devanagari", False),
    ("n06", "hostel ka khana bahut ghatiya aur ganda hai", "negative", "hi", "latin", True),
    ("n07", "mess food is not good at all here", "negative", "en", "latin", False),
    ("n08", "this is horrible <EMOJI_pensive_face> terrible mess", "negative", "en", "latin", False),
    ("n09", "hostel থেকে যে অভিজ্ঞতা পেলাম তা সত্যিই খারাপ", "negative", "bn", "bengali", True),
    ("n10", "বিষয় নিয়ে আমাদের দুজনের মধ্যে অনেক কষ্ট হলো", "negative", "bn", "bengali", False),
    ("n11", "ସେବା ଓ ରୋମ୍ ରକ୍ଷଣାର ଅବସ୍ଥା ବହୁତ ଖରାପ", "negative", "or", "oriya", True),
    # --- neutral ---
    ("t01", "placement process hai bas", "neutral", "hi", "latin", True),
    ("t02", "the exam was scheduled on monday", "neutral", "en", "latin", False),
    ("t03", "the hostel room is on the second floor", "neutral", "en", "latin", False),
    ("t04", "the class starts at nine in the morning", "neutral", "en", "latin", False),
    ("t05", "i attended the workshop yesterday", "neutral", "en", "latin", False),
    ("t06", "semester 2 fees were paid", "neutral", "hi", "latin", True),
    ("t07", "there is not much difference here", "neutral", "en", "latin", False),
    ("t08", "the seminar at <URL> starts today", "neutral", "en", "latin", False),
    ("t09", "সেমিনারটা আজ হবে না নিশ্চিত করা হয়নি", "neutral", "bn", "bengali", True),
    ("t10", "কীভাবে নীলা প্রোগ্রাম হিসেবে দেখব", "neutral", "bn", "bengali", False),
    ("t11", "ହସ୍ଟେଲ ରୋମ ନମ୍ବର ବାର୍ତ୍ତା ମୋତେ ଜଣା ନାହିଁ", "neutral", "or", "oriya", True),
]

# (document_id, preprocessed text, emotions, language, script, is_code_mixed)
EMOTION_FIXTURE = [
    # --- frustration ---
    ("f01", "this is so frustrating and stressful", ["frustration", "stress"], "en", "latin", False),
    ("f02", "i am frustrated with the mess food quality", ["frustration"], "en", "latin", False),
    ("f03", "so frustrating that nobody helps here", ["frustration", "anger"], "en", "latin", False),
    ("f04", "frustrating process and very stressful <EMOJI_loudly_crying_face>", ["frustration", "stress"], "en", "latin", False),
    # --- stress / anxiety ---
    ("x01", "placement season is so stressful", ["stress"], "en", "latin", False),
    ("x02", "exam stress aur pressure bahut zyada hai", ["stress", "anxiety"], "hi", "latin", True),
    ("x03", "this workload is incredibly stressful <EMOJI_tired_face>", ["stress"], "en", "latin", False),
    ("x04", "i am nervous and anxious about the interviews", ["anxiety"], "en", "latin", False),
    ("x05", "i am so anxious and worried about the fees", ["anxiety", "stress"], "en", "latin", False),
    ("x06", "chinta bahut ho rahi hai placement ki", ["anxiety"], "hi", "devanagari", False),
    # --- anger ---
    ("a01", "i am so angry about this delay", ["anger"], "en", "latin", False),
    ("a02", "this is unacceptable and i am furious", ["anger", "frustration"], "en", "latin", False),
    ("a03", "angry that nobody responded at all", ["anger"], "en", "latin", False),
    # --- joy / gratitude ---
    ("j01", "i am so happy about the placement", ["joy"], "en", "latin", False),
    ("j02", "what a wonderful day, i am delighted", ["joy", "gratitude"], "en", "latin", False),
    ("j03", "happy and excited for the results", ["joy"], "en", "latin", False),
    ("g01", "i am grateful to my mentors for the help", ["gratitude"], "en", "latin", False),
    ("g02", "thankful and grateful for the support", ["gratitude", "joy"], "en", "latin", False),
    ("g03", "i am grateful to my friends here", ["gratitude"], "en", "latin", False),
    # --- Bengali / Odia ---
    ("b01", "আমি খুব উদ্বিগ্ন এবং চাপে ভরা", ["anxiety", "stress"], "bn", "bengali", False),
    ("b02", "আমি খুব খুশি এবং কৃতজ্ঞ", ["joy", "gratitude"], "bn", "bengali", False),
    ("o01", "ମୁଁ ବହୁତ କାଷ୍ଟିରେ ପଡିଛି", ["frustration"], "or", "odia", False),
    # --- no emotion at all (valid negative supervision) ---
    ("z01", "the mess serves rice and dal", [], "en", "latin", False),
    ("z02", "placement process hai bas", [], "hi", "latin", True),
    ("z03", "the exam was on monday", [], "en", "latin", False),
    ("z04", "i attended the workshop yesterday", [], "en", "latin", False),
    # --- the brief's multi-emotion example ---
    ("m01", "placement pressure hai aur hostel ka stress bhi hai", ["stress", "frustration"], "hi", "latin", True),
]

# Held out from the fixtures above so evaluation is not self-scoring.
SENTIMENT_EVAL = [
    ("v01", "this placement season is horrible", "negative"),
    ("v02", "what a wonderful experience overall", "positive"),
    ("v03", "the placement process hai bas", "neutral"),
    ("v04", "i am so happy and grateful", "positive"),
    ("v05", "the mess food is terrible and bad", "negative"),
    ("v06", "the seminar is on monday", "neutral"),
    ("v07", "this campus is beautiful", "positive"),
    ("v08", "fees are ridiculous and unfair", "negative"),
    ("v09", "the class starts at nine", "neutral"),
    ("v10", "teachers are very helpful", "positive"),
    ("v11", "i am not happy with the hostel", "negative"),
    ("v12", "the workshop was on friday", "neutral"),
]

EMOTION_EVAL = [
    ("w01", "this is so frustrating and stressful", ["frustration", "stress"]),
    ("w02", "i am so happy about the placement", ["joy"]),
    ("w03", "i am angry that nobody helps", ["anger"]),
    ("w04", "i am nervous about the fees", ["anxiety"]),
    ("w05", "i am grateful for the support", ["gratitude"]),
    ("w06", "this workload is stressful", ["stress"]),
    ("w07", "the mess serves rice and dal", []),
    ("w08", "i am grateful and delighted", ["gratitude", "joy"]),
    ("w09", "so frustrating and i am furious", ["frustration", "anger"]),
    ("w10", "i am anxious about the results", ["anxiety"]),
    ("w11", "what a wonderful day", ["joy"]),
    ("w12", "i am so frustrated with the mess", ["frustration"]),
]

# A completely different taxonomy, to prove nothing is hard-coded.
ALTERNATE_SENTIMENT_FIXTURE = [
    ("c01", "the faculty were lovely and kind", "good"),
    ("c02", "the campus is well maintained", "good"),
    ("c03", "the canteen is quite decent", "good"),
    ("c04", "the faculty were rude and unhelpful", "bad"),
    ("c05", "the campus is badly maintained", "bad"),
    ("c06", "the canteen is quite awful", "bad"),
    ("c07", "the seminar happens on tuesday", "flat"),
    ("c08", "the library opens at eight", "flat"),
    ("c09", "the forms are due on friday", "flat"),
]

ALTERNATE_EMOTION_FIXTURE = [
    ("k01", "i feel completely calm about the result", ["calm"]),
    ("k02", "everything is peaceful today", ["calm"]),
    ("k03", "i am so irritated by the noise", ["annoyance"]),
    ("k04", "this noise is irritating me", ["annoyance"]),
    ("k05", "i feel calm and not annoyed", ["calm", "annoyance"]),
    ("k06", "the seminar is on tuesday", []),
    ("k07", "the library opens at eight", []),
    ("k08", "the forms are due on friday", []),
]

POSITIVE_TEXT = "this placement season is amazing"
NEGATIVE_TEXT = "this placement process is horrible"
NEUTRAL_TEXT = "the exam was scheduled on monday"
EMOJI_TEXT = (
    "placement ka scene bahut stressful hai bro <EMOJI_loudly_crying_face>"
)


def to_sentiment(rows) -> list:
    return [
        SentimentAnnotation(
            document_id=document_id,
            text=text,
            sentiment=sentiment,
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
            detected_languages=(["hi", "en"] if is_code_mixed else [language]),
        )
        for document_id, text, sentiment, language, script, is_code_mixed in rows
    ]


def to_emotion(rows) -> list:
    return [
        EmotionAnnotation(
            document_id=document_id,
            text=text,
            emotions=list(emotions),
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
            detected_languages=(["hi", "en"] if is_code_mixed else [language]),
        )
        for document_id, text, emotions, language, script, is_code_mixed in rows
    ]


def sentiment_annotations() -> list:
    return to_sentiment(SENTIMENT_FIXTURE)


def emotion_annotations() -> list:
    return to_emotion(EMOTION_FIXTURE)


def to_sentiment_from_pairs(rows) -> list:
    """Sentiment annotations from compact ``(document_id, text, sentiment)`` rows."""
    return [
        SentimentAnnotation(document_id=document_id, text=text, sentiment=sentiment)
        for document_id, text, sentiment in rows
    ]


def to_emotion_from_pairs(rows) -> list:
    """Emotion annotations from compact ``(document_id, text, emotions)`` rows."""
    return [
        EmotionAnnotation(
            document_id=document_id, text=text, emotions=list(emotions)
        )
        for document_id, text, emotions in rows
    ]


# ===========================================================================
# Sentiment annotations
# ===========================================================================


class TestSentimentAnnotationModel(unittest.TestCase):
    """The annotation hand-off format, and the CSV schema around it."""

    def test_defaults_are_documented_values(self):
        annotation = SentimentAnnotation(document_id="a", text="t", sentiment="good")
        self.assertEqual(annotation.language, "unknown")
        self.assertEqual(annotation.script, "unknown")
        self.assertFalse(annotation.is_code_mixed)
        self.assertEqual(annotation.detected_languages, [])
        self.assertIsNone(annotation.post_id)
        self.assertIsNone(annotation.topic_id)

    def test_sentiment_is_required(self):
        with self.assertRaises(Exception):
            SentimentAnnotation(document_id="a", text="t")

    def test_normalised_sentiment_trims_whitespace(self):
        annotation = SentimentAnnotation(
            document_id="a", text="t", sentiment="  positive  "
        )
        self.assertEqual(annotation.normalised_sentiment(), "positive")

    def test_blank_sentiment_is_rejected(self):
        annotation = SentimentAnnotation(document_id="a", text="t", sentiment="   ")
        with self.assertRaises(ValueError) as ctx:
            annotation.validate_sentiment()
        self.assertIn("no sentiment label", str(ctx.exception))

    def test_separator_in_sentiment_is_rejected(self):
        """Sentiment is single-label, so a pipe means the annotator used the
        multi-label syntax by mistake and should use the emotion file instead."""
        annotation = SentimentAnnotation(
            document_id="a", text="t", sentiment="positive|negative"
        )
        with self.assertRaises(ValueError) as ctx:
            annotation.validate_sentiment()
        self.assertIn("single-label", str(ctx.exception))

    def test_any_label_spelling_is_accepted(self):
        """No built-in sentiment vocabulary is enforced in code."""
        for value in ("positive", "Positive", "POSITIVE", "2", "chhota", "worst"):
            annotation = SentimentAnnotation(
                document_id="a", text="t", sentiment=value
            )
            annotation.validate_sentiment()

    def test_to_document_projects_text_and_metadata(self):
        annotation = SentimentAnnotation(
            document_id="a",
            text="normalized text",
            sentiment="positive",
            post_id="p1",
            comment_id="c1",
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
            topic_id=3,
        )
        document = annotation.to_document()
        self.assertEqual(document.document_id, "a")
        self.assertEqual(document.text, "normalized text")
        self.assertEqual(document.original_text, "normalized text")
        self.assertEqual(document.post_id, "p1")
        self.assertEqual(document.comment_id, "c1")
        self.assertEqual(document.language, "hi")
        self.assertTrue(document.is_code_mixed)
        self.assertEqual(document.detected_languages, ["hi", "en"])
        self.assertEqual(document.topic_id, 3)

    def test_corpus_counts_classes_and_labels(self):
        corpus = SentimentCorpus(annotations=sentiment_annotations())
        self.assertEqual(len(corpus), 35)
        self.assertEqual(corpus.labels, ["negative", "neutral", "positive"])
        self.assertEqual(
            corpus.class_counts(), {"negative": 11, "neutral": 11, "positive": 13}
        )
        self.assertEqual(len(corpus.document_ids), 35)
        self.assertEqual(len(list(corpus)), 35)

    def test_duplicate_document_ids_reported_not_dropped(self):
        corpus = SentimentCorpus(
            annotations=[
                SentimentAnnotation(document_id="d", text="t", sentiment="positive"),
                SentimentAnnotation(document_id="d", text="t2", sentiment="negative"),
                SentimentAnnotation(document_id="d", text="t3", sentiment="neutral"),
            ]
        )
        self.assertEqual(corpus.duplicate_document_ids(), ["d"])
        self.assertEqual(len(corpus), 3)

    def test_corpus_validate_surfaces_bad_rows(self):
        corpus = SentimentCorpus(
            annotations=[
                SentimentAnnotation(document_id="a", text="t", sentiment=" ")
            ]
        )
        with self.assertRaises(ValueError):
            corpus.validate()

    def test_csv_round_trip(self):
        corpus = SentimentCorpus(annotations=sentiment_annotations())
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "sentiment.csv")
            corpus.to_csv(path)
            self.assertTrue(os.path.exists(path))
            restored = SentimentCorpus.from_csv(path)
        self.assertEqual(len(restored), len(corpus))
        self.assertEqual(restored.labels, corpus.labels)
        self.assertEqual(restored.class_counts(), corpus.class_counts())
        original = corpus.annotations[6]
        copy = restored.annotations[6]
        self.assertEqual(copy.text, original.text)
        self.assertEqual(copy.sentiment, original.sentiment)
        self.assertEqual(copy.is_code_mixed, original.is_code_mixed)
        self.assertEqual(copy.detected_languages, original.detected_languages)

    def test_csv_column_order_is_documented(self):
        self.assertEqual(
            SENTIMENT_CSV_COLUMNS,
            [
                "document_id",
                "text",
                "sentiment",
                "post_id",
                "comment_id",
                "language",
                "script",
                "is_code_mixed",
                "detected_languages",
            ],
        )
        self.assertEqual(
            DEFAULT_SENTIMENT_CSV,
            os.path.join("data", "annotations", "sentiment_annotations.csv"),
        )

    def test_csv_missing_required_column_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("document_id,text\na,hello\n")
            with self.assertRaises(ValueError) as ctx:
                SentimentCorpus.from_csv(path)
        self.assertIn("sentiment", str(ctx.exception))

    def test_csv_missing_sentiment_value_reports_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("document_id,text,sentiment\n")
                handle.write("a,hello,positive\n")
                handle.write("b,world,\n")
            with self.assertRaises(ValueError) as ctx:
                SentimentCorpus.from_csv(path)
        self.assertIn(":3:", str(ctx.exception))
        self.assertIn("sentiment", str(ctx.exception))

    def test_csv_missing_file_names_expected_columns(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            SentimentCorpus.from_csv("no_such_sentiment_file.csv")
        self.assertIn("document_id", str(ctx.exception))

    def test_csv_skips_blank_lines_and_parses_bools(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "blanks.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("document_id,text,sentiment,is_code_mixed,detected_languages\n")
                handle.write("a,hello,positive,TRUE,hi|en\n")
                handle.write("\n")
                handle.write("   \n")
                handle.write("b,world,negative,no,\n")
            corpus = SentimentCorpus.from_csv(path)
        self.assertEqual(len(corpus), 2)
        self.assertTrue(corpus.annotations[0].is_code_mixed)
        self.assertEqual(corpus.annotations[0].detected_languages, ["hi", "en"])
        self.assertFalse(corpus.annotations[1].is_code_mixed)
        self.assertEqual(corpus.annotations[1].detected_languages, [])

    def test_csv_accepts_bom(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bom.csv")
            with open(path, "w", encoding="utf-8-sig") as handle:
                handle.write("document_id,text,sentiment\n")
                handle.write("a,hello,positive\n")
            corpus = SentimentCorpus.from_csv(path)
        self.assertEqual(len(corpus), 1)

    def test_from_records_helper(self):
        corpus = SentimentCorpus.from_records(
            [
                {"document_id": "a", "text": "t", "sentiment": "positive"},
                {"document_id": "b", "text": "u", "sentiment": "negative"},
            ]
        )
        self.assertEqual(corpus.labels, ["negative", "positive"])


# ===========================================================================
# Emotion annotations
# ===========================================================================


class TestEmotionAnnotationModel(unittest.TestCase):
    def test_empty_emotions_are_valid(self):
        annotation = EmotionAnnotation(document_id="a", text="t")
        self.assertEqual(annotation.emotions, [])
        self.assertFalse(annotation.is_multi_label)
        annotation.validate_emotions()

    def test_multi_label_is_valid(self):
        annotation = EmotionAnnotation(
            document_id="a",
            text="placement pressure hai aur hostel ka stress bhi hai",
            emotions=["stress", "frustration"],
        )
        self.assertTrue(annotation.is_multi_label)
        annotation.validate_emotions()

    def test_normalised_emotions_dedupe_and_sort(self):
        annotation = EmotionAnnotation(
            document_id="a", text="t", emotions=["stress", " anger ", "stress"]
        )
        self.assertEqual(annotation.normalised_emotions(), ["anger", "stress"])

    def test_blank_emotion_label_rejected(self):
        annotation = EmotionAnnotation(
            document_id="a", text="t", emotions=["stress", "  "]
        )
        with self.assertRaises(ValueError) as ctx:
            annotation.validate_emotions()
        self.assertIn("empty emotion label", str(ctx.exception))

    def test_separator_inside_label_rejected(self):
        annotation = EmotionAnnotation(
            document_id="a", text="t", emotions=["stress|anger"]
        )
        with self.assertRaises(ValueError) as ctx:
            annotation.validate_emotions()
        self.assertIn("split on that separator", str(ctx.exception))

    def test_any_label_spelling_is_accepted(self):
        annotation = EmotionAnnotation(
            document_id="a", text="t", emotions=["Joy", "ANGER", "frustrated"]
        )
        annotation.validate_emotions()

    def test_corpus_counts_including_empty_and_multi(self):
        corpus = EmotionCorpus(annotations=emotion_annotations())
        self.assertEqual(corpus.n_documents_without_emotions, 4)
        self.assertEqual(corpus.n_multi_label_documents, 11)
        counts = corpus.label_counts()
        for label in corpus.labels:
            self.assertGreaterEqual(counts[label], 2)
        self.assertEqual(sorted(corpus.labels), sorted(counts.keys()))

    def test_corpus_duplicate_ids_reported(self):
        corpus = EmotionCorpus(
            annotations=[
                EmotionAnnotation(document_id="d", text="t", emotions=["stress"]),
                EmotionAnnotation(document_id="d", text="u", emotions=["anger"]),
            ]
        )
        self.assertEqual(corpus.duplicate_document_ids(), ["d"])

    def test_csv_round_trip_with_multi_and_empty(self):
        corpus = EmotionCorpus(annotations=emotion_annotations())
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "emotion.csv")
            corpus.to_csv(path)
            restored = EmotionCorpus.from_csv(path)
        self.assertEqual(len(restored), len(corpus))
        self.assertEqual(restored.labels, corpus.labels)
        self.assertEqual(restored.label_counts(), corpus.label_counts())
        self.assertEqual(restored.n_documents_without_emotions, 4)
        self.assertEqual(restored.n_multi_label_documents, 11)
        multi = restored.annotations[0]
        self.assertEqual(multi.emotions, ["frustration", "stress"])
        empty = restored.annotations[-5]
        self.assertEqual(empty.emotions, [])

    def test_csv_column_order_and_default_path(self):
        self.assertEqual(
            EMOTION_CSV_COLUMNS,
            [
                "document_id",
                "text",
                "emotions",
                "post_id",
                "comment_id",
                "language",
                "script",
                "is_code_mixed",
                "detected_languages",
            ],
        )
        self.assertEqual(
            DEFAULT_EMOTION_CSV,
            os.path.join("data", "annotations", "emotion_annotations.csv"),
        )

    def test_csv_missing_emotions_column_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bad.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("document_id,text\na,hello\n")
            with self.assertRaises(ValueError) as ctx:
                EmotionCorpus.from_csv(path)
        self.assertIn("emotions", str(ctx.exception))

    def test_csv_missing_file_names_separator(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            EmotionCorpus.from_csv("no_such_emotion_file.csv")
        self.assertIn("|", str(ctx.exception))

    def test_to_document_projects_metadata(self):
        annotation = EmotionAnnotation(
            document_id="a",
            text="normalized",
            emotions=["stress"],
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
            topic_id=2,
        )
        document = annotation.to_document()
        self.assertEqual(document.text, "normalized")
        self.assertEqual(document.language, "hi")
        self.assertTrue(document.is_code_mixed)
        self.assertEqual(document.topic_id, 2)


# ===========================================================================
# Sentiment classifier
# ===========================================================================


class TestSentimentClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = TfidfLogisticSentimentClassifier()
        self.report = self.classifier.train(sentiment_annotations())
        self.documents = [annotation.to_document() for annotation in sentiment_annotations()]

    # -- interface ----------------------------------------------------

    def test_implements_abstract_interface(self):
        self.assertIsInstance(self.classifier, SentimentAnalysis)
        self.assertTrue(inspect.isabstract(SentimentAnalysis))

    def test_model_name_and_version_are_explicit(self):
        self.assertEqual(self.classifier.name, "tfidf_logistic")
        self.assertEqual(self.classifier.version, "v1")
        self.assertEqual(self.report.model_name, "tfidf_logistic")
        self.assertEqual(self.report.model_version, "v1")

    def test_not_trained_before_train(self):
        fresh = TfidfLogisticSentimentClassifier()
        self.assertFalse(fresh.is_trained)
        self.assertEqual(fresh.labels, [])

    def test_predict_before_training_raises(self):
        fresh = TfidfLogisticSentimentClassifier()
        with self.assertRaises(RuntimeError) as ctx:
            fresh.predict([SentimentEmotionDocument(document_id="d", text="hello")])
        self.assertIn("not been trained", str(ctx.exception))

    def test_evaluate_before_training_raises(self):
        fresh = TfidfLogisticSentimentClassifier()
        with self.assertRaises(RuntimeError):
            fresh.evaluate(sentiment_annotations())

    def test_abstract_base_forbids_instantiation(self):
        with self.assertRaises(TypeError):
            SentimentAnalysis()

    # -- training report ----------------------------------------------

    def test_classes_learned_from_annotation_data(self):
        self.assertEqual(self.report.labels, ["negative", "neutral", "positive"])
        self.assertEqual(self.report.n_classes, 3)
        self.assertEqual(self.classifier.labels, ["negative", "neutral", "positive"])

    def test_training_report_evidence(self):
        self.assertEqual(
            self.report.examples_per_class,
            {"negative": 11, "neutral": 11, "positive": 13},
        )
        self.assertEqual(self.report.n_training_documents, 35)
        self.assertGreater(self.report.n_features, 50)
        self.assertEqual(self.report.classes_per_document, 1)
        self.assertAlmostEqual(self.report.majority_class_share, 13 / 35, places=5)
        self.assertTrue(self.report.is_usable)

    def test_training_report_metadata(self):
        self.assertEqual(self.report.threshold, 0.5)
        self.assertIn("scikit-learn", self.report.library_versions)
        self.assertIn("numpy", self.report.library_versions)
        self.assertIsNone(self.report.trained_at)
        self.assertIn("tfidf", self.report.config)
        self.assertIn("logistic_regression", self.report.config)
        self.assertIn("data_guards", self.report.config)
        self.assertIn("features", self.report.config)
        self.assertEqual(
            self.report.config["logistic_regression"]["random_state"], 42
        )
        self.assertEqual(self.report.config["logistic_regression"]["solver"], "lbfgs")
        self.assertEqual(self.report.config["tfidf"]["stop_words"], None)
        self.assertTrue(
            self.report.config["features"]["drop_placeholder_tokens"]
        )

    def test_timestamp_is_opt_in(self):
        stamped = TfidfLogisticSentimentClassifier(include_timestamp=True)
        report = stamped.train(sentiment_annotations())
        self.assertIsNotNone(report.trained_at)
        result = stamped.predict(self.documents[:1])
        self.assertIsNotNone(result.trained_at)

    def test_training_report_property(self):
        self.assertIs(self.classifier.training_report, self.report)

    # -- prediction ---------------------------------------------------

    def test_predicts_positive(self):
        result = self.classifier.predict(
            [SentimentEmotionDocument(document_id="p", text=POSITIVE_TEXT)]
        )
        self.assertEqual(result.predictions[0].sentiment, "positive")

    def test_predicts_negative(self):
        result = self.classifier.predict(
            [SentimentEmotionDocument(document_id="n", text=NEGATIVE_TEXT)]
        )
        self.assertEqual(result.predictions[0].sentiment, "negative")

    def test_predicts_neutral(self):
        result = self.classifier.predict(
            [SentimentEmotionDocument(document_id="t", text=NEUTRAL_TEXT)]
        )
        self.assertEqual(result.predictions[0].sentiment, "neutral")

    def test_probabilities_are_a_normalised_distribution(self):
        """Sentiment is mutually exclusive, so the sigmoids are normalised."""
        result = self.classifier.predict(
            [SentimentEmotionDocument(document_id="p", text=POSITIVE_TEXT)]
        )
        prediction = result.predictions[0]
        self.assertEqual(
            sorted(prediction.sentiment_probabilities),
            ["negative", "neutral", "positive"],
        )
        self.assertAlmostEqual(
            sum(prediction.sentiment_probabilities.values()), 1.0, places=5
        )
        for probability in prediction.sentiment_probabilities.values():
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_confidence_is_the_winning_probability(self):
        result = self.classifier.predict(
            [SentimentEmotionDocument(document_id="p", text=POSITIVE_TEXT)]
        )
        prediction = result.predictions[0]
        self.assertEqual(
            prediction.confidence,
            max(prediction.sentiment_probabilities.values()),
        )
        self.assertEqual(prediction.top_sentiment(), prediction.sentiment)

    def test_result_metadata(self):
        result = self.classifier.predict(self.documents[:3])
        self.assertEqual(result.model_name, "tfidf_logistic")
        self.assertEqual(result.model_version, "v1")
        self.assertEqual(result.threshold, 0.5)
        self.assertEqual(result.n_documents, 3)
        self.assertEqual(result.known_sentiments, ["negative", "neutral", "positive"])
        self.assertEqual(result.n_abstained, 0)
        self.assertGreater(result.average_confidence(), 0.0)
        self.assertIn("tfidf", result.config)
        self.assertIn("scikit-learn", result.library_versions)

    def test_prediction_lookup_and_frequency(self):
        result = self.classifier.predict(self.documents)
        self.assertIsNotNone(result.prediction_for("s01"))
        self.assertIsNone(result.prediction_for("nope"))
        frequency = result.sentiment_frequency()
        self.assertEqual(sum(frequency.values()), result.n_documents)
        self.assertEqual(list(frequency), sorted(frequency))

    def test_empty_document_list(self):
        result = self.classifier.predict([])
        self.assertEqual(result.predictions, [])
        self.assertEqual(result.n_documents, 0)
        self.assertEqual(result.average_confidence(), 0.0)

    def test_blank_text_does_not_crash(self):
        result = self.classifier.predict(
            [
                SentimentEmotionDocument(document_id="blank", text=""),
                SentimentEmotionDocument(document_id="spaces", text="   "),
            ]
        )
        self.assertEqual(len(result.predictions), 2)
        for prediction in result.predictions:
            self.assertIsNotNone(prediction.sentiment_probabilities)

    # -- threshold / configuration -------------------------------------

    def test_threshold_override_abstains_more(self):
        document = SentimentEmotionDocument(document_id="n", text=NEGATIVE_TEXT)
        permissive = self.classifier.predict([document], threshold=0.3)
        strict = self.classifier.predict([document], threshold=0.95)
        self.assertIsNotNone(permissive.predictions[0].sentiment)
        self.assertIsNone(strict.predictions[0].sentiment)
        self.assertTrue(strict.predictions[0].abstained)
        self.assertEqual(strict.n_abstained, 1)
        self.assertEqual(permissive.n_abstained, 0)

    def test_abstention_reports_the_winner_anyway(self):
        document = SentimentEmotionDocument(document_id="n", text=NEGATIVE_TEXT)
        result = self.classifier.predict([document], threshold=0.99)
        prediction = result.predictions[0]
        self.assertIsNone(prediction.sentiment)
        self.assertFalse(prediction.has_sentiment)
        self.assertEqual(prediction.top_sentiment(), "negative")

    def test_sentiment_at_rethresholds_without_inference(self):
        result = self.classifier.predict(
            [SentimentEmotionDocument(document_id="n", text=NEGATIVE_TEXT)]
        )
        prediction = result.predictions[0]
        self.assertEqual(prediction.sentiment_at(0.1), "negative")
        self.assertIsNone(prediction.sentiment_at(0.99))

    def test_constructor_threshold_overrides_config(self):
        classifier = TfidfLogisticSentimentClassifier(threshold=0.7)
        self.assertEqual(classifier.threshold, 0.7)
        self.assertEqual(classifier.config.threshold, 0.5)
        classifier.train(sentiment_annotations())
        self.assertEqual(classifier.training_report.threshold, 0.7)
        self.assertEqual(classifier.config_dict()["threshold"], 0.7)

    def test_config_properties_are_exposed(self):
        self.assertIsInstance(self.classifier.config, SentimentModelConfig)
        self.assertEqual(self.classifier.config.C, 4.0)
        self.assertEqual(self.classifier.config.max_iter, 1000)
        self.assertEqual(self.classifier.config.solver, "lbfgs")
        self.assertEqual(self.classifier.config.random_state, 42)
        self.assertEqual(self.classifier.config.ngram_range, (1, 2))
        self.assertEqual(self.classifier.config.token_pattern, r"(?u)\b\w+\b")
        self.assertTrue(self.classifier.config.sublinear_tf)

    def test_tuning_C_changes_output(self):
        loose = TfidfLogisticSentimentClassifier(
            config=SentimentModelConfig(C=50.0)
        )
        loose.train(sentiment_annotations())
        documents = self.documents[:6]
        base_probs = [
            p.sentiment_probabilities
            for p in self.classifier.predict(documents).predictions
        ]
        loose_probs = [
            p.sentiment_probabilities for p in loose.predict(documents).predictions
        ]
        self.assertNotEqual(base_probs, loose_probs)

    def test_stop_words_english_is_allowed(self):
        classifier = TfidfLogisticSentimentClassifier(
            config=SentimentModelConfig(stop_words="english")
        )
        report = classifier.train(sentiment_annotations())
        self.assertEqual(report.config["tfidf"]["stop_words"], "english")

    def test_explicit_stop_word_list_is_allowed(self):
        classifier = TfidfLogisticSentimentClassifier(
            config=SentimentModelConfig(stop_words=["the", "a"])
        )
        classifier.train(sentiment_annotations())
        self.assertNotIn("the", classifier._vectorizer.vocabulary_)

    # -- determinism --------------------------------------------------

    def test_deterministic_across_independent_trains(self):
        other = TfidfLogisticSentimentClassifier()
        other.train(sentiment_annotations())
        first = self.classifier.predict(self.documents)
        second = other.predict(self.documents)
        self.assertEqual(
            [p.sentiment_probabilities for p in first.predictions],
            [p.sentiment_probabilities for p in second.predictions],
        )
        self.assertEqual(
            [p.sentiment for p in first.predictions],
            [p.sentiment for p in second.predictions],
        )

    def test_repeated_predict_is_stable(self):
        first = self.classifier.predict(self.documents)
        second = self.classifier.predict(self.documents)
        self.assertEqual(
            first.model_dump_json(), second.model_dump_json()
        )

    def test_retraining_replaces_class_set(self):
        self.assertIn("positive", self.classifier.labels)
        self.classifier.train(to_sentiment_from_pairs(ALTERNATE_SENTIMENT_FIXTURE))
        self.assertEqual(self.classifier.labels, ["bad", "flat", "good"])
        self.assertNotIn("positive", self.classifier.labels)
        result = self.classifier.predict(
            [SentimentEmotionDocument(document_id="d", text=POSITIVE_TEXT)]
        )
        self.assertEqual(
            sorted(result.predictions[0].sentiment_probabilities),
            ["bad", "flat", "good"],
        )

    def test_class_order_is_sorted_not_first_seen(self):
        """A different input order must not change the reported class order."""
        shuffled = list(reversed(sentiment_annotations()))
        other = TfidfLogisticSentimentClassifier()
        report = other.train(shuffled)
        self.assertEqual(report.labels, self.report.labels)

    # -- input immutability -------------------------------------------

    def test_input_documents_are_never_modified(self):
        document = SentimentEmotionDocument(
            document_id="p",
            text=EMOJI_TEXT,
            original_text="placement ka scene bahut stressful hai bro 😢",
            post_id="post-1",
            comment_id="comment-1",
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
            topic_id=4,
            topic_label="Placement",
        )
        before = document.model_dump()
        self.classifier.predict([document])
        self.assertEqual(document.model_dump(), before)
        self.assertIn("<EMOJI_loudly_crying_face>", document.text)
        self.assertIn("😢", document.original_text)

    def test_result_echoes_metadata_unchanged(self):
        document = SentimentEmotionDocument(
            document_id="p",
            text=EMOJI_TEXT,
            original_text="RAW <URL> placement",
            post_id="post-9",
            comment_id="comment-9",
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
            topic_id=1,
            topic_label="Topic 1",
            issue_labels=["placement_stress"],
        )
        prediction = self.classifier.predict([document]).predictions[0]
        self.assertEqual(prediction.text, EMOJI_TEXT)
        self.assertEqual(prediction.original_text, "RAW <URL> placement")
        self.assertEqual(prediction.post_id, "post-9")
        self.assertEqual(prediction.comment_id, "comment-9")
        self.assertEqual(prediction.language, "hi")
        self.assertEqual(prediction.script, "latin")
        self.assertTrue(prediction.is_code_mixed)
        self.assertEqual(prediction.detected_languages, ["hi", "en"])
        self.assertEqual(prediction.topic_id, 1)
        self.assertEqual(prediction.topic_label, "Topic 1")
        self.assertEqual(prediction.issue_labels, ["placement_stress"])

    def test_normalized_text_is_used_as_the_feature_source(self):
        """The model keys on `text`, and original_text is never featurised."""
        same_text = [
            SentimentEmotionDocument(
                document_id="a", text=POSITIVE_TEXT, original_text="totally different"
            ),
            SentimentEmotionDocument(
                document_id="b", text=POSITIVE_TEXT, original_text="another other one"
            ),
        ]
        result = self.classifier.predict(same_text)
        self.assertEqual(
            result.predictions[0].sentiment_probabilities,
            result.predictions[1].sentiment_probabilities,
        )

    # -- placeholders / negation ---------------------------------------

    def test_emoji_placeholders_dropped_by_default(self):
        vocabulary = self.classifier._vectorizer.vocabulary_
        for term in vocabulary:
            self.assertNotIn("emoji", term)
            self.assertNotIn("<", term)
            self.assertNotIn(">", term)
        # The rest of the training text is untouched, so real terms survive.
        self.assertIn("amazing", vocabulary)
        self.assertIn("horrible", vocabulary)

    def test_placeholder_keeping_can_be_enabled(self):
        classifier = TfidfLogisticSentimentClassifier(
            config=SentimentModelConfig(drop_placeholder_tokens=False)
        )
        classifier.train(sentiment_annotations())
        self.assertIn("emoji_tada", classifier._vectorizer.vocabulary_)
        self.assertIn("emoji_pensive_face", classifier._vectorizer.vocabulary_)

    def test_placeholder_dropping_documented_in_config(self):
        self.assertTrue(self.classifier.config.drop_placeholder_tokens)
        self.assertTrue(
            self.report.config["features"]["drop_placeholder_tokens"]
        )

    def test_emoji_text_still_scores_every_class(self):
        """
        Code-mixed text with an emoji placeholder is still featurised and scored.

        The placeholder is stripped from the model copy, but the rest of the text
        still produces a full probability vector.  The winning class may fall
        below the default confidence floor on a 35-document corpus, which is the
        documented abstain behaviour rather than a failure.
        """
        result = self.classifier.predict(
            [SentimentEmotionDocument(document_id="e", text=EMOJI_TEXT)]
        )
        prediction = result.predictions[0]
        self.assertEqual(len(prediction.sentiment_probabilities), 3)
        self.assertAlmostEqual(sum(prediction.sentiment_probabilities.values()), 1.0, 5)
        self.assertIsNotNone(prediction.top_sentiment())
        # Lowering the confidence floor re-derives a class with no re-inference.
        self.assertIsNotNone(prediction.sentiment_at(0.1))
        self.assertIsNone(prediction.sentiment_at(1.0))

    def test_negation_words_survive(self):
        """No stop-word list is applied by default, so negation is not stripped."""
        vocabulary = self.classifier._vectorizer.vocabulary_
        self.assertIn("not", vocabulary)
        self.assertIn("much", vocabulary)
        for term in ("hai", "aur", "bahut", "ka", "maine"):
            self.assertIn(term, vocabulary)

    def test_topic_features_off_by_default(self):
        vocabulary = self.classifier._vectorizer.vocabulary_
        self.assertFalse(any(term.startswith("topicid") for term in vocabulary))
        self.assertFalse(any(term.startswith("topickey") for term in vocabulary))

    def test_topic_features_can_be_enabled(self):
        """
        Enabling topic features injects weak topic tokens without changing the
        text model, and the topic context is still echoed back untouched.
        """
        classifier = TfidfLogisticSentimentClassifier(
            config=SentimentModelConfig(use_topic_features=True)
        )
        report = classifier.train(sentiment_annotations())
        self.assertTrue(report.config["features"]["use_topic_features"])

        result = classifier.predict(
            [
                SentimentEmotionDocument(
                    document_id="t",
                    text=POSITIVE_TEXT,
                    topic_id=7,
                    topic_label="campus life",
                    topic_keywords=["placement", "fest"],
                )
            ]
        )
        prediction = result.predictions[0]
        # Topic context is carried through for the downstream join, not used as
        # the reason for the class.
        self.assertEqual(prediction.topic_id, 7)
        self.assertEqual(prediction.topic_label, "campus life")
        self.assertEqual(prediction.topic_keywords, ["placement", "fest"])
        self.assertEqual(len(prediction.sentiment_probabilities), 3)

    # -- training refusals ---------------------------------------------

    def test_empty_training_data_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.classifier.train([])
        self.assertIn("no training data", str(ctx.exception))
        self.assertIn("sentiment_annotations.csv", str(ctx.exception))

    def test_wrong_record_type_refused(self):
        with self.assertRaises(TypeError) as ctx:
            self.classifier.train(emotion_annotations())
        self.assertIn("SentimentAnnotation", str(ctx.exception))

    def test_blank_sentiment_refused(self):
        annotations = sentiment_annotations()
        annotations[0] = SentimentAnnotation(
            document_id=annotations[0].document_id, text="some text", sentiment=" "
        )
        with self.assertRaises(ValueError):
            self.classifier.train(annotations)

    def test_too_few_documents_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.classifier.train(sentiment_annotations()[:3])
        self.assertIn("insufficient training data", str(ctx.exception))

    def test_single_class_refused(self):
        annotations = [
            SentimentAnnotation(
                document_id=f"x{index}", text=f"amazing great wonderful {index}",
                sentiment="positive",
            )
            for index in range(6)
        ]
        with self.assertRaises(ValueError) as ctx:
            self.classifier.train(annotations)
        self.assertIn("insufficient label variation", str(ctx.exception))

    def test_class_with_too_few_examples_refused(self):
        annotations = sentiment_annotations() + [
            SentimentAnnotation(
                document_id="rare", text="mixed bag not great", sentiment="mixed"
            )
        ]
        with self.assertRaises(ValueError) as ctx:
            self.classifier.train(annotations)
        self.assertIn("insufficient examples per class", str(ctx.exception))
        self.assertIn("mixed", str(ctx.exception))

    def test_unusable_vocabulary_refused(self):
        annotations = [
            SentimentAnnotation(
                document_id=f"b{index}", text="", sentiment="positive"
            )
            for index in range(4)
        ] + [
            SentimentAnnotation(
                document_id=f"c{index}", text="<URL> <EMOJI_fire>", sentiment="negative"
            )
            for index in range(4)
        ]
        classifier = TfidfLogisticSentimentClassifier()
        with self.assertRaises(ValueError) as ctx:
            classifier.train(annotations)
        self.assertIn("no usable TF-IDF features", str(ctx.exception))

    def test_failed_training_leaves_model_untrained(self):
        broken = TfidfLogisticSentimentClassifier()
        broken.train(sentiment_annotations())
        with self.assertRaises(ValueError):
            broken.train([])
        # The previous fit is still usable and still reports its own classes.
        self.assertTrue(broken.is_trained)
        self.assertEqual(broken.labels, ["negative", "neutral", "positive"])

    # -- configuration validation --------------------------------------

    def test_invalid_threshold_rejected(self):
        # `None` is excluded: it means "use the configured default", not invalid.
        for bad in (0.0, 1.0, -0.1, 1.5, "half", True):
            with self.assertRaises(ValueError):
                TfidfLogisticSentimentClassifier(threshold=bad)
            with self.assertRaises(ValueError):
                self.classifier.predict(self.documents[:1], threshold=bad)
        # Explicitly: None falls back to the configured floor rather than failing.
        self.assertEqual(
            self.classifier.predict(self.documents[:1], threshold=None).threshold,
            self.classifier.config.threshold,
        )

    def test_invalid_tfidf_config_rejected(self):
        cases = [
            {"max_features": 0},
            {"min_df": -1},
            {"max_df": 2.0},
            {"ngram_range": (0, 2)},
            {"ngram_range": (3, 2)},
            {"norm": "l3"},
            {"token_pattern": "([unclosed"},
        ]
        for case in cases:
            with self.assertRaises(ValueError, msg=str(case)):
                TfidfLogisticSentimentClassifier(config=SentimentModelConfig(**case))

    def test_stop_words_restricted_to_none_english_or_list(self):
        for bad in ("hindi", "bengali", "odia", "HINDI"):
            with self.assertRaises(ValueError) as ctx:
                TfidfLogisticSentimentClassifier(
                    config=SentimentModelConfig(stop_words=bad)
                )
            self.assertIn("english", str(ctx.exception))
        with self.assertRaises(ValueError):
            TfidfLogisticSentimentClassifier(
                config=SentimentModelConfig(stop_words=42)
            )
        with self.assertRaises(ValueError):
            TfidfLogisticSentimentClassifier(
                config=SentimentModelConfig(stop_words=[1, 2])
            )

    def test_invalid_regression_config_rejected(self):
        cases = [
            {"C": 0.0},
            {"C": -1.0},
            {"max_iter": 0},
            {"class_weight": "custom"},
            {"min_training_documents": 1},
            {"min_examples_per_class": 0},
            {"imbalance_warning_share": 0.0},
            {"imbalance_warning_share": 1.5},
        ]
        for case in cases:
            with self.assertRaises(ValueError, msg=str(case)):
                TfidfLogisticSentimentClassifier(config=SentimentModelConfig(**case))

    def test_valid_config_values_accepted(self):
        classifier = TfidfLogisticSentimentClassifier(
            config=SentimentModelConfig(
                max_features=500,
                min_df=1,
                max_df=0.95,
                ngram_range=(1, 3),
                norm="l1",
                use_idf=False,
                smooth_idf=False,
                lowercase=False,
                C=1.0,
                max_iter=100,
                class_weight="balanced",
                threshold=0.33,
                min_training_documents=10,
                min_examples_per_class=3,
                imbalance_warning_share=0.95,
            )
        )
        classifier.train(sentiment_annotations())
        self.assertEqual(classifier.config_dict()["tfidf"]["norm"], "l1")
        self.assertEqual(
            classifier.config_dict()["logistic_regression"]["class_weight"],
            "balanced",
        )

    # -- serialization -------------------------------------------------

    def test_result_json_round_trip(self):
        result = self.classifier.predict(self.documents)
        restored = SentimentAnalysisResult.model_validate_json(
            result.model_dump_json()
        )
        self.assertEqual(
            [p.sentiment_probabilities for p in restored.predictions],
            [p.sentiment_probabilities for p in result.predictions],
        )
        self.assertEqual(
            [p.sentiment for p in restored.predictions],
            [p.sentiment for p in result.predictions],
        )
        self.assertEqual(restored.model_name, "tfidf_logistic")
        self.assertEqual(restored.model_version, "v1")

    def test_training_report_json_round_trip(self):
        restored = SentimentTrainingReport.model_validate_json(
            self.report.model_dump_json()
        )
        self.assertEqual(restored.labels, self.report.labels)
        self.assertEqual(
            restored.examples_per_class, self.report.examples_per_class
        )

    def test_prediction_model_validates_standalone(self):
        prediction = SentimentPrediction(
            document_id="d", sentiment="positive",
            sentiment_probabilities={"positive": 1.0}, confidence=1.0,
        )
        self.assertTrue(prediction.has_sentiment)
        self.assertEqual(prediction.top_sentiment(), "positive")


# ===========================================================================
# Emotion classifier
# ===========================================================================


class TestEmotionClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = TfidfLogisticEmotionClassifier()
        self.report = self.classifier.train(emotion_annotations())
        self.documents = [annotation.to_document() for annotation in emotion_annotations()]

    def test_implements_abstract_interface(self):
        self.assertIsInstance(self.classifier, EmotionAnalysis)
        self.assertTrue(inspect.isabstract(EmotionAnalysis))

    def test_model_name_and_version_are_explicit(self):
        self.assertEqual(self.classifier.name, "tfidf_logistic_multilabel")
        self.assertEqual(self.classifier.version, "v1")
        self.assertEqual(self.report.model_name, "tfidf_logistic_multilabel")
        self.assertEqual(self.report.model_version, "v1")

    def test_not_trained_before_train(self):
        fresh = TfidfLogisticEmotionClassifier()
        self.assertFalse(fresh.is_trained)
        self.assertEqual(fresh.labels, [])

    def test_predict_before_training_raises(self):
        fresh = TfidfLogisticEmotionClassifier()
        with self.assertRaises(RuntimeError) as ctx:
            fresh.predict([SentimentEmotionDocument(document_id="d", text="hello")])
        self.assertIn("not been trained", str(ctx.exception))

    def test_abstract_base_forbids_instantiation(self):
        with self.assertRaises(TypeError):
            EmotionAnalysis()

    def test_labels_learned_from_annotation_data(self):
        self.assertEqual(
            self.report.labels,
            ["anger", "anxiety", "frustration", "gratitude", "joy", "stress"],
        )
        self.assertEqual(self.classifier.labels, self.report.labels)

    def test_training_report_evidence(self):
        self.assertEqual(self.report.n_training_documents, 27)
        self.assertEqual(self.report.n_labels, 6)
        self.assertEqual(self.report.n_documents_without_emotions, 4)
        self.assertEqual(self.report.n_multi_label_documents, 11)
        for label, count in self.report.positive_examples_per_label.items():
            self.assertGreaterEqual(count, 2, label)
        self.assertTrue(self.report.is_usable)
        self.assertIn("scikit-learn", self.report.library_versions)
        self.assertIsNone(self.report.trained_at)

    def test_config_snapshot(self):
        config = self.report.config
        self.assertEqual(config["logistic_regression"]["random_state"], 42)
        self.assertEqual(config["logistic_regression"]["solver"], "lbfgs")
        self.assertIsNone(config["tfidf"]["stop_words"])
        self.assertTrue(config["features"]["drop_placeholder_tokens"])
        self.assertIn("min_positive_examples_per_label", config["data_guards"])

    # -- prediction ---------------------------------------------------

    def test_single_emotion_prediction(self):
        """
        One clear emotion dominates, and one threshold picks it.

        The winning sigmoid is asserted directly; whether it also clears the
        default 0.5 boundary is checked through ``emotion_at`` rather than
        pinned, because on 27 training documents the winner can sit just under
        the boundary. The point being tested is the independent-per-label
        decision rule, not this corpus's exact calibration.
        """
        result = self.classifier.predict(
            [
                SentimentEmotionDocument(
                    document_id="s", text="i am so happy about the placement"
                )
            ]
        )
        prediction = result.predictions[0]
        self.assertEqual(prediction.top_emotion(), "joy")
        self.assertEqual(len(prediction.emotion_probabilities), 6)
        self.assertEqual(prediction.emotions_above(0.4), ["joy"])
        self.assertEqual(prediction.emotions_above(0.99), [])
        # Just joy clears a 0.4 floor: no second emotion is invented.
        self.assertFalse(prediction.is_multi_label)

    def test_multi_emotion_prediction(self):
        """The brief's example must be able to carry two emotions at once."""
        result = self.classifier.predict(
            [
                SentimentEmotionDocument(
                    document_id="m", text=EMOJI_TEXT, is_code_mixed=True
                )
            ],
            threshold=0.2,
        )
        prediction = result.predictions[0]
        self.assertGreaterEqual(len(prediction.emotions), 1)
        self.assertEqual(prediction.emotions, sorted(prediction.emotions))
        self.assertEqual(prediction.is_multi_label, len(prediction.emotions) > 1)

    def test_probabilities_are_independent_sigmoids_not_softmax(self):
        result = self.classifier.predict(
            [SentimentEmotionDocument(document_id="m", text=EMOJI_TEXT)]
        )
        prediction = result.predictions[0]
        total = sum(prediction.emotion_probabilities.values())
        self.assertEqual(len(prediction.emotion_probabilities), 6)
        # A softmax head would force this to be exactly 1.0.
        self.assertNotAlmostEqual(total, 1.0, places=3)
        for probability in prediction.emotion_probabilities.values():
            self.assertGreater(probability, 0.0)
            self.assertLess(probability, 1.0)

    def test_confidence_is_the_top_probability(self):
        result = self.classifier.predict(
            [SentimentEmotionDocument(document_id="s", text="i am so happy about the placement")]
        )
        prediction = result.predictions[0]
        self.assertEqual(
            prediction.confidence, max(prediction.emotion_probabilities.values())
        )
        self.assertIn(prediction.top_emotion(), prediction.emotion_probabilities)

    def test_result_metadata_and_counters(self):
        result = self.classifier.predict(self.documents)
        self.assertEqual(result.model_name, "tfidf_logistic_multilabel")
        self.assertEqual(result.model_version, "v1")
        self.assertEqual(result.n_documents, len(self.documents))
        self.assertEqual(
            result.known_emotions,
            ["anger", "anxiety", "frustration", "gratitude", "joy", "stress"],
        )
        self.assertEqual(
            result.n_documents_without_emotion + result.n_multi_label_documents
            + sum(
                1
                for p in result.predictions
                if len(p.emotions) == 1
            ),
            result.n_documents,
        )
        self.assertGreater(result.average_confidence(), 0.0)

    def test_emotion_frequency_and_lookup(self):
        result = self.classifier.predict(self.documents)
        frequency = result.emotion_frequency()
        self.assertEqual(list(frequency), sorted(frequency))
        self.assertIsNotNone(result.prediction_for("f01"))
        self.assertIsNone(result.prediction_for("nope"))

    def test_empty_document_list(self):
        result = self.classifier.predict([])
        self.assertEqual(result.predictions, [])
        self.assertEqual(result.n_documents, 0)
        self.assertEqual(result.average_confidence(), 0.0)

    def test_blank_text_does_not_crash(self):
        result = self.classifier.predict(
            [
                SentimentEmotionDocument(document_id="blank", text=""),
                SentimentEmotionDocument(document_id="none", text="<URL>"),
            ]
        )
        self.assertEqual(len(result.predictions), 2)

    # -- threshold ----------------------------------------------------

    def test_threshold_controls_how_many_emotions_fire(self):
        document = SentimentEmotionDocument(
            document_id="m", text="i am anxious and stressed about the fees"
        )
        strict = self.classifier.predict([document], threshold=0.8)
        loose = self.classifier.predict([document], threshold=0.2)
        self.assertLessEqual(
            len(strict.predictions[0].emotions), len(loose.predictions[0].emotions)
        )
        self.assertGreaterEqual(len(loose.predictions[0].emotions), 2)

    def test_high_threshold_yields_empty_emotion_set(self):
        result = self.classifier.predict(
            [SentimentEmotionDocument(document_id="m", text=EMOJI_TEXT)],
            threshold=0.99,
        )
        prediction = result.predictions[0]
        self.assertEqual(prediction.emotions, [])
        self.assertFalse(prediction.has_emotion)
        self.assertEqual(result.n_documents_without_emotion, 1)
        # The full vector is still returned so a consumer can decide otherwise.
        self.assertEqual(len(prediction.emotion_probabilities), 6)

    def test_emotions_above_rethresholds_without_inference(self):
        result = self.classifier.predict(
            [SentimentEmotionDocument(document_id="m", text=EMOJI_TEXT)]
        )
        prediction = result.predictions[0]
        for threshold in (0.1, 0.2, 0.3, 0.5):
            derived = prediction.emotions_above(threshold)
            expected = sorted(
                label
                for label, probability in prediction.emotion_probabilities.items()
                if probability > threshold
            )
            self.assertEqual(derived, expected)

    def test_constructor_threshold_override(self):
        classifier = TfidfLogisticEmotionClassifier(threshold=0.7)
        self.assertEqual(classifier.threshold, 0.7)
        classifier.train(emotion_annotations())
        self.assertEqual(classifier.training_report.threshold, 0.7)
        self.assertEqual(classifier.config_dict()["threshold"], 0.7)

    def test_invalid_threshold_rejected(self):
        for bad in (0, 1, -0.5, 1.2, "low", False):
            with self.assertRaises(ValueError):
                TfidfLogisticEmotionClassifier(threshold=bad)

    # -- unseen labels -------------------------------------------------

    def test_unseen_candidate_labels_are_reported_not_predicted(self):
        result = self.classifier.predict(
            self.documents[:2],
            candidate_labels=["stress", "boredom", "disgust", "boredom"],
        )
        self.assertEqual(result.unseen_labels, ["boredom", "disgust"])
        self.assertTrue(
            any("unseen_labels" in warning for warning in result.warnings)
        )
        for prediction in result.predictions:
            self.assertNotIn("boredom", prediction.emotions)
            self.assertNotIn("boredom", prediction.emotion_probabilities)

    def test_known_candidate_labels_are_not_flagged(self):
        result = self.classifier.predict(
            self.documents[:1], candidate_labels=self.classifier.labels
        )
        self.assertEqual(result.unseen_labels, [])

    # -- determinism ---------------------------------------------------

    def test_deterministic_across_independent_trains(self):
        other = TfidfLogisticEmotionClassifier()
        other.train(emotion_annotations())
        first = self.classifier.predict(self.documents)
        second = other.predict(self.documents)
        self.assertEqual(
            [p.emotion_probabilities for p in first.predictions],
            [p.emotion_probabilities for p in second.predictions],
        )

    def test_retraining_replaces_label_set(self):
        self.classifier.train(to_emotion_from_pairs(ALTERNATE_EMOTION_FIXTURE))
        self.assertEqual(self.classifier.labels, ["annoyance", "calm"])
        result = self.classifier.predict(self.documents[:1])
        self.assertEqual(
            sorted(result.predictions[0].emotion_probabilities),
            ["annoyance", "calm"],
        )

    # -- input handling -------------------------------------------------

    def test_input_documents_never_modified(self):
        document = SentimentEmotionDocument(
            document_id="p",
            text=EMOJI_TEXT,
            original_text="placement ka scene bahut stressful hai bro 😢",
            post_id="post-1",
            comment_id="comment-1",
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
            topic_id=2,
            topic_label="Hostel",
            issue_labels=["hostel"],
        )
        before = document.model_dump()
        self.classifier.predict([document])
        self.assertEqual(document.model_dump(), before)

    def test_result_echoes_metadata(self):
        document = SentimentEmotionDocument(
            document_id="p",
            text=EMOJI_TEXT,
            original_text="RAW <URL> text",
            post_id="post-7",
            comment_id="comment-7",
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
            topic_id=3,
            topic_label="Topic 3",
            issue_labels=["hostel", "mess_food"],
        )
        prediction = self.classifier.predict([document]).predictions[0]
        self.assertEqual(prediction.text, EMOJI_TEXT)
        self.assertEqual(prediction.original_text, "RAW <URL> text")
        self.assertEqual(prediction.post_id, "post-7")
        self.assertEqual(prediction.comment_id, "comment-7")
        self.assertTrue(prediction.is_code_mixed)
        self.assertEqual(prediction.issue_labels, ["hostel", "mess_food"])

    def test_emoji_placeholders_dropped_by_default(self):
        vocabulary = self.classifier._vectorizer.vocabulary_
        for term in vocabulary:
            self.assertNotIn("emoji", term)
            self.assertNotIn("<", term)
            self.assertNotIn(">", term)

    def test_placeholder_keeping_can_be_enabled(self):
        classifier = TfidfLogisticEmotionClassifier(
            config=EmotionModelConfig(drop_placeholder_tokens=False)
        )
        classifier.train(emotion_annotations())
        vocabulary = classifier._vectorizer.vocabulary_
        self.assertIn("emoji_loudly_crying_face", vocabulary)
        self.assertIn("emoji_tired_face", vocabulary)
        # The surrounding words are unaffected by the placeholder policy.
        self.assertIn("frustrating", vocabulary)

    def test_placeholder_dropping_documented_in_config(self):
        self.assertTrue(self.classifier.config.drop_placeholder_tokens)
        self.assertTrue(
            self.report.config["features"]["drop_placeholder_tokens"]
        )

    def test_topic_features_off_by_default(self):
        vocabulary = self.classifier._vectorizer.vocabulary_
        self.assertFalse(any(term.startswith("topicid") for term in vocabulary))

    def test_code_mixed_text_survives(self):
        vocabulary = self.classifier._vectorizer.vocabulary_
        for term in ("hai", "aur", "bahut", "ka", "chinta", "ki"):
            self.assertIn(term, vocabulary)

    # -- training refusals ---------------------------------------------

    def test_empty_training_data_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.classifier.train([])
        self.assertIn("emotion_annotations.csv", str(ctx.exception))

    def test_wrong_record_type_refused(self):
        with self.assertRaises(TypeError) as ctx:
            self.classifier.train(sentiment_annotations())
        self.assertIn("EmotionAnnotation", str(ctx.exception))

    def test_blank_emotion_label_refused(self):
        annotations = emotion_annotations()
        annotations[0] = EmotionAnnotation(
            document_id="f01", text="text", emotions=["  "]
        )
        with self.assertRaises(ValueError):
            self.classifier.train(annotations)

    def test_too_few_documents_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self.classifier.train(emotion_annotations()[:4])
        self.assertIn("insufficient training data", str(ctx.exception))

    def test_single_label_refused(self):
        annotations = [
            EmotionAnnotation(
                document_id=f"x{index}", text=f"so stressful today {index}",
                emotions=["stress"],
            )
            for index in range(6)
        ]
        with self.assertRaises(ValueError) as ctx:
            self.classifier.train(annotations)
        self.assertIn("insufficient label variation", str(ctx.exception))
        self.assertIn("never derived from sentiment", str(ctx.exception))

    def test_too_rare_label_refused(self):
        annotations = emotion_annotations() + [
            EmotionAnnotation(
                document_id="rare", text="i feel indifferent about it",
                emotions=["indifference"],
            )
        ]
        with self.assertRaises(ValueError) as ctx:
            self.classifier.train(annotations)
        self.assertIn("insufficient examples per label", str(ctx.exception))
        self.assertIn("indifference", str(ctx.exception))

    def test_degenerate_all_positive_label_refused(self):
        annotations = [
            EmotionAnnotation(
                document_id=f"a{index}",
                text=f"this is stressful and frustrating {index}",
                emotions=["stress", "frustration"],
            )
            for index in range(6)
        ]
        with self.assertRaises(ValueError) as ctx:
            self.classifier.train(annotations)
        self.assertIn("degenerate training data", str(ctx.exception))
        self.assertIn("no negative example", str(ctx.exception))

    def test_unusable_vocabulary_refused(self):
        annotations = [
            EmotionAnnotation(document_id=f"a{index}", text="", emotions=["stress"])
            for index in range(4)
        ] + [
            EmotionAnnotation(
                document_id=f"b{index}", text="<URL>", emotions=["anger"]
            )
            for index in range(4)
        ]
        with self.assertRaises(ValueError) as ctx:
            self.classifier.train(annotations)
        self.assertIn("no usable TF-IDF features", str(ctx.exception))

    def test_multi_label_path_warning_when_corpus_is_single_label(self):
        annotations = [
            EmotionAnnotation(
                document_id=f"a{index}",
                text=f"stressful frustrating {index}",
                emotions=["stress"],
            )
            for index in range(3)
        ] + [
            EmotionAnnotation(
                document_id=f"b{index}",
                text=f"joyful grateful {index}",
                emotions=["joy"],
            )
            for index in range(3)
        ]
        report = self.classifier.train(annotations)
        self.assertTrue(
            any("more than one emotion" in warning for warning in report.warnings)
        )

    # -- configuration -------------------------------------------------

    def test_invalid_config_rejected(self):
        cases = [
            {"max_features": 0},
            {"min_df": 0.5, "max_df": 0.1},
            {"ngram_range": (2, 1)},
            {"norm": None.__class__ and "nope"},
            {"token_pattern": "(["},
            {"stop_words": "hindi"},
            {"stop_words": {"a": 1}},
            {"C": 0},
            {"max_iter": 0},
            {"class_weight": "sqrt"},
            {"min_training_documents": 1},
            {"min_positive_examples_per_label": 0},
            {"threshold": 1.0},
        ]
        for case in cases:
            with self.assertRaises(ValueError, msg=str(case)):
                TfidfLogisticEmotionClassifier(config=EmotionModelConfig(**case))

    def test_valid_config_accepted(self):
        classifier = TfidfLogisticEmotionClassifier(
            config=EmotionModelConfig(
                ngram_range=(1, 1),
                stop_words="english",
                C=2.0,
                class_weight="balanced",
                threshold=0.25,
                min_positive_examples_per_label=3,
                use_topic_features=True,
                drop_placeholder_tokens=False,
            )
        )
        classifier.train(emotion_annotations())
        self.assertEqual(classifier.config_dict()["tfidf"]["ngram_range"], [1, 1])
        self.assertTrue(
            classifier.config_dict()["features"]["use_topic_features"]
        )
        self.assertFalse(
            classifier.config_dict()["features"]["drop_placeholder_tokens"]
        )

    # -- serialization --------------------------------------------------

    def test_result_json_round_trip(self):
        result = self.classifier.predict(self.documents)
        restored = EmotionAnalysisResult.model_validate_json(
            result.model_dump_json()
        )
        self.assertEqual(
            [p.emotion_probabilities for p in restored.predictions],
            [p.emotion_probabilities for p in result.predictions],
        )
        self.assertEqual(restored.model_name, "tfidf_logistic_multilabel")
        self.assertEqual(restored.model_version, "v1")

    def test_training_report_json_round_trip(self):
        restored = EmotionTrainingReport.model_validate_json(
            self.report.model_dump_json()
        )
        self.assertEqual(restored.labels, self.report.labels)
        self.assertEqual(
            restored.positive_examples_per_label,
            self.report.positive_examples_per_label,
        )

    def test_prediction_model_validates_standalone(self):
        prediction = EmotionPrediction(
            document_id="d", emotions=["stress", "anger"],
            emotion_probabilities={"stress": 0.9, "anger": 0.7}, confidence=0.9,
        )
        self.assertTrue(prediction.has_emotion)
        self.assertTrue(prediction.is_multi_label)
        self.assertEqual(prediction.top_emotion(), "stress")
        self.assertEqual(prediction.emotions_above(0.5), ["anger", "stress"])


# ===========================================================================
# Separation of the two dimensions
# ===========================================================================


class TestDimensionSeparation(unittest.TestCase):
    """Emotion must never be inferred from sentiment, and vice versa."""

    def test_sentiment_model_exposes_no_emotions(self):
        result = TfidfLogisticSentimentClassifier().train(
            sentiment_annotations()
        ) and TfidfLogisticSentimentClassifier()
        classifier = TfidfLogisticSentimentClassifier()
        classifier.train(sentiment_annotations())
        prediction_fields = set(SentimentPrediction.model_fields)
        self.assertNotIn("emotions", prediction_fields)
        self.assertNotIn("emotion_probabilities", prediction_fields)
        self.assertNotIn("emotion_labels", result.labels)

    def test_emotion_model_exposes_no_sentiment(self):
        classifier = TfidfLogisticEmotionClassifier()
        classifier.train(emotion_annotations())
        prediction_fields = set(EmotionPrediction.model_fields)
        self.assertNotIn("sentiment", prediction_fields)
        self.assertNotIn("sentiment_probabilities", prediction_fields)

    def test_emotion_predict_takes_no_sentiment_argument(self):
        """A caller cannot smuggle a sentiment prediction into the emotion model."""
        signature = inspect.signature(TfidfLogisticEmotionClassifier.predict)
        for parameter in signature.parameters:
            self.assertNotIn("sentiment", parameter)
        self.assertNotIn("sentiment", TfidfLogisticEmotionClassifier.__init__.__doc__ or "")

    def test_emotion_document_carries_no_sentiment_field(self):
        fields = set(SentimentEmotionDocument.model_fields)
        self.assertNotIn("sentiment", fields)
        self.assertNotIn("emotions", fields)

    def test_negative_document_can_have_no_emotion(self):
        """A complaint is not obliged to carry an emotional label."""
        classifier = TfidfLogisticEmotionClassifier()
        classifier.train(emotion_annotations())
        result = classifier.predict(
            [SentimentEmotionDocument(document_id="n", text="this is horrible and awful")]
        )
        prediction = result.predictions[0]
        if not prediction.emotions:
            self.assertEqual(result.n_documents_without_emotion, 1)
        self.assertEqual(len(prediction.emotion_probabilities), 6)

    def test_same_text_maps_to_different_emotions_per_corpus(self):
        """Emotion labels are annotation-driven, not sentiment-derived."""
        text = "this is really not good at all"
        first = TfidfLogisticEmotionClassifier()
        first.train(emotion_annotations())
        second = TfidfLogisticEmotionClassifier()
        second.train(to_emotion_from_pairs(ALTERNATE_EMOTION_FIXTURE))
        document = SentimentEmotionDocument(document_id="d", text=text)
        self.assertEqual(
            sorted(first.predict([document]).predictions[0].emotion_probabilities),
            ["anger", "anxiety", "frustration", "gratitude", "joy", "stress"],
        )
        self.assertEqual(
            sorted(second.predict([document]).predictions[0].emotion_probabilities),
            ["annoyance", "calm"],
        )

    def test_the_brief_example_lands_in_the_right_cell(self):
        """
        The brief's example: negative sentiment, frustration emotion, as two
        independent outputs.

        The sentiment assertion is on the winning class (``top_sentiment``) and
        the confidence floor is lowered for the check, because on a
        35-document baseline the winning probability can legitimately land just
        under 0.5.  That is the documented abstain behaviour, not a wrong class.
        """
        sentiment = TfidfLogisticSentimentClassifier()
        sentiment.train(sentiment_annotations())
        emotion = TfidfLogisticEmotionClassifier()
        emotion.train(emotion_annotations())
        document = SentimentEmotionDocument(
            document_id="x",
            text="placement pressure hai aur hostel ka stress bhi hai",
            language="hi",
            script="latin",
            is_code_mixed=True,
        )
        sentiment_prediction = sentiment.predict([document]).predictions[0]
        emotion_prediction = emotion.predict([document], threshold=0.15).predictions[0]

        self.assertEqual(sentiment_prediction.top_sentiment(), "negative")
        self.assertEqual(
            sentiment.predict([document], threshold=0.4).predictions[0].sentiment,
            "negative",
        )
        self.assertIn("frustration", emotion_prediction.emotions)
        self.assertIn("stress", emotion_prediction.emotions)
        # The two probability vectors are independent evidence.
        self.assertNotIn(
            sentiment_prediction.top_sentiment(),
            emotion_prediction.emotion_probabilities,
        )

    def test_sentiment_output_is_unchanged_by_emotion_training(self):
        """Training the emotion model must not touch the sentiment model."""
        emotion = TfidfLogisticEmotionClassifier()
        emotion.train(emotion_annotations())
        sentiment = TfidfLogisticSentimentClassifier()
        sentiment.train(sentiment_annotations())
        document = SentimentEmotionDocument(document_id="p", text=POSITIVE_TEXT)
        before = sentiment.predict([document]).predictions[0]
        emotion.predict([SentimentEmotionDocument(document_id="y", text=EMOJI_TEXT)])
        after = sentiment.predict([document]).predictions[0]
        self.assertEqual(before.sentiment_probabilities, after.sentiment_probabilities)


# ===========================================================================
# Multilingual, code-mixed, emoji
# ===========================================================================


class TestMultilingualAndCodeMixing(unittest.TestCase):
    def setUp(self):
        self.sentiment = TfidfLogisticSentimentClassifier()
        self.sentiment.train(sentiment_annotations())
        self.emotion = TfidfLogisticEmotionClassifier()
        self.emotion.train(emotion_annotations())

    def test_english(self):
        document = SentimentEmotionDocument(
            document_id="en", text=POSITIVE_TEXT, language="en", script="latin"
        )
        prediction = self.sentiment.predict([document]).predictions[0]
        self.assertEqual(prediction.sentiment, "positive")
        self.assertEqual(prediction.language, "en")

    def test_hindi_devanagari(self):
        document = SentimentEmotionDocument(
            document_id="hi",
            text="exam pressure ne mujhe puri tarah tod diya",
            language="hi",
            script="devanagari",
        )
        prediction = self.sentiment.predict([document]).predictions[0]
        self.assertEqual(prediction.sentiment, "negative")
        self.assertEqual(prediction.script, "devanagari")

    def test_romanized_hindi(self):
        document = SentimentEmotionDocument(
            document_id="hi-roman",
            text="hostel ka khana bahut ghatiya aur ganda hai",
            language="hi",
            script="latin",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
        )
        prediction = self.sentiment.predict([document]).predictions[0]
        self.assertEqual(prediction.sentiment, "negative")
        self.assertTrue(prediction.is_code_mixed)

    def test_bengali(self):
        """Native-script Bengali is tokenised and classified, not dropped."""
        document = SentimentEmotionDocument(
            document_id="bn",
            text="বিষয় নিয়ে আমাদের দুজনের মধ্যে অনেক কষ্ট হলো",
            language="bn",
            script="bengali",
        )
        prediction = self.sentiment.predict([document]).predictions[0]
        self.assertEqual(prediction.language, "bn")
        self.assertEqual(prediction.script, "bengali")
        self.assertEqual(prediction.text, document.text)
        self.assertEqual(prediction.sentiment, "negative")

    def test_odia(self):
        """Native-script Odia is tokenised and classified, not dropped."""
        document = SentimentEmotionDocument(
            document_id="or",
            text="ସେବା ଓ ରୋମ ରକ୍ଷଣାର ଅବସ୍ଥା ବହୁତ ଖରାପ",
            language="or",
            script="oriya",
        )
        prediction = self.sentiment.predict([document]).predictions[0]
        self.assertEqual(prediction.language, "or")
        self.assertEqual(prediction.script, "oriya")
        self.assertEqual(prediction.text, document.text)
        self.assertEqual(prediction.sentiment, "negative")

    def test_hinglish_emotion(self):
        document = SentimentEmotionDocument(
            document_id="hinglish",
            text="chinta bahut ho rahi hai placement ki",
            language="hi",
            script="devanagari",
            is_code_mixed=True,
            detected_languages=["hi", "en"],
        )
        prediction = self.emotion.predict([document], threshold=0.15).predictions[0]
        self.assertIsNotNone(prediction.emotions)
        self.assertTrue(prediction.is_code_mixed)

    def test_native_script_terms_survive_tokenisation(self):
        """
        The token pattern is script-agnostic, so Indic terms are learnable.

        Bengali and Odia are written without spaces, so the assertions look for
        the script's codepoint block in the vocabulary rather than for a
        particular word.
        """
        vocabulary = self.sentiment._vectorizer.vocabulary_
        self.assertIn("tod", vocabulary)  # Romanized Hindi
        self.assertTrue(
            any("আ" in term for term in vocabulary),
            "no Bengali-script term reached the vectorizer vocabulary",
        )
        self.assertTrue(
            any("ଅ" in term for term in vocabulary),
            "no Odia-script term reached the vectorizer vocabulary",
        )
        # The Indic blocks are genuinely populated, not just one stray term.
        bengali = [
            term for term in vocabulary if any("\u0980" <= c <= "\u09ff" for c in term)
        ]
        odia = [
            term for term in vocabulary if any("\u0b00" <= c <= "\u0b7f" for c in term)
        ]
        self.assertGreater(len(bengali), 20, bengali[:10])
        self.assertGreater(len(odia), 20, odia[:10])

    def test_all_scripts_predict_without_crashing(self):
        documents = [
            SentimentEmotionDocument(document_id="en", text=POSITIVE_TEXT),
            SentimentEmotionDocument(
                document_id="hi", text="exam pressure ne mujhe tod diya"
            ),
            SentimentEmotionDocument(
                document_id="roman", text="hostel ka khana bahut ghatiya hai"
            ),
            SentimentEmotionDocument(
                document_id="bn", text="আমি খুব খুশি এবং কৃতজ্ঞ"
            ),
            SentimentEmotionDocument(
                document_id="or", text="ମୁଁ ବହୁତ କାଷ୍ଟିରେ ପଡିଛି"
            ),
            SentimentEmotionDocument(document_id="emoji", text=EMOJI_TEXT),
        ]
        for classifier in (self.sentiment, self.emotion):
            result = classifier.predict(documents)
            self.assertEqual(len(result.predictions), len(documents))
            for prediction in result.predictions:
                self.assertIsNotNone(prediction)

    def test_emoji_semantic_token_does_not_become_vocabulary(self):
        for classifier in (self.sentiment, self.emotion):
            for term in classifier._vectorizer.vocabulary_:
                self.assertNotIn("emoji", term)

    def test_emoji_placeholder_does_not_leak_into_result_text(self):
        document = SentimentEmotionDocument(
            document_id="e", text=EMOJI_TEXT, original_text="😢 stressed bro"
        )
        prediction = self.emotion.predict([document]).predictions[0]
        self.assertEqual(prediction.text, EMOJI_TEXT)
        self.assertIn("<EMOJI_loudly_crying_face>", prediction.text)
        self.assertEqual(prediction.original_text, "😢 stressed bro")


# ===========================================================================
# Evaluation
# ===========================================================================


class TestSentimentEvaluation(unittest.TestCase):
    def setUp(self):
        self.classifier = TfidfLogisticSentimentClassifier()
        self.classifier.train(sentiment_annotations())
        self.eval_annotations = to_sentiment_from_pairs(SENTIMENT_EVAL)

    def test_meaningful_report_on_sufficient_data(self):
        report = self.classifier.evaluate(self.eval_annotations)
        self.assertIsInstance(report, SentimentEvaluationReport)
        self.assertTrue(report.is_meaningful, report.reason)
        self.assertIsNone(report.reason)
        self.assertEqual(report.n_documents, 12)
        self.assertEqual(report.n_classes, 3)
        self.assertEqual(sorted(report.class_order), ["negative", "neutral", "positive"])
        # Three held-out documents land below the confidence floor. They are
        # reported, not hidden, and they count as misses in accuracy.
        self.assertEqual(report.n_abstained, 3)
        self.assertEqual(report.n_unscoreable, 0)
        self.assertTrue(
            any("abstained" in warning for warning in report.warnings), report.warnings
        )

    def test_abstentions_counted_as_misses(self):
        """The three abstentions are wrong answers, not excluded rows."""
        report = self.classifier.evaluate(self.eval_annotations)
        predicted = self.classifier.predict(
            [
                SentimentEmotionDocument(document_id=record.document_id, text=record.text)
                for record in self.eval_annotations
            ]
        )
        self.assertEqual(
            sum(1 for p in predicted.predictions if p.abstained), report.n_abstained
        )
        self.assertEqual(sum(report.per_class_support.values()), report.n_documents)
        # Rows of the matrix only count emitted classes, so the shortfall is
        # exactly the number of abstentions.
        self.assertEqual(
            sum(sum(row) for row in report.confusion_matrix),
            report.n_documents - report.n_abstained,
        )
        self.assertAlmostEqual(
            report.accuracy,
            sum(sum(row) for row in report.confusion_matrix) / report.n_documents,
            places=5,
        )

    def test_known_baseline_misses_are_real(self):
        """
        Document the real weakness of this baseline rather than hiding it.

        The whole ``negative`` gold class is where it fails: of four negative
        held-out documents only one is emitted above the floor, and the other
        three are misses. Two of them have the right winning class but fall
        under the 0.5 floor, and one ("i am not happy with the hostel") is a
        negation this bag-of-words model simply does not represent. The numbers
        in the other tests should be read with that in mind.
        """
        report = self.classifier.evaluate(self.eval_annotations)
        index = {name: i for i, name in enumerate(report.class_order)}
        negative_row = report.confusion_matrix[index["negative"]]
        self.assertEqual(negative_row, [1, 0, 0])
        self.assertEqual(report.n_abstained, 3)
        # One negative document wins "negative" but is below the floor; the
        # abstention is reported rather than quietly counted as a hit.
        self.assertEqual(
            sum(report.per_class_predicted.values()), report.n_documents - report.n_abstained
        )
        self.assertAlmostEqual(report.accuracy, 9 / 12, places=5)

    def test_all_required_metrics_present(self):
        report = self.classifier.evaluate(self.eval_annotations)
        for field in (
            "accuracy",
            "macro_f1",
            "micro_f1",
            "macro_precision",
            "micro_precision",
            "macro_recall",
            "micro_recall",
            "per_class_f1",
            "per_class_precision",
            "per_class_recall",
            "per_class_support",
            "confusion_matrix",
            "classification_report",
        ):
            self.assertIsNotNone(getattr(report, field), field)
        self.assertGreaterEqual(report.accuracy, 0.0)
        self.assertLessEqual(report.accuracy, 1.0)
        self.assertGreaterEqual(report.macro_f1, 0.0)
        self.assertLessEqual(report.macro_f1, 1.0)
        # micro-F1 can exceed macro-F1 here precisely because abstentions are
        # counted as misses: an all-zero row is free in a micro average but
        # costs every class its recall. Micro is not the flattering number.
        self.assertGreaterEqual(report.micro_f1, report.macro_f1)

    def test_confusion_matrix_is_square_and_ordered(self):
        report = self.classifier.evaluate(self.eval_annotations)
        size = len(report.class_order)
        self.assertEqual(len(report.confusion_matrix), size)
        for row in report.confusion_matrix:
            self.assertEqual(len(row), size)
        index = {name: i for i, name in enumerate(report.class_order)}
        matrix = report.confusion_matrix
        # Every document with a gold class is either predicted into a row of its
        # own class or, if abstained, missing from the matrix. Both are stated.
        for name, support in report.per_class_support.items():
            self.assertLessEqual(sum(matrix[index[name]]), support)
        self.assertEqual(
            sum(sum(row) for row in matrix)
            + report.n_abstained
            + report.n_unscoreable,
            report.n_documents,
        )

    def test_confusion_matrix_rows_render(self):
        report = self.classifier.evaluate(self.eval_annotations)
        rows = report.confusion_matrix_as_rows()
        self.assertEqual(len(rows), len(report.class_order))

    def test_per_class_metrics_cover_every_class(self):
        report = self.classifier.evaluate(self.eval_annotations)
        for name in report.class_order:
            self.assertIn(name, report.per_class_f1)
            self.assertIn(name, report.per_class_support)
        self.assertEqual(sum(report.per_class_support.values()), 12)

    def test_metrics_withheld_when_too_small(self):
        report = self.classifier.evaluate(self.eval_annotations[:9])
        self.assertFalse(report.is_meaningful)
        self.assertIn("at least 10", report.reason)
        self.assertIsNone(report.accuracy)
        self.assertIsNone(report.macro_f1)
        self.assertIsNone(report.micro_f1)
        self.assertIsNone(report.confusion_matrix)
        self.assertIsNone(report.classification_report)
        # Support is still reported: it is a fact about the data.
        self.assertEqual(sum(report.per_class_support.values()), 9)

    def test_metrics_withheld_without_class_diversity(self):
        annotations = to_sentiment_from_pairs(
            [(f"u{i}", "the workshop is on monday", "neutral") for i in range(12)]
        )
        report = self.classifier.evaluate(annotations)
        self.assertFalse(report.is_meaningful)
        self.assertIn("sentiment class(es) present", report.reason)
        self.assertIsNone(report.accuracy)
        self.assertEqual(report.n_documents, 12)

    def test_thin_class_support_warns(self):
        annotations = to_sentiment_from_pairs(
            [("t1", "the workshop is on monday", "neutral")] * 11
            + [("t2", "this is horrible and terrible", "negative")]
        )
        report = self.classifier.evaluate(annotations)
        self.assertTrue(
            any("unstable" in warning for warning in report.warnings), report.warnings
        )

    def test_unseen_gold_class_is_reported(self):
        annotations = to_sentiment_from_pairs(SENTIMENT_EVAL) + [
            SentimentAnnotation(
                document_id="zz", text="mixed bag here", sentiment="mixed"
            )
        ]
        report = self.classifier.evaluate(annotations)
        self.assertTrue(
            any("never trained on" in warning for warning in report.warnings),
            report.warnings,
        )

    def test_abstention_counted_as_miss(self):
        report = self.classifier.evaluate(self.eval_annotations, threshold=0.999)
        self.assertTrue(report.is_meaningful)
        self.assertEqual(report.n_abstained, 12)
        self.assertEqual(report.accuracy, 0.0)
        self.assertTrue(
            any("abstained" in warning for warning in report.warnings)
        )

    def test_self_scoring_is_flagged(self):
        report = self.classifier.evaluate(sentiment_annotations())
        self.assertTrue(
            any("NOT held-out" in warning for warning in report.warnings),
            report.warnings,
        )

    def test_no_self_scoring_warning_on_held_out_data(self):
        report = self.classifier.evaluate(self.eval_annotations)
        self.assertFalse(
            any("NOT held-out" in warning for warning in report.warnings),
            report.warnings,
        )

    def test_empty_evaluation_set(self):
        report = self.classifier.evaluate([])
        self.assertFalse(report.is_meaningful)
        self.assertEqual(report.n_documents, 0)
        self.assertIsNone(report.accuracy)

    def test_format_summary_states_withholding(self):
        report = self.classifier.evaluate(self.eval_annotations[:3])
        summary = report.format_summary()
        self.assertIn("NOT MEANINGFUL", summary)
        self.assertIn("withheld", summary)

    def test_format_summary_prints_metrics(self):
        report = self.classifier.evaluate(self.eval_annotations)
        summary = report.format_summary()
        self.assertIn("accuracy", summary)
        self.assertIn("macro-F1", summary)
        self.assertIn("confusion matrix", summary)

    def test_evaluate_helper_directly(self):
        report = evaluate_sentiment(
            ["positive", "negative", "positive", "neutral"],
            ["positive", "negative", "neutral", "neutral"],
            ["negative", "neutral", "positive"],
        )
        self.assertFalse(report.is_meaningful)
        self.assertEqual(sum(report.per_class_support.values()), 4)

    def test_evaluate_helper_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            evaluate_sentiment(["a", "b"], ["a"], ["a", "b"])


class TestEmotionEvaluation(unittest.TestCase):
    def setUp(self):
        self.classifier = TfidfLogisticEmotionClassifier()
        self.classifier.train(emotion_annotations())
        self.eval_annotations = to_emotion_from_pairs(EMOTION_EVAL)

    def test_meaningful_report_on_sufficient_data(self):
        report = self.classifier.evaluate(self.eval_annotations)
        self.assertIsInstance(report, EmotionEvaluationReport)
        self.assertTrue(report.is_meaningful, report.reason)
        self.assertEqual(report.n_documents, 12)
        self.assertEqual(report.n_labels, 6)

    def test_all_required_metrics_present(self):
        report = self.classifier.evaluate(self.eval_annotations)
        for field in (
            "micro_f1",
            "macro_f1",
            "micro_precision",
            "macro_precision",
            "macro_recall",
            "per_label_f1",
            "per_label_precision",
            "per_label_recall",
            "per_label_support",
            "subset_accuracy",
            "hamming_accuracy",
            "hamming_loss",
            "classification_report",
        ):
            self.assertIsNotNone(getattr(report, field), field)
        self.assertLessEqual(report.subset_accuracy, report.micro_f1)

    def test_per_label_metrics_cover_every_label(self):
        report = self.classifier.evaluate(self.eval_annotations)
        self.assertEqual(len(report.per_label_f1), 6)
        self.assertEqual(sorted(report.per_label_f1), sorted(report.per_label_support))

    def test_documents_with_no_emotion_are_evaluable(self):
        report = self.classifier.evaluate(self.eval_annotations)
        self.assertGreaterEqual(
            sum(1 for a in self.eval_annotations if not a.emotions), 1
        )
        self.assertTrue(report.is_meaningful)

    def test_metrics_withheld_when_too_small(self):
        report = self.classifier.evaluate(self.eval_annotations[:6])
        self.assertFalse(report.is_meaningful)
        self.assertIn("at least 10", report.reason)
        self.assertIsNone(report.micro_f1)
        self.assertIsNone(report.macro_f1)
        self.assertIsNone(report.subset_accuracy)
        self.assertIsNone(report.hamming_accuracy)
        # Support counts gold *label positives*, not documents, so a slice with a
        # multi-emotion document sums to more than its document count.
        self.assertEqual(
            sum(report.per_label_support.values()),
            sum(len(a.emotions) for a in self.eval_annotations[:6]),
        )

    def test_metrics_withheld_without_label_diversity(self):
        """
        A single evaluated emotion cannot support macro-averages.

        Reached through ``evaluate_emotions`` directly, because a trained
        classifier always evaluates every label it knows — the gate protects the
        helper against a degenerate evaluation matrix.
        """
        report = evaluate_emotions(
            [[1]] * 12,
            [[1]] * 12,
            ["stress"],
        )
        self.assertFalse(report.is_meaningful)
        self.assertIn("emotion label(s) evaluated", report.reason)
        self.assertIsNone(report.micro_f1)
        self.assertIsNone(report.macro_f1)

    def test_no_zero_emotion_documents_warns_about_optimism(self):
        annotations = to_emotion_from_pairs(
            [(f"v{i}", f"i am stressed and anxious {i}", ["stress", "anxiety"])
             for i in range(12)]
        )
        report = self.classifier.evaluate(annotations)
        self.assertTrue(
            any("zero emotions" in warning for warning in report.warnings),
            report.warnings,
        )

    def test_self_scoring_is_flagged(self):
        report = self.classifier.evaluate(emotion_annotations())
        self.assertTrue(
            any("NOT held-out" in warning for warning in report.warnings),
            report.warnings,
        )

    def test_empty_evaluation_set(self):
        report = self.classifier.evaluate([])
        self.assertFalse(report.is_meaningful)
        self.assertEqual(report.n_documents, 0)

    def test_format_summary(self):
        summary = self.classifier.evaluate(self.eval_annotations).format_summary()
        self.assertIn("micro-F1", summary)
        self.assertIn("subset accuracy", summary)
        self.assertIn("per-label F1", summary)
        withheld = self.classifier.evaluate(self.eval_annotations[:2]).format_summary()
        self.assertIn("NOT MEANINGFUL", withheld)

    def test_evaluate_helper_directly(self):
        report = evaluate_emotions(
            [[1, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1]],
            [[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 0, 0]],
            ["a", "b", "c"],
        )
        self.assertFalse(report.is_meaningful)
        self.assertIn("at least 10", report.reason)
        self.assertEqual(sum(report.per_label_support.values()), 6)
        # Support is a fact about the data and is reported even when withheld.
        self.assertEqual(report.per_label_support, {"a": 2, "b": 2, "c": 2})

    def test_evaluate_helper_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            evaluate_emotions([[1, 0]], [[1, 0], [0, 1]], ["a", "b"])
        with self.assertRaises(ValueError):
            evaluate_emotions([[1, 0]], [[1, 0]], ["a"])


# ===========================================================================
# Architectural isolation
# ===========================================================================


def _package_modules():
    for directory, _dirs, files in os.walk(PACKAGE_ROOT):
        if "__pycache__" in directory:
            continue
        for name in sorted(files):
            if name.endswith(".py"):
                yield os.path.join(directory, name)


class TestArchitectureIsolation(unittest.TestCase):
    def test_package_contains_expected_modules(self):
        names = {os.path.basename(path) for path in _package_modules()}
        for expected in (
            "base.py",
            "annotation.py",
            "result.py",
            "evaluation.py",
            "tfidf_logistic_sentiment.py",
            "tfidf_logistic_emotion.py",
            "tfidf_logistic_base.py",
            "__init__.py",
        ):
            self.assertIn(expected, names)

    def test_package_imports_nothing_outside_itself(self):
        """
        No first-party coupling to preprocessing, topics, issues, or the database.

        Third-party libraries (pydantic, sklearn, numpy) and the standard
        library are fine; only ``app.*`` imports are constrained.
        """
        for path in _package_modules():
            with open(path, "r", encoding="utf-8") as handle:
                tree = ast.parse(handle.read(), filename=path)
            modules = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    modules.add(node.module)
            first_party = {
                module for module in modules if module == "app" or module.startswith("app.")
            }
            for module in first_party:
                self.assertTrue(
                    module.startswith("app.services.nlp.sentiment_emotion"),
                    f"{os.path.basename(path)} imports {module}",
                )

    def test_preprocessing_is_not_imported(self):
        for path in _package_modules():
            source = open(path, "r", encoding="utf-8").read()
            self.assertNotIn("app.services.nlp.preprocessing", source)
            self.assertNotIn("MultilingualTextPreprocessor", source)

    def test_no_database_or_ingestion_coupling(self):
        for path in _package_modules():
            source = open(path, "r", encoding="utf-8").read()
            for forbidden in (
                "sqlalchemy",
                "SessionLocal",
                "get_db",
                "app.database",
                "app.models",
                "app.services.ingest",
                "praw",
            ):
                self.assertNotIn(forbidden, source, f"{os.path.basename(path)}")

    def test_earlier_stages_are_not_modified_by_this_stage(self):
        """This stage must not reach into the completed stages."""
        for path in _package_modules():
            source = open(path, "r", encoding="utf-8").read()
            for forbidden in (
                "app.services.nlp.topic_discovery",
                "app.services.nlp.issue_identification",
                "app.services.nlp.language_analysis_service",
            ):
                self.assertNotIn(forbidden, source, f"{os.path.basename(path)}")

    def test_existing_stages_still_importable(self):
        """Adding this stage did not break the completed ones."""
        from app.services.nlp.issue_identification import (  # noqa: F401
            IssueModelConfig,
            TfidfLogisticIssueClassifier,
        )
        from app.services.nlp.topic_discovery import (  # noqa: F401
            NMFTopicDiscoverer,
        )

    def test_no_opinion_lexicon_is_shipped(self):
        """Labels must come from annotation data, not a hard-coded word list."""
        for path in _package_modules():
            source = open(path, "r", encoding="utf-8").read().lower()
            for lexicon_marker in (
                "positive_words",
                "negative_words",
                "sentiment_lexicon",
                "emotion_lexicon",
                "opinion_words",
                "wordnet",
                "vader",
                "nltk",
                "textblob",
            ):
                self.assertNotIn(lexicon_marker, source, os.path.basename(path))

    def test_input_document_defaults(self):
        document = SentimentEmotionDocument(document_id="d")
        self.assertEqual(document.text, "")
        self.assertEqual(document.language, "unknown")
        self.assertEqual(document.script, "unknown")
        self.assertFalse(document.is_code_mixed)
        self.assertEqual(document.detected_languages, [])
        self.assertEqual(document.issue_labels, [])
        self.assertEqual(document.issue_probabilities, {})
        self.assertFalse(document.has_issue)


if __name__ == "__main__":
    unittest.main()
