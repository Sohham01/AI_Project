"""
Demo script: SENTIMENT and EMOTION analysis as two separate models, on
synthetic IIM Sambalpur Community Intelligence data.

=============================================================================
ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE
Every document, annotation and metric below is synthetic and hand-written. The
corpora were not collected from Reddit and not annotated by real annotators.
The printed scores exist only to show that the code paths work end to end.
Real numbers require the annotation workflow in docs/sentiment_emotion.md.
=============================================================================

Sentiment and emotion are different dimensions, and this demo keeps them apart
on purpose:

    "placement season is horrible"
        sentiment = negative        (one mutually exclusive class)
        emotion   = frustration     (a set of labels)

    "I was nervous but it worked out well"
        sentiment = positive        (overall polarity)
        emotion   = anxiety         (NOT implied by the polarity)

Nothing here infers emotion from sentiment. The two dimensions have separate
annotation types, separate configurations, separate classifiers and separate
training runs, so the nine sections below cannot leak into each other:

    1. SENTIMENT ANNOTATIONS   synthetic SentimentAnnotation records
    2. SENTIMENT TRAINING      class set learned from those annotations
    3. SENTIMENT PREDICTION    one class + full distribution, or an abstention
    4. EMOTION ANNOTATIONS     synthetic EmotionAnnotation records
    5. EMOTION TRAINING        label set learned from those annotations
    6. EMOTION PREDICTION      independent sigmoids, several may fire at once
    7. DIMENSION SEPARATION    the same text, two independent outputs
    8. EVALUATION              held-out sanity check of both metric paths
    9. DATA ADEQUACY GUARDS    metrics withheld, and train() refusing bad data

Upstream context (topic id, issue labels) is carried through untouched and is
used as a feature only when explicitly enabled.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Windows consoles default to cp1252 and cannot render Devanagari / Bengali /
# Odia. Force UTF-8 so the multilingual output is readable.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.nlp.sentiment_emotion import (
    EmotionAnnotation,
    EmotionCorpus,
    EmotionModelConfig,
    SentimentAnnotation,
    SentimentCorpus,
    SentimentEmotionDocument,
    SentimentModelConfig,
    TfidfLogisticEmotionClassifier,
    TfidfLogisticSentimentClassifier,
)


# ---------------------------------------------------------------------------
# 1. SENTIMENT ANNOTATIONS
#
# (document_id, preprocessed text, sentiment, language, script, code-mixed)
# `text` is what the preprocessing stage would have produced: emoji already
# turned into <EMOJI_...> tokens, URLs into <URL>.
# Sentiment is SINGLE-LABEL and mutually exclusive, so the third field is one
# class, not a list.
# ---------------------------------------------------------------------------
SENTIMENT_TRAINING = [
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
    ("n11", "ସେବା ଓ ରୋମ ରକ୍ଷଣାର ଅବସ୍ଥା ବହୁତ ଖରାପ", "negative", "or", "oriya", True),
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

# Unseen documents for prediction.
# (document_id, text, original_text, language, script, code-mixed, topic, issues)
PREDICTION_DOCUMENTS = [
    ("d1", "this placement process is horrible", "This placement process is horrible.",
     "en", "latin", False, 1, ["placement_stress"]),
    ("d2", "what a wonderful experience overall", "What a wonderful experience overall!",
     "en", "latin", False, 4, []),
    ("d3", "the seminar is on monday", "The seminar is on Monday.",
     "en", "latin", False, 2, []),
    ("d4", "hostel ka khana bahut ghatiya hai", "Hostel ka khana bahut ghatiya hai 😡",
     "hi", "latin", True, 3, ["mess_food", "hostel"]),
    ("d5", "chinta bahut ho rahi hai placement ki", "Chinta bahut ho rahi hai placement ki 😟",
     "hi", "devanagari", True, 1, ["placement_stress"]),
    ("d6", "fees <EMOJI_loudly_crying_face> ROI <EMOJI_fire> placement stress",
     "Fees 💸 ROI 🔥 placement stress 😭", "en", "latin", False, 0, ["fees", "roi"]),
    ("d7", "বিষয় নিয়ে আমাদের দুজনের মধ্যে অনেক কষ্ট হলো",
     "বিষয় নিয়ে আমাদের দুজনের মধ্যে অনেক কষ্ট হলো।",
     "bn", "bengali", False, 1, []),
    ("d8", "ସେବା ଓ ରୋମ ରକ୍ଷଣାର ଅବସ୍ଥା ବହୁତ ଖରାପ",
     "ସେବା ଓ ରୋମ ରକ୍ଷଣାର ଅବସ୍ଥା ବହୁତ ଖରାପ।",
     "or", "oriya", True, 3, ["hostel"]),
    ("d9", "placement pressure hai aur hostel ka stress bhi hai",
     "Placement pressure hai aur hostel ka stress bhi hai 😫",
     "hi", "latin", True, 1, ["placement_stress", "hostel"]),
]

# Held out from the corpora below so section 8 is not self-scored.
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


# ---------------------------------------------------------------------------
# 4. EMOTION ANNOTATIONS
#
# (document_id, preprocessed text, emotions, language, script, code-mixed)
# Emotion is MULTI-LABEL, so the third field is a list and an empty list is
# valid supervision: "no emotion was annotated here".
# The label space is whatever the annotators agreed on. This demo happens to use
# six labels; nothing in the code knows that.
# ---------------------------------------------------------------------------
EMOTION_TRAINING = [
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
    # --- Bengali / Odia, same shared label space ---
    ("b01", "আমি খুব উদ্বিগ্ন এবং চাপে ভরা", ["anxiety", "stress"], "bn", "bengali", False),
    ("b02", "আমি খুব খুশি এবং কৃতজ্ঞ", ["joy", "gratitude"], "bn", "bengali", False),
    ("o01", "ମୁଁ ବହୁତ କାଷ୍ଟିରେ ପଡିଛି", ["frustration"], "or", "odia", False),
    # --- no emotion at all: valid negative supervision ---
    ("z01", "the mess serves rice and dal", [], "en", "latin", False),
    ("z02", "placement process hai bas", [], "hi", "latin", True),
    ("z03", "the exam was on monday", [], "en", "latin", False),
    ("z04", "i attended the workshop yesterday", [], "en", "latin", False),
    # --- the brief's multi-emotion example ---
    ("m01", "placement pressure hai aur hostel ka stress bhi hai", ["stress", "frustration"], "hi", "latin", True),
]

EMOTION_EVAL = [
    ("w01", "this is frustrating", ["frustration"]),
    ("w02", "the workload is stressful", ["stress"]),
    ("w03", "i am anxious about the results", ["anxiety"]),
    ("w04", "i am angry about this delay", ["anger"]),
    ("w05", "i am grateful for the help", ["gratitude"]),
    ("w06", "i am delighted with the outcome", ["joy"]),
    ("w07", "the seminar is on monday", []),
    ("w08", "i am nervous and worried", ["anxiety", "stress"]),
    ("w09", "this delay is unacceptable", ["anger"]),
    ("w10", "i am really happy about how this turned out", ["joy"]),
    ("w11", "this whole thing is frustrating and stressful", ["frustration", "stress"]),
    ("w12", "nothing much to report", []),
]

# A second annotation round with a deliberately different taxonomy, used in
# section 7c to show the emotion label space follows the data.
ALTERNATE_EMOTION_TRAINING = [
    ("k01", "i feel completely calm about the result", ["calm"]),
    ("k02", "everything is peaceful today", ["calm"]),
    ("k03", "i am so irritated by the noise", ["annoyance"]),
    ("k04", "this noise is irritating me", ["annoyance"]),
    ("k05", "i feel calm and not annoyed", ["calm", "annoyance"]),
    ("k06", "the seminar is on tuesday", []),
    ("k07", "the library opens at eight", []),
    ("k08", "the forms are due on friday", []),
]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def to_sentiment_annotations(rows) -> list:
    return [
        SentimentAnnotation(
            document_id=doc_id,
            text=text,
            sentiment=sentiment,
            language=language,
            script=script,
            is_code_mixed=code_mixed,
            detected_languages=([language, "en"] if code_mixed else [language]),
        )
        for doc_id, text, sentiment, language, script, code_mixed in rows
    ]


def to_emotion_annotations(rows) -> list:
    return [
        EmotionAnnotation(
            document_id=doc_id,
            text=text,
            emotions=list(emotions),
            language=language,
            script=script,
            is_code_mixed=code_mixed,
            detected_languages=([language, "en"] if code_mixed else [language]),
        )
        for doc_id, text, emotions, language, script, code_mixed in rows
    ]


def to_sentiment_eval(rows) -> list:
    """Sentiment annotations from compact (document_id, text, sentiment) rows."""
    return [
        SentimentAnnotation(document_id=doc_id, text=text, sentiment=sentiment)
        for doc_id, text, sentiment in rows
    ]


def to_emotion_eval(rows) -> list:
    """Emotion annotations from compact (document_id, text, emotions) rows."""
    return [
        EmotionAnnotation(document_id=doc_id, text=text, emotions=list(emotions))
        for doc_id, text, emotions in rows
    ]


def to_documents(rows) -> list:
    return [
        SentimentEmotionDocument(
            document_id=doc_id,
            text=text,
            original_text=original,
            language=language,
            script=script,
            is_code_mixed=code_mixed,
            detected_languages=([language, "en"] if code_mixed else [language]),
            topic_id=topic_id,
            topic_label=f"Topic {topic_id}",
            issue_labels=list(issue_labels),
        )
        for doc_id, text, original, language, script, code_mixed, topic_id, issue_labels
        in rows
    ]


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Sentiment annotations
# ---------------------------------------------------------------------------
def section_sentiment_annotations(annotations: list) -> None:
    rule("1. SENTIMENT ANNOTATIONS  (synthetic, hand-written, single-label)")
    corpus = SentimentCorpus(annotations=annotations)
    print()
    print(f"  annotations        : {len(corpus)}")
    print(f"  classes learned    : {len(corpus.labels)}  {corpus.labels}")
    print("  classes per document: 1 (mutually exclusive by definition)")
    print()
    print("  documents per class:")
    for label, count in corpus.class_counts().items():
        print(f"    {label:<20} {count}")
    print()
    print("  a few records:")
    for annotation in annotations[:2] + annotations[8:10] + annotations[-2:]:
        print(f"    {annotation.document_id}  sentiment={annotation.sentiment!r}  "
              f"({annotation.language}/{annotation.script})")
        print(f"      text: {annotation.text}")


# ---------------------------------------------------------------------------
# 2. Sentiment training
# ---------------------------------------------------------------------------
def section_sentiment_training(report) -> None:
    rule("2. SENTIMENT TRAINING  (class set LEARNED from the annotations)")
    print()
    print(f"  model              : {report.model_name} {report.model_version}")
    print(f"  training documents : {report.n_training_documents}")
    print(f"  classes learned    : {report.n_classes}  {report.labels}")
    print(f"  distinct features  : {report.n_features}")
    print(f"  confidence floor   : {report.threshold}")
    print(f"  majority share     : {report.majority_class_share:.3f}")
    print(f"  usable             : {report.is_usable}")
    print()
    print("  examples per class (the data-adequacy evidence):")
    for label, count in report.examples_per_class.items():
        print(f"    {label:<20} {count}")
    print()
    print("  config snapshot:")
    print(f"    tfidf             : {report.config['tfidf']}")
    print(f"    logistic_regression: {report.config['logistic_regression']}")
    if report.warnings:
        print()
        print("  warnings:")
        for warning in report.warnings:
            print(f"    ! {warning}")


# ---------------------------------------------------------------------------
# 3. Sentiment prediction
# ---------------------------------------------------------------------------
def section_sentiment_prediction(result, classifier) -> None:
    rule("3. SENTIMENT PREDICTION  (one class, or an abstention)")
    for prediction in result.predictions:
        print()
        preview = prediction.text
        if len(preview) > 62:
            preview = preview[:59] + "..."
        tag = f"{prediction.script}/{prediction.language}"
        tag += ", code-mixed" if prediction.is_code_mixed else ""
        print(f"Document        : {prediction.document_id}  {tag}")
        print(f"  preprocessed  : {preview}")
        print(f"  original_text : {prediction.original_text}")
        if prediction.abstained:
            print(f"  sentiment     : (ABSTAINED - winner {prediction.top_sentiment()!r} "
                  f"below the {result.threshold} floor)")
        else:
            print(f"  sentiment     : {prediction.sentiment}")
        print(f"  confidence    : {prediction.confidence:.4f}")
        ranked = ", ".join(
            f"{k}={v:.3f}"
            for k, v in sorted(prediction.sentiment_probabilities.items(), key=lambda kv: -kv[1])
        )
        print(f"  distribution  : {ranked}  (sums to 1: single-label)")
        print(f"  topic context : id={prediction.topic_id} label={prediction.topic_label!r} "
              f"keywords={prediction.topic_keywords}")
        print(f"  issue context : {prediction.issue_labels or '(none)'}")

    print()
    print("-" * 78)
    print(f"  documents              : {result.n_documents}")
    print(f"  abstained              : {result.n_abstained}")
    print("  an abstention is a real output: the model declines to name a class it")
    print("  is not confident about, instead of guessing 'neutral'.")

    print()
    print("-" * 78)
    print("  same text, different confidence floor (probabilities never change,")
    print("  only the decision moves):")
    document = SentimentEmotionDocument(
        document_id="demo",
        text="placement pressure hai aur hostel ka stress bhi hai",
    )
    for floor in (0.5, 0.4, 0.3):
        scored = classifier.predict([document], threshold=floor).predictions[0]
        print(f"    floor={floor}: {scored.sentiment or '(abstained)'}  "
              f"winner={scored.top_sentiment()}  p={scored.confidence:.4f}")


# ---------------------------------------------------------------------------
# 4. Emotion annotations
# ---------------------------------------------------------------------------
def section_emotion_annotations(annotations: list) -> None:
    rule("4. EMOTION ANNOTATIONS  (synthetic, hand-written, multi-label)")
    corpus = EmotionCorpus(annotations=annotations)
    print()
    print(f"  annotations           : {len(corpus)}")
    print(f"  labels learned        : {len(corpus.labels)}  {corpus.labels}")
    print(f"  multi-label documents : {sum(1 for a in corpus if a.is_multi_label)}")
    print(f"  documents with none   : {sum(1 for a in corpus if not a.emotions)}")
    print()
    print("  positive examples per label:")
    for label, count in corpus.label_counts().items():
        print(f"    {label:<20} {count}")
    print()
    print("  a few records:")
    for annotation in annotations[:2] + annotations[20:22] + annotations[22:25]:
        print(f"    {annotation.document_id}  emotions={annotation.emotions or '[] (none)'}")
        print(f"      text: {annotation.text}")


# ---------------------------------------------------------------------------
# 5. Emotion training
# ---------------------------------------------------------------------------
def section_emotion_training(report) -> None:
    rule("5. EMOTION TRAINING  (label set LEARNED from the annotations)")
    print()
    print(f"  model              : {report.model_name} {report.model_version}")
    print(f"  training documents : {report.n_training_documents}")
    print(f"  labels learned     : {report.n_labels}  {report.labels}")
    print(f"  distinct features  : {report.n_features}")
    print(f"  per-label boundary : {report.threshold}")
    print(f"  multi-label docs   : {report.n_multi_label_documents}")
    print(f"  docs with no emotion: {report.n_documents_without_emotions}")
    print()
    print("  positive examples per label (the data-adequacy evidence):")
    for label, count in report.positive_examples_per_label.items():
        print(f"    {label:<20} {count}")
    if report.warnings:
        print()
        print("  warnings:")
        for warning in report.warnings:
            print(f"    ! {warning}")


# ---------------------------------------------------------------------------
# 6. Emotion prediction
# ---------------------------------------------------------------------------
def section_emotion_prediction(result) -> None:
    rule("6. EMOTION PREDICTION  (independent sigmoids; several may fire)")
    for prediction in result.predictions:
        print()
        preview = prediction.text
        if len(preview) > 62:
            preview = preview[:59] + "..."
        tag = f"{prediction.script}/{prediction.language}"
        tag += ", code-mixed" if prediction.is_code_mixed else ""
        print(f"Document        : {prediction.document_id}  {tag}")
        print(f"  preprocessed  : {preview}")
        print(f"  emotions      : {prediction.emotions or '(none above the boundary)'}")
        ranked = ", ".join(
            f"{k}={v:.3f}"
            for k, v in sorted(prediction.emotion_probabilities.items(), key=lambda kv: -kv[1])
        )
        total = sum(prediction.emotion_probabilities.values())
        print(f"  sigmoids      : {ranked}")
        print(f"  sum           : {total:.4f}  (NOT constrained to 1 - these are")
        print("                   independent One-vs-Rest heads, not a softmax)")
        print(f"  topic context : id={prediction.topic_id} keywords={prediction.topic_keywords}")
        print(f"  issue context : {prediction.issue_labels or '(none)'}")

    print()
    print("-" * 78)
    print(f"  documents              : {result.n_documents}")
    print(f"  with no emotion above  : {result.n_documents_without_emotion}")
    print(f"  multi-emotion documents: {result.n_multi_label_documents}")
    print(f"  emotion frequency      : {result.emotion_frequency()}")


# ---------------------------------------------------------------------------
# 7. Dimension separation
# ---------------------------------------------------------------------------
def section_dimension_separation(sentiment_model, emotion_model, documents) -> None:
    rule("7. DIMENSION SEPARATION  (emotion is never inferred from sentiment)")
    print()
    print("  7a. The two models are separately trained and separately configured.")
    print(f"      sentiment: {sentiment_model.name} {sentiment_model.version} "
          f"threshold={sentiment_model.threshold} (a confidence floor)")
    print(f"      emotion  : {emotion_model.name} {emotion_model.version} "
          f"threshold={emotion_model.threshold} (a per-label boundary)")
    print(f"      a sentiment class is not an emotion label: "
          f"{sorted(sentiment_model.labels)} vs {sorted(emotion_model.labels)}")
    print(f"      overlap: {sorted(set(sentiment_model.labels) & set(emotion_model.labels))}")

    print()
    print("  7b. Same text, both dimensions, side by side.")
    document = SentimentEmotionDocument(
        document_id="x",
        text="placement pressure hai aur hostel ka stress bhi hai",
        language="hi",
        script="latin",
        is_code_mixed=True,
    )
    sentiment_prediction = sentiment_model.predict([document]).predictions[0]
    emotion_prediction = emotion_model.predict([document], threshold=0.15).predictions[0]
    print(f"      sentiment : {sentiment_prediction.sentiment or '(abstained)'}"
          f"  winner={sentiment_prediction.top_sentiment()}"
          f"  p={sentiment_prediction.confidence:.4f}")
    print(f"      emotion   : {emotion_prediction.emotions}")
    print("      the emotion vector was produced by the emotion model alone; nothing")
    print("      in the sentiment result feeds it, and vice versa.")

    print()
    print("  7c. The emotion label space comes from the annotation corpus, not from")
    print("      the code. A second corpus with a completely different taxonomy")
    print("      produces a different label space for the SAME text, which is")
    print("      impossible if emotion were being copied from sentiment:")
    text = "this is really not good at all"
    probe = SentimentEmotionDocument(document_id="probe", text=text)
    sentiment_verdict = sentiment_model.predict([probe]).predictions[0].sentiment
    print(f"      sentiment model says        : {sentiment_verdict or '(abstained)'}")

    alternate = TfidfLogisticEmotionClassifier()
    alternate.train(to_emotion_eval(ALTERNATE_EMOTION_TRAINING))
    print(f"      emotion corpus A labels     : {emotion_model.labels}")
    print(f"      emotion corpus B labels     : {alternate.labels}")
    print(f"      corpus A probability keys   : "
          f"{sorted(emotion_model.predict([probe]).predictions[0].emotion_probabilities)}")
    print(f"      corpus B probability keys   : "
          f"{sorted(alternate.predict([probe]).predictions[0].emotion_probabilities)}")
    print("      neither model can emit a label its annotations never contained, so")
    print("      a taxonomy change between annotation rounds is visible rather than")
    print("      silently absorbed by a hard-coded list.")

    print()
    print("  7d. A positive document is not required to carry a positive emotion, and")
    print("      a document with no emotion at all is a valid outcome:")
    rows = []
    for document in documents:
        s = sentiment_model.predict([document]).predictions[0]
        e = emotion_model.predict([document], threshold=0.3).predictions[0]
        rows.append((document.document_id, s.sentiment or "abstain", e.emotions or []))
    for doc_id, sent, emotions in rows:
        print(f"      {doc_id:<4} sentiment={sent:<9} emotions={emotions or '[]'}")

    print()
    print("  7e. Input objects are not mutated, and the echoed text is the")
    print("      preprocessed text with its placeholders intact:")
    document = SentimentEmotionDocument(
        document_id="emoji",
        text="fees <EMOJI_loudly_crying_face> ROI <EMOJI_fire>",
        original_text="Fees 💸 ROI 🔥",
    )
    before = document.model_dump()
    prediction = emotion_model.predict([document]).predictions[0]
    print(f"      input  text: {document.text}")
    print(f"      output text: {prediction.text}")
    print(f"      output original_text: {prediction.original_text}")
    print(f"      input unchanged: {document.model_dump() == before}")


# ---------------------------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------------------------
def section_evaluation(sentiment_report, emotion_report) -> None:
    rule("8. EVALUATION  (held out from training: ILLUSTRATIVE, NOT PERFORMANCE)")
    print()
    print("  ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE")
    print("  12 synthetic held-out records per dimension. The corpora are ~30")
    print("  hand-written sentences, so these numbers measure the metric code and")
    print("  the memorisation of a tiny corpus. They are not an estimate of how")
    print("  this model behaves on real campus discussion.")
    print()
    print(f"  Sentiment evaluated at its default confidence floor ({sentiment_report.n_documents} docs).")
    print(f"  Emotion evaluated at threshold 0.30, not the 0.50 default: on a corpus this")
    print("  small the default boundary leaves most documents with no emotion at all,")
    print("  and the operating point is a decision the project must make explicitly")
    print("  on real annotated data rather than inherit from a default.")
    print()
    print("-" * 78)
    print("  SENTIMENT (single-label, so accuracy is meaningful here)")
    print("-" * 78)
    print()
    print(sentiment_report.format_summary())
    print()
    print("-" * 78)
    print("  EMOTION (multi-label, so micro/macro, subset accuracy and Hamming)")
    print("-" * 78)
    print()
    print(emotion_report.format_summary())


# ---------------------------------------------------------------------------
# 9. Data-adequacy guards
# ---------------------------------------------------------------------------
def section_guards() -> None:
    rule("9. DATA ADEQUACY GUARDS  (withhold metrics, refuse bad training data)")
    classifier = TfidfLogisticSentimentClassifier()
    classifier.train(to_sentiment_annotations(SENTIMENT_TRAINING))

    print()
    print("  9a. Too few evaluation documents -> metrics withheld, not invented:")
    tiny = to_sentiment_eval(SENTIMENT_EVAL[:4])
    print("    " + classifier.evaluate(tiny).format_summary().replace("\n", "\n    "))

    print()
    print("  9b. No class diversity -> withheld, with the reason stated:")
    single = to_sentiment_eval(
        [(f"u{i}", "the workshop is on monday", "neutral") for i in range(12)]
    )
    print("    " + classifier.evaluate(single).format_summary().replace("\n", "\n    "))

    print()
    print("  9c. Scoring the training data against itself is flagged loudly:")
    self_scored = classifier.evaluate(to_sentiment_annotations(SENTIMENT_TRAINING))
    for warning in self_scored.warnings:
        if "held-out" in warning:
            print(f"    ! {warning}")

    print()
    print("  9d. train() refuses corpora it cannot honestly learn from:")
    cases = [
        ("empty corpus", []),
        ("one document per class", to_sentiment_eval(
            [("a", "good", "positive"), ("b", "bad", "negative"), ("c", "flat", "neutral")]
        )),
        ("missing labels", to_sentiment_eval(
            [(f"b{i}", "some text here", "positive") for i in range(6)]
        )),
    ]
    for name, annotations in cases:
        try:
            TfidfLogisticSentimentClassifier().train(annotations)
            print(f"    {name:<24} -> NOT REFUSED (this is a bug)")
        except ValueError as exc:
            print(f"    {name:<24} -> ValueError: {exc}")

    print()
    print("  9e. Emotion refuses a corpus where one emotion is in every document,")
    print("      because a per-label model fitted on a constant column learns")
    print("      nothing and would report a confident constant:")
    all_positive = to_emotion_eval(
        [(f"c{i}", f"this is stressful number {i}", ["stress", "frustration"])
         for i in range(6)]
    )
    try:
        TfidfLogisticEmotionClassifier().train(all_positive)
        print("    all-positive labels           -> NOT REFUSED (this is a bug)")
    except ValueError as exc:
        print(f"    all-positive labels           -> ValueError: {exc}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    sentiment_annotations = to_sentiment_annotations(SENTIMENT_TRAINING)
    emotion_annotations = to_emotion_annotations(EMOTION_TRAINING)
    documents = to_documents(PREDICTION_DOCUMENTS)

    print("=" * 78)
    print("SENTIMENT & EMOTION  |  two independent TF-IDF + Logistic Regression models")
    print("IIM Sambalpur Community Intelligence")
    print("=" * 78)
    print()
    print("*** ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE ***")
    print()

    # --- sentiment ---
    section_sentiment_annotations(sentiment_annotations)
    sentiment_model = TfidfLogisticSentimentClassifier(
        config=SentimentModelConfig(C=4.0, max_iter=1000, min_df=1, ngram_range=(1, 2))
    )
    sentiment_training = sentiment_model.train(sentiment_annotations)
    section_sentiment_training(sentiment_training)
    sentiment_result = sentiment_model.predict(documents)
    section_sentiment_prediction(sentiment_result, sentiment_model)

    # --- emotion ---
    section_emotion_annotations(emotion_annotations)
    emotion_model = TfidfLogisticEmotionClassifier(
        config=EmotionModelConfig(C=4.0, max_iter=1000, min_df=1, ngram_range=(1, 2))
    )
    emotion_training = emotion_model.train(emotion_annotations)
    section_emotion_training(emotion_training)
    emotion_result = emotion_model.predict(documents, threshold=0.3)
    section_emotion_prediction(emotion_result)

    # --- separation ---
    section_dimension_separation(sentiment_model, emotion_model, documents)

    # --- evaluation ---
    sentiment_report = sentiment_model.evaluate(to_sentiment_eval(SENTIMENT_EVAL))
    emotion_report = emotion_model.evaluate(
        to_emotion_eval(EMOTION_EVAL), threshold=0.30
    )
    section_evaluation(sentiment_report, emotion_report)

    # --- guards ---
    section_guards()

    # --- determinism, visible in the output ---
    print()
    print("-" * 78)
    print("  DETERMINISM CHECK (random_state=42, lbfgs):")
    repeat_sentiment = TfidfLogisticSentimentClassifier()
    repeat_sentiment.train(sentiment_annotations)
    repeat_emotion = TfidfLogisticEmotionClassifier()
    repeat_emotion.train(emotion_annotations)
    sentiment_same = [
        p.sentiment_probabilities for p in sentiment_result.predictions
    ] == [
        p.sentiment_probabilities
        for p in repeat_sentiment.predict(documents).predictions
    ]
    emotion_same = [
        p.emotion_probabilities for p in emotion_result.predictions
    ] == [
        p.emotion_probabilities
        for p in repeat_emotion.predict(documents, threshold=0.3).predictions
    ]
    print(f"    sentiment identical on retrain : {sentiment_same}")
    print(f"    emotion identical on retrain   : {emotion_same}")


if __name__ == "__main__":
    main()
