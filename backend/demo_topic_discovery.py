"""
Demo script: automatic topic discovery over a synthetic IIM Sambalpur
Community Intelligence corpus.

Pipeline shown:
    raw text
      -> LanguageAnalysisService   (language / script / code-mix)
      -> MultilingualTextPreprocessor  (deterministic normalisation)
      -> TopicDocument (already-preprocessed text + metadata)
      -> NMFTopicDiscoverer  (TF-IDF -> NMF -> keywords -> assignments)

The topic model consumes preprocessed text only; it never re-runs
preprocessing and never modifies raw text.

Topic categories are NOT hard-coded.  Nothing in the model or this script
knows that "placement" or "hostel" are IIM discussion areas — the model
discovers them from term co-occurrence and labels them "Topic <id>".
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Windows consoles default to cp1252 and cannot render Devanagari / Bengali /
# Odia.  Force UTF-8 so the multilingual output is readable.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.nlp.language import SimpleLanguageDetector
from app.services.nlp.script_detector import SimpleScriptDetector
from app.services.nlp.code_mix_detector import SimpleCodeMixDetector
from app.services.nlp.language_analysis_service import LanguageAnalysisService
from app.services.nlp.preprocessing import MultilingualTextPreprocessor
from app.services.nlp.topic_discovery import (
    NMFTopicDiscoverer,
    TFIDFConfig,
    TopicDocument,
)


# 24 representative synthetic documents spanning the discussion themes that
# actually show up in IIM subreddits, written in English, Hinglish, native
# script, with emoji, shouting, typos and code-switching.
SAMPLE_POSTS = [
    # --- placements / PPO ---
    "IIM Sambalpur placement season was very competitive and the PPO process was smooth for most students",
    "bhai placement ka kya scene hai PPO mila ya nhi confusion",
    "campus placements PPO selection process is quite unpredictable every single year",
    "placement aur PPO dono ke liye interview rounds bahut tough the",
    "🔥 PPO mil gaya finally! placement season complete",
    # --- CAT exam ---
    "CAT exam preparation needs serious mock tests and good coaching for the MBA admission cycle",
    "CAT exam di taiyari aur coaching MBA admission ke liye kitni zaroori hai",
    "mock tests for the CAT exam are the best way to prepare for MBA admission",
    "CAT pe mood off \U0001f625 coaching se thoda relief mila",
    # --- hostel / mess ---
    "hostel mess food quality is really bad and the mess charges are too high every month",
    "hostel ka mess aur food quality kharab hai mess charges bahut zyada hai",
    "mess food quality hostel charges should be reduced by the administration",
    "हॉस्टल का मेस खाना अच्छा नहीं है",
    # --- fees / scholarship / ROI ---
    "the fees are very high and the ROI is not worth it compared to the MBA return",
    "fees scholarship aur ROI ka kya scene hai MBA ka return justify karta hai ya nhi",
    "scholarship fees and ROI should be discussed openly with the students",
    # --- internships ---
    "internship opportunities during the summer are limited but the internship projects are good",
    "summer internship ke liye company kaise select kare internship projects mast hote hai",
    "internship MBA ke saath karna chahiye internship experience bahut important hai",
    # --- native script ---
    "आईआईएम संबলपुर में प्लेसमेंट बहुत अच्छा है",
    "क्या IIM संबलपुर अच्छा है",
    "ক্যাট পরীক্ষার প্রস্তুতি খুব কঠিন",
    "প্লেসমেন্ট প্রক্রিয়া ভালো হয়েছে",
    "ହସ୍ଟେଲ ଖାବା ଭଲ ନାହିଁ",
    # --- informal / emoji heavy ---
    "broooo \U0001f62d placement ka kya scene hai??? \U0001f525",
    "IIM Sambalpur \U0001f44d hostel life mast hai but mess food \U0001f626",
    "CAT exam MBA hostel mess placement internship fees all together bahut confusion",
    "😩😩 IIM Sambalpur placement internship mess hostel fees ROI CAT MBA",
]


def build_documents() -> list:
    """
    Run the upstream language-analysis and preprocessing stages and hand the
    topic model nothing but preprocessed text plus traceability metadata.
    """
    analysis = LanguageAnalysisService(
        language_detector=SimpleLanguageDetector(),
        script_detector=SimpleScriptDetector(),
        code_mix_detector=SimpleCodeMixDetector(),
    )
    preprocessor = MultilingualTextPreprocessor()

    documents = []
    for index, raw in enumerate(SAMPLE_POSTS):
        lang = analysis.analyze(raw)
        pre = preprocessor.preprocess(
            text=raw,
            language=lang.language,
            script=lang.script,
            is_code_mixed=lang.is_code_mixed,
            detected_languages=lang.detected_languages,
        )
        documents.append(
            TopicDocument(
                document_id=f"post-{index:03d}",
                text=pre.normalized_text,          # preprocessed, as required
                post_id=f"post-{index:03d}",
                original_text=pre.original_text,    # raw text kept for traceability
                language=lang.language,
                script=lang.script,
                is_code_mixed=lang.is_code_mixed,
                detected_languages=lang.detected_languages,
            )
        )
    return documents


def print_documents(result) -> None:
    print("=" * 78)
    print("PER-DOCUMENT TOPIC ASSIGNMENTS")
    print("=" * 78)

    for assignment in result.documents:
        topic = result.topic_by_id(assignment.dominant_topic_id)
        preview = assignment.text
        if len(preview) > 68:
            preview = preview[:65] + "..."

        print()
        print(f"Document        : {assignment.document_id}  "
              f"[{assignment.script}/{assignment.language}"
              f"{', code-mixed' if assignment.is_code_mixed else ''}]")
        print(f"  text          : {preview}")
        print(f"  original_text : {assignment.original_text}")
        print(f"Dominant Topic  : {assignment.dominant_topic_label}  (topic_id="
              f"{assignment.dominant_topic_id})")
        print(f"Topic Keywords  : {', '.join(topic.keywords[:8])}")
        print(f"Topic Score     : {assignment.dominant_topic_score:.4f}")
        distribution = ", ".join(
            f"{tid}={weight:.3f}"
            for tid, weight in sorted(
                assignment.topic_distribution.items(), key=lambda kv: -kv[1]
            )
        )
        print(f"Full Distribution: {distribution}")


def print_summary(result) -> None:
    print()
    print("=" * 78)
    print("SUMMARY OF DISCOVERED TOPICS")
    print("=" * 78)
    print()
    print(f"{'Topic':<10}{'Docs':>6}{'Share':>8}   Keywords")
    print("-" * 78)
    for topic in sorted(result.topics, key=lambda t: (-t.document_count, t.topic_id)):
        keywords = ", ".join(topic.keywords[:10])
        print(f"{topic.label:<10}{topic.document_count:>6}{topic.document_ratio:>8.1%}   {keywords}")

    print()
    print("-" * 78)
    print("MODEL METADATA")
    print("-" * 78)
    print(f"  topic_model            : {result.topic_model}")
    print(f"  topic_model_version    : {result.topic_model_version}")
    print(f"  n_topics               : {result.n_topics}")
    print(f"  n_documents            : {result.n_documents}")
    print(f"  vocabulary_size        : {result.vocabulary_size}")
    print(f"  discovered_at          : {result.discovered_at}")
    print(f"  library_versions       : {result.library_versions}")
    print(f"  tfidf config           : {result.config['tfidf']}")
    print(f"  nmf config             : {result.config['nmf']}")
    if result.warnings:
        print("  warnings               :")
        for warning in dict.fromkeys(result.warnings):
            print(f"      - {warning}")


def print_configurable_variant(documents) -> None:
    """
    Show that the stop-word handling is a real knob.

    The default is deliberately conservative (no stop-word removal at all).
    Supplying an explicit, clearly-labelled list of English + Romanized-Indic
    function words removes the 'ka / kya / scene hai' filler topic without the
    model knowing anything about IIM categories.
    """
    filler_function_words = [
        "hai", "ka", "ki", "ke", "ko", "kya", "scene", "bhai", "aur", "ya",
        "nahi", "nahin", "bahut", "accha", "acha", "the", "is", "a", "an",
        "and", "of", "for", "to", "in", "are", "on", "with", "that", "this",
        "se", "me", "mein", "par", "ho", "liye", "kar", "karte", "karke",
        "ka", "ki", "ki", "kar", "kare", "gaya", "kya",
    ]

    print()
    print("=" * 78)
    print("CONFIGURABLE VARIANT: explicit neutral stop-word list")
    print("=" * 78)
    print()
    print("The default configuration applies NO stop-word removal, so Romanized")
    print("function words can form a filler topic.  Passing an explicit list of")
    print("English + Romanized-Indic function words (never applied by default)")
    print("gives sharper topics:")
    print()

    variant = NMFTopicDiscoverer(
        n_topics=6,
        n_keywords=10,
        tfidf_config=TFIDFConfig(stop_words=sorted(set(filler_function_words))),
    ).discover(documents)

    for topic in sorted(variant.topics, key=lambda t: (-t.document_count, t.topic_id)):
        keywords = ", ".join(topic.keywords[:10])
        print(f"{topic.label:<10}{topic.document_count:>6}   {keywords}")

    print()
    print("Note: both runs are deterministic (init='nndsvda', random_state=42).")


def main() -> None:
    documents = build_documents()
    discoverer = NMFTopicDiscoverer(n_topics=6, n_keywords=10)
    result = discoverer.discover(documents)

    print("=" * 78)
    print("TOPIC DISCOVERY  |  TF-IDF -> NMF  |  IIM Sambalpur Community Intelligence")
    print("=" * 78)
    print(f"documents: {len(documents)}   model: {result.topic_model} "
          f"{result.topic_model_version}   topics: {result.n_topics}")
    print()

    print_documents(result)
    print_summary(result)
    print_configurable_variant(documents)

    # Determinism check, visible in the demo output.
    repeat = NMFTopicDiscoverer(n_topics=6, n_keywords=10).discover(documents)
    same = [t.keywords for t in result.topics] == [t.keywords for t in repeat.topics]
    print()
    print(f"DETERMINISM CHECK: repeated run produced identical topics -> {same}")


if __name__ == "__main__":
    main()
