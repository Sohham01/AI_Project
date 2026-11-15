"""
Demo script: multi-label ISSUE IDENTIFICATION on synthetic IIM Sambalpur
Community Intelligence data.

=============================================================================
ALL DATA IN THIS DEMO IS SYNTHETIC AND ILLUSTRATIVE.
It was written by hand to exercise the architecture. It was NOT collected from
Reddit, it was NOT annotated by real annotators, and the metrics printed below
are NOT project performance. They only demonstrate that the plumbing works.
Real numbers require the annotation workflow documented in
docs/issue_identification.md.
=============================================================================

The four stages below are deliberately separated:

    1. ANNOTATION EXAMPLES   synthetic IssueAnnotation records
    2. TRAINING              label set learned from those annotations
    3. PREDICTION            multi-label output on unseen synthetic posts
    4. EVALUATION            a sanity check of the metrics plumbing, held
                             out from training so it is at least not
                             self-scored

Topic discovery is upstream and separate: topics say WHAT is discussed,
issues say WHAT PROBLEM is expressed. This demo consumes the topic id that
topic discovery would have produced, purely as optional context.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Windows consoles default to cp1252 and cannot render Devanagari / Bengali /
# Odia.  Force UTF-8 so the multilingual output is readable.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.nlp.issue_identification import (
    AnnotationCorpus,
    IssueAnnotation,
    IssueDocument,
    IssueModelConfig,
    TfidfLogisticIssueClassifier,
)


# ---------------------------------------------------------------------------
# 1. ANNOTATION EXAMPLES
#
# (document_id, preprocessed text, labels, language, script, code-mixed)
# `text` is what the preprocessing stage would have produced: lowercase where
# appropriate, punctuation preserved, emoji already turned into
# <EMOJI_...> tokens.  Labels are multi-label and language-neutral.
# ---------------------------------------------------------------------------
TRAINING_ANNOTATIONS = [
    # --- fees ---
    ("a01", "the fees are way too high and unaffordable for most students", ["fees"], "en", "latin", False),
    ("a02", "fee structure is ridiculous and the tuition fee hike is unacceptable", ["fees"], "en", "latin", False),
    ("a03", "fees ka bhaari hai aur scholarship se bach bhi nahi milta", ["fees"], "hi", "latin", True),
    ("a04", "IIM fees badhte badhte bahut zyada ho gaye hain", ["fees"], "hi", "latin", True),
    ("a05", "the annual fee is too expensive compared to other institutes", ["fees"], "en", "latin", False),
    ("a06", "fee is unreasonable and the hostel mess charges are also high", ["fees", "mess_food"], "en", "latin", False),
    # --- roi ---
    ("a07", "the ROI of this MBA is not worth it at all", ["roi"], "en", "latin", False),
    ("a08", "return on investment is poor for an MBA programme", ["roi"], "en", "latin", False),
    ("a09", "ROI questionable hai MBA ka return justify nahi karta", ["roi"], "hi", "latin", True),
    ("a10", "the return on investment is questionable given the fees", ["roi", "fees"], "en", "latin", False),
    # --- placement_stress ---
    ("a11", "placement season is extremely stressful for students", ["placement_stress"], "en", "latin", False),
    ("a12", "placement season pressure and PPO uncertainty is stressful", ["placement_stress"], "en", "latin", False),
    ("a13", "placement season bahut stressful hai PPO ka tension", ["placement_stress"], "hi", "latin", True),
    ("a14", "campus placement stress is really high every year", ["placement_stress"], "en", "latin", False),
    # --- hostel ---
    ("a15", "hostel rooms are too small and cramped for two students", ["hostel"], "en", "latin", False),
    ("a16", "hostel room maintenance is poor and the room is very small", ["hostel"], "en", "latin", False),
    ("a17", "hostel room chhota hai aur maintenance bahut kharab", ["hostel"], "hi", "latin", True),
    ("a18", "the hostel room size and maintenance are both problems", ["hostel", "mess_food"], "en", "latin", False),
    # --- mess_food ---
    ("a19", "mess food quality is really bad every single day", ["mess_food"], "en", "latin", False),
    ("a20", "the hostel mess serves very bad and stale food", ["mess_food"], "en", "latin", False),
    ("a21", "mess ka khana kharab hai quality bahut kharab", ["mess_food"], "hi", "latin", True),
    ("a22", "mess food is unhealthy and the cooking quality is poor", ["mess_food"], "en", "latin", False),
    # --- exam_pressure ---
    ("a23", "CAT exam pressure is immense and preparation is stressful", ["exam_pressure"], "en", "latin", False),
    ("a24", "CAT exam preparation pressure is really high for students", ["exam_pressure"], "en", "latin", False),
    ("a25", "CAT exam ki preparation pressure bahut zyada hai", ["exam_pressure"], "hi", "latin", True),
    # --- native script, same shared label space ---
    ("a26", "ফি অনেক বেশি এবং ROI কোনো কাজে আসে না", ["fees", "roi"], "bn", "bengali", False),
    ("a27", "ছাত্রদের জন্য প্লেসমেন্ট চাপ অত্যন্ত বেশি", ["placement_stress"], "bn", "bengali", False),
    ("a28", "ଫି ବହୁତ ଅଧିକ ଏବଂ ROI ଉପଯୋଗୀ ନାହିଁ", ["fees", "roi"], "or", "odia", False),
    # --- emoji placeholder tokens produced by the preprocessing stage ---
    ("a29", "fees <EMOJI_loudly_crying_face> ROI <EMOJI_fire> placement stress <EMOJI_pensive_face>", ["fees", "roi", "placement_stress"], "en", "latin", False),
    # --- negatives: no issue expressed at all ---
    ("a30", "campus fest was amazing and the food was great", [], "en", "latin", False),
    ("a31", "IIM Sambalpur library has a great collection of books", [], "en", "latin", False),
    ("a32", "the new academic building is beautiful and spacious", [], "en", "latin", False),
    ("a33", "maine naya laptop liya aur course bhi achha hai", [], "hi", "latin", True),
]

# Unseen posts for prediction. topic_id is what topic discovery would supply.
PREDICTION_POSTS = [
    ("p1", "fees are way too high and ROI feels questionable", "en", "latin", False, 0, "Topic 0"),
    ("p2", "hostel rooms are too small and mess food is bad", "en", "latin", False, 3, "Topic 3"),
    ("p3", "placement season is extremely stressful", "en", "latin", False, 1, "Topic 1"),
    ("p4", "CAT exam pressure is immense and preparation is stressful", "en", "latin", False, 2, "Topic 2"),
    ("p5", "bhai mess ka khana kharab hai aur fees bhi zyada hai", "hi", "latin", True, 3, "Topic 3"),
    ("p6", "fees <EMOJI_loudly_crying_face> ROI <EMOJI_fire> placement stress <EMOJI_pensive_face>", "en", "latin", False, 0, "Topic 0"),
    ("p7", "campus fest was amazing and the food was great", "en", "latin", False, 4, "Topic 4"),
    ("p8", "the hostel mess food is terrible", "en", "latin", False, 3, "Topic 3"),
]

# Held out from TRAINING_ANNOTATIONS purely so the evaluation section is not
# scored on its own training data.
EVALUATION_ANNOTATIONS = [
    IssueAnnotation(document_id="e1", text="the hostel room is far too small", labels=["hostel"]),
    IssueAnnotation(document_id="e2", text="mess food quality is terrible", labels=["mess_food"]),
    IssueAnnotation(document_id="e3", text="fees are expensive and the ROI is poor", labels=["fees", "roi"]),
    IssueAnnotation(document_id="e4", text="placement season is stressful", labels=["placement_stress"]),
    IssueAnnotation(document_id="e5", text="CAT exam pressure is high", labels=["exam_pressure"]),
    IssueAnnotation(document_id="e6", text="the fee hike and mess food are both bad", labels=["fees", "mess_food"]),
    IssueAnnotation(document_id="e7", text="return on investment is not worth it", labels=["roi"]),
    IssueAnnotation(document_id="e8", text="hostel room maintenance is poor", labels=["hostel"]),
    IssueAnnotation(document_id="e9", text="the campus library is a great place", labels=[]),
    IssueAnnotation(document_id="e10", text="mess food is unhealthy and expensive", labels=["mess_food", "fees"]),
    IssueAnnotation(document_id="e11", text="PPO uncertainty makes placement stressful", labels=["placement_stress"]),
    IssueAnnotation(document_id="e12", text="the annual fee is too high", labels=["fees"]),
]


def to_annotations(rows) -> list:
    return [
        IssueAnnotation(
            document_id=doc_id,
            text=text,
            labels=list(labels),
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
            detected_languages=[language, "en"] if is_code_mixed else [language],
        )
        for doc_id, text, labels, language, script, is_code_mixed in rows
    ]


def print_annotations(annotations: list) -> None:
    print("=" * 78)
    print("1. ANNOTATION EXAMPLES  (synthetic, hand-written)")
    print("=" * 78)
    print()
    corpus = AnnotationCorpus(annotations=annotations)
    print(f"  annotations        : {len(corpus)}")
    print(f"  distinct labels    : {len(corpus.labels)}  {corpus.labels}")
    print(f"  multi-label docs   : {sum(1 for a in corpus if a.is_multi_label)}")
    print(f"  docs with no issue : {sum(1 for a in corpus if not a.labels)}")
    print()
    print("  per-label positive examples:")
    for label, count in corpus.label_counts().items():
        print(f"    {label:<20} {count}")
    print()
    print("  a few records:")
    for annotation in annotations[:4] + annotations[5:8]:
        print(f"    {annotation.document_id}  labels={annotation.labels}")
        print(f"      text: {annotation.text}")


def print_training(report) -> None:
    print()
    print("=" * 78)
    print("2. TRAINING  (label set LEARNED from the annotations, not hard-coded)")
    print("=" * 78)
    print()
    print(f"  model              : {report.model_name} {report.model_version}")
    print(f"  training documents : {report.n_training_documents}")
    print(f"  labels learned     : {report.n_labels}  {report.labels}")
    print(f"  distinct features  : {report.n_features}")
    print(f"  threshold          : {report.threshold}")
    print(f"  multi-label docs   : {report.n_multi_label_documents}")
    print(f"  docs without issue : {report.n_documents_without_labels}")
    print()
    print("  positive examples per label (the data-adequacy evidence):")
    for label, count in report.positive_examples_per_label.items():
        print(f"    {label:<20} {count}")
    if report.warnings:
        print()
        print("  warnings:")
        for warning in report.warnings:
            print(f"    ! {warning}")


def print_predictions(result) -> None:
    print()
    print("=" * 78)
    print("3. PREDICTION  (multi-label output on unseen synthetic posts)")
    print("=" * 78)
    for prediction in result.predictions:
        tags = f"[{prediction.script}/{prediction.language}"
        tags += ", code-mixed" if prediction.is_code_mixed else ""
        tags += f", {prediction.topic_label}]"
        preview = prediction.text
        if len(preview) > 60:
            preview = preview[:57] + "..."
        print()
        print(f"Document        : {prediction.document_id}  {tags}")
        print(f"  text          : {preview}")
        print(f"  original_text : {prediction.original_text}")
        print(f"Predicted Issues: {prediction.predicted_issue_labels or '(none)'}")
        print(f"Confidence      : {prediction.confidence:.4f}")
        ranked = sorted(prediction.issue_probabilities.items(), key=lambda kv: -kv[1])
        print("Probabilities    : " + ", ".join(f"{k}={v:.3f}" for k, v in ranked))

    print()
    print("-" * 78)
    print(f"  documents with an issue : {result.n_documents - result.n_documents_without_issue}"
          f" / {result.n_documents}")
    print(f"  multi-label documents   : {result.n_multi_label_documents}")
    print("  label frequency         : "
          + (str(result.label_frequency()) if result.label_frequency() else "{}"))


def print_threshold_effect(classifier) -> None:
    print()
    print("-" * 78)
    print("  same documents, different threshold (sigmoid scores are unchanged,")
    print("  only the decision boundary moves):")
    text = "hostel rooms are too small and mess food is bad"
    for threshold in (0.5, 0.4, 0.3):
        result = classifier.predict(
            [IssueDocument(document_id="demo", text=text)], threshold=threshold
        )
        prediction = result.predictions[0]
        ranked = ", ".join(
            f"{k}={v:.3f}" for k, v in
            sorted(prediction.issue_probabilities.items(), key=lambda kv: -kv[1])
        )
        print(f"    threshold={threshold}: {prediction.predicted_issue_labels}")
        print(f"      probabilities: {ranked}")


def print_evaluation(report) -> None:
    print()
    print("=" * 78)
    print("4. EVALUATION  (sanity check of the metrics plumbing only)")
    print("=" * 78)
    print()
    print("  REMINDER: evaluated on 12 SYNTHETIC held-out records. These numbers")
    print("  show the metric code path works. They are NOT project performance and")
    print("  must not be quoted as such.")
    print()
    print(report.format_summary())


def print_insufficient_data_note(classifier) -> None:
    print()
    print("-" * 78)
    print("  When the data is too small, metrics are WITHHELD rather than invented:")
    tiny = [
        IssueAnnotation(document_id="t1", text="fees are high", labels=["fees"]),
        IssueAnnotation(document_id="t2", text="roi is poor", labels=["roi"]),
        IssueAnnotation(document_id="t3", text="hostel is small", labels=["hostel"]),
    ]
    print(classifier.evaluate(tiny).format_summary())
    print()
    print("  And train() refuses data it cannot honestly learn from:")
    try:
        TfidfLogisticIssueClassifier().train(tiny)
    except ValueError as exc:
        print(f"    ValueError: {exc}")


def main() -> None:
    annotations = to_annotations(TRAINING_ANNOTATIONS)
    documents = [
        IssueDocument(
            document_id=doc_id,
            text=text,
            original_text=text,
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
            detected_languages=[language, "en"] if is_code_mixed else [language],
            topic_id=topic_id,
            topic_label=topic_label,
        )
        for doc_id, text, language, script, is_code_mixed, topic_id, topic_label
        in PREDICTION_POSTS
    ]

    print("=" * 78)
    print("ISSUE IDENTIFICATION  |  TF-IDF -> One-vs-Rest Logistic Regression")
    print("IIM Sambalpur Community Intelligence")
    print("=" * 78)
    print()
    print("*** ALL DATA BELOW IS SYNTHETIC AND ILLUSTRATIVE - NOT PROJECT DATA ***")
    print()

    print_annotations(annotations)

    classifier = TfidfLogisticIssueClassifier(
        config=IssueModelConfig(C=4.0, max_iter=1000, min_df=1, ngram_range=(1, 2))
    )
    report = classifier.train(annotations)
    print_training(report)

    # The taxonomy a downstream consumer would expect. Anything here that the
    # model never saw in training is reported as unseen, not silently dropped.
    taxonomy = report.labels + ["wifi", "exam_pressure_v2"]
    result = classifier.predict(documents, candidate_labels=taxonomy)
    print_predictions(result)

    print()
    print(f"  known labels    : {result.known_labels}")
    print(f"  unseen labels   : {result.unseen_labels}")
    print("  (labels absent from the annotations cannot be predicted - that is")
    print("   how taxonomy drift between annotation rounds becomes visible)")

    print_threshold_effect(classifier)
    print_evaluation(classifier.evaluate(EVALUATION_ANNOTATIONS))
    print_insufficient_data_note(classifier)

    # Determinism check, visible in the demo output.
    repeat = TfidfLogisticIssueClassifier()
    repeat.train(annotations)
    repeat_result = repeat.predict(documents)
    same = [p.issue_probabilities for p in result.predictions] == [
        p.issue_probabilities for p in repeat_result.predictions
    ]
    print()
    print(f"DETERMINISM CHECK: independent retrain produced identical "
          f"probabilities -> {same}")


if __name__ == "__main__":
    main()
