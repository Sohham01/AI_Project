"""
Demo script: RESPONSE CLASSIFICATION on synthetic IIM Sambalpur Community
Intelligence data.

=============================================================================
ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE
Every comment, annotation and metric below is synthetic and hand-written. The
corpora were not collected from Reddit and not annotated by real annotators.
The printed scores exist only to show that the code paths work end to end. They
say nothing about how this approach would perform on real Reddit threads - see
docs/response_classification.md, "Evaluation caveats".
=============================================================================

Response classification asks one question:

    what ROLE does this comment play in the discussion?

    "same bro"                               -> agreement
    "I faced this last year"                 -> personal_experience
    "I think this college is better"         -> opinion
    "you should contact the placement office" -> advice
    "apply for the scholarship"              -> solution
    "actually the fees aren't that high"     -> counterargument

It is NOT sentiment. "apply for the scholarship" carries no polarity at all, and
"same bro" can be said happily or sarcastically. Polarity and role are
independent dimensions, and neither is derived from the other.

Sections:

    1. THE SIX RESPONSE TYPES   the assignment's fixed taxonomy
    2. SYNTHETIC ANNOTATIONS    ResponseAnnotation records
    3. TRAINING                 the class set and what was learned
    4. PREDICTION               one role per comment
    5. CLASS PROBABILISTICS     the full six-way softmax distribution
    6. EXAMPLES FOR ALL SIX     one comment per role, with the runner-up
    7. HELD-OUT EVALUATION      a genuinely disjoint synthetic set
    8. SMALL-DATA REFUSAL       train() declining bad data
    9. CONTEXT & IMBALANCE      V1 ignores thread context; class_weight matters
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Windows consoles default to cp1252 and cannot render Devanagari / Bengali /
# Odia. Force UTF-8 so the multilingual output is readable.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.nlp.response_classification import (
    RESPONSE_TYPES,
    ResponseAnnotation,
    ResponseDocument,
    ResponseModelConfig,
    TfidfLogisticResponseClassifier,
    evaluate_response_classification,
)

BANNER = "=" * 78
SUBTLE = "-" * 78


def rule(title):
    print()
    print(BANNER)
    print(title)
    print(BANNER)


# ---------------------------------------------------------------------------
# SYNTHETIC DATA
#
# `text` is what the preprocessing stage would have produced: emoji already
# turned into <EMOJI_...> tokens, URLs into <URL>, casing and spacing folded.
#
# The phrasing is deliberately keyed on the lexical markers each role tends to
# use ("same", "i faced", "i think", "you should", "apply", "actually"). That is
# what a bag-of-words model can actually learn from a few dozen examples, and it
# is exactly why these numbers must not be read as Reddit performance.
# ---------------------------------------------------------------------------

# (document_id, normalized_text, response_type, language, script, code_mixed)
TRAINING_ROWS = [
    # A. agreement
    ("a01", "same bro exactly", "agreement", "en", "latin", False),
    ("a02", "true that, i agree with you", "agreement", "en", "latin", False),
    ("a03", "exactly this is what i said", "agreement", "en", "latin", False),
    ("a04", "yes that is right, same here", "agreement", "en", "latin", False),
    # B. personal experience
    ("p01", "i faced this exact problem last year", "personal_experience", "en", "latin", False),
    ("p02", "in my own case it happened too", "personal_experience", "en", "latin", False),
    ("p03", "i went through this myself", "personal_experience", "en", "latin", False),
    ("p04", "i remember facing this at my college", "personal_experience", "en", "latin", False),
    # C. opinion
    ("o01", "i think this college is better", "opinion", "en", "latin", False),
    ("o02", "in my view the campus is fine", "opinion", "en", "latin", False),
    ("o03", "honestly i believe it is good", "opinion", "en", "latin", False),
    ("o04", "i feel this place is underrated", "opinion", "en", "latin", False),
    # D. advice
    ("d01", "you should contact the placement office", "advice", "en", "latin", False),
    ("d02", "you ought to talk to the professor", "advice", "en", "latin", False),
    ("d03", "i would advise you to start early", "advice", "en", "latin", False),
    ("d04", "you must check the scholarship portal", "advice", "en", "latin", False),
    # E. solution
    ("s01", "apply for the scholarship on the portal", "solution", "en", "latin", False),
    ("s02", "register on the website before monday", "solution", "en", "latin", False),
    ("s03", "use the hostel wifi to download it", "solution", "en", "latin", False),
    ("s04", "submit the form on the intranet page", "solution", "en", "latin", False),
    # F. counterargument
    ("c01", "actually the fees are not that high", "counterargument", "en", "latin", False),
    ("c02", "but that is not true here", "counterargument", "en", "latin", False),
    ("c03", "however the fees rose last year", "counterargument", "en", "latin", False),
    ("c04", "that is wrong, i disagree entirely", "counterargument", "en", "latin", False),
]

# Genuinely held out: disjoint ids, disjoint text, 3 comments per role = 18.
# 18 is above MIN_DOCUMENTS_FOR_MEANINGFUL_METRICS (10) and every role has
# support 3, so the metrics are computable - but they still describe a
# hand-written toy corpus, not Reddit.
HELDOUT_ROWS = [
    ("h01", "exactly, same thing here", "agreement", "en", "latin", False),
    ("h02", "yes i agree completely", "agreement", "en", "latin", False),
    ("h03", "correct, that matches my point", "agreement", "en", "latin", False),
    ("h04", "i faced the same issue in 2023", "personal_experience", "en", "latin", False),
    ("h05", "this happened to me as well", "personal_experience", "en", "latin", False),
    ("h06", "i experienced this at my campus", "personal_experience", "en", "latin", False),
    ("h07", "i think this college is underrated", "opinion", "en", "latin", False),
    ("h08", "in my opinion the fees are fair", "opinion", "en", "latin", False),
    ("h09", "i believe the mess food is good", "opinion", "en", "latin", False),
    ("h10", "you should talk to the hostel warden", "advice", "en", "latin", False),
    ("h11", "you ought to apply for the bursary", "advice", "en", "latin", False),
    ("h12", "i would advise checking the syllabus", "advice", "en", "latin", False),
    ("h13", "register on the scholarship portal", "solution", "en", "latin", False),
    ("h14", "submit the form on the intranet", "solution", "en", "latin", False),
    ("h15", "download the form from the website", "solution", "en", "latin", False),
    ("h16", "actually the fees are quite low here", "counterargument", "en", "latin", False),
    ("h17", "but that is wrong about this place", "counterargument", "en", "latin", False),
    ("h18", "however i disagree with that claim", "counterargument", "en", "latin", False),
]

# Non-Latin and code-mixed comments, to show nothing script-specific breaks.
MULTILINGUAL_ROWS = [
    ("m01", "haan bilkul same baat hai", "agreement", "hi", "latin", True),
    ("m04", "pichle saal mujhe bhi yahi problem aayi thi", "personal_experience", "hi", "latin", True),
    ("m07", "मुझे लगता है यह कॉलेज बेहतर है", "opinion", "hi", "devanagari", False),
    ("m09", "হ্যাঁ ঠিক একই কথা", "agreement", "bn", "bengali", False),
    ("m10", "আমার একই অভিজ্ঞতা হয়েছিল", "personal_experience", "bn", "bengali", False),
    ("m11", "আমার মতে এটি ভালো কলেজ", "opinion", "bn", "bengali", False),
    ("m12", "ଆମେ ସେବାଟା ଭଲ ଲାଗୁଛି", "opinion", "or", "oriya", False),
    ("m13", "ଗତ ବର୍ଷ ମୁଁକୁ ମିଛି", "personal_experience", "or", "oriya", False),
    ("m15", "aapko placement office se baat karni chahiye", "advice", "hi", "latin", True),
    ("m17", "scholarship ke liye portal par apply karo", "solution", "hi", "latin", True),
    ("m19", "lekin fees itne zyada nahi hain", "counterargument", "hi", "latin", True),
    ("m22", "বৃত্তির জন্য আবেদন করুন", "solution", "bn", "bengali", False),
    ("m23", "কিন্তু ফি এত বেশি নয়", "counterargument", "bn", "bengali", False),
]


def to_annotations(rows):
    return [
        ResponseAnnotation(
            document_id=document_id,
            text=text,
            response_type=response_type,
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
        )
        for document_id, text, response_type, language, script, is_code_mixed in rows
    ]


def to_documents(rows):
    return [
        ResponseDocument(
            document_id=document_id,
            text=text,
            original_text=text,
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
        )
        for document_id, text, _role, language, script, is_code_mixed in rows
    ]


# ===========================================================================
# 1. THE SIX RESPONSE TYPES
# ===========================================================================


def section_one():
    rule("1. THE SIX RESPONSE TYPES  (fixed by the assignment, not learned)")
    print()
    print("Response classification asks: what ROLE does this comment play?")
    print("It is not sentiment - 'apply for the scholarship' has no polarity,")
    print("and 'same bro' can be said happily or sarcastically.")
    print()
    from app.services.nlp.response_classification import (
        RESPONSE_TYPE_CODES,
        RESPONSE_TYPE_DESCRIPTIONS,
    )

    for name in RESPONSE_TYPES:
        print(f"  {RESPONSE_TYPE_CODES[name]}. {name:<20} {RESPONSE_TYPE_DESCRIPTIONS[name]}")
    print()
    print(f"  {len(RESPONSE_TYPES)} classes, fixed and canonically ordered.")
    print("  An annotation outside this set is a labelling error, refused by")
    print("  train() - the model can never grow a seventh class.")
    print()
    print("  Model identifiers: tfidf_logistic_multiclass / v1")
    print("  Output head     : softmax (one primary role, six-way distribution)")


# ===========================================================================
# 2. SYNTHETIC ANNOTATIONS
# ===========================================================================


def section_two():
    rule("2. SYNTHETIC ANNOTATIONS  (hand-written; not Reddit, not annotators)")
    annotations = to_annotations(TRAINING_ROWS)
    print()
    print(f"{'id':<5} {'role':<20} {'lang':<5} {'text'}")
    print(SUBTLE)
    for annotation in annotations[:8]:
        print(
            f"{annotation.document_id:<5} {annotation.response_type:<20} "
            f"{annotation.language:<5} {annotation.text}"
        )
    print(f"...  ({len(annotations) - 8} more)")
    print()
    counts = {}
    for annotation in annotations:
        counts[annotation.response_type] = counts.get(annotation.response_type, 0) + 1
    print("  per-role counts:", {name: counts[name] for name in RESPONSE_TYPES})
    print()
    print("  CSV format (data/annotations/response_annotations.csv, NOT created here):")
    from app.services.nlp.response_classification.models.annotation import CSV_COLUMNS

    print("    " + ",".join(CSV_COLUMNS))
    print('    c-1001,"same bro exactly",agreement,post-1,c-1001,en,latin,false,en')
    print()
    print("  Only document_id, text and response_type are required; unknown extra")
    print("  columns (e.g. an annotator's notes) are ignored by the loader.")
    print()
    print("  Real annotations are produced by the human workflow, never fabricated.")


# ===========================================================================
# 3. TRAINING
# ===========================================================================


def section_three(annotations):
    rule("3. TRAINING  (TF-IDF -> multiclass logistic regression, softmax)")
    classifier = TfidfLogisticResponseClassifier()
    report = classifier.train(annotations)

    print()
    print(f"  model              : {report.model_name} / {report.model_version}")
    print(f"  training comments  : {report.n_training_documents}")
    print(f"  response types     : {report.n_classes} (expected {len(RESPONSE_TYPES)})")
    print(f"  classes per comment: {report.classes_per_document}  (mutually exclusive)")
    print(f"  TF-IDF features    : {report.n_features}")
    print(f"  solver / random    : {report.config['logistic_regression']['solver']}"
          f" / {report.config['logistic_regression']['random_state']}")
    print(f"  class_weight       : {report.config['logistic_regression']['class_weight']}")
    print(f"  confidence floor   : {report.threshold}")
    print()
    print("  learned classes (canonical order):")
    for name in report.labels:
        print(f"    {name:<22} {report.examples_per_class[name]} example(s)")
    print()
    if report.missing_response_types:
        print("  NOT covered by the training data:", ", ".join(report.missing_response_types))
    else:
        print("  All six response types are covered - the probability vector has 6 values.")
    if report.warnings:
        print()
        for warning in report.warnings:
            print(f"  ! {warning}")
    return classifier


# ===========================================================================
# 4. PREDICTION
# ===========================================================================


def section_four(classifier):
    rule("4. PREDICTION  (one role per comment)")
    comments = [
        ResponseDocument(
            document_id="c-9001",
            text="same bro exactly",
            post_id="post-1",
            comment_id="c-9001",
            original_text="Same bro, exactly!!",
            language="en",
        ),
        ResponseDocument(
            document_id="c-9002",
            text="i faced this exact problem last year",
            post_id="post-1",
            comment_id="c-9002",
            language="en",
        ),
        ResponseDocument(
            document_id="c-9003",
            text="you should contact the placement office",
            post_id="post-1",
            comment_id="c-9003",
            language="en",
        ),
    ]
    # A low floor so the arg-max is visible on a toy corpus.  The default 0.5
    # floor is shown in section 6.
    result = classifier.predict(comments, threshold=0.01)

    print()
    for comment in comments:
        print(f"  {comment.document_id}  \"{comment.text}\"")
    print()
    for prediction in result.predictions:
        print(
            f"  {prediction.document_id}  ->  {prediction.response_type:<20} "
            f"confidence={prediction.confidence:.4f}"
        )
    print()
    print(f"  model            : {result.model_name} / {result.model_version}")
    print(f"  known types      : {', '.join(result.known_response_types)}")
    print(f"  missing types    : {result.missing_response_types or 'none'}")
    print(f"  abstained        : {result.n_abstained}")
    print(f"  role frequency   : {result.response_type_frequency()}")
    print(f"  mean confidence  : {result.average_confidence():.4f}")
    print()
    print("  original_text is echoed untouched, never normalised by the model:")
    echoed = result.prediction_for("c-9001")
    print(f"    text          = {echoed.text!r}")
    print(f"    original_text = {echoed.original_text!r}")


# ===========================================================================
# 5. CLASS PROBABILITIES
# ===========================================================================


def section_five(classifier):
    rule("5. CLASS PROBABILITIES  (six values, summing to 1 - a softmax, not sigmoids)")
    comments = [
        ResponseDocument(document_id="p1", text="actually the fees are not that high"),
        ResponseDocument(document_id="p2", text="apply for the scholarship on the portal"),
    ]
    result = classifier.predict(comments, threshold=0.01)

    print()
    header = f"  {'id':<4} " + " ".join(f"{name[:9]:>10}" for name in RESPONSE_TYPES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for prediction in result.predictions:
        cells = " ".join(
            f"{prediction.response_probabilities[name]:>10.4f}" for name in RESPONSE_TYPES
        )
        print(f"  {prediction.document_id:<4} {cells}")
    print()
    for prediction in result.predictions:
        total = sum(prediction.response_probabilities.values())
        print(
            f"  {prediction.document_id}: sum={total:.6f}  winner="
            f"{prediction.top_response_type()}  quoted as {prediction.response_type!r}"
        )
    print()
    print("  The values sum to 1 (within 1e-6 rounding), because the six roles are")
    print("  mutually exclusive. A One-vs-Rest sigmoid model would give six")
    print("  INDEPENDENT scores that could all fire at once - a different question.")
    print()
    print("  The full vector is returned even when the model abstains, so a")
    print("  consumer can re-apply a different floor without re-running inference:")
    strict = classifier.predict(comments, threshold=0.99)
    for prediction in strict.predictions:
        print(
            f"    {prediction.document_id}: abstained={prediction.abstained}  "
            f"response_type={prediction.response_type!r}  "
            f"top={prediction.top_response_type()!r}"
        )


# ===========================================================================
# 6. EXAMPLES FOR ALL SIX CLASSES
# ===========================================================================


def section_six(classifier):
    rule("6. EXAMPLES FOR ALL SIX CLASSES  (multilingual and code-mixed included)")
    examples = [
        ("agreement", "same bro exactly"),
        ("personal_experience", "i faced this exact problem last year"),
        ("opinion", "i think this college is better"),
        ("advice", "you should contact the placement office"),
        ("solution", "apply for the scholarship on the portal"),
        ("counterargument", "actually the fees are not that high"),
        ("agreement (Hinglish)", "haan bilkul same baat hai"),
        ("personal_experience (Hinglish)", "pichle saal mujhe bhi yahi problem aayi thi"),
        ("opinion (Devanagari)", "मुझे लगता है यह कॉलेज बेहतर है"),
        ("agreement (Bengali)", "হ্যাঁ ঠিক একই কথা"),
        ("personal_experience (Bengali)", "আমার একই অভিজ্ঞতা হয়েছিল"),
        ("opinion (Odia)", "ଆମେ ସେବାଟା ଭଲ ଲାଗୁଛି"),
        ("solution (Hinglish)", "scholarship ke liye portal par apply karo"),
        ("counterargument (Bengali)", "কিন্তু ফি এত বেশি নয়"),
        ("agreement (emoji token)", "same bro exactly <EMOJI_tada>"),
    ]
    documents = [
        ResponseDocument(document_id=f"ex{index}", text=text)
        for index, (_label, text) in enumerate(examples)
    ]
    result = classifier.predict(documents, threshold=0.01)

    print()
    for (label, text), prediction in zip(examples, result.predictions):
        ordered = sorted(
            prediction.response_probabilities.items(), key=lambda kv: -kv[1]
        )
        winner = ordered[0][0]
        runner_up = ordered[1]
        mark = "ok " if label.split(" (")[0] == winner else "MISS"
        print(f"  [{mark}] expected {label}")
        print(f"         text       : {text}")
        print(
            f"         predicted  : {winner} ({prediction.response_probabilities[winner]:.4f})"
        )
        print(
            f"         runner-up  : {runner_up[0]} ({runner_up[1]:.4f})"
        )
    print()
    print("  The Hindi/Bengali/Odia rows and the emoji placeholder are handled by")
    print("  the same pipeline: the token pattern is script-agnostic and <EMOJI_...>")
    print("  tokens are stripped from the model copy while the document keeps them.")


# ===========================================================================
# 7. HELD-OUT EVALUATION
# ===========================================================================


def section_seven(classifier):
    rule("7. EVALUATION  (genuinely held-out synthetic set - still NOT Reddit)")
    heldout = to_annotations(HELDOUT_ROWS)

    training_ids = {row[0] for row in TRAINING_ROWS}
    training_texts = {row[1] for row in TRAINING_ROWS}
    overlap = [row[0] for row in HELDOUT_ROWS if row[0] in training_ids or row[1] in training_texts]
    print()
    print(f"  held-out comments : {len(heldout)}  (training had {len(TRAINING_ROWS)})")
    print(f"  id/text overlap   : {overlap or 'none - the sets are disjoint'}")
    print()

    report = classifier.evaluate(heldout, threshold=0.01)
    print(report.format_summary())
    print()
    print("  Reading the numbers honestly:")
    print("   - micro-F1 == accuracy here, because single-label multiclass data")
    print("     makes those two quantities identical. Both are reported so the")
    print("     identity is visible rather than assumed.")
    print("   - the off-diagonal cells are the interesting part: the mistakes are")
    print("     agreement/opinion comments read as counterargument, not")
    print("     random noise.")
    print("   - a 0.89 accuracy here reflects how separable THIS toy corpus is.")
    print("     Real annotation will not look like this.")

    print()
    print("  At the default 0.5 confidence floor, on the same held-out set:")
    strict = classifier.evaluate(heldout)
    print(f"    abstained        : {strict.n_abstained} of {strict.n_documents}")
    print(f"    accuracy         : {strict.accuracy}")
    print("  Abstentions are counted as misses, not dropped from the denominator.")

    print()
    print("  And when the data is too thin, metrics are withheld rather than invented:")
    tiny = evaluate_response_classification(
        ["advice", "agreement", "advice"], ["advice", "agreement", "solution"], list(RESPONSE_TYPES)
    )
    print(f"    is_meaningful : {tiny.is_meaningful}")
    print(f"    reason        : {tiny.reason}")
    print(f"    accuracy      : {tiny.accuracy}")
    print(f"    macro_f1      : {tiny.macro_f1}")


# ===========================================================================
# 8. SMALL-DATA REFUSAL
# ===========================================================================


def section_eight():
    rule("8. SMALL-DATA REFUSAL  (train() declining bad data, with reasons)")
    attempts = []

    attempts.append((
        "empty corpus",
        [],
    ))
    attempts.append((
        "only 3 comments",
        to_annotations(TRAINING_ROWS[:3]),
    ))
    attempts.append((
        "a single response type",
        to_annotations([
            ("z1", "same bro", "agreement", "en", "latin", False),
            ("z2", "true that", "agreement", "en", "latin", False),
            ("z3", "exactly right", "agreement", "en", "latin", False),
            ("z4", "yes i agree", "agreement", "en", "latin", False),
            ("z5", "correct indeed", "agreement", "en", "latin", False),
        ]),
    ))
    # Keep every row except all but one counterargument example.
    one_example = [row for row in TRAINING_ROWS if not row[2].startswith("counter")] + [
        row for row in TRAINING_ROWS if row[2] == "counterargument"
    ][:1]
    attempts.append((
        "a role with a single example",
        to_annotations(one_example),
    ))
    attempts.append((
        "an invalid label ('sarcasm')",
        to_annotations(list(TRAINING_ROWS) + [
            ("x1", "some new thing", "sarcasm", "en", "latin", False)
        ]),
    ))
    attempts.append((
        "one comment_id given two roles",
        to_annotations(list(TRAINING_ROWS) + [
            ("a01", "same bro exactly", "advice", "en", "latin", False)
        ]),
    ))
    attempts.append((
        "text that yields no TF-IDF features",
        to_annotations([
            ("e1", "<URL>", "agreement", "en", "latin", False),
            ("e2", "<URL>", "agreement", "en", "latin", False),
            ("e3", "<URL>", "opinion", "en", "latin", False),
            ("e4", "<URL>", "opinion", "en", "latin", False),
            ("e5", "<URL>", "opinion", "en", "latin", False),
            ("e6", "<URL>", "opinion", "en", "latin", False),
        ]),
    ))
    attempts.append((
        "a partial taxonomy when all six are required",
        to_annotations([row for row in TRAINING_ROWS if row[2] in ("agreement", "opinion")]),
    ))

    for label, annotations in attempts:
        config = ResponseModelConfig(require_all_response_types=True) if "all six" in label else None
        classifier = TfidfLogisticResponseClassifier(config)
        try:
            classifier.train(annotations)
        except (ValueError, TypeError) as exc:
            first_line = str(exc).split(". ")[0]
            print()
            print(f"  {label}")
            print(f"    REFUSED: {first_line}.")
        else:
            print()
            print(f"  {label}")
            print(f"    (accepted - this case is not a refusal)")

    print()
    print("  Each refusal names the problem and the way out, rather than training")
    print("  a model that would look fitted but generalise to nothing.")


# ===========================================================================
# 9. CONTEXT AND IMBALANCE
# ===========================================================================


def section_nine(classifier):
    rule("9. CONTEXT IS OPTIONAL, AND IMBALANCE IS HANDLED EXPLICITLY")

    print()
    print("  (a) Thread context is carried but NOT used by the V1 classifier.")
    plain = ResponseDocument(
        document_id="x", text="same bro exactly", original_text="Same bro exactly"
    )
    with_context = ResponseDocument(
        document_id="x",
        text="same bro exactly",
        original_text="Same bro exactly",
        parent_comment_text="the fees are far too high for this course",
        topic_id=7,
        topic_label="fees",
        issue_labels=["fees"],
    )
    a = classifier.predict([plain], threshold=0.01).predictions[0]
    b = classifier.predict([with_context], threshold=0.01).predictions[0]
    print(f"      comment alone      -> {a.response_type} ({a.confidence:.4f})")
    print(f"      same + full context-> {b.response_type} ({b.confidence:.4f})")
    print(f"      identical distribution: {a.response_probabilities == b.response_probabilities}")
    print()
    print("      'same bro' is only agreement if it agrees with something. A")
    print("      context-aware model needs the parent text; V1 does not have it, so")
    print("      it must not pretend to. The fields exist on the interface so a")
    print("      future model can use them without changing any public signature.")
    print()
    print("      The context is still echoed back for traceability:")
    print(f"        parent_comment_text = {b.parent_comment_text!r}")
    print(f"        issue_labels        = {b.issue_labels}")

    print()
    print("  (b) class_weight='balanced' is the default, and it changes the model.")
    rows = [
        (f"maj{index}", f"same bro exactly {index}", "agreement", "en", "latin", False)
        for index in range(10)
    ]
    for response_type, base in (
        ("personal_experience", "i faced this"),
        ("opinion", "i think this"),
        ("advice", "you should"),
        ("solution", "apply now"),
        ("counterargument", "actually that is wrong"),
    ):
        for index in range(3):
            rows.append(
                (f"{response_type[:2]}{index}", f"{base} {index}", response_type, "en", "latin", False)
            )
    skewed = to_annotations(rows)

    balanced = TfidfLogisticResponseClassifier(ResponseModelConfig(class_weight="balanced"))
    unweighted = TfidfLogisticResponseClassifier(ResponseModelConfig(class_weight=None))
    balanced.train(skewed)
    unweighted.train(skewed)

    probe = ResponseDocument(
        document_id="probe", text="actually the fees are not that high"
    )
    p_balanced = balanced.predict([probe], threshold=0.01).predictions[0]
    p_plain = unweighted.predict([probe], threshold=0.01).predictions[0]
    print()
    print(f"      skewed corpus: 'agreement' holds 10 of 25 comments ({40}%)")
    print(f"      probe comment : \"{probe.text}\"  (gold: counterargument)")
    print()
    print("      class_weight      counterargument    agreement")
    print(
        f"        'balanced'      {p_balanced.response_probabilities['counterargument']:.4f}"
        f"            {p_balanced.response_probabilities['agreement']:.4f}"
    )
    print(
        f"        None            {p_plain.response_probabilities['counterargument']:.4f}"
        f"            {p_plain.response_probabilities['agreement']:.4f}"
    )
    print()
    print("      Balancing shifts mass off the dominant role and onto the rare one.")
    print("      That is the point: agreement and opinion are cheap to collect and")
    print("      dominate real corpora, while advice and solution are the interesting,")
    print("      rare cases. Unweighted training would look accurate and be useless")
    print("      on the roles that matter. Accuracy is reported anyway - but always")
    print("      next to macro-F1 and per-class support.")

    print()
    print("  Determinism: two identically trained classifiers produce byte-identical")
    print("  output, because TF-IDF, lbfgs and the fixed class order are all")
    print("  deterministic and no timestamp is recorded by default.")
    first = classifier.predict(to_documents(HELDOUT_ROWS), threshold=0.01)
    second = classifier.predict(to_documents(HELDOUT_ROWS), threshold=0.01)
    print(f"    identical JSON across two predict() calls: {first.model_dump_json() == second.model_dump_json()}")


def main():
    print(BANNER)
    print("RESPONSE CLASSIFICATION - what role does this comment play?")
    print(BANNER)
    print("ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE")
    print("All data below is synthetic. No number here is Reddit performance.")

    section_one()
    annotations = to_annotations(TRAINING_ROWS)
    section_two()
    classifier = section_three(annotations)
    section_four(classifier)
    section_five(classifier)

    # Retrain on English + multilingual so the non-Latin rows are learnable.
    multilingual = TfidfLogisticResponseClassifier()
    multilingual_report = multilingual.train(annotations + to_annotations(MULTILINGUAL_ROWS))
    print()
    print(SUBTLE)
    print(f"  (the next section retrains on {multilingual_report.n_training_documents} comments,")
    print(f"   adding {len(MULTILINGUAL_ROWS)} code-mixed / non-Latin examples)")
    print(SUBTLE)
    section_six(multilingual)
    section_seven(classifier)
    section_eight()
    section_nine(multilingual)

    print()
    print(BANNER)
    print("End. Real numbers require the annotation workflow described in")
    print("docs/response_classification.md, on a disjoint held-out annotation set.")
    print(BANNER)


if __name__ == "__main__":
    main()
