"""
Demo script: SOLUTION DETECTION on synthetic IIM Sambalpur Community
Intelligence data.

=============================================================================
ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE
Every comment, annotation and metric below is synthetic and hand-written. The
corpora were not collected from Reddit and not annotated by real annotators.
The printed scores exist only to show that the code paths work end to end. They
say nothing about how this approach would perform on real Reddit threads - see
docs/solution_detection.md, "Evaluation caveats".
=============================================================================

Solution detection asks one question:

    does this comment provide an actionable solution/recommendation?

    "apply for the scholarship and submit the documents"  -> True
    "try applying before the deadline"                    -> True
    "you should contact the placement office"             -> True (advice)
    "same bro"                                           -> False
    "I faced this last year"                             -> False
    "actually fees aren't that high"                     -> False

It is NOT a taxonomy. The response-classification stage answers a different
question - "what role does this comment play?" - with one of six roles. Advice
and solution are both *candidates* here, but a candidate is not a solution:
"you should probably try emailing them, not sure" is advice and may contain
nothing actionable at all. is_solution comes only from this stage's trained
detector, never from the response role.

It is also NOT Issue-Solution Linking and NOT Resolution Detection. This stage
answers a per-comment question and stops: it never decides which issue a
solution answers, and never claims a problem was actually fixed.

Sections:

    1. THE ONE QUESTION         the binary target, in the annotator's words
    2. SYNTHETIC ANNOTATIONS    SolutionAnnotation records
    3. TRAINING                 what was fit, and from how much
    4. PREDICTION               one boolean per comment
    5. SOLUTION PROBABILITIES   the sigmoid, printed per comment
    6. ADVICE vs SOLUTION       candidate != solution, shown concretely
    7. HELD-OUT EVALUATION      a genuinely disjoint synthetic set
    8. THRESHOLD & ABSTENTION   the decision boundary, and the opt-in band
    9. GUARDRAILS              train() and evaluate() declining bad data
   10. MULTILINGUAL & EMOJI     code-mixed text, and what this V1 cannot do
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Windows consoles default to cp1252 and cannot render Devanagari / Bengali /
# Odia. Force UTF-8 so the multilingual output is readable.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.nlp.solution_detection import (
    DEFAULT_ANNOTATION_CSV,
    LABEL_NOT_SOLUTION,
    LABEL_SOLUTION,
    SolutionAnnotation,
    SolutionDocument,
    SolutionModelConfig,
    TfidfLogisticSolutionClassifier,
    classify_response_candidate,
    evaluate_solution_detection,
)

BANNER = "=" * 78
SUBTLE = "-" * 78


def rule(title):
    print()
    print(BANNER)
    print(title)
    print(BANNER)


def note(text):
    print(f"  {text}")


# ---------------------------------------------------------------------------
# SYNTHETIC DATA
#
# `text` is what the preprocessing stage would have produced: emoji already
# turned into <EMOJI_...> tokens, URLs into <URL>, casing and spacing folded.
# These strings are invented for this demo and are not quotes from any thread.
#
# The split between the two classes is deliberately lexical for the first few
# examples (imperatives and deontic markers vs social reactions), because that
# is what makes a 40-comment synthetic corpus separable at all. A real corpus is
# not this clean - see section 10 and the documentation's limitations.
# ---------------------------------------------------------------------------

TRAIN_ROWS = [
    # (document_id, text, is_solution, response_type, solution_type)
    ("tr-neg-01", "same bro exactly what i said", False, "agreement", None),
    ("tr-neg-02", "bhai ye toh bata dete pehle", False, "agreement", None),
    ("tr-neg-03", "i faced this same issue last year", False, "personal_experience", None),
    ("tr-neg-04", "i think the placement process is broken", False, "opinion", None),
    ("tr-neg-05", "actually fees arent that high compared to other iims", False, "counterargument", None),
    ("tr-neg-06", "thanks for the detailed explanation bro", False, "agreement", None),
    ("tr-neg-07", "this happened to me too it was stressful", False, "personal_experience", None),
    ("tr-neg-08", "i dont think this college is worth it", False, "opinion", None),
    ("tr-neg-09", "mera cgpa bhi 7.8 tha same situation", False, "personal_experience", None),
    ("tr-neg-10", "nope i dont agree with that at all", False, "counterargument", None),
    ("tr-neg-11", "just lurking here reading the thread", False, None, None),
    ("tr-neg-12", "bhai ye toh funny hai <EMOJI_loudly_crying_face>", False, "agreement", None),
    ("tr-neg-13", "what a waste of time everyone", False, "opinion", None),
    ("tr-neg-14", "i also had a bad experience with the hostel", False, "personal_experience", None),
    # Advice-shaped but with no concrete step. The annotation guideline calls
    # these NOT solutions: a deontic marker alone is not an actionable
    # recommendation, and "it depends" hands the reader nothing to do.
    ("tr-neg-15", "you should probably do something about this, not sure how", False, "advice", None),
    ("tr-neg-16", "i would say just wait and see what happens", False, "advice", None),
    ("tr-neg-17", "you might want to look into this sometime", False, "advice", None),
    ("tr-pos-01", "you should contact the placement office directly", True, "advice", None),
    ("tr-pos-02", "apply for the scholarship on the student portal", True, "solution", None),
    ("tr-pos-03", "try submitting the form before the deadline closes", True, "solution", None),
    ("tr-pos-04", "you must submit your marksheet along with the application", True, "advice", None),
    ("tr-pos-05", "you can apply for the fee waiver if your family income is low", True, "solution", "fee_waiver"),
    ("tr-pos-06", "try emailing the dean they respond within two days", True, "solution", None),
    ("tr-pos-07", "i would suggest joining the coding club it helped me", True, "advice", "extracurriculars"),
    ("tr-pos-08", "apply now and keep the acknowledgement receipt safe", True, "solution", None),
    ("tr-pos-09", "you should start preparing for the aptitude test early", True, "advice", "placement_preparation"),
    ("tr-pos-10", "check the eligibility criteria before you pay the fee", True, "solution", None),
    ("tr-pos-11", "try applying to the scholarship again next semester", True, "solution", "fee_waiver"),
    ("tr-pos-12", "you must upload your photo in the portal or it fails", True, "advice", None),
    ("tr-pos-13", "submit the application form at the office counter instead", True, "solution", None),
    ("tr-pos-14", "join the alumni network it helps with referrals", True, "advice", "extracurriculars"),
]

HELDOUT_ROWS = [
    ("ho-neg-01", "same story bro exactly", False, "agreement", None),
    ("ho-neg-02", "i had a similar experience in my first year", False, "personal_experience", None),
    ("ho-neg-03", "the hostel food is terrible honestly", False, "opinion", None),
    ("ho-neg-04", "i disagree fees are higher than other colleges", False, "counterargument", None),
    ("ho-neg-05", "bhai ye toh sad hai <EMOJI_pleading_face>", False, "agreement", None),
    ("ho-neg-06", "reading this thread made my day", False, "agreement", None),
    ("ho-neg-07", "i dont want to go there again", False, "opinion", None),
    ("ho-neg-08", "my roommate had the same problem", False, "personal_experience", None),
    ("ho-pos-01", "you should email the registrar about the fee revision", True, "advice", None),
    ("ho-pos-02", "apply through the scholarship portal before the cutoff", True, "solution", None),
    ("ho-pos-03", "try the hostel allotment again in the next round", True, "solution", None),
    ("ho-pos-04", "you must carry your id proof to the verification", True, "advice", None),
    ("ho-pos-05", "submit the fee receipt at the accounts section", True, "solution", None),
    ("ho-pos-06", "you should start the aptitude prep at least two months early", True, "advice", "placement_preparation"),
    # Two deliberately awkward cases, so the held-out numbers are not a
    # flawless 1.0: a perfect matrix on 16 synthetic comments would be a fixture
    # artefact wearing the costume of a result. A short imperative-free solution
    # and a concrete-step-free "advice" are both easy for a lexical model to get
    # wrong, and which one it gets wrong is not worth predicting in a demo.
    ("ho-pos-07", "just pay the fee online", True, "solution", None),
    ("ho-neg-09", "you should sort this out somehow", False, "advice", None),
]


def annotations_from(rows):
    return [
        SolutionAnnotation(
            document_id=document_id,
            text=text,
            is_solution=is_solution,
            response_type=response_type,
            solution_type=solution_type,
            rationale=(
                "contains an actionable step" if is_solution else "no actionable step"
            ),
        )
        for document_id, text, is_solution, response_type, solution_type in rows
    ]


TRAIN = annotations_from(TRAIN_ROWS)
HELDOUT = annotations_from(HELDOUT_ROWS)

# Comment texts used for the qualitative sections, with whatever upstream
# context a caller would have available. V1 reads none of it as a feature.
SHOWCASE = [
    ("apply for the scholarship on the portal", "solution", "en", "latin", False),
    ("you should contact the placement office", "advice", "en", "latin", False),
    ("try applying before the deadline closes", "solution", "en", "latin", False),
    ("you should probably do something about this, not sure how", "advice", "en", "latin", False),
    ("same bro", "agreement", "en", "latin", False),
    ("i faced this same issue last year", "personal_experience", "en", "latin", False),
    ("i think the fees are too high", "opinion", "en", "latin", False),
    ("apply for the scholarship and submit the documents", "solution", "hi", "latin", True),
    ("apply for the scholarship on the portal bhai", "solution", "hi", "latin", True),
    ("aapko scholarship ke liye apply karna chahiye", "advice", "hi", "latin", True),
    ("bhai ye toh sad hai <EMOJI_pleading_face>", "agreement", "hi", "latin", True),
    ("আবেদন করতে হবে পোর্টালে", "solution", "bn", "bengali", False),
]


def main():
    classifier = TfidfLogisticSolutionClassifier()
    training_report = classifier.train(TRAIN)

    # ------------------------------------------------------------------
    rule("1. THE ONE QUESTION")
    # ------------------------------------------------------------------
    note("is_solution: does this comment provide an actionable")
    note("solution/recommendation?")
    print()
    print(f"  targets (canonical order) : {LABEL_NOT_SOLUTION} (0), {LABEL_SOLUTION} (1)")
    note("The order is fixed, so index 1 always means 'provides a solution'")
    note("and two runs produce the same confusion-matrix axes.")
    print()
    note("Not a taxonomy. solution_type is free-form annotation data,")
    note("reported for coverage and never learned. There is no category list")
    note("in the package, on purpose: a fixed list would encode one community's")
    note("vocabulary into a general model.")
    print()
    note("NOT this stage's job:")
    note("- which ISSUE a solution answers   (Issue-Solution Linking)")
    note("- whether the problem was FIXED    (Resolution Detection)")

    # ------------------------------------------------------------------
    rule("2. SYNTHETIC ANNOTATIONS")
    # ------------------------------------------------------------------
    note(f"{len(TRAIN)} hand-written SolutionAnnotation records (synthetic).")
    print()
    for annotation in TRAIN[:4] + TRAIN[-2:]:
        mark = "1" if annotation.is_solution else "0"
        role = annotation.response_type or "-"
        print(f"  is_solution={mark}  response_type={role:<20} {annotation.text}")
    print()
    note("An empty is_solution cell is REFUSED, never read as False: a missing")
    note("annotation that defaulted to a negative would teach the detector to")
    note("ignore solutions, which is the most damaging error this stage has.")
    print()
    note(f"Real annotations belong at: {DEFAULT_ANNOTATION_CSV}")
    note("That file is created by the human annotation workflow and is")
    note("deliberately not shipped - fabricated labels make every number a lie.")

    # ------------------------------------------------------------------
    rule("3. TRAINING")
    # ------------------------------------------------------------------
    print(f"  model                 : {training_report.model_name} / {training_report.model_version}")
    print(f"  training documents    : {training_report.n_training_documents}")
    print(f"  per class             : {training_report.examples_per_class}")
    print(f"  features (TF-IDF)     : {training_report.n_features}")
    print(f"  threshold             : {training_report.threshold}")
    print(f"  abstain_threshold     : {training_report.abstain_threshold}  (None = never abstain)")
    print(f"  library versions      : {training_report.library_versions}")
    print()
    note("TF-IDF unigrams + bigrams -> binary logistic regression.")
    note("With K=2 classes, scikit-learn's softmax is algebraically identical")
    note("to one sigmoid, so predict_proba[:,1] IS sigma(w.x + b) exactly.")
    print()
    note("stop_words=None on purpose: 'you should', 'you must' and 'try to'")
    note("are the class's most reliable features, and an English stop-word")
    note("list deletes precisely the deontic markers that make text a solution.")
    if training_report.warnings:
        print()
        note("training warnings:")
        for warning in training_report.warnings:
            print(f"    ! {warning}")

    # ------------------------------------------------------------------
    rule("4. PREDICTION")
    # ------------------------------------------------------------------
    documents = [
        SolutionDocument(
            document_id=f"demo-{index:02d}",
            text=text,
            response_type=response_type,
            language=language,
            script=script,
            is_code_mixed=is_code_mixed,
        )
        for index, (text, response_type, language, script, is_code_mixed) in enumerate(SHOWCASE, start=1)
    ]
    result = classifier.predict(documents)
    for prediction in result.predictions:
        mark = "TRUE " if prediction.is_solution else "false"
        print(f"  is_solution={mark}  p={prediction.solution_probability:.4f}  {prediction.text}")
    print()
    print(f"  documents                    : {result.n_documents}")
    print(f"  solutions                    : {result.n_solutions}")
    print(f"  not solutions                : {result.n_not_solutions}")
    print(f"  solution candidates (role)   : {result.n_candidates}")
    print(f"  candidate, not a solution    : {result.n_candidates_not_solutions}")
    print(f"  solution, not a candidate    : {result.n_solutions_not_candidates}")
    print(f"  abstained                    : {result.n_abstained}")
    print()
    for warning in result.warnings:
        print(f"    ! {warning}")

    # ------------------------------------------------------------------
    rule("5. SOLUTION PROBABILITIES")
    # ------------------------------------------------------------------
    note("The two probabilities always sum to 1; index 1 is the solution:")
    for prediction in result.predictions[:8]:
        total = prediction.solution_probability + prediction.not_solution_probability
        print(
            f"  P(solution)={prediction.solution_probability:.6f}  "
            f"P(not)={prediction.not_solution_probability:.6f}  sum={total:.6f}  "
            f"{prediction.text}"
        )
    print()
    note("The probability is kept alongside the boolean on every prediction, so")
    note("a downstream linker can re-threshold without re-running the model.")

    # ------------------------------------------------------------------
    rule("6. ADVICE vs SOLUTION")
    # ------------------------------------------------------------------
    note("A candidate is not a solution. The response role decides only")
    note("whether a comment is worth CHECKING; the detector decides whether")
    note("it is one.")
    print()
    for text, role, _language, _script, _mixed in SHOWCASE[:7]:
        if role not in ("advice", "solution"):
            continue
        assessment = classify_response_candidate(role)
        prediction = next(p for p in result.predictions if p.text == text)
        verdict = prediction.is_solution
        print(
            f"  role={role:<9} is_candidate={str(assessment.is_candidate):<5} "
            f"is_solution={str(verdict):<5}  {text}"
        )
    print()
    note("'you should contact the placement office' is advice AND a real")
    note("solution, so the candidate flag and the boolean agree.")
    print()
    note("'you should probably do something about this, not sure how' is")
    note("advice but concrete-action-free - a deontic marker with no step to")
    note("take - and the model says false. That is exactly the case where")
    note("equating advice with a solution would have been wrong. The")
    note("annotation guideline, not the model, is what draws this line: see")
    note("docs/solution_detection.md, 'Advice is a candidate, not a solution'.")
    print()
    note("Structurally, CandidateAssessment.is_solution starts as None, so")
    note("'advice implies solution' cannot be written down without running")
    note("the detector. CandidateAssessment for advice:")
    print(f"    {classify_response_candidate('advice').model_dump_json()}")
    print(f"    {classify_response_candidate('solution').model_dump_json()}")
    print(f"    {classify_response_candidate('opinion').model_dump_json()}")

    # ------------------------------------------------------------------
    rule("7. HELD-OUT EVALUATION")
    # ------------------------------------------------------------------
    note(f"{len(HELDOUT)} disjoint synthetic comments - no overlap with the")
    note("training ids or texts, so this is memorisation measured separately.")
    evaluation = classifier.evaluate(HELDOUT)
    print()
    print(evaluation.format_summary())
    note("On a corpus this small and this clean, these numbers describe the")
    note("fixture, not the approach. Quote positive-class precision and recall")
    note("together; accuracy flatters a model on a mostly-negative corpus.")

    # ------------------------------------------------------------------
    rule("8. THRESHOLD & ABSTENTION")
    # ------------------------------------------------------------------
    note("The threshold is a decision boundary: is_solution = p >= threshold.")
    note("Below it the answer is false - a real answer, not a shrug.")
    print()
    for threshold in (0.2, 0.5, 0.8, 0.95):
        run = classifier.predict(documents, threshold=threshold)
        print(
            f"  threshold={threshold:<5} -> solutions={run.n_solutions:<2} "
            f"not_solutions={run.n_not_solutions:<2} "
            f"abstained={run.n_abstained}"
        )
    print()
    note("Abstention is opt-in and sits ABOVE the boundary: the band")
    note("threshold <= p < abstain_threshold is where the model leans positive")
    note("but is not confident enough to assert a link downstream.")
    banded = classifier.predict(documents, threshold=0.5, abstain_threshold=0.99)
    print()
    print("  with abstain_threshold=0.99:")
    for prediction in banded.predictions:
        if prediction.abstained:
            print(
                f"    is_solution={str(prediction.is_solution):<5} "
                f"abstained={str(prediction.abstained):<5} "
                f"p={prediction.solution_probability:.4f}  {prediction.text}"
            )
    print(f"    ... the other {banded.n_documents - banded.n_abstained} comment(s)")
    print(f"        are unaffected: they were already below the threshold.")
    print()
    print(f"  n_abstained: {banded.n_abstained}")
    note("Evaluation counts an abstention as a miss and excludes it from the")
    note("confusion matrix, so an abstention band cannot inflate the metrics.")
    note("With the default (None) the band is off and n_abstained is 0.")

    # ------------------------------------------------------------------
    rule("9. GUARDRAILS")
    # ------------------------------------------------------------------
    note("train() refuses data that cannot honestly produce a model, and")
    note("evaluate() withholds metrics the data cannot support.")
    print()

    refusals = [
        (
            "no data at all",
            [],
        ),
        (
            f"fewer than {SolutionModelConfig().min_training_documents} documents",
            TRAIN[:5],
        ),
        (
            "only one target class",
            [
                SolutionAnnotation(document_id=f"only{i}", text=f"apply for the scholarship {i}", is_solution=True)
                for i in range(12)
            ],
        ),
        (
            "a blank text body",
            TRAIN[:6]
            + [SolutionAnnotation(document_id="blank", text="   ", is_solution=True)]
            + TRAIN[6:12],
        ),
    ]
    for description, data in refusals:
        try:
            TfidfLogisticSolutionClassifier().train(data)
        except ValueError as exc:
            first_line = str(exc).split(".")[0]
            print(f"  REFUSED  {description}")
            print(f"           {first_line}.")
        else:
            print(f"  ACCEPTED {description}  <-- this should not happen")
    print()
    note("A blank is_solution cell is refused the same way - never read as False.")
    try:
        SolutionAnnotation.from_csv_row(
            {"document_id": "c1", "text": "apply now", "is_solution": ""}
        )
    except ValueError as exc:
        print(f"  REFUSED  blank is_solution cell")
        print(f"           {str(exc).split('.')[0]}.")
    print()
    note("evaluate() withholds rather than inventing a number:")
    thin = evaluate_solution_detection(
        [True, False, True, False], [True, False, True, False], [LABEL_NOT_SOLUTION, LABEL_SOLUTION]
    )
    print(f"    4 documents  -> meaningful={thin.is_meaningful}  {thin.reason}")
    print(f"                    accuracy={thin.accuracy}  (withheld, not 1.0)")
    one_class = evaluate_solution_detection(
        [True] * 12, [True] * 12, [LABEL_NOT_SOLUTION, LABEL_SOLUTION]
    )
    print(f"    one target   -> meaningful={one_class.is_meaningful}  {one_class.reason}")
    print(f"                    precision={one_class.precision}  (withheld, not 0.0)")
    print()
    note("Count facts still survive a withheld report:")
    print(
        f"    documents={thin.n_documents} scorable={thin.n_scorable} "
        f"abstained={thin.n_abstained} unscoreable={thin.n_unscoreable}"
    )

    # ------------------------------------------------------------------
    rule("10. MULTILINGUAL & EMOJI")
    # ------------------------------------------------------------------
    note("Code-mixed text is normal here, so token_pattern keeps every script")
    note("and lowercase folds case defensively.")
    print()
    for prediction in result.predictions[7:]:
        document = next(d for d in documents if d.document_id == prediction.document_id)
        note(
            f"  {document.language}/{document.script}"
            f"{'  code-mixed' if document.is_code_mixed else '':<14}"
            f"p={prediction.solution_probability:.4f} "
            f"is_solution={str(prediction.is_solution):<5} {document.text}"
        )
    print()
    note("What this V1 cannot do, stated plainly:")
    note("- <EMOJI_...> tokens are lossy: one token can stand for many visually")
    note("  distinct emojis, so tone-carrying emoji are dropped as features.")
    note("- Romanized Hindi shares no whitespace cues with English morphology;")
    note("  a model trained only on English will lean on 'apply' and 'should'.")
    note("- Bangla-script text is in the vocabulary only if the annotations")
    note("  contain it. V1 has no cross-lingual transfer, and none is claimed.")
    note("- Negation scope is weak: 'do NOT apply for the scholarship' keeps")
    note("  every feature of a solution and flips only the target.")
    note("- No thread context: parent_comment_text, topic_id and issue_labels")
    note("  are carried and echoed, never used as features.")
    print()
    print(BANNER)
    print("END - all data above is synthetic. Not project performance.")
    print(BANNER)


if __name__ == "__main__":
    main()
