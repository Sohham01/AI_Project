"""
Demo: Resolution / Unresolved Issue Detection.

    python demo_resolution_detection.py

Every example below is **synthetic**.  The numbers were chosen to make each
mechanism visible, not sampled from a corpus, and the discussions they stand for
were written for this file.  Nothing here measures the stage.

    ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE

What the demo shows, in order:

1. A well-covered discussion resolving.
2. The distinction the stage exists for: one confident solution in a long thread
   is *not* a resolution.
3. A busy thread where solutions were proposed but never linked to the issue.
4. A thin discussion - reported as a verdict with ``evidence_sufficient=False``,
   not as a fourth status.
5. A multi-label issue whose two labels resolve differently.
6. Configuration, determinism and the recomputation of a score by hand.
7. The evaluation refusal: a metric cannot be produced without human adjudication.
"""

from app.services.nlp.resolution_detection import (
    EvidenceBasedResolver,
    IssueResolutionEvidence,
    ResolutionAnnotation,
    ResolutionCorpus,
    ResolutionModelConfig,
    ResolutionStatus,
    TemporalEvidence,
    derive_evidence,
    evaluate_resolution,
)

BANNER = "=" * 78
WARNING = "ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE"


def heading(number: int, title: str) -> None:
    print(f"\n{BANNER}\n{number}. {title}\n{BANNER}")


# ---------------------------------------------------------------------------
# 1. Well covered
# ---------------------------------------------------------------------------


def demo_well_covered() -> None:
    heading(1, "A well-covered discussion")
    print(
        "12 comments, 4 solutions linked to this issue, 3 of them strong links,\n"
        "solid link scores and detector confidence.  The coverage is a large\n"
        "fraction of the discussion, so the densities are high too."
    )
    record = IssueResolutionEvidence(
        issue_id="post-1001",
        issue_label="fees",
        post_id="post-1001",
        total_comments=12,
        solution_candidate_count=5,
        linked_solution_count=4,
        strong_link_count=3,
        link_scores=[0.81, 0.74, 0.69, 0.33],
        solution_probabilities=[0.91, 0.87, 0.83, 0.58],
    )
    result = EvidenceBasedResolver().detect(record).results[0]
    print()
    print(result.format_summary())
    print(f"  score components : {result.score_components}")
    print(f"  weights          : {result.weights}")


# ---------------------------------------------------------------------------
# 2. One solution is not resolution
# ---------------------------------------------------------------------------


def demo_one_solution_is_not_resolution() -> None:
    heading(2, "One confident solution in a long thread is NOT a resolved issue")
    print(
        "40 comments and a single solution, matched very well indeed.  Every\n"
        "other term is a fraction of 40, so the score stays in the partial band.\n"
        "This is the failure mode the stage exists to prevent: reading a linker\n"
        "result and calling the issue resolved."
    )
    record = IssueResolutionEvidence(
        issue_id="post-1002",
        issue_label="network",
        total_comments=40,
        solution_candidate_count=1,
        linked_solution_count=1,
        strong_link_count=1,
        link_scores=[0.95],
        solution_probabilities=[0.97],
    )
    result = EvidenceBasedResolver().detect(record).results[0]
    print()
    print(result.format_summary())
    print(
        f"  density {result.solution_density:.4f} = 1 linked solution / 40 comments.\n"
        f"  A perfect link cannot compensate for a discussion with no other\n"
        f"  coverage, because coverage is weighted {result.weights['solution_density']:.2f}\n"
        f"  + {result.weights['strong_solution_density']:.2f} against "
        f"{result.weights['best_link_score']:.2f} for link strength."
    )


# ---------------------------------------------------------------------------
# 3. Candidates that were never linked
# ---------------------------------------------------------------------------


def demo_unlinked_candidates() -> None:
    heading(3, "Busy thread, solutions proposed, none linked to this issue")
    print(
        "8 solution candidates in the discussion and 0 linked to this issue and\n"
        "label.  Candidates are context about the discussion, not evidence that\n"
        "this issue got fixed, so they raise nothing at all - the score is 0."
    )
    record = IssueResolutionEvidence(
        issue_id="post-1003",
        issue_label="ui",
        total_comments=22,
        solution_candidate_count=8,
        linked_solution_count=0,
        strong_link_count=0,
    )
    result = EvidenceBasedResolver().detect(record).results[0]
    print()
    print(result.format_summary())
    print(
        f"  The linker found nothing for this pair.  That is a statement about the\n"
        f"  linker's output, not proof that no fix exists in the thread - which is\n"
        f"  why the warning says so rather than claiming the issue is unfixable."
    )


# ---------------------------------------------------------------------------
# 4. Insufficient evidence
# ---------------------------------------------------------------------------


def demo_thin_discussion() -> None:
    heading(4, "A thin discussion: reported, not relabelled")
    print(
        "2 comments.  There is still a verdict, and it is one of the three\n"
        "required states - but evidence_sufficient is False, a warning is attached,\n"
        "and the confidence is halved.  A fourth 'insufficient_evidence' status\n"
        "was rejected: that describes our confidence, not the issue."
    )
    record = IssueResolutionEvidence(
        issue_id="post-1004",
        issue_label="roi",
        total_comments=2,
        solution_candidate_count=1,
        linked_solution_count=1,
        strong_link_count=1,
        link_scores=[0.83],
        solution_probabilities=[0.88],
    )
    result = EvidenceBasedResolver().detect(record).results[0]
    print()
    print(result.format_summary())
    print(f"  status value           : {result.status.value!r} (still one of the three)")
    print(f"  evidence_sufficient    : {result.evidence_sufficient}")


# ---------------------------------------------------------------------------
# 5. Multi-label
# ---------------------------------------------------------------------------


def demo_multi_label() -> None:
    heading(5, "A multi-label issue whose two labels resolve differently")
    print(
        "One document tagged both 'fees' and 'roi'.  The fees question attracts\n"
        "linked answers; the roi question attracts none.  A single verdict for the\n"
        "document would have to collapse both into one number, and the solved fees\n"
        "problem would be recorded as a solved ROI problem."
    )
    batch = EvidenceBasedResolver().detect_batch([
        IssueResolutionEvidence(
            issue_id="post-1005", issue_label="fees", total_comments=10,
            solution_candidate_count=3, linked_solution_count=3, strong_link_count=2,
            link_scores=[0.78, 0.66, 0.51], solution_probabilities=[0.86, 0.79, 0.6],
        ),
        IssueResolutionEvidence(
            issue_id="post-1005", issue_label="roi", total_comments=10,
            solution_candidate_count=3, linked_solution_count=0,
        ),
    ])
    for result in batch.results:
        print()
        print(result.format_summary())
    print()
    print(f"  per-label status for post-1005 : {batch.statuses_for_issue('post-1005')}")
    print(f"  documents / label verdicts     : "
          f"{batch.report.n_issue_documents} / {batch.report.n_issue_labels}")
    print(f"  still open (not fully resolved) : "
          f"{[r.pair_key() for r in batch.not_resolved()]}")


# ---------------------------------------------------------------------------
# 6. Configuration, determinism, hand recomputation
# ---------------------------------------------------------------------------


def demo_configuration_and_determinism() -> None:
    heading(6, "Configuration, determinism, and recomputing a score by hand")
    record = IssueResolutionEvidence(
        issue_id="post-1006", issue_label="account", total_comments=9,
        solution_candidate_count=3, linked_solution_count=2, strong_link_count=1,
        link_scores=[0.72, 0.44], solution_probabilities=[0.81, 0.6],
        temporal=TemporalEvidence(solution_after_issue=True, follow_up_count=4,
                                  follow_up_resolution_signal=True),
    )

    resolver = EvidenceBasedResolver()
    result = resolver.detect(record).results[0]
    derived = derive_evidence(record)
    weights = result.weights

    print("The four terms:")
    total = 0.0
    for name, value in derived.as_components().items():
        contribution = weights[name] * value
        total += contribution
        print(f"  {name:<26} = {value:.6f}  x  {weights[name]:.2f}  = {contribution:.6f}")
    print(f"  {'sum':<26}   {'':<11}    {total:.6f}")
    print()
    print(f"  resolver's score          : {result.resolution_score:.12f}")
    print(f"  recomputed from components: {result.recompute_score:.12f}")
    print(f"  identical                 : "
          f"{abs(result.resolution_score - result.recompute_score) < 1e-12}")
    print()
    print(f"  result.status             : {result.status.value}")
    print(f"  confidence (boundary dist.): {result.confidence:.4f}   "
          f"- a heuristic, not a probability")
    print()
    print("  Temporal evidence was recorded and is NOT scored in V1:")
    print(f"    solution_after_issue          = {result.temporal.solution_after_issue}")
    print(f"    follow_up_count               = {result.temporal.follow_up_count}")
    print(f"    follow_up_resolution_signal   = "
          f"{result.temporal.follow_up_resolution_signal}")
    print("    A confirmed 'that fixed it' does not raise the score, because no")
    print("    measurement justifies a weight for it.  The clearest V2 candidate.")

    # Stricter thresholds, same evidence.  The score is unchanged because it does
    # not depend on where the cut-points are; only the verdict moves.
    stricter = EvidenceBasedResolver(ResolutionModelConfig(
        unresolved_upper=0.5, partial_upper=0.6, resolved_lower=0.7
    )).detect(record).results[0]
    print()
    print(f"  with stricter thresholds (unresolved<{stricter.thresholds['unresolved_upper']}, "
          f"resolved>={stricter.thresholds['resolved_lower']}): "
          f"{result.status.value} -> {stricter.status.value}")
    print(f"  the score did not move ({stricter.resolution_score:.12f}) - only the verdict did.")

    repeat = EvidenceBasedResolver().detect(record).results[0]
    print(f"  deterministic across runs  : {repeat.model_dump() == result.model_dump()}")


# ---------------------------------------------------------------------------
# 7. Evaluation refuses to invent a metric
# ---------------------------------------------------------------------------


def demo_evaluation_refusal() -> None:
    heading(7, "Evaluation refuses to produce a metric without human adjudication")
    batch = EvidenceBasedResolver().detect_batch([
        IssueResolutionEvidence(
            issue_id="post-1007", issue_label="fees", total_comments=8,
            solution_candidate_count=2, linked_solution_count=2, strong_link_count=2,
            link_scores=[0.79, 0.71], solution_probabilities=[0.88, 0.8],
        ),
        IssueResolutionEvidence(
            issue_id="post-1008", issue_label="ui", total_comments=7,
            solution_candidate_count=1, linked_solution_count=0,
        ),
    ])

    print("Calling the evaluator with nothing annotated:")
    try:
        evaluate_resolution(batch, ResolutionCorpus())
    except ValueError as exc:
        print(f"  ValueError: {exc}")
    print()
    print("The refusal is the feature.  A resolver that always answered")
    print("'unresolved' would score accuracy 1.0 against an all-unresolved")
    print("corpus, so an unannotated pipeline must produce a number that cannot")
    print("be quoted rather than a plausible-looking one.")
    print()
    print("With an explicitly supplied corpus, the arithmetic can be checked.")
    print("The labels below are SYNTHETIC and exist only to exercise the code:")
    corpus = ResolutionCorpus.from_records([
        {"issue_id": "post-1007", "issue_label": "fees", "gold_status": "resolved",
         "rationale": "synthetic label written to exercise the evaluator"},
        {"issue_id": "post-1008", "issue_label": "ui", "gold_status": "unresolved",
         "rationale": "synthetic label written to exercise the evaluator"},
    ])
    evaluation = evaluate_resolution(batch, corpus)
    print()
    for line in evaluation.format_report().splitlines()[:12]:
        print(f"  {line}")
    print()
    print("  Every figure above describes a two-row corpus of invented labels.")
    print("  It is not a measurement of this stage, and the report says so.")
    print()
    print(f"  The real annotation schema is ready for a human round: "
          f"{list(ResolutionAnnotation(issue_id='x', issue_label='y', gold_status='resolved').to_csv_row())}")
    print("  No such dataset ships with this project, and none was fabricated.")


def main() -> None:
    print(BANNER)
    print(WARNING)
    print(BANNER)
    print()
    print("Resolution Detection - what this stage is for")
    print()
    print("  Issue identification asks what problem is expressed.")
    print("  Solution detection asks whether a comment offers a fix.")
    print("  Linking asks whether that fix addresses that problem.")
    print("  This stage asks the only question that is about a whole discussion:")
    print()
    print("      considering the whole discussion, how resolved does this issue appear?")
    print()
    print("  All examples are synthetic.  No accuracy, F1 or confusion matrix is")
    print("  reported for the stage, because no human-annotated resolution dataset")
    print("  exists to report one against.")

    demo_well_covered()
    demo_one_solution_is_not_resolution()
    demo_unlinked_candidates()
    demo_thin_discussion()
    demo_multi_label()
    demo_configuration_and_determinism()
    demo_evaluation_refusal()

    print()
    print(BANNER)
    print(WARNING)
    print(BANNER)


if __name__ == "__main__":
    main()
