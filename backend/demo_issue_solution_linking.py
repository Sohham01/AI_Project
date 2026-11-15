"""
Issue-Solution Linking - synthetic end-to-end demonstration.

Run from the ``backend`` directory::

    python demo_issue_solution_linking.py

Everything below is INVENTED.  The comments are written by this file, the issue
labels and solution probabilities are assigned by this file, and the scores are
the linker scoring text that this file wrote.  Nothing here measures how the
system performs on real data, and no precision or recall figure is printed -
because no human-annotated link dataset exists to compute one from.  The only
thing this demonstrates is that the stage runs, scopes its candidates, and
reports what it did.

Three complete discussions, so the thread-scoping rule is visible rather than
described:

1. p1  a fees complaint in a hostel thread, with two genuine answers and a
   multi-label issue comment
2. p2  a wifi complaint in the same hostel
3. p3  an unrelated cooking thread that contains a highly confident "solution"
    - the case the cross-thread rule exists to reject

Also shown: a solution that is lexically similar to the issue but answers a
different question, and a comment that both looks like a solution and is not
about the issue, to make the point that a link is a claim about a *pair*.
"""

import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(__file__))

# Windows consoles default to cp1252 and cannot render Bengali / Odia.  Force
# UTF-8 so the multilingual threads below are actually readable rather than
# crashing the run on a print.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.services.nlp.issue_solution_linking import (
    IssueEvidence,
    LinkCorpus,
    SolutionEvidence,
    TfidfCosineIssueSolutionLinker,
)

BANNER = "=" * 78
DISCLAIMER = (
    "ILLUSTRATIVE / PLUMBING CHECK - NOT PROJECT PERFORMANCE. Synthetic comments, "
    "synthetic labels, synthetic probabilities. Shows mechanism only."
)


def print_banner(title: str) -> None:
    print()
    print(BANNER)
    print(title)
    print(DISCLAIMER)
    print(BANNER)


def build_issues() -> List[IssueEvidence]:
    """
    Three identified issues.  ``issue_labels`` and the probabilities stand in for
    what the issue-identification stage would emit; this stage only consumes them.
    """
    return [
        IssueEvidence(
            document_id="issue-fees",
            post_id="post-hostel-fees",
            comment_id="c-fees-1",
            original_text="The hostel mess charges 9000 a month which is way too high, "
            "and the rooms are barely cleaned <EMOJI_SAD>",
            normalized_text="the hostel mess charges 9000 a month which is way too high "
            "and the rooms are barely cleaned",
            issue_labels=["fees", "hostel"],
            issue_probabilities={"fees": 0.93, "hostel": 0.61},
            language="en",
            script="Latin",
        ),
        IssueEvidence(
            document_id="issue-wifi",
            post_id="post-hostel-wifi",
            comment_id="c-wifi-1",
            original_text="Wifi disconnects every ten minutes in the library and nobody "
            "from the desk helps",
            normalized_text="wifi disconnects every ten minutes in the library and nobody "
            "from the desk helps",
            issue_labels=["network"],
            issue_probabilities={"network": 0.88},
            language="en",
            script="Latin",
        ),
        IssueEvidence(
            document_id="issue-bengali",
            post_id="post-hostel-bengali",
            comment_id="c-bn-1",
            original_text="হোস্টেলের মেসের খরচ অনেক বেশি, আমি সামলাতে পারছি না",
            normalized_text="হোস্টেলের মেসের খরচ অনেক বেশি আমি সামলাতে পারছি না",
            issue_labels=["fees"],
            issue_probabilities={"fees": 0.84},
            language="bn",
            script="Bengali",
        ),
    ]


def build_solutions() -> List[SolutionEvidence]:
    """
    Solution-stage output, standing in for real upstream predictions.

    ``c-unrelated`` is the important one: probability 0.97, role ``solution``,
    and a comment that talks about money - in a different thread.  It is exactly
    the kind of pair a cross product would happily link.
    """
    return [
        SolutionEvidence(
            document_id="sol-warden",
            post_id="post-hostel-fees",
            comment_id="c-fees-2",
            parent_comment_id="c-fees-1",
            original_text="Talk to the warden directly about the mess charges, they can "
            "reduce the monthly fee if you show them the receipts",
            normalized_text="talk to the warden directly about the mess charges they can "
            "reduce the monthly fee if you show them the receipts",
            is_solution=True,
            solution_probability=0.94,
            response_type="solution",
            response_probabilities={"solution": 0.94, "advice": 0.04, "question": 0.02},
            language="en",
            script="Latin",
        ),
        SolutionEvidence(
            document_id="sol-scholarship",
            post_id="post-hostel-fees",
            comment_id="c-fees-3",
            parent_comment_id="c-fees-1",
            original_text="Apply for the hardship scholarship before the deadline, it "
            "covers part of the mess charge each month",
            normalized_text="apply for the hardship scholarship before the deadline it "
            "covers part of the mess charge each month",
            is_solution=True,
            solution_probability=0.81,
            response_type="advice",
            response_probabilities={"solution": 0.81, "advice": 0.17, "question": 0.02},
            language="en",
            script="Latin",
        ),
        SolutionEvidence(
            document_id="sol-emoji-only",
            post_id="post-hostel-fees",
            comment_id="c-fees-4",
            original_text="<EMOJI_SAD> <EMOJI_SAD> <EMOJI_PLEADING> <URL>",
            normalized_text="<EMOJI_SAD> <EMOJI_SAD> <EMOJI_PLEADING> <URL>",
            is_solution=True,
            solution_probability=0.55,
            response_type="advice",
            language="en",
            script="Latin",
        ),
        SolutionEvidence(
            document_id="sol-agree",
            post_id="post-hostel-fees",
            comment_id="c-fees-5",
            original_text="I also faced high hostel mess fees last year, it went up twice",
            normalized_text="i also faced high hostel mess fees last year it went up twice",
            is_solution=False,
            solution_probability=0.09,
            response_type="question",
            language="en",
            script="Latin",
        ),
        SolutionEvidence(
            document_id="sol-router",
            post_id="post-hostel-wifi",
            comment_id="c-wifi-2",
            parent_comment_id="c-wifi-1",
            original_text="Restart the router at the desk and reconnect, the wifi drops "
            "because the access point overloads at peak hours",
            normalized_text="restart the router at the desk and reconnect the wifi drops "
            "because the access point overloads at peak hours",
            is_solution=True,
            solution_probability=0.89,
            response_type="solution",
            language="en",
            script="Latin",
        ),
        SolutionEvidence(
            document_id="sol-bn",
            post_id="post-hostel-bengali",
            comment_id="c-bn-2",
            parent_comment_id="c-bn-1",
            original_text="ওয়ার্ডেনের সাথে কথা বলে মেসের খরচ কমানো যাবে",
            normalized_text="ওয়ার্ডেনের সাথে কথা বলে মেসের খরচ কমানো যাবে",
            is_solution=True,
            solution_probability=0.86,
            response_type="solution",
            language="bn",
            script="Bengali",
        ),
        SolutionEvidence(
            document_id="sol-unrelated",
            post_id="post-cooking",
            comment_id="c-cook-1",
            original_text="Reduce the oil and add more garlic, that is the whole solution "
            "to a bland curry",
            normalized_text="reduce the oil and add more garlic that is the whole solution "
            "to a bland curry",
            is_solution=True,
            solution_probability=0.97,
            response_type="solution",
            language="en",
            script="Latin",
        ),
    ]


def show_discussions(issues: List[IssueEvidence], solutions: List[SolutionEvidence]) -> None:
    """The input, grouped by thread, so the scope of the run is visible."""
    print_banner("1. INPUT - three complete synthetic discussions")

    by_post: dict = {}
    for item in list(issues) + list(solutions):
        by_post.setdefault(item.post_id, []).append(item)

    for post_id, items in by_post.items():
        print()
        print(f"  thread {post_id}")
        for item in items:
            if isinstance(item, IssueEvidence):
                labels = ", ".join(item.issue_labels)
                print(
                    f"    ISSUE    {item.document_id:<18} labels=[{labels}] "
                    f"{item.language}/{item.script}"
                )
            else:
                print(
                    f"    SOLUTION {item.document_id:<18} is_solution={str(item.is_solution):<5} "
                    f"p={item.solution_probability:.2f} role={item.response_type}"
                )
            text = item.normalized_text
            if len(text) > 68:
                text = text[:65] + "..."
            print(f"             \"{text}\"")


def show_default_run(issues: List[IssueEvidence], solutions: List[SolutionEvidence]) -> None:
    """The default configuration: thread-scoped, no cross-post matching."""
    print_banner("2. DEFAULT RUN - same-thread candidates only")

    linker = TfidfCosineIssueSolutionLinker()
    result = linker.link(issues, solutions)
    print(f"  model: {result.model_name} {result.model_version}")
    print(f"  link_threshold: {result.link_threshold}")
    print(f"  TF-IDF terms fitted for this run: {result.n_features}")
    print()
    print(result.format_summary())

    print()
    print("  Candidate rejection counts (why nothing else was scored):")
    for reason, count in sorted(result.candidate_rejections.items()):
        print(f"    {reason:<28} {count}")
    print()
    print("  Note sol-unrelated: probability 0.97, role 'solution', and it talks about")
    print("  money - but it is in post-cooking, so it is never scored against the fee")
    print("  issue. Enabling cross-post matching is what would change that, and it is")
    print("  off by default for exactly this reason.")


def show_link_detail(issues: List[IssueEvidence], solutions: List[SolutionEvidence]) -> None:
    """Every score behind the decisions, asserted or not."""
    print_banner("3. EVERY SCORED CANDIDATE - including the ones not asserted")

    result = TfidfCosineIssueSolutionLinker().link(issues, solutions)
    header = (
        f"  {'issue':<14} {'label':<7} {'solution':<16} {'sim':>6} {'p':>6} "
        f"{'thr':>5} {'role':>5} {'score':>7} {'conf':>6}  linked"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for link in result.links:
        print(
            f"  {link.issue_document_id:<14} {link.issue_label:<7} "
            f"{link.solution_document_id:<16} {link.similarity_score:>6.3f} "
            f"{link.solution_probability:>6.2f} {link.thread_relevance:>5.2f} "
            f"{link.response_role_score:>5.2f} {link.link_score:>7.3f} "
            f"{link.confidence:>6.3f}  {str(link.linked):<5} [{link.relation}]"
        )
    print()
    print("  'score' is the weighted blend; 'conf' is distance from the threshold, not")
    print("  a probability. sol-agree is lexically close to the fees issue but is not a")
    print("  solution, so it was never a candidate. sol-emoji-only is a candidate with no")
    print("  usable terms, so its similarity is 0.0 and the flag is set on the link.")


def show_scoring_arithmetic(issues: List[IssueEvidence], solutions: List[SolutionEvidence]) -> None:
    """Recompute one score by hand from the reported components."""
    print_banner("4. SCORING ARITHMETIC - one link recomputed by hand")

    result = TfidfCosineIssueSolutionLinker().link(issues, solutions)
    link = next(
        l for l in result.links
        if l.issue_document_id == "issue-fees" and l.solution_document_id == "sol-warden"
    )
    weights = link.signals["weights"]
    print()
    for name, value in link.score_components.items():
        print(f"  {name:<22} = {value:.6f}  x  weight {weights[name]:.4f}")
    manual = sum(weights[name] * link.score_components[name] for name in weights)
    print(f"  {'':<22}   {'':<9}  {'-' * 21}")
    print(f"  {'link_score':<22} = {manual:.6f}   (reported: {link.link_score:.6f})")
    print()
    print(
        f"  weights sum to {sum(weights.values()):.4f}, so the score is on a 0-1 scale "
        f"and comparable to link_threshold={link.link_threshold}."
    )
    print(f"  linked = {link.link_score} >= {link.link_threshold} -> {link.linked}")


def show_multilabel(issues: List[IssueEvidence], solutions: List[SolutionEvidence]) -> None:
    """One issue, two labels, two independent link claims."""
    print_banner("5. MULTI-LABEL - each label is a separate link claim")

    result = TfidfCosineIssueSolutionLinker().link(issues, solutions)
    print()
    print("  issue-fees carries two labels, so it produces two claims per candidate:")
    for link in result.links_for_issue("issue-fees"):
        print(
            f"    label={link.issue_label:<7} solution={link.solution_document_id:<18} "
            f"score={link.link_score:.3f} linked={link.linked}"
        )
    print()
    print("  The two labels share a score, and that is V1's real limitation, not an")
    print("  oversight: the representation is the issue's text, so the label is metadata")
    print("  about which claim is being made rather than a term in the vector. What the")
    print("  split buys is addressability - a consumer can hold the fees links and the")
    print("  hostel links separately, and the evaluation unit is one claim, not one")
    print("  comment. What it does not buy is a label-specific score: separating 'this")
    print("  answer fixes the fees but says nothing about the rooms' would need a")
    print("  label-aware representation (include_issue_label_in_text, or embeddings),")
    print("  and neither has been measured here.")


def show_cross_post(issues: List[IssueEvidence], solutions: List[SolutionEvidence]) -> None:
    """What changes when the operator explicitly asks for cross-thread matching."""
    print_banner("6. CROSS-POST MATCHING - off by default, shown on request")

    thread_scoped = TfidfCosineIssueSolutionLinker().link(issues, solutions)
    linker = TfidfCosineIssueSolutionLinker(allow_cross_post=True, link_threshold=0.35)
    result = linker.link(issues, solutions)
    print(f"  allow_cross_post=True, link_threshold={result.link_threshold}")
    print(
        f"  candidates scored: {result.n_scored_links} "
        f"(was {thread_scoped.n_scored_links} thread-scoped)"
    )
    print()
    asserted = [l for l in result.links if l.linked and l.relation == "cross_post"]
    for link in result.links:
        if link.relation == "cross_post":
            print(
                f"    {link.issue_document_id:<14} <- {link.solution_document_id:<16} "
                f"sim={link.similarity_score:.3f} score={link.link_score:.3f} "
                f"linked={link.linked}"
            )
    print()
    print("  The cooking comment is now a candidate for every issue in the batch, and at")
    print(f"  this threshold {len(asserted)} of those cross-thread pairs are asserted -")
    print("  including the fees issue being 'solved' by a comment about garlic. That is")
    print("  the cross product, arriving through the front door because the operator")
    print("  opened it. Cross-thread matching multiplies the candidate pool by roughly")
    print("  the number of threads, so it needs its own threshold chosen against")
    print("  annotated data; the default is not automatically appropriate for it, and no")
    print("  such data exists yet.")


def show_threshold_sensitivity(issues: List[IssueEvidence], solutions: List[SolutionEvidence]) -> None:
    """Re-derive the decision at other thresholds without re-running the model."""
    print_banner("7. THRESHOLD SENSITIVITY - one run, re-decided at several thresholds")

    result = TfidfCosineIssueSolutionLinker().link(issues, solutions)
    print()
    print(f"  {'threshold':<11} {'asserted':<9} {'top link'}")
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7):
        links = result.links_at_threshold(threshold)
        asserted = [l for l in links if l.linked]
        top = f"{asserted[0].issue_document_id} <- {asserted[0].solution_document_id}" if asserted else "-"
        print(f"  {threshold:<11} {len(asserted):<9} {top}")
    print()
    print("  The scores never move; only the boolean does. Nothing was re-fitted and")
    print("  nothing was re-scored, so a threshold sweep costs no extra compute.")
    print("  Choosing a real threshold needs an annotated link dataset, which does not")
    print("  exist yet - these numbers show the mechanism, not the right cut-off.")


def show_evaluation_refusal(issues: List[IssueEvidence], solutions: List[SolutionEvidence]) -> None:
    """The stage refuses to report a metric it cannot justify."""
    print_banner("8. EVALUATION - why no precision or recall is printed here")

    result = TfidfCosineIssueSolutionLinker().link(issues, solutions)
    print()
    print("  This project has no human-annotated link dataset. Evaluating a linker needs")
    print("  one: precision and recall are questions about relations a human judged, and")
    print("  no test in this repository can answer them.")
    print()
    print("  The schema for such a file is defined and shipped:")
    from app.services.nlp.issue_solution_linking import DEFAULT_LINK_CSV, LINK_CSV_COLUMNS

    print(f"    path    : {DEFAULT_LINK_CSV}")
    print(f"    columns : {', '.join(LINK_CSV_COLUMNS)}")
    print()
    print("  The evaluator refuses to produce a number without one:")
    try:
        from app.services.nlp.issue_solution_linking import evaluate_linking

        evaluate_linking(result, LinkCorpus())
    except ValueError as exc:
        print(f"    ValueError: {exc}")
    print()
    print("  Annotating those pairs is the next piece of work. Until it is done, any")
    print("  link accuracy figure quoted for this stage would be fiction.")


def show_audit_trail(issues: List[IssueEvidence], solutions: List[SolutionEvidence]) -> None:
    """What one link carries, so a reviewer can check the decision by hand."""
    print_banner("9. AUDIT TRAIL - one link in full")

    result = TfidfCosineIssueSolutionLinker().link(issues, solutions)
    link = next(l for l in result.links if l.solution_document_id == "sol-emoji-only")
    print()
    print(f"  {link.issue_document_id} / {link.issue_label} <- {link.solution_document_id}")
    print(f"    linked              : {link.linked}")
    print(f"    model               : {link.model_name} {link.model_version}")
    print(f"    score / threshold   : {link.link_score} / {link.link_threshold}")
    print(f"    confidence          : {link.confidence} (boundary distance, not a probability)")
    print(f"    relation            : {link.relation}")
    print(f"    issue text (echoed) : {link.issue_text[:60]}")
    print(f"    solution text       : {link.solution_text[:60]}")
    print(f"    issue_text_used     : {link.signals['issue_text_used'][:60]!r}")
    print(f"    solution_text_used  : {link.signals['solution_text_used']!r}")
    print(f"    zero vector (issue) : {link.signals['issue_zero_vector']}")
    print(f"    zero vector (soln)  : {link.signals['solution_zero_vector']}")
    if "zero_vector_warning" in link.signals:
        print(f"    warning             : {link.signals['zero_vector_warning']}")
    print()
    print("  The echoed text keeps the placeholders upstream produced; the *_text_used")
    print("  fields show what the vectoriser actually consumed. Neither the evidence")
    print("  objects nor the echoed text were modified to get here.")


def show_zero_vector_behaviour(issues: List[IssueEvidence], solutions: List[SolutionEvidence]) -> None:
    """Placeholder-only text degrades visibly rather than silently."""
    print_banner("10. DEGRADED INPUT - reported, not swallowed")

    result = TfidfCosineIssueSolutionLinker().link(issues, solutions)
    print()
    print(f"  zero-vector issues   : {result.n_zero_vector_issues}")
    print(f"  zero-vector solutions: {result.n_zero_vector_solutions}")
    print(f"  TF-IDF terms         : {result.n_features}")
    print()
    print("  Run warnings:")
    for warning in result.warnings:
        print(f"    ! {warning}")
    print()
    print("  sol-emoji-only is a genuine empty representation: <EMOJI_...> placeholders")
    print("  are the whole comment. It is scored (it is a candidate) and its similarity")
    print("  is 0.0 rather than NaN, and the link says why. Silently treating it as a")
    print("  match, or crashing the run, would both be worse.")


def show_input_immutability(issues: List[IssueEvidence], solutions: List[SolutionEvidence]) -> None:
    """Evidence objects are never rewritten by linking."""
    print_banner("11. INPUT IMMUTABILITY - the linker reads, it does not rewrite")

    before_issue = issues[0].model_dump()
    before_solution = solutions[0].model_dump()
    TfidfCosineIssueSolutionLinker().link(issues, solutions)
    same = issues[0].model_dump() == before_issue and solutions[0].model_dump() == before_solution
    print()
    print(f"  evidence objects unchanged after a full run: {same}")
    print()
    print("  Placeholder stripping, label handling and score rounding all happen on")
    print("  this stage's own copies. An upstream stage's data is read-only input, so")
    print("  linking can be re-run, re-thresholded or extended without a hidden")
    print("  dependency on the order things happened to be processed in.")


def main() -> int:
    issues = build_issues()
    solutions = build_solutions()

    show_discussions(issues, solutions)
    show_default_run(issues, solutions)
    show_link_detail(issues, solutions)
    show_scoring_arithmetic(issues, solutions)
    show_multilabel(issues, solutions)
    show_cross_post(issues, solutions)
    show_threshold_sensitivity(issues, solutions)
    show_evaluation_refusal(issues, solutions)
    show_audit_trail(issues, solutions)
    show_zero_vector_behaviour(issues, solutions)
    show_input_immutability(issues, solutions)

    print()
    print(BANNER)
    print("END OF DEMO")
    print(
        "Nothing above is a performance measurement. The stage runs, scopes its\n"
        "candidates to a thread, scores what survives, and reports the rest."
    )
    print(BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
