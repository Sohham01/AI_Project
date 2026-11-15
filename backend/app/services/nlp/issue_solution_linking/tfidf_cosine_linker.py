"""
TF-IDF + cosine Issue-Solution Linker.

How a link is made
------------------
1. **Candidate generation** (:mod:`.models.candidate`) decides which triples are
   even scored: solutions worth considering, in the same thread by default.
2. **Representation.**  Every issue *text* and every solution *text* in the run is
   fitted into one TF-IDF vocabulary, then L2-normalised.  Fitting over the run
   rather than a fixed corpus is a deliberate V1 choice: a linker that only ever
   sees today's batch has no stale-index problem and no train/serve skew, at the
   cost of similarities that are only comparable within a run.  Cross-run
   comparison of raw ``similarity_score`` values is therefore meaningless, and
   the docs say so.
3. **Similarity.**  Cosine between the two normalised sparse vectors - a dot
   product, because L2 normalisation makes the denominator 1.  A text with no
   usable terms has a zero vector and scores 0.0 against everything, which is
   reported in ``signals`` rather than hidden as a NaN.
4. **Blending.**  The similarity is one of four weighted terms
   (:class:`~.base.LinkModelConfig`); the total is compared to
   ``link_threshold`` to produce ``linked``.

Why the blend and not a max
---------------------------
``max(similarity, probability, thread)`` looks reasonable and is wrong: it lets
any single strong signal assert a link on its own, so one lucky vocabulary
overlap outvotes a thread mismatch and a low detector probability.  A weighted
sum requires signals to *cooperate*, and it keeps every term visible in the
output, so a surprising link can be traced to the term that caused it.

Why the label is not in the text
--------------------------------
By default the issue's label is metadata, not a search term.  A linker's job is
to decide whether a solution comment addresses an issue *in context*; if the
label token were in the vector, a one-word overlap between label and comment
would manufacture a link no reader would accept.  ``include_issue_label_in_text``
exists for the case where an author has measured that it helps, and it is off.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.services.nlp.issue_solution_linking.base import (
    IssueEvidence,
    IssueSolutionLinking,
    LinkModelConfig,
    SolutionEvidence,
)
from app.services.nlp.issue_solution_linking.models.candidate import (
    CandidateDecision,
    generate_candidates,
    normalised_labels,
)
from app.services.nlp.issue_solution_linking.models.labels import (
    clean_issue_text,
    strip_placeholder_tokens,
)
from app.services.nlp.issue_solution_linking.models.result import (
    LinkPrediction,
    LinkingResult,
    LinkingRunReport,
)

#: Stable model identity.  Both values are reported on every link, asserted or not.
LINK_MODEL_NAME = "tfidf_cosine_issue_solution"
LINK_MODEL_VERSION = "v1"

#: Separator for the composite vector key used when a label is folded into an
#: issue's text.  A control character cannot occur in a document id or a label.
_LABEL_KEY_SEP = "\x1f"


def _library_versions() -> Dict[str, str]:
    """Versions of the two libraries that determine the output, for the report."""
    versions: Dict[str, str] = {}
    try:
        import sklearn

        versions["scikit-learn"] = sklearn.__version__
    except Exception:  # pragma: no cover - provenance must never break a run
        versions["scikit-learn"] = "unavailable"
    try:
        import scipy

        versions["scipy"] = getattr(scipy, "__version__", "unavailable")
    except Exception:  # pragma: no cover
        versions["scipy"] = "unavailable"
    return versions


class TfidfCosineIssueSolutionLinker(IssueSolutionLinking):
    """
    Baseline linker: thread-scoped TF-IDF cosine, blended with detector evidence.

    Deterministic by construction.  The vectoriser is fitted per run, no sampling
    and no randomness is involved anywhere, so the same inputs produce the same
    links, the same scores, and the same order.  That is a hard requirement here
    rather than a nicety: a published "this comment answers that question" edge
    has to be reproducible, and a nondeterministic linker cannot be reviewed.
    """

    def __init__(self, config: Optional[LinkModelConfig] = None, **overrides: Any) -> None:
        """
        Accept a config, keyword overrides, or both - overrides win.

        Keyword overrides exist because the common case is "same model, one knob
        moved" (a threshold sweep, cross-post on for one run), and rebuilding a
        ``LinkModelConfig`` by hand to do that invites typos in field names.
        """
        if config is None:
            self._config = LinkModelConfig(**overrides)
        else:
            if overrides:
                merged = config.model_dump()
                merged.update(overrides)
                self._config = LinkModelConfig(**merged)
            else:
                self._config = config.model_copy()
        self._validate_config(self._config)

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return LINK_MODEL_NAME

    @property
    def version(self) -> str:
        return LINK_MODEL_VERSION

    @property
    def config(self) -> LinkModelConfig:
        return self._config

    @property
    def link_threshold(self) -> float:
        return self._config.link_threshold

    @property
    def config_dict(self) -> Dict[str, Any]:
        """
        JSON-serialisable snapshot of every knob that affects the output.

        Includes the role score table and the weights, so a link can be
        recomputed by hand from the report alone.
        """
        snapshot = self._config.model_dump(mode="json")
        snapshot["_model_name"] = LINK_MODEL_NAME
        snapshot["_model_version"] = LINK_MODEL_VERSION
        snapshot["_confidence_definition"] = (
            "confidence = 1 - 2 * abs(link_score - link_threshold), clipped to "
            "[0, 1]. A boundary-distance heuristic, NOT a calibrated probability: "
            "no annotated link dataset exists to calibrate against."
        )
        return snapshot

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def link(
        self,
        issues: Sequence[IssueEvidence],
        solutions: Sequence[SolutionEvidence],
        link_threshold: Optional[float] = None,
    ) -> LinkingResult:
        """
        Link solution candidates to issue labels.

        ``link_threshold`` overrides the configured threshold for this call only;
        scores are unaffected, so sweeping it is free.  The source evidence
        objects are never modified.
        """
        threshold = self.link_threshold if link_threshold is None else link_threshold
        self._validate_threshold(threshold)
        config = self._config

        issues = list(issues or [])
        solutions = list(solutions or [])

        warnings: List[str] = []
        candidates, rejections = generate_candidates(issues, solutions, config)

        labels_by_document: Dict[str, List[str]] = {}
        for issue in issues:
            labels_by_document[issue.document_id] = normalised_labels(issue)

        issue_texts = self._build_issue_texts(issues, labels_by_document, config)
        solution_texts = self._build_solution_texts(solutions, config)

        zero_issue_keys = {key for key, text in issue_texts.items() if not text}
        zero_solution_ids = {
            solution.document_id
            for solution in solutions
            if not solution_texts.get(solution.document_id)
        }
        if zero_issue_keys and candidates:
            warnings.append(
                f"{len(zero_issue_keys)} issue/label representation(s) had no usable "
                "terms after placeholder stripping, so they score 0.0 similarity "
                "against every candidate. Check that normalized_text is populated, "
                "not just original_text."
            )
        if zero_solution_ids and candidates:
            warnings.append(
                f"{len(zero_solution_ids)} solution(s) had no usable terms after "
                "placeholder stripping, so they score 0.0 similarity against every issue."
            )

        vectors, n_features = self._fit_vectors(issue_texts, solution_texts, config)
        if n_features == 0 and candidates:
            warnings.append(
                "TF-IDF produced an empty vocabulary from this batch, so every "
                "similarity is 0.0 and only the non-semantic terms can carry a link. "
                "Link scores are not trustworthy on this run."
            )
        if not candidates:
            warnings.append(
                "no candidate was generated: no solution cleared candidacy, no issue "
                "carried a label, or thread placement rejected every pair. See "
                "candidate_rejections for which."
            )

        issue_by_id = {issue.document_id: issue for issue in issues}
        solution_by_id = {solution.document_id: solution for solution in solutions}
        weights = self._weights()

        links: List[LinkPrediction] = []
        for decision in candidates:
            issue = issue_by_id[decision.issue_document_id]
            solution = solution_by_id[decision.solution_document_id]
            issue_key = self._issue_key(decision.issue_document_id, decision.issue_label, config)
            similarity = self._similarity(vectors, issue_key, decision.solution_document_id)
            link = self._build_link(
                issue=issue,
                solution=solution,
                decision=decision,
                similarity=similarity,
                threshold=threshold,
                weights=weights,
                issue_text_used=issue_texts.get(issue_key, ""),
                solution_text_used=solution_texts.get(solution.document_id, ""),
                issue_zero_vector=issue_key in zero_issue_keys,
                solution_zero_vector=solution.document_id in zero_solution_ids,
                config=config,
            )
            links.append(link)

        links = self._sort_links(links)

        linked_links = [link for link in links if link.linked]
        n_candidate_solutions = len({d.solution_document_id for d in candidates})
        n_issue_labels = sum(len(labels) for labels in labels_by_document.values())

        return LinkingResult(
            model_name=LINK_MODEL_NAME,
            model_version=LINK_MODEL_VERSION,
            link_threshold=float(threshold),
            links=links,
            n_issues=len(issues),
            n_solutions=len(solutions),
            n_solution_candidates=n_candidate_solutions,
            n_issue_labels=n_issue_labels,
            n_scored_links=len(links),
            n_linked=len(linked_links),
            n_linked_issues=len({link.issue_document_id for link in linked_links}),
            n_linked_solutions=len({link.solution_document_id for link in linked_links}),
            n_features=n_features,
            n_zero_vector_issues=len(zero_issue_keys),
            n_zero_vector_solutions=len(zero_solution_ids),
            candidate_rejections=dict(sorted(rejections.items())),
            issue_labels_by_document=labels_by_document,
            config=self.config_dict,
            library_versions=_library_versions(),
            linked_at=self._timestamp(config),
            warnings=warnings,
        )

    def run_report(
        self,
        issues: Sequence[IssueEvidence],
        solutions: Sequence[SolutionEvidence],
        result: Optional[LinkingResult] = None,
        link_threshold: Optional[float] = None,
    ) -> LinkingRunReport:
        """
        Provenance for a run, kept separate from its conclusions.

        Runs the link if no result is supplied, so a report is always obtainable
        from just the inputs.  Counts are read off the result rather than
        recomputed, so a report can never disagree with the result it describes.
        """
        if result is None:
            result = self.link(issues, solutions, link_threshold=link_threshold)
        return LinkingRunReport(
            model_name=LINK_MODEL_NAME,
            model_version=LINK_MODEL_VERSION,
            link_threshold=result.link_threshold,
            n_issue_documents=result.n_issues,
            n_issue_labels=result.n_issue_labels,
            n_solution_documents=result.n_solutions,
            n_solution_candidates=result.n_solution_candidates,
            n_candidates_scored=result.n_scored_links,
            n_links_asserted=result.n_linked,
            n_features=result.n_features,
            n_zero_vector_issues=result.n_zero_vector_issues,
            n_zero_vector_solutions=result.n_zero_vector_solutions,
            candidate_rejections=dict(result.candidate_rejections),
            config=dict(result.config),
            library_versions=dict(result.library_versions),
            linked_at=result.linked_at,
            warnings=list(result.warnings),
        )

    # ------------------------------------------------------------------
    # Text preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _issue_key(issue_document_id: str, issue_label: str, config: LinkModelConfig) -> str:
        """
        Vector key for one (issue, label) pair.

        One key per document by default, because a label folded into the text
        would otherwise give the label copy a second, artificially lowered IDF
        weight for the same terms.  When ``include_issue_label_in_text`` is on,
        the key is per pair - the representations genuinely differ, so they need
        genuinely separate rows.
        """
        if config.include_issue_label_in_text:
            return f"{issue_document_id}{_LABEL_KEY_SEP}{issue_label}"
        return issue_document_id

    def _build_issue_texts(
        self,
        issues: Sequence[IssueEvidence],
        labels_by_document: Dict[str, List[str]],
        config: LinkModelConfig,
    ) -> Dict[str, str]:
        """
        The issue texts TF-IDF sees, keyed by :meth:`_issue_key`.

        Placeholders are stripped from this copy only; the evidence objects and
        the text echoed on every link stay exactly as upstream produced them.
        """
        texts: Dict[str, str] = {}
        for issue in issues:
            body = issue.normalized_text or issue.original_text or ""
            if config.drop_placeholder_tokens:
                body = strip_placeholder_tokens(body)
            else:
                body = str(body or "")
            if not config.include_issue_label_in_text:
                texts[issue.document_id] = body.strip()
                continue
            for label in labels_by_document.get(issue.document_id, []):
                texts[self._issue_key(issue.document_id, label, config)] = clean_issue_text(
                    body,
                    include_label=True,
                    label=label,
                    drop_placeholders=False,  # already stripped above
                )
        return texts

    def _build_solution_texts(
        self, solutions: Sequence[SolutionEvidence], config: LinkModelConfig
    ) -> Dict[str, str]:
        """Mirror of :meth:`_build_issue_texts` for solutions, keyed by document id."""
        texts: Dict[str, str] = {}
        for solution in solutions:
            body = solution.normalized_text or solution.original_text or ""
            if config.drop_placeholder_tokens:
                body = strip_placeholder_tokens(body)
            else:
                body = str(body or "")
            texts[solution.document_id] = body.strip()
        return texts

    # ------------------------------------------------------------------
    # Vectorisation
    # ------------------------------------------------------------------

    def _fit_vectors(
        self,
        issue_texts: Dict[str, str],
        solution_texts: Dict[str, str],
        config: LinkModelConfig,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Fit one TF-IDF vocabulary over the run and return the row vectors.

        Fitted over *all* issue and solution texts, not only the candidates'
        ones, so a term's IDF reflects the batch rather than only the pairs being
        scored.  That also keeps the vocabulary stable while candidates are
        capped or filtered, which makes a capped run comparable to an uncapped
        one instead of silently changing every score.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer

        keys: List[str] = []
        corpus: List[str] = []
        seen: set = set()
        for mapping in (issue_texts, solution_texts):
            for key, text in mapping.items():
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)
                corpus.append(text)

        usable = [text for text in corpus if text and text.strip()]
        if not usable:
            return {}, 0

        vectorizer = TfidfVectorizer(
            max_features=config.max_features,
            min_df=config.min_df,
            max_df=config.max_df,
            ngram_range=tuple(config.ngram_range),
            stop_words=config.stop_words,
            lowercase=config.lowercase,
            sublinear_tf=config.sublinear_tf,
            token_pattern=config.token_pattern,
            norm=config.norm,
            use_idf=config.use_idf,
            smooth_idf=config.smooth_idf,
        )
        try:
            matrix = vectorizer.fit_transform(corpus)
        except ValueError as exc:
            # An empty vocabulary is a legitimate outcome (all-punctuation text,
            # or a min_df/max_df pair that removes every term) and must not crash
            # the pipeline.  Reported through the warning path by the caller.
            if "vocabulary" in str(exc).lower():
                return {}, 0
            raise

        vectors = {key: matrix[index] for index, key in enumerate(keys)}
        return vectors, int(len(vectorizer.vocabulary_))

    @staticmethod
    def _similarity(
        vectors: Dict[str, Any], issue_document_id: str, solution_document_id: str
    ) -> float:
        """
        Cosine similarity between two L2-normalised TF-IDF rows.

        Returns 0.0 for a missing or zero vector instead of raising or returning
        NaN: an empty text is a real state, and a link whose similarity is
        undefined must be visibly weak, not silently strong.
        """
        issue_vector = vectors.get(issue_document_id)
        solution_vector = vectors.get(solution_document_id)
        if issue_vector is None or solution_vector is None:
            return 0.0
        try:
            value = float(issue_vector.dot(solution_vector.T).toarray().ravel()[0])
        except (AttributeError, IndexError, TypeError, ValueError):
            return 0.0
        if value != value:  # NaN guard
            return 0.0
        # Cosine of non-negative TF-IDF is in [0, 1]; clamp for safety so a
        # floating-point overshoot can never assert a link on its own.
        return max(0.0, min(1.0, value))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _weights(self) -> Dict[str, float]:
        """
        The four weights, normalised to sum to 1 when configured.

        The score is only comparable to ``link_threshold`` if it is on a 0-1
        scale, so normalisation is on by default and turning it off is documented
        as making the threshold meaningless.
        """
        config = self._config
        weights = {
            "semantic_similarity": float(config.semantic_similarity_weight),
            "solution_probability": float(config.solution_probability_weight),
            "thread_relevance": float(config.thread_relevance_weight),
            "response_role": float(config.response_role_weight),
        }
        negative = [name for name, value in weights.items() if value < 0.0]
        if negative:
            raise ValueError(f"link weights must be non-negative; got negative: {negative}")
        if config.normalise_weights:
            total = sum(weights.values())
            if total <= 0.0:
                raise ValueError(
                    "the link weights sum to 0, so no score can be computed. At least "
                    "one weight must be positive."
                )
            weights = {name: value / total for name, value in weights.items()}
        return weights

    def _role_score(self, solution: SolutionEvidence) -> float:
        """
        Evidence weight for the solution's response role.

        An unknown or absent role gets ``default_role_score`` rather than 0.0: an
        unclassified comment is not a proven non-solution, and zeroing it would
        penalise the pipeline for a gap in an upstream stage.  The role never
        gates candidacy and never promotes ``advice`` into a solution.
        """
        config = self._config
        role = (solution.response_type or "").strip().lower()
        if not role:
            return float(config.default_role_score)
        for known, value in config.role_scores.items():
            if str(known).strip().lower() == role:
                return float(value)
        return float(config.default_role_score)

    def _confidence(self, score: float, threshold: float) -> float:
        """
        Boundary distance in [0, 1]: 0.0 exactly at the threshold, 1.0 at either
        extreme.

        Deliberately *not* a probability.  A score barely above the threshold is
        the one a human should check first, and this mapping makes that visible
        instead of hiding it behind a number that looks like evidence.
        """
        return max(0.0, min(1.0, 1.0 - 2.0 * abs(score - threshold)))

    def _build_link(
        self,
        issue: IssueEvidence,
        solution: SolutionEvidence,
        decision: CandidateDecision,
        similarity: float,
        threshold: float,
        weights: Dict[str, float],
        issue_text_used: str,
        solution_text_used: str,
        issue_zero_vector: bool,
        solution_zero_vector: bool,
        config: LinkModelConfig,
    ) -> LinkPrediction:
        """Assemble one scored triple into a :class:`LinkPrediction`."""
        probability = max(0.0, min(1.0, float(solution.solution_probability or 0.0)))
        thread_relevance = max(0.0, min(1.0, float(decision.thread_relevance)))
        role_score = self._role_score(solution)

        components = {
            "semantic_similarity": similarity,
            "solution_probability": probability,
            "thread_relevance": thread_relevance,
            "response_role": role_score,
        }
        link_score = sum(weights[name] * value for name, value in components.items())
        link_score = max(0.0, min(1.0, link_score))
        linked = link_score >= threshold

        signals: Dict[str, Any] = {
            "candidacy_basis": decision.candidacy_basis,
            "same_post": bool(decision.same_post),
            "parent_child": bool(decision.parent_child),
            "thread_reason": decision.reason,
            "response_type": solution.response_type or "",
            "response_role_score": role_score,
            "issue_zero_vector": issue_zero_vector,
            "solution_zero_vector": solution_zero_vector,
            "weights": weights,
            "is_code_mixed_issue": bool(issue.is_code_mixed),
            "is_code_mixed_solution": bool(solution.is_code_mixed),
            # What TF-IDF actually saw - placeholders already stripped, and the
            # label appended only when include_issue_label_in_text is on.  The
            # unstripped text stays on the echoed ``issue_text`` /
            # ``solution_text`` fields, so the audit shows both the input and the
            # representation, and a reviewer can see exactly what was stripped.
            "issue_text_used": issue_text_used,
            "solution_text_used": solution_text_used,
        }
        if issue_zero_vector or solution_zero_vector:
            signals["zero_vector_warning"] = (
                "one side had no usable terms; similarity is 0.0 by definition and "
                "this link rests entirely on thread placement and detector confidence"
            )

        return LinkPrediction(
            issue_document_id=issue.document_id,
            issue_label=decision.issue_label,
            solution_document_id=solution.document_id,
            similarity_score=round(similarity, 6),
            solution_probability=round(probability, 6),
            thread_relevance=round(thread_relevance, 6),
            response_role_score=round(role_score, 6),
            score_components={name: round(value, 6) for name, value in components.items()},
            link_score=round(link_score, 6),
            linked=linked,
            link_threshold=float(threshold),
            confidence=round(self._confidence(link_score, threshold), 6),
            model_name=LINK_MODEL_NAME,
            model_version=LINK_MODEL_VERSION,
            signals=signals,
            issue_post_id=issue.post_id,
            issue_comment_id=issue.comment_id,
            issue_text=issue.normalized_text or "",
            issue_original_text=issue.original_text,
            solution_post_id=solution.post_id,
            solution_comment_id=solution.comment_id,
            solution_text=solution.normalized_text or "",
            solution_original_text=solution.original_text,
            language=solution.language or issue.language or "unknown",
            script=solution.script or issue.script or "unknown",
            is_code_mixed=bool(issue.is_code_mixed or solution.is_code_mixed),
        )

    @staticmethod
    def _sort_links(links: List[LinkPrediction]) -> List[LinkPrediction]:
        """
        Total, input-independent ordering: asserted links first by score, then the
        rest by score; ties broken on ids so the order never depends on dict or set
        iteration.

        Rounding to 6 dp before comparing prevents a floating-point tie from
        reordering two otherwise-equal links between runs.
        """
        return sorted(
            links,
            key=lambda link: (
                not link.linked,
                -link.link_score,
                link.issue_document_id,
                link.issue_label,
                link.solution_document_id,
            ),
        )

    # ------------------------------------------------------------------
    # Validation and reporting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _timestamp(config: LinkModelConfig) -> Optional[str]:
        if not config.include_timestamp:
            return None
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _validate_threshold(threshold: Any) -> float:
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(
                f"link_threshold must be a number in [0.0, 1.0], got "
                f"{type(threshold).__name__}"
            )
        value = float(threshold)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"link_threshold must be in [0.0, 1.0], got {value}")
        return value

    @staticmethod
    def _validate_config(config: LinkModelConfig) -> None:
        """
        Reject a configuration that cannot produce a meaningful score.

        The linker would not crash on any of these - it would produce confident
        nonsense, which is worse.  Each check below corresponds to a specific way
        a run can be quietly broken.
        """
        for name, value in (
            ("link_threshold", config.link_threshold),
            ("candidate_solution_threshold", config.candidate_solution_threshold),
            ("cross_post_thread_relevance", config.cross_post_thread_relevance),
            ("parent_child_thread_relevance", config.parent_child_thread_relevance),
            ("default_role_score", config.default_role_score),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0], got {value}")

        for name in (
            "semantic_similarity_weight",
            "solution_probability_weight",
            "thread_relevance_weight",
            "response_role_weight",
        ):
            if float(getattr(config, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(config, name)}")

        if config.normalise_weights:
            total = sum(
                float(value)
                for value in (
                    config.semantic_similarity_weight,
                    config.solution_probability_weight,
                    config.thread_relevance_weight,
                    config.response_role_weight,
                )
            )
            if total <= 0.0:
                raise ValueError(
                    "the four link weights sum to 0, so no score can be computed. "
                    "Give at least one a positive weight."
                )

        if config.max_candidates_per_issue is not None and config.max_candidates_per_issue < 1:
            raise ValueError(
                f"max_candidates_per_issue must be at least 1 or None, got "
                f"{config.max_candidates_per_issue}"
            )

        if isinstance(config.min_df, int) and config.min_df < 1:
            raise ValueError(
                f"min_df as an int is a document count and must be >= 1, got "
                f"{config.min_df}. Use a float in (0, 1] for a fraction."
            )
        if isinstance(config.max_df, float) and not 0.0 < config.max_df <= 1.0:
            raise ValueError(
                f"max_df as a float is a proportion and must be in (0.0, 1.0], got "
                f"{config.max_df}"
            )
        if isinstance(config.max_df, int) and config.max_df < 1:
            raise ValueError(
                f"max_df as an int is a document count and must be >= 1, got "
                f"{config.max_df}"
            )
        if isinstance(config.min_df, float) and isinstance(config.max_df, float):
            if not 0.0 < config.min_df <= config.max_df <= 1.0:
                raise ValueError(
                    f"min_df must be in (0.0, max_df] and max_df in (0.0, 1.0], got "
                    f"min_df={config.min_df}, max_df={config.max_df}"
                )

        low, high = config.ngram_range
        if int(low) < 1 or int(high) < int(low):
            raise ValueError(
                f"ngram_range must satisfy 1 <= low <= high, got {config.ngram_range}"
            )

        for role, value in config.role_scores.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(
                    f"role_scores[{role!r}] must be in [0.0, 1.0], got {value}"
                )
