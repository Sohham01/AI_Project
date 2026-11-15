"""
The deterministic V1 resolver.

What it does
------------
Four weighted terms over one issue's evidence -> a score -> one of three statuses,
with the volume guard applied to confidence rather than to the verdict.  The full
equation is documented on :class:`~.base.ResolutionModelConfig`; this module
implements it and reports every input to it.

Why a plain function and not a trained model
--------------------------------------------
There is no annotated resolution dataset, so there is nothing to fit.  A fitted
model here would be a set of coefficients derived from... the author's intuition,
with a confidence interval invented afterwards and a name implying rigour.  V1 is
the honest version: stated weights, stated thresholds, every number reported.

The one judgement call is the weights themselves, and they are defensible on
their own terms rather than by measurement.  ``solution_density`` and
``strong_link_density`` (0.30 and 0.20) outrank ``best_link_score`` (0.25)
deliberately: *how many* relevant solutions a discussion attracted is harder to
produce by accident than how similar any single comment's words are.  Give
lexical overlap the dominant weight and a thread where one reply happens to
reuse the poster's vocabulary - even a reply proposing the wrong fix - carries the
verdict.  Link strength and solution confidence (0.25 each) still matter; they
express how much to believe the *quality* of the coverage, not merely its
existence.

Determinism
-----------
Same evidence plus same configuration gives the same numbers, always: no sampling,
no random seeds, no wall-clock field unless ``include_timestamp`` is set.  Two
runs that disagree mean the input differed, which is a diagnosable problem.  A
stage whose output drifts between runs cannot be used to answer "did this get
worse?", which is the main thing anyone wants from a resolution metric.

What is deliberately not an input
---------------------------------
**Sentiment.**  A happy poster and a fixed problem are different facts.  A
thread where the author thanks everyone and describes the problem still
unresolved would score 0.0 here and score well on sentiment, and that is correct:
nothing in the evidence says anyone confirmed a fix.  Wiring positive sentiment
into the score would make this stage quietly reward issues people are happy
about, and the resulting correlation with satisfaction would be mistaken for
evidence of resolution.  The same reasoning applies to emotion, comment counts as
a proxy for engagement, and solution *candidates* that were never linked to this
issue.
"""

import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.services.nlp.resolution_detection.base import (
    STATUS_ORDER,
    IssueResolutionEvidence,
    ResolutionDetection,
    ResolutionModelConfig,
    ResolutionStatus,
    TemporalEvidence,
)
from app.services.nlp.resolution_detection.models.evidence import (
    DerivedEvidence,
    derive_evidence,
    validate_evidence,
)
from app.services.nlp.resolution_detection.models.result import (
    ResolutionBatchResult,
    ResolutionResult,
    ResolutionRunReport,
)

MODEL_NAME = "evidence_based_resolution"
MODEL_VERSION = "v1"


class EvidenceBasedResolver(ResolutionDetection):
    """
    Evidence-based resolution detector.

    Usage::

        resolver = EvidenceBasedResolver()
        result = resolver.detect(evidence)            # one issue
        batch = resolver.detect_batch(evidence_list)  # many issues

    Configuration is validated in ``__init__`` rather than lazily, so an
    overlapping threshold is refused at construction time instead of producing
    quietly wrong statuses for every issue in a run.
    """

    def __init__(self, config: Optional[ResolutionModelConfig] = None) -> None:
        self._config = config if config is not None else ResolutionModelConfig()
        self._validate_config(self._config)
        self._weights = self._resolve_weights(self._config)

    # -- interface -------------------------------------------------------------

    @property
    def name(self) -> str:
        return MODEL_NAME

    @property
    def version(self) -> str:
        return MODEL_VERSION

    @property
    def config(self) -> ResolutionModelConfig:
        return self._config

    @property
    def thresholds(self) -> Dict[str, float]:
        return {
            "unresolved_upper": float(self._config.unresolved_upper),
            "partial_upper": float(self._config.partial_upper),
            "resolved_lower": float(self._config.resolved_lower),
        }

    @property
    def config_dict(self) -> Dict[str, Any]:
        """Every knob that affects the output, for reproducibility."""
        data = self._config.model_dump()
        data["weights"] = dict(self._weights)
        data["model_name"] = MODEL_NAME
        data["model_version"] = MODEL_VERSION
        return data

    # -- configuration validation -----------------------------------------------

    @staticmethod
    def _validate_config(config: ResolutionModelConfig) -> None:
        """
        Refuse a configuration whose score or thresholds cannot mean what they say.

        Weights are checked for finiteness and non-negativity, thresholds for
        their ordering.  A zero total weight is refused outright: it would divide
        by zero, and with ``normalise_weights=False`` it would make every score
        0.0 - technically computable, entirely uninformative, and a configuration
        mistake that would otherwise pass unnoticed.
        """
        for name in (
            "solution_density_weight",
            "strong_link_weight",
            "link_strength_weight",
            "solution_confidence_weight",
        ):
            value = float(getattr(config, name))
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
            if value < 0.0:
                raise ValueError(
                    f"{name}={value} is negative. Every term is in [0, 1] and a negative "
                    "weight would invert its evidence."
                )

        if not config.normalise_weights:
            total = sum(float(getattr(config, name)) for name in (
                "solution_density_weight",
                "strong_link_weight",
                "link_strength_weight",
                "solution_confidence_weight",
            ))
            if total <= 0.0:
                raise ValueError(
                    "all weights are zero and normalise_weights is False, so every "
                    "resolution_score would be 0.0 regardless of the evidence. This "
                    "cannot distinguish a resolved issue from an unresolved one."
                )

        for name in ("unresolved_upper", "partial_upper", "resolved_lower"):
            value = float(getattr(config, name))
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"{name} must be a finite number, got {value!r}")
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name}={value} is outside [0.0, 1.0]. Thresholds are compared "
                    "against a normalised score, which is itself in [0, 1]."
                )

        if not config.unresolved_upper <= config.partial_upper <= config.resolved_lower:
            raise ValueError(
                f"thresholds must satisfy unresolved_upper <= partial_upper <= "
                f"resolved_lower, got unresolved_upper={config.unresolved_upper}, "
                f"partial_upper={config.partial_upper}, "
                f"resolved_lower={config.resolved_lower}. The partially-resolved "
                "band needs a lower and an upper edge, and they are the same numbers "
                "as the resolved band's lower edge and the unresolved band's upper "
                "edge. Overlapping thresholds would leave a score able to satisfy two "
                "conditions at once."
            )

        if config.minimum_discussion_volume < 0:
            raise ValueError(
                f"minimum_discussion_volume={config.minimum_discussion_volume} is "
                "negative; a discussion cannot have fewer than zero comments."
            )

        scale = float(config.insufficient_evidence_confidence_scale)
        if not 0.0 <= scale <= 1.0:
            raise ValueError(
                f"insufficient_evidence_confidence_scale={scale} is outside [0.0, 1.0]. "
                "It is a multiplier on confidence, so a value above 1.0 would inflate "
                "confidence on the weakest evidence in the set."
            )

        threshold = float(config.strong_link_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"strong_link_threshold={threshold} is outside [0.0, 1.0]. It is "
                "compared against per-link scores."
            )

    @staticmethod
    def _resolve_weights(config: ResolutionModelConfig) -> Dict[str, float]:
        """The four weights, normalised to sum to 1 when configured."""
        weights = config.weight_map()
        total = sum(weights.values())
        if config.normalise_weights:
            if total <= 0.0:
                raise ValueError(
                    "all weights are zero, so no score can be computed. Set at least "
                    "one weight above zero."
                )
            return {name: value / total for name, value in weights.items()}
        return weights

    # -- scoring ----------------------------------------------------------------

    def score(self, derived: DerivedEvidence) -> float:
        """
        The weighted sum of the four terms, clamped to [0, 1].

        Clamping is a no-op for consistent evidence; it exists so a floating-point
        overshoot at exactly 1.0 cannot push a score over the resolved threshold.
        """
        total = sum(self._weights[name] * value for name, value in derived.as_components().items())
        return min(1.0, max(0.0, total))

    def decide_status(self, score: float) -> ResolutionStatus:
        """
        Apply the two effective cut-points.

        Two cut-points for three bands: the partially-resolved band needs a lower
        and an upper edge, and the upper edge is the resolved band's lower edge.
        """
        if score < self._config.unresolved_upper:
            return ResolutionStatus.UNRESOLVED
        if score < self._config.resolved_lower:
            return ResolutionStatus.PARTIALLY_RESOLVED
        return ResolutionStatus.RESOLVED

    def confidence_for(self, score: float, status: ResolutionStatus) -> float:
        """
        Distance from the nearest status boundary, scaled to [0, 1].

        A score exactly on a boundary gets 0.0: a verdict that a rounding change
        would flip is not a confident verdict, whatever the status.  A score 0.5
        or more from every boundary saturates at 1.0.

        This is a *heuristic*, not a probability.  Nothing here has been
        calibrated against adjudicated outcomes because none exist, and reporting
        it as "0.9 confident" would invite exactly the misreading that
        ``format_summary``'s ``conf=`` label is meant to prevent.
        """
        if status == ResolutionStatus.UNRESOLVED:
            distance = self._config.unresolved_upper - score
        elif status == ResolutionStatus.PARTIALLY_RESOLVED:
            distance = min(
                score - self._config.unresolved_upper,
                self._config.resolved_lower - score,
            )
        else:
            distance = score - self._config.resolved_lower

        # The farthest two cut-points are min(unresolved_upper, 1 - resolved_lower)
        # apart at most; use the actual half-spread so 1.0 is reachable.
        lower = self._config.unresolved_upper
        upper = self._config.resolved_lower
        half_spread = max((upper - lower) / 2.0, 0.5)
        return min(1.0, max(0.0, distance / half_spread))

    # -- single issue -----------------------------------------------------------

    def detect_one(self, evidence: IssueResolutionEvidence) -> ResolutionResult:
        """Assess one issue-label pair."""
        validate_evidence(evidence)
        derived = derive_evidence(evidence)
        score = self.score(derived)
        status = self.decide_status(score)
        confidence = self.confidence_for(score, status)

        warnings: List[str] = []
        sufficient = derived.effective_discussion_volume >= self._config.minimum_discussion_volume
        if not sufficient:
            warnings.append(
                f"insufficient evidence: discussion_volume="
                f"{derived.effective_discussion_volume} is below "
                f"minimum_discussion_volume={self._config.minimum_discussion_volume}. "
                f"The status '{status.value}' is a required state, not a conclusion "
                "this evidence supports. Treat it as provisional."
            )
            confidence *= self._config.insufficient_evidence_confidence_scale

        if derived.linked_solution_count == 0:
            warnings.append(
                "no solutions are linked to this issue and label. This is a statement "
                "about the linker's output for this pair, not proof that no fix exists "
                "in the discussion."
            )
        if (
            derived.solution_candidate_count > derived.linked_solution_count
            and derived.linked_solution_count == 0
        ):
            warnings.append(
                f"{derived.solution_candidate_count} solution candidate(s) were assessed "
                "in the discussion but none was linked to this issue. Candidates are "
                "context, not evidence of resolution, and deliberately do not raise the "
                "score."
            )
        if derived.link_count and derived.best_link_score < self._config.strong_link_threshold:
            warnings.append(
                f"the strongest link is {derived.best_link_score:.3f}, below the "
                f"strong_link_threshold={self._config.strong_link_threshold:.2f}. "
                "Nothing here clears the bar the linker stage itself uses for a strong "
                "link."
            )

        return ResolutionResult(
            issue_id=evidence.issue_id,
            issue_label=evidence.issue_label,
            post_id=evidence.post_id,
            status=status,
            resolution_score=score,
            confidence=confidence,
            # A deep copy, so a result stays true to the evidence it was judged on
            # even if the caller mutates the original record afterwards.  Pydantic
            # reuses nested model instances by default, which would otherwise make
            # a result quietly change its own history.
            evidence=evidence.model_copy(deep=True),
            score_components=derived.as_components(),
            weights=dict(self._weights),
            thresholds=self.thresholds,
            signals=derived.as_signal_summary(),
            solution_density=derived.solution_density,
            strong_solution_density=derived.strong_solution_density,
            best_link_score=derived.best_link_score,
            solution_probability_mean=derived.solution_probability_mean,
            linked_solution_count=derived.linked_solution_count,
            strong_link_count=derived.strong_link_count,
            solution_candidate_count=derived.solution_candidate_count,
            effective_discussion_volume=derived.effective_discussion_volume,
            evidence_sufficient=sufficient,
            warnings=warnings,
            temporal=evidence.temporal,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            config=self.config_dict,
            linked_at=self._timestamp(),
        )

    # -- batch ------------------------------------------------------------------

    def detect_batch(
        self,
        evidence: Sequence[IssueResolutionEvidence],
        evidence_scores: Optional[Dict[str, float]] = None,
    ) -> ResolutionBatchResult:
        """
        Assess many issues, in input order.

        Duplicate ``(issue_id, issue_label)`` pairs keep their **first** occurrence
        and are listed in the run report.  First-wins is arbitrary, but it is
        deterministic, which is what a metric needs: last-wins would make the
        reported status for a repeated pair depend on input ordering, and averaging
        would invent a verdict from two records the pipeline has not reconciled.
        Reporting the duplicates leaves the underlying question - why does the
        pipeline produce two records for one pair? - visible instead of resolving
        it silently here.

        The caller's sequence and every record in it are left untouched.
        """
        records = list(evidence)
        seen: Dict[str, int] = {}
        duplicates: List[str] = []
        ordered: List[IssueResolutionEvidence] = []

        for record in records:
            key = record.pair_key()
            if key in seen:
                if key not in duplicates:
                    duplicates.append(key)
                continue
            seen[key] = len(ordered)
            ordered.append(record)

        results = [self.detect_one(record) for record in ordered]

        # The report is built from the *original* sequence, not the deduplicated
        # one: "3 records in, 2 assessed" is the fact worth reporting, and
        # passing the deduplicated list would make a lossy run look complete.
        report = self._build_report(records, results, duplicates)
        return ResolutionBatchResult(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            results=results,
            thresholds=self.thresholds,
            weights=dict(self._weights),
            config=self.config_dict,
            library_versions=self._library_versions(),
            linked_at=self._timestamp(),
            report=report,
        )

    def detect(
        self,
        evidence: Any,
    ) -> ResolutionBatchResult:
        """
        Assess one issue, or many.

        Accepts either a single :class:`IssueResolutionEvidence` or a sequence of
        them, and always returns a :class:`ResolutionBatchResult` so a caller
        never has to branch on which it passed.

        The single-record case needs its own branch because a Pydantic model is
        iterable - it yields ``(field_name, value)`` tuples, not records.  Left
        unhandled, ``detect(one_record)`` would iterate that model's fields and
        then fail on a tuple, with an error that says nothing about the real
        mistake.
        """
        if isinstance(evidence, IssueResolutionEvidence):
            return self.detect_batch([evidence])
        return self.detect_batch(evidence)

    # -- reporting --------------------------------------------------------------

    def _build_report(
        self,
        records: Sequence[IssueResolutionEvidence],
        results: Sequence[ResolutionResult],
        duplicates: Sequence[str],
    ) -> ResolutionRunReport:
        """Run-level provenance, independent of the verdicts."""
        status_counts = {status: 0 for status in STATUS_ORDER}
        sufficient = 0
        with_temporal = 0
        with_follow_up_signal = 0
        for result in results:
            status_counts[ResolutionStatus(result.status)] += 1
            if result.evidence_sufficient:
                sufficient += 1
            if result.temporal is not None and result.temporal.has_any:
                with_temporal += 1
                if result.temporal.follow_up_resolution_signal:
                    with_follow_up_signal += 1

        mismatches = [
            record.pair_key()
            for record in records
            if self._response_counts_mismatch(record)
        ]

        warnings: List[str] = []
        if not results:
            warnings.append(
                "no issue was assessed. The evidence sequence was empty."
            )
        elif status_counts[ResolutionStatus.RESOLVED] == 0 and sufficient == 0:
            warnings.append(
                f"no verdict in this run rests on sufficient evidence: every one of the "
                f"{len(results)} issue(s) has a discussion below "
                f"minimum_discussion_volume={self._config.minimum_discussion_volume}. "
                "This usually means the guard is set above this corpus's typical thread "
                "size, not that every issue in it is unresolved."
            )
        if duplicates:
            warnings.append(
                f"{len(duplicates)} (issue, label) pair(s) appeared more than once; the "
                "first occurrence was kept. The pipeline is producing duplicate records "
                "for pairs it should be producing one of."
            )
        if mismatches:
            warnings.append(
                f"{len(mismatches)} record(s) have a response_counts histogram that "
                "disagrees with the scalar role counts. Both views are reported; the "
                "scalar fields are the ones this stage would read, and response roles "
                "are not score inputs in V1 anyway."
            )
        if with_follow_up_signal:
            warnings.append(
                f"{with_follow_up_signal} issue(s) carried an explicit follow-up "
                "resolution signal. Temporal evidence is recorded but not scored in V1, "
                "so a confirmed 'that fixed it' does not raise the score. This is the "
                "clearest available candidate for V2 once outcomes are annotated."
            )

        return ResolutionRunReport(
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            n_evidence_records=len(records),
            n_issue_documents=len({record.issue_id for record in records}),
            n_issue_labels=len(results),
            n_resolved=status_counts[ResolutionStatus.RESOLVED],
            n_partially_resolved=status_counts[ResolutionStatus.PARTIALLY_RESOLVED],
            n_unresolved=status_counts[ResolutionStatus.UNRESOLVED],
            n_evidence_sufficient=sufficient,
            n_evidence_insufficient=len(results) - sufficient,
            duplicate_pair_keys=list(duplicates),
            response_count_mismatches=mismatches,
            n_with_temporal_evidence=with_temporal,
            n_with_follow_up_resolution_signal=with_follow_up_signal,
            thresholds=self.thresholds,
            weights=dict(self._weights),
            config=self.config_dict,
            library_versions=self._library_versions(),
            linked_at=self._timestamp(),
            warnings=warnings,
        )

    @staticmethod
    def _response_counts_mismatch(record: IssueResolutionEvidence) -> bool:
        """
        True when the role histogram disagrees with the scalar role counts.

        A warning rather than an error.  The histogram is the field a caller is
        most likely to fill in incompletely, and refusing the record would throw
        away an otherwise valid resolution assessment over a field V1 never
        scores.  Since neither view is a score input, the disagreement cannot
        corrupt the verdict - it only means the record's self-description is
        imprecise, which is worth knowing before V2 uses those fields.
        """
        if not record.response_counts:
            return False
        scalars = {
            "solution": record.solution_count,
            "opinion": record.opinion_count,
            "agreement": record.agreement_count,
            "experience": record.experience_count,
            "advice": record.advice_count,
            "counterargument": record.counterargument_count,
        }
        for role, value in record.response_counts.items():
            if role not in scalars:
                # An unrecognised role name is a vocabulary difference, not a
                # contradiction; the stage must not care what roles exist.
                continue
            if int(value) != int(scalars[role]):
                return True
        return False

    @staticmethod
    def _library_versions() -> Dict[str, str]:
        """
        The versions that could change a number, and nothing else.

        Reported for reproducibility.  Deliberately short: a long version list is
        mostly noise, and none of the other entries would ever influence a
        resolution score.
        """
        import platform

        import pydantic

        return {
            "python": platform.python_version(),
            "pydantic": pydantic.VERSION,
        }

    def _timestamp(self) -> Optional[str]:
        """UTC timestamp, or ``None`` - reproducibility is the default."""
        if not self._config.include_timestamp:
            return None
        return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
