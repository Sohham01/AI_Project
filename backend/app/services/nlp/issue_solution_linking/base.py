"""
Base interfaces, configuration and evidence models for Issue-Solution Linking.

What this stage does
-------------------
Issue identification answers "what problem is being expressed?".  Solution
detection answers "does this comment offer an actionable fix?".  This stage
answers the third, cross-comment question:

    which detected solutions address which identified issues?

It is the *only* stage that puts two documents side by side, and that is exactly
why it is a separate stage.  The first two are per-comment classifiers that can be
trained, scored, swapped and versioned independently.  Linking is a relation over
a *set* of comments, so its failure modes are different: it can be wrong by
matching the wrong comment, by matching a comment in the wrong thread, or by
matching everything to everything.  Those errors do not show up in either
classifier's metrics, which is why they need their own component.

The failure this stage is built to prevent
------------------------------------------
"Do not simply link every solution comment to every issue."

A cross product is the trivial implementation and it is wrong in a way that looks
fine: every link gets a plausible score, the output fills up, and the knowledge
base fills with confident nonsense.  Two rules prevent it here:

1. **Candidate generation is thread-scoped by default.**  A solution is only
   considered for an issue in the same ``post_id``, unless that is explicitly
   disabled.  Cross-post matching is a real requirement at corpus scale and a
   real precision problem; it is a configuration decision, not a default.
2. **Every score is a blend, not a maximum.**  Semantic similarity is one signal
   among several, so a high similarity cannot on its own assert a link and a low
   one cannot veto an otherwise strong match.

What this stage deliberately does NOT answer
--------------------------------------------
- **Is the issue resolved?** A link says a comment *addresses* an issue.  It does
  not say anybody used the comment, that the problem went away, or that the
  poster accepted it.  Resolution detection is a later stage with different
  evidence requirements.
- **Is this issue still open?** Same answer: that needs a link *plus* downstream
  evidence, and it is not modelled here.
- **What should the reader do?** No answer generation, no generated solutions.
  The output is an edge between two comments a human already wrote.

Independence
------------
This package imports nothing from any other NLP stage.  ``IssueEvidence`` and
``SolutionEvidence`` are neutral containers: they define the fields this stage
*needs*, and any upstream stage can be adapted to fill them without either side
knowing the other exists.  Response role, solution probability and issue
probabilities all arrive as plain data - a string, a float, a mapping - so a
renamed or replaced upstream stage is a wiring change, not a breaking import.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class IssueEvidence(BaseModel):
    """
    One identified issue: a comment that carries issue labels, plus the text that
    produced them.

    Why the text is the representation and the label is metadata
    ------------------------------------------------------------
    A label alone is not semantic context.  "fees" is one token; it shares no
    vocabulary with "the hostel mess charges 9000 a month and it is not worth
    it", and a linker that compares labels to solution text will either miss the
    real match or invent one on a shared word.  So the representation is
    :attr:`normalized_text` - the actual text the issue prediction was made from -
    and :attr:`issue_labels` is carried as metadata describing *which* issue the
    text expresses.

    Multi-label is the normal case, not an edge case: "fees are way too high and
    the ROI is questionable" is two issues.  Each label becomes its own link
    candidate against the same text, so the linker can say "this comment solves
    the fees problem" and, separately, "this comment says nothing about ROI" -
    which collapsing to one label would make impossible to say.
    """

    document_id: str

    # -- source traceability -------------------------------------------------
    post_id: Optional[str] = None
    comment_id: Optional[str] = None
    original_text: Optional[str] = None
    #: ALREADY-preprocessed text.  The linker never imports or re-runs
    #: preprocessing; upstream stages own normalisation.
    normalized_text: str = ""

    # -- the identified issue ------------------------------------------------
    #: Every predicted issue label on this document, in the order the upstream
    #: stage produced them.  Each one is linked separately.
    issue_labels: List[str] = Field(default_factory=list)
    #: Per-label probabilities, when the upstream stage produced them.  Used for
    #: provenance and reported on links; never a similarity term.
    issue_probabilities: Dict[str, float] = Field(default_factory=dict)
    #: Optional topic context, echoed for traceability.  Not a similarity term.
    topic_id: Optional[int] = None

    # -- language-analysis metadata (echoed, never branched on) ---------------
    language: str = "unknown"
    script: str = "unknown"
    is_code_mixed: bool = False
    detected_languages: List[str] = Field(default_factory=list)

    @property
    def has_thread(self) -> bool:
        """True when this issue can be placed in a discussion."""
        return bool(self.post_id and str(self.post_id).strip())

    @property
    def has_labels(self) -> bool:
        return bool([label for label in self.issue_labels if str(label).strip()])

    @property
    def comparable_text(self) -> str:
        """
        The text the issue is *represented* by, for similarity purposes.

        Falls back to ``original_text`` when ``normalized_text`` is empty, which
        is what the linker itself does - two different notions of "the issue's
        text" in one package would let a caller read a property that disagrees
        with the score it sits next to.  Falls back at all because an upstream
        stage that fills only ``original_text`` is a wiring gap worth scoring
        over, not a reason to emit a zero vector and a confident-looking link on
        the remaining signals; the linker reports that case as a zero vector only
        when both are empty.
        """
        return self.normalized_text or self.original_text or ""


class SolutionEvidence(BaseModel):
    """
    One solution candidate: a comment the solution stage assessed, plus the text
    that produced the assessment.

    ``is_solution`` and ``solution_probability`` arrive as data.  The linker never
    re-derives them and never imports the stage that produced them; it uses the
    boolean as one signal and the probability as another, because they are not the
    same fact.  A comment asserted as a solution at 0.51 and at 0.99 both arrive
    with ``is_solution=True``, and they should not be scored the same.
    """

    document_id: str

    # -- source traceability -------------------------------------------------
    post_id: Optional[str] = None
    comment_id: Optional[str] = None
    original_text: Optional[str] = None
    #: ALREADY-preprocessed text.  Placeholders are stripped from a *copy* the
    #: model reads; these fields are never modified.
    normalized_text: str = ""

    # -- the solution assessment --------------------------------------------
    #: The upstream stage's boolean.  A ``True`` makes this a candidate outright.
    is_solution: bool = False
    #: The upstream stage's probability.  One of the four weighted terms.
    solution_probability: float = 0.0
    #: Optional response role, as a plain string.  Adjusts evidence weight; never
    #: gates candidacy, and never turns ``advice`` into a solution.
    response_type: Optional[str] = None
    response_probabilities: Dict[str, float] = Field(default_factory=dict)

    # -- optional thread context --------------------------------------------
    #: Set when this comment is a reply.  A direct reply to the issue's comment
    #: is the strongest thread signal available.
    parent_comment_id: Optional[str] = None
    parent_comment_text: Optional[str] = None

    # -- language-analysis metadata (echoed, never branched on) ---------------
    language: str = "unknown"
    script: str = "unknown"
    is_code_mixed: bool = False
    detected_languages: List[str] = Field(default_factory=list)

    @property
    def has_thread(self) -> bool:
        """True when this solution can be placed in a discussion."""
        return bool(self.post_id and str(self.post_id).strip())

    @property
    def comparable_text(self) -> str:
        """
        The text the solution is *represented* by, for similarity purposes.

        Same ``normalized_text`` then ``original_text`` fallback as
        :attr:`IssueEvidence.comparable_text`, for the same reason.
        """
        return self.normalized_text or self.original_text or ""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class LinkModelConfig(BaseModel):
    """
    Configuration for the TF-IDF + cosine baseline linker.

    Every number the linker uses is here.  Nothing is buried in the
    implementation, because a linker's output is a published claim: "this
    community-generated comment answers this issue" needs to be reproducible from
    its configuration, and a reviewer needs to be able to see which knob moved.

    Weights
    -------
    The four weights are combined into a single score.  With
    ``normalise_weights=True`` (the default) they are divided by their sum, so
    ``link_score`` is on a 0-1 scale regardless of how the weights are written::

        link_score = w_sem * similarity
                   + w_prob * solution_probability
                   + w_thread * thread_relevance
                   + w_role * response_role_score

    The defaults are a judgement call, not a measurement: semantic overlap is the
    most informative single signal, detector confidence is second, thread
    placement is a strong prior that should not be able to outvote the text, and
    response role is weak evidence by construction.  See
    docs/issue_solution_linking.md for why each number sits where it does, and for
    what would justify changing it.

    Multilingual defaults mirror the earlier stages and are not negotiable
    conveniences:
    - ``stop_words`` defaults to ``None``.  No Hindi/Bengali/Odia stop-word lists
      are applied, and no text is ever translated.  In Romanized code-mixed text
      those function words carry the code-switch structure, so removing them by
      default is a silent quality loss.
    - ``token_pattern`` keeps every script, so Bengali "আবেদন করুন" and English
      "apply" reach the same vectoriser.
    """

    # --- TF-IDF (fitted per run over the issue + solution texts) -------------
    max_features: Optional[int] = 20000
    min_df: Union[int, float] = 1
    max_df: Union[int, float] = 1.0
    #: Unigrams only, unlike the classifier stages.  An issue and the solution
    #: that answers it almost never share a bigram: a fees complaint and
    #: "ask the warden to lower the mess charges" overlap on individual content
    #: words, and the bigrams each side invents are all distinctive, so they
    #: dominate the vector norm and *lower* the cosine for exactly the pairs
    #: that should match most.  Measured on a five-comment thread, switching
    #: (1, 2) -> (1, 1) raised a related pair's similarity from 0.12 to 0.19.
    #: Bigrams are still available for a measured reason; the default is the one
    #: that scores short, rephrased comment pairs honestly.
    ngram_range: Tuple[int, int] = (1, 1)
    stop_words: Optional[Any] = None
    lowercase: bool = True
    sublinear_tf: bool = True
    token_pattern: str = r"(?u)\b\w+\b"
    norm: str = "l2"
    use_idf: bool = True
    smooth_idf: bool = True

    # --- link score weights --------------------------------------------------
    semantic_similarity_weight: float = 0.50
    solution_probability_weight: float = 0.30
    thread_relevance_weight: float = 0.15
    response_role_weight: float = 0.05
    #: Divide the weights by their sum so the score stays on 0-1.  Turn it off
    #: only if you intend the raw weighted sum, which is no longer comparable to
    #: ``link_threshold``.
    normalise_weights: bool = True

    # --- decision rule -------------------------------------------------------
    #: Only links at or above this score are emitted with ``linked=True``.
    #: Scores are always preserved either way.
    link_threshold: float = 0.5

    # --- candidate generation ------------------------------------------------
    #: A solution becomes a candidate when ``is_solution`` is True OR its
    #: probability reaches this value.  The OR is deliberate: the upstream
    #: boolean is a decision someone already made, and this stage does not
    #: overrule it by re-thresholding.
    candidate_solution_threshold: float = 0.5
    #: Restrict candidates to the same ``post_id``.  This is the rule that stops
    #: the model becoming a cross product.
    require_same_post: bool = True
    #: Allow cross-thread matching.  Off by default; turn it on knowingly.
    allow_cross_post: bool = False
    #: Count a direct reply (``parent_comment_id`` matching the issue's
    #: ``comment_id``) as the strongest thread signal.
    use_parent_context: bool = True
    #: Optional cap on candidates per (issue, label), largest score first.  ``None``
    #: means no cap.  Present for cost control at corpus scale, not for accuracy.
    max_candidates_per_issue: Optional[int] = None
    #: Thread relevance given to a cross-post pair when ``allow_cross_post`` is
    #: on.  Kept low: cross-thread matches are hypotheses, not findings.
    cross_post_thread_relevance: float = 0.0
    #: Weight of the direct-reply signal within the thread term.
    parent_child_thread_relevance: float = 1.0
    parent_child_thread_bonus: float = 0.0

    # --- response-role evidence ---------------------------------------------
    #: Role name -> evidence weight.  ``advice`` is a legitimate candidate and is
    #: scored lower than a direct ``solution``; it is never promoted to one.
    role_scores: Dict[str, float] = Field(
        default_factory=lambda: {"solution": 1.0, "advice": 0.8}
    )
    #: Weight for a role outside ``role_scores``, and for no role at all.  Neutral
    #: rather than zero: an unclassified comment is not a proven non-solution.
    default_role_score: float = 0.5

    # --- issue representation -----------------------------------------------
    #: Append the issue label to the text used for similarity.  Off by default:
    #: a bare label like "fees" is a single token, and weighting it equally with
    #: a full sentence would let a lexical match manufacture a link.
    include_issue_label_in_text: bool = False

    # --- preprocessing-stage placeholders ----------------------------------
    #: Strip ``<URL>`` / ``<MENTION>`` / ``<EMOJI_...>`` from the *model's copy*.
    #: The echoed text is never touched.
    drop_placeholder_tokens: bool = True

    # --- reporting ----------------------------------------------------------
    #: Opt-in only: a wall-clock field would break reproducibility.
    include_timestamp: bool = False


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class IssueSolutionLinking(ABC):
    """
    Abstract interface for Issue-Solution Linking.

    Any implementation (TF-IDF + cosine, a dense bi-encoder, an LLM judge shown
    the two comments) must:

    - accept :class:`IssueEvidence` and :class:`SolutionEvidence` and mutate
      neither;
    - generate its own candidates rather than comparing every solution to every
      issue;
    - return a :class:`~.models.result.LinkingResult` whose every link carries the
      underlying scores *and* the model name and version, whether or not it was
      emitted as ``linked=True``.

    Linking is not a trained model in V1 - the vectoriser is fitted per run over
    the documents being linked - so there is no ``train()``.  A future learned
    reranker would add one behind this same interface.
    """

    @abstractmethod
    def link(
        self,
        issues: Sequence[IssueEvidence],
        solutions: Sequence[SolutionEvidence],
        link_threshold: Optional[float] = None,
    ) -> Any:
        """Link solution candidates to issue labels; return a structured result."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable model identifier, e.g. ``"tfidf_cosine_issue_solution"``."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Explicit model version, e.g. ``"v1"``."""
        ...

    @property
    @abstractmethod
    def config(self) -> LinkModelConfig:
        ...

    @property
    @abstractmethod
    def link_threshold(self) -> float:
        ...

    @property
    @abstractmethod
    def config_dict(self) -> Dict[str, Any]:
        """JSON-serialisable snapshot of every knob that affects the output."""
        ...
