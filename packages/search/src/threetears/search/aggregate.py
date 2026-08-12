"""Aggregate -- many calls into one corpus (search-spec.md §3.4, D1/D2/D3).

Owns the dedup key, the merge rule, fan-out accounting (SR-H2, SR-H3) and
the spend rollup. Owns no ranking: fusion is offered here because it is a
corpus-wide operation, but it is never applied unless asked for, and
ordering the result is Select's business.

**Dedup keys on identity, verbatim.** ``Candidate.identity`` is by
convention the canonical URL, and normalising it further (stripping
``utm_*``, unifying case, dropping trailing slashes) is refused: the two
failure directions are not symmetric. Under-merging costs a duplicate the
caller can see. Over-merging silently destroys a distinct result nobody
learns was there. ``?page=2`` is not tracking chaff, and hosts that serve
different content at case-differing paths exist. A consumer wanting
normalisation supplies ``key=``, or does it where the identity is minted --
the adapter, which knows the provider's conventions.

**One failing call never poisons its siblings** (SR-H3). A call that raised
contributes its spend and a notice and no candidates; it does not propagate.
This is the shape :mod:`threetears.search.extract` already ruled for
per-candidate outcomes, applied one layer up.

**Fusion is offered, never default** (§3.4: MAY implement, MUST NOT
require). Reciprocal-rank fusion consumes rank *position*, not score value,
which makes it the one fusion correct-by-construction under D1 -- it never
has to pretend SearXNG's unbounded weight and Tavily's [0, 1] relevance
share a scale, because it never reads either.

**Fusion happens during accumulation, and that placement is load-bearing.**
A rank is a candidate's position *within its own call*, and grouping by
identity destroys that order -- two calls returning the same two results in
opposite orders leave nothing in the grouped structure to recover which was
first where. So ranks are read while the ``CandidateSet``s are still
intact, not reconstructed from the corpus afterwards. Producer candidates
(D3) hold no rank in any call and therefore contribute nothing to a fusion,
which is the honest answer rather than a fabricated position.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Final

from threetears.search.contracts.candidate import Candidate, CandidateSet
from threetears.search.contracts.corpus import Corpus, CorpusEntry
from threetears.search.contracts.criteria import CriterionDisposition, Disposition
from threetears.search.contracts.errors import SearchFailure
from threetears.search.contracts.scores import SCALE_RANK, ScoreEntry
from threetears.search.contracts.spend import Spend

__all__ = [
    "FUSION_RRF_K",
    "RRF_SCORE_NAME",
    "RRF_STAGE_SOURCE",
    "aggregate",
]

#: the score name a reciprocal-rank fusion writes.
RRF_SCORE_NAME: Final[str] = "reciprocal-rank-fusion"

#: the ``source`` a fused entry names -- a pipeline stage, never a provider
#: instance. Only a stage that normalised across providers may declare a
#: score comparable (SR-A4), and this is that stage.
RRF_STAGE_SOURCE: Final[str] = "aggregate.reciprocal-rank-fusion"

#: the conventional RRF damping constant, from the original Cormack et al.
#: formulation. Exposed so a caller may tune it, defaulted so nobody must.
FUSION_RRF_K: Final[int] = 60

#: how the weakest-answer rollup orders dispositions: the further down, the
#: weaker. A criterion one provider pushed down and another could not
#: satisfy is reported unsatisfied, because a partial answer that reads as
#: complete is the defect P8 exists to prevent.
_DISPOSITION_STRENGTH: Final[tuple[Disposition, ...]] = (
    "pushdown",
    "local",
    "ignored-unknown",
    "unsatisfied",
)


def _identity(candidate: Candidate) -> str:
    """Key a candidate by its identity, verbatim.

    :param candidate: the candidate to key
    :ptype candidate: Candidate
    :return: the dedup key
    :rtype: str
    """
    return candidate.identity


def aggregate(
    results: Iterable[CandidateSet | SearchFailure],
    *,
    key: Callable[[Candidate], str] = _identity,
    extra_candidates: Iterable[Candidate] = (),
    fuse: bool = False,
    fusion_k: int = FUSION_RRF_K,
) -> Corpus:
    """Accumulate many call results into one corpus.

    Accepts failures alongside successes so a fan-out never has to choose
    between losing a sibling's candidates and losing a failure's spend
    (SR-H3, D4). A failure contributes its spend and a notice.

    ``extra_candidates`` is the producer seam (D3): candidates a producer
    made rather than an adapter fetched. They dedup and merge exactly like
    fetched ones -- provenance keeps the producer classes distinct, so
    nothing here needs to know which is which, and a model-mediated
    candidate can never impersonate an API provider by arriving through
    this door.

    :param results: what the fan-out returned, successes and failures alike
    :ptype results: Iterable[CandidateSet | SearchFailure]
    :param key: how to derive the dedup key; defaults to identity verbatim.
        Supply one to normalise, accepting the over-merge risk
    :ptype key: Callable[[Candidate], str]
    :param extra_candidates: externally produced candidates (D3)
    :ptype extra_candidates: Iterable[Candidate]
    :param fuse: add a reciprocal-rank-fusion score to each entry. Off by
        default: §3.4 offers fusion and MUST NOT require it
    :ptype fuse: bool
    :param fusion_k: RRF damping constant; larger flattens the advantage of
        top ranks. Ignored unless ``fuse``
    :ptype fusion_k: int
    :return: the accumulated corpus, entries in accumulation order
    :rtype: Corpus
    :raises ValueError: when ``fuse`` is set and ``fusion_k`` is negative,
        which would make a rank's contribution negative or undefined
    """
    if fuse and fusion_k < 0:
        raise ValueError(f"reciprocal-rank fusion requires a non-negative k, got {fusion_k}")

    grouped: dict[str, list[Candidate]] = {}
    dispositions: dict[str, list[CriterionDisposition]] = {}
    notices: list[str] = []
    fusion: dict[str, float] = {}
    spend = Spend()

    for result in results:
        if isinstance(result, SearchFailure):
            spend = spend + result.spend
            notices.append(_failure_notice(result))
            continue
        spend = spend + result.spend
        notices.extend(result.notices)
        for disposition in result.dispositions:
            dispositions.setdefault(disposition.criterion_key, []).append(disposition)
        # rank is position within *this* call, read here because grouping
        # by identity below destroys the order it comes from.
        for rank, candidate in enumerate(result.candidates, start=1):
            dedup_key = key(candidate)
            grouped.setdefault(dedup_key, []).append(candidate)
            if fuse:
                fusion[dedup_key] = fusion.get(dedup_key, 0.0) + 1.0 / (fusion_k + rank)

    for candidate in extra_candidates:
        grouped.setdefault(key(candidate), []).append(candidate)

    entries = tuple(
        CorpusEntry(
            identity=identity,
            contributions=tuple(contributions),
            derived_scores=_fused_score(fusion.get(identity)) if fuse else (),
        )
        for identity, contributions in grouped.items()
    )
    return Corpus(
        entries=entries,
        dispositions=_rollup(dispositions),
        spend=spend,
        notices=tuple(notices),
    )


def _fused_score(total: float | None) -> tuple[ScoreEntry, ...]:
    """Build the derived RRF entry, or none when nothing ranked.

    A producer candidate that appeared in no call has no rank anywhere, and
    gets no fused score rather than a fabricated zero -- absent and
    last-place are different claims.

    This is the one place a ``comparable=True`` score is minted, and the
    condition :class:`~threetears.search.contracts.scores.ScoreEntry` states
    for it is met exactly: a pipeline stage normalised across providers.

    :param total: the summed reciprocal ranks, or None when unranked
    :ptype total: float | None
    :return: the derived entry, or an empty tuple
    :rtype: tuple[ScoreEntry, ...]
    """
    if total is None:
        return ()
    return (
        ScoreEntry(
            name=RRF_SCORE_NAME,
            value=total,
            scale=SCALE_RANK,
            source=RRF_STAGE_SOURCE,
            comparable=True,
        ),
    )


def _failure_notice(failure: SearchFailure) -> str:
    """Describe a failed sibling in one line, class first.

    Class first because that is the fact an operator acts on -- the same
    reasoning ``page_finder`` records for its typed failure classes. The
    class is ``failure_class``, the wire-stable name, not the Python type
    name: a notice an operator greps for must not change under a rename.

    :param failure: the failure a sibling call raised
    :ptype failure: SearchFailure
    :return: the notice text
    :rtype: str
    """
    scope = getattr(failure, "scope", None)
    qualified = f"{failure.failure_class}:{scope}" if scope else failure.failure_class
    return f"call failed ({qualified}): {failure}"


def _rollup(dispositions: Mapping[str, Sequence[CriterionDisposition]]) -> tuple[CriterionDisposition, ...]:
    """Reduce per-call dispositions to one weakest honest answer per criterion.

    Where contributors disagreed, ``detail`` names the divergence rather
    than hiding it: a caller told only "unsatisfied" cannot tell that three
    of four providers did satisfy it.

    :param dispositions: per-criterion answers gathered across calls
    :ptype dispositions: Mapping[str, Sequence[CriterionDisposition]]
    :return: one disposition per criterion, in first-seen order
    :rtype: tuple[CriterionDisposition, ...]
    """
    rolled: list[CriterionDisposition] = []
    for criterion_key, answers in dispositions.items():
        weakest = max(answers, key=lambda a: _DISPOSITION_STRENGTH.index(a.disposition))
        distinct = {a.disposition for a in answers}
        detail = weakest.detail
        if len(distinct) > 1:
            spread = ", ".join(sorted(distinct))
            trailer = f". {weakest.detail}" if weakest.detail else ""
            detail = f"contributors disagreed ({spread}); reporting the weakest{trailer}"
        rolled.append(CriterionDisposition(criterion_key=criterion_key, disposition=weakest.disposition, detail=detail))
    return tuple(rolled)
