"""The insight ledger as the bundle reads it: which insights stand, which are retracted, which are superseded.

An insight minted by an ARCHIVED analysis is retracted (:func:`retracted_insights`), and a newer insight
restating an older one supersedes it (:func:`superseding_insights`). :func:`_prior_insights` is the capped,
deduplicated list the bundle carries.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

from threetears.evals.kernel.campaign import EvalInsight
from threetears.evals.analysis.bundle.caps import (
    _capped,
    _MAX_PRIOR_INSIGHTS,
)


class InsightStanding(NamedTuple):
    """Where each listed insight's minting analysis stands — the two states a reader must be told.

    Attributes:
        retracted: Insight id → the ARCHIVED analysis that minted it. No reader may present it as
            live, and no generation reads it as prior context.
        orphaned: Insight id → the analysis it names that no longer resolves (a hard delete keeps the
            insights it minted). NOT retracted — nothing says it was shown false, so generations still
            read it — but its provenance can no longer be followed, and a listing must say so.
    """

    retracted: dict[str, str]
    orphaned: dict[str, str]


def insight_standing(
    insights: Iterable[EvalInsight],
    analysis_archived: Callable[[str], bool | None],
) -> InsightStanding:
    """Classify each insight by where its minting analysis stands, asking the store once per analysis.

    An analysis is archived when it was shown false, and the insights it minted are the part of
    it that keeps travelling: they are fed back to every later generation over the same subject
    as prior context, so an archive that left them in place kept the falsehood steering analyses
    while the report that asserted it was marked retired. Deleting each such insight is still the
    intended cleanup; this is what makes the archive itself sufficient to stop the steering, so a
    delete that is forgotten or deferred no longer leaves the generator reading a retracted claim.

    **Derived from the analysis, never copied onto the insight.** The archived flag is the one
    source of truth, so un-archiving an analysis restores its insights with nothing to un-mark, and
    no reader can disagree with another about whether an insight is retracted. Every reader that
    presents insights asks this function — the bundle's ``prior_insights``, and the ledger listing.

    An insight naming no analysis is neither: it never had a back-reference to lose. One naming an
    analysis that no longer resolves is ORPHANED rather than retracted, and the listings disclose it
    from ``orphaned``.

    Args:
        insights: The insights to classify.
        analysis_archived: Whether one analysis is archived, ``None`` when it does not resolve.
            Called at most once per distinct analysis, and only for insights that name one.

    Returns:
        Both maps, each in insight-id order and empty when nothing is in that state.
    """
    standing: dict[str, bool | None] = {}
    retracted: dict[str, str] = {}
    orphaned: dict[str, str] = {}
    for insight in insights:
        source = insight.source_analysis_id
        if not source:
            continue
        if source not in standing:
            standing[source] = analysis_archived(source)
        if standing[source] is True:
            retracted[insight.id] = source
        elif standing[source] is None:
            orphaned[insight.id] = source
    return InsightStanding(dict(sorted(retracted.items())), dict(sorted(orphaned.items())))


def retracted_insights(
    insights: Iterable[EvalInsight],
    analysis_archived: Callable[[str], bool | None],
) -> dict[str, str]:
    """Which of these insights an ARCHIVED analysis minted — :func:`insight_standing`'s ``retracted``.

    Args:
        insights: The insights to classify.
        analysis_archived: Whether one analysis is archived, ``None`` when it does not resolve.

    Returns:
        Insight id → the id of the archived analysis that minted it, in insight-id order.
    """
    return insight_standing(insights, analysis_archived).retracted


def insight_restatement_key(statement: str) -> str:
    """The claim an insight states, as two insights stating it compare — case, spacing and a final period aside.

    The one rule for "these two insights say the same thing", read by the bundle (which carries one insight per
    claim) and by the ledger write (which replaces a live insight a new one restates rather than adding a
    duplicate). Deliberately literal: two sentences that mean the same thing in different words are two keys,
    because deciding they are one claim is a judgement, and a wrong merge here would retire a claim nobody
    restated.

    Args:
        statement: An insight's statement.

    Returns:
        The comparison key.
    """
    return " ".join(statement.casefold().split()).rstrip(".").rstrip()


def superseding_insights(
    minted: Sequence[EvalInsight],
    ledger: Iterable[EvalInsight],
    analysis_archived: Callable[[str], bool | None],
) -> list[EvalInsight]:
    """The insights a generation writes: each one minted, taking the id of the live insight it restates.

    A generation mints one insight per finding that states one, and regenerating over the same evidence
    states the same claims again. Written as new rows, every regeneration grew the ledger by its whole
    output, and every one of those rows rode into the next paid prompt. So a minted insight whose claim
    (:func:`insight_restatement_key`) a LIVE ledger insight of the subject already states is written under
    that insight's id: the store's upsert replaces the old row with the restatement — its statement,
    confidence, evidence and minting analysis now the newer ones — and the ledger keeps its size. That is the
    insight's ``invalidation_trigger``, carried out.

    A RETRACTED insight (its analysis archived, :func:`retracted_insights`) is not live and is never replaced:
    a new analysis stating a claim an archive withdrew mints it afresh, and archiving that new analysis is
    what would withdraw it again. Where the ledger already holds several live insights stating one claim —
    written before this rule — the newest is the one replaced.

    Args:
        minted: The insights a generation returned, in its order.
        ledger: The subject's prior insights.
        analysis_archived: Whether one analysis is archived, ``None`` when it does not resolve.

    Returns:
        The insights to write, in ``minted`` order, each under its own id or the id of the insight it replaces.
        A claim the generation stated twice is written once.
    """
    prior = list(ledger)
    retracted = retracted_insights(prior, analysis_archived)
    live: dict[str, EvalInsight] = {}
    for insight in _sorted_insights([insight for insight in prior if insight.id not in retracted]):
        live.setdefault(insight_restatement_key(insight.statement), insight)
    written: list[EvalInsight] = []
    seen: set[str] = set()
    for insight in minted:
        key = insight_restatement_key(insight.statement)
        if key in seen:
            continue
        seen.add(key)
        replaced = live.get(key)
        written.append(insight if replaced is None else insight.model_copy(update={"id": replaced.id}))
    return written


def _prior_insights(live: list[EvalInsight]) -> tuple[list[EvalInsight], int]:
    """The live insights a bundle carries — the newest per claim, at most the cap — and how many it leaves out.

    Args:
        live: The subject's insights as of the cutoff, less the retracted ones.

    Returns:
        ``(carried, omitted)``: newest first, deterministic for the fingerprint.
    """
    newest: dict[str, EvalInsight] = {}
    for insight in _sorted_insights(live):
        newest.setdefault(insight_restatement_key(insight.statement), insight)
    carried, _beyond_cap = _capped(list(newest.values()), _MAX_PRIOR_INSIGHTS, weight=None)
    return carried, len(live) - len(carried)


def _sorted_insights(insights: list[EvalInsight]) -> list[EvalInsight]:
    """Order insights newest-first with an id tie-break — a stable fingerprint slice.

    ``query_insights`` already returns newest-first, but its equal-``observed_at``
    tie-break is storage's, not ours. The bundle fingerprint (and the prompt-A/B
    invariant it guards) must not depend on that, so we re-sort deterministically
    here: ``(observed_at, id)`` descending keeps newest-first and makes ties total.

    Args:
        insights: The subject's prior insights, in storage order.

    Returns:
        The same insights, deterministically ordered.
    """
    return sorted(insights, key=lambda insight: (insight.observed_at, insight.id), reverse=True)


__all__ = [
    "insight_restatement_key",
    "insight_standing",
    "InsightStanding",
    "retracted_insights",
    "superseding_insights",
]
