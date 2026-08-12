"""Aggregate: dedup that keeps provenance, scores that never combine, fan-out that survives a failure.

The tests search-task-02 owes, plus the two regressions its rulings name
explicitly: a mixed corpus must expose no combined score value anywhere
(R3), and reciprocal-rank fusion must read the rank a candidate held in its
*own* call rather than one reconstructed after grouping.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from threetears.search.aggregate import (
    FUSION_RRF_K,
    RRF_SCORE_NAME,
    RRF_STAGE_SOURCE,
    aggregate,
)
from threetears.search.contracts import (
    PRODUCER_API_PROVIDER,
    PRODUCER_MODEL_MEDIATED,
    SCALE_RANK,
    SCALE_UNBOUNDED,
    SCALE_UNIT_INTERVAL,
    Candidate,
    CandidateSet,
    ContentSlot,
    CriterionDisposition,
    Locator,
    Provenance,
    RateLimited,
    ScoreEntry,
    Spend,
)

RETRIEVED_AT = datetime(2026, 8, 12, 9, 0, 0, tzinfo=UTC)


def _candidate(
    url: str,
    *,
    provider: str,
    query: str = "capybara",
    score: ScoreEntry | None = None,
    title: str | None = None,
    content: ContentSlot | None = None,
    producer: str = PRODUCER_API_PROVIDER,
) -> Candidate:
    """Build a candidate keyed on ``url`` from a named provider instance."""
    return Candidate(
        identity=url,
        locators=(Locator(url=url),),
        provenance=Provenance(
            query=query,
            provider_instance=provider,
            retrieved_at=RETRIEVED_AT,
            producer=producer,
        ),
        title=title,
        scores=() if score is None else (score,),
        content=content,
    )


def _searxng_score(value: float) -> ScoreEntry:
    """A SearXNG-shaped weight: unbounded above."""
    return ScoreEntry.provider_native(
        name="relevance", value=value, scale=SCALE_UNBOUNDED, provider_instance="searxng-main"
    )


def _tavily_score(value: float) -> ScoreEntry:
    """A Tavily-shaped relevance: [0, 1]."""
    return ScoreEntry.provider_native(
        name="relevance", value=value, scale=SCALE_UNIT_INTERVAL, provider_instance="tavily-main"
    )


def test_same_url_from_two_providers_becomes_one_entry_with_both_provenances() -> None:
    """R1: merge is a view -- neither provenance is discarded."""
    corpus = aggregate(
        [
            CandidateSet(candidates=(_candidate("https://a", provider="searxng-main"),)),
            CandidateSet(candidates=(_candidate("https://a", provider="tavily-main"),)),
        ]
    )

    assert len(corpus.entries) == 1
    entry = corpus.entries[0]
    assert len(entry.contributions) == 2
    assert [p.provider_instance for p in entry.provenances] == ["searxng-main", "tavily-main"]


def test_merged_scores_stay_distinct_and_are_never_combined() -> None:
    """R3/D1: an unbounded 9.0 beside a [0,1] 0.8 produces no third number.

    The regression that matters: averaging them yields 4.9, a plausible
    figure that means nothing and would silently order a mixed corpus.
    """
    corpus = aggregate(
        [
            CandidateSet(candidates=(_candidate("https://a", provider="searxng-main", score=_searxng_score(9.0)),)),
            CandidateSet(candidates=(_candidate("https://a", provider="tavily-main", score=_tavily_score(0.8)),)),
        ]
    )

    scores = corpus.entries[0].scores
    assert [s.value for s in scores] == [9.0, 0.8]
    assert {s.source for s in scores} == {"searxng-main", "tavily-main"}
    assert all(not s.comparable for s in scores)
    assert {s.scale for s in scores} == {SCALE_UNBOUNDED, SCALE_UNIT_INTERVAL}


def test_a_failed_call_contributes_spend_and_a_notice_but_never_poisons_siblings() -> None:
    """R7/SR-H3: one failure never takes down the fan-out."""
    failure = RateLimited("provider returned 429", spend=Spend(money=Decimal("0.01"), calls=1))
    corpus = aggregate(
        [
            CandidateSet(candidates=(_candidate("https://a", provider="searxng-main"),), spend=Spend(calls=1)),
            failure,
            CandidateSet(candidates=(_candidate("https://b", provider="tavily-main"),), spend=Spend(calls=1)),
        ]
    )

    assert {e.identity for e in corpus.entries} == {"https://a", "https://b"}
    assert corpus.spend.calls == 3
    assert corpus.spend.money == Decimal("0.01")
    assert any(RateLimited.failure_class in notice for notice in corpus.notices)


def test_the_failure_notice_names_the_wire_stable_class_not_the_python_type() -> None:
    """A notice an operator greps for must survive a class rename."""
    failure = RateLimited("provider returned 429", spend=Spend())
    corpus = aggregate([failure])

    assert corpus.notices[0].startswith(f"call failed ({RateLimited.failure_class}")


def test_spend_rolls_up_across_calls_that_bought_nothing() -> None:
    """D4: budget follows the bill, and the bill does not care what it bought."""
    corpus = aggregate(
        [
            CandidateSet(spend=Spend(money=Decimal("0.02"), calls=1)),
            CandidateSet(spend=Spend(money=Decimal("0.03"), calls=1)),
        ]
    )

    assert corpus.entries == ()
    assert corpus.spend.money == Decimal("0.05")
    assert corpus.spend.calls == 2


def test_dispositions_report_the_weakest_answer_and_name_the_divergence() -> None:
    """R8/P8: a filtered corpus that is not filtered must not read as filtered."""
    corpus = aggregate(
        [
            CandidateSet(dispositions=(CriterionDisposition(criterion_key="time-range", disposition="pushdown"),)),
            CandidateSet(dispositions=(CriterionDisposition(criterion_key="time-range", disposition="unsatisfied"),)),
        ]
    )

    assert len(corpus.dispositions) == 1
    rolled = corpus.dispositions[0]
    assert rolled.disposition == "unsatisfied"
    assert rolled.detail is not None
    assert "disagreed" in rolled.detail


def test_agreeing_dispositions_keep_their_detail_untouched() -> None:
    """No divergence means no divergence note -- the detail is the adapter's."""
    corpus = aggregate(
        [
            CandidateSet(
                dispositions=(
                    CriterionDisposition(criterion_key="language", disposition="pushdown", detail="mapped to lang"),
                )
            ),
            CandidateSet(dispositions=(CriterionDisposition(criterion_key="language", disposition="pushdown"),)),
        ]
    )

    assert corpus.dispositions[0].detail == "mapped to lang"


def test_content_survives_the_merge_so_a_mixed_corpus_never_refetches() -> None:
    """SR-A2: Tavily bought the text; a SearXNG sibling must not lose it."""
    slot = ContentSlot(text="capybaras range widely", origin="provider-response")
    corpus = aggregate(
        [
            CandidateSet(candidates=(_candidate("https://a", provider="searxng-main"),)),
            CandidateSet(candidates=(_candidate("https://a", provider="tavily-main", content=slot),)),
        ]
    )

    assert corpus.entries[0].content == slot


def test_identity_is_not_normalised() -> None:
    """R2: under-merging costs a duplicate; over-merging destroys a result."""
    corpus = aggregate(
        [
            CandidateSet(
                candidates=(
                    _candidate("https://a/page", provider="searxng-main"),
                    _candidate("https://a/page?utm_source=x", provider="searxng-main"),
                )
            )
        ]
    )

    assert len(corpus.entries) == 2


def test_a_caller_supplied_key_can_normalise_when_it_accepts_the_risk() -> None:
    """The lever exists; the default declines to pull it."""
    corpus = aggregate(
        [
            CandidateSet(
                candidates=(
                    _candidate("https://a/page", provider="searxng-main"),
                    _candidate("https://a/page?utm_source=x", provider="searxng-main"),
                )
            )
        ],
        key=lambda c: c.identity.split("?", 1)[0],
    )

    assert len(corpus.entries) == 1
    assert len(corpus.entries[0].contributions) == 2


def test_producer_candidates_enter_at_aggregate_without_impersonating_a_provider() -> None:
    """D3: the producer seam, and provenance keeping the classes distinct."""
    produced = _candidate("work:vermeer/milkmaid", provider="curation-engine", producer=PRODUCER_MODEL_MEDIATED)
    corpus = aggregate(
        [CandidateSet(candidates=(_candidate("https://a", provider="searxng-main"),))],
        extra_candidates=[produced],
    )

    by_identity = {e.identity: e for e in corpus.entries}
    assert by_identity["work:vermeer/milkmaid"].contributions[0].provenance.producer == PRODUCER_MODEL_MEDIATED
    assert by_identity["https://a"].contributions[0].provenance.producer == PRODUCER_API_PROVIDER


# -- fusion -----------------------------------------------------------------


def test_no_fusion_by_default() -> None:
    """§3.4: MAY implement, MUST NOT require."""
    corpus = aggregate([CandidateSet(candidates=(_candidate("https://a", provider="searxng-main"),))])

    assert corpus.entries[0].derived_scores == ()
    assert all(s.name != RRF_SCORE_NAME for s in corpus.entries[0].scores)


def test_fusion_emits_one_derived_comparable_rank_entry_sourced_to_the_stage() -> None:
    """R4: the one place a comparable score is minted, and it names a stage."""
    corpus = aggregate(
        [CandidateSet(candidates=(_candidate("https://a", provider="searxng-main"),))],
        fuse=True,
    )

    derived = corpus.entries[0].derived_scores
    assert len(derived) == 1
    assert derived[0].name == RRF_SCORE_NAME
    assert derived[0].scale == SCALE_RANK
    assert derived[0].source == RRF_STAGE_SOURCE
    assert derived[0].comparable is True
    assert derived[0].value == pytest.approx(1.0 / (FUSION_RRF_K + 1))


def test_fusion_reads_the_rank_a_candidate_held_in_its_own_call() -> None:
    """The regression grouping would hide: two calls, opposite orders.

    Call A ranks x first and y second; call B ranks them the other way. Both
    entries must therefore score identically. Reconstructing rank from the
    grouped corpus instead assigns both of B's candidates the order they
    happen to appear in *A*, and the two scores come out different.
    """
    call_a = CandidateSet(
        candidates=(
            _candidate("https://x", provider="searxng-main", query="q"),
            _candidate("https://y", provider="searxng-main", query="q"),
        )
    )
    call_b = CandidateSet(
        candidates=(
            _candidate("https://y", provider="tavily-main", query="q"),
            _candidate("https://x", provider="tavily-main", query="q"),
        )
    )

    corpus = aggregate([call_a, call_b], fuse=True)
    scores = {e.identity: e.derived_scores[0].value for e in corpus.entries}

    expected = 1.0 / (FUSION_RRF_K + 1) + 1.0 / (FUSION_RRF_K + 2)
    assert scores["https://x"] == pytest.approx(expected)
    assert scores["https://y"] == pytest.approx(expected)


def test_fusion_ranks_a_repeated_result_higher_than_a_single_appearance() -> None:
    """The property RRF exists for, stated as an assertion rather than assumed."""
    call_a = CandidateSet(
        candidates=(
            _candidate("https://both", provider="searxng-main", query="q"),
            _candidate("https://only-a", provider="searxng-main", query="q"),
        )
    )
    call_b = CandidateSet(candidates=(_candidate("https://both", provider="tavily-main", query="q"),))

    corpus = aggregate([call_a, call_b], fuse=True)
    scores = {e.identity: e.derived_scores[0].value for e in corpus.entries}

    assert scores["https://both"] > scores["https://only-a"]


def test_a_producer_candidate_that_ranked_nowhere_gets_no_fused_score() -> None:
    """Absent and last-place are different claims; fabricating a rank conflates them."""
    produced = _candidate("work:vermeer/milkmaid", provider="curation-engine", producer=PRODUCER_MODEL_MEDIATED)
    corpus = aggregate(
        [CandidateSet(candidates=(_candidate("https://a", provider="searxng-main"),))],
        extra_candidates=[produced],
        fuse=True,
    )

    by_identity = {e.identity: e for e in corpus.entries}
    assert by_identity["work:vermeer/milkmaid"].derived_scores == ()
    assert len(by_identity["https://a"].derived_scores) == 1


def test_a_negative_fusion_constant_is_refused() -> None:
    """A negative k makes a rank's contribution negative or undefined."""
    with pytest.raises(ValueError, match="non-negative"):
        aggregate([CandidateSet()], fuse=True, fusion_k=-1)


def test_fusion_does_not_reorder_the_corpus() -> None:
    """Fusion adds a judgment; ordering by it is Select's business."""
    call_a = CandidateSet(
        candidates=(
            _candidate("https://only-a", provider="searxng-main", query="q"),
            _candidate("https://both", provider="searxng-main", query="q"),
        )
    )
    call_b = CandidateSet(candidates=(_candidate("https://both", provider="tavily-main", query="q"),))

    corpus = aggregate([call_a, call_b], fuse=True)

    assert [e.identity for e in corpus.entries] == ["https://only-a", "https://both"]
