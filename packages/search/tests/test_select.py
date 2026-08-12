"""Select: the cull that reads criteria, and the ranker slot that stays empty.

The tests search-task-02 owes for R5 and R6, plus P4's acceptance test stated
as assertions rather than as a design claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from threetears.search.contracts import (
    FACET_MEDIA_CATEGORY,
    SCALE_UNBOUNDED,
    Candidate,
    Corpus,
    CorpusEntry,
    Criterion,
    CriterionDisposition,
    Locator,
    Provenance,
    ScoreEntry,
)
from threetears.search.select import select

RETRIEVED_AT = datetime(2026, 8, 12, 9, 0, 0, tzinfo=UTC)


def _entry(
    url: str,
    *,
    carrier: str | None = None,
    score: float | None = None,
    published_at: datetime | None = None,
) -> CorpusEntry:
    """One corpus entry keyed on ``url``."""
    candidate = Candidate(
        identity=url,
        locators=(Locator(url=url),),
        provenance=Provenance(query="q", provider_instance="searxng-main", retrieved_at=RETRIEVED_AT),
        published_at=published_at,
        scores=()
        if score is None
        else (
            ScoreEntry.provider_native(
                name="relevance", value=score, scale=SCALE_UNBOUNDED, provider_instance="searxng-main"
            ),
        ),
        facets={} if carrier is None else {FACET_MEDIA_CATEGORY: carrier},
    )
    return CorpusEntry(identity=url, contributions=(candidate,))


class _ReversingRanker:
    """A consumer-supplied ranker that only reorders."""

    name = "reversing-test-ranker"

    def rank(self, entries: Sequence[CorpusEntry], /) -> Sequence[CorpusEntry]:
        """Return the same entries, reversed."""
        return list(reversed(entries))


class _CullingRanker:
    """A ranker that oversteps by dropping an entry."""

    name = "culling-test-ranker"

    def rank(self, entries: Sequence[CorpusEntry], /) -> Sequence[CorpusEntry]:
        """Return fewer entries than it was given."""
        return list(entries)[:1]


def test_output_is_unranked_by_default_and_says_so() -> None:
    """R5/SR-L2: absent ranker is a stated fact, not an absence."""
    shortlist = select(Corpus(entries=(_entry("https://a"), _entry("https://b"))))

    assert shortlist.ranked is False
    assert shortlist.ranker is None
    assert [e.identity for e in shortlist.entries] == ["https://a", "https://b"]


def test_a_zero_scoring_engine_survives_the_cull() -> None:
    """R6/D1: a ``priority: low`` engine scores everything 0 and is not irrelevant.

    The named failure: a cull reading ``score > 0`` as "relevant" would empty
    this shortlist entirely.
    """
    corpus = Corpus(entries=(_entry("https://a", score=0.0), _entry("https://b", score=0.0)))

    shortlist = select(corpus)

    assert len(shortlist.entries) == 2


def test_the_cull_applies_max_results() -> None:
    """The one bound Select applies without being told a score."""
    corpus = Corpus(entries=(_entry("https://a"), _entry("https://b"), _entry("https://c")))

    shortlist = select(corpus, criteria=[Criterion.max_results(2)])

    assert [e.identity for e in shortlist.entries] == ["https://a", "https://b"]
    assert shortlist.dispositions[0].disposition == "local"


def test_a_consumer_wanting_the_cull_pays_for_no_ranker() -> None:
    """P4's acceptance test, first half."""
    corpus = Corpus(entries=(_entry("https://a"), _entry("https://b"), _entry("https://c")))

    shortlist = select(corpus, criteria=[Criterion.max_results(1)])

    assert len(shortlist.entries) == 1
    assert shortlist.ranked is False


def test_a_consumer_supplying_a_ranker_can_still_constrain_carrier() -> None:
    """P4's acceptance test, second half: the two never condition each other."""
    corpus = Corpus(
        entries=(
            _entry("https://doc", carrier="document"),
            _entry("https://img", carrier="image"),
            _entry("https://img2", carrier="image"),
        )
    )

    shortlist = select(corpus, criteria=[Criterion.carrier("image")], ranker=_ReversingRanker())

    assert [e.identity for e in shortlist.entries] == ["https://img2", "https://img"]
    assert shortlist.ranked is True
    assert shortlist.ranker == "reversing-test-ranker"


def test_a_ranker_that_culls_is_refused() -> None:
    """Ordering is the slot's job; culling would apply a criterion nothing answered for."""
    corpus = Corpus(entries=(_entry("https://a"), _entry("https://b")))

    with pytest.raises(ValueError, match="different set of entries"):
        select(corpus, ranker=_CullingRanker())


def test_the_cull_runs_after_ranking_so_max_results_keeps_the_best() -> None:
    """A cap applied before ordering would keep an arbitrary subset and then sort it."""
    corpus = Corpus(entries=(_entry("https://a"), _entry("https://b"), _entry("https://c")))

    shortlist = select(corpus, criteria=[Criterion.max_results(2)], ranker=_ReversingRanker())

    assert [e.identity for e in shortlist.entries] == ["https://c", "https://b"]


def test_a_pushed_down_criterion_is_not_reapplied_locally() -> None:
    """The provider filtered on data it holds and this layer does not.

    The regression: re-applying a pushed-down time window against
    ``published_at`` -- routinely absent -- drops results the provider
    correctly kept.
    """
    corpus = Corpus(
        entries=(_entry("https://a"), _entry("https://b")),
        dispositions=(CriterionDisposition(criterion_key="time-range", disposition="pushdown"),),
    )
    window = Criterion.time_range(start=datetime(2026, 1, 1, tzinfo=UTC), end=datetime(2026, 8, 1, tzinfo=UTC))

    shortlist = select(corpus, criteria=[window])

    assert len(shortlist.entries) == 2
    assert shortlist.dispositions[0].disposition == "pushdown"


def test_a_locally_applied_window_drops_undateable_candidates_and_says_how_many() -> None:
    """R6's honesty rule: a cull nobody can see is the defect P8 exists to prevent."""
    corpus = Corpus(
        entries=(
            _entry("https://dated", published_at=datetime(2026, 3, 1, tzinfo=UTC)),
            _entry("https://undated"),
        )
    )
    window = Criterion.time_range(start=datetime(2026, 1, 1, tzinfo=UTC), end=datetime(2026, 8, 1, tzinfo=UTC))

    shortlist = select(corpus, criteria=[window])

    assert [e.identity for e in shortlist.entries] == ["https://dated"]
    assert any("lacked the data" in notice for notice in shortlist.notices)
    assert shortlist.dispositions[0].detail is not None
    assert "missing data" in shortlist.dispositions[0].detail


def test_a_criterion_select_cannot_apply_is_unsatisfied_not_ignored() -> None:
    """SR-B3: an honest no beats a silent drop."""
    shortlist = select(Corpus(entries=(_entry("https://a"),)), criteria=[Criterion.language("en")])

    assert shortlist.dispositions[0].criterion_key == "language"
    assert shortlist.dispositions[0].disposition == "unsatisfied"
    assert len(shortlist.entries) == 1


def test_domain_include_matches_subdomains_but_not_lookalikes() -> None:
    """A bare substring test would keep ``notexample.org``."""
    corpus = Corpus(
        entries=(
            _entry("https://docs.example.org/a"),
            _entry("https://example.org/b"),
            _entry("https://notexample.org/c"),
        )
    )

    shortlist = select(corpus, criteria=[Criterion(key="domains-include", value=["example.org"])])

    assert [e.identity for e in shortlist.entries] == ["https://docs.example.org/a", "https://example.org/b"]


def test_spend_is_carried_forward_untouched() -> None:
    """Select filters candidates and never spend."""
    corpus = Corpus(entries=(_entry("https://a"), _entry("https://b")))

    shortlist = select(corpus, criteria=[Criterion.max_results(1)])

    assert shortlist.spend == corpus.spend
