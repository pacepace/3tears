"""Select -- criteria and a cull, and a ranker slot it never fills (search-spec.md §3.6).

Owns local criteria application and the cull. Owns no ranking: §4.14 puts
rerank in ``agent-memory`` and ``3tears-models``, and
:mod:`threetears.search.contracts.ranker` is the seam a consumer fills.

**The cull reads criteria, never a score threshold.** D1 names the failure
directly: *"Select's cull MUST NOT read ``score > 0`` as 'relevant' -- a
``priority: low`` engine scores everything 0."* A caller wanting a threshold
names the score and supplies the bound; there is no implicit one.

**A criterion the provider already pushed down is not re-applied.** The
corpus records how each contributing call answered (SR-B2), and re-filtering
a pushed-down criterion locally is not a harmless double-check: the provider
filtered on data it holds and this layer often does not. A time window the
provider honoured would be re-applied here against ``published_at``, which is
frequently absent, and the second pass would drop results the first correctly
kept. So pushdown wins and Select applies only what was left unsatisfied.

**Local application that cannot see the data drops and says so.** A candidate
with no ``published_at`` is not *known* to fall in a requested window, so it
does not survive a locally-applied one -- keeping it would mean the filter
did not filter. The count is recorded as a notice, because a cull nobody can
see is the defect P8 exists to prevent.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Final
from urllib.parse import urlparse

from threetears.search.contracts.corpus import Corpus, CorpusEntry
from threetears.search.contracts.criteria import (
    CRITERION_CARRIER,
    CRITERION_DOMAINS_EXCLUDE,
    CRITERION_DOMAINS_INCLUDE,
    CRITERION_MAX_RESULTS,
    CRITERION_MIN_RESOLUTION,
    CRITERION_RIGHTS_CLASS,
    CRITERION_TIME_RANGE,
    Criterion,
    CriterionDisposition,
)
from threetears.search.contracts.facets import (
    FACET_HEIGHT,
    FACET_MEDIA_CATEGORY,
    FACET_RIGHTS_STATUS,
    FACET_WIDTH,
)
from threetears.search.contracts.ranker import Ranker
from threetears.search.contracts.shortlist import Shortlist

__all__ = ["LOCALLY_APPLICABLE", "select"]

#: the criteria Select can apply itself, given only what a candidate carries.
#: Anything else is answered ``unsatisfied`` rather than silently ignored --
#: ``language`` is the standing example: nothing on a candidate records it,
#: so Select cannot honour it and says so (SR-B3).
LOCALLY_APPLICABLE: Final[frozenset[str]] = frozenset(
    {
        CRITERION_CARRIER,
        CRITERION_DOMAINS_EXCLUDE,
        CRITERION_DOMAINS_INCLUDE,
        CRITERION_MAX_RESULTS,
        CRITERION_MIN_RESOLUTION,
        CRITERION_RIGHTS_CLASS,
        CRITERION_TIME_RANGE,
    }
)


def select(
    corpus: Corpus,
    *,
    criteria: Sequence[Criterion] = (),
    ranker: Ranker | None = None,
) -> Shortlist:
    """Apply criteria to a corpus, cull it, and order it if a ranker was given.

    P4's acceptance test is structural here rather than tested by convention:
    filtering and ranking never touch, so a consumer supplying its own ranker
    still constrains carrier type, and a consumer wanting the cull pays for
    no ranker at all.

    :param corpus: what Aggregate accumulated
    :ptype corpus: Corpus
    :param criteria: what the caller asked for; each gets one honest answer
    :ptype criteria: Sequence[Criterion]
    :param ranker: a consumer-supplied ranker, or None to leave the order
        unranked and marked as such (SR-L2)
    :ptype ranker: Ranker | None
    :return: the ordered, filtered subset and how each criterion was answered
    :rtype: Shortlist
    :raises ValueError: when a ranker returns a different set of entries than
        it was given -- ordering is its job, culling is not
    """
    pushed_down = {d.criterion_key for d in corpus.dispositions if d.disposition == "pushdown"}
    entries = list(corpus.entries)
    notices = list(corpus.notices)
    dispositions: list[CriterionDisposition] = [d for d in corpus.dispositions if d.criterion_key in pushed_down]
    answered = set(pushed_down)

    cap: int | None = None
    for criterion in criteria:
        if criterion.key in answered:
            continue
        answered.add(criterion.key)
        if criterion.key == CRITERION_MAX_RESULTS:
            cap = int(criterion.value) if isinstance(criterion.value, int) else None
            dispositions.append(_local(criterion.key, "cull applied after ordering"))
            continue
        if criterion.key not in LOCALLY_APPLICABLE:
            dispositions.append(_unsatisfied(criterion))
            continue
        kept, dropped_blind = _apply(criterion, entries)
        entries = kept
        detail = "applied locally"
        if dropped_blind:
            detail = f"applied locally; {dropped_blind} candidate(s) dropped for missing data"
            notices.append(f"{criterion.key}: {dropped_blind} candidate(s) lacked the data to be judged")
        dispositions.append(_local(criterion.key, detail))

    ranked = False
    ranker_name: str | None = None
    if ranker is not None:
        entries = _ranked(ranker, entries)
        ranked = True
        ranker_name = ranker.name

    if cap is not None:
        entries = entries[:cap]

    return Shortlist(
        entries=tuple(entries),
        ranked=ranked,
        ranker=ranker_name,
        dispositions=tuple(dispositions),
        spend=corpus.spend,
        notices=tuple(notices),
    )


def _ranked(ranker: Ranker, entries: list[CorpusEntry]) -> list[CorpusEntry]:
    """Order entries through the slot, checking the ranker only reordered.

    A ranker is consumer code, and one that drops an entry has applied a
    constraint no disposition answers for (SR-B2). Checked rather than
    trusted, and by identity multiset so a legitimate reorder passes and a
    silent cull does not.

    :param ranker: the consumer's ranker
    :ptype ranker: Ranker
    :param entries: the entries to order
    :ptype entries: list[CorpusEntry]
    :return: the reordered entries
    :rtype: list[CorpusEntry]
    :raises ValueError: when the returned set differs from the given one
    """
    ordered = list(ranker.rank(entries))
    if sorted(e.identity for e in ordered) != sorted(e.identity for e in entries):
        raise ValueError(
            f"ranker {ranker.name!r} returned a different set of entries: "
            f"ordering is the slot's job, culling is Select's"
        )
    return ordered


def _local(key: str, detail: str) -> CriterionDisposition:
    """Record a criterion Select applied itself."""
    return CriterionDisposition(criterion_key=key, disposition="local", detail=detail)


def _unsatisfied(criterion: Criterion) -> CriterionDisposition:
    """Record a criterion nothing could honour, naming why (SR-B3)."""
    return CriterionDisposition(
        criterion_key=criterion.key,
        disposition="unsatisfied",
        detail="no provider pushed it down and a candidate carries nothing Select could apply it to",
    )


def _apply(criterion: Criterion, entries: list[CorpusEntry]) -> tuple[list[CorpusEntry], int]:
    """Filter entries by one criterion, counting those it could not judge.

    :param criterion: the criterion to apply
    :ptype criterion: Criterion
    :param entries: the entries to filter
    :ptype entries: list[CorpusEntry]
    :return: the survivors, and how many were dropped for missing data
    :rtype: tuple[list[CorpusEntry], int]
    """
    kept: list[CorpusEntry] = []
    blind = 0
    for entry in entries:
        verdict = _judge(criterion, entry)
        if verdict is None:
            blind += 1
            continue
        if verdict:
            kept.append(entry)
    return kept, blind


def _judge(criterion: Criterion, entry: CorpusEntry) -> bool | None:
    """Answer whether one entry satisfies one criterion.

    Three-valued deliberately: None means *the data to judge this is absent*,
    which is neither a pass nor a fail and is counted separately so the cull
    can say how much it dropped blind.

    :param criterion: the criterion to test
    :ptype criterion: Criterion
    :param entry: the entry to test it against
    :ptype entry: CorpusEntry
    :return: True to keep, False to drop, None when unjudgeable
    :rtype: bool | None
    """
    if criterion.key == CRITERION_CARRIER:
        return _facet_equals(entry, FACET_MEDIA_CATEGORY, criterion.value)
    if criterion.key == CRITERION_RIGHTS_CLASS:
        return _facet_equals(entry, FACET_RIGHTS_STATUS, criterion.value)
    if criterion.key == CRITERION_DOMAINS_INCLUDE:
        return _domain_match(entry, criterion.value, include=True)
    if criterion.key == CRITERION_DOMAINS_EXCLUDE:
        return _domain_match(entry, criterion.value, include=False)
    if criterion.key == CRITERION_TIME_RANGE:
        return _within_window(entry, criterion.value)
    if criterion.key == CRITERION_MIN_RESOLUTION:
        return _meets_resolution(entry, criterion.value)
    return None


def _facet_equals(entry: CorpusEntry, facet: str, wanted: object) -> bool | None:
    """Compare a facet across contributions, tolerating absence."""
    values = [c.facets[facet] for c in entry.contributions if facet in c.facets]
    if not values:
        return None
    return any(value == wanted for value in values)


def _domain_match(entry: CorpusEntry, value: object, *, include: bool) -> bool | None:
    """Test an entry's locator hosts against a domain list.

    A host matches a listed domain when it equals it or is a subdomain of
    it, so ``example.org`` covers ``docs.example.org`` and never
    ``notexample.org`` -- a bare substring test would match the latter.
    """
    if not isinstance(value, list):
        return None
    domains = [str(d).lower().lstrip(".") for d in value]
    hosts = {
        (urlparse(locator.url).hostname or "").lower()
        for contribution in entry.contributions
        for locator in contribution.locators
    } - {""}
    if not hosts or not domains:
        return None
    hit = any(host == domain or host.endswith(f".{domain}") for host in hosts for domain in domains)
    return hit if include else not hit


def _within_window(entry: CorpusEntry, value: object) -> bool | None:
    """Test publication time against an absolute window."""
    if not isinstance(value, dict):
        return None
    published = next((c.published_at for c in entry.contributions if c.published_at is not None), None)
    if published is None:
        return None
    start = _parse_bound(value.get("start"))
    end = _parse_bound(value.get("end"))
    if start is not None and published < start:
        return False
    return not (end is not None and published > end)


def _parse_bound(raw: object) -> datetime | None:
    """Read one ISO-8601 window bound, tolerating an absent one."""
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _meets_resolution(entry: CorpusEntry, value: object) -> bool | None:
    """Test pixel dimensions against a minimum."""
    if not isinstance(value, dict):
        return None
    want_width = value.get("width")
    want_height = value.get("height")
    for contribution in entry.contributions:
        width = contribution.facets.get(FACET_WIDTH)
        height = contribution.facets.get(FACET_HEIGHT)
        if not isinstance(width, int) or not isinstance(height, int):
            continue
        wide = not isinstance(want_width, int) or width >= want_width
        tall = not isinstance(want_height, int) or height >= want_height
        if wide and tall:
            return True
        return False
    return None
