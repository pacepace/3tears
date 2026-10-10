"""Measurement windows: WHEN a run was measured, and whether two runs' windows overlap.

:func:`measurement_window` reads a run's window off its results' ``scored_at`` stamps,
:func:`classify_window_pairs` sorts every pair into overlapping and disjoint, and
:func:`measurement_window_disclosure` says so when runs measured over non-overlapping spans are compared.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, NamedTuple

from threetears.evals.schema.base import EvalBaseModel

if TYPE_CHECKING:
    from threetears.evals.schema.models import (
        EvalResult,
    )


class MeasurementWindow(EvalBaseModel):
    """The wall-clock span a run's cells were actually measured over — DERIVED.

    **Deliberately not called a "window" bare**, because that word is already
    taken one level up: ``CampaignWindow`` is a campaign's span, derived from its
    member runs' ``created_at``. This is a different quantity on a different
    basis, and the two disagree by hours on the same data, so they carry different
    names and different derivations rather than one name meaning two things.

    **The basis is every result's ``scored_at``, never the run's
    ``created_at``/``completed_at``.** A run document is saved — stamping
    ``created_at`` — *before* its task is created, and the task then waits on a
    semaphore that admits two jobs at a time, so ``created_at`` is when the run
    was enqueued and not when anything was measured. The gap is not academic: a
    ten-arm sweep whose arms were enqueued within half a minute of each other
    executed across four hours, so on a ``created_at`` basis every arm's span
    contains every other arm's and no pair can ever read as disjoint. A window
    derived that way would be silent on precisely the runs it exists to describe.
    ``scored_at`` is stamped when a cell's result object is built, at the end of
    that cell, which makes min/max over a run's results the span its cells really
    occupied.

    Both bounds are the ISO-8601 UTC strings the models stamp, compared as strings
    throughout: every eval timestamp comes from one stamper that always emits a
    ``+00:00`` offset, so lexicographic order is chronological order. Nothing here
    parses a datetime, which is what keeps a malformed stamp from raising on a
    read surface.
    """

    run_id: str
    #: Earliest ``scored_at`` among the run's results.
    start: str
    #: Latest ``scored_at`` among the run's results.
    end: str


def measurement_window(run_id: str, results: Sequence[EvalResult]) -> MeasurementWindow | None:
    """Derive one run's measurement window from the results it produced.

    Args:
        run_id: The run the window describes.
        results: That run's stored results. Callers filter to one run; nothing
            here checks ``eval_run_id``, so passing another run's results
            produces a window describing a population that never existed.

    Returns:
        The window, or ``None`` when the run produced no results. ``None`` is an
        honest absence and never a zero-length span at some arbitrary instant: a
        run that measured nothing did not measure it "at" a time, and a synthetic
        point would compare against other windows as though it had.
    """
    stamps = [r.scored_at for r in results if r.scored_at]
    if not stamps:
        return None
    return MeasurementWindow(run_id=run_id, start=min(stamps), end=max(stamps))


class WindowGap(NamedTuple):
    """One pair of runs whose measurement spans do not overlap, and how far apart they were.

    The pair is oriented — ``earlier`` finished before ``later`` began — so the gap
    is a forward duration rather than a signed difference a reader has to interpret.
    """

    earlier: MeasurementWindow
    later: MeasurementWindow
    #: Seconds between ``earlier.end`` and ``later.start``, or ``None`` when the
    #: recorded stamps cannot be read as instants. See :func:`_gap_seconds`.
    seconds: float | None


def _gap_seconds(earlier_end: str, later_start: str) -> float | None:
    """How long a run's span sat idle before the next one began, if that is knowable.

    **The lexicographic discipline is unchanged and this does not weaken it.** Ordering
    and the disjointness predicate still compare the ISO-8601 strings directly, so a
    malformed stamp cannot make a comparison raise. What is added is a *guarded*
    parse used for magnitude only: a stamp that will not parse, or a pair that mixes an
    offset-aware stamp with a naive one, yields no duration and the disclosure says so
    rather than failing.

    Args:
        earlier_end: The last ``scored_at`` of the run that finished first.
        later_start: The first ``scored_at`` of the run that began after it.

    Returns:
        The non-negative gap in seconds, or ``None`` when it cannot be computed
        honestly — an unparseable stamp, a naive/aware mix, or a parsed order that
        contradicts the lexicographic one (which means the two stamps are not on a
        shared clock and no duration between them is meaningful).
    """
    try:
        end = datetime.fromisoformat(earlier_end)
        start = datetime.fromisoformat(later_start)
    except TypeError, ValueError:
        # NOSILENT: None IS the report -- format_window_gap renders it as the uncomputable-gap clause
        return None
    try:
        delta = (start - end).total_seconds()
    except TypeError:
        # NOSILENT: One stamp carried a UTC offset and the other did not. Subtracting those is
        # not a duration anybody measured, so it is reported as uncomputable (None renders as
        # the uncomputable-gap clause).
        return None
    return delta if delta >= 0 else None


class OverlappingWindows(NamedTuple):
    """One pair of runs whose measurement spans DID overlap.

    Unoriented, unlike :class:`WindowGap`: there is no earlier and later to name when
    two spans share wall-clock time, and no duration between them to report. It carries
    the two windows and nothing else, because the only claim it supports is that these
    two runs were not measured apart.
    """

    first: MeasurementWindow
    second: MeasurementWindow


class WindowPairs(NamedTuple):
    """Every pair of a run set, split by whether the two spans overlapped.

    Both halves come off one enumeration. The disclosure counts the disjoint pairs
    against the total, and the total includes the overlapping half. Complementing one half
    somewhere else would put the overlap predicate in two places, which is the drift this
    type exists to prevent.
    """

    disjoint: list[WindowGap]
    overlapping: list[OverlappingWindows]

    @property
    def total(self) -> int:
        """How many pairs the run set has.

        Returns:
            The P a partial disclosure counts its disjoint pairs against.
        """
        return len(self.disjoint) + len(self.overlapping)


def classify_window_pairs(windows: Sequence[MeasurementWindow]) -> WindowPairs:
    """Split every pair of these runs by whether their measurement spans overlapped.

    The single derivation behind both the predicate and the prose: the badge, the
    quantifier the sentence states and the magnitudes it names are all read off this one
    enumeration, so they cannot come apart about which pairs are disjoint.

    Only windows that RESOLVED are passed in, and callers must keep it that way. A run
    with no results has no window, and letting its absence contribute would report a
    difference on the strength of what one run could not say — the same rule the
    case-set and attribution comparisons follow.

    **Any pair, not all pairs**, and the prose a caller renders from this must state
    that quantifier and not a stronger one. Three arms where two ran together and the
    third ran the next morning are still a set nothing held fixed across, so requiring
    every pair to be disjoint would report nothing for the common sweep shape of a few
    arms at a time — but printing the existential as "these runs were measured over
    non-overlapping spans" is the falsehood the generator escalated into a false
    headline. Fewer than two windows yields both halves
    empty: one run cannot be measured apart from itself, and zero runs assert nothing.

    Bounds are treated as closed and touching counts as overlap: two runs where one's
    last cell and the other's first share an instant were running against the same
    conditions, and the whole point of the question is whether they were.

    Args:
        windows: The resolved windows of the runs being compared.

    Returns:
        The pairs, split. Both halves are in the order the pairs are enumerated from
        ``windows``, and both are empty for fewer than two windows — one run cannot be
        measured apart from itself, and it cannot overlap itself either.
    """
    # Checked exhaustively over every pair. A sorted single pass is the shape that
    # answers "do ALL of them overlap", which is a different question and the one
    # a comparison does not want asked; over the handful of runs a comparison set
    # holds, the pairwise loop costs nothing and says what it means.
    disjoint: list[WindowGap] = []
    overlapping: list[OverlappingWindows] = []
    for i, a in enumerate(windows):
        for b in windows[i + 1 :]:
            if a.end < b.start:
                earlier, later = a, b
            elif b.end < a.start:
                earlier, later = b, a
            else:
                overlapping.append(OverlappingWindows(first=a, second=b))
                continue
            disjoint.append(WindowGap(earlier=earlier, later=later, seconds=_gap_seconds(earlier.end, later.start)))
    return WindowPairs(disjoint=disjoint, overlapping=overlapping)


def disjoint_window_pairs(windows: Sequence[MeasurementWindow]) -> list[WindowGap]:
    """Every pair of these runs whose measurement spans do not overlap, with its gap.

    The disjoint half of :func:`classify_window_pairs`, which holds the predicate and
    the reasoning. Kept as its own name because the disclosure and the badge ask only
    this question, and reading ``.disjoint`` at every call site would say less.

    Args:
        windows: The resolved windows of the runs being compared.

    Returns:
        One :class:`WindowGap` per non-overlapping pair, in the order the pairs are
        enumerated from ``windows``. Empty when every pair overlaps, and for fewer than
        two windows — one run cannot be measured apart from itself.
    """
    return classify_window_pairs(windows).disjoint


# The distinguishing clause of the disjoint-window disclosure, split out for the
# same reason :data:`DEGRADED_RUN_CLAUSE` is: a test pins the CLAUSE, so the
# sentence around it stays free to be rewritten for a reader.
DISJOINT_WINDOWS_CLAUSE = "measured over non-overlapping spans"


#: Above this many windows the disclosure summarises instead of listing every span.
#:
#: The listing form is the better disclosure and stays the default for the sizes an
#: operator actually reads. It stops being a disclosure at scale: a 27-run group
#: renders every span inline as one unbroken paragraph, and the sentence directing the
#: reader to "read it rather than the badge" becomes advice nobody can take. Four keeps
#: the pairwise and small-sweep cases — the ones where every span is the point —
#: verbatim.
MAX_INLINE_MEASUREMENT_WINDOWS = 4

#: How many non-overlapping pairs the disclosure names before it counts the rest.
#:
#: Six is exactly the pair count of :data:`MAX_INLINE_MEASUREMENT_WINDOWS` runs, so the
#: sizes an operator actually reads never truncate. The cap exists for the other end: a
#: ``full=true`` read of a 22-run campaign has 231 pairs, and a paragraph of them is the
#: same non-disclosure the uncapped span list was.
MAX_RENDERED_WINDOW_GAPS = 6

#: What a gap renders as when the recorded stamps cannot be read as instants. Said
#: rather than omitted: a pair silently missing its magnitude reads as a pair with no
#: gap, which is the opposite of what it means.
UNCOMPUTABLE_GAP_CLAUSE = "gap not computable from the recorded stamps"


def format_window_gap(seconds: float | None) -> str:
    """Render a gap between two measurement spans as a magnitude, in its own units.

    Two significant units and no unit words: ``50m31s``, ``1h04m``, ``5d23h``. The
    compact form is deliberate — this is a magnitude the reader weighs, not a sentence,
    and it sits inside a list of pairs where spelled-out units would bury the numbers.

    **It states the size and never what the size means.** Each disclosure computes a magnitude
    in the measure's own units; deciding whether that magnitude is material is the descriptor's
    threshold to declare and is not this function's.

    Args:
        seconds: The gap, or ``None`` when it could not be computed.

    Returns:
        The magnitude, or :data:`UNCOMPUTABLE_GAP_CLAUSE` when there is none.
    """
    if seconds is None:
        return UNCOMPUTABLE_GAP_CLAUSE
    total = round(seconds)
    days, remainder = divmod(total, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _gap_phrase(gap: WindowGap) -> str:
    """Name one non-overlapping pair and how far apart it was."""
    if gap.seconds is None:
        return f"{gap.earlier.run_id} and {gap.later.run_id}, {UNCOMPUTABLE_GAP_CLAUSE}"
    return f"{gap.earlier.run_id} and {gap.later.run_id}, {format_window_gap(gap.seconds)} apart"


def _widest_gaps_first(pairs: Sequence[WindowGap]) -> list[WindowGap]:
    """Order non-overlapping pairs so the biggest gap is read first.

    Widest first because that is the pair a reader most needs to see, and because
    truncating the tail then drops the least, never the most. Pairs whose gap could not
    be computed sort last — they are an admission rather than a measurement, and putting
    an admission where the widest gap belongs would read as a ranking. Ties break on run
    id so one set of windows always renders one way.
    """
    return sorted(
        pairs,
        key=lambda p: (0 if p.seconds is not None else 1, -(p.seconds or 0.0), p.earlier.run_id, p.later.run_id),
    )


def measurement_window_disclosure(windows: Sequence[MeasurementWindow], *, full: bool = False) -> str | None:
    """The sentence a comparison must carry when its runs did not share a clock.

    Descriptive, never a verdict: it states when each run was measured, which pairs did
    not overlap and by how much, and stops there. No threshold decides how far apart is
    too far, no severity is assigned, and nothing is corrected — how much a gap matters
    depends on what else moved in it, which this surface cannot see and the operator can.

    **It states the quantifier it can defend.** The condition is existential — see
    :func:`disjoint_window_pairs` — so the universal form ("these runs were measured over
    non-overlapping spans") is printed only when every pair really is disjoint. Otherwise
    the sentence counts: *D of the P pairs among these N runs*, with the remainder named
    as overlapping and the closing attribution scoped to the pairs that earned it.
    Rendering the existential as a universal is the defect this exists for — on the
    campaign that motivated the fix, three of six pairs overlapped while the sentence
    denied it, and the generator escalated the denial into a false headline.

    **Each non-overlapping pair carries its gap magnitude**, widest first, because
    a disclosure computes a magnitude in the measure's own
    units. An earlier form deliberately reported bounds and no durations, on the grounds
    that no read surface may parse a datetime; that guarantee is kept — ordering and the
    overlap predicate are still purely lexicographic — and only the magnitude is parsed,
    under a guard that yields :data:`UNCOMPUTABLE_GAP_CLAUSE` rather than raising.

    Above :data:`MAX_INLINE_MEASUREMENT_WINDOWS` the span list is replaced by the group's
    outer bounds and its two extreme windows, and the gap list collapses to the widest
    pair; both say how much they did not name and where to get it.

    Args:
        windows: The resolved windows of the runs being compared.
        full: List every span regardless of count. For surfaces whose consumer
            can collapse the text itself, and for an operator who asked. The pair
            list still caps at :data:`MAX_RENDERED_WINDOW_GAPS`, because pairs grow
            quadratically where spans grow linearly.

    Returns:
        The disclosure, rendered verbatim by every surface, or ``None`` when every
        pair of runs overlaps or there are too few windows to ask.
    """
    pairs = disjoint_window_pairs(windows)
    if not pairs:
        return None
    ordered = sorted(windows, key=lambda w: (w.start, w.end))
    n_runs = len(ordered)
    n_pairs = n_runs * (n_runs - 1) // 2
    inline = full or n_runs <= MAX_INLINE_MEASUREMENT_WINDOWS
    if inline:
        detail = "; ".join(f"{w.run_id} {w.start} to {w.end}" for w in ordered)
    else:
        first, last = ordered[0], ordered[-1]
        # The lower bound is `first.start` by construction — the sort is by start.
        # The upper bound is NOT `last.end` for the same reason: sorting by start
        # says nothing about which member ends last, so a run that began early and
        # overran holds the latest end while sitting first. Taking `last.end` would
        # understate the group's span in exactly that case.
        detail = (
            f"{n_runs} runs, measured between {first.start} "
            f"and {max(w.end for w in ordered)}; earliest {first.run_id} {first.start} to {first.end}; "
            f"latest {last.run_id} {last.start} to {last.end}; "
            f"{n_runs - 2} further span(s) not shown — pass full=true for every span"
        )

    widest_first = _widest_gaps_first(pairs)
    if inline:
        shown = widest_first[:MAX_RENDERED_WINDOW_GAPS]
        gap_detail = "; ".join(_gap_phrase(gap) for gap in shown)
        withheld = len(pairs) - len(shown)
        if withheld:
            gap_detail += f"; +{withheld} further non-overlapping pair(s), all narrower than these"
        gaps = f"Non-overlapping pairs, widest first: {gap_detail}."
    else:
        gaps = f"Widest non-overlapping pair: {_gap_phrase(widest_first[0])}"
        if len(pairs) > 1:
            gaps += f"; {len(pairs) - 1} further non-overlapping pair(s) not named"
        gaps += "."

    if len(pairs) == n_pairs:
        lead = f"These runs were {DISJOINT_WINDOWS_CLAUSE} of wall-clock time — every pair of them ({detail})."
        subject = "a difference between them"
    else:
        lead = (
            f"{len(pairs)} of the {n_pairs} pairs among the {n_runs} runs named here were "
            f"{DISJOINT_WINDOWS_CLAUSE} of wall-clock time; the other {n_pairs - len(pairs)} overlap ({detail})."
        )
        subject = "a difference between the two runs of a non-overlapping pair"
    return (
        f"{lead} {gaps} "
        "Anything that changed on the machine, the providers or the account between those spans "
        f"varies with the runs, so {subject} is not attributable to the runs alone."
    )


__all__ = [
    "classify_window_pairs",
    "disjoint_window_pairs",
    "DISJOINT_WINDOWS_CLAUSE",
    "format_window_gap",
    "MAX_INLINE_MEASUREMENT_WINDOWS",
    "MAX_RENDERED_WINDOW_GAPS",
    "measurement_window",
    "measurement_window_disclosure",
    "MeasurementWindow",
    "OverlappingWindows",
    "UNCOMPUTABLE_GAP_CLAUSE",
    "WindowGap",
    "WindowPairs",
]
