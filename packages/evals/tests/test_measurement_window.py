"""Measurement windows — when a run's cells were actually measured, and what that discloses.

The subject is a disclosure, not a correction: these tests pin *that* the surface
reports non-overlapping measurement spans and *that* it says nothing about how bad
that is. The basis is deliberately load-bearing and has its own test — a window
derived from the run document's timestamps instead of its results' would be silent
on exactly the runs it exists to describe.
"""

from __future__ import annotations


import pytest

from threetears.evals.analysis import reporting
from threetears.evals.analysis.reporting import (
    DISJOINT_WINDOWS_CLAUSE,
    MAX_INLINE_MEASUREMENT_WINDOWS,
    MeasurementWindow,
    disjoint_window_pairs,
    measurement_window,
    measurement_window_disclosure,
)
from threetears.evals.analysis.lenses.comparison_sets import BADGE_MEASUREMENT_WINDOWS_DISJOINT, compute_comparison_sets
from threetears.evals.analysis.reads import comparison_sets
from threetears.evals.schema import SubjectSnapshot
from threetears.evals.schema.models import EvalResult, EvalRun
from threetears.evals.run.reads import list_runs
from threetears.evals.kernel.identity import IDENTITY_VERSION
from packages.evals.tests.factories import as_listed
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile


#: The host whose vocabulary every grouping here reads: the toy host. Nothing in this module is
#: about any particular host's levers -- every run carries the same subject and an empty payload --
#: so the toy host stands in for whichever host serves the grouping.
_HOST = toyhost_profile()


def bool_disjoint(windows) -> bool:
    """Whether ANY pair of these windows is disjoint.

    ``reporting`` exports the pair list, not a boolean. A wrapper named
    ``measurement_windows_are_disjoint`` used to provide one, and it read as a
    universal ("these windows are disjoint") while returning an existential —
    the exact misreading this module exists to prevent. It lost its last
    production caller when the disclosure moved onto the pair list directly, so
    the boolean is a test convenience now and its name says which quantifier it
    means.
    """
    return bool(disjoint_window_pairs(windows))


def _result(run_id: str, scored_at: str) -> EvalResult:
    """A stored result carrying only what a window is derived from."""
    return EvalResult(
        scope_id="u",
        eval_run_id=run_id,
        test_case_id="tc",
        model="m",
        k_iteration=1,
        scored_at=scored_at,
        termination="completed",
        cost_usd=0.0,
        cost_roles=["candidate", "inner_agent", "judge", "simulator"],
        usage=[],
        covariates={},
        phase_timings={},
        host_measures={},
        variant_key="vk-1",
        identity_version=IDENTITY_VERSION,
    )


def _window(run_id: str, start: str, end: str) -> MeasurementWindow:
    return MeasurementWindow(run_id=run_id, start=start, end=end)


def _subject(subject_id: str, label: str) -> SubjectSnapshot:
    """A subject with no components and no carried state -- identity is all a grouping reads of it."""
    return SubjectSnapshot(
        subject_id=subject_id, subject_label=label, state={}, captured_at="2026-08-04T00:00:00+00:00"
    )


def _comparison_sets(storage: _WindowStorage) -> dict:
    """The grouping as a host serves it: the engine's composition over the engine's run listing.

    A host's service delegates exactly this, so driving it here drives the production seam --
    the one place the results argument is supplied.
    """
    return comparison_sets(
        storage,
        "u",
        list_runs=lambda scope_id, **kwargs: list_runs(toyhost_host(storage=storage), scope_id, **kwargs),
        profile=_HOST,
    )


def _grouped_run(run_id: str) -> EvalRun:
    """A run that groups with its siblings on every other coordinate.

    Same subject, template, models and frozen case set, so the only thing a badge
    can be reporting is the measurement window. A fixture that differed elsewhere
    would let a passing assertion be carried by the wrong badge.
    """
    return EvalRun(
        apparatus_provenance="commissioned",
        id=run_id,
        scope_id="u",
        template_id="tpl",
        subject_snapshot=_subject("ent-maple", "Maple"),
        candidate_model="m",
        test_case_ids=["tc"],
        status="completed",
        candidate_kind="test-kind",
        k_runs=1,
        rubric_scales={},
    )


class _WindowStorage:
    """The two scope-wide reads the served grouping composes over.

    Deliberately not a ``MagicMock``: a mock answers ``query_eval_results`` with a
    mock whatever the service does, so it cannot tell a service that reads the
    results from one that does not.
    """

    def __init__(self, runs: list[EvalRun], results: list[EvalResult]) -> None:
        self._runs = runs
        self._results = results

    def query_eval_runs(
        self, scope_id: str, status: str | None = None, *, elide_payload: frozenset[str] = frozenset()
    ) -> list[EvalRun]:
        """Return the seeded runs in the scope, filtered by status as the real store does.

        Args:
            scope_id: Partition key.
            status: Optional run-status filter.

        Returns:
            The matching runs.
        """
        return as_listed(
            [r for r in self._runs if r.scope_id == scope_id and (not status or r.status == status)],
            elide_payload,
        )

    def query_eval_results(self, scope_id: str) -> list[EvalResult]:
        """Return the seeded results in the scope.

        Args:
            scope_id: Partition key.

        Returns:
            The matching results.
        """
        return [r for r in self._results if r.scope_id == scope_id]


# =============================================================================
# Derivation — from the results, never from the run document
# =============================================================================


def test_measurement_window_is_the_span_of_its_results_scored_at():
    results = [
        _result("r1", "2026-08-04T23:04:32.100000+00:00"),
        _result("r1", "2026-08-04T22:22:09.000000+00:00"),
        _result("r1", "2026-08-04T22:51:00.000000+00:00"),
    ]

    window = measurement_window("r1", results)

    assert window == _window("r1", "2026-08-04T22:22:09.000000+00:00", "2026-08-04T23:04:32.100000+00:00")


def test_measurement_window_is_derived_from_result_timestamps_and_not_the_runs_own_span():
    """The corrected premise, pinned as an assertion rather than left as prose.

    A run document is saved — stamping ``created_at`` — before its execution task
    is even created, and that task then queues behind a semaphore. So ``created_at``
    is when the run was *enqueued*. The fixture reproduces the shape that makes the
    difference matter: two arms enqueued seconds apart that executed hours apart. On
    the run document's own span they are indistinguishable and overlapping; on the
    results' span they are two separate measurement sessions, which is the fact a
    reader needs.
    """
    enqueued_a = "2026-08-04T22:22:09.000000+00:00"
    enqueued_b = "2026-08-04T22:22:40.000000+00:00"
    finished = "2026-08-05T02:18:42.000000+00:00"

    arm_a = EvalRun(
        apparatus_provenance="commissioned",
        id="arm-a",
        scope_id="u",
        subject_snapshot=_subject("e", "Maple"),
        candidate_model="m",
        test_case_ids=["tc"],
        created_at=enqueued_a,
        completed_at=finished,
        candidate_kind="test-kind",
        k_runs=1,
        rubric_scales={},
    )
    arm_b = EvalRun(
        apparatus_provenance="commissioned",
        id="arm-b",
        scope_id="u",
        subject_snapshot=_subject("e", "Maple"),
        candidate_model="m",
        test_case_ids=["tc"],
        created_at=enqueued_b,
        completed_at=finished,
        candidate_kind="test-kind",
        k_runs=1,
        rubric_scales={},
    )

    # Enqueue spans: arm_b's sits wholly inside arm_a's. Nothing derived from them
    # can ever read as disjoint, whatever the arms actually did.
    assert arm_a.created_at < arm_b.created_at < arm_a.completed_at

    windows = [
        measurement_window(arm_a.id, [_result("arm-a", "2026-08-04T22:23:00.000000+00:00")]),
        measurement_window(arm_b.id, [_result("arm-b", "2026-08-05T01:44:00.000000+00:00")]),
    ]

    assert bool_disjoint([w for w in windows if w is not None])


def test_a_run_with_no_results_has_no_measurement_window():
    """``None``, never a zero-length span.

    A run that measured nothing did not measure it "at" some instant, and a
    synthetic point would compare against real windows as though it had — which is
    how an absence starts evidencing a difference.
    """
    assert measurement_window("r1", []) is None


def test_a_result_carrying_no_timestamp_does_not_anchor_a_window_at_the_epoch():
    """A blank stamp is unknown, not "the beginning of time"."""
    results = [_result("r1", ""), _result("r1", "2026-08-04T22:22:09.000000+00:00")]

    window = measurement_window("r1", results)

    assert window == _window("r1", "2026-08-04T22:22:09.000000+00:00", "2026-08-04T22:22:09.000000+00:00")
    assert measurement_window("r1", [_result("r1", "")]) is None


# =============================================================================
# Overlap — the predicate a caveat would be gated on
# =============================================================================


def test_disjoint_measurement_windows_are_reported_as_disjoint():
    windows = [
        _window("morning", "2026-08-04T09:00:00+00:00", "2026-08-04T10:30:00+00:00"),
        _window("evening", "2026-08-04T21:00:00+00:00", "2026-08-04T22:15:00+00:00"),
    ]

    assert bool_disjoint(windows) is True


def test_overlapping_measurement_windows_are_not():
    windows = [
        _window("a", "2026-08-04T09:00:00+00:00", "2026-08-04T11:00:00+00:00"),
        _window("b", "2026-08-04T10:00:00+00:00", "2026-08-04T12:00:00+00:00"),
    ]

    assert bool_disjoint(windows) is False
    assert measurement_window_disclosure(windows) is None


def test_windows_that_merely_touch_still_overlap():
    """Two runs sharing an instant were running against the same conditions."""
    windows = [
        _window("a", "2026-08-04T09:00:00+00:00", "2026-08-04T11:00:00+00:00"),
        _window("b", "2026-08-04T11:00:00+00:00", "2026-08-04T12:00:00+00:00"),
    ]

    assert bool_disjoint(windows) is False


def test_any_disjoint_pair_is_enough_even_when_two_of_three_overlap():
    """Three arms, two concurrent and one the next morning, is the common sweep shape.

    Requiring *every* pair to be disjoint would report nothing there, which is the
    reading that lets a set nothing was held fixed across pass unremarked.
    """
    windows = [
        _window("a", "2026-08-04T09:00:00+00:00", "2026-08-04T11:00:00+00:00"),
        _window("b", "2026-08-04T10:00:00+00:00", "2026-08-04T12:00:00+00:00"),
        _window("c", "2026-08-05T09:00:00+00:00", "2026-08-05T10:00:00+00:00"),
    ]

    assert bool_disjoint(windows) is True


def test_a_run_with_no_resolvable_window_does_not_evidence_a_difference():
    """Mixing a windowed run with a window-less one must report nothing.

    The absence-as-difference failure, which the attribution comparison had to be
    fixed for twice: a run that cannot say when it was measured is not thereby
    evidence that it was measured somewhere else.
    """
    resolved = [_window("a", "2026-08-04T09:00:00+00:00", "2026-08-04T11:00:00+00:00")]
    unresolved = measurement_window("b", [])

    assert unresolved is None
    assert bool_disjoint(resolved) is False
    assert measurement_window_disclosure(resolved) is None


def test_a_single_window_and_no_window_at_all_are_both_silent():
    assert bool_disjoint([]) is False
    assert bool_disjoint([_window("a", "2026-08-04T09:00:00+00:00", "2026-08-04T11:00:00+00:00")]) is False


# =============================================================================
# Disclosure — descriptive, carrying the windows, carrying no verdict
# =============================================================================


def test_the_disclosure_names_every_window_it_was_derived_from():
    """ "The operator can judge severity" is only true if the spans are on the page.

    A bare caveat word says a gap exists; the reader still cannot tell four minutes
    from four hours, which is the whole of the judgement being handed to them.
    """
    windows = [
        _window("evening", "2026-08-04T21:00:00+00:00", "2026-08-04T22:15:00+00:00"),
        _window("morning", "2026-08-04T09:00:00+00:00", "2026-08-04T10:30:00+00:00"),
    ]

    disclosure = measurement_window_disclosure(windows)

    assert disclosure is not None
    for window in windows:
        assert window.run_id in disclosure
        assert window.start in disclosure
        assert window.end in disclosure
    assert DISJOINT_WINDOWS_CLAUSE in disclosure
    # Earliest first, so the reader reads the runs in the order they happened
    # rather than in whatever order the group was assembled.
    assert disclosure.index("morning") < disclosure.index("evening")


def test_a_large_group_summarises_to_its_outer_bounds_instead_of_every_span():
    """Past a handful of runs the listing form stops being a disclosure.

    A 27-run group rendered every span inline as one unbroken paragraph, while the
    badge legend told the reader to "read it rather than the badge" — advice nobody
    can take at that length. The summary keeps what the disclosure exists to answer,
    whether the group spans minutes or days, and says how many spans it withheld so
    the omission is never silent.
    """
    windows = [
        _window(f"run{i:02d}", f"2026-08-{4 + i:02d}T09:00:00+00:00", f"2026-08-{4 + i:02d}T10:00:00+00:00")
        for i in range(MAX_INLINE_MEASUREMENT_WINDOWS + 3)
    ]

    disclosure = measurement_window_disclosure(windows)

    assert disclosure is not None
    assert DISJOINT_WINDOWS_CLAUSE in disclosure
    # The outer bounds are what answers "minutes or days" without any span list.
    assert "2026-08-04T09:00:00+00:00" in disclosure
    assert windows[-1].end in disclosure
    # The extremes are named; the interior is not, and the count of what is missing
    # is on the page so a reader never mistakes the summary for the whole group.
    assert "run00" in disclosure
    assert windows[-1].run_id in disclosure
    assert "run03" not in disclosure
    assert f"{len(windows) - 2} further span(s) not shown" in disclosure
    assert "full=true" in disclosure


def test_the_summary_takes_the_latest_end_not_the_last_runs_end():
    """Sorting by START does not put the latest END last.

    One long-running arm that began early can finish after every later arm. Reading
    the group's upper bound off the last element understates the span precisely when
    a run overran, which is when the disclosure matters most.
    """
    windows = [
        _window("overrunner", "2026-08-04T09:00:00+00:00", "2026-08-09T23:00:00+00:00"),
        *(
            _window(f"later{i}", f"2026-08-{5 + i:02d}T09:00:00+00:00", f"2026-08-{5 + i:02d}T10:00:00+00:00")
            for i in range(MAX_INLINE_MEASUREMENT_WINDOWS + 1)
        ),
    ]

    disclosure = measurement_window_disclosure(windows)

    assert disclosure is not None
    # Pinned to the BOUNDS clause, not merely to the timestamp appearing somewhere.
    # The overrunner is also the earliest window, so its end is printed by the
    # "earliest ... to {end}" clause whatever the upper bound says — an assertion
    # that only looked for the timestamp passed with the bound computed wrongly.
    assert "measured between 2026-08-04T09:00:00+00:00 and 2026-08-09T23:00:00+00:00" in disclosure


def test_full_lists_every_span_however_many_there_are():
    """The detail stays reachable — summarising is a default, not a ceiling."""
    windows = [
        _window(f"run{i:02d}", f"2026-08-{4 + i:02d}T09:00:00+00:00", f"2026-08-{4 + i:02d}T10:00:00+00:00")
        for i in range(MAX_INLINE_MEASUREMENT_WINDOWS + 3)
    ]

    disclosure = measurement_window_disclosure(windows, full=True)

    assert disclosure is not None
    for window in windows:
        assert window.run_id in disclosure
        assert window.start in disclosure
    assert "not shown" not in disclosure


def test_the_disclosure_states_the_magnitude_and_still_carries_no_verdict():
    """The gap's SIZE is reported; what the size MEANS is not.

    Reporting presents and compares but never editorializes: the surface's job is
    to say the runs did not share a clock and how far apart they were, and the
    operator's is to decide what that costs. A word like "invalid" or
    "significant" here would be this module deciding it for them, on a question it
    has no data about.

    **This test previously asserted the opposite of its second half** — "it does
    not decide how far apart is too far, so it states no interval either" — and
    the reversal is deliberate, not a weakening. Each disclosure computes a
    magnitude in the measure's own units ("windows 50m31s apart"), because a
    reader cannot weigh a gap they are not shown. What stays undecided is the
    *threshold* — the descriptor-declared value that decides materiality — so the
    magnitude is now required here and the verdict is still forbidden.
    """
    windows = [
        _window("a", "2026-08-04T09:00:00+00:00", "2026-08-04T10:30:00+00:00"),
        _window("b", "2026-08-04T21:00:00+00:00", "2026-08-04T22:15:00+00:00"),
    ]

    disclosure = measurement_window_disclosure(windows)

    assert disclosure is not None
    lowered = disclosure.lower()
    for verdict_word in ("invalid", "unusable", "significant", "severe", "critical", "warning", "should", "must not"):
        assert verdict_word not in lowered, f"the disclosure editorializes: {verdict_word!r}"
    # 10:30 to 21:00 is 10h30m — stated as a magnitude, with no claim about it.
    assert "10h30m apart" in disclosure
    # No threshold language: the size is given, the judgement is not made.
    for threshold_word in ("too far", "threshold", "acceptable", "material", "negligible"):
        assert threshold_word not in lowered, f"the disclosure decides materiality: {threshold_word!r}"


# =============================================================================
# The quantifier
#
# The predicate is existential ("at least one pair"). Printing it as a universal
# is the falsehood the generator escalated into a false BLUF, so these pin what
# the prose is entitled to claim rather than only that it fires.
# =============================================================================


def test_a_partly_overlapping_group_does_not_claim_every_run_was_measured_apart():
    """The bug, reproduced from a real campaign's shape: three of six pairs disjoint, all four claimed.

    These four spans have a real campaign's shape. `run-a` contains
    `run-b` and overlaps `run-c`, which in turn contains `run-d`; only three of the six
    pairs are genuinely disjoint. The old sentence opened "These runs were measured over
    non-overlapping spans", which is false of every one of the three overlapping pairs.
    """
    windows = [
        _window("run-a", "2026-01-15T10:00:00+00:00", "2026-01-15T10:12:00+00:00"),
        _window("run-b", "2026-01-15T10:02:00+00:00", "2026-01-15T10:08:00+00:00"),
        _window("run-c", "2026-01-15T10:11:00+00:00", "2026-01-15T10:38:00+00:00"),
        _window("run-d", "2026-01-15T10:13:00+00:00", "2026-01-15T10:24:00+00:00"),
    ]

    disclosure = measurement_window_disclosure(windows)

    assert disclosure is not None
    assert not disclosure.startswith("These runs were"), "the existential is still printed as a universal"
    assert "3 of the 6 pairs among the 4 runs named here" in disclosure
    assert "the other 3 overlap" in disclosure
    # The attribution warning is scoped to the pairs that earned it, not to the group.
    assert "a difference between the two runs of a non-overlapping pair is not attributable" in disclosure


def test_a_wholly_disjoint_group_still_states_the_universal_because_it_is_true():
    """The fix must not over-hedge: when every pair is disjoint, say so plainly.

    A quantifier fix that downgraded the true universal to a count would trade one
    inaccuracy for another and make the common two-arm sequential case read as
    though something were unresolved.
    """
    windows = [
        _window("a", "2026-08-04T09:00:00+00:00", "2026-08-04T10:00:00+00:00"),
        _window("b", "2026-08-04T11:00:00+00:00", "2026-08-04T12:00:00+00:00"),
        _window("c", "2026-08-04T13:00:00+00:00", "2026-08-04T14:00:00+00:00"),
    ]

    disclosure = measurement_window_disclosure(windows)

    assert disclosure is not None
    assert disclosure.startswith(f"These runs were {DISJOINT_WINDOWS_CLAUSE} of wall-clock time — every pair of them")
    # The counting form, and only it, says some pairs overlapped.
    assert "pairs among the" not in disclosure
    assert "overlap (" not in disclosure
    assert "a difference between them is not attributable" in disclosure


def test_every_non_overlapping_pair_is_named_with_its_gap_widest_first():
    """Which pairs, and how far — the two things the badge alone cannot carry.

    Widest first so the pair the reader most needs leads, and so a truncated tail
    drops the least rather than the most.
    """
    windows = [
        _window("a", "2026-08-04T09:00:00+00:00", "2026-08-04T09:30:00+00:00"),
        _window("b", "2026-08-04T09:31:00+00:00", "2026-08-04T10:00:00+00:00"),
        _window("c", "2026-08-04T12:00:00+00:00", "2026-08-04T12:30:00+00:00"),
    ]

    disclosure = measurement_window_disclosure(windows)

    assert disclosure is not None
    assert "a and b, 1m00s apart" in disclosure
    assert "a and c, 2h30m apart" in disclosure
    assert "b and c, 2h00m apart" in disclosure
    # Widest first: a↔c (2h30m), then b↔c (2h00m), then a↔b (1m).
    assert disclosure.index("a and c, 2h30m") < disclosure.index("b and c, 2h00m") < disclosure.index("a and b, 1m00s")


def test_a_gap_that_cannot_be_computed_is_admitted_rather_than_omitted():
    """A malformed stamp must not raise, and must not read as "no gap".

    The lexicographic discipline is what keeps this surface from raising on a
    stamp nothing can parse; the magnitude is the only thing that parses, and it
    says so when it cannot.
    """
    windows = [
        # Lexicographically ordered and genuinely disjoint — the string comparisons
        # that decide overlap still work — but `fromisoformat` cannot read the end.
        _window("malformed", "2026-08-04T09:00:00+00:00", "2026-08-04T10:00:00 UTC"),
        _window("iso", "2026-08-04T21:00:00+00:00", "2026-08-04T22:00:00+00:00"),
    ]

    disclosure = measurement_window_disclosure(windows)

    assert disclosure is not None
    assert "malformed and iso, gap not computable from the recorded stamps" in disclosure


def test_a_naive_stamp_beside_an_aware_one_yields_no_duration_rather_than_a_wrong_one():
    """Subtracting a naive instant from an aware one is not a duration anybody measured."""
    windows = [
        _window("naive", "2026-08-04T09:00:00", "2026-08-04T10:00:00"),
        _window("aware", "2026-08-04T21:00:00+00:00", "2026-08-04T22:00:00+00:00"),
    ]

    disclosure = measurement_window_disclosure(windows)

    assert disclosure is not None
    assert reporting.UNCOMPUTABLE_GAP_CLAUSE in disclosure


def test_the_summary_form_names_the_widest_gap_and_counts_the_rest():
    """Above the inline cap the pair list collapses the way the span list does.

    The earlier form refused to name a widest gap at all, on the ground that no
    read surface may parse a datetime. The guarantee is kept — ordering and the
    overlap predicate are still string comparisons — and the summary now answers
    the question the disclosure exists for.
    """
    windows = [
        _window(f"run{i:02d}", f"2026-08-{4 + i:02d}T09:00:00+00:00", f"2026-08-{4 + i:02d}T10:00:00+00:00")
        for i in range(MAX_INLINE_MEASUREMENT_WINDOWS + 3)
    ]

    disclosure = measurement_window_disclosure(windows)

    assert disclosure is not None
    # run00 ends 08-04 10:00; run06 starts 08-10 09:00 — 5 days 23 hours.
    assert "Widest non-overlapping pair: run00 and run06, 5d23h apart" in disclosure
    assert "20 further non-overlapping pair(s) not named" in disclosure


def test_the_pair_list_caps_even_when_every_span_is_listed():
    """Spans grow linearly and pairs grow quadratically, so `full` cannot uncap both.

    A `full=true` read of a 22-run campaign holds 231 pairs; a paragraph of them
    is the same non-disclosure the uncapped span list was. The withheld count is
    stated, never dropped.
    """
    windows = [
        _window(f"run{i:02d}", f"2026-08-{4 + i:02d}T09:00:00+00:00", f"2026-08-{4 + i:02d}T10:00:00+00:00")
        for i in range(MAX_INLINE_MEASUREMENT_WINDOWS + 3)
    ]

    disclosure = measurement_window_disclosure(windows, full=True)

    assert disclosure is not None
    named = disclosure.count(" apart")
    assert named == reporting.MAX_RENDERED_WINDOW_GAPS
    assert (
        f"+{21 - reporting.MAX_RENDERED_WINDOW_GAPS} further non-overlapping pair(s), all narrower than these"
        in disclosure
    )


def test_the_predicate_and_the_pair_list_are_one_derivation():
    """The badge and the prose read the same list, so they cannot disagree.

    `measurement_windows_are_disjoint` used to walk its own pairwise loop. Two
    loops answering one question is how a badge comes to fire over a sentence that
    names no pair — the drift this module already refuses everywhere else.
    """
    overlapping = [
        _window("a", "2026-08-04T09:00:00+00:00", "2026-08-04T11:00:00+00:00"),
        _window("b", "2026-08-04T10:00:00+00:00", "2026-08-04T12:00:00+00:00"),
    ]
    partly = [*overlapping, _window("c", "2026-08-04T20:00:00+00:00", "2026-08-04T21:00:00+00:00")]

    assert reporting.disjoint_window_pairs(overlapping) == []
    assert bool_disjoint(overlapping) is False
    assert len(reporting.disjoint_window_pairs(partly)) == 2
    assert bool_disjoint(partly) is True


def test_the_split_covers_every_pair_and_names_the_overlapping_half():
    """Both halves come off one enumeration, so neither can be the other's complement.

    The disclosure counts disjoint pairs against the total, and the model is handed the
    overlapping half to read before claiming a pair was measured apart. Deriving one half and
    complementing it at the second caller is how the two would come to disagree about
    a pair — the same drift the single pair list already prevents for the badge.
    """
    windows = [
        _window("a", "2026-08-04T09:00:00+00:00", "2026-08-04T11:00:00+00:00"),
        _window("b", "2026-08-04T10:00:00+00:00", "2026-08-04T12:00:00+00:00"),
        _window("c", "2026-08-04T20:00:00+00:00", "2026-08-04T21:00:00+00:00"),
    ]

    pairs = reporting.classify_window_pairs(windows)

    assert pairs.total == 3, "every pair is classified — a run set of three has three of them"
    assert pairs.disjoint == reporting.disjoint_window_pairs(windows), "the disjoint half IS the exported list"
    assert [(both.first.run_id, both.second.run_id) for both in pairs.overlapping] == [("a", "b")]


def test_a_run_set_too_small_to_pair_classifies_as_neither():
    """One run cannot be measured apart from itself, and it cannot overlap itself either.

    The empty case matters because a partial-disclosure rule keyed on ``0 < D < P``
    reads a fabricated total as a real one.
    """
    lone = [_window("a", "2026-08-04T09:00:00+00:00", "2026-08-04T11:00:00+00:00")]

    assert reporting.classify_window_pairs(lone).total == 0
    assert reporting.classify_window_pairs([]).total == 0


@pytest.mark.parametrize(
    "seconds,rendered",
    [
        (0, "0s"),
        (55, "55s"),
        (61, "1m01s"),
        (3031, "50m31s"),
        (3600, "1h00m"),
        (86400, "1d00h"),
        (516600, "5d23h"),
        (None, reporting.UNCOMPUTABLE_GAP_CLAUSE),
    ],
)
def test_a_gap_renders_as_two_significant_units(seconds, rendered):
    """Compact and unit-word-free, so a list of pairs reads as numbers to weigh."""
    assert reporting.format_window_gap(seconds) == rendered


# =============================================================================
# Reaching a caller — the grouping surface, where an operator actually meets it
#
# A predicate and a sentence that nothing calls answer no question. These pin the
# path from stored results to the badge on a comparison group, which is what the
# formatter tests above cannot: every one of them would still pass with the
# grouping surface unchanged.
# =============================================================================


def test_a_group_whose_runs_were_measured_hours_apart_is_badged_and_names_the_spans():
    """The badge fires and the sentence carries the spans, on one predicate.

    Both halves are asserted together on purpose: the flag alone cannot tell four
    minutes from four hours, and a sentence nobody is pointed at by a flag is not
    a caveat on the comparison.
    """
    morning, evening = _grouped_run("arm-morning"), _grouped_run("arm-evening")
    results = [
        _result("arm-morning", "2026-08-04T22:23:00.000000+00:00"),
        _result("arm-morning", "2026-08-04T23:04:32.000000+00:00"),
        _result("arm-evening", "2026-08-05T01:44:00.000000+00:00"),
        _result("arm-evening", "2026-08-05T02:18:42.000000+00:00"),
    ]

    group = compute_comparison_sets([morning, evening], results=results, profile=_HOST).comparison_sets[0]

    assert BADGE_MEASUREMENT_WINDOWS_DISJOINT in group.badges
    assert group.measurement_window_disclosure is not None
    for run_id in ("arm-morning", "arm-evening"):
        assert run_id in group.measurement_window_disclosure
    assert "2026-08-04T22:23:00.000000+00:00" in group.measurement_window_disclosure
    assert "2026-08-05T02:18:42.000000+00:00" in group.measurement_window_disclosure


def test_runs_measured_over_the_same_stretch_are_not_badged():
    """The caveat must fire on the condition, not on every group that has results."""
    a, b = _grouped_run("arm-a"), _grouped_run("arm-b")
    results = [
        _result("arm-a", "2026-08-04T22:00:00.000000+00:00"),
        _result("arm-a", "2026-08-04T23:30:00.000000+00:00"),
        _result("arm-b", "2026-08-04T22:30:00.000000+00:00"),
        _result("arm-b", "2026-08-04T23:00:00.000000+00:00"),
    ]

    group = compute_comparison_sets([a, b], results=results, profile=_HOST).comparison_sets[0]

    assert BADGE_MEASUREMENT_WINDOWS_DISJOINT not in group.badges
    assert group.measurement_window_disclosure is None


def test_the_badge_and_its_sentence_are_never_reported_apart():
    """One predicate decides both, so no group can carry a flag with nothing behind it.

    Checked across a group that fires and one that does not, in a single call, so
    a surface reading only the badge and a surface reading only the sentence
    cannot come to different answers about the same scope.
    """
    disjoint_a, disjoint_b = _grouped_run("split-a"), _grouped_run("split-b")
    together_a, together_b = _grouped_run("together-a"), _grouped_run("together-b")
    for run in (together_a, together_b):
        run.subject_snapshot = _subject("ent-bea", "Bea")
    results = [
        _result("split-a", "2026-08-04T09:00:00.000000+00:00"),
        _result("split-b", "2026-08-04T21:00:00.000000+00:00"),
        # Two stamps each, so these two really do span a shared stretch. One stamp
        # per run would be two instants, which are disjoint unless they coincide.
        _result("together-a", "2026-08-04T09:00:00.000000+00:00"),
        _result("together-a", "2026-08-04T10:00:00.000000+00:00"),
        _result("together-b", "2026-08-04T09:30:00.000000+00:00"),
        _result("together-b", "2026-08-04T10:30:00.000000+00:00"),
    ]

    groups = compute_comparison_sets(
        [disjoint_a, disjoint_b, together_a, together_b], results=results, profile=_HOST
    ).comparison_sets

    assert len(groups) == 2
    for group in groups:
        assert (BADGE_MEASUREMENT_WINDOWS_DISJOINT in group.badges) is (group.measurement_window_disclosure is not None)
    assert {BADGE_MEASUREMENT_WINDOWS_DISJOINT in g.badges for g in groups} == {True, False}


def test_a_run_that_produced_no_results_cannot_pull_the_badge():
    """An absence is not evidence that the run was measured somewhere else.

    The same rule the case-basis and attribution arms follow, asserted through the
    grouping surface rather than only against the predicate: one run with a window
    and one without leaves nothing to compare, so the group is silent.
    """
    measured, silent = _grouped_run("measured"), _grouped_run("silent")

    group = compute_comparison_sets(
        [measured, silent],
        results=[_result("measured", "2026-08-04T09:00:00.000000+00:00")],
        profile=_HOST,
    ).comparison_sets[0]

    assert BADGE_MEASUREMENT_WINDOWS_DISJOINT not in group.badges
    assert group.measurement_window_disclosure is None


def test_a_caller_that_supplies_no_results_asks_a_narrower_question_and_gets_no_badge():
    """Undecidable, and silent rather than clean-looking on some invented basis.

    A caller with only the run documents cannot answer this — a run is stamped
    ``created_at`` when it is enqueued — so the surface reports nothing instead of
    substituting a basis that would be silent on exactly the runs it exists to
    describe.
    """
    a, b = _grouped_run("arm-a"), _grouped_run("arm-b")

    group = compute_comparison_sets([a, b], profile=_HOST).comparison_sets[0]

    assert BADGE_MEASUREMENT_WINDOWS_DISJOINT not in group.badges
    assert group.measurement_window_disclosure is None


# =============================================================================
# The production seam — the read that makes the badge decidable at all
#
# `comparison_sets(runs, results=())` DEFAULTS to silence, so everything above
# would keep passing with the served grouping never reading a result: each of
# those tests hands `results=` in itself. These drive the engine's composition
# over seeded storage, which is the only place that argument is supplied in
# production.
# =============================================================================


def test_the_service_reads_the_scopes_results_so_the_badge_can_reach_an_operator():
    """End to end from stored documents: two arms measured hours apart, badged.

    The grouping cannot answer this from the run documents — a run is stamped
    ``created_at`` when it is enqueued — so the service's second read *is* the
    feature. Asserted through the served composition rather than through the
    reporting function, because the argument that carries it has a silent default.
    """
    storage = _WindowStorage(
        runs=[_grouped_run("arm-morning"), _grouped_run("arm-evening")],
        results=[
            _result("arm-morning", "2026-08-04T22:23:00.000000+00:00"),
            _result("arm-morning", "2026-08-04T23:04:32.000000+00:00"),
            _result("arm-evening", "2026-08-05T01:44:00.000000+00:00"),
            _result("arm-evening", "2026-08-05T02:18:42.000000+00:00"),
        ],
    )

    payload = _comparison_sets(storage)

    group = payload["comparison_sets"][0]
    assert BADGE_MEASUREMENT_WINDOWS_DISJOINT in group["badges"]
    disclosure = group["measurement_window_disclosure"]
    assert disclosure is not None
    assert "2026-08-04T22:23:00.000000+00:00" in disclosure
    assert "2026-08-05T02:18:42.000000+00:00" in disclosure


def test_the_service_leaves_runs_that_shared_a_stretch_unbadged():
    """The other half of the seam: reading the results must not badge everything.

    A test that only asserts the badge fires passes just as well against a
    service that badges unconditionally, which would be the more damaging bug —
    a caveat on every group is a caveat on none.
    """
    storage = _WindowStorage(
        runs=[_grouped_run("arm-a"), _grouped_run("arm-b")],
        results=[
            _result("arm-a", "2026-08-04T22:00:00.000000+00:00"),
            _result("arm-a", "2026-08-04T23:30:00.000000+00:00"),
            _result("arm-b", "2026-08-04T22:30:00.000000+00:00"),
            _result("arm-b", "2026-08-04T23:00:00.000000+00:00"),
        ],
    )

    group = _comparison_sets(storage)["comparison_sets"][0]

    assert BADGE_MEASUREMENT_WINDOWS_DISJOINT not in group["badges"]
    assert group["measurement_window_disclosure"] is None


# =============================================================================
# The operator doc — the third copy of the badge vocabulary
# =============================================================================
