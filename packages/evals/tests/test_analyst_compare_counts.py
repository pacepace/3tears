"""What a comparison's interval counts, whether it is corrected, and when it may be too narrow: read, never asserted.

Contracts held here (``threetears.evals.analysis.analyst.compare`` and the ``compare.*`` keys'
``basis`` in ``analysis.report.figure_keys``):

- **correction read from the family**: a paired row's basis says its interval is corrected only where the one
  contrast family on the contrast's aspect names a correction other than ``none``; where the family's correction is
  ``none`` it says not corrected, and where no one family or no correction is stated it says so; never "corrected
  within its family" against a family that says none (the W2 regression prototype's r4 contradiction);
- **what the interval counts**: an independent row cites what each arm's count counts (its ``spread.`` key); a
  paired row says its contrast states no count (``counted_unstated``) and the page's notes say so once;
- **may be too narrow**: a row is marked where what its interval counts is trials, observations or unstated, its
  spread was not read on distinct cases, and an arm ran more trials than cases, citing each such arm's trials and
  cases; never where every arm's trials equal its cases. Expectations are computed from each input. The note says it
  as a may (``may be too narrow if these intervals count trials nested in cases``), never that reading on the cases
  would widen it, and never calls a paired row, whose unit is unstated, counting trials.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from packages.evals.tests.analyst_support import KAIRO, fixture_inputs, fixture_snapshots
from packages.evals.tests.fixtures.report_plans.misfits import reference_data
from threetears.evals.analysis.analyst.compare import ArmsComparison, ComparisonRow, arms_compare
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import on_lead_aspect, resolve
from threetears.evals.analysis.report_input import Contrast, ReportInput
from threetears.evals.kernel.errors import RefusedInput

#: Every row a test reads.
WHOLE = 200


def _inputs() -> dict[str, ReportInput]:
    inputs = dict(fixture_inputs())
    inputs[KAIRO] = frozen_input(fixture_snapshots("s")[KAIRO])
    return inputs


def _paired_contrasts(held: ReportInput) -> Iterator[Contrast]:
    """The contrasts a comparison overall reads as its paired estimate: on the lead aspect, against the control, with a
    difference and an interval, one per arm and measure."""
    control = next((arm.id for arm in held.arms if arm.role == "control"), None)
    seen: set[tuple[str, str]] = set()
    for contrast in held.contrasts:
        held_once = [
            other
            for other in held.contrasts
            if (other.arm, other.measure) == (contrast.arm, contrast.measure) and on_lead_aspect(held, other)
        ]
        if (
            control is None
            or contrast.control not in (None, control)
            or not on_lead_aspect(held, contrast)
            or contrast.delta is None
            or contrast.interval is None
            or len(held_once) != 1
            or (contrast.arm, contrast.measure) in seen
            or contrast.measure not in {*held.measures, *held.judged}
        ):
            continue
        seen.add((contrast.arm, contrast.measure))
        yield contrast


def _expected_correction(held: ReportInput, contrast: Contrast) -> str:
    families = [family for family in held.contrast_families if family.aspect == contrast.aspect]
    if len(families) != 1:
        return "no one contrast family"
    correction = (families[0].correction or "").strip()
    if not correction:
        return "states no correction"
    if correction.lower() == "none":
        return "not corrected"
    return f"corrected within its family ({correction})"


def _pairs() -> list[tuple[str, str, str]]:
    return [
        (name, contrast.arm, contrast.measure)
        for name, held in _inputs().items()
        for contrast in _paired_contrasts(held)
    ]


@pytest.mark.parametrize(("name", "arm", "measure"), _pairs(), ids=lambda value: str(value))
def test_a_paired_rows_correction_is_its_familys_own_and_never_asserted(name: str, arm: str, measure: str) -> None:
    held = _inputs()[name]
    contrast = next(c for c in _paired_contrasts(held) if (c.arm, c.measure) == (arm, measure))
    control = next(a.id for a in held.arms if a.role == "control")
    basis = resolve(f"compare.{arm}.{control}.{measure}.overall.basis", held).value
    assert isinstance(basis, str) and basis.startswith("paired:")
    expected = _expected_correction(held, contrast)
    assert expected in basis, (basis, expected)
    assert ("corrected within its family" in basis) == expected.startswith("corrected within"), basis


def test_the_fixtures_hold_a_family_saying_none_and_one_naming_its_correction() -> None:
    """The control for the test above: both kinds of family are among the pairs it reads."""
    found = {_expected_correction(_inputs()[name], c) for name in _inputs() for c in _paired_contrasts(_inputs()[name])}
    assert "not corrected" in found
    assert any(words.startswith("corrected within its family (") for words in found)


def test_r4s_contrast_is_not_called_corrected_where_its_family_says_none() -> None:
    """The W2 regression prototype's r4 finding: arms_compare called the contrast "corrected within its family" while
    its family says its correction is none."""
    held = fixture_inputs()["r4"]
    (family,) = held.contrast_families
    assert family.correction == "none"
    contrast = next(_paired_contrasts(held))
    control = next(a.id for a in held.arms if a.role == "control")
    row = arms_compare(held, [contrast.arm, control], contrast.measure).rows[0]
    assert row.basis is not None and isinstance(row.basis.value, str)
    assert "corrected within its family" not in row.basis.value and "not corrected" in row.basis.value


def _comparisons(held: ReportInput) -> Iterator[ArmsComparison]:
    """Each candidate against the control on every measure both hold, where an interval reads somewhere."""
    control = next((arm.id for arm in held.arms if arm.role == "control"), held.arms[0].id)
    for arm in held.arms:
        if arm.id == control:
            continue
        for key in dict.fromkeys([*held.measures, *held.judged]):
            try:
                yield arms_compare(held, [arm.id, control], key, limit=WHOLE)
            except RefusedInput:
                continue


def _more_trials_than_cases(held: ReportInput, arm: str) -> bool:
    board = held.scoreboard.get(arm)
    if board is None or board.cases is None:
        return False
    trials = board.usable if board.usable is not None else board.ran
    return trials is not None and trials > board.cases


def _expected_loose(held: ReportInput, first: str, second: str, key: str, row: ComparisonRow) -> list[str]:
    """The arms whose trials may make a row's interval too narrow, read from the input: on a paired row both; on an
    independent row each whose count counts trials or observations with no distinct cases stated there."""
    if row.interval is None:
        return []
    if row.paired_with is not None:
        candidates = [first, second]
    else:
        candidates = []
        for arm in (first, second):
            prefix = f"spread.{arm}.{key}.{row.where}"
            try:
                counted = resolve(f"{prefix}.counts", held).value
            except UnresolvableReference:
                continue
            try:
                resolve(f"{prefix}.cases", held)
            except UnresolvableReference:
                if counted in ("trials", "observations"):
                    candidates.append(arm)
    return [arm for arm in candidates if _more_trials_than_cases(held, arm)]


@pytest.mark.parametrize("name", sorted(_inputs()))
def test_a_row_says_what_its_interval_counts_and_when_it_may_be_too_narrow(name: str) -> None:
    held = _inputs()[name]
    for comparison in _comparisons(held):
        first, second = (arm.removeprefix("arm:") for arm in (comparison.first, comparison.second))
        for row in comparison.rows:
            where = f"{comparison.first} - {comparison.second} on {comparison.measure} at {row.where}"
            loose = _expected_loose(held, first, second, comparison.measure, row)
            assert row.may_be_narrow == bool(loose), where
            expected_keys = [
                key
                for arm in loose
                for key in (
                    f"scoreboard.{arm}.{'usable' if held.scoreboard[arm].usable is not None else 'ran'}",
                    f"scoreboard.{arm}.cases",
                )
            ]
            assert [figure.key for figure in row.nested] == expected_keys, where
            for figure in row.nested:
                assert resolve(figure.key, held).value == figure.value
            assert row.counted_unstated == (row.interval is not None and row.paired_with is not None), where
            if row.interval is not None and row.paired_with is None:
                assert all(figure.key.endswith(".counts") for figure in row.counts), where
            else:
                assert row.counts == [], where
        assert any("counted_unstated" in note for note in comparison.notes) == any(
            row.counted_unstated for row in comparison.rows
        )
        assert any("may_be_narrow" in note for note in comparison.notes) == any(
            row.may_be_narrow for row in comparison.rows
        )


def test_r4s_paired_contrast_on_replies_in_conversations_is_marked_as_possibly_too_narrow() -> None:
    """The W2 regression prototype's r4 editor finding: the paired interval reads per reply, with every arm's replies
    nested in fewer conversations; the row says it may be too narrow and cites the trials and cases."""
    held = fixture_inputs()["r4"]
    contrast = next(_paired_contrasts(held))
    control = next(a.id for a in held.arms if a.role == "control")
    comparison = arms_compare(held, [contrast.arm, control], contrast.measure)
    row = comparison.rows[0]
    assert row.paired_with == f"arm:{contrast.arm}" and row.counted_unstated and row.may_be_narrow
    trials, cases = (figure.value for figure in row.nested[:2])
    assert isinstance(trials, int | float) and isinstance(cases, int | float) and trials > cases


def test_the_narrow_note_says_a_may_and_never_that_the_cases_would_widen_a_paired_interval() -> None:
    """The W2c finding: the note said "read on the cases it would be wider" of every marked row, though a paired row's
    unit is unstated and Kairo's per-case contrast is already over the cases. It says the interval may be too narrow
    if it counts trials nested in cases, in the phrase the page's note shares."""
    held = fixture_inputs()["r4"]
    contrast = next(_paired_contrasts(held))
    control = next(a.id for a in held.arms if a.role == "control")
    comparison = arms_compare(held, [contrast.arm, control], contrast.measure)
    assert comparison.rows[0].may_be_narrow and comparison.rows[0].counted_unstated
    (note,) = [note for note in comparison.notes if "may_be_narrow" in note]
    assert "may be too narrow if these intervals count trials nested in cases" in note
    assert "would be wider" not in note and "counts trials, observations or an unstated unit" not in note
    assert "unstated" in note and "may already be read over the cases" in note


def test_a_row_whose_arms_ran_each_case_once_is_never_marked_as_possibly_too_narrow() -> None:
    """The control: with every arm's trials equal to its cases there is no nesting to narrow an interval."""
    data = reference_data("r4")
    data["scoreboard"] = {arm: {**row, "ran": row["cases"]} for arm, row in data["scoreboard"].items()}
    held = ReportInput.model_validate(data)
    contrast = next(_paired_contrasts(held))
    control = next(a.id for a in held.arms if a.role == "control")
    comparison = arms_compare(held, [contrast.arm, control], contrast.measure, limit=WHOLE)
    assert comparison.rows[0].counted_unstated
    assert not any(row.may_be_narrow for row in comparison.rows)
    assert not any("may_be_narrow" in note for note in comparison.notes)
