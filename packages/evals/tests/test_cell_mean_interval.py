"""An overall that is the plain mean of its target-matrix cells reads an interval derived from theirs.

Contracts held here (``threetears.evals.analysis.cell_mean``, read through the figure keys, the gate check and the
decision's implied caveats):

- **r3 derives one**: its balanced accuracy, the plain mean of the nine cells' accuracies, gets standard error
  ``sqrt(sum(se_i ** 2)) / 9`` at the cells' 95%, checked against the arithmetic done by hand on the fixture's own
  half-widths; ``overall_interval.*`` names it, every figure marked derived; the overall gate reads it (its ends by
  ``criteria.overall.<arm>.low`` and ``.high``, derived; a borderline reason calls it derived), so no
  ``point_estimate:overall`` caveat is implied;
- **qualifying is structural**: renamed, with its definition's words and the eval's kind changed, the same figures
  derive the same interval; a weighted mean (traffic accuracy) and a figure the cells do not average to (macro F1)
  derive none;
- **nothing is derived without every cell's spread**: one cell stating none for every arm derives nothing and the
  caveat stays; one cell stating none for one arm derives nothing for that arm alone;
- **refusals**: cells at a level other than the overall's, cells whose level is unstated, and cells that do not cover
  the target matrix are refused, each saying why;
- **a case-weighted mean that fits as well**: where the cells state differing cases and their case-weighted mean fits
  the overall as well as their plain one (close figures, or near-equal counts), the overall derives from the wider of
  the two standard errors, checked by hand, and its basis says it fits both; cells of one count read the plain one;
- **a cell figure that is no number** is absent for its own arm alone, which derives nothing; the others derive;
- **settling reads it too**: Nano-XS few-shot's borderline on the overall gate, which rests on the derived interval,
  gets settling sizes and a wrong-side share checked by hand, its spread's basis naming the cells; an overall that
  does not qualify still finds no spread.
"""

from __future__ import annotations

import math
import re
from typing import Any

import pytest

from packages.evals.tests.analyst_support import fixture_inputs
from threetears.evals.analysis.analyst.cited import cite, cite_if
from threetears.evals.analysis.analyst.settle import settle_cases
from threetears.evals.analysis.errors import UnstatedFigure
from threetears.evals.analysis.report.figure_keys import resolve
from threetears.evals.analysis.report_decision import implied_caveats
from threetears.evals.analysis.report_gates import gate_check
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_margins import criterion_figures

#: The two-sided normal quantile at 95%, as a hand calculation writes it.
Z95 = 1.959964

#: Each cell's accuracy half-width for two of r3's arms, copied from the fixture's cells in its order, and the overall
#: balanced accuracy the fixture states for each.
HALF_WIDTHS = {
    "brisk-s.few": (0.8, 0.3, 0.7, 0.9, 0.6, 0.9, 1.1, 1.4, 1.2),
    "nano-xs.few": (0.7, 0.2, 0.7, 0.9, 0.6, 0.9, 1.1, 1.3, 1.4),
}
OVERALL = {"brisk-s.few": 90.7, "nano-xs.few": 90.1}

#: By hand: sqrt(7.81) / 9 and sqrt(7.86) / 9, the half-width of a plain mean of nine independent cells, each cell's
#: half-width at the one level they share.
HAND_HALF_WIDTH = {"brisk-s.few": 0.310516, "nano-xs.few": 0.311508}


def _r3() -> ReportInput:
    return fixture_inputs()["r3"]


def _changed(change: Any) -> ReportInput:
    """r3 with its JSON changed in place by ``change``."""
    data = _r3().model_dump(mode="json", by_alias=True, exclude_none=True)
    change(data)
    return ReportInput.model_validate(data)


def _key(arm: str, field: str, measure: str = "balanced_accuracy") -> str:
    return f"overall_interval.{measure}.{arm}.{field}"


@pytest.mark.parametrize("arm", list(HALF_WIDTHS))
def test_r3_s_balanced_accuracy_reads_the_interval_its_cells_give(arm: str) -> None:
    held = _r3()
    # The hand values are the fixture's: its cells' half-widths, squared, summed and rooted over nine, and its
    # stated overall the cells' plain mean.
    stated = tuple(float(cell.per_arm[arm]["accuracy"].half_width or 0) for cell in held.cells.values())
    assert stated == HALF_WIDTHS[arm]
    assert math.sqrt(sum(width**2 for width in stated)) / 9 == pytest.approx(HAND_HALF_WIDTH[arm], abs=1e-6)
    means = [float(cell.per_arm[arm]["accuracy"].mean or 0) for cell in held.cells.values()]
    assert sum(means) / 9 == pytest.approx(OVERALL[arm], abs=0.1)
    half = HAND_HALF_WIDTH[arm]
    assert cite(_key(arm, "half_width"), held).value == pytest.approx(half, abs=1e-5)
    assert cite(_key(arm, "se"), held).value == pytest.approx(half / Z95, abs=1e-5)
    low, high = cite(_key(arm, "interval"), held).value  # type: ignore[misc]
    assert (low, high) == pytest.approx((OVERALL[arm] - half, OVERALL[arm] + half), abs=1e-5)
    assert cite(_key(arm, "low"), held).value == pytest.approx(low)
    assert cite(_key(arm, "high"), held).value == pytest.approx(high)
    assert cite(_key(arm, "mean"), held).value == OVERALL[arm]
    assert cite(_key(arm, "level"), held).reads == "95%"
    assert cite(_key(arm, "cells"), held).value == 9
    assert cite(_key(arm, "of"), held).value == "Accuracy"
    basis = cite(_key(arm, "basis"), held).value
    assert isinstance(basis, str) and "plain mean of Accuracy" in basis and "independent" in basis
    fields = ("mean", "se", "half_width", "interval", "low", "high", "level", "of", "cells", "basis")
    assert all(cite(_key(arm, field), held).derived for field in fields)


def test_the_overall_gate_reads_the_derived_interval_and_implies_no_point_estimate() -> None:
    held = _r3()
    gate = next(criterion for criterion in held.criteria if criterion.id == "overall")
    (figure,) = criterion_figures(held, gate, "brisk-s.few")
    assert figure.derived and figure.ci == pytest.approx((90.7 - 0.310516, 90.7 + 0.310516), abs=1e-5)
    for end in ("low", "high"):
        cited = cite(f"criteria.overall.brisk-s.few.{end}", held)
        assert cited.derived and cited.value == cite(_key("brisk-s.few", end), held).value
    assert cite("criteria.overall.brisk-s.few.value", held).derived is False
    assert not any(caveat.id == "point_estimate:overall" for caveat in implied_caveats(held))
    # Nano-XS few-shot's 90.1 clears the 90 gate on its point estimate, and the derived interval reaches below it.
    check = gate_check(held, "nano-xs.few", criterion="overall")
    assert check.status == "borderline"
    assert check.borders == ("Overall accuracy 90.1%, derived interval 89.8–90.4%",)


def test_qualifying_reads_the_numbers_never_the_name_the_definition_or_the_kind() -> None:
    def disguise(data: dict[str, Any]) -> None:
        data["measures"]["score_x"] = data["measures"].pop("balanced_accuracy")
        data["measures"]["score_x"].pop("means", None)
        data["measures"]["score_x"]["label"] = "Score X"
        for criterion in data["criteria"]:
            if criterion["measure"] == "balanced_accuracy":
                criterion["measure"] = "score_x"
        data["meta"]["kind"] = "measurement"
        data["meta"].pop("design_shape", None)

    disguised = _changed(disguise)
    held = _r3()
    for arm in HALF_WIDTHS:
        assert cite(_key(arm, "interval", "score_x"), disguised).value == cite(_key(arm, "interval"), held).value


@pytest.mark.parametrize(
    ("measure", "why"),
    [
        ("traffic_accuracy", "is not the plain mean of any measure's cells"),
        ("macro_f1", "is not the plain mean of any measure's cells"),
    ],
)
def test_a_weighted_mean_or_a_figure_its_cells_do_not_average_to_derives_none(measure: str, why: str) -> None:
    held = _r3()
    assert len({cell.n_cases for cell in held.cells.values()}) > 1, "the control: the cells' cases differ"
    with pytest.raises(UnstatedFigure, match=why):
        resolve(_key("brisk-s.few", "interval", measure), held)
    assert cite_if(_key("brisk-s.few", "interval", measure), held) is None


def _without_spread(cell: str, arms: tuple[str, ...] | None = None) -> ReportInput:
    def strip(data: dict[str, Any]) -> None:
        for arm, figures in data["cells"][cell]["per_arm"].items():
            if arms is None or arm in arms:
                figures["accuracy"].pop("half_width")

    return _changed(strip)


def test_one_cell_stating_no_spread_derives_nothing_and_the_caveat_stays() -> None:
    held = _without_spread("complex|news")
    with pytest.raises(UnstatedFigure, match=re.escape("cell 'complex|news' states no spread of its own")):
        resolve(_key("brisk-s.few", "interval"), held)
    gate = next(criterion for criterion in held.criteria if criterion.id == "overall")
    (figure,) = criterion_figures(held, gate, "brisk-s.few")
    assert (figure.ci, figure.half_width, figure.derived) == (None, None, False)
    (caveat,) = [caveat for caveat in implied_caveats(held) if caveat.id == "point_estimate:overall"]
    assert caveat.arms == ("brisk-s.few",) and caveat.at == (("brisk-s.few", None, None),)


def test_one_cell_stating_no_spread_for_one_arm_derives_nothing_for_that_arm_alone() -> None:
    held = _without_spread("simple|conv", ("brisk-s.few",))
    assert cite_if(_key("brisk-s.few", "interval"), held) is None
    assert cite_if(_key("nano-xs.few", "interval"), held) is not None


#: r3's cells' stated cases, in its order, and two near-balanced sets a quota grid that lost a few cases might state.
R3_CASES = (3120, 14880, 4410, 4260, 7350, 5190, 3440, 1960, 3600)
NEAR_EQUAL = (500, 498, 502, 497, 501, 499, 500, 503, 496)

#: By hand, on brisk-s.few's cell half-widths: sqrt(sum((n_i / sum(n) * hw_i) ** 2)), the case-weighted mean's
#: half-width, with r3's own cases (narrower than the plain mean's 0.310516) and with the near-equal ones (wider).
WEIGHTED_HALF_WIDTH = {R3_CASES: 0.238462, NEAR_EQUAL: 0.310640}

#: What ``overall_interval.<measure>.<arm>.weights`` reads, plain and fitting both.
PLAIN = "the plain mean of the cells"
BOTH = "both the plain and the case-weighted mean of the cells; the interval is the wider of the two readings'"


def _weighted_half_width(cases: tuple[int, ...], widths: tuple[float, ...]) -> float:
    total = sum(cases)
    return math.sqrt(sum((count / total * width) ** 2 for count, width in zip(cases, widths, strict=True)))


def _with_cases(cases: tuple[int, ...]) -> Any:
    def change(data: dict[str, Any]) -> None:
        for cell, count in zip(data["cells"].values(), cases, strict=True):
            cell["n_cases"] = count

    return change


def _close(cases: tuple[int, ...], *, weighted: bool) -> ReportInput:
    """r3 with every arm's cells at 90.1 and 90.2 alternately, stating ``cases``, and each arm's overall their
    case-weighted mean (``weighted``) or their plain mean, rounded to one decimal as the input states figures."""

    def close(data: dict[str, Any]) -> None:
        _with_cases(cases)(data)
        figures = [90.1 if at % 2 == 0 else 90.2 for at in range(len(cases))]
        for arm in data["measures"]["balanced_accuracy"]["per_arm"]:
            for cell, figure in zip(data["cells"].values(), figures, strict=True):
                cell["per_arm"][arm]["accuracy"]["mean"] = figure
            weights = cases if weighted else (1,) * len(cases)
            mean = sum(w * f for w, f in zip(weights, figures, strict=True)) / sum(weights)
            data["measures"]["balanced_accuracy"]["per_arm"][arm]["mean"] = round(mean, 1)

    return _changed(close)


def test_the_hand_weighted_half_widths_are_the_fixture_s() -> None:
    held = _r3()
    assert tuple(cell.n_cases for cell in held.cells.values()) == R3_CASES
    for cases, expected in WEIGHTED_HALF_WIDTH.items():
        assert _weighted_half_width(cases, HALF_WIDTHS["brisk-s.few"]) == pytest.approx(expected, abs=1e-6)
    assert cite(_key("brisk-s.few", "weights"), held).value == PLAIN


def test_an_overall_a_case_weighted_mean_fits_as_well_derives_from_the_wider_standard_error() -> None:
    """Cells within rounding of one another (90.1 and 90.2) and an overall that is their case-weighted mean: the plain
    mean fits it too, so it is either. It derives, from the wider of the two standard errors (here the plain one,
    0.310516 against the case-weighted 0.238462 on r3's own cases), and its basis says it fits both."""
    held = _close(R3_CASES, weighted=True)
    arm = "brisk-s.few"
    widest = max(HAND_HALF_WIDTH[arm], WEIGHTED_HALF_WIDTH[R3_CASES])
    assert cite(_key(arm, "half_width"), held).value == pytest.approx(widest, abs=1e-5)
    assert cite(_key(arm, "weights"), held).value == BOTH
    basis = cite(_key(arm, "basis"), held).value
    assert isinstance(basis, str) and "fits both a plain and a case-weighted mean" in basis and "wider" in basis
    spread = cite(f"spread.{arm}.balanced_accuracy.overall.basis", held).value
    assert isinstance(spread, str) and "both their plain and their case-weighted mean" in spread
    gate = next(criterion for criterion in held.criteria if criterion.id == "overall")
    (figure,) = criterion_figures(held, gate, arm)
    assert figure.derived and figure.ci is not None


def test_near_equal_counts_with_spread_apart_figures_derive_from_the_wider_standard_error() -> None:
    """r3's own figures, spread 14 points apart, on cells whose counts are nearly equal (a quota grid that lost a few
    cases): the case-weighted mean sits within rounding of the plain one, so the figure fits both and derives, here
    from the case-weighted standard error, 0.310640 by hand, the wider."""
    held = _changed(_with_cases(NEAR_EQUAL))
    arm = "brisk-s.few"
    assert WEIGHTED_HALF_WIDTH[NEAR_EQUAL] > HAND_HALF_WIDTH[arm], "the control: the case-weighted one is wider"
    assert cite(_key(arm, "half_width"), held).value == pytest.approx(WEIGHTED_HALF_WIDTH[NEAR_EQUAL], abs=1e-5)
    assert cite(_key(arm, "weights"), held).value == BOTH


def test_equal_counts_with_close_figures_derive_the_plain_standard_error() -> None:
    """Cells of one count weigh alike, so their plain and case-weighted means are one reading, and its standard
    error the plain one."""
    held = _close((500,) * 9, weighted=False)
    arm = "brisk-s.few"
    assert cite(_key(arm, "half_width"), held).value == pytest.approx(HAND_HALF_WIDTH[arm], abs=1e-5)
    assert cite(_key(arm, "weights"), held).value == PLAIN
    basis = cite(_key(arm, "basis"), held).value
    assert isinstance(basis, str) and "is the plain mean of Accuracy" in basis and "case-weighted" not in basis


def test_a_cell_figure_that_is_no_number_is_absent_for_its_own_arm_alone() -> None:
    def not_a_number(data: dict[str, Any]) -> None:
        data["cells"]["simple|conv"]["per_arm"]["brisk-s.few"]["accuracy"]["mean"] = float("nan")

    held = _changed(not_a_number)
    with pytest.raises(UnstatedFigure, match="states no spread of its own"):
        resolve(_key("brisk-s.few", "interval"), held)
    nano = cite(_key("nano-xs.few", "half_width"), held).value
    assert nano == pytest.approx(HAND_HALF_WIDTH["nano-xs.few"], abs=1e-5)


def test_cells_at_a_level_other_than_the_overall_s_are_refused() -> None:
    def overall_at_90(data: dict[str, Any]) -> None:
        data["measures"]["balanced_accuracy"]["level"] = "90%"

    held = _changed(overall_at_90)
    with pytest.raises(UnstatedFigure, match="cells at a level other than the overall's are not combined"):
        resolve(_key("brisk-s.few", "interval"), held)
    assert any(caveat.id == "point_estimate:overall" for caveat in implied_caveats(held))

    def overall_at_95(data: dict[str, Any]) -> None:
        data["measures"]["balanced_accuracy"]["level"] = "0.95"

    assert cite_if(_key("brisk-s.few", "interval"), _changed(overall_at_95)) is not None, "one level, two spellings"


def test_cells_whose_level_is_unstated_are_refused() -> None:
    def unstated(data: dict[str, Any]) -> None:
        data["measures"]["accuracy"].pop("level")

    with pytest.raises(UnstatedFigure, match="state no level to build the interval at"):
        resolve(_key("brisk-s.few", "interval"), _changed(unstated))


def test_cells_that_do_not_cover_the_target_matrix_are_refused() -> None:
    def drop(data: dict[str, Any]) -> None:
        data["cells"].pop("complex|news")

    with pytest.raises(UnstatedFigure, match="do not cover the target matrix once each"):
        resolve(_key("brisk-s.few", "interval"), _changed(drop))


#: Nano-XS few-shot's overall cases (its scoreboard's, what its spread's count is) and the one- and two-sided normal
#: quantiles at 95%, as a hand calculation writes them.
NANO_CASES = 48210
Z95_ONE_SIDED = 1.644854


def test_a_borderline_on_the_derived_interval_gets_a_settling_size() -> None:
    """Nano-XS few-shot reads 90.1 against the 90 gate, borderline on the interval its cells give. By hand: one case's
    spread is se * sqrt(48210) with se = hw / 1.96 and hw**2 = 7.86 / 81, so a two-sided interval clears the line at
    48210 * (hw / 0.1)**2 = 48210 * 7.86 / 0.81 = 467,815.6 cases (467,816), a one-sided bound at that times
    (1.644854 / 1.959964)**2 (329,482.8, so 329,483), and a fresh run of today's size reads under the line with
    probability Phi(-0.1 / se) = Phi(-0.6292) = 0.2646."""
    held = _r3()
    (row,) = settle_cases(held, arms=["nano-xs.few"], criterion="overall").rows
    two_sided = NANO_CASES * 7.86 / 0.81
    assert row.cases_needed_two_sided is not None and row.cases_needed is not None and row.wrong_side is not None
    assert row.cases_needed_two_sided.value == math.ceil(two_sided) == 467_816
    assert row.cases_needed.value == pytest.approx(two_sided * (Z95_ONE_SIDED / Z95) ** 2, abs=1)
    se = math.sqrt(7.86) / 9 / Z95
    assert row.wrong_side.value == pytest.approx(0.5 * math.erfc(0.1 / se / math.sqrt(2)), abs=1e-5)
    assert row.wrong_side.value == pytest.approx(0.2646, abs=1e-4)
    assert [figure.value for figure in row.counts] == [NANO_CASES]
    (basis,) = row.basis
    assert basis.derived and isinstance(basis.value, str) and "derived from its target-matrix cells" in basis.value
    assert all(figure.derived for figure in (row.cases_needed, row.cases_needed_two_sided, row.wrong_side))
    spread = cite("spread.nano-xs.few.balanced_accuracy.overall.se", held)
    assert spread.derived and spread.value == pytest.approx(se, abs=1e-6)
    interval = cite("spread.nano-xs.few.balanced_accuracy.overall.interval", held).value
    assert interval == pytest.approx(cite(_key("nano-xs.few", "interval"), held).value), "the gate's interval"


def test_an_overall_that_does_not_qualify_still_finds_no_spread() -> None:
    held = _r3()
    assert cite("spread.nano-xs.few.balanced_accuracy.overall.se", held).derived, "the control: r3's qualifies"
    for measure in ("traffic_accuracy", "macro_f1"):
        assert cite_if(f"spread.nano-xs.few.{measure}.overall.se", held) is None, measure
    stripped = _without_spread("complex|news")
    assert cite_if("spread.nano-xs.few.balanced_accuracy.overall.se", stripped) is None
    assert cite_if("settle.criteria.overall.nano-xs.few.overall.cases_needed", stripped) is None
