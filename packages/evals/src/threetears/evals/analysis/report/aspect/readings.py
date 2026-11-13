"""The readings by arm, by stratum and by a classifier's label, under the contrasts."""

from __future__ import annotations

from collections.abc import Mapping

from threetears.evals.analysis.arms import surface_order
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.cells import cell_ref
from threetears.evals.analysis.quantities import PERCENT, delta_unit
from threetears.evals.analysis.report.aspect.common import _at, _cell_namer, _interval, _note, _stratum_word, _unit_on
from threetears.evals.analysis.report.figures import (
    NO_SUCCESSFUL_RESULTS,
    SHARE,
    MeasureRuler,
    Rulers,
    surface_order_sentence,
)
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.model import (
    CellValue,
    DisclosureBlock,
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
    table_row,
)
from threetears.evals.analysis.report.unit_once import interval_beside
from threetears.evals.analysis.report.verdict import MIN_CASES
from threetears.evals.analysis.report.words import (
    CASES_READING,
    judged_heading,
    one_figure_each,
    reading_heading,
)
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.analysis.style import ArmName, declared_check_name
from threetears.evals.kernel.analysis_measures import MeasureSummary
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, MATCH_MEASURE, ClassifierStatistic, classifier_label_of
from threetears.evals.kernel.numbers import ABSENT
from threetears.evals.kernel.surface import (
    STRATUM_MIN_CASES,
    CellFacts,
    DecisionSurface,
    JudgedReading,
    PredictionPoweredReading,
    StratumFacts,
)

#: Said beside a stratum's case count when it holds fewer than :data:`STRATUM_MIN_CASES` cases.
TOO_FEW_CASES = "too few cases to read alone"


def _readings_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    names: dict[str, ArmName],
    rulers: Rulers,
    forms: PageForms,
) -> list[ReportBlock]:
    """A one-arm page's readings over all its cases, one row each: the figure with its interval, or its count under
    the case floor, and the cases it rests on — nothing on a page with a contrast or more than one arm, whose
    readings the contrasts state.

    The readings are the ones a contrast would have tested, had there been a control (every measure with a better
    end on a merit axis, a guardrail among them, and every judged dimension), so the one-arm page reads the same
    figures a comparison would. The per-label statistics are the per-label table's, and the strata the stratum
    table's.
    """
    contrasted = any(family.comparisons for family in bundle.multiple_comparisons.families)
    if contrasted or len({cell.variant_key for cell in surface.cells}) != 1:
        return []
    cell_name = _cell_namer(surface, names)
    catalog = bundle.measure_catalog
    rows: list[dict[str, CellValue]] = []
    # Each row's figure is where a finding citing that reading at that cell lands.
    places: list[tuple[CellFacts, str, str, float | None]] = []
    for cell in surface.cells:
        label = cell_name(cell.variant_key, cell.apparatus_class_id)
        measures = {summary.name: summary for summary in cell.measures.measures}
        readings = [
            name
            for name, descriptor in catalog.items()
            if name in measures
            and descriptor.higher_is_better is not None
            and (descriptor.merit_axis is not None or descriptor.guardrail)
            and classifier_label_of(name) is None
            and not measures[name].texts
            and not measures[name].categories
        ]
        ordered = sorted(readings, key=lambda name: reading_heading(surface, name, declared=bundle.check_name(name)))
        for name in one_figure_each(ordered):
            summary = measures[name]
            places.append((cell, name, "measure", summary.rate if summary.rate is not None else summary.mean))
            rows.append(
                {
                    "arm": label,
                    "reading": rulers.of(name, "measure").scaled(
                        reading_heading(surface, name, declared=bundle.check_name(name))
                    ),
                    "value": _value_text(summary, rulers.of(name, "measure"), forms),
                    "cases": summary.n_independent or None,
                }
            )
        for reading in sorted(cell.judged, key=lambda reading: judged_heading(reading.dimension)):
            ruler = rulers.of(reading.dimension, "judged")
            places.append((cell, reading.dimension, "judged", reading.mean))
            rows.append(
                {
                    "arm": label,
                    "reading": judged_heading(reading.dimension),
                    "value": None
                    if reading.mean is None
                    else _figure_text(
                        reading.mean, reading.sem, ruler, forms, measure=reading.dimension, kind="judged"
                    ),
                    "cases": reading.n_independent or None,
                }
            )
    if not rows:
        return []
    columns = [
        *([TableColumn(key="arm", header="Arm")] if len(surface.cells) > 1 else []),
        TableColumn(key="reading", header="Reading", term="reading"),
        TableColumn(key="value", header="Value", term="interval"),
        TableColumn(key="cases", header="Cases", term="cases"),
    ]
    return [
        TableBlock(
            section="contrasts",
            name="readings",
            title="All cases",
            columns=columns,
            rows=[
                {
                    column.key: forms.of_measure(
                        _at(TableCell(value=row["value"]), surface, *place), measure=place[1], kind=place[2]
                    )
                    if column.key == "value"
                    else forms.count_cell(row["cases"])
                    if column.key == "cases"
                    else TableCell(value=row[column.key])
                    for column in columns
                }
                for row, place in zip(rows, places, strict=True)
            ],
            order="each measure by name, then each judged dimension",
            total_rows=len(rows),
        )
    ]


def _value_text(summary: MeasureSummary, ruler: MeasureRuler, forms: PageForms) -> str:
    """One measure's figure as a reading's value cell: a rate as a count under :data:`MIN_CASES` cases, else the
    figure with its interval (``98.9% [88.1%, 100%]``), or with its spread where no interval was computed."""
    name = summary.name
    if summary.rate is not None:
        if summary.n_independent < MIN_CASES:
            return f"{forms.hits(summary.rate, summary.n, measure=name, kind='measure').text} of {summary.n}"
        rate = forms.ruled_level(SHARE, summary.rate, measure=name, kind="measure").text
        return f"{rate}{_interval(summary.ci_low, summary.ci_high, SHARE, forms, measure=name, kind='measure')}"
    if summary.mean is None:
        return ABSENT
    if summary.ci_low is not None and summary.ci_high is not None:
        mean = _unit_on(forms.ruled_level(ruler, summary.mean, measure=name, kind="measure").text, ruler)
        return f"{mean}{_interval(summary.ci_low, summary.ci_high, ruler, forms, measure=name, kind='measure')}"
    return _figure_text(summary.mean, summary.sem, ruler, forms, measure=name, kind="measure")


def _figure_text(
    mean: float, sem: float | None, ruler: MeasureRuler, forms: PageForms, *, measure: str, kind: str
) -> str:
    """A mean on its ruler with its standard error beside it, where one was estimated."""
    spread = (
        f" ± {forms.ruled_spread(ruler, sem, mean=mean, measure=measure, kind=kind).text}" if sem is not None else ""
    )
    return f"{forms.ruled_level(ruler, mean, measure=measure, kind=kind).text}{spread}"


def _strata_blocks(
    surface: DecisionSurface,
    arms: dict[str, ArmName],
    rulers: Rulers,
    forms: PageForms,
    *,
    check_names: Mapping[str, str],
    stratum_names: Mapping[str, str],
) -> list[ReportBlock]:
    """Each arm's figures per stratum beside its pooled figure — or nothing, when no case declares a stratum."""
    control = surface.control_variant_key
    cells = surface_order(
        (cell for cell in surface.cells if cell.strata),
        control=control,
        names={key: name.label for key, name in arms.items()},
    )
    if not cells:
        return []
    cell_name = _cell_namer(surface, arms)
    labels = {
        cell_ref(cell.variant_key, cell.apparatus_class_id): cell_name(cell.variant_key, cell.apparatus_class_id)
        for cell in surface.cells
    }
    names = sorted(
        {stratum.stratum for cell in cells for stratum in cell.strata}, key=lambda name: (name is None, name or "")
    )
    columns = {name: f"stratum_{index}" for index, name in enumerate(names)}
    # Where every stratum is too small to read alone, one line says so and no cell repeats it.
    sizes = [stratum.n_cases for cell in cells for stratum in cell.strata]
    all_thin = max(sizes) < STRATUM_MIN_CASES
    rows: list[dict[str, CellValue]] = []
    thin: list[str] = []
    for cell in cells:
        arm = labels[cell_ref(cell.variant_key, cell.apparatus_class_id)]
        arm = f"▸ {arm}" if cell.variant_key == control else arm
        by_name = {stratum.stratum: stratum for stratum in cell.strata}
        cases: dict[str, CellValue] = {
            "arm": arm,
            "reading": CASES_READING,
            "all": _cases_text(cell.n_observations, cell.n_cases),
        }
        for name in names:
            cases[columns[name]] = None
        for name, stratum in by_name.items():
            cases[columns[name]] = _cases_text(stratum.n_observations, stratum.n_cases)
            if stratum.n_cases < STRATUM_MIN_CASES and not all_thin:
                cases[columns[name]] = f"{cases[columns[name]]} — {TOO_FEW_CASES}"
                plural = "" if stratum.n_cases == 1 else "s"
                thin.append(
                    f"{_stratum_word(name, stratum_names)} in {arm.removeprefix('▸ ')} ({stratum.n_cases} case{plural})"
                )
        rows.append(cases)
        rows.extend(_measure_rows(arm, cell, by_name, columns, surface, rulers, forms, check_names=check_names))
        rows.extend(_judged_rows(arm, cell, by_name, columns, rulers, forms))
    order = (
        surface_order_sentence(has_control_row=any(cell.variant_key == control for cell in cells))
        + " Within each arm: its cases, then each measure by name, then each judged dimension."
    )
    table = TableBlock(
        section="contrasts",
        name="strata",
        title="By stratum",
        columns=[
            TableColumn(key="arm", header="Arm", term="control" if control is not None else None),
            TableColumn(key="reading", header="Reading"),
            TableColumn(key="all", header="All cases", term="cases"),
            *(
                TableColumn(key=columns[name], header=_stratum_word(name, stratum_names), term="stratum")
                for name in names
            ),
        ],
        rows=[table_row(row) for row in rows],
        order=order,
        total_rows=len(rows),
        details=True,
        key_columns=2,
    )
    blocks: list[ReportBlock] = [_by_stratum_row(table) if len(names) > _STRATA_COLUMNS else table]
    if all_thin:
        low, high = min(sizes), max(sizes)
        span = (
            f"{low} case{'' if low == 1 else 's'}" if low == high else f"{interval_beside(str(low), str(high))} cases"
        )
        blocks.insert(
            0,
            DisclosureBlock(
                section="contrasts",
                source="strata",
                text=f"Every stratum holds {span}; a stratum is read alone from {STRATUM_MIN_CASES} cases.",
            ),
        )
    elif thin:
        # A list of counts belongs beside the table it qualifies, in the methods' fold, not in a line of prose.
        blocks.append(
            _note(
                f"Too few cases to read alone (a stratum needs at least {STRATUM_MIN_CASES} cases), so read their "
                "intervals: " + "; ".join(thin) + ".",
                "strata",
            )
        )
    return blocks


#: The most strata the table by stratum lays out as columns, beside the arm, the reading and the pooled figure; more,
#: and a stratum is a row, each arm a column, so the table stays as wide as the arms, never as the strata.
_STRATA_COLUMNS = 6


def _by_stratum_row(table: TableBlock) -> TableBlock:
    """The table by stratum turned a stratum per row: all cases first, then each stratum, each reading under it in the
    arms' own order, every arm a column holding its cell as it was; a row no arm holds is left out."""
    arms = list(dict.fromkeys(row["arm"].text for row in table.rows))
    readings = list(dict.fromkeys(row["reading"].text for row in table.rows))
    held = {(row["arm"].text, row["reading"].text): row for row in table.rows}
    places = [column for column in table.columns if column.key not in ("arm", "reading")]
    rows: list[dict[str, TableCell]] = []
    for place in places:
        for reading in readings:
            cells = {
                f"arm_{at}": held[arm, reading][place.key] if (arm, reading) in held else TableCell(value=None)
                for at, arm in enumerate(arms)
            }
            if all(cell.value is None for cell in cells.values()):
                continue
            rows.append({"stratum": TableCell(value=place.header), "reading": TableCell(value=reading), **cells})
    control = next((column.term for column in table.columns if column.key == "arm"), None)
    return table.model_copy(
        update={
            "columns": [
                TableColumn(key="stratum", header="Stratum", term="stratum"),
                TableColumn(key="reading", header="Reading"),
                *(
                    TableColumn(key=f"arm_{at}", header=arm, term=control if arm.startswith("▸ ") else None)
                    for at, arm in enumerate(arms)
                ),
            ],
            "rows": rows,
            "order": "All cases, then each stratum by name; within each, its cases, then each measure by name, then "
            "each judged dimension. The arms are the columns, in the order the arms' rows would take: "
            + table.order.split(" Within each arm")[0],
            "total_rows": len(rows),
        }
    )


def _cases_text(n_trials: int, n_cases: int | None) -> str:
    """How many cases and trials a figure rests on — cases first, since they are the independent draws; one case run
    once is its count alone (``1 case``), and a count of one is never plural."""
    trials = f"{n_trials} trial{'' if n_trials == 1 else 's'}"
    if n_cases is None:
        return f"{trials}, cases unrecorded"
    if n_cases == 1 and n_trials == 1:
        return "1 case"
    return f"{n_cases} case{'' if n_cases == 1 else 's'}, {trials}"


def _measure_rows(
    arm: str,
    cell: CellFacts,
    by_name: dict[str | None, StratumFacts],
    columns: dict[str | None, str],
    surface: DecisionSurface,
    rulers: Rulers,
    forms: PageForms,
    *,
    check_names: Mapping[str, str],
) -> list[dict[str, CellValue]]:
    """One row per measure the arm or any of its strata holds — every one but a text measure."""
    pooled = {summary.name: summary for summary in cell.measures.measures}
    per_stratum = {
        name: {summary.name: summary for summary in stratum.measures.measures} for name, stratum in by_name.items()
    }
    measures = sorted(
        {
            summary.name
            for summaries in (pooled, *per_stratum.values())
            for summary in summaries.values()
            if not summary.texts
        }
    )
    # A classifier's accuracy is its trials' label match: one row, named as the accuracy, in the match's own form,
    # which counts a stratum too small to estimate a rate on.
    doubled = ACCURACY_MEASURE in measures and MATCH_MEASURE in measures
    rows: list[dict[str, CellValue]] = []
    for measure in measures:
        if doubled and measure == ACCURACY_MEASURE:
            continue
        named = ACCURACY_MEASURE if doubled and measure == MATCH_MEASURE else measure
        ruler = rulers.of(measure, "measure")

        def figure(summaries: dict[str, MeasureSummary], facts_of: CellFacts | StratumFacts) -> CellValue:
            if measure in summaries:
                return _summary_text(summaries[measure], ruler, forms)
            return NO_SUCCESSFUL_RESULTS if facts_of.all_failed else None

        row: dict[str, CellValue] = {
            "arm": arm,
            "reading": ruler.labelled(
                reading_heading(surface, named, declared=declared_check_name(check_names, named))
            ),
            "all": figure(pooled, cell),
        }
        for name in columns:
            row[columns[name]] = figure(per_stratum[name], by_name[name]) if name in per_stratum else None
        rows.append(row)
    return rows


def _judged_rows(
    arm: str,
    cell: CellFacts,
    by_name: dict[str | None, StratumFacts],
    columns: dict[str | None, str],
    rulers: Rulers,
    forms: PageForms,
) -> list[dict[str, CellValue]]:
    """One row per judged dimension the arm or any of its strata was scored on."""
    pooled = {reading.dimension: reading for reading in cell.judged}
    per_stratum = {
        name: {reading.dimension: reading for reading in stratum.judged} for name, stratum in by_name.items()
    }
    dimensions = sorted({dimension for readings in (pooled, *per_stratum.values()) for dimension in readings})
    rows: list[dict[str, CellValue]] = []
    for dimension in dimensions:
        ruler = rulers.of(dimension, "judged")
        row: dict[str, CellValue] = {
            "arm": arm,
            "reading": judged_heading(dimension),
            "all": _judged_text(pooled[dimension], ruler, forms) if dimension in pooled else None,
        }
        for name in columns:
            readings = per_stratum.get(name, {})
            row[columns[name]] = _judged_text(readings[dimension], ruler, forms) if dimension in readings else None
        rows.append(row)
    return rows


def _summary_text(summary: MeasureSummary, ruler: MeasureRuler, forms: PageForms) -> str:
    """One measure's figure with the n it rests on, on its ruler: a rate as a count below :data:`MIN_CASES` cases.

    A rate is a share by its shape, so it reads as a percent (:data:`SHARE`) whatever the catalog said of its
    measure; a mean is stated with its unit on the figure, and its spread as a bare number beside it on the same
    scale, money in the mean's one form (``98.9% ± 6.5``, ``$0.046 ± $0.007``).
    """
    n = _n_text(summary.n, summary.n_independent)
    name = summary.name
    if summary.rate is not None:
        if summary.n_independent < MIN_CASES:
            return f"{forms.hits(summary.rate, summary.n, measure=name, kind='measure').text} of {summary.n} {n}"
        rate = forms.ruled_level(SHARE, summary.rate, measure=name, kind="measure").text
        return f"{rate}{_interval(summary.ci_low, summary.ci_high, SHARE, forms, measure=name, kind='measure')} {n}"
    if summary.mean is not None:
        return f"{_figure_text(summary.mean, summary.sem, ruler, forms, measure=name, kind='measure')} {n}"
    counts = sorted(summary.categories.items(), key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{category}: {count}" for category, count in counts) + f" {n}"


def _judged_text(reading: JudgedReading, ruler: MeasureRuler, forms: PageForms) -> str:
    """One judged dimension's mean, with its standard error and the scores it rests on, on the dimension's ruler."""
    if reading.mean is None:
        return f"no score (n={reading.n})"
    figure = _figure_text(reading.mean, reading.sem, ruler, forms, measure=reading.dimension, kind="judged")
    text = f"{figure} {_n_text(reading.n, reading.n_independent)}"
    if (powered := reading.prediction_powered) is not None:
        text += f"; {_prediction_powered_text(powered, ruler, forms, reading.dimension)}"
    return text


def _prediction_powered_text(
    powered: PredictionPoweredReading, ruler: MeasureRuler, forms: PageForms, dimension: str
) -> str:
    """The prediction-powered estimate beside a judged mean: its value and interval, or why it is not available."""
    rated = f"{powered.n_labelled} rated by people"
    if powered.mean is None:
        return f"with people's ratings: not available ({rated}, under {powered.min_labelled})"
    rectifier = forms.in_unit(
        (powered.rectifier or 0.0) * ruler.factor,
        delta_unit(ruler.unit),
        role="change",
        measure=dimension,
        kind="judged",
    ).text
    mean = forms.ruled_level(ruler, powered.mean, measure=dimension, kind="judged").text
    interval = _interval(powered.ci_low, powered.ci_high, ruler, forms, measure=dimension, kind="judged")
    return f"with people's ratings: {mean}{interval} (rectifier {rectifier}; {rated})"


def _n_text(n: int, n_cases: int) -> str:
    """A figure's n, and where cases repeat, how many cases it is over."""
    if 0 < n_cases < n:
        return f"(n={n} over {n_cases} case{'' if n_cases == 1 else 's'})"
    return f"(n={n})"


#: The columns of the per-label table after the label and the arm, by the statistic each holds.
_LABEL_STATISTICS: tuple[tuple[ClassifierStatistic, str], ...] = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
)


def _label_blocks(
    surface: DecisionSurface, names: dict[str, ArmName], rulers: Rulers, forms: PageForms
) -> list[ReportBlock]:
    """A classifier's per-label precision, recall and F1, every arm's, as one table — or nothing for no classifier."""
    control = surface.control_variant_key
    cell_name = _cell_namer(surface, names)
    labels = {
        cell_ref(cell.variant_key, cell.apparatus_class_id): cell_name(cell.variant_key, cell.apparatus_class_id)
        for cell in surface.cells
    }
    figures: dict[str, dict[str, dict[str, MeasureSummary]]] = {}
    for cell in surface.cells:
        ref = cell_ref(cell.variant_key, cell.apparatus_class_id)
        for summary in cell.measures.measures:
            if (classifier := classifier_label_of(summary.name)) is None:
                continue
            statistic, label = classifier
            figures.setdefault(label, {}).setdefault(ref, {})[statistic] = summary
    if not figures:
        return []
    order = sorted(
        ((cell_ref(cell.variant_key, cell.apparatus_class_id), cell) for cell in surface.cells),
        key=lambda item: (item[1].variant_key != control, labels[item[0]], item[1].apparatus_class_id),
    )
    classifying = {ref for by_cell in figures.values() for ref in by_cell}
    rows: list[dict[str, CellValue]] = []
    missing = False
    for label in sorted(figures):
        for ref, cell in order:
            if ref not in classifying:
                continue
            by_statistic = figures[label].get(ref, {})
            row: dict[str, CellValue] = {
                "label": label,
                "arm": f"▸ {labels[ref]}" if cell.variant_key == control else labels[ref],
            }
            for statistic, _ in _LABEL_STATISTICS:
                figure = by_statistic.get(statistic)
                row[statistic] = (
                    None if figure is None else _summary_text(figure, rulers.of(figure.name, "measure"), forms)
                )
                missing = missing or figure is None
            rows.append(row)
    said = [
        "Precision is counted over the trials an arm predicted as the label and recall over those expected as it, each "
        f"with its {forms.in_unit(INTERVAL_LEVEL * 100, PERCENT, role='confidence').text} Wilson interval over the "
        "arm's cases; below "
        f"{MIN_CASES} cases a rate is stated as a count. F1 has no interval by construction."
    ]
    if missing:
        said.append(
            "A label an arm never predicted has no precision, one it never met has no recall, and either has no F1: "
            "each is shown as —."
        )
    return [
        TableBlock(
            section="contrasts",
            name="labels",
            title="Per label",
            columns=[
                TableColumn(key="label", header="Label", term="label"),
                TableColumn(key="arm", header="Arm", term="control" if control is not None else None),
                *(TableColumn(key=statistic, header=header) for statistic, header in _LABEL_STATISTICS),
            ],
            rows=[table_row(row) for row in rows],
            order="by label, then the control first and every other arm by name",
            total_rows=len(rows),
            details=True,
        ),
        _note(" ".join(said)),
    ]


__all__ = [
    "TOO_FEW_CASES",
]
