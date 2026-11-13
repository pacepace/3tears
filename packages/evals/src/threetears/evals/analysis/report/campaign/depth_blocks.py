"""How deep a campaign's arms ran, for every kind: each arm's cases, trials and repeats on each aspect, pass^k at the
common depth, the opt-in model of pass^k at a depth no arm reached, and what limits each comparison's interval.

A table is printed only where it says more than a sentence would (rule 2 of the senior-report plan):

- the depth table is one sentence where every arm ran each case alike on every aspect but a few, which it names
  (:func:`~threetears.evals.analysis.depth.repeats_sentence`);
- the pass^k depth table is one sentence where pass^k compares pass rates on every aspect, at a common and planned
  depth of 1 with no case left out;
- the precision table drops a column empty on every row, and sits folded under the Methods: what limits an interval is
  how it was read, one step further from the answer than the scorecard;
- what extra repeats cannot lower is said once, in Design & provenance, where a between-case share is read
  (:func:`depth_provenance_blocks`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.bundle.schema import TRIAL_PASS_NAME, AnalysisContextBundle
from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.depth import (
    COMMON_DEPTH_ONE_EVERYWHERE,
    MODEL_ASSUMES,
    REPEATS_FLOOR,
    Precision,
    repeats_sentence,
)
from threetears.evals.analysis.depth_reading import AspectDepth, SubjectDepth, aspect_depth
from threetears.evals.analysis.report.campaign.common import named_arms
from threetears.evals.analysis.report.campaign.names import AspectNames
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.model import (
    DisclosureBlock,
    Report,
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
    table_row,
)
from threetears.evals.kernel.campaign import EvalCampaign

__all__ = ["depth_blocks", "depth_provenance_blocks"]


def _label_of(bundle: AnalysisContextBundle, labels: Mapping[str, str]) -> Callable[[str], str]:
    """What the page calls the arm a variant key names on one aspect: the campaign's label, else its variant's name."""
    names = named_arms(bundle, labels)
    return lambda key: names[key].label if key in names else f"unplaced arm {short_digest(key)}"


def depth_blocks(
    campaign: EvalCampaign,
    readings: CampaignReadings,
    reports: Mapping[str, Report],
    *,
    names: AspectNames,
    forms: PageForms,
) -> list[ReportBlock]:
    """How deep each arm ran on each aspect, what pass^k compares at, and what limits each comparison's interval:
    each a table with one sentence under it where the arms differ, or one sentence where that says all a table would.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :param names: What the page calls each aspect.
    :ptype names: AspectNames
    :param forms: The page's forms.
    :ptype forms: PageForms
    :return: The tables and their sentences; nothing where no arm ran.
    :rtype: list[ReportBlock]
    """
    labels = readings.arm_labels()
    depths = [
        (
            reports[aspect.label].bundle,
            aspect_depth(
                reports[aspect.label].bundle,
                _label_of(reports[aspect.label].bundle, labels),
                name=names(aspect.label),
            ),
        )
        for aspect in campaign.aspects
    ]
    rows = [
        table_row(
            {
                "aspect": names.cell(depth.aspect),
                "arm": arm,
                "cases": forms.count_cell(held.cases),
                "trials": forms.count_cell(held.trials),
                "fewest": forms.count_cell(held.repeats_min),
                "median": forms.count_cell(held.repeats_median),
                "most": forms.count_cell(held.repeats_max),
            }
        )
        for _, depth in depths
        for arm, held in depth.arms.items()
    ]
    if not rows:
        return []
    once = repeats_sentence({names(depth.aspect): depth.arms for _, depth in depths})
    if once is not None:
        blocks: list[ReportBlock] = [DisclosureBlock(section="scorecard", source="comparisons", text=once)]
    else:
        blocks = [*_depth_table(rows), *_differ_lines(depths)]
    blocks += _pass_k_depth_blocks(depths, labels, names, forms)
    # What limits each interval is a method, read one step from the answer only by someone who asks, so it is folded under
    # the page's methods, never on the scorecard.
    blocks += _precision_blocks(readings, names, forms)
    return blocks


def _depth_table(rows: list[dict[str, TableCell]]) -> list[ReportBlock]:
    """Each arm's cases, trials and repeats on each aspect, in one table."""
    return [
        TableBlock(
            section="scorecard",
            name="depth",
            title="Depth",
            columns=[
                TableColumn(key="aspect", header="Aspect", term="aspect"),
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="cases", header="Cases", term="cases"),
                TableColumn(key="trials", header="Trials", term="usable"),
                TableColumn(key="fewest", header="Fewest repeats", term="depth"),
                TableColumn(key="median", header="Median repeats", term="depth"),
                TableColumn(key="most", header="Most repeats", term="depth"),
            ],
            rows=rows,
            order="aspect by aspect, each arm",
            total_rows=len(rows),
        )
    ]


def _differ_lines(depths: Sequence[tuple[AnalysisContextBundle, AspectDepth]]) -> list[ReportBlock]:
    """Under the depth table, each aspect's sentence saying its arms ran to different depths."""
    return [
        DisclosureBlock(section="scorecard", source="comparisons", text=depth.said)
        for _, depth in depths
        if depth.said is not None
    ]


def _pass_rates_alike(subjects: Sequence[SubjectDepth]) -> bool:
    """Whether every compared subject reads pass^k as a pass rate and nothing more: planned and compared at depth 1,
    with no case left out, so its table would print each arm's pass rate again."""
    return bool(subjects) and all(
        subject.k == 1 and subject.common_depth == 1 and subject.left_out == 0 for subject in subjects
    )


def _pass_k_depth_blocks(
    depths: Sequence[tuple[AnalysisContextBundle, AspectDepth]],
    labels: Mapping[str, str],
    names: AspectNames,
    forms: PageForms,
) -> list[ReportBlock]:
    """Each frontier point's depth beside its pass^k: the planned depth, the cases short of it, and its reading at the
    common depth, the fewest runs any arm gave a shared case; a sentence on the common depth, or that the arms share no
    case, and on each arm pass^k is unmeasured for. Every subject of every aspect.

    Where every subject compares pass rates (:func:`_pass_rates_alike`), one sentence says so in place of the table:
    the subject's own where there is one, else one for the campaign."""
    compared = [subject for _, depth in depths for subject in depth.subjects if subject.compared]
    if _pass_rates_alike(compared):
        lines = compared[0].lines() if len(compared) == 1 else [COMMON_DEPTH_ONE_EVERYWHERE]
        return [
            *(DisclosureBlock(section="scorecard", source="comparisons", text=line) for line in lines),
            *_model_blocks(depths, labels, names, forms),
        ]
    rows: list[dict[str, TableCell]] = []
    for bundle, depth in depths:
        name = _label_of(bundle, labels)
        subjects = [subject for subject in bundle.frontier.subjects if subject.points]
        for subject, said in zip(subjects, depth.subjects, strict=True):
            if not said.compared:
                continue
            for point in subject.points:
                rows.append(
                    _of_the_pass_rate(
                        table_row(
                            {
                                "aspect": TableCell(value=said.where, ref=names.folds.get(bundle.aspect)),
                                "arm": name(point.variant_key),
                                "planned": forms.count_cell(subject.k),
                                "short": forms.count_cell(point.n_cases_short_of_k),
                                "common": forms.count_cell(subject.common_depth),
                                "reading": _pass_k_cell(forms, point.common_pass_hat_k),
                                "left_out": forms.count_cell(point.n_cases_left_out),
                            }
                        ),
                        ("reading",),
                        forms,
                    )
                )
    if not rows:
        return []
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scorecard",
            name="pass_k_depth",
            title="Pass^k depth",
            columns=[
                TableColumn(key="aspect", header="Aspect", term="aspect"),
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="planned", header="Planned depth", term="depth"),
                TableColumn(key="short", header="Cases short", term="cases"),
                TableColumn(key="common", header="Common depth", term="common depth"),
                TableColumn(key="reading", header="Pass^k there (%)"),
                TableColumn(key="left_out", header="Cases left out", term="cases"),
            ],
            rows=rows,
            order="aspect by aspect, each arm as the frontier lists it",
            total_rows=len(rows),
        )
    ]
    blocks += [
        DisclosureBlock(section="scorecard", source="comparisons", text=line)
        for _, depth in depths
        for line in depth.pass_k_lines()
    ]
    return blocks + _model_blocks(depths, labels, names, forms)


#: The two arms a precision row compares.
ARMS_PAIR = "{first} and {second}"

#: What the page says under the model's table: what the figures are, and that nothing ranks on them.
_MODEL_SAID = f"Each figure here is {MODEL_ASSUMES}. Its arm ran no case that deep, so nothing ranks on it."


def _model_blocks(
    depths: Sequence[tuple[AnalysisContextBundle, AspectDepth]],
    labels: Mapping[str, str],
    names: AspectNames,
    forms: PageForms,
) -> list[ReportBlock]:
    """The opt-in model-based pass^k of each point at a depth it never reached, in a table of its own, labelled a
    model with what it assumes: never beside a measured figure as one."""
    rows = [
        _of_the_pass_rate(
            table_row(
                {
                    "aspect": TableCell(value=said.where, ref=names.folds.get(bundle.aspect)),
                    "arm": _label_of(bundle, labels)(point.variant_key),
                    "depth": forms.count_cell(point.model_pass_hat_k.k),
                    "model": _pass_k_cell(forms, point.model_pass_hat_k.pass_hat_k),
                }
            ),
            ("model",),
            forms,
        )
        for bundle, depth in depths
        for subject, said in zip([sub for sub in bundle.frontier.subjects if sub.points], depth.subjects, strict=True)
        for point in subject.points
        if point.model_pass_hat_k is not None
    ]
    if not rows:
        return []
    return [
        TableBlock(
            section="scorecard",
            name="model_pass_k",
            title="Model-based pass^k",
            columns=[
                TableColumn(key="aspect", header="Aspect", term="aspect"),
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="depth", header="Depth", term="depth"),
                TableColumn(key="model", header="Model pass^k (%)"),
            ],
            rows=rows,
            order="aspect by aspect, each arm the depth is unreached for",
            total_rows=len(rows),
        ),
        DisclosureBlock(
            section="scorecard",
            source="comparisons",
            text=_MODEL_SAID,
        ),
    ]


def _precision_blocks(readings: CampaignReadings, names: AspectNames, forms: PageForms) -> list[ReportBlock]:
    """What limits each pass-rate comparison's interval: its half-width, each arm's own, the arm limiting it and the
    between-case share of its variance, folded under the methods; a column no row fills is left out."""
    compared: list[tuple[str, Precision]] = [(row.aspect, row.precision) for row in readings.readings if row.precision]
    compared += [(pair.aspect, pair.precision) for pair in readings.pairs if pair.precision]
    if not compared:
        return []
    labels = readings.arm_labels()

    def named(key: str) -> str:
        return labels.get(key, f"unplaced arm {short_digest(key)}")

    rows: list[dict[str, TableCell]] = []
    for aspect, precision in compared:
        # The sides in the order the comparison read them: the control, or the arm first in key order, first.
        (first, first_width), (second, second_width) = precision.half_widths.items()
        rows.append(
            _of_the_pass_rate(
                table_row(
                    {
                        "aspect": names.cell(aspect),
                        "arms": ARMS_PAIR.format(first=named(first), second=named(second)),
                        "half_width": _half_width_cell(forms, precision.half_width),
                        "first": _half_width_cell(forms, first_width),
                        "second": _half_width_cell(forms, second_width),
                        "limited_by": named(precision.limited_by) if precision.limited_by else None,
                        "between": None
                        if precision.between_share is None
                        else _share_cell(forms, precision.between_share),
                    }
                ),
                ("half_width", "first", "second"),
                forms,
            )
        )
    columns = [
        TableColumn(key="aspect", header="Aspect", term="aspect"),
        TableColumn(key="arms", header="Arms"),
        TableColumn(key="half_width", header="Half-width (pts)", term="interval"),
        TableColumn(key="first", header="First arm's own (pts)", term="interval"),
        TableColumn(key="second", header="Second arm's own (pts)", term="interval"),
        TableColumn(key="limited_by", header="Limited by"),
        TableColumn(key="between", header="Between cases (%)", term="between cases"),
    ]
    filled = [column for column in columns if any(row[column.key].value is not None for row in rows)]
    return [
        TableBlock(
            section="methods",
            name="precision",
            title="Precision",
            columns=filled,
            rows=[{column.key: row[column.key] for column in filled} for row in rows],
            order="aspect by aspect, each comparison as the readings list it",
            total_rows=len(rows),
            details=True,
        )
    ]


def depth_provenance_blocks(readings: CampaignReadings) -> list[ReportBlock]:
    """What depth says once in the page's Design & provenance section, in the slot
    :mod:`~threetears.evals.analysis.report.campaign.page` gives it there: that extra repeats of a case cannot lower
    the spread between cases, wherever a comparison's between-case share is read, which needs a repeated case.

    :param readings: What code read off the campaign.
    :ptype readings: CampaignReadings
    :return: The one sentence; nothing where no comparison reads a between-case share.
    :rtype: list[ReportBlock]
    """
    read = [row.precision for row in readings.readings] + [pair.precision for pair in readings.pairs]
    if not any(precision is not None and precision.between_share is not None for precision in read):
        return []
    return [DisclosureBlock(section="provenance", source="comparisons", text=REPEATS_FLOOR)]


def _of_the_pass_rate(row: dict[str, TableCell], keys: Sequence[str], forms: PageForms) -> dict[str, TableCell]:
    """A depth row whose cells under ``keys`` are figures of the pass rate (a pass^k, an interval's half-width), which
    the page's column rules record them under."""
    return {
        key: forms.of_measure(cell, measure=TRIAL_PASS_NAME, kind="trial") if key in keys else cell
        for key, cell in row.items()
    }


def _pass_k_cell(forms: PageForms, value: float | None) -> TableCell:
    """A pass^k as its depth table's cell holds it, bare under the column's ``(%)``, at the places the page's forms
    read it at."""
    printed = forms.pass_k(TRIAL_PASS_NAME, value, kind="trial")
    return (
        TableCell(value=printed.held, value_text=printed.digits) if printed.held is not None else TableCell(value=None)
    )


def _half_width_cell(forms: PageForms, value: float) -> TableCell:
    """An interval's half-width as the precision table's cell holds it, bare under the column's ``(pts)``, at the
    places the page's forms read the pass rate's changes at."""
    printed = forms.half_width(TRIAL_PASS_NAME, value, kind="trial")
    return TableCell(value=printed.held, value_text=printed.digits)


def _share_cell(forms: PageForms, share: float) -> TableCell:
    """A between-case share as the precision table's cell holds it, bare under the column's ``(%)``, in whole
    percent."""
    printed = forms.share(share)
    return TableCell(value=printed.held, value_text=printed.digits)
