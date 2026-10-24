"""How deep a campaign's arms ran, for every kind: each arm's cases, trials and repeats on each aspect, pass^k at the
common depth, the opt-in model of pass^k at a depth no arm reached, and what limits each comparison's interval;
tables, with one sentence under each where the arms differ."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from threetears.evals.analysis.arms import short_digest
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.campaign_readings import CampaignReadings
from threetears.evals.analysis.depth import MODEL_ASSUMES, REPEATS_FLOOR, Precision
from threetears.evals.analysis.depth_reading import AspectDepth, aspect_depth
from threetears.evals.analysis.report.campaign.common import named_arms
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

__all__ = ["depth_blocks"]


def _label_of(bundle: AnalysisContextBundle, labels: Mapping[str, str]) -> Callable[[str], str]:
    """What the page calls the arm a variant key names on one aspect: the campaign's label, else its variant's name."""
    names = named_arms(bundle, labels)
    return lambda key: names[key].label if key in names else f"unplaced arm {short_digest(key)}"


def depth_blocks(
    campaign: EvalCampaign, readings: CampaignReadings, reports: Mapping[str, Report]
) -> list[ReportBlock]:
    """How deep each arm ran on each aspect, what pass^k compares at, and what limits each comparison's interval:
    tables, with one sentence under each where the arms differ.

    :param campaign: The campaign.
    :ptype campaign: EvalCampaign
    :param readings: What code read off it.
    :ptype readings: CampaignReadings
    :param reports: Each aspect's own page, by its label.
    :ptype reports: Mapping[str, Report]
    :return: The tables and their sentences; nothing where no arm ran.
    :rtype: list[ReportBlock]
    """
    labels = readings.arm_labels()
    depths = [
        (
            reports[aspect.label].bundle,
            aspect_depth(reports[aspect.label].bundle, _label_of(reports[aspect.label].bundle, labels)),
        )
        for aspect in campaign.aspects
    ]
    rows = [
        table_row(
            {
                "aspect": depth.aspect,
                "arm": arm,
                "cases": held.cases,
                "trials": held.trials,
                "fewest": held.repeats_min,
                "median": held.repeats_median,
                "most": held.repeats_max,
            }
        )
        for _, depth in depths
        for arm, held in depth.arms.items()
    ]
    if not rows:
        return []
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scoreboard",
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
    blocks += [
        DisclosureBlock(section="scoreboard", source="comparisons", text=depth.said)
        for _, depth in depths
        if depth.said is not None
    ]
    blocks += _pass_k_depth_blocks(depths, labels)
    blocks += _precision_blocks(readings)
    return blocks


def _pass_k_depth_blocks(
    depths: Sequence[tuple[AnalysisContextBundle, AspectDepth]], labels: Mapping[str, str]
) -> list[ReportBlock]:
    """Each frontier point's depth beside its pass^k: the planned depth, the cases short of it, and its reading at the
    common depth, the fewest runs any arm gave a shared case; a sentence on the common depth, or that the arms share no
    case, and on each arm pass^k is unmeasured for. Every subject of every aspect."""
    rows: list[dict[str, TableCell]] = []
    for bundle, depth in depths:
        name = _label_of(bundle, labels)
        subjects = [subject for subject in bundle.frontier.subjects if subject.points]
        for subject, said in zip(subjects, depth.subjects, strict=True):
            if not said.compared:
                continue
            for point in subject.points:
                rows.append(
                    table_row(
                        {
                            "aspect": said.where,
                            "arm": name(point.variant_key),
                            "planned": subject.k,
                            "short": point.n_cases_short_of_k,
                            "common": subject.common_depth,
                            "reading": None
                            if point.common_pass_hat_k is None
                            else round(100 * point.common_pass_hat_k, 1),
                            "left_out": point.n_cases_left_out,
                        }
                    )
                )
    if not rows:
        return []
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scoreboard",
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
        DisclosureBlock(section="scoreboard", source="comparisons", text=line)
        for _, depth in depths
        for line in depth.pass_k_lines()
    ]
    return blocks + _model_blocks(depths, labels)


#: The two arms a precision row compares.
ARMS_PAIR = "{first} and {second}"

#: What the page says under the model's table: what the figures are, and that nothing ranks on them.
_MODEL_SAID = f"Each figure here is {MODEL_ASSUMES}. Its arm ran no case that deep, so nothing ranks on it."


def _model_blocks(
    depths: Sequence[tuple[AnalysisContextBundle, AspectDepth]], labels: Mapping[str, str]
) -> list[ReportBlock]:
    """The opt-in model-based pass^k of each point at a depth it never reached, in a table of its own, labelled a
    model with what it assumes: never beside a measured figure as one."""
    rows = [
        table_row(
            {
                "aspect": said.where,
                "arm": _label_of(bundle, labels)(point.variant_key),
                "depth": point.model_pass_hat_k.k,
                "model": round(100 * point.model_pass_hat_k.pass_hat_k, 1),
            }
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
            section="scoreboard",
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
            section="scoreboard",
            source="comparisons",
            text=_MODEL_SAID,
        ),
    ]


def _precision_blocks(readings: CampaignReadings) -> list[ReportBlock]:
    """What limits each pass-rate comparison's interval: its half-width, each arm's own, the arm limiting it and the
    between-case share of its variance, with the one sentence saying what extra repeats can and cannot lower."""
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
            table_row(
                {
                    "aspect": aspect,
                    "arms": ARMS_PAIR.format(first=named(first), second=named(second)),
                    "half_width": round(precision.half_width, 1),
                    "first": round(first_width, 1),
                    "second": round(second_width, 1),
                    "limited_by": named(precision.limited_by) if precision.limited_by else None,
                    "between": None if precision.between_share is None else round(100 * precision.between_share),
                }
            )
        )
    return [
        TableBlock(
            section="scoreboard",
            name="precision",
            title="Precision",
            columns=[
                TableColumn(key="aspect", header="Aspect", term="aspect"),
                TableColumn(key="arms", header="Arms"),
                TableColumn(key="half_width", header="Half-width (pts)", term="interval"),
                TableColumn(key="first", header="First arm's own (pts)", term="interval"),
                TableColumn(key="second", header="Second arm's own (pts)", term="interval"),
                TableColumn(key="limited_by", header="Limited by"),
                TableColumn(key="between", header="Between cases (%)", term="between cases"),
            ],
            rows=rows,
            order="aspect by aspect, each comparison as the readings list it",
            total_rows=len(rows),
        ),
        DisclosureBlock(section="scoreboard", source="comparisons", text=REPEATS_FLOOR),
    ]
