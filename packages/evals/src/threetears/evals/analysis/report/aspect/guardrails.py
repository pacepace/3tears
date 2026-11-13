"""The guardrails and the bars: each read on each arm (:func:`guardrail_verdicts`, :func:`bar_verdicts`), and the
blocks that lay them out under the contrasts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from threetears.evals.analysis.quantities import delta_unit
from threetears.evals.analysis.report.aspect.common import _at, _interval_on_delta, _level_text, _note
from threetears.evals.analysis.report.campaign.scorecard import bar_threshold
from threetears.evals.analysis.report.figures import MeasureRuler, Rulers, report_rulers
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.model import (
    DisclosureBlock,
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
    Verdict,
    VerdictReason,
)
from threetears.evals.analysis.report.words import BAR_DECISION_WORDS, guardrail_words, reading_heading, reason_words
from threetears.evals.analysis.stats import INTERVAL_LEVEL
from threetears.evals.analysis.style import declared_check_name, humanize
from threetears.evals.kernel.campaign import ReadingKind
from threetears.evals.kernel.surface import DecisionSurface, GuardrailCheck, GuardrailReadings

#: How a guardrail is decided, said once in the methods' disclosures.
_GUARDRAILS_DISCLOSURE = (
    "A guardrail is what an arm must not get worse on. Each is decided on its own 95% interval on the difference "
    "from the control: held when the whole interval lies on the good side of the margin, breached when it lies "
    "wholly beyond it, undecided otherwise. On a declared range the interval is the bounded test's, which holds "
    "its error rate at any number of cases; a reading with no declared range is never held, since no test holds "
    "its rate there, and can only be shown breached. A measure's margin is its declared materiality threshold, a "
    "judged dimension's the one its campaign declares; with none, a guardrail is held at zero change. Guardrails "
    "join no comparison and no composite, so no gain elsewhere offsets one; a breached guardrail warns, the writer may "
    "not recommend its arm, and an undecided guardrail is not known to be safe."
)


def _interval_header(levels: set[float], forms: PageForms) -> str:
    """The interval column's header, carrying the one level every row holds at; bare where the rows differ."""
    return f"Interval ({_level_text(next(iter(levels)), forms)})" if len(levels) == 1 else "Interval"


def _straddled(check: GuardrailCheck, ruler: MeasureRuler, forms: PageForms) -> str:
    """Why a guardrail whose interval straddles its line is undecided, on the page's ruler for its reading: the line
    in points for a rate, beside the interval and margin columns that state it so. The interval itself is its own
    column's, with the level in that column's header, so the reason does not say it twice."""
    line = -check.margin if check.higher_is_better else check.margin
    declared = forms.in_unit(
        line * ruler.factor, delta_unit(ruler.unit), role="declared", measure=check.name, kind=check.reading
    )
    said = f"{declared.text}, the declared margin" if check.margin_declared else "zero, as no margin is declared"
    return f"the interval reaches both sides of {said}, so the arm is shown neither within it nor beyond it"


def guardrail_verdicts(
    surface: DecisionSurface,
    guardrails: GuardrailReadings | None,
    arm: Callable[[str, str], str],
    rulers: Rulers | None = None,
    *,
    check_names: Mapping[str, str] | None = None,
    forms: PageForms,
) -> list[Verdict]:
    """Each guardrail check as a typed verdict, in the guardrails table's row order; the table prints their words.

    An undecided check whose interval straddles its line says why on ``rulers`` (the surface's own where None), the
    ruler the guardrails table states that reading on, its line in the page's ``forms``. ``check_names``
    (``AnalysisContextBundle.named_checks``) heads a goal check by its declared display name.
    """
    if guardrails is None or guardrails.withheld is not None:
        return []
    rulers = rulers or report_rulers(surface)
    verdicts = []
    for check in guardrails.checks:
        reason: VerdictReason = (
            "within_margin"
            if check.decision == "held"
            else "beyond_margin"
            if check.decision == "breached"
            else "no_interval"
            if check.interval is None
            else "interval_straddles"
        )
        verdicts.append(
            Verdict(
                kind="guardrail",
                outcome=check.decision,
                reason=reason,
                reason_detail=reason_words(check.undecided_reason),
                reading=check.reading,
                name=check.name,
                heading=reading_heading(
                    surface, check.name, check.reading, declared=declared_check_name(check_names or {}, check.name)
                ),
                arm=arm(check.contrast.variant_key, check.contrast.apparatus_class_id),
                variant_key=check.contrast.variant_key,
                apparatus_class_id=check.contrast.apparatus_class_id,
                control=arm(check.control.variant_key, check.control.apparatus_class_id),
                delta=check.delta,
                interval=check.interval,
                interval_level=INTERVAL_LEVEL if check.interval is not None else None,
                margin=check.margin,
                margin_source=(
                    None if not check.margin_declared else "campaign" if check.reading == "judged" else "measure"
                ),
                guardrail=True,
                words=guardrail_words(
                    check.decision,
                    arm(check.contrast.variant_key, check.contrast.apparatus_class_id),
                    arm(check.control.variant_key, check.control.apparatus_class_id),
                )
                + (
                    f" ({_straddled(check, rulers.of(check.name, check.reading), forms)})"
                    if reason == "interval_straddles"
                    else f" ({reason_words(check.undecided_reason)})"
                    if check.undecided_reason
                    else ""
                ),
            )
        )
    return verdicts


def bar_verdicts(
    surface: DecisionSurface, arm: Callable[[str, str], str], *, check_names: Mapping[str, str] | None = None
) -> list[Verdict]:
    """Each adjudicated bar's reading on each arm as a typed verdict, bar by bar in the surface's order; a bar on a
    goal check headed by its declared display name in ``check_names`` (``AnalysisContextBundle.named_checks``)."""
    reasons: dict[str, VerdictReason] = {
        "cleared": "interval_clears",
        "missed": "interval_misses",
        "undecided": "interval_straddles",
        "no_interval": "too_few_observations",
        "no_data": "no_observations",
    }
    verdicts = []
    for bar in surface.bars:
        if bar.state != "adjudicated":
            continue
        reading: ReadingKind = "judged" if bar.measure_id in surface.dimensions else "measure"
        for read in bar.verdicts:
            interval = (read.ci_low, read.ci_high) if read.ci_low is not None and read.ci_high is not None else None
            verdicts.append(
                Verdict(
                    kind="bar",
                    outcome=read.decision,
                    reason=reasons[read.decision],
                    reading=reading,
                    name=bar.measure_id,
                    heading=reading_heading(
                        surface,
                        bar.measure_id,
                        reading,
                        declared=declared_check_name(check_names or {}, bar.measure_id),
                    ),
                    arm=arm(read.variant_key, read.apparatus_class_id),
                    variant_key=read.variant_key,
                    apparatus_class_id=read.apparatus_class_id,
                    interval=interval,
                    value=read.value,
                    threshold=bar.threshold,
                    margin=read.margin,
                    margin_source="measure" if read.margin is not None else None,
                    words=BAR_DECISION_WORDS[read.decision],
                )
            )
    return verdicts


def _guardrail_blocks(
    surface: DecisionSurface,
    guardrails: GuardrailReadings | None,
    verdicts: Sequence[Verdict],
    rulers: Rulers,
    forms: PageForms,
) -> list[ReportBlock]:
    """Each guardrail, held, breached or undecided for each arm against the control; the rule said once beside it."""
    if guardrails is None:
        return []
    if guardrails.withheld is not None:
        return [DisclosureBlock(section="contrasts", source="guardrails", text=guardrails.withheld)]
    if not guardrails.checks:
        return []
    rows: list[dict[str, TableCell]] = []
    for check, verdict in zip(guardrails.checks, verdicts, strict=True):
        ruler = rulers.of(check.name, check.reading)
        form = forms.change_money(check.name, check.reading)
        interval = _interval_on_delta(check.interval, ruler, forms, form=form, measure=check.name, kind=check.reading)
        rows.append(
            {
                "guardrail": TableCell(value=ruler.scaled(verdict.heading)),
                "arm": TableCell(value=verdict.arm),
                "control_mean": _at(
                    forms.ruled_level_cell(ruler, check.control.mean, measure=check.name, kind=check.reading),
                    surface,
                    check.control,
                    check.name,
                    check.reading,
                    check.control.mean,
                ),
                "arm_mean": _at(
                    forms.ruled_level_cell(ruler, check.contrast.mean, measure=check.name, kind=check.reading),
                    surface,
                    check.contrast,
                    check.name,
                    check.reading,
                    check.contrast.mean,
                ),
                "cases": TableCell(
                    value=(
                        f"{check.control.n_cases} paired"
                        if check.test == "paired"
                        else f"{check.contrast.n_cases} vs {check.control.n_cases}"
                    )
                ),
                "delta": forms.ruled_change_cell(ruler, check.delta, form=form, measure=check.name, kind=check.reading),
                "interval": TableCell(
                    value=None
                    if interval is None
                    else interval + (" (t: no declared range)" if check.interval_basis == "t" else "")
                ),
                "margin": TableCell(
                    value=forms.in_unit(
                        check.margin * ruler.factor,
                        delta_unit(ruler.unit),
                        role="declared",
                        measure=check.name,
                        kind=check.reading,
                    ).text
                    if check.margin_declared
                    else "0 (none declared)"
                ),
                "decision": TableCell(value=verdict.words),
            }
        )
    return [
        TableBlock(
            section="contrasts",
            name="guardrails",
            title="Guardrails",
            columns=[
                TableColumn(key="guardrail", header="Guardrail", term="guardrail"),
                TableColumn(key="arm", header="Arm"),
                TableColumn(key="control_mean", header="Control mean", term="control mean"),
                TableColumn(key="arm_mean", header="Arm mean", term="arm mean"),
                TableColumn(key="cases", header="Cases", term="cases"),
                TableColumn(key="delta", header="Change", term="change"),
                TableColumn(key="interval", header=_interval_header({INTERVAL_LEVEL}, forms), term="interval"),
                TableColumn(key="margin", header="Margin", term="margin"),
                TableColumn(key="decision", header="Decision", term="decision"),
            ],
            rows=rows,
            order="by guardrail, then arm",
            total_rows=len(rows),
        ),
        _note(_GUARDRAILS_DISCLOSURE, "guardrails"),
    ]


def _bar_blocks(
    surface: DecisionSurface, verdicts: Sequence[Verdict], rulers: Rulers, forms: PageForms
) -> list[ReportBlock]:
    """Each adjudicated bar, read on each arm, and every bar no arm could be read against."""
    blocks: list[ReportBlock] = []
    if verdicts:
        rows: list[dict[str, TableCell]] = []
        for verdict in verdicts:
            ruler = rulers.of(verdict.name, verdict.reading)
            # A bar's value and its interval read at their reading's places, as its contrasts do.
            places = forms.contrast_places(verdict.name, verdict.reading)
            rows.append(
                {
                    "bar": TableCell(value=ruler.scaled(verdict.heading)),
                    "arm": TableCell(value=verdict.arm),
                    "value": forms.of_measure(
                        _at(
                            forms.ruled_level_cell(
                                ruler, verdict.value, places, measure=verdict.name, kind=verdict.reading
                            ),
                            surface,
                            verdict,
                            verdict.name,
                            verdict.reading,
                            verdict.value,
                        ),
                        measure=verdict.name,
                        kind=verdict.reading,
                    ),
                    # A bar's interval reads as its value does: money in the measure's one form
                    # (``$0.019 to $0.021``, never ``$0.01858 to $0.02142``).
                    "interval": forms.of_measure(
                        TableCell(
                            value=None
                            if verdict.interval is None
                            else forms.ruled_interval(
                                ruler, *verdict.interval, places, measure=verdict.name, kind=verdict.reading
                            ).text
                        ),
                        measure=verdict.name,
                        kind=verdict.reading,
                    ),
                    # As it was declared, never at the measure's places (rule 4).
                    "threshold": bar_threshold(
                        forms, ruler, verdict.threshold, measure=verdict.name, kind=verdict.reading
                    )
                    if verdict.threshold is not None
                    else TableCell(value=None),
                    "decision": TableCell(value=verdict.words),
                }
            )
        blocks.append(
            TableBlock(
                section="contrasts",
                name="bars",
                title="Bars",
                columns=[
                    TableColumn(key="bar", header="Bar", term="bar"),
                    TableColumn(key="arm", header="Arm"),
                    TableColumn(key="value", header="Value"),
                    TableColumn(key="interval", header="Interval", term="interval"),
                    TableColumn(key="threshold", header="Threshold"),
                    TableColumn(key="decision", header="Decision", term="decision"),
                ],
                rows=rows,
                order="bar by bar, each arm in the surface's order",
                total_rows=len(rows),
                details=True,
            )
        )
    blocks.extend(
        _note(
            f"The {bar.source} bar on {humanize('dimension', bar.measure_id)} could be read against no arm: {bar.reason}."
        )
        for bar in surface.bars
        if bar.state != "adjudicated" and bar.reason
    )
    return blocks


__all__ = [
    "bar_verdicts",
    "guardrail_verdicts",
]
