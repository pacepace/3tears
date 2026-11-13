"""The screens that answer the question: **Decision**, the aspect's floors on each candidate and what would show
an unclear one holds, then **What it means**, the unasked readings and the contrasts, guardrails and readings laid out
in order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from threetears.evals.analysis.bundle.schema import FLOOR_COST_MEASURE, AnalysisContextBundle
from threetears.evals.analysis.quantities import is_money
from threetears.evals.analysis.report.aspect.common import (
    _candidates,
    _ChartLedger,
    _control_key,
    _cost_ruler,
    _floor_verdicts,
    _namer,
    _per_reply,
    _printed_strata,
    _readings_table,
)
from threetears.evals.analysis.report.aspect.contrasts import _comparison_blocks
from threetears.evals.analysis.report.aspect.gates import _guardrail_blocks
from threetears.evals.analysis.report.aspect.readings import _label_blocks, _readings_blocks, _strata_blocks
from threetears.evals.analysis.report.cost_unit import money_ruler
from threetears.evals.analysis.report.decision import (
    FloorVerdict,
    SampleProjection,
    case_cost_usd,
    floor_projection,
    floors_projection,
    projected_cell,
    projection_note,
)
from threetears.evals.analysis.report.figures import MeasureRuler, Rulers
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.model import (
    CellValue,
    DisclosureBlock,
    ReportBlock,
    TableBlock,
    TableColumn,
    Verdict,
    table_row,
)
from threetears.evals.analysis.report.verdict import guardrail_lines
from threetears.evals.analysis.report.words import PASS_RATE, judged_heading, one_figure_each
from threetears.evals.analysis.reporting import self_judging_text
from threetears.evals.analysis.style import ArmName, measure_name
from threetears.evals.kernel.declaration import CostLower
from threetears.evals.kernel.surface import DecisionSurface


def _dollars(amount: float | None, ruler: MeasureRuler, forms: PageForms) -> str | None:
    """A cost as the scoreboard prints it, on its ruler and in its one form (the one owner of money's scale, #993):
    ``$3.09`` per 1,000 replies beside every other arm's; None where none was measured."""
    if amount is None:
        return None
    if ruler.money is None or not is_money(ruler.unit):
        raise ValueError("a cost is spelled on a money ruler, and this one holds no form")
    return forms.ruled_level(ruler, amount, measure=FLOOR_COST_MEASURE, kind="measure").text


def _decision_blocks(
    bundle: AnalysisContextBundle, arm: Callable[[str], str], aspect: str, forms: PageForms
) -> list[ReportBlock]:
    """The aspect's floors, each on each candidate against the control, as a table, never prose.

    Each floor's row opens with the floor as a sentence naming its arm. The control heads its column by name, as does
    a lone other arm; with several, an ``Arm`` column names each row's and its figures are under ``Value``. A rate
    floor has a row per rate it reads, each with that rate's own status and reason, so a status never sits beside a
    reading it is not about. A broken floor is also a line under the verdict (:func:`_floor_warning_blocks`); here it
    reads as plainly as a held one.

    An aspect with no floor declared has no floor block at all: the page says what it holds, never what it lacks.
    Where a floor is unclear, a ``To show the floors hold``
    table follows: for each unclear floor, what showing it holds needs if the arms do not differ and if they differ as
    observed, each count with its cost
    (:func:`~threetears.evals.analysis.report.decision.floor_projection`), then one line saying how each count was
    found and that it is about, what repeats do, and, where holding needs more cases than the run used on every side,
    that it does. A floor broken or held projects nothing.
    """
    control = _control_key(bundle)
    read = _floor_verdicts(bundle, arm)
    declared = bundle.declared
    if declared is not None and declared.unread_without_control:
        # A margin or floors stored on an aspect of a campaign naming no control are read against nothing: said once,
        # where the floors would have been.
        return [DisclosureBlock(section="decision", source="floor", text=declared.unread_without_control)]
    if control is None or not _candidates(bundle):
        return []
    if not read:
        return []
    cost = _cost_ruler(bundle)
    others = list(read)
    rows: list[dict[str, CellValue]] = []
    projected: list[tuple[FloorVerdict, str, SampleProjection]] = []
    joined: list[SampleProjection] = []
    pointed = False
    per_case = case_cost_usd(bundle)
    for key, verdicts in read.items():
        if (together := floors_projection(verdicts, per_case_usd=per_case)) is not None:
            # Only an unclear floor projects: holding is what more cases can show.
            joined.append(together)
            projected += [
                (verdict, arm(key), projection)
                for verdict in verdicts
                if (projection := floor_projection(verdict, per_case_usd=per_case)) is not None
            ]
        for verdict in verdicts:
            shown: list[tuple[str, str | None, str | None, str]] = []
            readings = verdict.readings
            if isinstance(verdict.floor.clause, CostLower):
                shown = [
                    (
                        f"cost {_per_reply(bundle)}",
                        _dollars(readings.control_cost if readings else None, cost, forms),
                        _dollars(readings.candidate_cost if readings else None, cost, forms),
                        f"{verdict.outcome}: {forms.said(verdict.reason)}",
                    )
                ]
            else:
                # A reading past the margin on its point estimate alone says so: the glossary defines the term.
                pointed = pointed or any(rate.reason.mentions("point estimate") for rate in verdict.rates)
                # Each rate its own status, so a row's status is about the reading on that row and no other.
                shown = [
                    (
                        PASS_RATE if rate.rate.measure is None else rate.rate.name,
                        f"{rate.rate.control_passed} of {rate.rate.control_n}",
                        f"{rate.rate.candidate_passed} of {rate.rate.candidate_n}",
                        f"{rate.outcome}: {forms.said(rate.reason)}",
                    )
                    for rate in verdict.rates
                ]
            for index, (reading, ours, theirs, outcome) in enumerate(
                shown or [("—", None, None, f"{verdict.outcome}: {forms.said(verdict.reason)}")]
            ):
                rows.append(
                    {
                        "floor": forms.said(verdict.statement) if index == 0 else None,
                        "arm": arm(key),
                        "reading": reading,
                        "control": ours,
                        "candidate": theirs,
                        "outcome": outcome,
                    }
                )
    # A lone other arm heads its own column by name; several share a "Value" column, each row naming its arm.
    single = len(others) == 1
    columns = [
        TableColumn(key="floor", header="Floor", term="floor"),
        *([] if single else [TableColumn(key="arm", header="Arm")]),
        TableColumn(key="reading", header="Reading", term="reading"),
        TableColumn(key="control", header=arm(control)),
        TableColumn(key="candidate", header=arm(others[0]) if single else "Value"),
        TableColumn(key="outcome", header="Status", term="floor status"),
    ]
    return [
        TableBlock(
            section="decision",
            name="decision",
            title=f"Floors on {aspect}",
            columns=columns,
            rows=[table_row({column.key: row[column.key] for column in columns}) for row in rows],
            order="floor by floor, each arm against the control",
            total_rows=len(rows),
        ),
        DisclosureBlock(
            section="decision",
            source="floor",
            text=FLOORS_WARN_ONLY,
            term="point estimate" if pointed else None,
        ),
        *_projection_blocks(bundle, projected, joined, forms, single=single),
    ]


#: What the Decision section says once, under the floors: a floor warns and decides nothing, and which interval it
#: reads, so a floor unclear beside a contrast that is not separated reads as the two rulers it is.
FLOORS_WARN_ONLY = (
    "A floor warns and never decides: a broken one warns on the first screen. A floor reads its own interval, "
    "uncorrected, which can differ from the contrasts table's."
)


def _projection_blocks(
    bundle: AnalysisContextBundle,
    projected: Sequence[tuple[FloorVerdict, str, SampleProjection]],
    joined: Sequence[SampleProjection],
    forms: PageForms,
    *,
    single: bool,
) -> list[ReportBlock]:
    """The ``To show the floors hold`` table, one row per unclear floor and arm, and the line under it, which reads each arm's
    projection over its floors (``joined``); empty with none. A cost floor's no-difference cell is blank: it has no
    margin, and the line says so once."""
    if not projected:
        return []
    columns = [
        TableColumn(key="floor", header="Floor", term="floor"),
        *([] if single else [TableColumn(key="arm", header="Arm")]),
        TableColumn(key="none", header="If no difference"),
        TableColumn(key="observed", header="If as observed"),
    ]
    # Every projected cost the table states is one set of totals, in one form (#993).
    costs = money_ruler(
        (
            forms.projected_cost(cost)
            for verdict, _, projection in projected
            for need in (
                (projection.if_observed,)
                if isinstance(verdict.floor.clause, CostLower)
                else (projection.if_none, projection.if_observed)
            )
            if (cost := projection.cost(need)) is not None
        ),
        per=0,
    ).form
    rows: list[dict[str, CellValue]] = [
        {
            "floor": forms.said(verdict.statement),
            "arm": name,
            "none": None
            if isinstance(verdict.floor.clause, CostLower)
            else projected_cell(projection, projection.if_none, costs, forms=forms),
            "observed": projected_cell(projection, projection.if_observed, costs, forms=forms),
        }
        for verdict, name, projection in projected
    ]
    run_cases = max((board.cases for board in bundle.scoreboard), default=0)
    costed = any(isinstance(verdict.floor.clause, CostLower) for verdict, *_ in projected)
    return [
        TableBlock(
            section="decision",
            name="projection",
            title="To show the floors hold",
            columns=columns,
            rows=[table_row({column.key: row[column.key] for column in columns}) for row in rows],
            order="floor by floor, each arm against the control",
            total_rows=len(rows),
        ),
        DisclosureBlock(
            section="decision",
            source="floor",
            text=projection_note(joined, run_cases=run_cases, cost_clause=costed, forms=forms),
        ),
    ]


def _question_blocks(bundle: AnalysisContextBundle) -> list[ReportBlock]:
    """On a code-only report: the readings the declared question did not ask about."""
    scope = bundle.reading_scope
    blocks: list[ReportBlock] = []
    catalog = bundle.measure_catalog
    exploratory = [
        *(
            measure_name(name, catalog[name].reader_name if name in catalog else None, declared=bundle.check_name(name))
            for name in one_figure_each(scope.exploratory_measures, read=catalog)
        ),
        *(judged_heading(name) for name in scope.exploratory_dimensions),
    ]
    if exploratory and scope.questions_declared:
        blocks.append(_readings_table("unasked", "Unasked readings", [{"reading": name} for name in exploratory]))
    return blocks


def _contrast_blocks(
    bundle: AnalysisContextBundle,
    surface: DecisionSurface,
    contrasts_read: Sequence[Verdict],
    guardrails_read: Sequence[Verdict],
    charts: _ChartLedger,
    names: dict[str, ArmName],
    rulers: Rulers,
    forms: PageForms,
) -> list[ReportBlock]:
    """The contrasts against the control, their charts, the guardrails, and the readings by stratum and label, in the
    page's ``forms``.

    Engine text that explains how they were tested goes to the methods' closed ``Disclosures`` fold.
    """
    arm = _namer(names)
    blocks: list[ReportBlock] = []
    blocks.extend(_self_judging_blocks(bundle, surface, arm))
    blocks.extend(_guardrail_standing_blocks(bundle, arm))
    blocks.extend(_comparison_blocks(bundle, surface, contrasts_read, charts, arm, rulers, forms))
    blocks.extend(_readings_blocks(bundle, surface, names, rulers, forms))
    blocks.extend(_guardrail_blocks(surface, bundle.guardrails, guardrails_read, rulers, forms))
    blocks.extend(
        _strata_blocks(
            surface, names, rulers, forms, check_names=bundle.named_checks(), stratum_names=_printed_strata(bundle)
        )
    )
    blocks.extend(_label_blocks(surface, names, rulers, forms))
    return blocks


def _self_judging_blocks(
    bundle: AnalysisContextBundle, surface: DecisionSurface, arm: Callable[[str], str]
) -> list[ReportBlock]:
    """A line for each arm whose answers its judge's own model produced: its judged contrasts may favour it."""
    variant_of_run = {run_id: cell.variant_key for cell in surface.cells for run_id in cell.run_ids}
    shared: dict[str, list[str]] = {}
    for summary in bundle.run_summaries:
        variant = variant_of_run.get(summary.run_id)
        if variant is None or not summary.judge_shares_candidate_model:
            continue
        models = shared.setdefault(variant, [])
        models.extend(model for model in summary.judge_shares_candidate_model if model not in models)
    return [
        DisclosureBlock(
            section="contrasts",
            source="apparatus",
            text=self_judging_text(models, f"arm {arm(variant)}")
            + "; a contrast on a judged dimension may favour that arm for that reason alone.",
        )
        for variant, models in sorted(shared.items(), key=lambda item: arm(item[0]))
    ]


def _guardrail_standing_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str]) -> list[ReportBlock]:
    """A line for each arm a guardrail leaves not shown safe, above the contrasts it might be picked from, in the
    verdict line's own words (:func:`~threetears.evals.analysis.report.verdict.guardrail_lines`)."""
    return [
        DisclosureBlock(section="contrasts", source="guardrails", text=line, term="guardrail")
        for line in guardrail_lines(bundle, arm)
    ]


__all__: list[str] = []
