"""The first screen's scoreboard: each arm's counts and cost per reply, its outcome bar, and the short names its
charts print in tight places.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from threetears.evals.analysis.arms import arm_levels, distinguishing_axes, elide_level
from threetears.evals.analysis.bundle.schema import (
    FLOOR_COST_MEASURE,
    TRIAL_PASS_NAME,
    AnalysisContextBundle,
    ArmScoreboard,
)
from threetears.evals.analysis.bundle.surface import bundle_decision_surface
from threetears.evals.analysis.report.aspect.common import _at, _control_key, _cost_ruler, _per_reply
from threetears.evals.analysis.report.figures import MeasureRuler, report_rulers
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.model import (
    CellValue,
    ChartBlock,
    DisclosureBlock,
    OutcomeBar,
    OutcomeSegment,
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
)
from threetears.evals.analysis.report.number_forms.kinds import Role
from threetears.evals.analysis.report.verdict import cause_short, caveat_lines
from threetears.evals.analysis.report.words import campaign_arm_tags
from threetears.evals.analysis.style import GLOSSARY, ArmName, arm_display_name, lever_reader_name
from threetears.evals.kernel.host.sweepables import CANDIDATE_MODEL_LEVER
from threetears.evals.kernel.surface import DecisionSurface, earlier_control_mark


def _short_names(
    bundle: AnalysisContextBundle,
    names: Mapping[str, ArmName],
    given: Mapping[str, str],
    declared: Mapping[str, str],
) -> dict[str, str]:
    """Each arm's short name, for a chart's tight places (GENERATOR.md §6.1): the short name the campaign declares for
    it, as declared; else the label the campaign gives it; the control's ``control``; else only the levels where it
    differs from the control, the model first and trimmed of the prefix it shares with the control's. With no control,
    or nothing that differs, an arm's short name is its name, and so is it where that name is no longer than the short
    one (a quick run's ``v4-flash``, never ``candidate v4-flash``). Declared names are unique, which the campaign
    refuses otherwise, so only a computed one can collide: one that agrees with any other arm's short name is named in
    full, so no two read alike."""
    control = _control_key(bundle)
    entries = {entry.variant_key: entry for entry in bundle.variant_index}
    distinguishing = distinguishing_axes(bundle.variant_index)
    base = {level.axis_id: level for level in arm_levels(entries.get(control) if control else None, distinguishing)}
    shorts: dict[str, str] = {}
    for key, name in names.items():
        if key in declared:
            shorts[key] = declared[key]
            continue
        if key in given:
            shorts[key] = given[key]
        elif key == control:
            shorts[key] = "control"
        elif control is None:
            shorts[key] = name.label
        else:
            differing = [
                level
                for level in arm_levels(entries.get(key), distinguishing)
                if level.axis_id not in base or base[level.axis_id].content_hash != level.content_hash
            ]
            ordered = sorted(differing, key=lambda level: level.axis_id != CANDIDATE_MODEL_LEVER)
            trimmed = [level.axis_id == CANDIDATE_MODEL_LEVER and level.axis_id in base for level in ordered]
            parts = [
                _trimmed(level.display, base[level.axis_id].display)
                if cut
                else f"{lever_reader_name(level.axis_id, bundle.lever_names)} {elide_level(level.display)}"
                for level, cut in zip(ordered, trimmed, strict=True)
            ]
            # Computed from the levers it differs on, so printed by the one display-name rule; a model trimmed of the
            # control's prefix is a model's name, which no lever's reader name opens.
            levers = [level.axis_id for level, cut in zip(ordered, trimmed, strict=True) if not cut]
            shorts[key] = arm_display_name(" · ".join(parts), levers, bundle.lever_names) or name.label
        if len(name.label) <= len(shorts[key]):
            shorts[key] = name.label
    taken = list(shorts.values())
    return {
        key: short if key in declared or taken.count(short) == 1 else names[key].label for key, short in shorts.items()
    }


#: What separates the parts of a model's name, where a shared prefix may be cut.
_MODEL_SEPARATORS = frozenset("-_ .")


def _trimmed(model: str, base: str) -> str:
    """``model`` without the prefix it shares with ``base``, cut at a separator, as GENERATOR.md §6.1 trims a
    candidate's model against the control's (``Meridian-4.1-flash`` against ``Meridian-4.1`` is ``flash``); the whole
    model where under four characters are shared or nothing is left."""
    shared = 0
    while shared < min(len(model), len(base)) and model[shared] == base[shared]:
        shared += 1
    if not (shared == len(base) and shared < len(model) and model[shared] in _MODEL_SEPARATORS):
        while shared > 0 and model[shared - 1] not in _MODEL_SEPARATORS:
            shared -= 1
    rest = model[shared:].lstrip("".join(_MODEL_SEPARATORS))
    return rest if shared >= 4 and rest else model


def _scoreboard_blocks(
    bundle: AnalysisContextBundle,
    arm: Callable[[str], str],
    shorts: Mapping[str, str],
    forms: PageForms,
    given_tags: Mapping[str, str] | None = None,
) -> list[ReportBlock]:
    """The scoreboard table, its outcome bar, then only what changes how the verdict is read, in normal weight. Each
    arm's cost is a figure of the cost the floors read, which the page's forms hold the cell under.

    The outcome bar tags each arm by ``given_tags``, a campaign's, where they tag every arm of the scoreboard, and
    else by this page's own order.

    A fact that changes how the verdict should be read belongs on the first screen, and nothing else does: the case
    floor, then an arm whose every trial failed. Everything else that qualifies the table is a
    step down: what was not charted and where a cost is a floor are in the methods; a failure cause the scoreboard
    counts apart from checks and the judge heads its own column where it is the only one, and the outcome bar names
    each where there are several.
    """
    boards = bundle.scoreboard
    if not boards:
        return [
            DisclosureBlock(
                section="scoreboard", source="runs", text="No trial ran: the campaign resolved no result in this scope."
            )
        ]
    control = _control_key(bundle)
    keys = [board.variant_key for board in boards]
    tags = (
        dict(given_tags)
        if given_tags is not None and all(key in given_tags for key in keys)
        else campaign_arm_tags([bundle], control)
    )
    earlier = bundle.design.earlier_control
    lost_any = any(board.lost for board in boards)
    other_causes = sorted({cause for board in boards for cause in board.failed if cause not in ("check", "judged")})
    cost = _cost_ruler(bundle)
    # The share of each arm's trials that passed, on the pass rate's ruler, as the scorecard states the rate.
    passing = report_rulers(bundle_decision_surface(bundle), bundle.measure_catalog).of(TRIAL_PASS_NAME, "trial")
    rows: list[dict[str, CellValue]] = []
    for board in boards:
        row: dict[str, CellValue] = {
            "arm": ("▸ " if board.variant_key == control else "")
            + arm(board.variant_key)
            + (f" ({earlier_control_mark(earlier)})" if earlier and board.variant_key == earlier.variant_key else ""),
            "ran": board.ran,
            "usable": board.usable,
            "passed": board.passed,
            "check": board.failed.get("check", 0),
            "judged": board.failed.get("judged", 0),
            # A number on the cost's ruler, not its text: the page right-aligns a column of numbers, and the cell
            # spells it with its currency sign (``$3.092``) the way every money cell does.
            "cost": None if board.reply_cost_usd is None else board.reply_cost_usd * cost.factor,
        }
        if lost_any:
            row["lost"] = sum(board.lost.values())
        if other_causes:
            row["other"] = sum(board.failed.get(cause, 0) for cause in other_causes)
        rows.append(row)
    columns = [
        # The ▸ marking the control is defined where the glossary defines the control.
        TableColumn(key="arm", header="Arm", term="control" if control is not None else None),
        TableColumn(key="ran", header="Ran", term="ran"),
        *([TableColumn(key="lost", header="Lost", term="lost")] if lost_any else []),
        TableColumn(key="usable", header="Usable", term="usable"),
        TableColumn(key="passed", header="Passed", term="passed"),
        TableColumn(key="check", header="Failed by check", term="failed by check"),
        TableColumn(key="judged", header="Failed by judge", term="failed by judge"),
        *([TableColumn(key="other", header=_other_failures_header(other_causes))] if other_causes else []),
        # The term the header says, so the glossary defines the words a reader sees.
        TableColumn(
            key="cost",
            header=f"Cost {_per_reply(bundle)}",
            term=f"cost {_per_reply(bundle)}" if f"cost {_per_reply(bundle)}" in GLOSSARY else "cost per reply",
        ),
    ]
    rows = [{column.key: row[column.key] for column in columns} for row in rows]
    surface = bundle_decision_surface(bundle)
    cells = [
        {
            **_scoreboard_cells(row, forms),
            # Money in the one form its ruler chose over every arm's cost, held as a figure of the cost the floors read.
            "cost": forms.of_measure(
                _arm_cost_at(
                    forms.ruled_level_cell(cost, board.reply_cost_usd, measure=FLOOR_COST_MEASURE, kind="measure"),
                    surface,
                    board.variant_key,
                    board.reply_cost_usd,
                ),
                measure=FLOOR_COST_MEASURE,
                kind="measure",
            ),
        }
        for row, board in zip(rows, boards, strict=True)
    ]
    blocks: list[ReportBlock] = [
        TableBlock(
            section="scoreboard",
            name="scoreboard",
            title="Scoreboard",
            columns=columns,
            rows=cells,
            order="the control first, then the other arms by name",
            total_rows=len(rows),
        ),
        ChartBlock(
            section="scoreboard",
            chart="outcome_bar",
            title="Outcomes, share of each arm's trials",
            bars=[
                _outcome_bar(board, arm, tags[board.variant_key], shorts.get(board.variant_key), forms, passing)
                for board in boards
            ],
        ),
    ]
    blocks += [
        DisclosureBlock(section="scoreboard", source="caveat", text=line, term="case floor")
        for line in caveat_lines(bundle, arm)
    ]
    if bundle.all_failed_cells:
        every_failed = list(dict.fromkeys(arm(cell.variant_key) for cell in bundle.all_failed_cells))
        blocks.append(
            DisclosureBlock(
                section="scoreboard",
                source="runs",
                text=(
                    f"Every trial of {', '.join(every_failed)} failed: the model refused or errored on every call. "
                    "It took no turn, so it has no cost or latency to read, and every failure counts against it."
                ),
            )
        )
    return blocks


def _arm_cost_at(cell: TableCell, surface: DecisionSurface, variant_key: str, figure: float | None) -> TableCell:
    """An arm's cost per reply, carrying the id a finding citing that cost lands on where the arm ran under one test
    setup, so the arm is one cell, and the figure is that cell's own (:func:`_at`)."""
    held = [cell for cell in surface.cells if cell.variant_key == variant_key]
    if len(held) != 1:
        return cell
    return _at(cell, surface, held[0], FLOOR_COST_MEASURE, "measure", figure)


def _scoreboard_cells(row: Mapping[str, CellValue], forms: PageForms) -> dict[str, TableCell]:
    """A scoreboard row's cells but its cost, which its ruler's one form spells: counts as a table prints them, and
    words as they are."""
    return {key: forms.count_cell(value) for key, value in row.items() if key != "cost"}


def _other_failures_header(causes: Sequence[str]) -> str:
    """The header over failures the scoreboard counts apart from checks and the judge: the one cause's own words
    (``Model failed``), or ``Failed otherwise`` over several, which the outcome bar then names one by one."""
    if len(causes) == 1:
        words = cause_short(causes[0])
        return words[:1].upper() + words[1:]
    return "Failed otherwise"


def _outcome_bar(
    board: ArmScoreboard,
    arm: Callable[[str], str],
    tag: str,
    short: str | None,
    forms: PageForms,
    passing: MeasureRuler,
) -> OutcomeBar:
    """One arm's census as its stacked bar, tagged and short-named as every chart of the page tags and names it:
    passed, then each failure cause, then each loss cause; the share that passed in the pass rate's one form on the
    page, as the scorecard prints it."""
    segments = [OutcomeSegment(state="passed", label="passed", count=board.passed)] if board.passed else []
    segments += [
        OutcomeSegment(state="failed", label=f"failed: {cause_short(cause)}", count=count)
        for cause, count in sorted(board.failed.items(), key=lambda item: (-item[1], item[0]))
    ]
    segments += [
        OutcomeSegment(state="lost", label=f"lost: {cause_short(cause)}", count=count)
        for cause, count in sorted(board.lost.items(), key=lambda item: (-item[1], item[0]))
    ]
    ran = sum(segment.count for segment in segments)
    # The note's share is of every trial the bar draws, lost ones too, at the pass rate's digits: the pass rate itself
    # where no trial was lost (its passes of its usable trials), so the note and the scorecard print it alike; else a
    # share of its own, recorded as one.
    role: Role = "level" if ran == board.usable else "share"
    share = (
        forms.ruled_level(passing, board.passed / ran, measure=TRIAL_PASS_NAME, kind="trial", role=role).text
        if ran
        else None
    )
    return OutcomeBar(arm=arm(board.variant_key), segments=segments, tag=tag, short_name=short, passed_text=share)


__all__: list[str] = []
