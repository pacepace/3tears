"""The words code writes on a planned page: every line the plan has no prose for, one template each.

The planner writes the page's sentences; code writes its frame: the decision's confidence (GENERATOR §2.5's three forms,
verbatim), the decision slots' titles by mode and kind, the options table's role words and column heads, each
warning an answer answers, the arms key's eyebrow and takeaway, an example card's missing reason, the captions a
chart needs that no planner sentence holds, the explore part's flip table, and the Full results tables' titles,
heads and status words. Held here so a translation replaces them in one place.
"""

from __future__ import annotations

__all__ = [
    "CASES_WORD",
    "COMPUTED_REPEATS_NOTE",
    "PAIRED_BASIS_TEMPLATE",
    "SETTLE_FLOOR_TEMPLATE",
    "SETTLE_MARGIN_TEMPLATE",
    "SETTLE_EACH",
    "SETTLE_NOTE",
    "BAR_WORD",
    "BAR_STRIP_NOTE",
    "OVERALL_WORD",
    "LEVER_CROSS_GROUP",
    "LEVER_CROSS_TEMPLATE",
    "LEVER_EFFECT_GROUP",
    "LEVER_MOVE_TEMPLATE",
    "LEVERS_NOTE",
    "LEVERS_INTERACTION_NOTE",
    "LEVERS_READ_NOTE",
    "COMPARE_NOTE",
    "BASIS_NOTE",
    "ANSWER_TEMPLATES",
    "APPENDIX_TITLE",
    "ARM_HEADERS",
    "ARM_ROLE_WORDS",
    "ARMS_EYEBROW",
    "ARMS_TAKEAWAY",
    "ASPECT_MEASURE_TEMPLATE",
    "CONFIDENCE_LABEL",
    "COMPUTED_NOTE",
    "CONFIDENCE_WORDS",
    "CONTRAST_VERDICT_WORDS",
    "EXPECTED_TERM",
    "EXPLORE_HEADERS",
    "EXPLORE_HOLDS",
    "EXPLORE_TITLE",
    "FULL_RESULTS_HEADERS",
    "FULL_RESULTS_TITLE",
    "FULL_RESULTS_TITLES",
    "GATE_WORDS",
    "HUMAN_PREF_NOTE",
    "INPUT_TERM",
    "LEVER_NOTE",
    "LINE_TEMPLATES",
    "CRITERION_LINE_TEMPLATE",
    "CRITERION_LINE_WHERE",
    "MADE_TEMPLATE",
    "MARGIN_STATUS_WORDS",
    "MISSING_TEMPLATE",
    "NO_FIGURE",
    "NO_REASON",
    "OVERALL",
    "OPTIONS_TITLE",
    "OPTION_HEADERS",
    "PASSES_TEMPLATE",
    "REFERENCE_TEMPLATE",
    "REPEAT_TEMPLATE",
    "ROLE_WORDS",
    "RUN_TEMPLATE",
    "SYNTHETIC",
    "STATISTIC_NOTE",
    "UNCORRECTED_NOTE",
    "GATE_HEADERS",
    "GATE_LINE_WORDS",
    "GATE_PLACE_WORDS",
    "CROSS_CUTTING_FLAG",
    "FLOOR_ON_TEMPLATE",
    "FLOOR_READS_TEMPLATE",
    "FLOOR_REASON_TEMPLATE",
    "FLOOR_WORDS",
    "LOG_AXIS_NOTE",
    "NO_READING",
    "NOT_GATED",
    "PASSES_EVERY_GATE",
    "PICK_LABEL",
    "REGION_LABEL",
    "TASK_DETAIL_TEMPLATE",
    "TASK_SHARE_TEMPLATE",
    "change_words",
    "reading_words",
    "slot_title",
    "FLOOR_STRIP_NOTE",
    "MISSES_TEMPLATE",
    "LOG_X_NOTE",
    "RANGE_TEMPLATE",
    "P95_TEMPLATE",
    "NO_MEDIAN_TEMPLATE",
    "CASES_TEMPLATE",
    "BELOW_TEMPLATE",
    "DIFFERENCE_NOTE",
    "POINTS_WORDS",
    "UNIT_WORDS",
    "CONFUSION_NOTE",
    "PRICE_OPEN_NOTE",
    "PRICE_STEP_NOTES",
    "PRICE_STEP_TEMPLATE",
    "TABLE_HEADERS",
    "TABLE_PER_THOUSAND",
    "COST_HEADER",
    "INTERVAL_HEADER",
    "LOG_BARS_NOTE",
    "CELL_TEMPLATE",
    "TABLE_ORDER",
]

#: GENERATOR §2.5: how sure the decision is, in exactly three forms, verbatim.
CONFIDENCE_WORDS: dict[str, str] = {
    "high": "High",
    "high_if_immaterial": "High, if the caveats below aren't material",
    "low": "Low",
}

#: What the confidence box's first line opens with.
CONFIDENCE_LABEL = "Confidence"

#: GENERATOR §2.5: each slot's title. A decide plan's second slot depends on the kind (:func:`slot_title`).
_SLOT_TITLES: dict[str, str] = {
    "recommendation": "Recommendation",
    "watch": "Watch",
    "next": "Next",
    "the_trade": "The trade",
    "where_it_flips": "Where the answer flips",
    "our_lean": "Our lean",
    "what_settles_it": "What would settle it",
}

#: The second decide slot's title: why not the candidates against a control, why not the others in a choice.
_WHY_NOT_TITLES: dict[bool, str] = {True: "Why not the candidates", False: "Why not the others"}


def slot_title(key: str, *, change: bool) -> str:
    """A decision slot's title (GENERATOR §2.5).

    :param key: The slot's key.
    :ptype key: str
    :param change: Whether the campaign measures candidates against a control.
    :ptype change: bool
    :return: The title.
    :rtype: str
    """
    return _WHY_NOT_TITLES[change] if key == "why_not" else _SLOT_TITLES[key]


#: GENERATOR §2.6: why an option is in the options table, as its role column says it.
ROLE_WORDS: dict[str, str] = {
    "recommended": "Recommended",
    "lean": "Our lean",
    "control": "Control",
    "near_miss": "Near miss",
    "most_accurate": "Most accurate",
    "cheapest": "Cheapest",
    "alternative": "Alternative",
}

#: The options table's title, which a reader of its values (a screen reader, Markdown) is given.
OPTIONS_TITLE = "Options"

#: The options table's column heads: the option and its role, then each column family's (a judged dimension's and a
#: measure's head is its reader name).
OPTION_HEADERS: dict[str, str] = {
    "option": "Option",
    "role": "Role",
    "pass_rate": "Pass rate",
    "reading.change": "Change (points)",
    "reading.verdict": "Reading",
    "gate": "Gates",
}

#: The cost column's head, wherever a table states the page's one cost: what it is per, as the input states it
#: (:func:`~threetears.evals.analysis.report.cost_unit.per_count`, ``per 1,000 conversations``).
COST_HEADER = "Cost {per}"

#: A count out of a total, as a cell's marker reads it: passes over usable trials, usable trials over those run.
PASSES_TEMPLATE = "{passed} of {usable}"

#: An arm's role, as the arms key and the options table's control row say it.
ARM_ROLE_WORDS: dict[str, str] = {"control": "Control", "challenger": "Candidate"}

#: The arms key's column heads.
ARM_HEADERS: dict[str, str] = {
    "arm": "Arm",
    "role": "Role",
    "settings": "Configuration",
    "served_model": "Served model",
    "reasoning": "Share of output spent reasoning",
    "usable": "Trials usable",
}

#: GENERATOR §3: an arm's status against the gates, as its pill says it.
GATE_WORDS: dict[str, str] = {"pass": "Passes", "borderline": "Borderline", "fail": "Fails"}

#: Each warning code raised on the recommended arm, as the answer to it names it: one template per warning kind.
ANSWER_TEMPLATES: dict[str, str] = {
    "floor_broken": "The {name} floor on {aspect} is broken.",
    "floor_unclear": "The {name} floor on {aspect} is unclear.",
    "behind_on": "{name} is shown ahead on {aspect}.",
    "bar_missed": "The bar on {name} is missed on {aspect}.",
}

#: The arms key's eyebrow (GENERATOR §2.8): the plan writes no prose for it.
ARMS_EYEBROW = "The arms"

#: The arms key's one-line takeaway, in code's words.
ARMS_TAKEAWAY = "Each arm's full configuration, once. The rest of the page names an arm by its letter tag and name."

#: The Full results part's title (GENERATOR §2.11).
FULL_RESULTS_TITLE = "Full results"

#: Each Full results table's title, by the plan's name for it (GENERATOR §2.11); ``tasks`` is the strata table's
#: second part, where the input holds tasks.
FULL_RESULTS_TITLES: dict[str, str] = {
    "arms_by_measure": "Every arm by measure",
    "contrasts": "Every corrected comparison",
    "strata": "Every stratum",
    "tasks": "Every task",
    "gate_matrix_full": "Every gate, wherever it is read",
    "configs": "Every arm's configuration",
    "cells": "Every cell",
}

#: The Full results tables' column heads (an arm's column is headed by its name, a measure's by its reader name, a
#: factor's by its name).
FULL_RESULTS_HEADERS: dict[str, str] = {
    "arm": "Arm",
    "pass_rate": "Pass rate",
    "measure": "Measure",
    "aspect": "Aspect",
    "change": "Change",
    "interval": "Interval",
    "level": "Level",
    "p_adjusted": "Adjusted p",
    "verdict": "Verdict",
    "confounds": "Confounds",
    "stratum": "Stratum",
    "task": "Task",
    "cell": "Cell",
    "cases": "Cases",
    "where": "Where",
    "criterion": "Criterion",
    "line": "Line",
}

#: A corrected comparison's verdict as its pill says it, by the pipeline's verdict.
CONTRAST_VERDICT_WORDS: dict[str, str] = {
    "improved": "Improved",
    "regressed": "Regressed",
    "equivalent": "Equivalent",
    "not_separated": "Not separated",
    "untested": "Untested",
}

#: A contrast's measure, as a table or change chart names it where the measure's selected contrasts sit on more than one
#: aspect: each is told apart by the aspect it was read on.
ASPECT_MEASURE_TEMPLATE = "{measure}, on {aspect}"

#: Where a criterion read on an arm's overall figure is read, in the gate matrix's Where column.
OVERALL = "Overall"

#: A criterion's line, by its direction.
LINE_TEMPLATES: dict[str, str] = {"higher_is_better": "at least {line}", "lower_is_better": "at most {line}"}

#: A declared criterion's line as a chart labels it: its reader name and its kind (gate, floor, bar, guardrail), the
#: side that passes (:data:`LINE_TEMPLATES`), and, where it reads its figure elsewhere than the chart does, where
#: (:data:`CRITERION_LINE_WHERE`): an overall gate drawn in each task's panel is the overall line, not a task's.
CRITERION_LINE_TEMPLATE = "{name} {kind}, {side}{where}"
CRITERION_LINE_WHERE = {
    "overall": " overall",
    "each_task": " in each task",
    "task": " in {task}",
    "cell": " in each cell",
}

#: What a gate matrix cell says beside its status where the input holds no figure there.
NO_FIGURE = "no figure"

#: The explore part's flip table: its title (a screen reader and Markdown are given it) and its heads.
EXPLORE_TITLE = "Where each reading flips"
EXPLORE_HEADERS: dict[str, str] = {
    "arm": "Arm",
    "aspect": "Aspect",
    "flip": "Holds at any margin above",
    "declared": "At the declared margin",
}

#: A reading at its declared margin, as its pill says it.
MARGIN_STATUS_WORDS: dict[bool, str] = {True: "Within the margin", False: "Outside the margin"}

#: Whether the call holds at every margin, as the line under the flip table says it.
EXPLORE_HOLDS: dict[bool, str] = {
    True: "The call holds at every margin.",
    False: "The call changes at some margin: a reading is within any margin above the point it flips at.",
}

#: How a stored analysis's planned page was made, in its one line at the foot (GENERATOR §2: the made line).
MADE_TEMPLATE = (
    "Planned by {model} on {date} as analysis {analysis}; code drew every figure and chart from the campaign's "
    "evidence as frozen with it."
)

#: The closed appendix's title: the evidence record a page built from a plan keeps below Method.
APPENDIX_TITLE = "Evidence record"

#: The masthead's mark on a page of made-up data (GENERATOR §2.1).
SYNTHETIC = "Synthetic example data"

#: The masthead's run date.
RUN_TEMPLATE = "Run {date}"

#: An example card whose trial has no judge reason.
NO_REASON = "No judge reason recorded."

#: An example card's terms for what the case gave and what it expected.
INPUT_TERM = "Input"
EXPECTED_TERM = "Expected"

#: An example card's repeat.
REPEAT_TEMPLATE = "repeat {repeat}"

#: A chart's caption naming each comparison its query asks for that the input does not hold (GENERATOR §2: never
#: fill it in).
MISSING_TEMPLATE = "Not in the evidence: {missing}."

#: A reference line's label: its kind and where it sits.
REFERENCE_TEMPLATE = "{kind} {value}"

#: A chart of per-slice or per-level changes code computed from point figures (GENERATOR §5.2: descriptive, not the
#: pipeline's tests).
UNCORRECTED_NOTE = (
    "Changes computed from each arm's figures, uncorrected. Bold rows are separated at 95% before any correction; "
    "read them as leads."
)

#: A chart of lever effects (GENERATOR §1, shape B): code's averages, not the pipeline's tests.
LEVER_NOTE = (
    "Each level against its factor's first level, averaged over every combination of the other factors that ran "
    "both: descriptive, uncorrected."
)

#: A bar's line, as its label names it, and what a strip of bar reads says.
BAR_WORD = "bar"
BAR_STRIP_NOTE = "Each dot is the arm's value, with the interval its bar was read on; a hollow dot misses its bar."

#: A settle chart's or table's caption: what its figure is, ``{each}`` naming where it sits for the kind drawn
#: (:data:`SETTLE_EACH`, a bar's length, or a table's column by its header).
SETTLE_NOTE = (
    "{each} how often a fresh run of today's size would read the figure on the other side of its line, if the "
    "truth is today's estimate ({level}); it plans a run and tests nothing."
)
SETTLE_EACH = {"bars": "Each bar is", "table": 'In each row, "{column}" is'}

#: What a settle row against an aspect's margin, or a floor, is settled against.
SETTLE_MARGIN_TEMPLATE = "the margin on {aspect}"
SETTLE_FLOOR_TEMPLATE = "the floor {floor} on {aspect}"

#: A compare chart's caption: whose difference each row is, and on what.
COMPARE_NOTE = "Each row is {first} less {second} on {measure}, overall and in each place both hold it."

#: How a compare chart's intervals were read, in the analysis's own words.
BASIS_NOTE = "The interval is {basis}."

#: How a compare row's interval reads where a contrast pairs the two arms, in reader names.
PAIRED_BASIS_TEMPLATE = (
    "paired: the contrast of {arm} against the control on {measure}, its own interval, corrected within its family"
)

#: What a settle table's sizes count on a repeated design: the distinct cases.
CASES_WORD = "cases"

#: What a compare row overall is called, beside the slices' names.
OVERALL_WORD = "Overall"

#: A lever chart's caption: what an effect is, and, where the chart draws any, what an interaction is
#: (:data:`LEVERS_INTERACTION_NOTE` after it), then how both are read.
LEVERS_NOTE = "Each effect is one setting moved between two arms that differ in it alone, on {measure}{interactions}."
LEVERS_INTERACTION_NOTE = (
    "; each interaction is the same move's effect in one context less its effect in another, the two contexts "
    "differing in one other setting"
)
LEVERS_READ_NOTE = "Descriptive: read as two arms' own spreads, never a test."

#: A lever effect's label (the arm it moves from, and the value it moves to), and the group effects sit under.
LEVER_MOVE_TEMPLATE = "{start} → {value}"
LEVER_EFFECT_GROUP = "{setting}"

#: An interaction's label: the second move's effect less the first's, each move as an effect's label names it (the
#: arm it starts from, and the value it moves to), so two interactions read apart wherever their moves do.
LEVER_CROSS_TEMPLATE = "({second}) less ({first})"
LEVER_CROSS_GROUP = "{setting} effect, across {across}"

#: A chart whose intervals code computed from each rate's binomial spread, where its figures state none.
COMPUTED_NOTE = (
    "Each interval is computed here from the rate's binomial spread, by the normal approximation, held to 0 to 100%. "
    "A rate of none or all has none."
)

#: Beside it, where a case ran more than once: what count the spread was read on.
COMPUTED_REPEATS_NOTE = (
    "Where a case ran more than once, the interval is read on the distinct cases, each counted once: the widest "
    "spread its repeats allow."
)

#: A chart of blind human preference, about the no-preference point.
HUMAN_PREF_NOTE = "The share of {n} blind ratings preferring {control} over {arm}; 50% is no preference."

#: A chart whose figures have no mean, drawn at a percentile instead.
STATISTIC_NOTE = "Each bar is the arm's {statistic}."

#: GENERATOR §3: which way a change reads, by the direction of the change, never by the margin.
_CHANGE_WORDS: dict[str, str] = {"worse": "worse", "better": "better", "": "not separated"}


def change_words(direction: str) -> str:
    """A change's direction as a chart note says it (GENERATOR §3): ``worse``, ``better``, or ``not separated``.

    :param direction: ``worse``, ``better``, or empty where the interval crosses zero.
    :ptype direction: str
    :return: The words.
    :rtype: str
    """
    return _CHANGE_WORDS[direction]


#: GENERATOR §3: a reading's status as the options table's pill says it, the margin verdict in the words.
_READING_WORDS: dict[str, str] = {"worse": "Worse", "better": "Better", "": "Not separated"}
_MARGIN_WORDS: dict[bool, str] = {True: "within margin", False: "outside margin"}


def reading_words(direction: str, within: bool | None) -> str:
    """A reading's status as its pill says it: its direction, then whether it is within the margin, where one is
    declared (``Worse, within margin``).

    :param direction: ``worse``, ``better``, or empty where the interval crosses zero.
    :ptype direction: str
    :param within: Whether the reading is within its margin; None where no margin is declared.
    :ptype within: bool | None
    :return: The words.
    :rtype: str
    """
    said = _READING_WORDS[direction]
    return said if within is None else f"{said}, {_MARGIN_WORDS[within]}"


# --- the gate family's charts (GENERATOR §3, §5.1) ---------------------------------------------------------------

#: A gate matrix's first column's heading, by what its rows are.
GATE_HEADERS: dict[str, str] = {
    "tasks": "Task",
    "readings": "Aspect",
    "floors": "Floor",
    "measures": "Gate",
    "configs": "Gate",
}

#: The pick on a gate matrix's routable row (GENERATOR §3 Routing).
PICK_LABEL = "Cheapest passing"

#: A cross-cutting task's row: its gates hold for every candidate that would take routed traffic (GENERATOR §3).
CROSS_CUTTING_FLAG = "Applies to all traffic"

#: A task row's line: its cases, then its share of traffic where it carries some.
TASK_DETAIL_TEMPLATE = "{cases} cases"
TASK_SHARE_TEMPLATE = "{share} of traffic"

#: A cell where no gate is read in its task: no status to state, only the figures.
NOT_GATED = "Not gated here"

#: A readings cell for an arm the input holds no reading of on that aspect.
NO_READING = "No reading"

#: A floor's status as its pill says it: held is good, unclear borderline, broken a failure.
FLOOR_WORDS: dict[str, str] = {"held": "Held", "unclear": "Unclear", "broken": "Broken"}

#: Why a floor reads as it does: the worst drop its interval allows, against its margin.
FLOOR_REASON_TEMPLATE = "worst drop {drop} against a {margin} margin"

#: A floor that reads another rate than the pass rate (GENERATOR §3: shown apart, never merged with the reading).
FLOOR_READS_TEMPLATE = "reads {rate}"

#: Such a floor beside a reading: the rate it reads, and its status.
FLOOR_ON_TEMPLATE = "Floor on {rate}: {status}"

#: A gate's line as a row's detail says it, by the criterion's direction.
GATE_LINE_WORDS: dict[str, str] = {"higher_is_better": "at least {line}", "lower_is_better": "at most {line}"}

#: Where a gate is read, beyond the whole comparison, as a row's detail adds it.
GATE_PLACE_WORDS: dict[str, str] = {"task": "in every task", "cell": "in every cell", "one_task": "in {task}"}

#: A gate scatter's shaded region: where a point clears both gates.
REGION_LABEL = "Meets both gates"

#: A point's or tile's tooltip line where it clears every gate.
PASSES_EVERY_GATE = "Passes every gate"

#: A scatter's caption where an axis is drawn on a log scale (GENERATOR §5.2).
LOG_AXIS_NOTE = "The {axis} axis is on a log scale: its values span more than ten times."


#: The caption under a dot strip of floors.
FLOOR_STRIP_NOTE = "Each dot is the largest drop its interval allows; a floor holds where the dot stays above its line."

#: A dot's tooltip line where it misses its line.
MISSES_TEMPLATE = "Misses the {line}"

#: The caption under bars drawn on a log axis.
LOG_BARS_NOTE = "The bars are drawn on a log scale: their values span more than ten times, so each step is a multiple."

#: The caption under a scatter whose x axis is logarithmic.
LOG_X_NOTE = "The horizontal axis is logarithmic: each step is a multiple."

#: A range bar's tooltip line: its two ends.
RANGE_TEMPLATE = "p50 {p50} · p95 {p95}"

#: A range bar's tooltip line where the input holds only its p95, which is drawn as its end mark alone.
P95_TEMPLATE = "p95 {p95}; no p50 in the evidence"

#: The caption naming the range bars drawn as their p95 alone.
NO_MEDIAN_TEMPLATE = "Drawn at the p95 alone, with no p50 in the evidence: {missing}."

#: A heatmap cell's tooltip line under its place: its value and the cases under it.
CASES_TEMPLATE = "{value} · n = {n}"

#: A heatmap cell's tooltip line where its value is below the floor.
BELOW_TEMPLATE = "Below the {line}"

#: The caption naming which arm a difference heatmap draws less which, and in what unit.
DIFFERENCE_NOTE = "Each cell is {compared} less {base}, {unit}; the small figure is {compared}'s own."

#: What a rate's difference is stated in, in a difference heatmap's caption.
POINTS_WORDS = "in points"

#: What any other difference is stated in, in a difference heatmap's caption.
UNIT_WORDS = "in its own unit"

#: The caption under confusion matrices.
CONFUSION_NOTE = "Rows are the true label and columns the predicted one; each row is shared out to 100%."

#: A plain table chart's column header, by column key.
TABLE_HEADERS = {
    "arm": "Arm",
    "measure": "Measure",
    "value": "Value",
    "change": "Change",
    "reading": "Reading",
    "verdict": "Verdict",
    "figure": "Figure",
    "floor": "Floor",
    "status": "Status",
    "worst_drop": "Largest drop",
    "margin": "Margin",
    "slice": "Slice",
    "trial": "Trial",
    "case": "Case",
    "stratum": "Stratum or task",
    "outcome": "Outcome",
    "reason": "Reason",
    "cell": "Cell",
    "cases": "Cases",
    "axis": "Axis",
    "expected": "True label",
    "predicted": "Predicted label",
    "prefers_control": "Prefers the control",
    "step": "Step along the frontier",
    "per_point": "Price per point",
    "aspect": "Aspect",
    "bar": "Bar",
    "place": "Place",
    "first": "First arm",
    "second": "Second arm",
    "difference": "Difference",
    "interval": "Interval",
    "level": "Level",
    "basis": "Read as",
    "against": "Against",
    "estimate": "Today's figure",
    "line": "Line",
    "distance": "Distance toward passing",
    "settles": "Settles",
    "counted": "Sizes count",
    "cases_needed": "Needed, one-sided",
    "cases_needed_two_sided": "Needed, two-sided",
    "today": "Today",
    "trials_needed": "Trials needed, one-sided",
    "trials_needed_two_sided": "Trials needed, two-sided",
    "repeats": "Repeats of a case",
    "today_trials": "Trials today",
    "wrong_side": "A fresh run reads the other side",
    "setting": "Setting",
    "move": "Move",
}

#: How a step along a price frontier is named: the option it moves from, then the one it moves to.
PRICE_STEP_TEMPLATE = "{first} → {second}"

#: What a chart of price steps says its prices are: a month's at the declared volume, or per 1,000 of what the cost
#: counts, ``{per}`` the phrase (:func:`~threetears.evals.analysis.report.cost_unit.per_count`).
PRICE_STEP_NOTES = {
    True: "Each step's price per point a month at the declared volume: where the best option moves as a point's "
    "price rises.",
    False: "Each step's price per point {per}: where the best option moves as a point's price rises.",
}

#: What a chart or table of price steps says of each step whose gain's interval reaches zero: its price is open above,
#: ``{steps}`` the steps by name (:data:`PRICE_STEP_TEMPLATE`), joined by semicolons.
PRICE_OPEN_NOTE = (
    "No upper bound on {steps}: the gain's interval reaches zero there, so the evidence allows any price per point "
    "above the low end shown."
)

#: An interval column's head in a plain table: the chart's interval level, where it states one (``95% interval``).
INTERVAL_HEADER = "{level} interval"

#: What a plain table states a money figure per reply per, beside its measure's name: its count, as the input states
#: it (``per 1,000 replies``).
TABLE_PER_THOUSAND = "{name}, {per}"

#: How a target-matrix cell's two levels read together, in a plain table.
CELL_TEMPLATE = "{row} × {column}"

#: How a plain table's rows are ordered, in words.
TABLE_ORDER = "as the chart's query selects them, in the input's order"
