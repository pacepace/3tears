"""The words code writes on a planned page: every line the plan has no prose for, one template each.

The planner writes the page's sentences; code writes its frame: the decision's confidence (GENERATOR §2.5's three forms,
verbatim), the decision slots' titles by mode and kind, the options table's role words and column heads, each
warning an answer answers, the arms key's eyebrow and takeaway, an example card's missing reason, and the captions a
chart needs that no planner sentence holds. Held here so a translation replaces them in one place.
"""

from __future__ import annotations

from threetears.evals.analysis.report.figures import PER_THOUSAND_REPLIES

__all__ = [
    "ANSWER_TEMPLATES",
    "APPENDIX_TITLE",
    "ARM_HEADERS",
    "ARM_ROLE_WORDS",
    "ARMS_EYEBROW",
    "ARMS_TAKEAWAY",
    "CONFIDENCE_LABEL",
    "CONFIDENCE_WORDS",
    "EXPECTED_TERM",
    "FULL_RESULTS_TITLE",
    "GATE_WORDS",
    "HUMAN_PREF_NOTE",
    "INPUT_TERM",
    "LEVER_NOTE",
    "MISSING_TEMPLATE",
    "NO_REASON",
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
    "change_words",
    "reading_words",
    "slot_title",
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
    "cost_per_1k": "Cost " + PER_THOUSAND_REPLIES,
}

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
