"""The one cost rule: each arm's cost per 1,000 of what it counts, read from the first source that states one.

What the decision ranks options by on its cost axis, and what every page prints as an arm's cost: one rule, read on
the numbers side, so a page's wording never moves the call. The sources, in the order they are tried: a measure stated
per 1,000; one stated per 1,000,000; the one dollar measure on the cost merit axis; the scoreboard's cost per reply;
the one dollar measure where lower is better; then each of those measures blended over the tasks by traffic share.

**What a cost counts** (:func:`per_noun`). Money per one thing, a ``usd`` measure, and the scoreboard's
``reply_cost_usd``, is per reply: the package counts one trial as one reply. Money per 1,000 or per 1,000,000 of
something is per what its label says it counts: the words after the count in the label (``Cost per 1k
conversations``: ``conversations``). A label that names no count, or a count other than its unit's, states nothing.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final

from threetears.evals.analysis.report_input import Measure, ReportInput, ScoreboardRow

__all__ = [
    "COUNT_SPELLINGS",
    "PER_THOUSAND_FACTORS",
    "REPLIES",
    "SCOREBOARD_COST",
    "cost_measures",
    "cost_source",
    "costs_per_thousand",
    "per_noun",
    "with_reply_costs",
]

#: The dollar scales, each with the factor that states its figure per 1,000 of what it counts.
PER_THOUSAND_FACTORS: Final[Mapping[str, float]] = {"usd": 1000.0, "usd_per_1k": 1.0, "usd_per_1m": 0.001}

#: What money per one trial is per, as the package counts trials: one trial is one reply.
REPLIES = "replies"

#: Each money unit's count, the one a figure in it is stated per.
_UNIT_COUNTS: Final[Mapping[str, int]] = {"usd_per_1k": 1_000, "usd_per_1m": 1_000_000}

#: How a label may write each count: as digits (with any thousands separator, or a ``k``/``m`` suffix) or as a word.
COUNT_SPELLINGS: Final[Mapping[int, str]] = {
    1_000: r"1\s?k|1[,.   ]?000|thousand",
    1_000_000: r"1\s?m|1(?:[,.   ]?000){2}|million",
}

#: The longest noun a label's count is read with, in words: a count's noun is a word or a short phrase.
_MOST_WORDS = 4


def per_noun(measure: Measure) -> str:
    """What one money measure is per, as a plural noun; empty where it is no money measure or does not say.

    :param measure: The measure.
    :return: The noun (``replies`` for money per one thing; the label's noun for money per a count).
    """
    if measure.unit == "usd":
        return REPLIES
    count = _UNIT_COUNTS.get(measure.unit or "")
    if count is None or not measure.label:
        return ""
    found = re.search(rf"(?<!\w)(?:{COUNT_SPELLINGS[count]})\s+(?P<noun>[^\d()\[\],;:]+)", measure.label, re.I)
    if found is None:
        return ""
    noun = found.group("noun").strip()
    return noun if 0 < len(noun.split()) <= _MOST_WORDS else ""


def cost_measures(report_input: ReportInput) -> tuple[list[str], list[str]]:
    """The measures an arm's cost may be read from, in the order they are tried: each one alone of its kind.

    First the measures read before the scoreboard: the one stated per 1,000 (``usd_per_1k``), per 1,000,000,
    the one in dollars on the cost merit axis. Then the one dollar measure, in any of those units, where lower is
    better, read after it.
    """
    measures = report_input.measures

    def alone(keys: list[str]) -> list[str]:
        return keys if len(keys) == 1 else []

    first = [
        *alone([key for key, m in measures.items() if m.unit == "usd_per_1k"]),
        *alone([key for key, m in measures.items() if m.unit == "usd_per_1m"]),
        *alone([key for key, m in measures.items() if m.unit == "usd" and m.merit_axis == "cost"]),
    ]
    lower = alone(
        [key for key, m in measures.items() if m.unit in PER_THOUSAND_FACTORS and m.higher_is_better is False]
    )
    return first, lower


def _overall_cost(report_input: ReportInput, key: str) -> dict[str, float]:
    measure = report_input.measures[key]
    factor = PER_THOUSAND_FACTORS[measure.unit or ""]
    return {arm: stat.mean * factor for arm, stat in measure.per_arm.items() if stat.mean is not None}


def _blended_cost(report_input: ReportInput, key: str) -> dict[str, float]:
    """Each arm's cost on one measure blended over the tasks: its task figures weighted by each task's traffic share.

    An arm is blended over the tasks that carry traffic and hold its figure, each weighted by its share of their
    traffic, and left out where any of those figures is unstated.
    """
    factor = PER_THOUSAND_FACTORS[report_input.measures[key].unit or ""]
    blended: dict[str, float] = {}
    for arm in report_input.arms:
        parts = [
            (float(task.share), task.per_arm[arm.id][key].mean)
            for task in report_input.tasks
            if task.share and arm.id in task.per_arm and key in task.per_arm[arm.id]
        ]
        means = [mean for _, mean in parts if mean is not None]
        if parts and len(means) == len(parts):
            total = sum(share for share, _ in parts)
            blended[arm.id] = factor * sum(share * mean for (share, _), mean in zip(parts, means, strict=True)) / total
    return blended


#: The cost source that is the scoreboard's cost per reply.
SCOREBOARD_COST = "scoreboard"


def cost_source(report_input: ReportInput) -> tuple[str, dict[str, float]]:
    """The first cost source that holds a figure, by name (a measure's key, or :data:`SCOREBOARD_COST`), and each
    arm's cost per 1,000 on it; ``("", {})`` where none does (:func:`costs_per_thousand`'s order)."""
    first, lower = cost_measures(report_input)
    sources = [(key, _overall_cost(report_input, key)) for key in first]
    sources.append(
        (
            SCOREBOARD_COST,
            {
                a: row.reply_cost_usd * 1000
                for a, row in report_input.scoreboard.items()
                if row.reply_cost_usd is not None
            },
        )
    )
    sources += [(key, _overall_cost(report_input, key)) for key in lower]
    sources += [(key, _blended_cost(report_input, key)) for key in dict.fromkeys([*first, *lower])]
    return next(((name, source) for name, source in sources if source), ("", {}))


def costs_per_thousand(report_input: ReportInput) -> dict[str, float]:
    """Each arm's cost per 1,000 of what it counts, from the first source that holds one: the one rule a page and the
    decision read cost by (``cost_per_1k.<arm>``).

    The arms' overall figures on a measure stated per 1,000; else on one stated per 1,000,000, divided by 1,000; else
    on the one dollar measure on the cost merit axis, times 1,000; else the scoreboard's cost per reply, times 1,000;
    else the one dollar measure where lower is better. Where none of those states a figure, each measure in the same
    order blended over the tasks by traffic share (:func:`_blended_cost`). What the count counts is
    :func:`~threetears.evals.analysis.report.keys.reads.cost_basis`'s.
    """
    source, costs = cost_source(report_input)
    return with_reply_costs(report_input, source, costs, report_input.scoreboard)


def with_reply_costs(
    report_input: ReportInput, source: str, costs: dict[str, float], board: Mapping[str, ScoreboardRow]
) -> dict[str, float]:
    """``costs`` (a source's figures, per 1,000) with each arm the source lacks and a scoreboard's cost per reply
    states, in that figure times 1,000.

    A source can leave an arm out that the scoreboard states: the cost per reply is the same cost over the same count
    where the source counts replies, so the arm is not left without a cost beside arms that carry one. Every table and
    key reads an arm's cost here, so one arm states one cost. A source counting anything else takes no fallback.
    """
    if source == SCOREBOARD_COST:
        return costs
    measure = report_input.measures.get(source)
    if measure is None or per_noun(measure) != REPLIES:
        return costs
    return {
        **{arm: row.reply_cost_usd * 1000 for arm, row in board.items() if row.reply_cost_usd is not None},
        **costs,
    }
