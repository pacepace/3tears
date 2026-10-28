"""What a cost is per: the count a report input's money is stated against, read from what the input states.

A cost is money per something: per reply, per 1,000 conversations, per 1,000,000 turns. A page that prints one
cost "per 1,000 replies" when the campaign measured it per 1,000 conversations states a different figure, and the
digit rule then refuses the prose that would put it right. So every phrase a page prints a cost's count with
(``per 1,000 conversations``) is written here, from the input, and never assumed.

**What the input states.** A money measure declares its unit (``usd``, ``usd_per_1k``, ``usd_per_1m``) and a label:

- money per one thing, a ``usd`` measure, and the scoreboard's ``reply_cost_usd``, is per reply: the input contract
  states ``usd`` as dollars per reply, and the package counts one trial as one reply
  (:data:`~threetears.evals.analysis.report.figures.PER_THOUSAND_REPLIES`);
- money per 1,000 or per 1,000,000 of something, a ``usd_per_1k`` or ``usd_per_1m`` measure, is per what its label
  says it counts: the words after the count in the label (``Cost per 1k conversations``: ``conversations``; the
  count written ``1k``, ``1,000``, ``1000`` or ``thousand``, ``1m``, ``1,000,000`` or ``million``, the count its unit
  states). A label that names no count, or a count other than its unit's, states nothing, and the count stays
  unstated.

*(inferred: the label is the only place a reference input states the count today; reading it is reading the input,
and a label in another language that writes the count as digits reads the same.)*

**Unstated is said, not filled in.** Where the input does not state what a cost is per, the phrase is neutral
(``per 1,000 (unit not stated)``) and :func:`per_noun` returns empty, so a caller that raises
caveats sees that the cost's count is unstated (``CostBasis.stated`` in
:func:`~threetears.evals.analysis.report.figure_keys.cost_basis`).

Nothing here refuses: a label that does not read is a count unstated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from threetears.evals.kernel.numbers import format_number, group_thousands
from threetears.evals.analysis.report_input import Measure

__all__ = [
    "REPLIES",
    "CostBasis",
    "per_count",
    "per_noun",
    "per_one",
]

#: What money per one trial is per, as the package counts trials: one trial is one reply.
REPLIES = "replies"

#: What a count reads where the input does not say what a cost counts.
_UNSTATED = "(unit not stated)"

#: How a count of what a cost is per reads: ``per 1,000 conversations``.
_PER_TEMPLATE = "per {count} {noun}"

#: Each money unit's count, the one a figure in it is stated per.
_UNIT_COUNTS: dict[str, int] = {"usd_per_1k": 1_000, "usd_per_1m": 1_000_000}

#: How a label may write each count: as digits (with any thousands separator, or a ``k``/``m`` suffix) or as a word.
_COUNT_SPELLINGS: dict[int, str] = {
    1_000: r"1\s?k|1[,.   ]?000|thousand",
    1_000_000: r"1\s?m|1(?:[,.   ]?000){2}|million",
}

#: The longest noun a label's count is read with, in words: a count's noun is a word or a short phrase.
_MOST_WORDS = 4


@dataclass(frozen=True)
class CostBasis:
    """What a page's cost (``cost_per_1k.<arm>``) is per, and where it is read from.

    :ivar noun: What one of the count is, as a plural noun (``conversations``); empty where the input does not say.
    :ivar source: The measure the cost is read from, or ``scoreboard``; empty where the input states no cost.
    """

    noun: str
    source: str

    @property
    def stated(self) -> bool:
        """Whether the input says what its cost is per."""
        return bool(self.noun)


def per_noun(measure: Measure) -> str:
    """What one money measure is per, as a plural noun; empty where it is no money measure or does not say.

    :param measure: The measure.
    :ptype measure: Measure
    :return: The noun (``replies`` for money per one thing; the label's noun for money per a count).
    :rtype: str
    """
    if measure.unit == "usd":
        return REPLIES
    count = _UNIT_COUNTS.get(measure.unit or "")
    if count is None or not measure.label:
        return ""
    found = re.search(rf"(?<!\w)(?:{_COUNT_SPELLINGS[count]})\s+(?P<noun>[^\d()\[\],;:]+)", measure.label, re.I)
    if found is None:
        return ""
    noun = found.group("noun").strip()
    return noun if 0 < len(noun.split()) <= _MOST_WORDS else ""


def per_one(noun: str) -> str:
    """The words a cost per one of ``noun`` is printed with: ``per reply``, ``per conversation``.

    :param noun: What the cost counts, as a plural noun (:func:`per_noun`); empty where the input does not say.
    :ptype noun: str
    :return: ``per`` and the noun made singular (``replies`` to ``reply``, ``calls`` to ``call``); ``per`` and
        ``(unit not stated)`` where the noun is empty.
    :rtype: str
    """
    if not noun:
        return f"per {_UNSTATED}"
    *rest, last = noun.split(" ")
    if last.endswith("ies"):
        last = last[:-3] + "y"
    elif last.endswith("s") and not last.endswith("ss"):
        last = last[:-1]
    return "per " + " ".join([*rest, last])


def per_count(noun: str, count: int = 1_000) -> str:
    """The words a cost per ``count`` of ``noun`` is printed with: ``per 1,000 conversations``.

    :param noun: What one of the count is, as :func:`per_noun` reads it; empty where the input does not say.
    :ptype noun: str
    :param count: How many of it the cost is per.
    :ptype count: int
    :return: The phrase; the count with ``(unit not stated)`` where the noun is empty.
    :rtype: str
    """
    return _PER_TEMPLATE.format(count=group_thousands(format_number(float(count))), noun=noun or _UNSTATED)
