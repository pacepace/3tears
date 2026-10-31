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
from collections.abc import Iterable
from dataclasses import dataclass

from threetears.evals.kernel.numbers import MoneyForm, format_number, group_thousands, money_form
from threetears.evals.analysis.report_input import Measure, ReportInput
from threetears.evals.analysis.report_margins import aspect_board

__all__ = [
    "PRICED_SHARE_TEMPLATE",
    "PRICED_TRIALS_TEMPLATE",
    "MoneyRuler",
    "money_ruler",
    "REPLIES",
    "CostBasis",
    "per_count",
    "per_noun",
    "per_one",
    "priced_in_part",
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


#: How many of the calls an arm's cost counts were priced, said beside an amount priced over only part of them: a
#: smaller total, which no page states bare.
PRICED_SHARE_TEMPLATE = "{priced} of {calls} calls priced"

#: The words beside a run's total where only part of its trials' spend was priced, which counts trials, not calls.
PRICED_TRIALS_TEMPLATE = "{priced} of {trials} trials priced"


def priced_in_part(report_input: ReportInput, arm: str, aspect: str | None = None) -> str | None:
    """Whether an arm's cost counts only part of its calls, as its pricing says (``ScoreboardRow.pricing``): the words
    every surface states beside such an amount (``850 of 897 calls priced``); None where all were priced, or the input
    does not say.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param arm: The arm, by id.
    :ptype arm: str
    :param aspect: The aspect whose scoreboard says (:func:`~threetears.evals.analysis.report_margins.aspect_board`);
        None for the lead aspect's, the input's own.
    :ptype aspect: str | None
    :return: The words, or None.
    :rtype: str | None
    """
    row = aspect_board(report_input, aspect).get(arm)
    pricing = row.pricing if row is not None else None
    if (
        pricing is None
        or pricing.calls is None
        or pricing.priced_calls is None
        or pricing.priced_calls >= pricing.calls
    ):
        return None
    return PRICED_SHARE_TEMPLATE.format(priced=pricing.priced_calls, calls=pricing.calls)


#: The most of what money counts a ruler states it per: a set under a cent even per 1,000,000 reads there, to the
#: decimals its least amount needs.
_MOST_COUNTED = 1_000_000


@dataclass(frozen=True)
class MoneyRuler:
    """How a set of money amounts is stated (#993), from :func:`money_ruler`, the one owner of money's scale: the form
    every amount is spelled in, what an amount is multiplied by to be drawn, and how many of what it counts each drawn
    amount is per.

    :ivar form: The drawn amounts' one form (:func:`~threetears.evals.kernel.numbers.money_form`).
    :ivar scale: What an amount as stored is multiplied by to be drawn (1000 for money per reply drawn per 1,000
        replies; 1 for a total, which is never scaled).
    :ivar counted: How many of what the money counts each drawn amount is per; 0 for a total (a run's cost, a month's),
        which counts nothing.
    """

    form: MoneyForm
    scale: float
    counted: int

    def drawn(self, value: float) -> float:
        """An amount as stored, as drawn."""
        return value * self.scale

    def spell(self, value: float, *, signed: bool = False) -> str:
        """An amount as stored, spelled as drawn, in the set's form."""
        return self.form.spell(value * self.scale, signed=signed)

    def words(self, noun: str) -> str:
        """The words a drawn amount is per (``per 1,000,000 calls``); empty for a total."""
        return per_count(noun, self.counted) if self.counted else ""

    def label(self, label: str, noun: str, own: int) -> str:
        """A label naming the money (a measure's name, a column's head) with the drawn count in it: the count the
        label states for its own unit (``own``: 1,000 in ``Cost per 1k calls``) replaced by the drawn one (``Cost per
        1,000,000 calls``), or, where the label states none (money per reply, whose label names no count), the drawn
        count's words after it (``Cost per reply, per 1,000 replies``). Never two counts: unchanged where the drawn
        count is the label's own, and for a total.

        :param label: The label as the input or the page writes it.
        :ptype label: str
        :param noun: What the money counts, as a plural noun.
        :ptype noun: str
        :param own: The count the money's own unit is per (1 for money per reply, 1,000 for a price per 1,000).
        :ptype own: int
        :return: The label with the drawn count.
        :rtype: str
        """
        if not self.counted or self.counted == own:
            return label
        if own == 1 and noun:
            # Money per one of what it counts states its count as ``per reply``: the drawn count replaces it.
            single = re.search(rf"(?<!\w){re.escape(per_one(noun))}(?!\w)", label, re.I)
            if single is not None:
                return label[: single.start()] + self.words(noun) + label[single.end() :]
        spellings = _COUNT_SPELLINGS.get(own)
        if spellings is not None:
            found = (
                re.search(rf"(?<!\w)(?:per\s+)?(?:{spellings})\s+{re.escape(noun)}(?!\w)", label, re.I)
                if noun
                else None
            )
            if found is not None:
                return label[: found.start()] + self.words(noun) + label[found.end() :]
        return f"{label}, {self.words(noun)}"

    def for_changes(self, changes: Iterable[float | None], widths: Iterable[float | None] = ()) -> MoneyRuler:
        """The ruler changes between the set's amounts are stated on: the set's own scale and count, so a change
        chart scales as the amounts it compares do, in the changes' form (their own precision, capped by their
        widest interval: :meth:`~threetears.evals.kernel.numbers.MoneyForm.for_changes`).

        :param changes: The changes, as stored.
        :ptype changes: Iterable[float | None]
        :param widths: The widths of their intervals, as stored.
        :ptype widths: Iterable[float | None]
        :return: The changes' ruler.
        :rtype: MoneyRuler
        """
        form = self.form.for_changes(
            (change * self.scale if change is not None else None for change in changes),
            (width * self.scale if width is not None else None for width in widths),
        )
        return MoneyRuler(form=form, scale=self.scale, counted=self.counted)


def money_ruler(amounts: Iterable[float | None], *, per: int, start: int = 0) -> MoneyRuler:
    """The one owner of how a set of money amounts is stated (:class:`MoneyRuler`): every chart, table and figure key
    reads its scale, its form and its count from here, so no surface chooses its own.

    An amount that counts something (money per reply, per 1,000 or per 1,000,000 of what it counts) is drawn per
    ``start`` of it, and, where some amount is under a cent there, per a thousand times more, up to 1,000,000: the
    ruler moves, and the per-1,000 mark is never stacked on it. A set still under a cent per 1,000,000 moves no
    further: it reads at that count, in dollars to the decimals its least amount needs for two significant figures
    (``$0.0030 per 1,000,000 replies``), never with the mark beside the count. A total (``per`` 0: a run's cost, a
    month's) counts nothing and is never scaled: a set of totals under a cent reads per 1,000 with its mark
    (``$8.00/1000``) on every surface, a chart's ticks included (the kit's ``usdpk``), as a key and a table write it.

    :param amounts: Every amount the set holds, as stored.
    :ptype amounts: Iterable[float | None]
    :param per: How many of what it counts one stored amount is per: 1 for money per reply, 1,000 or 1,000,000 for a
        price per that many; 0 for a total.
    :ptype per: int
    :param start: The least count to state it per (a chart states money per reply per 1,000 replies); ``per`` where
        unset.
    :ptype start: int
    :return: The ruler.
    :rtype: MoneyRuler
    """
    held = [amount for amount in amounts if amount is not None]
    if per == 0:
        return MoneyRuler(form=money_form(held), scale=1.0, counted=0)
    count = max(start, per)
    while money_form(amount * count / per for amount in held).per_thousand and count < _MOST_COUNTED:
        count *= 1000
    # A counted set states its count, so it never carries the per-1,000 mark beside it, at the cap included.
    form = money_form((amount * count / per for amount in held), marked=False)
    return MoneyRuler(form=form, scale=count / per, counted=count)


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
