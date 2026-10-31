"""The mechanical checks over report prose that a report plan and a report document share: typed digits, a unit a
key already prints, and internal ids a reader should not see.

Each check reads prose as ``(field path, text)`` pairs, so it holds for whatever names its fields:
:mod:`~threetears.evals.analysis.report_plan_checks` walks a plan's ``ModelProse`` fields and option labels,
:mod:`~threetears.evals.analysis.report_doc_checks` walks a document's prose through
:func:`~threetears.evals.kernel.report_doc.report_doc_prose`. A fault is ``(field path, why)``.

- :func:`literal_faults`: prose typing a number itself rather than through a figure key (GENERATOR §6.2's D3 rule,
  :func:`~threetears.evals.analysis.report.figure_keys.literal_numbers`), with an optional pattern of further phrases
  a caller allows.
- :func:`unit_faults`: prose writing a unit right after a key that prints it, and money per reply under one cent keyed
  per reply on a page that states its costs per 1,000.
- :func:`internal_ids` and :func:`typed_ids`: the ids that read as ids and differ from their reader names (an arm's
  id where it differs from the name the page prints for the arm, a hyphen or a digit marking it too), and where a text
  types one outside its keys and outside every name a reader is shown (:func:`reads_as_id` is the rule).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence

from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.quantities import MONEY, UNIT_NAMES
from threetears.evals.analysis.report.cost_unit import per_count
from threetears.evals.analysis.report.figure_keys import (
    INTERVAL_TEMPLATE,
    PERCENTILE_NAMES,
    Figure,
    FigureKey,
    cost_basis,
    format_figure,
    keys_in,
    literal_numbers,
    printed_arm_name,
    reader_names,
    resolve,
)
from threetears.evals.analysis.report_input import ReportInput

__all__ = [
    "internal_ids",
    "literal_faults",
    "reads_as_id",
    "typed_ids",
    "unit_faults",
]

#: One fault: the field path it sits at and why it is refused.
_Fault = tuple[str, str]


# --- typed digits --------------------------------------------------------------------------------------------------


def literal_faults(
    texts: Iterable[tuple[str, str]], report_input: ReportInput, *, allowed: re.Pattern[str] | None = None
) -> Iterator[_Fault]:
    """Each text that types a number outside its figure keys and the names the input holds.

    :param texts: The prose, as ``(field path, text)``.
    :ptype texts: Iterable[tuple[str, str]]
    :param report_input: The report input the prose was written against.
    :ptype report_input: ReportInput
    :param allowed: Further phrases the caller lets prose write with their digits; a typed number wholly inside a match
        is set aside. None for none.
    :ptype allowed: re.Pattern[str] | None
    :return: One fault per text that types a number, naming every number it types.
    :rtype: Iterator[tuple[str, str]]
    """
    for path, text in texts:
        spans = literal_numbers(text, report_input)
        if allowed is not None:
            phrases = [match.span() for match in allowed.finditer(text)]
            spans = [span for span in spans if not any(lo <= span[0] and span[1] <= hi for lo, hi in phrases)]
        if typed := [text[start:end] for start, end in spans]:
            yield (
                path,
                (
                    f"types the number(s) {typed}: every figure in prose is a figure key, and a level, a count or a unit "
                    "the input does not hold is written in words or left out"
                ),
            )


# --- units a key already prints ------------------------------------------------------------------------------------

#: Money under this, per reply, reads per 1,000 replies on a page that states its costs so (one cost basis per page).
_CENT = 0.01
#: The filter that reads money per 1,000 replies, and the filters a key keeps beside it (it is a spelling, so it
#: replaces any other).
_PER_THOUSAND_FILTER = "usd_per_1k"
_KEPT_BESIDE_A_SPELLING = frozenset({"abs", "interval"})
#: The count a cost's unit opens with (``per 1,000``, ``per 1,000,000``), whatever it counts: prose that repeats it
#: after a key that prints it doubles the unit, even where it names another noun.
_PER_COUNT = re.compile(r"^per [\d,]+")
#: What joins an interval's two ends where code spells each with its unit (``1.2 s to 3.4 s``).
_INTERVAL_JOIN = INTERVAL_TEMPLATE.format(low="", high="")
#: A figure's number as code spells it, at the start of its spelling: a bound, a sign, money's sign, the digits, and a
#: range's second end (``< 0.000001``, ``-$0.01``, ``27–28``). What follows it is the unit the figure prints.
_SPELLED_NUMBER = re.compile(r"^[<>]?\s?[+\-−]?\$?\d[\d.,]*(?:–[+\-−]?\$?\d[\d.,]*)?")


def _states_cost_per_thousand(report_input: ReportInput) -> bool:
    """Whether the page states the arms' costs per 1,000: the ``cost_per_1k`` figure resolves for some arm."""
    for arm in report_input.arms:
        try:
            resolve(f"cost_per_1k.{arm.id}", report_input)
        except UnresolvableReference:
            continue  # NOSILENT: an arm that states no cost states none per 1,000; the others are read
        return True
    return False


def _unit_words(spelled: str) -> list[str]:
    """The words a figure's spelling already prints its unit by, longest first: the unit after its number (``ms``,
    ``%``, ``per 1,000 replies``), and every name :data:`~threetears.evals.analysis.quantities.UNIT_NAMES` gives that
    unit, money's whenever the spelling carries its sign."""
    last = spelled.rpartition(_INTERVAL_JOIN)[2]
    number = _SPELLED_NUMBER.match(last)
    suffix = last[number.end() :].strip() if number is not None else ""
    words = {suffix} if suffix else set()
    if (count := _PER_COUNT.match(suffix)) is not None:
        words.add(count.group(0))
    for unit, names in UNIT_NAMES.items():
        if suffix.casefold() in {name.casefold() for name in names} or (unit == MONEY and "$" in spelled):
            words.update(names)
    return sorted(words, key=len, reverse=True)


def _repeated_unit(after: str, words: Sequence[str]) -> str | None:
    """The unit word ``after`` opens with (past any space), bounded where it ends in a word character; None for none."""
    for word in words:
        bound = r"(?!\w)" if re.match(r"\w", word[-1:]) else ""
        if re.match(rf"\s*{re.escape(word)}{bound}", after, re.IGNORECASE):
            return word
    return None


def _sub_cent(figure: Figure, key: FigureKey) -> bool:
    """A money figure per reply under one cent, not keyed per 1,000 replies."""
    if figure.scale != MONEY or _PER_THOUSAND_FILTER in key.filters:
        return False
    values = figure.value if isinstance(figure.value, tuple) else (figure.value,)
    sizes = [abs(value) for value in values if isinstance(value, float)]
    return bool(sizes) and 0 < max(sizes) < _CENT


def unit_faults(texts: Iterable[tuple[str, str]], report_input: ReportInput) -> Iterator[_Fault]:
    """No prose writes a unit right after a key that prints it, and money per reply under one cent is keyed per
    1,000 where the page states its costs so.

    A key that does not parse, resolve or suit its filters is passed over here: the key check refuses it at its path.

    :param texts: The prose, as ``(field path, text)``.
    :ptype texts: Iterable[tuple[str, str]]
    :param report_input: The report input the prose was written against.
    :ptype report_input: ReportInput
    :return: Each fault, in text and then key order.
    :rtype: Iterator[tuple[str, str]]
    """
    per_thousand = _states_cost_per_thousand(report_input)
    basis = per_count(cost_basis(report_input).noun)
    for path, text in texts:
        try:
            keys = keys_in(text)
        except UnresolvableReference:
            continue  # NOSILENT: prose whose keys do not parse is refused by the caller's key check at this path
        for key in keys:
            assert key.span is not None
            start, end = key.span
            try:
                figure = resolve(key.path, report_input)
                if figure.kind not in ("number", "interval"):
                    continue
                spelled = format_figure(figure, key.filters)
            except UnresolvableReference:
                continue  # NOSILENT: a key that does not resolve or suit its filters is refused by the key check
            written = text[start:end]
            if (word := _repeated_unit(text[end:], _unit_words(spelled))) is not None:
                yield (
                    path,
                    (
                        f"writes {word!r} right after {written}, which code spells {spelled!r}, its unit included: "
                        "a key prints its own unit, so drop the word"
                    ),
                )
            if per_thousand and _sub_cent(figure, key):
                kept = [name for name in key.filters if name in _KEPT_BESIDE_A_SPELLING]
                wanted = "|".join([key.path, *kept, _PER_THOUSAND_FILTER])
                yield (
                    path,
                    (
                        f"keys {written}, money per reply under one cent ({spelled}), and the page states its costs "
                        f"{basis}: key it {basis}, {{{wanted}}}, so the page reads one cost basis"
                    ),
                )


# --- internal ids --------------------------------------------------------------------------------------------------

#: A character no reader name carries and an id often does: an id holding one reads as an id.
_ID_MARKS = re.compile(r"[_.:/=()\[\]\"]")


def reads_as_id(identifier: str) -> bool:
    """Whether an id reads as an id to a reader: it holds a character no reader name carries (an underscore, a dot, a
    colon, a slash, an equals sign, a bracket or a quote). A plain word (``billing``, ``overall``) reads as a word.

    :param identifier: The id.
    :ptype identifier: str
    :return: Whether it holds one of those characters.
    :rtype: bool
    """
    return _ID_MARKS.search(identifier) is not None


#: A character that marks an arm's id as an id where the page prints the arm by another name: every id mark, and a
#: hyphen or a digit. Prose joins words with a hyphen ("cost-per-reply"), so a hyphen alone marks nothing in an
#: aspect's or a measure's id; an arm's id is the campaign's label for a run (``fanout-1``), and where the page prints
#: the arm otherwise (``fanout 1``), that label is never a reader's word.
_ARM_ID_MARKS = re.compile(r"[-\d_.:/=()\[\]\"]")


def _internal_arm_ids(report_input: ReportInput) -> dict[str, str]:
    """Each arm's id a reader is never shown: one that differs, case-folded, from both the name the page prints for
    its arm (:func:`~threetears.evals.analysis.report.figure_keys.printed_arm_name`) and its served model, and holds
    an id mark, a hyphen or a digit. A model arm, whose id is its served model's name, is the reader's word; a lever
    arm's id is not. The alias exempts nothing beyond the printed name it falls back to: an input built from a run
    sets every arm's alias to its id."""
    found: dict[str, str] = {}
    for arm in report_input.arms:
        printed = printed_arm_name(arm)
        names = {name.casefold() for name in (printed, arm.served_model) if name}
        if arm.id.casefold() not in names and _ARM_ID_MARKS.search(arm.id):
            found[arm.id] = printed
    return found


def internal_ids(report_input: ReportInput) -> dict[str, tuple[str, str]]:
    """Every id a reader is never shown: id, then its kind and the name a reader is shown for it.

    The named kinds are those :func:`~threetears.evals.analysis.report.figure_keys.reader_names` names: aspects,
    measures, judged dimensions, strata, tasks and criteria, each id that reads as an id (:func:`reads_as_id`) and
    differs from its reader name. A percentile's name (``p95``) is the reader's own. Then the arms: each arm's id that
    differs from the name the page prints for it and from its served model, and holds an id mark, a hyphen or a digit
    (``fanout-1`` where the page prints "fanout 1"), as kind ``arm``.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Each such id, to ``(kind, name)``, the first kind that names it.
    :rtype: dict[str, tuple[str, str]]
    """
    found: dict[str, tuple[str, str]] = {}
    for kind, names in reader_names(report_input).items():
        for identifier, name in names.items():
            if identifier.lower() in PERCENTILE_NAMES or identifier.casefold() == name.casefold():
                continue
            if reads_as_id(identifier):
                found.setdefault(identifier, (kind, name))
    for identifier, printed in _internal_arm_ids(report_input).items():
        found.setdefault(identifier, ("arm", printed))
    return found


#: A key as prose holds it, set aside before ids are looked for.
_KEY_SPAN = re.compile(r"\{[^{}]*\}")


def _bounded(name: str, *, exact: bool = False) -> re.Pattern[str]:
    """``name`` where a non-word character or the text's edge bounds it on both sides, in any case unless ``exact``:
    D3's rule."""
    return re.compile(rf"(?<!\w){re.escape(name)}(?!\w)", 0 if exact else re.IGNORECASE)


def typed_ids(text: str, ids: Mapping[str, object], shown: Iterable[str], *, exact: Collection[str] = ()) -> list[str]:
    """The ids ``text`` types outside its keys and outside any name a reader is shown, each once, in order found.

    Names and ids are matched in one pass, longest first, each masked where it matched, so a shown name that is a
    bounded prefix of an id (``Tone`` in ``tone.kept``) never hides the id, and an id inside a longer shown name is
    that name.

    :param text: The prose.
    :ptype text: str
    :param ids: The ids to look for, as the keys of a mapping (what each maps to is the caller's).
    :ptype ids: Mapping[str, object]
    :param shown: The names a reader is shown, which hide an id inside them.
    :ptype shown: Iterable[str]
    :param exact: The ids matched in their own case only, where a word of another case is not the id (a case id
        ``q3`` against "Q3"); every other name is matched in any case.
    :ptype exact: Collection[str]
    :return: The ids typed, in the order they first appear.
    :rtype: list[str]
    """
    masked = list(_KEY_SPAN.sub(lambda match: "_" * len(match.group()), text))
    candidates = {name: False for name in shown} | dict.fromkeys(ids, True)
    found: dict[str, int] = {}
    for name in sorted(candidates, key=len, reverse=True):
        for match in _bounded(name, exact=name in exact).finditer("".join(masked)):
            masked[match.start() : match.end()] = "_" * (match.end() - match.start())
            if candidates[name]:
                found.setdefault(name, match.start())
    return sorted(found, key=found.__getitem__)
