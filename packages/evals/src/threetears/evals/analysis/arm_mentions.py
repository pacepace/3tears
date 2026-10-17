"""Which arms a finding's title names — read off the names the bundle gives each arm, never off a list of aliases.

**Why it exists.** A production memo (2026-10) titled a finding "GPT-6 Luna Pro and Ling 3.1 Flash quadrupled
evaluation cost" while the finding's readings were taken at two other arms only, so the reader was shown no reading
of Ling's cost at all. Rule 9 of the writer's prompt already said a title claims no more than its evidence; nothing
checked it. A title naming an arm whose readings the finding does not cite states a result the reader cannot see.

**Structural, with one question read off the words**, the way :mod:`threetears.evals.analysis.null_claims` is. Which
arms a finding cites is structure: the cells of its evidence rows and the arms of its trials. Only "which arms does
this title name" is read off the words, and only by the names the bundle itself gives each arm
(:func:`arm_vocabulary`): the display of each level that tells it apart (:func:`~threetears.evals.analysis.arms.distinguishing_axes`),
and for a provider-qualified level (``acme/ling-3.1-flash``) the part after its last ``/`` too, since that is the
model's own name. Nothing invents an alias: a writer shortening ``gpt-6-luna-pro`` to "Luna Pro" names nothing here,
and the reporter eval's judge measures the rest.

**How a name matches.** Case-insensitively, as a run of the name's words in order (letters and digits split apart,
so ``gpt4o`` is ``gpt 4 o``), each word lowercased and never case-folded (``straße-2`` keeps its own spelling), joined
in the title by up to two spaces, hyphens or dashes (``U+2010`` to ``U+2014``), dots, slashes, colons or underscores:
``ling-3.1-flash`` matches "Ling 3.1 Flash" and "ling-3.1-flash" alike. A name glued to more name ("luna" inside
"luna-pro", "gpt-5" inside "gpt-5.1", or inside "GPT-5-mini" whichever dash joins it) is not a match, and where two
names match overlapping words the longer wins, so "Luna Pro" names the ``luna-pro`` arm and never also the ``luna``
one.

**Chosen to miss rather than to refuse a sound title.** A false refusal buys a paid repair, and two lose the whole
analysis. So a level names an arm only where it reads as a name: any level of the candidate model (a model id is a
proper noun, and a model level a host labels in words, "current model", means that arm wherever a title writes it),
and on every other lever a level carrying both a letter and a digit (``terse-v2``, ``fanout-2``, ``v2``, ``x4``,
``8k-ctx``), unless it is a quantity: wholly a number followed by letters, its unit. A quantity whose unit is one or
two letters (``4x``, ``8k``, ``100ms``, ``2.5 GB``) is ordinary prose ("a 4x rise in cost", "under 8k tokens") and
names no arm. One whose unit is longer (``1024tok``, the display an interval level with a unit is given, or
``8chunks``) names its arm only written as one token, number and unit unspaced: "1024tok" in a title is the arm's
label, where "8 chunks" is a count. A bare word
(``high``, ``wide``) is ordinary prose as often as it is an arm, and a bare number is a figure as often as a level; a
level longer than :data:`~threetears.evals.analysis.arms.LABEL_LEVEL_CHARS` (a prompt, a description) is
never written in a title. A lever every arm carried at one level names none of them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from threetears.evals.analysis.arms import LABEL_LEVEL_CHARS, applicable_levers, distinguishing_axes
from threetears.evals.kernel.campaign import VariantIndexEntry
from threetears.evals.kernel.host.sweepables import CANDIDATE_MODEL_LEVER

#: A name's words: runs of letters and runs of digits, split apart where one turns into the other.
_WORDS = re.compile(r"[^\W\d_]+|\d+")

#: A hyphen or a dash: ASCII ``-`` and ``U+2010`` (hyphen) to ``U+2014`` (em dash), as a character-class body.
_DASHES = r"\-\u2010-\u2014"

#: What may join two of a name's words in a title: nothing, or up to two spaces, hyphens or dashes, dots, slashes,
#: colons or underscores.
_JOIN = rf"[\s{_DASHES}_./:]{{0,2}}"

#: A letter or digit.
_ALNUM = r"[^\W_]"

#: What glues a name to more name: a hyphen or dash, dot, slash or underscore with a letter or digit beyond it.
_GLUE = rf"[{_DASHES}_./]"

#: A quantity: the whole level a number followed by letters, its unit (``4x``, ``8k``, ``1024tok``, ``2.5 GB``).
_QUANTITY = re.compile(r"\d+(?:\.\d+)?\s?([^\W\d_]+)")

#: The fewest letters a quantity's unit carries for the quantity to name its arm, written unspaced: ``tok`` does,
#: ``k``, ``x`` and ``ms`` do not.
_NAMING_UNIT_LETTERS = 3


@dataclass(frozen=True)
class ArmName:
    """One name the bundle gives one or more arms, and how a title is matched against it.

    :ivar words: The name as the bundle spells it — the first level display it was read from.
    :ivar variants: Every arm carrying a level of that name, by variant key.
    :ivar pattern: The compiled match over a title.
    """

    words: str
    variants: frozenset[str]
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class ArmMention:
    """One arm name a text carries, as written there.

    :ivar written: The words the text used, exactly as it spelled them.
    :ivar name: The name they matched.
    """

    written: str
    name: ArmName


def _reads_as_a_name(axis: str, form: str) -> bool:
    """Whether a level's display is a name a writer would put in a title (see the module docstring)."""
    if len(form) > LABEL_LEVEL_CHARS or not re.search(r"[^\W\d_]", form):
        return False
    if axis == CANDIDATE_MODEL_LEVER:
        return True
    if not re.search(r"\d", form):
        return False
    quantity = _QUANTITY.fullmatch(form)
    return quantity is None or len(quantity.group(1)) >= _NAMING_UNIT_LETTERS


def _is_quantity(axis: str, form: str) -> bool:
    """Whether a level's display is a quantity, whose unit a title must write against its number to name it."""
    return axis != CANDIDATE_MODEL_LEVER and _QUANTITY.fullmatch(form) is not None


def _forms(display: str) -> list[str]:
    """The names one level display offers: itself, and for a provider-qualified one the part after its last ``/``."""
    whole = " ".join(display.split())
    tail = whole.rsplit("/", 1)[-1].strip()
    return [whole] if not tail or tail == whole else [whole, tail]


def _pattern(words: tuple[str, ...], *, unspaced_unit: bool) -> re.Pattern[str]:
    joins = [_JOIN] * (len(words) - 1)
    if unspaced_unit:
        joins[-1] = ""
    body = re.escape(words[0]) + "".join(join + re.escape(word) for join, word in zip(joins, words[1:], strict=True))
    return re.compile(rf"(?<!{_ALNUM})(?<!{_ALNUM}{_GLUE}){body}(?!{_ALNUM})(?!{_GLUE}{_ALNUM})", re.IGNORECASE)


def arm_vocabulary(entries: Sequence[VariantIndexEntry]) -> list[ArmName]:
    """Every name the bundle gives an arm that a title could carry.

    :param entries: The bundle's variant index.
    :ptype entries: Sequence[VariantIndexEntry]
    :return: One :class:`ArmName` per distinct run of words, each with every arm carrying a level of that name, in
        index order of first appearance; empty when no arm carries a level that reads as a name.
    :rtype: list[ArmName]
    """
    distinguishing = distinguishing_axes(entries)
    spelled: dict[tuple[str, ...], str] = {}
    unspaced_unit: dict[tuple[str, ...], bool] = {}
    carriers: dict[tuple[str, ...], set[str]] = {}
    for entry in entries:
        if entry.levels_unavailable is not None:
            continue
        for axis, level in applicable_levers(entry):
            if axis not in distinguishing:
                continue
            for form in _forms(level.display):
                words = tuple(word.lower() for word in _WORDS.findall(form))
                if not words or not _reads_as_a_name(axis, form):
                    continue
                spelled.setdefault(words, form)
                unspaced_unit.setdefault(words, _is_quantity(axis, form))
                carriers.setdefault(words, set()).add(entry.variant_key)
    return [
        ArmName(spelled[words], frozenset(carriers[words]), _pattern(words, unspaced_unit=unspaced_unit[words]))
        for words in spelled
    ]


def arm_mentions(text: str, vocabulary: Sequence[ArmName]) -> list[ArmMention]:
    """The arm names ``text`` carries, in the order they appear, the longer winning where two overlap.

    :param text: A finding's title.
    :ptype text: str
    :param vocabulary: The bundle's :func:`arm_vocabulary`.
    :ptype vocabulary: Sequence[ArmName]
    :return: Each match once, in text order; empty when the text names no arm.
    :rtype: list[ArmMention]
    """
    found = [(match.start(), match.end(), name) for name in vocabulary for match in name.pattern.finditer(text)]
    kept: list[tuple[int, int, ArmName]] = []
    for start, end, name in sorted(found, key=lambda hit: (hit[0] - hit[1], hit[0])):
        if all(end <= other_start or start >= other_end for other_start, other_end, _ in kept):
            kept.append((start, end, name))
    return [ArmMention(text[start:end], name) for start, end, name in sorted(kept, key=lambda hit: hit[0])]


__all__ = ["ArmMention", "ArmName", "arm_mentions", "arm_vocabulary"]
