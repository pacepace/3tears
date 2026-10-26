"""A key check: what each figure key a report cites reads on the snapshot, or why it reads nothing.

Design note. An analyst drafting prose, and the editor testing it, need to know before submitting that every key a
report cites resolves, and what it prints. The W2 prototype found no tool that said so: six analysts and three
editors each improvised one (a throwaway block whose title held the keys, a script), and one analyst spent most of
its calls that way. :func:`figure_keys_check` is that tool.

**Its input** is a list of keys as a report writes them: a bare path (``reading.fast.v2.change``), a path with its
filters (``reading.fast.v2.change|abs|pts``), either in braces (``{cost_per_1k.base|usd}``), or an arm
(``arm:fast.v2``). Each is checked once, in the order given; a key repeated is checked at its first place. At most
:data:`KEYS_MOST` keys a call, and the rows are paged.

**Each key's row** is one of three:

- ``reads``: the figure, written through its filters as the page writes it (a
  :class:`~threetears.evals.analysis.analyst.cited.WrittenFigure`: its key, value, kind, scale, unit and ``reads``),
  the family it reads by (its form, what the family reads, the help page listing it), the arms and aspects it is
  about, and the filters the figure takes;
- ``unstated``: a well-formed key whose figure the snapshot does not state (an interval the pipeline did not compute,
  a monthly figure with no declared volume). The path's form is still given, and the remedy is to say in words that
  the snapshot does not state it, or cite a figure it holds;
- ``refused``: a key that names nothing the snapshot holds, reads two ways, carries a filter the grammar does not
  define, or a filter its figure does not suit. The refusal is the resolver's own words, which name what could come
  next where the path stopped matching; the remedy names the filters the figure takes, or the grammar's help page.

**What it is not.** It computes nothing: every figure is :func:`~threetears.evals.analysis.analyst.cited.cite_written`
over the key resolver every report and page read (:mod:`~threetears.evals.analysis.report.figure_keys`), so a key
that reads here reads the same on the page. It reads one snapshot and stores nothing, so the same keys always return
the same rows. It never reads what kind of eval a campaign is.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.cited import WrittenFigure, cite_written
from threetears.evals.analysis.errors import UnresolvableReference, UnstatedFigure
from threetears.evals.analysis.report.figure_keys import (
    FILTER_SPELLINGS,
    figure_filters,
    key_form,
    key_subjects,
    parse_key,
    resolve,
)
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.kernel.errors import RefusedInput

__all__ = ["KEYS_LIMIT", "KEYS_MOST", "KeyCheck", "KeysChecked", "figure_keys_check"]

#: The most keys one call checks: a report's whole prose is checked a few calls at a time.
KEYS_MOST = 120

#: The most rows a page holds unless the caller asks for fewer: each is a figure and its family, a few lines.
KEYS_LIMIT = 40

#: What every result says it is.
_READS_AS = (
    "each key as a report writes it, read on the snapshot: what it prints through its filters, the family it reads "
    "by, or why it reads nothing and what to write instead"
)

#: The remedy for a key whose figure the snapshot leaves unstated.
_UNSTATED_REMEDY = (
    "the snapshot does not state this figure: say so in words, or cite a figure it holds (help "
    "topic='figure_keys.{group}' lists this form's statistics)"
)

#: The remedy for a key whose filters the grammar does not read.
_PARSE_REMEDY = (
    "write the path, then any filters each after a bar: one spelling at most ({spellings}), plus abs and interval"
)

#: The remedy for a path the grammar cannot read.
_PATH_REMEDY = (
    "write the path as the refusal says what may come next, ids exactly as campaign_overview spells them; help "
    "topic='figure_keys' lists every form"
)

#: The remedy for a filter its figure does not suit.
_FILTER_REMEDY = "write the key with a filter its figure takes: {taken}"

#: The remedy for a figure that takes no filter.
_NO_FILTER_REMEDY = "write the key with no filter: its figure is {kind}, which takes none"

#: How a key check reads one key.
KeyStatus = Literal["reads", "unstated", "refused"]

_MODEL = ConfigDict(frozen=True, extra="forbid")


class KeyCheck(BaseModel):
    """One key as a report writes it, read on the snapshot."""

    model_config = _MODEL

    asked: str = Field(description="The key as it was given.")
    status: KeyStatus = Field(
        description="reads, unstated (well formed, the snapshot states no such figure) or refused."
    )
    figure: WrittenFigure | None = Field(description="What it reads, written through its filters; None unless reads.")
    form: str | None = Field(description="The form of the family it reads by; None where the path matches none.")
    what: str | None = Field(description="What that family reads, as the grammar says it.")
    arms: list[str] = Field(description="The arms the key is about, as arm:<id>.")
    aspects: list[str] = Field(description="The aspects it is about, by id.")
    filters_taken: list[str] = Field(description="Every filter its figure takes; empty for text, an arm, or none read.")
    refusal: str | None = Field(description="Why it reads nothing; None where it reads.")
    remedy: str | None = Field(description="What to write instead; None where it reads.")


class KeysChecked(BaseModel):
    """The keys asked about, each read on the snapshot or refused with what to write instead."""

    model_config = _MODEL

    reads_as: str = Field(description="What this is.")
    keys: list[KeyCheck] = Field(description="Each key on this page, in the order asked.")
    total: int = Field(description="The distinct keys asked about.")
    offset: int = Field(description="The first key this page holds.")
    next_offset: int | None = Field(description="The offset of the next page; None when this is the last.")


def _unbraced(written: str) -> str:
    """A key as written, its one pair of braces taken off where it carries them."""
    text = written.strip()
    if text.startswith("{") and text.endswith("}") and text.count("{") == 1 and text.count("}") == 1:
        return text[1:-1].strip()
    return text


def _refused(asked: str, why: str, remedy: str, form: tuple[str, str] | None = None) -> KeyCheck:
    return KeyCheck(
        asked=asked,
        status="refused",
        figure=None,
        form=form[0] if form else None,
        what=form[1] if form else None,
        arms=[],
        aspects=[],
        filters_taken=[],
        refusal=why,
        remedy=remedy,
    )


def _check(asked: str, report_input: ReportInput) -> KeyCheck:
    """One key, read or refused."""
    try:
        key = parse_key(_unbraced(asked))
    except UnresolvableReference as refused:
        return _refused(asked, str(refused), _PARSE_REMEDY.format(spellings=", ".join(FILTER_SPELLINGS)))
    try:
        form = key_form(key.path, report_input)
    except UnresolvableReference as refused:
        return _refused(asked, str(refused), _PATH_REMEDY)
    try:
        figure = resolve(key.path, report_input)
    except UnstatedFigure as unstated:
        return KeyCheck(
            asked=asked,
            status="unstated",
            figure=None,
            form=form.form,
            what=form.what,
            arms=[],
            aspects=[],
            filters_taken=[],
            refusal=str(unstated),
            remedy=_UNSTATED_REMEDY.format(group=form.group),
        )
    except UnresolvableReference as refused:
        return _refused(asked, str(refused), _PATH_REMEDY, (form.form, form.what))
    taken = list(figure_filters(figure))
    try:
        written = cite_written(key.path, key.filters, report_input)
    except UnresolvableReference as refused:
        remedy = _FILTER_REMEDY.format(taken=", ".join(taken)) if taken else _NO_FILTER_REMEDY.format(kind=figure.kind)
        return _refused(asked, str(refused), remedy, (form.form, form.what))
    subjects = key_subjects(key.path, report_input)
    return KeyCheck(
        asked=asked,
        status="reads",
        figure=written,
        form=form.form,
        what=form.what,
        arms=[f"arm:{arm}" for arm in subjects.arms],
        aspects=list(subjects.aspects),
        filters_taken=taken,
        refusal=None,
        remedy=None,
    )


def figure_keys_check(
    report_input: ReportInput, keys: Sequence[str], *, limit: int = KEYS_LIMIT, offset: int = 0
) -> KeysChecked:
    """What each key reads on the snapshot, or why it reads nothing and what to write instead.

    :param report_input: The snapshot's frozen report input.
    :ptype report_input: ReportInput
    :param keys: The keys, as a report writes them: a path, with any filters, in braces or not, or ``arm:<id>``.
    :ptype keys: Sequence[str]
    :param limit: The most keys one page holds.
    :ptype limit: int
    :param offset: The first key the page holds.
    :ptype offset: int
    :return: Each key on the page, read or refused.
    :rtype: KeysChecked
    :raises RefusedInput: No key is given, more than :data:`KEYS_MOST`, or one is empty.
    """
    distinct = list(dict.fromkeys(key.strip() for key in keys))
    if not distinct or any(not key for key in distinct):
        raise RefusedInput(
            "keys names no key to check, or an empty one",
            remedy="name each key as a report writes it, such as {cost_per_1k.<arm>|usd} or arm:<id>",
            parameters=("keys",),
        )
    if len(distinct) > KEYS_MOST:
        raise RefusedInput(
            f"keys names {len(distinct)} distinct keys, more than one call checks",
            remedy=f"check them in calls of {KEYS_MOST} keys or fewer",
            parameters=("keys",),
        )
    window = distinct[offset : offset + limit]
    end = offset + len(window)
    return KeysChecked(
        reads_as=_READS_AS,
        keys=[_check(key, report_input) for key in window],
        total=len(distinct),
        offset=offset,
        next_offset=end if end < len(distinct) else None,
    )
