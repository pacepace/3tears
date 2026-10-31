"""The report editor's rubric as the shipped skill states it: the one source of its items' names and definitions.

``editor.md``'s ``## The rubric`` section numbers seven items, each a bold name and its definition on one line, and
says in one sentence what a 2, a 1 and a 0 mean. Two readers take the rubric from that text, read here and nowhere
else, so neither carries a copy that could drift from it:

- the editor's answer is checked against its item names (:mod:`~threetears.evals.ops.report_writer.editor`);
- a report writer template's scoring judge scores the same items, each a rubric dimension
  (:func:`~threetears.evals.ops.report_writer.writer_contract.writer_scoring_rubric`).

**Always the shipped skill's text.** An arm under test may run an ``editor.md`` of its own; its editor reads that text,
and nothing here does. What the scoring judge asks is copied from the shipped text into the writer template when the
template is created, so an arm's edit never moves it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cache

from threetears.evals.analysis.analyst.skill import load_skill

#: The names with a caller outside this module: the editor's answer check and the writer template's scoring rubric
#: read the rubric through :func:`shipped_rubric`; neither names its types.
__all__ = ["shipped_rubric"]

#: The rubric's section, from its heading to the next heading of its level.
_SECTION = re.compile(r"^## The rubric\n(?P<body>.*?)(?=^## )", re.DOTALL | re.MULTILINE)

#: One item: its number, its bold name ending in a full stop, and its definition, on one line.
_ITEM = re.compile(r"^(?P<n>\d+)\. \*\*(?P<name>[^*]+?)\.\*\* (?P<definition>\S.*)$", re.MULTILINE)

#: What each score means, in the sentence that says it.
_SCALE = re.compile(r"A 2 means (?P<top>[^;]+); a 1 means (?P<middle>[^;]+); a 0 means (?P<bottom>[^.]+)\.")


@dataclass(frozen=True, slots=True)
class RubricItem:
    """One rubric item, as ``editor.md`` states it.

    :ivar n: Its number, from 1, in the rubric's order.
    :ivar name: Its name, the bold words without their full stop.
    :ivar definition: What it asks, the rest of its line.
    """

    n: int
    name: str
    definition: str


@dataclass(frozen=True, slots=True)
class EditorRubric:
    """The rubric's items, in order, and what each of the editor's three scores means.

    :ivar items: The items, numbered from 1 with no gap.
    :ivar top: What a 2 means.
    :ivar middle: What a 1 means.
    :ivar bottom: What a 0 means.
    """

    items: tuple[RubricItem, ...]
    top: str
    middle: str
    bottom: str


def _read(text: str) -> EditorRubric:
    """The rubric ``text`` states, refusing one whose section, items or scale sentence do not read."""
    section = _SECTION.search(text)
    if section is None:
        raise ValueError("editor.md has no '## The rubric' section followed by another section")
    body = section["body"]
    items = tuple(
        RubricItem(n=int(match["n"]), name=match["name"].strip(), definition=match["definition"].strip())
        for match in _ITEM.finditer(body)
    )
    if not items or [item.n for item in items] != list(range(1, len(items) + 1)):
        raise ValueError(
            f"editor.md's rubric numbers its items {[item.n for item in items]}, not 1 upward with no gap: each item "
            "is one line, '<n>. **<Name>.** <definition>'"
        )
    scale = _SCALE.search(body)
    if scale is None:
        raise ValueError("editor.md's rubric does not say what a 2, a 1 and a 0 mean in one sentence")
    return EditorRubric(
        items=items, top=scale["top"].strip(), middle=scale["middle"].strip(), bottom=scale["bottom"].strip()
    )


@cache
def shipped_rubric() -> EditorRubric:
    """The rubric of the skill this build ships, read once from its ``editor.md``.

    :return: The rubric.
    :rtype: EditorRubric
    :raises ValueError: The shipped ``editor.md`` states no rubric that reads, which a build must not ship.
    """
    return _read(load_skill().editor().text)
