"""relational steps' vocabulary: a step (one ``SELECT`` and its binds), a relation already quoted, the column pairs two
relations match on, and the declaration of a change relation. The steps are built and run in
``threetears.core.keyed.steps``."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Final

from threetears.interfaces.keyed.filters import Derived, quote_identifier

__all__ = ["Match", "Quoted", "Rel", "Step", "change_relation"]


@dataclass(frozen=True)
class Step:
    """a relational step: one ``SELECT`` and its parameters, its placeholders numbered from ``$1``. Built by the
    helpers here, never written by hand; read by another step, or run by :func:`read_step` and its siblings.

    :param sql: the ``SELECT``, TRUSTED
    :ptype sql: str
    :param params: its parameters, ``$1`` first
    :ptype params: tuple[Any, ...]
    """

    sql: str
    params: tuple[Any, ...] = ()


#: a relation name as SQL writes it: identifiers, each bare or double-quoted, joined by dots (``"report"."lsf_rows"``)
_QUOTED_NAME: Final = re.compile(
    r'(?:"(?:[^"]|"")+"|[A-Za-z_][A-Za-z0-9_$]*)(?:\.(?:"(?:[^"]|"")+"|[A-Za-z_][A-Za-z0-9_$]*))*'
)


@dataclass(frozen=True)
class Quoted:
    """a relation's name already quoted as SQL names it (a copy's run-time relation, ``snapshot.relation(name)``:
    ``"report"."lsf_rows"``): a step reads it as written, never quoting it again. Only a name: dotted identifiers,
    each bare or double-quoted. The no-whole-reads rule judges a step over it as reading the relation it names.

    :param name: the relation, TRUSTED, quoted
    :ptype name: str
    :raises ValueError: for anything but a quoted relation name
    """

    name: str

    def __post_init__(self) -> None:
        if not _QUOTED_NAME.fullmatch(self.name):
            raise ValueError(f"a quoted relation is a name, dotted identifiers, not {self.name!r}")


#: what a step reads: a change relation's TRUSTED name, one already quoted, a derived select, or another step
Rel = str | Quoted | Derived | Step


@dataclass(frozen=True)
class Match:
    """one column pair two relations are matched on.

    :param column: the column of the step's relation (``t``)
    :ptype column: str
    :param other: the other relation's column (``k``); the same name when None
    :ptype other: str | None
    :param null_safe: whether two NULLs match (``IS NOT DISTINCT FROM``); ``=`` otherwise
    :ptype null_safe: bool
    """

    column: str
    other: str | None = None
    null_safe: bool = False

    def condition(self, left: str, right: str) -> str:
        """the pair's condition between ``left`` and ``right`` (the aliases).

        :param left: the step relation's alias
        :ptype left: str
        :param right: the other relation's alias
        :ptype right: str
        :return: the condition, TRUSTED
        :rtype: str
        """
        mine = f"{left}.{quote_identifier(self.column)}"
        theirs = f"{right}.{quote_identifier(self.other or self.column)}"
        return f"{mine} IS NOT DISTINCT FROM {theirs}" if self.null_safe else f"{mine} = {theirs}"


def change_relation(name: str) -> str:
    """declare ``name`` a change relation: one a change registers (its produced rows, the held rows it touches, a
    partition's chunk), bounded by that change. The no-whole-reads rule reads the steps over it as change-bounded.

    :param name: the relation's TRUSTED name, as registered
    :ptype name: str
    :return: the name
    :rtype: str
    :raises ValueError: when the name is empty
    """
    if not name.strip():
        raise ValueError("a change relation is named")
    return name
