"""what a read of a source's change record past a saved mark answers. Pure: the reads run in ``threetears.core.keyed``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["MarkedRead", "PageAfter"]


@dataclass(frozen=True)
class MarkedRead:
    """what a read past the marks answered.

    :param rows: the rows past any mark
    :ptype rows: list[dict[str, Any]]
    :param marks: each stamp's mark to save next: the newest read, or the one given when none moved
    :ptype marks: dict[str, Any]
    """

    rows: list[dict[str, Any]]
    marks: dict[str, Any]


@dataclass(frozen=True)
class PageAfter:
    """a page of rows after an id.

    :param rows: the rows, in id order
    :ptype rows: list[dict[str, Any]]
    :param last: the last id read: the next page's ``after``; the one given when no row was read
    :ptype last: Any
    :param more: whether the page was full, so more may follow
    :ptype more: bool
    """

    rows: list[dict[str, Any]]
    last: Any
    more: bool
