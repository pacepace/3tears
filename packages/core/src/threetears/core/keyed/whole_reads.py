"""the whys a component may read (or write, or drop) a table whole, each counted: one registry, not one a module.

A load path reads by key. The few whole reads it may still make are named in advance, each with what it means, and
counted by table and why the moment one is made, so a replay, a soak or a dashboard sees every one. Only the
``expected`` whys (a true cold start, an operator's reset) may happen in steady state; every other is a change some
writer did not record by key, and a replay holds it at zero (:attr:`WholeReads.gated`).

One component declares one registry at module level::

    WHOLE_READS = WholeReads(
        "copy_follower_whole_reads_total",
        subject="tables a copy follower read whole from L3",
        whys={"cold_start": "a first start with no checkpoint", "reset": "an operator's reset of the table", ...},
        expected={"cold_start", "reset"},
    )
    ...
    WHOLE_READS.count("cold_start", table=table.name)

A why the registry does not name is refused where it is counted (a typo would otherwise count nothing a replay
gates on). Counting is cheap and IO-free: a counter increment and one log line, at INFO for an expected why and at
WARNING for every other.

**A stamped table's live rows, whole** (:func:`read_live_rows`): the one spelling of a cold start's fill of an empty
cache from a stamped L3 table (``ROW_STAMPED``), ``SELECT <columns> FROM <table> WHERE NOT gone``, the gone filter in
the database. Unnarrowed, so its caller declares it (``whole_read(kind="cold_start", ...)``), as it declares any other
whole pager of 3tears' (``read_l3_rows``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Final

from threetears.core.keyed._filters import reads_l3
from threetears.interfaces.keyed.keys import stamped_table
from threetears.observe import get_logger
from threetears.observe.metrics import counter

__all__ = ["WholeReads", "read_live_rows"]

log = get_logger(__name__)

#: a TRUSTED column :func:`read_live_rows` writes bare: a plain name
_COLUMN: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

#: a TRUSTED table :func:`read_live_rows` writes bare: a plain name, or ``schema.name``
_TABLE: Final = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?")


async def read_live_rows(conn: Any, table: str, columns: Sequence[str]) -> list[Any]:
    """every live row of a stamped L3 table (``ROW_STAMPED``), in one statement: ``SELECT <columns> FROM <table> WHERE
    NOT gone``, so a row a delete marked gone in place is left out by the database, never in Python.

    A cold start's fill of an empty cache (a small table read whole once at start and again only after a write that
    changed the whole table); its caller declares it so (``whole_read(kind="cold_start", ...)``). One statement, never
    paged: a transport that cuts an answer at its row cap says so (``L3AnswerCutError``), so a short answer is never
    taken for the table. A table no stamped declaration names (or a read that does not go to L3, whose copy keeps no
    gone flag) is refused: its gone rows are not marked by the flag this filters on.

    :param conn: the L3 pool, connection or backend (anything with an asyncpg-shaped ``fetch``)
    :ptype conn: Any
    :param table: the stamped table, a TRUSTED plain identifier (``name`` or ``schema.name``)
    :ptype table: str
    :param columns: the columns read, TRUSTED plain identifiers
    :ptype columns: Sequence[str]
    :return: the rows, as the connection answers them
    :rtype: list[Any]
    :raises ValueError: when ``table`` is not a stamped L3 table, or a name is not a plain identifier
    """
    plain = _TABLE.fullmatch(table) is not None and all(_COLUMN.fullmatch(column) is not None for column in columns)
    if not plain:
        raise ValueError(f"read_live_rows: {table!r} or one of {list(columns)!r} is not a plain identifier")
    if not columns:
        raise ValueError(f"read_live_rows: {table} names no column to read")
    declared = stamped_table(table, l3=reads_l3(conn))
    if declared is None:
        raise ValueError(
            f"read_live_rows: {table} is not a stamped L3 table (ROW_STAMPED), so no gone flag marks its deleted rows"
        )
    rows: list[Any] = list(
        await conn.fetch(f"SELECT {', '.join(columns)} FROM {table} WHERE NOT {declared.gone}")  # noqa: S608 - TRUSTED
    )
    return rows


class WholeReads:
    """a component's named whole reads, counted by why.

    :param metric: the counter's name (``<component>_whole_reads_total``)
    :ptype metric: str
    :param subject: what the counter counts, the start of its description
    :ptype subject: str
    :param whys: why -> what it means; every why the component may count
    :ptype whys: Mapping[str, str]
    :param expected: the whys that may happen in steady state (a cold start, an operator's reset)
    :ptype expected: Collection[str]
    :param labels: the counter's labels beside ``why``
    :ptype labels: tuple[str, ...]
    :raises ValueError: when no why is named, or an expected why is not one of them
    """

    def __init__(
        self,
        metric: str,
        *,
        subject: str,
        whys: Mapping[str, str],
        expected: Collection[str],
        labels: tuple[str, ...] = ("table",),
    ) -> None:
        if not whys:
            raise ValueError(f"{metric}: a whole-read registry names its whys")
        unknown = sorted(set(expected) - set(whys))
        if unknown:
            raise ValueError(f"{metric}: expected whys {unknown} are not among its whys")
        self._metric = metric
        self._subject = subject
        self._whys = dict(whys)
        self._expected = frozenset(expected)
        self._labels = labels

    @property
    def metric(self) -> str:
        """the counter's name.

        :return: the name
        :rtype: str
        """
        return self._metric

    @property
    def whys(self) -> Mapping[str, str]:
        """every why, and what it means.

        :return: why -> meaning
        :rtype: Mapping[str, str]
        """
        return dict(self._whys)

    @property
    def gated(self) -> frozenset[str]:
        """the whys a replay or a soak holds at zero: every one but the expected.

        :return: the whys
        :rtype: frozenset[str]
        """
        return frozenset(self._whys) - self._expected

    def _counter(self) -> Any:
        return counter(
            self._metric,
            description=f"{self._subject}, by {', '.join(self._labels)} and why ("
            + "; ".join(f"{why}: {what}" for why, what in self._whys.items())
            + f"); zero but {', '.join(sorted(self._expected)) or 'none'}",
            label_names=(*self._labels, "why"),
        )

    def count(self, why: str, **labels: str) -> None:
        """count one whole read for ``why``, and say so in the log.

        :param why: one of :attr:`whys`
        :ptype why: str
        :param labels: the counter's other labels (``table=``)
        :ptype labels: str
        :return: nothing
        :rtype: None
        :raises ValueError: when ``why`` is not one of the registry's, or the labels are not the counter's
        """
        if why not in self._whys:
            raise ValueError(f"{self._metric}: {why!r} is not one of its whys {sorted(self._whys)}")
        if set(labels) != set(self._labels):
            raise ValueError(f"{self._metric}: counts by {list(self._labels)}, got {sorted(labels)}")
        self._counter().labels(**labels, why=why).inc()
        log.log(
            logging.INFO if why in self._expected else logging.WARNING,
            "read whole",
            extra={"extra_data": {"metric": self._metric, "why": why, "what": self._whys[why], **labels}},
        )
