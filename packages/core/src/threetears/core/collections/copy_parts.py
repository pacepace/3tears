"""The parts a copied table is held in: one per value of its declared scope column once it is big, else one.

**Why parts.** A copy publishes each version of a table as files, and a change rewrites the files its rows land in
and links every other from the version before. Cut by the scope its table already declares (``delta_state_scope``,
the copy's :attr:`~threetears.core.collections.copy_follower.FollowedTable.scope`: the state, for reports), a change
to one state rewrites that state's rows, not the table's.

**Why a size floor** (``design-copy-worker-backend.md``, decision 2). Every part is one more Arrow chunk a reader's
DuckDB scans, at a fixed cost a chunk, so a small table cut in 51 reads slower than held whole, and saves its apply
little: below :data:`SPLIT_FLOOR_BYTES` a table is one part (:data:`WHOLE_PART`). A table with no scope column is
one part at every size.

**A key without the scope column.** A table keyed without its scope column (a result row keyed by its race) names,
for a deleted key, no row that says its part. Such a table declares a locator: a key column naming one scope value
(the race names one state). :class:`Locator` holds each locator value's part, from the rows the copy held, and
:func:`parts_of_keys` finds each key's part through it, never by a scan of the parts.

**Arrow end to end.** Rows are split by one sort and one run-end encoding of the scope column, each part a slice of
the sorted rows; a locator is a two-column Arrow table, matched by Arrow's hash kernels (``index_in``, ``is_in``),
never Acero (``tests/enforcement/test_no_acero_tables.py``). Python objects here are one a table or one a part, never
one a row or a key.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final

import pyarrow as pa
import pyarrow.compute as pc

from threetears.core.frozen_state import EMPTY, frozen

if TYPE_CHECKING:
    from threetears.core.collections.copy_follower import FollowedTable

__all__ = [
    "SPLIT_FLOOR_BYTES",
    "WHOLE_PART",
    "Locator",
    "Part",
    "PartRule",
    "part_rules",
    "parts_of_keys",
    "split_parts",
]

#: the one part a table below the floor, or with no scope column, is held as
WHOLE_PART: Final = "all"

#: a scoped table holding at least this many bytes (Arrow ``nbytes``) is cut a part per scope value; below it, one part.
#:
#: **Measured** 2026-11-25 (``scripts/measure-copy-part-floor.py``; Apple M4, pyarrow 25.0.1, DuckDB 1.5.6 on one
#: thread, the fastest of 12-60 runs on a shared machine; a second run within 10-30% of these). A 7-column table (two
#: texts and an int of key, two counts, a text, a ``timestamptz``; 76 bytes a row) over 51 scope values, and a 100-row
#: change to one scope, half of it replacing held rows. A rebuild is the apply's: the held rows less those the change
#: names (``arrow_keys.rows_named``), plus its rows, in key order, written as one Arrow IPC file to memory.
#:
#: ===========  ========  ==============  ================  ================
#: rows         MB        whole rebuild   one part of 51    saved a change
#: ===========  ========  ==============  ================  ================
#: 1,000        0.07      0.26 ms         0.35 ms           none
#: 10,000       0.74      1.35 ms         0.20 ms           1.1 ms
#: 20,000       1.5       3.23 ms         0.24 ms           3.0 ms
#: 50,000       3.8       7.59 ms         0.31 ms           7.3 ms
#: 100,000      7.5       18.7 ms         0.46 ms           18.2 ms
#: 200,000      15.1      37.0 ms         0.77 ms           36.3 ms
#: 500,000      38.0      107 ms          3.7 ms            103 ms
#: 1,000,000    76.0      315 ms          5.5 ms            310 ms
#: ===========  ========  ==============  ================  ================
#:
#: A read of the table as 51 parts (51 chunks) against one part, over three queries (one state's group-by, every
#: state's group-by, one race's rows): 0.9-1.9 ms slower a read from 1k to 50k rows, 0.1-1.8 ms at 100k-200k, flat with
#: size (DuckDB's fixed cost a chunk; reports measured 2.8 against 0.7 ms at 16k rows, and Race Detail's 42 statements
#: 20-30% faster whole), and even or faster split from 500k rows (-2.9 to +1.2 ms).
#:
#: **So:** a split saves each change landing in the table its whole rebuild less one part's, which grows with the
#: table, and costs each read of it about 1.3 ms. At 16 MiB a change saves about 40 ms, the cost of some 30 reads.
#: The saving lands on the copy's one serial apply worker, which every change waits on before any page shows it (the
#: hot path on election night); the reads spread over the query workers. So the floor leans to the apply, short of
#: splitting tables whose change saves only a few reads' worth (under 4 MB). 16 MB is also where reports cut by hand
#: until now: every table it held whole was under 16 MB at the end of cobalt dev's 2026-10-30 night, and its long
#: tables were split (``aibots_reports.pod.shared_backend.PER_STATE_TABLES``).
SPLIT_FLOOR_BYTES: Final = 16 << 20

#: a table already split stays split until it holds under the floor divided by this: a table near the floor (a
#: retention letting runs go, the next run adding them) would otherwise flip shape, and each flip rewrites it whole
_STAY_SPLIT_DIVISOR: Final = 2

#: a :class:`Locator`'s columns: a locator value and its part, both as text
_VALUE: Final = "value"
_PART: Final = "part"
_LOCATED: Final = pa.schema(
    [pa.field(_VALUE, pa.string(), nullable=False), pa.field(_PART, pa.string(), nullable=False)]
)


@dataclass(frozen=True, slots=True)
class PartRule:
    """how one table is cut into parts.

    :ivar table: the table
    :ivar key: its key's columns
    :ivar scope_column: the column a part per value is cut by; None for a table held as one part (:data:`WHOLE_PART`)
    :ivar locator: a key column naming one scope value, for a split table whose key holds no ``scope_column``
    """

    table: str
    key: tuple[str, ...]
    scope_column: str | None = None
    locator: str | None = None

    def __post_init__(self) -> None:
        if self.locator is not None and self.scope_column is None:
            raise ValueError(f"{self.table}: a locator names a scope, and a table held as one part has none")
        if self.locator is not None and self.locator not in self.key:
            raise ValueError(f"{self.table}: the locator {self.locator!r} is not a key column {list(self.key)}")
        if self.scope_column is not None and self.scope_column not in self.key and self.locator is None:
            raise ValueError(
                f"{self.table}: a key without its scope column {self.scope_column!r} needs a locator: a key column "
                f"naming one scope value, so a deleted key finds its part"
            )

    @property
    def split(self) -> bool:
        """whether the table is cut a part per scope value.

        :return: True when cut by :attr:`scope_column`
        :rtype: bool
        """
        return self.scope_column is not None


@dataclass(frozen=True, slots=True)
class Part:
    """one part's rows.

    :ivar name: the part: the scope value as text, or :data:`WHOLE_PART`
    :ivar rows: its rows: a slice of the one sorted copy :func:`split_parts` makes, or the rows themselves when whole
    """

    name: str
    rows: pa.Table


def part_rules(
    tables: Sequence[FollowedTable],
    sizes: Mapping[str, int],
    *,
    locators: Mapping[str, str] = EMPTY,
    held: Mapping[str, PartRule] = EMPTY,
    floor: int = SPLIT_FLOOR_BYTES,
) -> Mapping[str, PartRule]:
    """each table's parts: by its declared scope column when it holds at least ``floor`` bytes, else one part; a table
    with no scope column is one part.

    A scoped table whose key lacks its scope column must be given a locator at every size, so a declaration missing
    one is refused when it is made, not when the table first grows past the floor.

    :param tables: the copied tables, each with its declared scope (``FollowedTable.scope``)
    :ptype tables: Sequence[FollowedTable]
    :param sizes: each table's bytes as the copy holds it (Arrow ``nbytes``); every table must be named
    :ptype sizes: Mapping[str, int]
    :param locators: each table keyed without its scope column, and the key column naming its scope value
    :ptype locators: Mapping[str, str]
    :param held: the rules the copy holds its tables by now: a table split stays split down to half the floor
    :ptype held: Mapping[str, PartRule]
    :param floor: the bytes a table is split at (:data:`SPLIT_FLOOR_BYTES`)
    :ptype floor: int
    :return: table -> its rule, read-only
    :rtype: Mapping[str, PartRule]
    :raises ValueError: for a table with no size, a scoped table keyed without its scope and no locator, or a locator
        for a table not given or not scoped
    """
    named = frozenset(table.name for table in tables)
    stray = sorted(set(locators) - named)
    if stray:
        raise ValueError(f"locators for tables not copied: {stray}")
    unsized = sorted(named - set(sizes))
    if unsized:
        raise ValueError(f"no size for {unsized}: a table's parts follow its size")
    rules: list[tuple[str, PartRule]] = []
    for table in tables:
        key = tuple(table.key)
        locator = locators.get(table.name)
        if table.scope is None:
            if locator is not None:
                raise ValueError(f"{table.name}: a locator names a scope, and the table declares none")
            rules.append((table.name, PartRule(table.name, key)))
            continue
        column = table.scope[0]
        # the declaration is checked at every size: a locator missing is refused now, not when the table grows
        declared = PartRule(table.name, key, scope_column=column, locator=locator)
        was_split = table.name in held and held[table.name].split
        big = sizes[table.name] >= (floor // _STAY_SPLIT_DIVISOR if was_split else floor)
        rules.append((table.name, declared if big else PartRule(table.name, key)))
    return frozen(rules)


def _scope_text(rule: PartRule, column: pa.ChunkedArray | pa.Array) -> pa.Array:
    """a scope column's values as part names (text), refusing a null: a row of no scope has no part."""
    values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if values.null_count:
        raise ValueError(f"{rule.table}: {values.null_count} rows hold no {rule.scope_column}, so no part")
    texts: pa.Array = pc.cast(values, pa.string())
    return texts


def split_parts(rule: PartRule, rows: pa.Table) -> tuple[Part, ...]:
    """``rows`` cut into the parts ``rule`` names, each part's rows in their order in ``rows``.

    One stable sort of the scope column and one run-end encoding of it; each part a slice of the sorted rows.

    :param rule: the table's rule
    :ptype rule: PartRule
    :param rows: the rows, holding the scope column when the table is split
    :ptype rows: pa.Table
    :return: the parts, by name ascending; none for no rows
    :rtype: tuple[Part, ...]
    :raises ValueError: for a row whose scope column is null
    """
    if not rows.num_rows:
        return ()
    if rule.scope_column is None:
        return (Part(WHOLE_PART, rows),)
    names = _scope_text(rule, rows[rule.scope_column])
    order = pc.sort_indices(names)
    ordered = rows.take(order)
    runs = pc.run_end_encode(names.take(order))
    ends = runs.run_ends.to_numpy()
    values = runs.values
    parts: list[Part] = []
    start = 0
    for at, end in enumerate(ends):
        parts.append(Part(str(values[at].as_py()), ordered.slice(start, int(end) - start)))
        start = int(end)
    return tuple(parts)


@dataclass(frozen=True, slots=True)
class Locator:
    """each locator value's part, as the rows the copy held named them: a two-column Arrow table (the value and its
    part, both text), each value once. Replaced whole, never changed in place (:meth:`extended`).

    :ivar located: the pairs
    """

    located: pa.Table = field(default_factory=_LOCATED.empty_table)

    def extended(self, rule: PartRule, rows: pa.Table) -> Locator:
        """this locator with each locator value ``rows`` hold placed in its row's part; a value's newest part wins.

        :param rule: the table's rule, split and keyed without its scope column
        :ptype rule: PartRule
        :param rows: rows the copy now holds of the table (its scope column and locator column at least)
        :ptype rows: pa.Table
        :return: the locator after them
        :rtype: Locator
        :raises ValueError: for a rule with no locator, or rows naming one locator value in two parts
        """
        if rule.locator is None or rule.scope_column is None:
            raise ValueError(f"{rule.table}: only a split table keyed without its scope column is located")
        if not rows.num_rows:
            return self
        values = rows[rule.locator].combine_chunks()
        if values.null_count:
            raise ValueError(f"{rule.table}: {values.null_count} rows hold no {rule.locator}, so locate nothing")
        texts: pa.Array = pc.cast(values, pa.string())
        parts = _scope_text(rule, rows[rule.scope_column])
        # each value once with the part of a row naming it, then every row's part held to its value's: two hash
        # lookups, no sort and no NumPy
        named: pa.Array = pc.unique(texts)
        part_of: pa.Array = parts.take(pc.index_in(named, value_set=texts))
        if not pc.all(pc.equal(part_of.take(pc.index_in(texts, value_set=named)), parts)).as_py():
            raise ValueError(f"{rule.table}: rows name one {rule.locator} in two {rule.scope_column} values")
        kept = self.located.filter(pc.invert(pc.is_in(self.located[_VALUE], value_set=named)))
        pairs = pa.Table.from_arrays([named, part_of], schema=_LOCATED)
        return Locator(pa.concat_tables([kept, pairs]).combine_chunks())


def parts_of_keys(rule: PartRule, keys: pa.Table, locator: Locator | None = None) -> pa.Array:
    """each key's part: :data:`WHOLE_PART` for a table held as one part, its scope column's value for a key holding
    it, else its locator value's part (null for a value the copy never held, so a key no part holds).

    :param rule: the table's rule
    :ptype rule: PartRule
    :param keys: the keys (the key's columns; others are not read)
    :ptype keys: pa.Table
    :param locator: the table's locator, for a split table keyed without its scope column
    :ptype locator: Locator | None
    :return: each key's part, a key's at its row number
    :rtype: pa.Array
    :raises ValueError: for a split table keyed without its scope column and no locator given
    """
    found: pa.Array
    if rule.scope_column is None:
        found = pa.repeat(pa.scalar(WHOLE_PART, pa.string()), keys.num_rows)
    elif rule.scope_column in keys.column_names:
        found = _scope_text(rule, keys[rule.scope_column])
    else:
        if locator is None or rule.locator is None:
            raise ValueError(f"{rule.table}: a key without {rule.scope_column} is found through its locator")
        values = pc.cast(keys[rule.locator].combine_chunks(), pa.string())
        at = pc.index_in(values, value_set=locator.located[_VALUE].combine_chunks())
        found = locator.located[_PART].combine_chunks().take(at)
    return found
