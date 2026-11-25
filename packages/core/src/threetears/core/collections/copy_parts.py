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

**A part's name** is its scope value's text when that is a plain store name (:func:`part_name`), else the value
escaped (``e.`` and each other byte as ``.XX``), or for a value too long even so a hash of it (``h.``): any scope
value names a part the store takes, and a part's name gives its value back (:func:`scope_of_part`; a hashed one's
value is its rows' scope column).

**Arrow end to end.** Rows are split by one sort and one run-end encoding of the scope column, each part a slice of
the sorted rows; a locator is a two-column Arrow table, matched by Arrow's hash kernels (``index_in``, ``is_in``),
never Acero (``tests/enforcement/test_no_acero_tables.py``). Python objects here are one a table or one a part, never
one a row or a key.
"""

from __future__ import annotations

import hashlib
import re
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
    "part_name",
    "parts_of_keys",
    "scope_of_part",
    "split_parts",
    "tier_of",
    "tier_part",
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

#: between a scope value and its runs in a per-run part's name: ``VA-r000000005001-000000005010-1`` (scope value, first
#: run, last run, level), so a scope's parts sort by their runs and a merged part sorts into its group's place (the
#: store's ``merge_parts`` requires both). Moved from reports (``aibots_reports.pod.follower_parts``)
_TIER_MARK: Final = "-r"

_TIER: Final = re.compile(r"^(?P<scope>.+)-r(?P<first>\d{12})-(?P<last>\d{12})-(?P<level>\d+)$")

#: a table already split stays split until it holds under the floor divided by this: a table near the floor (a
#: retention letting runs go, the next run adding them) would otherwise flip shape, and each flip rewrites it whole
_STAY_SPLIT_DIVISOR: Final = 2

#: the longest a scope value's part name may be: the store takes a part name of 128 at most
#: (``shared_arrow.plain_name``), and a per-run part adds its runs and level to it (:func:`tier_part`, 30 more for
#: levels under 100)
_PART_NAME_MAX: Final = 96

#: a scope value named as it is: the store's name characters but ``.``, which marks an escaped or hashed name, so the
#: three never collide
_PLAIN_PART: Final = re.compile(r"[A-Za-z0-9_-]{1,96}")
#: the same, for Arrow's regex kernels (RE2)
_PLAIN_PART_RE2: Final = r"^[A-Za-z0-9_-]{1,96}$"
#: the characters an escaped name keeps as they are; ``-`` is escaped there, so no escaped name reads as a tier
_ESCAPE_KEEPS: Final = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")
_ESCAPED: Final = "e."
_HASHED: Final = "h."

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
    :ivar run_column: with ``scope_column``, a key column of runs (an integer): a split table's rows are cut by scope
        value and run, in tiers (:func:`tier_part`): a new run is a level-0 part of its own, merged later with its
        neighbours into parts of higher levels (``shared_copy.merge_due``), so a change to the newest run rewrites
        that run's rows alone, and a scope of N runs holds about ``FAN_OUT * log(N)`` parts
    """

    table: str
    key: tuple[str, ...]
    scope_column: str | None = None
    locator: str | None = None
    run_column: str | None = None

    def __post_init__(self) -> None:
        if self.run_column is not None and self.scope_column is None:
            raise ValueError(
                f"{self.table}: a per-run table is cut by its scope too, and a table held as one part has none"
            )
        if self.run_column is not None and self.run_column not in self.key:
            raise ValueError(f"{self.table}: the run column {self.run_column!r} is not a key column {list(self.key)}")
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

    :ivar name: the part: its scope value's name (:func:`part_name`), or :data:`WHOLE_PART`
    :ivar rows: its rows: a slice of the one sorted copy :func:`split_parts` makes, or the rows themselves when whole
    """

    name: str
    rows: pa.Table


def part_rules(
    tables: Sequence[FollowedTable],
    sizes: Mapping[str, int],
    *,
    locators: Mapping[str, str] = EMPTY,
    runs: Mapping[str, str] = EMPTY,
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
    :param runs: each per-run table and its run column: split, it is cut by scope value and run, in tiers
        (:attr:`PartRule.run_column`); held as one part, by nothing
    :ptype runs: Mapping[str, str]
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
    stray = sorted((set(locators) | set(runs)) - named)
    if stray:
        raise ValueError(f"locators or run columns for tables not copied: {stray}")
    unsized = sorted(named - set(sizes))
    if unsized:
        raise ValueError(f"no size for {unsized}: a table's parts follow its size")
    rules: list[tuple[str, PartRule]] = []
    for table in tables:
        key = tuple(table.key)
        locator = locators.get(table.name)
        if table.scope is None:
            if locator is not None or table.name in runs:
                raise ValueError(
                    f"{table.name}: a locator or a run column cuts by a scope, and the table declares none"
                )
            rules.append((table.name, PartRule(table.name, key)))
            continue
        column = table.scope[0]
        # the declaration is checked at every size: a locator missing is refused now, not when the table grows
        declared = PartRule(table.name, key, scope_column=column, locator=locator, run_column=runs.get(table.name))
        was_split = table.name in held and held[table.name].split
        big = sizes[table.name] >= (floor // _STAY_SPLIT_DIVISOR if was_split else floor)
        rules.append((table.name, declared if big else PartRule(table.name, key)))
    return frozen(rules)


def tier_part(scope: str, level: int, first: int, last: int) -> str:
    """a per-run part's name (:attr:`PartRule.run_column`).

    :param scope: the scope value, as text
    :ptype scope: str
    :param level: its tier: 0 for one run
    :ptype level: int
    :param first: its first run
    :ptype first: int
    :param last: its last run
    :ptype last: int
    :return: the name
    :rtype: str
    """
    return f"{scope}{_TIER_MARK}{first:012d}-{last:012d}-{level}"


def tier_of(part: str) -> tuple[str, int, int, int] | None:
    """a per-run part's scope value, level, first and last run; None for any other part.

    :param part: the part's name
    :ptype part: str
    :return: ``(scope, level, first, last)``, or None
    :rtype: tuple[str, int, int, int] | None
    """
    found = _TIER.match(part)
    return None if found is None else (found["scope"], int(found["level"]), int(found["first"]), int(found["last"]))


def part_name(value: str) -> str:
    """the part a scope value's text names: the text itself when it is a plain store name (``[A-Za-z0-9_-]``, 96 at
    most, neither :data:`WHOLE_PART` nor a tier's name), else ``e.`` and the text with every byte but ``[A-Za-z0-9_]``
    as ``.XX`` (its UTF-8 in hex), else (still over 96) ``h.`` and 32 hex digits of its SHA-256.

    **Why.** A scope value is any text (a race's name, a county with a space or an apostrophe), and the store refuses a
    part name of other characters at publish: such a value's change failed there, after its whole apply. Escaped, the
    name is the value's, read back by :func:`scope_of_part`; the three forms never collide (a plain name holds no
    ``.``; an escaped one no ``-``, so never reads as a tier's).

    :param value: the scope value's text
    :ptype value: str
    :return: its part's name, a name the store takes
    :rtype: str
    """
    if _PLAIN_PART.fullmatch(value) and value != WHOLE_PART and _TIER.match(value) is None:
        return value
    escaped = _ESCAPED + "".join(
        char if char in _ESCAPE_KEEPS else "".join(f".{byte:02X}" for byte in char.encode()) for char in value
    )
    if len(escaped) <= _PART_NAME_MAX:
        return escaped
    return _HASHED + hashlib.sha256(value.encode()).hexdigest()[:32]


def scope_of_part(name: str) -> str | None:
    """the scope value's text a part's name (:func:`part_name`, or a tier's scope, :func:`tier_of`) names; None for a
    hashed name (its value is its rows' scope column).

    :param name: the part's name
    :ptype name: str
    :return: the value's text, or None
    :rtype: str | None
    """
    if name.startswith(_HASHED):
        return None
    if not name.startswith(_ESCAPED):
        return name
    text = name[len(_ESCAPED) :]
    raw = bytearray()
    at = 0
    while at < len(text):
        if text[at] == ".":
            raw += bytes.fromhex(text[at + 1 : at + 3])
            at += 3
        else:
            raw += text[at].encode()
            at += 1
    return raw.decode()


def _part_names(texts: pa.Array) -> pa.Array:
    """each scope value's part name (:func:`part_name`), in Arrow: a column of plain names is itself; else each
    distinct other value is named once (one Python object a part, never a row) and every row takes its value's."""
    plain = pc.and_(
        pc.match_substring_regex(texts, _PLAIN_PART_RE2),
        pc.and_(pc.not_equal(texts, WHOLE_PART), pc.invert(pc.match_substring_regex(texts, _TIER.pattern))),
    )
    if pc.all(plain).as_py() is not False:
        return texts
    others: pa.Array = pc.unique(texts.filter(pc.invert(plain)))
    named = pa.array([part_name(value) for value in others.to_pylist()], pa.string())
    found: pa.Array = pc.if_else(plain, texts, named.take(pc.index_in(texts, value_set=others)))
    return found


def _scope_text(rule: PartRule, column: pa.ChunkedArray | pa.Array) -> pa.Array:
    """a scope column's values as part names (:func:`part_name`), refusing a null: a row of no scope has no part."""
    values = column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column
    if values.null_count:
        raise ValueError(f"{rule.table}: {values.null_count} rows hold no {rule.scope_column}, so no part")
    return _part_names(pc.cast(values, pa.string()))


def split_parts(rule: PartRule, rows: pa.Table) -> tuple[Part, ...]:
    """``rows`` cut into the parts ``rule`` names, each part's rows in their order in ``rows``.

    One stable sort of the scope column and one run-end encoding of it; each part a slice of the sorted rows.

    :param rule: the table's rule
    :ptype rule: PartRule
    :param rows: the rows, holding the scope column when the table is split
    :ptype rows: pa.Table
    :return: the parts, by name ascending; none for no rows
    :rtype: tuple[Part, ...]
    :raises ValueError: for a row whose scope column is null, or a per-run table (its rows' parts are the tiers the
        copy holds: ``shared_copy`` cuts it)
    """
    if rule.run_column is not None:
        raise ValueError(f"{rule.table}: a per-run table's parts are the tiers the copy holds, not its scope's alone")
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

    def overlaid(self, newer: Locator) -> Locator:
        """this locator with every value ``newer`` places placed as it places it (a per-scope reload's locator over the
        copy's): one hash lookup, no row read.

        :param newer: the pairs that win
        :ptype newer: Locator
        :return: the locator after them
        :rtype: Locator
        """
        if not newer.located.num_rows:
            return self
        values = newer.located[_VALUE].combine_chunks()
        kept = self.located.filter(pc.invert(pc.is_in(self.located[_VALUE], value_set=values)))
        return Locator(pa.concat_tables([kept, newer.located.cast(_LOCATED)]).combine_chunks())


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
