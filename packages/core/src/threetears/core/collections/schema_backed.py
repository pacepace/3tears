"""schema-backed collection primitive.

collapses the CRUD boilerplate that every
:class:`~threetears.core.collections.base.BaseCollection` subclass used
to hand-roll (``fetch_from_postgres`` / ``save_to_postgres`` /
``delete_from_postgres`` / ``serialize`` / ``deserialize``) into one
schema-driven implementation. subclasses declare a
:class:`TableSchema` as a class attribute and stop writing SQL by hand
for the standard row lifecycle. domain-specific query methods
(``find_by_agent`` / ``hybrid_search`` / etc.) stay on subclasses
because query shape is legitimately per-collection; CRUD is not.

per CLAUDE.md (14-eng-ai-bot) "NO BACKWARDS-COMPATIBILITY SHIMS": a
subclass that adopts :class:`SchemaBackedCollection` deletes its
hand-rolled CRUD in the same commit. there is no dual-path
"override-wins" fallback -- schema is the declaration, the base class
is the implementation.

**pgvector:** ``VECTOR_TYPE`` is deliberately generic (not named
``EMBEDDING_TYPE``) because pgvector rows can carry non-embedding
vectors. write path emits the textual bracketed form plus ``::vector``
cast; read path coerces the textual form back to ``list[float]``.

**composite pk:** mirrors the
:class:`~threetears.core.collections.base.BaseCollection` composite-pk
contract added in namespace-task-01 phase 8.5l-1. declare
``TableSchema.primary_key = ("a", "b")`` and the generator emits
``WHERE a = $1 AND b = $2`` for fetch / delete, ``ON CONFLICT (a, b)``
for upsert, and ``WHERE a = $1 AND b = $2 AND date_updated = $N`` for
CAS.
"""

from __future__ import annotations

import base64
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import wraps
from typing import Any, ClassVar, Generic, Literal, TypeVar, overload
from uuid import UUID

from threetears.core.collections.base import BaseCollection, EntityT
from threetears.observe import get_logger

__all__ = [
    "BOOL_TYPE",
    "BYTES_TYPE",
    "Column",
    "DATETIME_TYPE",
    "DATETIMETZ_TYPE",
    "INT_TYPE",
    "JSONB_TYPE",
    "OnConflict",
    "PartitionEnforcementError",
    "SchemaBackedCollection",
    "STRING_TYPE",
    "TableSchema",
    "UUID_TYPE",
    "VECTOR_TYPE",
    "spans_partitions",
]

# on_conflict enum values for :class:`TableSchema`.
#
# - ``"raise"``: emit plain INSERT with no ``ON CONFLICT`` clause; duplicate
#   primary-key inserts surface the asyncpg ``UniqueViolationError`` directly
#   to the caller. journal / append-only tables.
# - ``"ignore"``: emit ``INSERT ... ON CONFLICT (pk) DO NOTHING``; duplicate
#   primary-key inserts are silently dropped. dedup-on-redelivery tables
#   (audit envelopes, idempotent event logs).
# - ``"update"``: emit ``INSERT ... ON CONFLICT (pk) DO UPDATE SET <mutable>``
#   (default). standard upsert behaviour for editable rows.
OnConflict = Literal["raise", "ignore", "update"]

log = get_logger(__name__)


# sentinel strings for column type tags. chosen over real Python types
# because several logical types collapse to the same Python type
# (``list[float]`` vs ``list[dict]`` for vector vs jsonb arrays) and
# because the serialization layer already distinguishes types by tag
# rather than by runtime ``isinstance`` checks.
UUID_TYPE = "uuid"
STRING_TYPE = "string"
# DATETIME_TYPE: column declared as ``TIMESTAMP`` (timezone-naive
# wall-clock) in the L3 DDL. asyncpg's TIMESTAMP codec accepts naive
# datetimes only and rejects aware ones with ``DataError``; the write
# coercion strips ``tzinfo`` (after projecting any aware input to UTC)
# so callers that hold aware-UTC datetimes throughout the platform
# (per CLAUDE.md datetime rule) round-trip cleanly.
DATETIME_TYPE = "datetime"
# DATETIMETZ_TYPE: column declared as ``TIMESTAMPTZ`` (timezone-aware
# instant) in the L3 DDL. asyncpg's TIMESTAMPTZ codec calls
# ``obj.astimezone(utc)`` on every value, which interprets a naive
# datetime as the **client's local timezone** (not UTC). that means
# stripping ``tzinfo`` before binding silently shifts the wire value
# by the developer's local TZ offset on non-UTC hosts -- the row gets
# stored as ``utc + local_offset``, the next read returns
# ``utc + local_offset``, and any subsequent CAS predicate that strips
# tzinfo again shifts a second time and fails to match. the write
# coercion for this type ensures a tz-aware UTC datetime regardless of
# input shape (naive inputs are wrapped with ``UTC``; aware inputs are
# normalized via ``astimezone(UTC)``) so the codec's
# ``obj.astimezone(utc)`` is a no-op and the wire value is the true
# UTC instant. read coercion ensures the returned value carries
# ``tzinfo=UTC`` so downstream callers (and CAS predicates that
# round-trip through this collection) see a stable shape.
DATETIMETZ_TYPE = "datetimetz"
JSONB_TYPE = "jsonb"
BYTES_TYPE = "bytes"
INT_TYPE = "int"
BOOL_TYPE = "bool"
VECTOR_TYPE = "vector"


@dataclass(frozen=True)
class Column:
    """column descriptor for :class:`TableSchema`.

    :cvar name: column name (matches the L3 DDL and the row-dict key)
    :cvar column_type: one of the module-level type tags
        (``UUID_TYPE``, ``STRING_TYPE``, etc.); used by
        :meth:`SchemaBackedCollection._normalize_write_value` /
        :meth:`SchemaBackedCollection._normalize_read_value` to pick
        the right coercion rule and by the SQL generator to decide
        whether a parameter needs a ``::jsonb`` / ``::vector`` cast
    :cvar immutable: when ``True``, column is emitted on INSERT but
        excluded from ``DO UPDATE SET`` and from the CAS UPDATE path.
        primary-key columns are implicitly immutable (enforced by
        :class:`TableSchema`). typical uses: ``date_created``,
        ``agent_id``, ``customer_id``, ``conversation_id``
    :cvar nullable: when ``True``, missing values in the write dict
        default to ``None``; when ``False``, a missing key raises
        :class:`KeyError` at write time so an incomplete payload
        fails loudly instead of silently writing ``NULL``
    :cvar partition: when ``True``, column is the partition key for
        the table. partition columns must be part of
        :attr:`TableSchema.primary_key` so the schema enforces row
        uniqueness through the partition. partition columns are also
        :attr:`immutable` by definition (the value identifies WHICH
        partition the row belongs to and changing it would corrupt
        the partition map); the :class:`TableSchema` validator coerces
        ``immutable=True`` automatically when ``partition=True`` to
        keep declarations terse. on a Collection mixed with the
        :class:`SchemaBackedCollection` partition guard, every
        public read / write method must either accept the partition
        column as a required argument, be decorated
        ``@spans_partitions`` to opt into multi-partition reads, or
        be allowlisted via ``_partition_exempt_methods`` with an
        explicit rationale -- protects against the cross-partition
        bleed class of bug
    :cvar nullable: when ``True``, missing values default to ``None``;
        ignored when ``partition=True`` (partition columns must be
        non-null by construction)

    :param name: column name
    :ptype name: str
    :param column_type: one of the module-level type tags
    :ptype column_type: str
    :param immutable: INSERT-only flag; defaults to ``False``
    :ptype immutable: bool
    :param nullable: defaultable flag; defaults to ``False``
    :ptype nullable: bool
    :param partition: partition-column flag; defaults to ``False``
    :ptype partition: bool
    """

    name: str
    column_type: str
    immutable: bool = False
    nullable: bool = False
    partition: bool = False


@dataclass(frozen=True)
class TableSchema:
    """table descriptor for :class:`SchemaBackedCollection`.

    :cvar name: table name (matches the L3 DDL)
    :cvar primary_key: single column name (single-pk shape) or tuple
        of column names in declared order (composite-pk shape)
    :cvar columns: list of :class:`Column` in the order they should
        appear in generated INSERT statements. order is stable across
        invocations so asyncpg parameter positions (``$1``, ``$2``, ...)
        remain predictable for tests that assert on positional args
    :cvar cas_column: column used as the optimistic-concurrency fence
        on the UPDATE path. when set (typical value: ``"date_updated"``)
        and the caller passes a non-``None`` ``original_timestamp`` to
        :meth:`SchemaBackedCollection.save_to_postgres`, the generator
        emits a separate UPDATE fenced on ``cas_column = $N``. when
        ``None``, the ``original_timestamp`` argument is silently
        ignored
    :cvar on_conflict: behaviour of
        :meth:`SchemaBackedCollection.save_to_postgres` on primary-key
        conflict. ``"update"`` (default) emits ``ON CONFLICT (pk) DO
        UPDATE SET <mutable>`` -- the standard upsert path. ``"raise"``
        emits plain INSERT with no ``ON CONFLICT`` clause; duplicate
        primary keys raise :class:`asyncpg.exceptions.UniqueViolationError`
        from asyncpg naturally (journal / append-only tables).
        ``"ignore"`` emits ``ON CONFLICT (pk) DO NOTHING``; duplicate
        primary keys are silently dropped (dedup-on-redelivery tables
        like ``audit_events`` keyed on ``(correlation_id, event_type)``)
    """

    name: str
    primary_key: str | tuple[str, ...]
    columns: list[Column]
    cas_column: str | None = None
    on_conflict: OnConflict = "update"

    _pk_columns: tuple[str, ...] = field(init=False, repr=False)
    _by_name: dict[str, Column] = field(init=False, repr=False)
    _partition_column: str | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """validate + index the schema after construction.

        :return: nothing
        :rtype: None
        :raises ValueError: when a pk column is not declared in
            :attr:`columns`, when :attr:`cas_column` is set but missing
            from :attr:`columns`, when more than one column is flagged
            ``partition=True`` (only one partition column per table),
            or when a partition column is not part of
            :attr:`primary_key` (the schema must enforce row uniqueness
            through the partition)
        """
        pk = self.primary_key
        pk_cols: tuple[str, ...] = (pk,) if isinstance(pk, str) else tuple(pk)
        # coerce partition columns to immutable so callers do not have
        # to declare both flags. partition columns identify the slot a
        # row belongs to; mutating the value would corrupt the partition
        # map.
        coerced_columns: list[Column] = []
        partition_cols: list[str] = []
        for col in self.columns:
            if col.partition:
                partition_cols.append(col.name)
                if not col.immutable:
                    col = Column(  # noqa: PLW2901 -- intentional coercion
                        name=col.name,
                        column_type=col.column_type,
                        immutable=True,
                        nullable=col.nullable,
                        partition=True,
                    )
            coerced_columns.append(col)
        if len(partition_cols) > 1:
            raise ValueError(
                f"TableSchema(name={self.name!r}): only one partition "
                f"column per table is supported, got {partition_cols!r}",
            )
        partition_column: str | None = partition_cols[0] if partition_cols else None
        if partition_column is not None and partition_column not in pk_cols:
            raise ValueError(
                f"TableSchema(name={self.name!r}): partition column "
                f"{partition_column!r} must be part of primary_key "
                f"(got primary_key={pk_cols!r}) so the schema enforces "
                f"row uniqueness through the partition",
            )
        by_name = {c.name: c for c in coerced_columns}
        for pk_col in pk_cols:
            if pk_col not in by_name:
                raise ValueError(
                    f"TableSchema(name={self.name!r}): primary_key column "
                    f"{pk_col!r} not found in columns",
                )
        if self.cas_column is not None and self.cas_column not in by_name:
            raise ValueError(
                f"TableSchema(name={self.name!r}): cas_column "
                f"{self.cas_column!r} not found in columns",
            )
        # dataclass is frozen, use object.__setattr__ to install derived
        # attributes
        object.__setattr__(self, "columns", coerced_columns)
        object.__setattr__(self, "_pk_columns", pk_cols)
        object.__setattr__(self, "_by_name", by_name)
        object.__setattr__(self, "_partition_column", partition_column)

    @property
    def pk_columns(self) -> tuple[str, ...]:
        """return primary-key column names as a tuple (always tuple-shape).

        :return: pk column names in declared order
        :rtype: tuple[str, ...]
        """
        return self._pk_columns

    @property
    def partition_column(self) -> str | None:
        """return the partition-column name, or ``None`` for non-partitioned tables.

        :return: partition column name when one column is flagged
            ``partition=True``, ``None`` otherwise
        :rtype: str | None
        """
        return self._partition_column

    def column(self, name: str) -> Column:
        """resolve a column descriptor by name.

        :param name: column name
        :ptype name: str
        :return: matching :class:`Column`
        :rtype: Column
        :raises KeyError: when no column with the given name exists
        """
        return self._by_name[name]

    def get_column(self, name: str) -> Column | None:
        """resolve a column descriptor by name, returning ``None`` on miss.

        non-raising variant of :meth:`column`. used by the read /
        deserialize paths when a row carries extra columns not
        declared in the schema (ad-hoc joined projections,
        migration-in-flight rows with stray fields) -- those pass
        through untouched rather than raising.

        :param name: column name
        :ptype name: str
        :return: matching :class:`Column` or ``None`` when not declared
        :rtype: Column | None
        """
        return self._by_name.get(name)

    def mutable_columns(self) -> list[Column]:
        """return non-pk, non-immutable columns in declared order.

        used by the upsert generator (``DO UPDATE SET``) and the CAS
        UPDATE generator.

        :return: list of columns eligible for update
        :rtype: list[Column]
        """
        pk_set = set(self._pk_columns)
        return [c for c in self.columns if not c.immutable and c.name not in pk_set]


def _coerce_uuid(value: Any) -> UUID | None:
    """coerce a value to a stdlib :class:`UUID`.

    asyncpg returns ``asyncpg.pgproto.pgproto.UUID`` for UUID columns,
    which is a subclass of stdlib UUID. SQLite's parameter binder
    rejects the subclass (it special-cases stdlib UUID only), so every
    read coerces to stdlib UUID unconditionally.

    :param value: UUID-shaped input (stdlib UUID, pgproto UUID, or str)
    :ptype value: Any
    :return: stdlib UUID or ``None`` pass-through
    :rtype: UUID | None
    """
    if value is None:
        result: UUID | None = None
    elif type(value) is UUID:  # noqa: E721 -- exact type match, see docstring
        result = value
    elif isinstance(value, UUID):  # pgproto subclass
        result = UUID(str(value))
    else:
        result = UUID(str(value))
    return result


def _coerce_datetime_for_write(value: Any) -> datetime | None:
    """coerce a datetime to naive-UTC form for TIMESTAMP column writes.

    belt-and-suspenders alongside
    :meth:`BaseCollection.save_entity`: aware datetimes become naive
    (UTC-projected), naive pass through. ``None`` pass-through.

    :param value: datetime input (aware / naive) or ``None``
    :ptype value: Any
    :return: naive datetime or ``None``
    :rtype: datetime | None
    """
    if value is None:
        result: datetime | None = None
    elif not isinstance(value, datetime):
        result = value
    elif value.tzinfo is None:
        result = value
    else:
        result = value.astimezone(UTC).replace(tzinfo=None)
    return result


def _coerce_datetime_for_write_tz(value: Any) -> datetime | None:
    """coerce a datetime to aware-UTC form for TIMESTAMPTZ column writes.

    asyncpg's TIMESTAMPTZ codec runs ``obj.astimezone(utc)`` on every
    bound parameter. for a naive input the call interprets the wall-
    clock value as the **client's local timezone**, so a naive value
    that semantically represents UTC (which is what
    :meth:`BaseCollection.save_entity` passes after stripping ``tzinfo``
    for the legacy TIMESTAMP path) gets shifted by the host's local
    offset before it ever reaches the database. on non-UTC hosts that
    leak silently corrupts every TIMESTAMPTZ write and breaks every
    subsequent CAS predicate (the predicate value goes through the
    same shift on the way out).

    this coercion eliminates the ambiguity by ensuring the value is
    aware-UTC before binding: a naive input is interpreted as UTC
    (``replace(tzinfo=UTC)``) -- the contract for
    :class:`SchemaBackedCollection` is "core code holds aware UTC
    everywhere; if a value reached this point as naive it was a
    UTC-projected naive (per CLAUDE.md datetime rule)". an aware
    input is normalized via ``astimezone(UTC)`` so non-UTC tzinfo
    values (rare but legal) flatten to UTC before the codec runs its
    own no-op ``astimezone(utc)``. ``None`` pass-through.

    :param value: datetime input (aware / naive) or ``None``
    :ptype value: Any
    :return: aware-UTC datetime or ``None``
    :rtype: datetime | None
    """
    if value is None:
        result: datetime | None = None
    elif not isinstance(value, datetime):
        result = value
    elif value.tzinfo is None:
        result = value.replace(tzinfo=UTC)
    else:
        result = value.astimezone(UTC)
    return result


def _coerce_datetime_for_read_tz(value: Any) -> datetime | None:
    """coerce a TIMESTAMPTZ read value to aware-UTC form.

    asyncpg's TIMESTAMPTZ decoder already returns aware-UTC datetimes,
    so the typical path is a passthrough. this coercion exists for the
    edge cases where a TIMESTAMPTZ value re-enters the read normalizer
    from a non-asyncpg source (L2 JSON deserialization, hand-rolled
    proxy pools that round-trip through string isoformat) and arrives
    naive. naive values are wrapped with ``UTC`` (the column type
    asserts "this is a UTC instant"), aware values are normalized to
    UTC tzinfo so equality comparisons against asyncpg-decoded values
    are byte-stable. ``None`` pass-through.

    :param value: datetime input (aware / naive) or ``None``
    :ptype value: Any
    :return: aware-UTC datetime or ``None``
    :rtype: datetime | None
    """
    if value is None:
        result: datetime | None = None
    elif not isinstance(value, datetime):
        result = value
    elif value.tzinfo is None:
        result = value.replace(tzinfo=UTC)
    else:
        result = value.astimezone(UTC)
    return result


def _encode_vector(value: Any) -> str | None:
    """encode a vector for pgvector's ``::vector`` cast.

    asyncpg has no native pgvector codec; values must be passed as the
    literal bracketed textual representation (``"[1.0, 2.0, ...]"``).
    list inputs get JSON-encoded at the WRITE boundary. already-encoded
    strings pass through. ``None`` pass-through.

    :param value: list of floats, pre-encoded string, or ``None``
    :ptype value: Any
    :return: textual vector, passthrough string, or ``None``
    :rtype: str | None
    """
    if value is None:
        result: str | None = None
    elif isinstance(value, str):
        result = value
    elif isinstance(value, list):
        result = json.dumps(value)
    else:
        result = str(value)
    return result


def _decode_vector(value: Any) -> list[float] | None:
    """decode a pgvector textual vector back to ``list[float]``.

    :param value: textual vector (e.g. ``"[1.0,2.0]"``), list passthrough,
        or ``None``
    :ptype value: Any
    :return: list of floats or ``None``
    :rtype: list[float] | None
    """
    if value is None:
        result: list[float] | None = None
    elif isinstance(value, list):
        result = [float(v) for v in value]
    elif isinstance(value, str):
        parsed = json.loads(value)
        result = [float(v) for v in parsed] if isinstance(parsed, list) else None
    else:
        result = None
    return result


def _encode_jsonb(value: Any) -> str | None:
    """encode a JSONB column value for the ``::jsonb`` cast.

    dict/list values become JSON strings; pre-encoded strings pass
    through; ``None`` pass-through.

    :param value: dict, list, pre-encoded string, or ``None``
    :ptype value: Any
    :return: JSON string or ``None``
    :rtype: str | None
    """
    if value is None:
        result: str | None = None
    elif isinstance(value, str):
        result = value
    elif isinstance(value, (dict, list)):
        result = json.dumps(value, default=_json_default)
    else:
        result = json.dumps(value, default=_json_default)
    return result


def _decode_jsonb(value: Any) -> Any:
    """decode a JSONB column value into a Python dict / list.

    asyncpg returns JSONB as a string unless a codec is registered;
    callers downstream expect the decoded structure. malformed payload
    surfaces as the raw string so the corruption is visible rather than
    silently swallowed.

    :param value: JSON string, decoded object, or ``None``
    :ptype value: Any
    :return: decoded structure (typically ``dict``), passthrough, or
        ``None``
    :rtype: Any
    """
    if value is None:
        result: Any = None
    elif isinstance(value, str) and value:
        try:
            result = json.loads(value)
        except (ValueError, TypeError):
            result = value
    else:
        result = value
    return result


def _json_default(obj: object) -> Any:
    """default handler for :func:`json.dumps` covering platform types.

    :param obj: value that :func:`json.dumps` cannot encode natively
    :ptype obj: object
    :return: JSON-compatible representation
    :rtype: Any
    :raises TypeError: when ``obj`` type is not covered
    """
    if isinstance(obj, UUID):
        result: Any = str(obj)
    elif isinstance(obj, datetime):
        result = obj.isoformat()
    elif isinstance(obj, bytes):
        result = base64.b64encode(obj).decode("ascii")
    else:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    return result


def _decode_l2_value(column: Column, value: Any) -> Any:
    """decode a single field from the L2 JSON payload back to native type.

    :param column: column descriptor controlling the coercion rule
    :ptype column: Column
    :param value: JSON-decoded raw value
    :ptype value: Any
    :return: value coerced to the column's native Python type
    :rtype: Any
    """
    if value is None:
        result: Any = None
    elif column.column_type == UUID_TYPE and isinstance(value, str):
        result = UUID(value)
    elif column.column_type == DATETIME_TYPE and isinstance(value, str):
        result = datetime.fromisoformat(value)
    elif column.column_type == DATETIMETZ_TYPE and isinstance(value, str):
        # isoformat strings carrying ``+00:00`` round-trip to aware UTC
        # naturally; strings without an offset (legacy L2 payloads from
        # before this column type existed) are interpreted as UTC.
        parsed = datetime.fromisoformat(value)
        result = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    elif column.column_type == BYTES_TYPE and isinstance(value, str):
        result = base64.b64decode(value.encode("ascii"))
    elif column.column_type == BOOL_TYPE and isinstance(value, (bool, int)):
        result = bool(value)
    elif column.column_type == INT_TYPE and isinstance(value, (int, float)):
        result = int(value)
    elif column.column_type == VECTOR_TYPE:
        result = _decode_vector(value)
    elif column.column_type == JSONB_TYPE:
        # JSONB round-trips through JSON natively; value is already the
        # decoded structure
        result = value
    else:
        result = value
    return result


class PartitionEnforcementError(TypeError):
    """raised at class-definition time when a partitioned collection
    declares a public method that neither accepts the partition column
    nor opts into the cross-partition path via :func:`spans_partitions`.

    catches the cross-partition bleed class of bug at import time --
    long before a stray ``find_by_user`` lacking ``conversation_id``
    can ship to production and silently surface another conversation's
    rows.

    resolution: add the partition column to the method signature, mark
    the method with ``@spans_partitions``, or list it in
    ``cls._partition_exempt_methods`` with a ``# rationale: ...``
    inline comment. blanket exemptions ("internal helper", "tests need
    this") are rejected by code review.
    """


F = TypeVar("F", bound=Callable[..., Any])


@overload
def spans_partitions(method: F) -> F: ...


@overload
def spans_partitions(
    method: None = None, *, marker_only: bool = False,
) -> Callable[[F], F]: ...


def spans_partitions(
    method: F | None = None, *, marker_only: bool = False,
) -> F | Callable[[F], F]:
    """mark a Collection method as deliberately cross-partition.

    decorated methods are exempt from the
    :class:`SchemaBackedCollection` partition-column guard; they may
    span multiple partitions intentionally (e.g. an authorized
    cross-agent retrieval where the caller has resolved the authorized
    set of agent IDs upstream and passes them in as a tuple).

    contract: at call time, the decorator validates that the caller
    passed a ``tuple`` (not a list, not a single value, not a
    generator) for the partition-spanning argument. tuples are an
    explicit "I deliberately fan out across these partitions" signal;
    anything else is treated as a programming error and raises
    :class:`TypeError`. the wrapped callable inspects the method
    signature to find the first parameter whose name ends in ``_ids``,
    a plural form of the partition column. callers that need a
    different sentinel should override the convention by passing the
    ``tuple`` argument under any name ending in ``_ids``.

    when no ``_ids``-suffix parameter is found, decoration FAILS at
    class-definition time with :class:`PartitionEnforcementError`
    unless the caller explicitly passed ``marker_only=True``. the
    marker-only opt-in is a deliberate noise -- adding it to a method
    requires the author to justify why the method is structurally
    cross-partition by other means (e.g. an opaque pre-built SQL
    string where ACL is enforced upstream). this closes the silent-
    degrade failure mode flagged by review-task-01 finding D-2 in
    partition-hardening-task-01 sub-task 2.

    two calling forms are supported:

    - ``@spans_partitions`` (bare) -- method must declare an
      ``_ids``-suffix parameter; runtime guard validates tuple shape.
    - ``@spans_partitions(marker_only=True)`` -- method has no
      ``_ids`` parameter; runtime guard is skipped entirely; the
      decorator becomes a pure compile-time exemption marker against
      :meth:`SchemaBackedCollection.__init_subclass__`'s partition
      discipline.

    :param method: collection method to decorate (bare-decorator form)
    :ptype method: Callable[..., Any] | None
    :param marker_only: when ``True``, skip the ``_ids`` parameter
        scan + runtime guard; method becomes a pure compile-time
        exemption marker
    :ptype marker_only: bool
    :return: wrapped method, or a decorator factory when invoked with
        keyword arguments
    :rtype: Callable[..., Any] | Callable[[F], F]
    :raises PartitionEnforcementError: at decoration time when the
        target method has no ``_ids``-suffix parameter and
        ``marker_only=True`` was not passed
    """
    def _decorate(target: F) -> F:
        sig = inspect.signature(target)
        plural_param: str | None = None
        for name in sig.parameters:
            if name.endswith("_ids"):
                plural_param = name
                break
        if plural_param is None and not marker_only:
            raise PartitionEnforcementError(
                f"{target.__qualname__}: @spans_partitions found no "
                f"parameter ending in '_ids'. either rename the "
                f"partition-spanning parameter to end in '_ids' so the "
                f"runtime tuple-shape guard can validate the fan-out, "
                f"OR opt into the marker-only path with "
                f"@spans_partitions(marker_only=True) -- the latter "
                f"declares 'this method is structurally cross-partition "
                f"by other means' (e.g. opaque pre-built SQL with ACL "
                f"enforced upstream) and skips the runtime guard.",
            )

        @wraps(target)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """call-site guard for ``@spans_partitions``-marked methods.

            :raises TypeError: when the cross-partition argument is not
                a ``tuple`` or has length zero
            """
            if plural_param is not None and not marker_only:
                bound = sig.bind(*args, **kwargs)
                value = bound.arguments.get(plural_param)
                if value is not None:
                    if not isinstance(value, tuple):
                        raise TypeError(
                            f"{target.__qualname__}: cross-partition "
                            f"argument {plural_param!r} must be a tuple "
                            f"(deliberate fan-out signal); got "
                            f"{type(value).__name__}",
                        )
                    if len(value) == 0:
                        raise TypeError(
                            f"{target.__qualname__}: cross-partition "
                            f"argument {plural_param!r} is an empty "
                            f"tuple; refusing to emit a query that "
                            f"spans zero partitions",
                        )
            return target(*args, **kwargs)

        setattr(wrapper, "_spans_partitions", True)
        return wrapper  # type: ignore[return-value]

    if method is not None:
        return _decorate(method)
    return _decorate


def _is_partition_exempt(method: Any) -> bool:
    """true iff ``method`` is decorated with :func:`spans_partitions`.

    :param method: function or method object
    :ptype method: Any
    :return: presence of the marker attribute
    :rtype: bool
    """
    return getattr(method, "_spans_partitions", False) is True


def _method_accepts_partition(method: Any, partition_column: str) -> bool:
    """true iff ``method`` declares a parameter named ``partition_column``.

    used by the :class:`SchemaBackedCollection` ``__init_subclass__``
    hook to verify every public method on a partitioned Collection
    accepts the partition column. matches positional and keyword-only
    parameters.

    :param method: function or method object
    :ptype method: Any
    :param partition_column: name of the table's partition column
    :ptype partition_column: str
    :return: presence of a parameter named ``partition_column``
    :rtype: bool
    """
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return partition_column in sig.parameters


class SchemaBackedCollection(BaseCollection[EntityT], Generic[EntityT]):
    """base collection that generates CRUD SQL from a :class:`TableSchema`.

    subclasses declare::

        class MyCollection(SchemaBackedCollection[MyEntity]):
            schema = TableSchema(
                name="my_table",
                primary_key="id",
                columns=[
                    Column("id", UUID_TYPE),
                    Column("name", STRING_TYPE),
                    Column("date_created", DATETIME_TYPE, immutable=True),
                    Column("date_updated", DATETIME_TYPE),
                ],
                cas_column="date_updated",
            )
            entity_class = MyEntity
            primary_key_column = "id"

    and inherit the full CRUD implementation:
    :meth:`fetch_from_postgres`, :meth:`save_to_postgres`,
    :meth:`delete_from_postgres`, :meth:`serialize`, :meth:`deserialize`.
    domain-specific query methods (e.g. ``find_by_agent``) stay on the
    subclass because query shape is per-collection.

    the L3 pool is read from :attr:`l3_pool` (the public extension
    point on :class:`BaseCollection`). subclasses that accept a
    ``postgres_pool`` constructor kwarg should assign it onto
    ``self.l3_pool`` in their ``__init__`` after calling ``super()``
    so the generic path finds the pool uniformly.

    **partition-column guard.** when :attr:`schema` declares a
    partition column (one :class:`Column` flagged ``partition=True``),
    every public read / write method on the subclass must satisfy at
    least one of:

    1. accept the partition column as a parameter (positional or
       keyword) -- the SQL generator will then enforce the
       partition predicate naturally.
    2. be decorated with :func:`spans_partitions` -- the caller is
       deliberately fanning out across an authorized set of
       partition values supplied as a tuple.
    3. appear in :attr:`_partition_exempt_methods` -- a narrow,
       audited allowlist for methods that legitimately span every
       partition (e.g. ``count_total`` on a per-pod operational
       summary).

    a subclass that violates the guard fails at class-definition
    time with :class:`PartitionEnforcementError`. resolution priority:
    (1) add the partition column to the method signature; (2) decorate
    with :func:`spans_partitions`; (3) extend
    :attr:`_partition_exempt_methods` with a documented rationale.
    """

    schema: ClassVar[TableSchema]
    _partition_exempt_methods: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """validate the partition-column contract at class-definition time.

        executed once per subclass at import time. when
        :attr:`schema` declares a partition column, every public
        async method that is not inherited from the framework must
        either accept the partition column as a parameter, be
        decorated with :func:`spans_partitions`, or be listed in
        :attr:`_partition_exempt_methods`.

        framework-level methods inherited from
        :class:`BaseCollection` / :class:`SchemaBackedCollection`
        are exempt by construction (they thread the pk through
        :meth:`BaseCollection.normalize_pk` which already includes
        the partition column when the table has one). only methods
        defined directly on the subclass are checked.

        :param kwargs: PEP-487 keyword arguments forwarded to base
        :ptype kwargs: Any
        :return: nothing
        :rtype: None
        :raises PartitionEnforcementError: when a public subclass
            method violates the partition contract
        """
        super().__init_subclass__(**kwargs)
        schema = cls.__dict__.get("schema")
        if schema is None or not isinstance(schema, TableSchema):
            return None
        partition_column = schema.partition_column
        if partition_column is None:
            return None
        exempt = cls._partition_exempt_methods
        violations: list[str] = []
        for name, member in cls.__dict__.items():
            if not callable(member):
                continue
            if name.startswith("_"):
                continue
            if isinstance(member, (classmethod, staticmethod, property)):
                continue
            if name in exempt:
                continue
            if _is_partition_exempt(member):
                continue
            if _method_accepts_partition(member, partition_column):
                continue
            violations.append(name)
        if violations:
            raise PartitionEnforcementError(
                f"{cls.__name__}: schema declares partition column "
                f"{partition_column!r}; method(s) {violations!r} neither "
                f"accept it nor opt into @spans_partitions, and are not "
                f"in _partition_exempt_methods. add the partition "
                f"column to the signature, decorate the method with "
                f"@spans_partitions, or extend _partition_exempt_methods "
                f"with a documented rationale.",
            )
        return None

    # --- SQL generation ---

    def _build_where_pk(self, start: int = 1) -> tuple[str, tuple[str, ...]]:
        """build ``WHERE pk...`` fragment for fetch / delete / CAS.

        :param start: first positional-parameter index
        :ptype start: int
        :return: (where clause fragment, tuple of pk column names in
            declared order)
        :rtype: tuple[str, tuple[str, ...]]
        """
        pk_cols = self.schema.pk_columns
        parts = [f"{col} = ${start + i}" for i, col in enumerate(pk_cols)]
        return " AND ".join(parts), pk_cols

    def _render_param(self, column: Column, idx: int) -> str:
        """render ``$N`` with optional ``::jsonb`` / ``::vector`` cast.

        :param column: column descriptor
        :ptype column: Column
        :param idx: positional-parameter index (1-based)
        :ptype idx: int
        :return: placeholder string ready for SQL interpolation
        :rtype: str
        """
        if column.column_type == JSONB_TYPE:
            result = f"${idx}::jsonb"
        elif column.column_type == VECTOR_TYPE:
            result = f"${idx}::vector"
        else:
            result = f"${idx}"
        return result

    def _build_insert_sql(self) -> str:
        """build INSERT SQL with the conflict clause selected by schema.

        column order matches :attr:`TableSchema.columns` exactly; tests
        that assert on positional asyncpg parameters can rely on that.

        branches on :attr:`TableSchema.on_conflict`:

        * ``"raise"``: plain INSERT with no ON CONFLICT clause
        * ``"ignore"``: ``INSERT ... ON CONFLICT (pk) DO NOTHING``
        * ``"update"`` (default): ``INSERT ... ON CONFLICT (pk) DO
          UPDATE SET <mutable>``; falls back to ``DO NOTHING`` when
          every non-pk column is immutable so existing rows are not
          clobbered

        :return: parameterized INSERT SQL
        :rtype: str
        """
        schema = self.schema
        cols = schema.columns
        col_names = ", ".join(c.name for c in cols)
        placeholders = ", ".join(self._render_param(c, i + 1) for i, c in enumerate(cols))
        sql = f"INSERT INTO {schema.name} ({col_names}) VALUES ({placeholders})"
        pk_cols = ", ".join(schema.pk_columns)
        if schema.on_conflict == "raise":
            result = sql
        elif schema.on_conflict == "ignore":
            result = f"{sql} ON CONFLICT ({pk_cols}) DO NOTHING"
        else:
            mutable = schema.mutable_columns()
            if mutable:
                set_clause = ", ".join(f"{c.name} = EXCLUDED.{c.name}" for c in mutable)
                result = f"{sql} ON CONFLICT ({pk_cols}) DO UPDATE SET {set_clause}"
            else:
                # pk-only table or every non-pk column is immutable:
                # emit DO NOTHING so existing rows are not clobbered.
                result = f"{sql} ON CONFLICT ({pk_cols}) DO NOTHING"
        return result

    def _build_cas_update_sql(self) -> str:
        """build UPDATE SQL for the CAS (optimistic-concurrency) path.

        parameter ordering: pk values ($1..$Npk), then mutable columns
        in declared order, then the CAS fence value as the last
        parameter. callers match this layout in :meth:`save_to_postgres`.

        :return: parameterized UPDATE SQL
        :rtype: str
        :raises RuntimeError: when the schema has no :attr:`cas_column`
            (caller misuse -- CAS path is unreachable without a fence)
        """
        schema = self.schema
        if schema.cas_column is None:
            raise RuntimeError(
                f"TableSchema(name={schema.name!r}): CAS UPDATE requires "
                f"cas_column to be set",
            )
        pk_cols = schema.pk_columns
        mutable = schema.mutable_columns()
        # $1..$Npk are pk values; $Npk+1..$Npk+Nmut are mutable values
        pk_where = " AND ".join(f"{col} = ${i + 1}" for i, col in enumerate(pk_cols))
        set_parts: list[str] = []
        next_idx = len(pk_cols) + 1
        for col in mutable:
            set_parts.append(f"{col.name} = {self._render_param(col, next_idx)}")
            next_idx += 1
        # final $ is the CAS fence value
        cas_idx = next_idx
        set_clause = ", ".join(set_parts)
        result = (
            f"UPDATE {schema.name} SET {set_clause} "
            f"WHERE {pk_where} AND {schema.cas_column} = ${cas_idx}"
        )
        return result

    def _build_fetch_sql(self) -> str:
        """build SELECT SQL for pk-based fetch.

        :return: parameterized SELECT SQL
        :rtype: str
        """
        where, _ = self._build_where_pk(start=1)
        return f"SELECT * FROM {self.schema.name} WHERE {where}"

    def _build_delete_sql(self) -> str:
        """build DELETE SQL for pk-based delete.

        :return: parameterized DELETE SQL
        :rtype: str
        """
        where, _ = self._build_where_pk(start=1)
        return f"DELETE FROM {self.schema.name} WHERE {where}"

    # --- value normalization ---

    def _normalize_write_value(self, column: Column, value: Any) -> Any:
        """apply the column's write-boundary coercion.

        :param column: column descriptor
        :ptype column: Column
        :param value: raw value from the caller's data dict
        :ptype value: Any
        :return: coerced value ready to pass to asyncpg
        :rtype: Any
        """
        if value is None:
            result: Any = None
        elif column.column_type == UUID_TYPE:
            result = _coerce_uuid(value)
        elif column.column_type == DATETIME_TYPE:
            result = _coerce_datetime_for_write(value)
        elif column.column_type == DATETIMETZ_TYPE:
            result = _coerce_datetime_for_write_tz(value)
        elif column.column_type == JSONB_TYPE:
            result = _encode_jsonb(value)
        elif column.column_type == VECTOR_TYPE:
            result = _encode_vector(value)
        else:
            result = value
        return result

    def _normalize_read_value(self, column: Column, value: Any) -> Any:
        """apply the column's read-boundary coercion.

        :param column: column descriptor
        :ptype column: Column
        :param value: raw value from asyncpg
        :ptype value: Any
        :return: coerced value ready to hand back to callers
        :rtype: Any
        """
        if value is None:
            result: Any = None
        elif column.column_type == UUID_TYPE:
            # always coerce to stdlib UUID -- pgproto UUID is a subclass
            # that SQLite's binder rejects
            result = _coerce_uuid(value)
        elif column.column_type == DATETIMETZ_TYPE:
            # asyncpg's TIMESTAMPTZ decoder returns aware-UTC datetimes
            # already; the coercion is a defensive normalization that
            # also handles the edge case where a value flows in from a
            # non-asyncpg source (L2 JSON, hand-rolled proxy pool) and
            # arrives naive. result is always aware-UTC -- callers can
            # rely on a stable shape downstream.
            result = _coerce_datetime_for_read_tz(value)
        elif column.column_type == JSONB_TYPE:
            result = _decode_jsonb(value)
        elif column.column_type == VECTOR_TYPE:
            result = _decode_vector(value)
        else:
            result = value
        return result

    def _pull_value(self, data: dict[str, Any], column: Column) -> Any:
        """read a column's value from the caller's data dict.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :param column: column descriptor
        :ptype column: Column
        :return: raw value from the dict
        :rtype: Any
        :raises KeyError: when the column is required (``nullable=False``)
            and the key is missing from ``data``
        """
        if column.name in data:
            result: Any = data[column.name]
        elif column.nullable:
            result = None
        else:
            raise KeyError(
                f"{self.schema.name}.save_to_postgres: required column "
                f"{column.name!r} missing from data",
            )
        return result

    def _build_insert_params(self, data: dict[str, Any]) -> list[Any]:
        """build the parameter list for the INSERT path in declared order.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :return: parameter list matching :attr:`TableSchema.columns` order
        :rtype: list[Any]
        """
        return [self._normalize_write_value(c, self._pull_value(data, c)) for c in self.schema.columns]

    def _build_cas_params(
        self, data: dict[str, Any], original_timestamp: datetime,
    ) -> list[Any]:
        """build the parameter list for the CAS UPDATE path.

        layout: pk values in declared order, then mutable columns in
        declared order, then the CAS fence value last. matches the
        placeholder layout produced by :meth:`_build_cas_update_sql`.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: pre-mutation CAS fence value
        :ptype original_timestamp: datetime
        :return: parameter list matching the CAS UPDATE placeholder
            layout
        :rtype: list[Any]
        """
        schema = self.schema
        params: list[Any] = []
        for pk_name in schema.pk_columns:
            pk_col = schema.column(pk_name)
            params.append(self._normalize_write_value(pk_col, self._pull_value(data, pk_col)))
        for col in schema.mutable_columns():
            params.append(self._normalize_write_value(col, self._pull_value(data, col)))
        # CAS fence value: CAS column is typically date_updated; caller
        # supplies the pre-mutation value directly
        if schema.cas_column is None:
            # unreachable -- _build_cas_update_sql enforces the guard
            raise RuntimeError(
                f"TableSchema(name={schema.name!r}): CAS path reached without cas_column",
            )
        cas_col = schema.column(schema.cas_column)
        params.append(self._normalize_write_value(cas_col, original_timestamp))
        return params

    # --- BaseCollection contract implementations ---

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch one row from L3 by primary key.

        accepts scalar ``entity_id`` for single-pk tables or a tuple of
        pk values for composite-pk tables; shape is validated by
        :meth:`BaseCollection.normalize_pk`.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk)
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        result: dict[str, Any] | None
        if self.l3_pool is None:
            result = None
        else:
            pk_values = self.normalize_pk(entity_id)
            # coerce pk values uniformly (str -> UUID when column is
            # declared UUID, etc.)
            coerced: list[Any] = []
            for pk_name, raw in zip(self.schema.pk_columns, pk_values, strict=True):
                coerced.append(self._normalize_write_value(self.schema.column(pk_name), raw))
            row = await self.l3_pool.fetchrow(self._build_fetch_sql(), *coerced)
            if row is None:
                result = None
            else:
                result = self._coerce_row(dict(row))
        return result

    async def save_to_postgres(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """persist one row to L3.

        behaviour branches on schema configuration:

        * ``on_conflict="raise"``: INSERT only, no ``ON CONFLICT``;
          duplicate-key conflicts raise from asyncpg naturally
        * ``on_conflict="ignore"``: ``INSERT ... ON CONFLICT DO NOTHING``;
          duplicate-key conflicts silently dropped
        * ``cas_column`` set AND ``original_timestamp is not None`` AND
          ``on_conflict="update"``: emit UPDATE fenced on ``cas_column``;
          returns 0 on fence mismatch
          (:class:`~threetears.core.exceptions.ConcurrentModificationError`
          surfaces from the caller)
        * otherwise: upsert via ``INSERT ... ON CONFLICT (pk) DO UPDATE``

        :param data: row payload keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: pre-mutation CAS fence value; ignored
            when :attr:`TableSchema.cas_column` is ``None`` or when
            :attr:`TableSchema.on_conflict` is not ``"update"``
        :ptype original_timestamp: datetime | None
        :param conn: optional asyncpg-compatible connection that
            overrides :attr:`l3_pool` for this single write. when
            supplied, the INSERT/UPDATE binds to the caller's
            transaction so the write commits atomically with the
            caller's other operations on the same connection.
            ``None`` defers to the collection's own pool (legacy
            behaviour)
        :ptype conn: Any
        :return: rows affected reported by asyncpg (0 on CAS failure or
            ON CONFLICT DO NOTHING miss, 1 on success)
        :rtype: int
        """
        result: int
        executor: Any = conn if conn is not None else self.l3_pool
        if executor is None:
            result = 0
        else:
            schema = self.schema
            cas_eligible = (
                schema.cas_column is not None
                and original_timestamp is not None
                and schema.on_conflict == "update"
            )
            if cas_eligible:
                sql = self._build_cas_update_sql()
                params = self._build_cas_params(data, original_timestamp)  # type: ignore[arg-type]
            else:
                sql = self._build_insert_sql()
                params = self._build_insert_params(data)
            status = await executor.execute(sql, *params)
            result = self._parse_rowcount(status)
        return result

    async def delete_from_postgres(self, entity_id: Any) -> None:
        """delete one row from L3 by primary key.

        :param entity_id: pk value (single-pk) or tuple of pk values
            (composite-pk)
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is not None:
            pk_values = self.normalize_pk(entity_id)
            coerced: list[Any] = []
            for pk_name, raw in zip(self.schema.pk_columns, pk_values, strict=True):
                coerced.append(self._normalize_write_value(self.schema.column(pk_name), raw))
            await self.l3_pool.execute(self._build_delete_sql(), *coerced)

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize row dict to JSON bytes for L2 storage.

        :param data: row dict keyed by column name
        :ptype data: dict[str, Any]
        :return: UTF-8 JSON bytes
        :rtype: bytes
        """
        return json.dumps(data, default=_json_default).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize JSON bytes from L2 back to a row dict.

        :param data: UTF-8 JSON bytes produced by :meth:`serialize`
        :ptype data: bytes
        :return: row dict with typed values
        :rtype: dict[str, Any]
        """
        raw: dict[str, Any] = json.loads(data.decode("utf-8"))
        result: dict[str, Any] = {}
        for key, value in raw.items():
            col = self.schema.get_column(key)
            if col is None:
                result[key] = value
            else:
                result[key] = _decode_l2_value(col, value)
        return result

    # --- row / rowcount helpers ---

    def _coerce_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """apply read-boundary coercion to every known column in a row.

        columns not declared in the schema pass through untouched so
        ad-hoc joined projections still round-trip cleanly.

        :param row: row dict as returned by asyncpg
        :ptype row: dict[str, Any]
        :return: same-shape dict with coerced values
        :rtype: dict[str, Any]
        """
        result: dict[str, Any] = {}
        for key, value in row.items():
            col = self.schema.get_column(key)
            if col is None:
                result[key] = value
            else:
                result[key] = self._normalize_read_value(col, value)
        return result

    @staticmethod
    def _parse_rowcount(status: Any) -> int:
        """parse asyncpg's command-tag string into a row count.

        asyncpg returns a string like ``"INSERT 0 1"`` /
        ``"UPDATE 1"`` / ``"DELETE 1"``. empty / falsy values
        (mock pools sometimes return ``None`` or ``""``) resolve to 0.

        :param status: asyncpg status tag
        :ptype status: Any
        :return: rows-affected count
        :rtype: int
        """
        result: int
        if not status:
            result = 0
        elif isinstance(status, str):
            parts = status.split()
            if parts:
                try:
                    result = int(parts[-1])
                except ValueError:
                    result = 0
            else:
                result = 0
        else:
            result = 0
        return result
