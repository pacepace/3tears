"""which tenant a table's change records belong to, so a pod reads only its own tenants' changes.

The epoch change log (``{ns}-epoch-changes``) keys every record by tenant first, then table:
``{tenant}.{table}.e``. A pod's grant names its own tenants' keys (its agent's namespace, and the platform
tables its role already reads for its customer and for ``_all``), so it reads only the records of the rows
it may read, whose values (``ChangedRow.value``) are then no more readable than the collections bucket's.
Platform services (the hub, the registry, the gateway) read every tenant through the wildcard
``*.{table}.e``.

**A table declares its tenancy** (:attr:`BaseCollection.tenancy`, or ``TableSchema.tenancy``), never
guessed:

- :data:`TENANT_BY_SCHEMA`: an agent-schema table: the tenant is the schema the row lives in;
- :func:`tenant_column`: a platform table with a tenant column (the customer an RLS policy names): the
  tenant is that column's value. The column joins the class's invalidation columns, so every record and
  note carries it;
- :func:`no_tenant`: a platform-wide table, with the reason: one ``_all`` key every follower reads.

A class that declares none is :data:`TENANCY_UNDECLARED`: its records go on ``_all``, which is correct
for every follower but isolates nothing; the collection census lists it. A whole-table change (a
truncate, a file load) or one whose keys are unknown always goes on ``_all``: it spans tenants, and
carries no row.

**Detection is unchanged.** A follower walks each key it may read after its one mark per table, merged by
the log's sequence; the token is still the log's global sequence. An interior record of a live log cannot
disappear (the stream denies deletes and refuses a full append), so a lost record is detected per key the
way it was per table: by the log's first sequence (``aged_out``) and its incarnation.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

if TYPE_CHECKING:
    from threetears.core.collections.epoch_change import ChangedRow

__all__ = [
    "ALL_TENANTS",
    "TENANCY_UNDECLARED",
    "TENANT_BY_SCHEMA",
    "Tenancy",
    "epoch_change_filter",
    "epoch_change_key",
    "no_tenant",
    "rows_with_tenants",
    "schema_tenant",
    "split_epoch_change_key",
    "tenant_column",
    "value_tenant",
]

#: the tenant every follower of a table reads: platform-wide tables, whole-table and keys-unknown changes
ALL_TENANTS: Final = "_all"

#: a tenant token: one subject token the grants can name
_TOKEN: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]{1,62}$")

#: a table as the key's middle: subject tokens of identifier characters
_TABLE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]+(\.[A-Za-z0-9_-]+)*$")

#: the token that ends every record key, so a grant on ``{tenant}.{table}.>`` names exactly one table
_SUFFIX: Final = "e"

#: the digest length a token that cannot be a subject token is replaced by
_DIGEST: Final = 24


@dataclass(frozen=True, slots=True)
class Tenancy:
    """how a table's change records are partitioned by tenant.

    :ivar kind: ``schema``, ``column``, ``none`` or ``undeclared``
    :ivar column: the tenant column, for ``column``
    :ivar reason: why the table has no tenant, for ``none``
    """

    kind: Literal["schema", "column", "none", "undeclared"]
    column: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        """refuse a declaration missing what it needs.

        :return: nothing
        :rtype: None
        :raises ValueError: a column tenancy without a column, or a ``none`` without a reason
        """
        if self.kind == "column" and (not self.column or not _TABLE.match(self.column)):
            raise ValueError(f"a column tenancy names its tenant column; got {self.column!r}")
        if self.kind == "none" and (not isinstance(self.reason, str) or not self.reason.strip()):
            raise ValueError("no_tenant needs a reason: say why this table's changes are platform-wide")


#: an agent-schema table: the tenant is the schema the row lives in
TENANT_BY_SCHEMA: Final = Tenancy("schema")

#: a table that declares no tenancy: its records go on ``_all``, and the census lists it
TENANCY_UNDECLARED: Final = Tenancy("undeclared")


def tenant_column(column: str) -> Tenancy:
    """a platform table whose tenant is ``column``'s value (the customer an RLS policy names).

    :param column: the tenant column
    :ptype column: str
    :return: the declaration
    :rtype: Tenancy
    """
    return Tenancy("column", column=column)


def no_tenant(reason: str) -> Tenancy:
    """a platform-wide table: every follower of it reads its one ``_all`` key.

    :param reason: why its changes belong to no tenant
    :ptype reason: str
    :return: the declaration
    :rtype: Tenancy
    """
    return Tenancy("none", reason=reason)


def _token(prefix: str, value: str) -> str:
    """``{prefix}-{value}`` when that is one subject token, otherwise ``h-`` and a digest of ``value``.

    The same rule as the change-note trigger's (:func:`threetears.core.ddl.change_notes.tenant_sql`), so a
    note's tenant and a record's agree.

    :param prefix: the token's kind
    :ptype prefix: str
    :param value: the schema or column value
    :ptype value: str
    :return: the token
    :rtype: str
    """
    token = f"{prefix}-{value}"
    if _TOKEN.match(token):
        return token
    return f"h-{hashlib.sha256(value.encode()).hexdigest()[:_DIGEST]}"


def schema_tenant(schema: str) -> str:
    """the tenant of a row in ``schema`` (an agent's namespace schema).

    :param schema: the schema
    :ptype schema: str
    :return: the tenant token
    :rtype: str
    """
    return _token("s", schema)


def value_tenant(value: str) -> str:
    """the tenant of a row whose tenant column holds ``value`` (as text).

    :param value: the column's value
    :ptype value: str
    :return: the tenant token
    :rtype: str
    """
    return _token("v", value)


def _row_tenant(tenancy: Tenancy, columns: Mapping[str, str | None] | None) -> str | None:
    """the tenant of one changed row, from its invalidation columns; ``None`` when it is ``_all``'s.

    A schema tenancy is named by whoever knows the row's schema (the writer that ran the statement),
    never from the row: ``None`` here.

    :param tenancy: the table's declaration
    :ptype tenancy: Tenancy
    :param columns: the row's invalidation columns
    :ptype columns: Mapping[str, str | None] | None
    :return: the tenant token, or ``None``
    :rtype: str | None
    """
    if tenancy.kind != "column" or tenancy.column is None or not columns:
        return None
    value = columns.get(tenancy.column)
    return None if value is None else value_tenant(str(value))


def _table_segment(table: str) -> str:
    """the table as the key's middle: verbatim when it is subject tokens, otherwise a digest.

    :param table: the table
    :ptype table: str
    :return: the segment
    :rtype: str
    """
    if _TABLE.match(table):
        return table
    return f"d-{hashlib.sha256(table.encode()).hexdigest()[:_DIGEST]}"


def epoch_change_key(table: str, tenant: str | None) -> str:
    """the change-log key a record of ``table`` for ``tenant`` is appended under.

    :param table: the table
    :ptype table: str
    :param tenant: the tenant token; ``None`` for ``_all``
    :ptype tenant: str | None
    :return: ``{tenant}.{table}.e``
    :rtype: str
    :raises ValueError: when ``tenant`` is not a subject token
    """
    owner = tenant or ALL_TENANTS
    if not _TOKEN.match(owner):
        raise ValueError(f"a tenant is one subject token of {_TOKEN.pattern}; got {owner!r}")
    return f"{owner}.{_table_segment(table)}.{_SUFFIX}"


def epoch_change_filter(table: str) -> str:
    """the subject filter of every tenant's records of ``table``, for a platform reader.

    :param table: the table
    :ptype table: str
    :return: ``*.{table}.e``
    :rtype: str
    """
    return f"*.{_table_segment(table)}.{_SUFFIX}"


def split_epoch_change_key(key: str) -> tuple[str, str] | None:
    """the tenant and the table segment a record key names, or ``None`` for a key of another shape.

    :param key: the key
    :ptype key: str
    :return: ``(tenant, table segment)``
    :rtype: tuple[str, str] | None
    """
    tenant, _, rest = key.partition(".")
    table, _, suffix = rest.rpartition(".")
    if not tenant or not table or suffix != _SUFFIX:
        return None
    return tenant, table


def rows_with_tenants(tenancy: Tenancy, rows: Sequence[ChangedRow] | None) -> Sequence[ChangedRow] | None:
    """``rows`` with each row's tenant named from ``tenancy``'s tenant column.

    A row whose tenant its writer already named keeps it (a schema tenancy, named by whoever ran the
    statement in that schema); a whole-table change has no rows, and goes on ``_all``.

    :param tenancy: the table's declaration
    :ptype tenancy: Tenancy
    :param rows: the rows, or ``None`` for the whole table
    :ptype rows: Sequence[ChangedRow] | None
    :return: the rows, each with its tenant
    :rtype: Sequence[ChangedRow] | None
    """
    if rows is None or tenancy.kind != "column":
        return rows
    return [
        row if row.tenant is not None else row.model_copy(update={"tenant": _row_tenant(tenancy, row.columns)})
        for row in rows
    ]
