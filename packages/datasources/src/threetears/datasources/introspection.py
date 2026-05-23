"""Tier-2 change-detection primitives for the datasource introspector.

bundles the python-side cross-language hash helper (from
``datasource-task-02``) + the diff dataclass (from
``datasource-task-03``) + the pure diff-computation function. these
are the building blocks the Hub introspector calls into for the
change-driven introspection path (datasource-task-13).

byte-equivalence with the driver-side SQL is load-bearing:

- :func:`compute_column_hash` MUST produce the same MD5 hex digest
  as ``MD5(STRING_AGG(column_name || ':' || data_type || ':' ||
  COALESCE(is_nullable, ''), ',' ORDER BY ordinal_position))`` (the
  formula used by :class:`AsyncpgDriver` /
  :class:`RedshiftDriver`'s ``table_hashes`` SQL). the integration
  tests in ``tests/integration/test_*_driver_live.py`` already
  verify the cross-language invariant -- this module is the python
  side of that contract.

these primitives live in 3tears (not Hub) so future 3tears
consumers reuse them. shard 13's DS-13-14 flags this lift as a
follow-up; bringing them in here now makes the lift unnecessary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

__all__ = [
    "IntrospectionDiff",
    "compute_column_hash",
    "compute_introspection_diff",
]


def compute_column_hash(cols: list[dict[str, Any]]) -> str:
    """MD5 over the canonical column-shape payload (Tier-2 change probe).

    payload format (byte-identical to the driver-side SQL in
    ``_POSTGRES_TABLE_HASHES_SQL`` / ``_REDSHIFT_TABLE_HASHES_SQL``):

    .. code-block:: text

        column_name:data_type:is_nullable,column_name:data_type:is_nullable,...

    where columns are sorted by ``ordinal_position`` ascending,
    ``is_nullable`` is the RAW warehouse string (``'YES'`` / ``'NO'``
    / ``''``) -- NOT a python ``bool`` (the warehouse-side ``COALESCE
    (is_nullable, '')`` makes NULL render as the empty string, so
    we mirror with ``or ""`` here for the same byte output), and
    ``data_type`` is the raw warehouse string.

    :param cols: list of column-row dicts; each MUST carry
        ``column_name`` (str), ``data_type`` (str), ``is_nullable``
        (str | None), ``ordinal_position`` (int). extra keys are
        ignored. the function does NOT mutate ``cols``.
    :ptype cols: list[dict[str, Any]]
    :return: 32-char lowercase MD5 hex digest
    :rtype: str
    """
    payload = ",".join(
        f"{c['column_name']}:{c['data_type']}:{c.get('is_nullable') or ''}"
        for c in sorted(cols, key=lambda c: c["ordinal_position"])
    )
    return hashlib.md5(payload.encode()).hexdigest()  # noqa: S324 -- change-detect signal, not a security boundary


@dataclass(frozen=True)
class IntrospectionDiff:
    """result of comparing warehouse-side hashes against stored hashes.

    carries both the work lists the orchestrator acts on AND the
    summary counts the orchestrator returns to the caller for
    metrics. immutable so callers can pass the diff around without
    worrying about downstream mutation.

    work lists:

    :ivar tables_to_introspect: ``(schema, table)`` tuples whose
        column hash differs from storage, OR whose stored hash is
        NULL (treated as "force re-introspect"), OR which are
        present in the warehouse but absent from storage. the
        orchestrator fetches columns for these tables, recomputes
        the hash, and upserts both the table row + column rows
    :vartype tables_to_introspect: tuple[tuple[str, str], ...]
    :ivar tables_to_delete: ``(schema, table)`` tuples present in
        storage but absent from the warehouse probe. the
        orchestrator deletes these (table row + cascading column
        rows)
    :vartype tables_to_delete: tuple[tuple[str, str], ...]

    summary counts (filled by :func:`compute_introspection_diff` at
    diff-build time, or by the orchestrator as it walks the work
    lists):

    :ivar tables_checked: total number of tables in the warehouse
        probe (= ``len(warehouse_hashes)``)
    :vartype tables_checked: int
    :ivar tables_unchanged: warehouse hashes that matched the
        stored value (no work needed)
    :vartype tables_unchanged: int
    :ivar tables_changed: warehouse hashes that differed from
        non-null stored values
    :vartype tables_changed: int
    :ivar tables_added: present in warehouse, absent from storage
    :vartype tables_added: int
    :ivar tables_removed: present in storage, absent from warehouse
    :vartype tables_removed: int
    :ivar columns_added: per-column delta from the per-table
        re-introspect step; 0 from :func:`compute_introspection_diff`
        and filled by the orchestrator
    :vartype columns_added: int
    :ivar columns_removed: see ``columns_added``
    :vartype columns_removed: int
    :ivar columns_changed: see ``columns_added``
    :vartype columns_changed: int
    :ivar elapsed_ms: wall-clock elapsed time of the orchestrator
        call (probe + diff + per-table work); 0 from
        :func:`compute_introspection_diff` and filled by the
        orchestrator
    :vartype elapsed_ms: int
    """

    tables_to_introspect: tuple[tuple[str, str], ...]
    tables_to_delete: tuple[tuple[str, str], ...]
    tables_checked: int
    tables_unchanged: int
    tables_changed: int
    tables_added: int
    tables_removed: int
    columns_added: int = 0
    columns_removed: int = 0
    columns_changed: int = 0
    elapsed_ms: int = 0

    @property
    def has_changes(self) -> bool:
        """True iff the orchestrator has any work to do.

        :return: True iff either work list is non-empty
        :rtype: bool
        """
        return bool(self.tables_to_introspect) or bool(self.tables_to_delete)


def compute_introspection_diff(
    warehouse_hashes: dict[tuple[str, str], str],
    stored_hashes: dict[tuple[str, str], str | None],
) -> IntrospectionDiff:
    """compare warehouse-side hashes against stored ones; return the diff.

    pure function -- no I/O, no mutation of inputs. the orchestrator
    consumes the returned diff's work lists, performs the actual
    introspection / deletion + updates the columns_* + elapsed_ms
    counters before returning to its caller.

    classification rules:

    - present in both, hashes match: ``tables_unchanged += 1``
    - present in both, hashes differ: ``tables_changed += 1`` AND
      added to ``tables_to_introspect``
    - present in both, stored hash is None: ``tables_changed += 1``
      AND added to ``tables_to_introspect`` (NULL stored hash means
      "force re-introspect" -- this happens after the column_hash
      migration backfills NULL on existing rows)
    - present in warehouse only: ``tables_added += 1`` AND added
      to ``tables_to_introspect``
    - present in storage only: ``tables_removed += 1`` AND added
      to ``tables_to_delete``

    :param warehouse_hashes: ``{(schema, table): hash}`` from the
        driver's ``table_hashes`` call. hash is a non-null hex digest
    :ptype warehouse_hashes: dict[tuple[str, str], str]
    :param stored_hashes: ``{(schema, table): hash | None}`` from the
        Hub's ``datasource_tables.column_hash`` column. None signals
        "force re-introspect" (NULL stored hash post-migration)
    :ptype stored_hashes: dict[tuple[str, str], str | None]
    :return: the diff with work lists + table-level counts populated;
        columns_* + elapsed_ms remain 0 for the orchestrator to fill
    :rtype: IntrospectionDiff
    """
    to_introspect: list[tuple[str, str]] = []
    to_delete: list[tuple[str, str]] = []

    tables_unchanged = 0
    tables_changed = 0
    tables_added = 0
    tables_removed = 0

    warehouse_keys = set(warehouse_hashes.keys())
    stored_keys = set(stored_hashes.keys())

    for key in sorted(warehouse_keys & stored_keys):
        stored = stored_hashes[key]
        warehouse = warehouse_hashes[key]
        if stored is None or stored != warehouse:
            tables_changed += 1
            to_introspect.append(key)
        else:
            tables_unchanged += 1

    for key in sorted(warehouse_keys - stored_keys):
        tables_added += 1
        to_introspect.append(key)

    for key in sorted(stored_keys - warehouse_keys):
        tables_removed += 1
        to_delete.append(key)

    return IntrospectionDiff(
        tables_to_introspect=tuple(to_introspect),
        tables_to_delete=tuple(to_delete),
        tables_checked=len(warehouse_keys),
        tables_unchanged=tables_unchanged,
        tables_changed=tables_changed,
        tables_added=tables_added,
        tables_removed=tables_removed,
    )
