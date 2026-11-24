"""which tables this process's listed collection modules declare, and the report of a lookup for one they do not.

The class registries in :mod:`threetears.core.collections.generation` (tenancy, write generation, L2-first) know
only the classes this process imported. A process that never imported a table's class answered for it from the
fallback: the hub never imported ``MemoriesCollection``, so a user merge's record of ``memories`` went out under
``_all`` instead of the agent's tenant. :func:`~threetears.core.collections.manifest.load_declared_collections`
imports every module the installed packages list under the ``threetears.collections`` entry-point group, once at
process start, and records here the tables they declare.

A registry lookup for a table no listed module declares (:func:`report_undeclared_table`) keeps the fallback this
release, but logs at ERROR once per table and lookup and counts ``threetears_undeclared_table_total{table, lookup}``.
NEXT RELEASE: the fallback becomes a refusal (``SchemaTenantUnknownError``-style raise) -- the test
``test_an_undeclared_table_is_refused`` in ``packages/core/tests/unit/collections/test_manifest.py`` is marked for it.

A leaf module (observe only): :mod:`generation` reads it on every fallback, and the loader, which imports the
collection modules and so :mod:`generation` itself, writes it.
"""

from __future__ import annotations

from typing import Final

from threetears.observe import BuildOnce, get_logger
from threetears.observe.metrics import counter

__all__ = [
    "record_declared_tables",
    "report_undeclared_table",
]

log = get_logger(__name__)

_UNDECLARED: Final = counter(
    "threetears_undeclared_table_total",
    description=(
        "registry lookups (tenancy, write generation, L2-first) for a table no module listed under the "
        "threetears.collections entry points declares; counted once per table and lookup per process"
    ),
    label_names=("table", "lookup"),
)

#: the tables the listed collection modules declare, set once by the startup loader
_DECLARED: BuildOnce[None, frozenset[str]] = BuildOnce()

#: each (table, lookup) already reported, so each logs and counts once per process
_REPORTED: BuildOnce[tuple[str, str], bool] = BuildOnce()


def record_declared_tables(tables: frozenset[str]) -> frozenset[str]:
    """record the tables the listed collection modules declare; the first record stands.

    :param tables: the declared table names
    :ptype tables: frozenset[str]
    :return: the recorded set (the first one recorded in this process)
    :rtype: frozenset[str]
    """
    return _DECLARED.get(None, lambda: tables)


def _report(table: str, lookup: str, loaded: bool) -> bool:
    """log and count one undeclared lookup; the build of :data:`_REPORTED`.

    :param table: the table looked up
    :ptype table: str
    :param lookup: which registry: ``tenancy``, ``write_generation`` or ``l2_first``
    :ptype lookup: str
    :param loaded: whether the startup loader ran in this process
    :ptype loaded: bool
    :return: ``True``
    :rtype: bool
    """
    why = (
        "no module listed under the threetears.collections entry points declares it"
        if loaded
        else "load_declared_collections() never ran in this process"
    )
    log.error(
        f"collection registry lookup for an undeclared table {table!r} ({lookup}): {why}; answering with the "
        "fallback this release, refused from the next. List the module defining its collection class in its "
        'package\'s pyproject [project.entry-points."threetears.collections"], and call '
        "threetears.core.collections.load_declared_collections() where the process starts.",
        extra={"extra_data": {"table": table, "lookup": lookup, "loader_ran": loaded}},
    )
    _UNDECLARED.labels(table=table, lookup=lookup).inc()
    return True


def report_undeclared_table(table: str, lookup: str) -> None:
    """report a registry lookup for ``table`` when no listed module declares it: ERROR and count, once.

    Called on a registry's fallback path only; a declared table (one whose class simply does not switch the
    registry's feature on) passes silently.

    :param table: the table looked up
    :ptype table: str
    :param lookup: which registry: ``tenancy``, ``write_generation`` or ``l2_first``
    :ptype lookup: str
    :return: nothing
    :rtype: None
    """
    declared = _DECLARED.peek(None)
    if declared is not None and table in declared:
        return
    _REPORTED.get((table, lookup), lambda: _report(table, lookup, declared is not None))
