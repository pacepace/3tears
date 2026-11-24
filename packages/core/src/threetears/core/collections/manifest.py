"""the startup loader: import every collection module the installed packages list, once, and record their tables.

**Why.** The class registries behind :func:`~threetears.core.collections.generation.tenancy_of_table`,
:func:`~threetears.core.collections.generation.tables_with_write_generation` and
:func:`~threetears.core.collections.generation.written_l2_first` are filled by class definition, so they were
complete only for the classes a process happened to import. The hub never imported ``MemoriesCollection``, and a
user merge's record of ``memories`` went out under ``_all`` (the hub fixed that call site in November 2026; this
fixes the cause).

**How.** Each package and app lists the modules that define its collection classes in its pyproject::

    [project.entry-points."threetears.collections"]
    "threetears.agent.memory.collections" = "threetears.agent.memory.collections"

An entry point whose modules need an extra names it (``"threetears.mcp.rbac [grants]"``); it is loaded only where
every distribution that extra requires is installed, so a package installed without the extra still starts.

:func:`load_declared_collections` imports every listed module once per process (``BuildOnce``) and records the
tables their collection classes name (:mod:`threetears.core.collections.undeclared_tables`). Each process calls it
where it starts, before it serves: the hub's app and its other serving processes, the SDK's runtime bootstrap, and
3tears' tool-pod bootstrap and registry server. A listed module that fails to import fails the start: a process
missing one would answer its tables from the fallback, the bug this exists to end.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from dataclasses import dataclass
from typing import Final

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.generation import table_named_by_class
from threetears.core.collections.undeclared_tables import record_declared_tables
from threetears.observe import BuildOnce, get_logger

__all__ = ["load_declared_collections"]

log = get_logger(__name__)

#: the entry-point group each package lists its collection modules under
COLLECTIONS_ENTRY_POINT_GROUP: Final = "threetears.collections"

#: a requirement's distribution name, at the start of its requirement string (PEP 508)
_REQUIREMENT_NAME: Final = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

#: the extra a requirement's marker names, e.g. ``extra == "grants"``
_EXTRA_MARKER: Final = re.compile(r"""extra\s*==\s*["']([^"']+)["']""")


class DeclaredCollectionsError(RuntimeError):
    """a module listed under the ``threetears.collections`` entry points could not be imported at startup."""


@dataclass(frozen=True)
class DeclaredCollections:
    """what the startup loader imported and recorded.

    :ivar modules: every listed module imported, in name order
    :ivar skipped: entry points left out because an extra they need is not installed, as ``name [extras]``
    :ivar tables: the tables the imported modules' collection classes name
    """

    modules: tuple[str, ...]
    skipped: tuple[str, ...]
    tables: frozenset[str]


def _canonical(name: str) -> str:
    """a distribution name in PEP 503 normal form.

    :param name: the name
    :ptype name: str
    :return: lower case, runs of ``-_.`` as one ``-``
    :rtype: str
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def _missing_for_extras(entry: importlib.metadata.EntryPoint, installed: frozenset[str]) -> tuple[str, ...]:
    """the distributions ``entry``'s extras require that are not installed.

    :param entry: the entry point
    :ptype entry: importlib.metadata.EntryPoint
    :param installed: every installed distribution's canonical name
    :ptype installed: frozenset[str]
    :return: the missing distributions' names; empty when the entry names no extra or all are installed
    :rtype: tuple[str, ...]
    """
    if not entry.extras or entry.dist is None:
        return ()
    wanted = set(entry.extras)
    missing: list[str] = []
    for requirement in entry.dist.requires or ():
        marker = _EXTRA_MARKER.search(requirement)
        name = _REQUIREMENT_NAME.match(requirement)
        if marker is None or name is None or marker.group(1) not in wanted:
            continue
        if _canonical(name.group(1)) not in installed:
            missing.append(name.group(1))
    return tuple(missing)


def _tables_of(module_name: str) -> frozenset[str]:
    """import ``module_name`` and name the tables its own collection classes declare.

    :param module_name: the listed module
    :ptype module_name: str
    :return: the table names its classes (defined in it, not imported into it) name on the class
    :rtype: frozenset[str]
    :raises DeclaredCollectionsError: when the module cannot be imported
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise DeclaredCollectionsError(
            f"\n\n  the collection module {module_name!r}, listed under the {COLLECTIONS_ENTRY_POINT_GROUP!r} entry "
            f"points, failed to import: {exc}\n"
            "  a process missing it answers its tables' tenancy and write generation from the fallback.\n"
            "  fix: install what it needs, or, when it needs an optional extra, name the extra on its entry point\n"
            '  ("module [extra]") so it loads only where the extra is installed.\n'
        ) from exc
    # every collection class the module made, read off the class tree rather than the module's names: a factory's
    # classes (reports' pod.tables builds one a table, held in a dict) are named for the module but bound to no
    # attribute of it, and a census of its attributes found none of them
    return frozenset(
        name
        for value in _collection_classes()
        if value.__module__ == module.__name__ and (name := table_named_by_class(value)) is not None
    )


def _collection_classes() -> tuple[type, ...]:
    """every subclass of :class:`BaseCollection` defined in this process, at any depth.

    :return: the classes, each once
    :rtype: tuple[type, ...]
    """
    seen: dict[type, None] = {}
    pending: list[type] = [BaseCollection]
    while pending:
        for sub in pending.pop().__subclasses__():
            if sub not in seen:
                seen[sub] = None
                pending.append(sub)
    return tuple(seen)


def _load() -> DeclaredCollections:
    """import every listed module and record the tables they declare; the build of :data:`_LOADED`.

    :return: what was imported and recorded
    :rtype: DeclaredCollections
    :raises DeclaredCollectionsError: when a listed module cannot be imported
    """
    installed = frozenset(
        _canonical(dist.metadata["Name"]) for dist in importlib.metadata.distributions() if dist.metadata["Name"]
    )
    modules: set[str] = set()
    skipped: list[str] = []
    for entry in importlib.metadata.entry_points(group=COLLECTIONS_ENTRY_POINT_GROUP):
        missing = _missing_for_extras(entry, installed)
        if missing:
            skipped.append(f"{entry.module} [{','.join(entry.extras)}]")
            continue
        modules.add(entry.module)
    tables: frozenset[str] = frozenset()
    for module_name in sorted(modules):
        tables = tables | _tables_of(module_name)
    recorded = record_declared_tables(tables)
    log.info(
        "declared collection modules loaded",
        extra={"extra_data": {"modules": len(modules), "tables": len(recorded), "skipped_for_extras": sorted(skipped)}},
    )
    return DeclaredCollections(modules=tuple(sorted(modules)), skipped=tuple(sorted(skipped)), tables=recorded)


#: the one load per process
_LOADED: BuildOnce[None, DeclaredCollections] = BuildOnce()


def load_declared_collections() -> DeclaredCollections:
    """import every module listed under ``threetears.collections`` once, and record the tables they declare.

    Call where the process starts, before it serves: the imports run on the calling thread, so never from the
    event loop once serving. Every call after the first returns the first one's result.

    :return: what was imported and recorded
    :rtype: DeclaredCollections
    :raises DeclaredCollectionsError: when a listed module cannot be imported
    """
    return _LOADED.get(None, _load)
