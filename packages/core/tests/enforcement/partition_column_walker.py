"""3tears' partition-column rule: the packages it walks and the columns tied to their partitions.

The walker is :mod:`threetears.enforcement.partition_column` (one walker; the hub's gate runs it too). This module
binds 3tears' own configuration, for the enforcement test and its meta-test.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from pathlib import Path

from threetears.enforcement.no_whole_reads.schema import SchemaCatalog, collect_schema
from threetears.enforcement.partition_column import (
    PARTITIONED_TABLES,
    PartitionRule,
    tied_columns,
    unique_keys,
    walk_python_files,
)
from threetears.enforcement.partition_column import violations_in_file as rule_violations

__all__ = [
    "PACKAGE_SRC_ROOTS",
    "PARTITIONED_TABLES",
    "repo_tied_columns",
    "repo_unique_keys",
    "tied_columns",
    "unique_keys",
    "violations_in_file",
    "walk_python_files",
]

# packages to walk. additions go here when a package is brought under the partition-column doctrine; paths that do
# not exist are skipped by ``walk_python_files``.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
PACKAGE_SRC_ROOTS: list[Path] = [
    _REPO_ROOT / "packages" / "core" / "src",
    _REPO_ROOT / "packages" / "conversations" / "src",
    _REPO_ROOT / "packages" / "scheduled-jobs" / "src",
    _REPO_ROOT / "packages" / "agent" / "tools" / "src",
    _REPO_ROOT / "packages" / "agent" / "memory" / "src",
    _REPO_ROOT / "packages" / "agent" / "workspace" / "src",
    _REPO_ROOT / "packages" / "agent" / "skills" / "src",
    _REPO_ROOT / "packages" / "agent" / "wake" / "src",
    _REPO_ROOT / "packages" / "agent" / "acl" / "src",
    _REPO_ROOT / "packages" / "agent" / "audit" / "src",
]


@functools.cache
def _repo_catalog() -> SchemaCatalog:
    """the schema catalog of the walked packages' own declarations.

    :return: the catalog
    :rtype: SchemaCatalog
    """
    roots = tuple(root for root in PACKAGE_SRC_ROOTS if root.exists())
    return collect_schema(_REPO_ROOT, roots)


def repo_tied_columns() -> Mapping[str, frozenset[str]]:
    """:func:`tied_columns` over the walked packages' own declarations.

    :return: table onto its tied columns
    :rtype: Mapping[str, frozenset[str]]
    """
    return tied_columns(_repo_catalog())


def repo_unique_keys() -> Mapping[str, frozenset[frozenset[str]]]:
    """:func:`unique_keys` over the walked packages' own declarations.

    :return: table onto its unique keys
    :rtype: Mapping[str, frozenset[frozenset[str]]]
    """
    return unique_keys(_repo_catalog())


def violations_in_file(
    path: Path,
    tied: Mapping[str, frozenset[str]] | None = None,
    unique: Mapping[str, frozenset[frozenset[str]]] | None = None,
) -> tuple[list[str], list[str]]:
    """``(strict, deferred)`` violations in ``path`` under 3tears' rule.

    :param path: source file under inspection
    :ptype path: Path
    :param tied: the tied columns; None takes the walked packages' own (:func:`repo_tied_columns`)
    :ptype tied: Mapping[str, frozenset[str]] | None
    :param unique: the unique keys; None takes the walked packages' own (:func:`repo_unique_keys`)
    :ptype unique: Mapping[str, frozenset[frozenset[str]]] | None
    :return: the strict and the deferred violations
    :rtype: tuple[list[str], list[str]]
    """
    return rule_violations(
        path,
        PartitionRule(
            tied=repo_tied_columns() if tied is None else tied,
            unique=repo_unique_keys() if unique is None else unique,
        ),
    )
