"""enforcement: every 3tears collection that declares a ``TableSchema`` hands back, from every tier, what was written.

A tier that changes a value's type or loses its precision serves the changed value on every read it
answers: a ``NUMERIC`` read back from L1 as a ``float`` (``Decimal * float`` raises), an instant
without its zone. :func:`threetears.core.testing.tier_parity.tier_parity_findings` writes one row of
exacting values through the SQLite L1, the DuckDB L1 and the collection's L2 codec, and names the
collection, column and tier of every difference. This runs it over every collection class in the
workspace's source with a schema; the live half (the migrated Postgres table, and asyncpg's read of
it) runs beside the live parity tests of the packages whose tables 3tears' tests can migrate.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

from threetears.core.collections.base import NO_L2, BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import TableSchema
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.tier_parity import tier_parity_findings

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SOURCE_ROOTS = tuple(
    sorted(
        path
        for pattern in ("packages/*/src", "packages/agent/*/src")
        for path in _REPO_ROOT.glob(pattern)
        if (path.parent / "pyproject.toml").is_file()
    )
)


def _collections_with_a_schema() -> list[type[BaseCollection[Any]]]:
    """every collection class in the workspace's source that declares a ``TableSchema``, imported.

    :return: the classes, by qualified name
    :rtype: list[type[BaseCollection[Any]]]
    """
    for root in _SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "TableSchema" in path.read_text(encoding="utf-8") and "testing" not in path.parts:
                importlib.import_module(".".join(path.relative_to(root).with_suffix("").parts))
    found: dict[str, type[BaseCollection[Any]]] = {}
    pending: list[type[Any]] = [BaseCollection]
    while pending:
        for sub in pending.pop().__subclasses__():
            pending.append(sub)
            if sub.__module__.startswith("threetears.") and isinstance(getattr(sub, "schema", None), TableSchema):
                found[f"{sub.__module__}.{sub.__qualname__}"] = sub
    return [found[name] for name in sorted(found)]


def _build(cls: type[BaseCollection[Any]]) -> BaseCollection[Any]:
    """a collection of ``cls`` for its L2 codec, every other dependency its constructor requires left ``None``.

    Some collections take their pool or authorizer as a constructor argument; the codec uses none.

    :param cls: the collection class
    :ptype cls: type[BaseCollection[Any]]
    :return: the collection
    :rtype: BaseCollection[Any]
    """
    required = {
        name: None
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name not in {"self", "registry", "config", "nats_client"}
        and parameter.default is inspect.Parameter.empty
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    config = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return cls(registry=CollectionRegistry(), config=config, nats_client=NO_L2, **required)


def test_every_collection_hands_back_from_every_tier_what_was_written() -> None:
    collections = _collections_with_a_schema()
    names = {cls.__qualname__ for cls in collections}
    # the gate would pass vacuously on a discovery that found nothing
    assert {"CapabilitySourceCollection", "DataSourceColumnCollection", "McpToolGrantCollection"} <= names
    findings = tier_parity_findings(collections, build=_build)
    assert not findings, "\n".join(findings)
