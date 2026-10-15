"""enforcement: a collection over a ``TableSchema`` and its schema agree about the write-generation switch.

The switch is the table's declaration (``TableSchema.write_generation``): the migration package that
creates the table renders the write-mark trigger from it, and the collection class reads its own
``write_generation`` from it. A class switched on over a schema that is not -- or the other way
round -- would advance a generation for writes no trigger marks, or mark writes nothing advances.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path

from threetears.core.collections.base import BaseCollection, tables_with_write_generation
from threetears.core.collections.generation import WriteGeneration
from threetears.core.collections.schema_backed import SchemaBackedCollection, TableSchema

_REPO_ROOT = Path(__file__).resolve().parents[2]

_SOURCE_ROOTS = tuple(
    sorted(
        path
        for pattern in ("packages/*/src", "packages/agent/*/src")
        for path in _REPO_ROOT.glob(pattern)
        if (path.parent / "pyproject.toml").is_file()
    )
)


def class_is_switched(cls: type[BaseCollection]) -> bool:
    """whether a collection class's writes advance its table's generation.

    :param cls: the class
    :ptype cls: type[BaseCollection]
    :return: ``True`` when it declares ``WRITE_GENERATION`` or caches absences
    :rtype: bool
    """
    return isinstance(cls.write_generation, WriteGeneration) or cls.negative_cache_max_age is not None


def disagreements(classes: Iterable[type[BaseCollection]]) -> list[str]:
    """the classes over a ``TableSchema`` whose switch and whose schema's switch differ.

    :param classes: collection classes
    :ptype classes: Iterable[type[BaseCollection]]
    :return: ``Class: class says X, schema says Y`` for each
    :rtype: list[str]
    """
    found: list[str] = []
    for cls in classes:
        schema = getattr(cls, "schema", None)
        if not isinstance(schema, TableSchema):
            continue
        schema_on = isinstance(schema.write_generation, WriteGeneration)
        if class_is_switched(cls) != schema_on:
            found.append(f"{cls.__qualname__}: class says {class_is_switched(cls)}, schema says {schema_on}")
    return found


def _schema_backed_classes() -> list[type[BaseCollection]]:
    """every ``SchemaBackedCollection`` subclass the source defines, imported.

    :return: the classes
    :rtype: list[type[BaseCollection]]
    """
    for root in _SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            if "SchemaBackedCollection" in path.read_text(encoding="utf-8"):
                importlib.import_module(".".join(path.relative_to(root).with_suffix("").parts))
    found: list[type[BaseCollection]] = []
    pending: list[type] = [SchemaBackedCollection]
    while pending:
        for sub in pending.pop().__subclasses__():
            found.append(sub)
            pending.append(sub)
    return [cls for cls in found if cls.__module__.startswith("threetears.")]


def test_every_schema_backed_class_and_its_schema_agree_on_the_switch() -> None:
    classes = _schema_backed_classes()
    assert any(getattr(c, "schema", None) is not None for c in classes), "no schema-backed class found"
    assert "coordination_revocations" in tables_with_write_generation()
    assert not disagreements(classes)


def test_a_class_and_schema_that_disagree_are_reported() -> None:
    """a class that switches itself on over a schema that is not, and the reverse, are both named."""
    from threetears.core.collections.generation import WRITE_GENERATION, WRITE_GENERATION_UNDECLARED
    from threetears.core.collections.schema_backed import Column

    off = TableSchema(name="t_off", primary_key="id", columns=[Column("id", "string")])
    on = TableSchema(name="t_on", primary_key="id", columns=[Column("id", "string")], write_generation=WRITE_GENERATION)

    # stand-ins with a collection's switch attributes, never registered as collections
    class SelfSwitched:
        schema = off
        write_generation = WRITE_GENERATION
        negative_cache_max_age = None

    class NotReadingItsSchema:
        schema = on
        write_generation = WRITE_GENERATION_UNDECLARED
        negative_cache_max_age = None

    class Agrees:
        schema = on
        write_generation = WRITE_GENERATION
        negative_cache_max_age = None

    found = disagreements([SelfSwitched, NotReadingItsSchema, Agrees])  # type: ignore[list-item]
    assert [entry.split(":")[0].split(".")[-1] for entry in found] == ["SelfSwitched", "NotReadingItsSchema"]
