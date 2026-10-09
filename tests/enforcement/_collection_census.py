"""every ``BaseCollection`` subclass a set of source trees defines, with its table and its declaration.

Run as a script, in a process of its own, by ``test_write_generation_declarations.py``: it imports
every module under the trees it is given, which the test process should not do to itself. It
prints one JSON object per class to stdout.

For each class:

- ``table``: the table it names, when that can be read off the class without building it -- a
  ``schema`` whose ``name`` is a string, or a ``table_name`` property that answers with the class
  standing in for an instance (a literal, or a class attribute). ``None`` when the table is named
  per instance (a constructor argument, a scope) or the class names none of its own.
- ``abstract``: whether the class leaves an abstract method unimplemented.
- ``declaration``: ``on``, ``opted_out``, ``undeclared``, or ``invalid``, read off
  ``write_generation`` as the class resolves it; ``reason`` is an opt-out's reason.
- ``ancestors``: every other census class it subclasses, by qualified name.

Usage: ``python _collection_census.py <src root> [<src root> ...]``
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any


def _module_names(root: Path) -> list[str]:
    """every importable module under one source root, ``__main__`` modules left out.

    :param root: a package source root (``packages/<name>/src``)
    :ptype root: Path
    :return: dotted module names, sorted
    :rtype: list[str]
    """
    names: list[str] = []
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[-1] == "__main__":
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def _qualified(cls: type) -> str:
    """``module.QualName`` for a class.

    :param cls: the class
    :ptype cls: type
    :return: its qualified name
    :rtype: str
    """
    return f"{cls.__module__}.{cls.__qualname__}"


def _table_of(cls: type) -> str | None:
    """the table a class names without being built, or ``None``.

    :param cls: a collection class
    :ptype cls: type
    :return: the table name, or ``None`` when it is named per instance or not at all
    :rtype: str | None
    """
    schema = getattr(cls, "schema", None)
    name = getattr(schema, "name", None)
    if isinstance(name, str):
        return name
    prop = inspect.getattr_static(cls, "table_name", None)
    if isinstance(prop, property) and prop.fget is not None:
        try:
            # the class stands in for an instance: a literal or a class attribute answers, and a
            # table named per instance raises, which is what "not readable off the class" means
            answer: Any = prop.fget(cls)
        # prawduct:allow prawduct/broad-except -- any failure means the table is not readable off the class
        except Exception:  # noqa: BLE001
            return None
        return answer if isinstance(answer, str) else None
    return None


def _declaration(cls: type) -> tuple[str, str | None]:
    """which write-generation declaration a class carries, and an opt-out's reason.

    :param cls: a collection class
    :ptype cls: type
    :return: ``(kind, reason)``
    :rtype: tuple[str, str | None]
    """
    from threetears.core.collections.generation import (
        NoWriteGeneration,
        UndeclaredWriteGeneration,
        WriteGeneration,
    )

    declared = getattr(cls, "write_generation", None)
    if isinstance(declared, WriteGeneration):
        return "on", None
    if isinstance(declared, NoWriteGeneration):
        return "opted_out", declared.reason
    if isinstance(declared, UndeclaredWriteGeneration):
        return "undeclared", None
    return "invalid", None


def census(roots: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """import every module under ``roots`` and describe every collection class they define.

    :param roots: package source roots
    :ptype roots: list[Path]
    :return: one record per class, and every module that failed to import with why
    :rtype: tuple[list[dict[str, Any]], list[str]]
    """
    failures: list[str] = []
    modules: set[str] = set()
    for root in roots:
        sys.path.insert(0, str(root))
    for root in roots:
        for name in _module_names(root):
            try:
                importlib.import_module(name)
            # prawduct:allow prawduct/broad-except -- every import failure is reported to the test, which fails on it
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            modules.add(name)

    from threetears.core.collections.base import BaseCollection

    found: list[type] = []
    pending = [BaseCollection]
    while pending:
        for sub in pending.pop().__subclasses__():
            if sub not in found:
                found.append(sub)
                pending.append(sub)
    members = [cls for cls in found if cls.__module__ in modules]
    records: list[dict[str, Any]] = []
    for cls in members:
        kind, reason = _declaration(cls)
        records.append(
            {
                "name": _qualified(cls),
                "table": _table_of(cls),
                "abstract": inspect.isabstract(cls),
                "declaration": kind,
                "reason": reason,
                "ancestors": sorted(
                    _qualified(other) for other in members if other is not cls and issubclass(cls, other)
                ),
            }
        )
    return sorted(records, key=lambda record: str(record["name"])), failures


if __name__ == "__main__":
    described, failed = census([Path(arg) for arg in sys.argv[1:]])
    json.dump({"classes": described, "import_failures": failed}, sys.stdout)
