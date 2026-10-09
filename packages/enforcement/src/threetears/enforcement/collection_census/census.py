"""every ``BaseCollection`` subclass a set of source trees defines, with its table and its declaration.

Run in a process of its own (``python -m threetears.enforcement.collection_census``), by
:func:`~threetears.enforcement.collection_census.run_census`: it imports every module under the
trees it is given, which a test process should not do to itself. It prints one JSON object.

For each class:

- ``table``: the table it names, when that can be read off the class without building it -- a
  ``schema`` whose ``name`` is a string, or a ``table_name`` property that answers with the class
  standing in for an instance (a literal, or a class attribute). ``None`` when the table is named
  per instance (a constructor argument, a scope) or the class names none of its own.
- ``abstract``: whether the class leaves an abstract method unimplemented.
- ``declaration``: ``on``, ``opted_out``, ``undeclared``, or ``invalid``, read off
  ``write_generation`` as the class resolves it; ``reason`` is an opt-out's reason.
- ``ancestors``: every other census class it subclasses, by qualified name.
- ``framework``: whether the class is the 3tears framework's rather than the trees' own.

**Across repositories.** With ``--framework``, every module of the installed ``threetears``
packages is imported too, and their collection classes join the census marked ``framework``. A
product's census then sees a class of its own that names a table the framework already has a class
for -- the second class for one table that no census inside either repository can see. A framework
module that does not import in the product's environment (an extra it does not install) is listed
under ``framework_import_failures``; its classes cannot be compared, and a product that uses one
imports it from its own trees, which brings it into the census anyway.

Usage: ``python -m threetears.enforcement.collection_census [--framework] <src root> [<src root> ...]``
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

__all__ = ["census", "main"]

#: the namespace every framework package lives under.
_FRAMEWORK_PACKAGE = "threetears"


def _module_names(root: Path) -> list[str]:
    """every importable module under one source root, ``__main__`` modules left out.

    :param root: a package source root (``packages/<name>/src``, or a product's ``src``)
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
        if parts:
            names.append(".".join(parts))
    return names


def _framework_module_names() -> list[str]:
    """every module of the installed ``threetears`` packages, ``__main__`` modules left out.

    :return: dotted module names, sorted
    :rtype: list[str]
    """
    # ``threetears`` and ``threetears.agent`` are namespace packages spread over every installed
    # distribution's source tree, which ``pkgutil`` does not descend into; the files are read instead.
    package = importlib.import_module(_FRAMEWORK_PACKAGE)
    names = {
        name
        for directory in package.__path__
        for name in _module_names(Path(directory).parent)
        if name.startswith(f"{_FRAMEWORK_PACKAGE}.") and ".tests" not in name
    }
    return sorted(names)


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
            # NOSILENT: a getter that raises for the class standing in for an instance is the answer
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


def _import_all(names: list[str], failures: list[str]) -> set[str]:
    """import each module, recording why any did not.

    :param names: dotted module names
    :ptype names: list[str]
    :param failures: where a failure is recorded, as ``module: Type: message``
    :ptype failures: list[str]
    :return: the modules that imported
    :rtype: set[str]
    """
    imported: set[str] = set()
    for name in names:
        try:
            importlib.import_module(name)
        # prawduct:allow prawduct/broad-except -- every import failure is reported to the caller, which decides
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        imported.add(name)
    return imported


def census(roots: list[Path], *, framework: bool = False) -> dict[str, Any]:
    """import every module under ``roots`` and describe every collection class they define.

    :param roots: package source roots
    :ptype roots: list[Path]
    :param framework: whether the installed framework's classes join the census too
    :ptype framework: bool
    :return: ``classes`` (one record per class, sorted by name), ``import_failures`` (the trees'
        modules that did not import) and ``framework_import_failures``
    :rtype: dict[str, Any]
    """
    failures: list[str] = []
    framework_failures: list[str] = []
    for root in roots:
        sys.path.insert(0, str(root))
    modules: set[str] = set()
    for root in roots:
        modules |= _import_all(_module_names(root), failures)
    framework_modules: set[str] = set()
    if framework:
        framework_modules = _import_all([n for n in _framework_module_names() if n not in modules], framework_failures)

    from threetears.core.collections.base import BaseCollection

    found: list[type] = []
    pending: list[type] = [BaseCollection]
    while pending:
        for sub in pending.pop().__subclasses__():
            if sub not in found:
                found.append(sub)
                pending.append(sub)
    members = [cls for cls in found if cls.__module__ in modules or cls.__module__ in framework_modules]
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
                "framework": cls.__module__ not in modules,
            }
        )
    return {
        "classes": sorted(records, key=lambda record: str(record["name"])),
        "import_failures": failures,
        "framework_import_failures": framework_failures,
    }


def main(argv: list[str]) -> dict[str, Any]:
    """the census the command line asks for.

    :param argv: ``[--framework] <src root> ...``
    :ptype argv: list[str]
    :return: the census
    :rtype: dict[str, Any]
    """
    framework = "--framework" in argv
    return census([Path(arg) for arg in argv if arg != "--framework"], framework=framework)
