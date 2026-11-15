"""Record what a host's production code uses of ``3tears-evals``, from a checkout of the host.

Dev tooling for ``3tears-evals``: it lives beside the package, not in it, and is not installed. It writes
``tests/fixtures/host_imports/<host>.txt``, which ``tests/test_host_imports_are_declared.py`` holds to the declared
surface and ``tests/test_declared_holdbacks.py`` reads as the host's use of each name (admission criterion 2), and
``tests/fixtures/host_test_imports/<host>.txt``, the host's tests' use, which ``tests/test_declared_holdbacks.py``
reads for a kit of the ``tests`` kind only (owner ruling R-a, 2026-11-15: a kit a host runs in its own tests is used
there). Both files are generated, never edited by hand: re-run this against the host's newest develop.

    uv run python packages/evals/scripts/record_host_imports.py discodon /path/to/discodon/checkout

It reads the host's committed tree at ``HEAD`` (``git ls-files``, ``git show``), never the working tree, and changes
nothing there.

**What counts as production.** The host's tracked Python files, less its tests and its planning notes:

- A test is a file under ``tests/``, a ``conftest.py``, or a ``test_*.py`` or ``*_test.py``. A test's imports never
  count: a name only a host's tests import is not used (owner ruling, 2026-11-15: "a test alone does not count").
- A file under a top-level directory whose name starts with a dot (``.prawduct/``, ``.claude/``) is a planning note
  or an agent's tooling, not code the host runs.
- A script under ``tools/`` counts only when a file that is neither a test nor such a note names it (by its file
  name or its module path): an operator command the host's code or its documentation points at. A tool only tests
  name, or that nothing names, is not production.
- Everything else (the host's package source) counts.

**What counts as use.** Each ``from threetears.evals… import name``, each ``import threetears.evals…`` with every
attribute read off the module it binds, and each attribute read off a module a ``from`` statement binds
(``from threetears.evals import kernel`` then ``kernel.name``). Lines are fully qualified ``root.name`` as written in
the host's code; a line naming a module is kept, and the test reading this file skips it.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import re
import subprocess
from pathlib import Path, PurePosixPath

#: The package directory, whose fixtures this writes.
PACKAGE = Path(__file__).resolve().parents[1]

#: Where each host's list is written.
HOST_IMPORTS = PACKAGE / "tests" / "fixtures" / "host_imports"

#: Where each host's tests' list is written.
HOST_TEST_IMPORTS = PACKAGE / "tests" / "fixtures" / "host_test_imports"

PREFIX = "threetears.evals"


def _git(checkout: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True, text=True).stdout


def is_test(path: str) -> bool:
    """Whether a host's file is a test: under ``tests/``, a ``conftest.py``, a ``test_*.py`` or ``*_test.py``."""
    parts = PurePosixPath(path).parts
    name = parts[-1]
    return (
        "tests" in parts[:-1]
        or name == "conftest.py"
        or (name.startswith("test_") and name.endswith(".py"))
        or name.endswith("_test.py")
    )


def is_note(path: str) -> bool:
    """Whether a host's file is a planning note or agent tooling: under a top-level dot directory."""
    return PurePosixPath(path).parts[0].startswith(".")


def is_module(name: str) -> bool:
    """Whether ``name`` is an importable module (``threetears.evals.kernel``), not a name in one."""
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def used_names(source: str, filename: str) -> set[str]:
    """Every ``threetears.evals`` name one file imports or reads off an imported module."""
    tree = ast.parse(source, filename=filename)
    found: set[str] = set()
    modules: dict[str, str] = {}  # a local name bound to a package module -> that module
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(PREFIX) and node.level == 0:
            for alias in node.names:
                qualified = f"{node.module}.{alias.name}"
                found.add(qualified)
                if is_module(qualified):
                    modules[alias.asname or alias.name] = qualified
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PREFIX):
                    found.add(alias.name)
                    if alias.asname:
                        modules[alias.asname] = alias.name
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base = dotted(node.value)
        if base is None:
            continue
        head, _, rest = base.partition(".")
        module = f"{modules[head]}{'.' + rest if rest else ''}" if head in modules else base
        if module.startswith(PREFIX) and is_module(module):
            found.add(f"{module}.{node.attr}")
    return found


def dotted(node: ast.expr) -> str | None:
    """A ``Name`` or a chain of attributes on one, as dotted text (``kernel.host``); ``None`` for anything else."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and (base := dotted(node.value)) is not None:
        return f"{base}.{node.attr}"
    return None


def production_files(checkout: Path) -> list[str]:
    """The host's tracked Python files that are production by the rule above, at ``HEAD``."""
    tracked = _git(checkout, "ls-files", "--", "*.py").split()
    candidates = [path for path in tracked if not is_test(path) and not is_note(path)]
    tools = [path for path in candidates if PurePosixPath(path).parts[0] == "tools"]
    if not tools:
        return candidates
    kept = [path for path in candidates if path not in tools]
    for tool in tools:
        stem = PurePosixPath(tool).with_suffix("")
        pattern = rf"{re.escape(stem.name)}\.py|{re.escape('.'.join(stem.parts))}\b"
        named_by = subprocess.run(
            ["git", "-C", str(checkout), "grep", "-l", "-E", pattern, "HEAD"], capture_output=True, text=True
        ).stdout.split()
        referrers = {hit.removeprefix("HEAD:") for hit in named_by} - {tool}
        if any(not is_test(path) and not is_note(path) for path in referrers):
            kept.append(tool)
    return sorted(kept)


def test_files(checkout: Path) -> list[str]:
    """The host's tracked Python test files at ``HEAD``, its planning notes left out."""
    return [path for path in _git(checkout, "ls-files", "--", "*.py").split() if is_test(path) and not is_note(path)]


def host_uses(checkout: Path, *, tests: bool = False) -> set[str]:
    """Every package name the host's production files use, at ``HEAD``; with ``tests``, its test files instead."""
    found: set[str] = set()
    for path in test_files(checkout) if tests else production_files(checkout):
        source = _git(checkout, "show", f"HEAD:{path}")
        if PREFIX in source:
            found |= used_names(source, path)
    return found


def record(host: str, checkout: Path, *, into: Path = HOST_IMPORTS, tests_into: Path = HOST_TEST_IMPORTS) -> list[Path]:
    """Write ``host``'s production use of the package, and its tests' use, and return the files written."""
    commit = _git(checkout, "rev-parse", "--short", "HEAD").strip()
    branch = _git(checkout, "rev-parse", "--abbrev-ref", "HEAD").strip()
    date = _git(checkout, "log", "-1", "--format=%cs", "HEAD").strip()
    source = f"# From {host} {branch} at {commit} ({date})."
    lists = {
        into / f"{host}.txt": (
            host_uses(checkout),
            [
                f"# {host}'s production use of 3tears-evals, generated by packages/evals/scripts/record_host_imports.py: do not",
                "# edit by hand. Production is the host's package source and the operator tools it names, never its tests",
                "# (the script states the rule).",
            ],
        ),
        tests_into / f"{host}.txt": (
            host_uses(checkout, tests=True),
            [
                f"# {host}'s tests' use of 3tears-evals, generated by packages/evals/scripts/record_host_imports.py: do not",
                "# edit by hand. Its tests only (the script states the rule); tests/test_declared_holdbacks.py counts a",
                "# name here only for a kit of the `tests` kind.",
            ],
        ),
    }
    for target, (found, header) in lists.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join([*header, source, *sorted(found)]) + "\n", encoding="utf-8")
    return list(lists)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "host",
        help="the host's name: the lists are written to tests/fixtures/host_imports/<host>.txt and host_test_imports/",
    )
    parser.add_argument("checkout", type=Path, help="a git checkout of the host, read at HEAD")
    arguments = parser.parse_args()
    for path in record(arguments.host, arguments.checkout):
        print(path)


if __name__ == "__main__":
    main()
