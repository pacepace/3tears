"""Every name a host imports is declared: ``host-contract`` or ``stable``, never ``provisional``.

A host's production code binds to what it imports, so a package that exports a name a host imports promises it. Each
file under ``tests/fixtures/host_imports/`` is one host's import list, one fully qualified
``threetears.evals.<module>.<name>`` per line, as the host reported it. A line naming a module rather than a name in
one (``threetears.evals``, ``threetears.evals.ops``) is skipped. Every other line must be a public name the snapshot
records, tiered ``host-contract`` or ``stable``; the failure names the host's file and the name.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from packages.evals.tests.script_loader import load_script

surface = load_script("public_surface.py")

HOST_IMPORTS = Path(__file__).resolve().parent / "fixtures" / "host_imports"


def _is_module(line: str) -> bool:
    try:
        importlib.import_module(line)
    except ImportError:
        return False
    return True


def host_imports() -> list[tuple[str, str]]:
    """Each ``(host file, name)`` the host import lists carry, modules skipped."""
    found: list[tuple[str, str]] = []
    for path in sorted(HOST_IMPORTS.glob("*.txt")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                found.append((path.name, line))
    return found


def test_there_is_a_host_import_list() -> None:
    assert host_imports(), f"no host import list under {HOST_IMPORTS}, so the test below says nothing"


def test_no_name_a_host_imports_is_provisional_or_missing() -> None:
    recorded = surface.load_snapshot()
    problems: list[str] = []
    for host_file, name in host_imports():
        if name in recorded:
            if not surface.declared(recorded[name].tier):
                problems.append(f"{host_file}: {name} is {recorded[name].tier}")
        elif not _is_module(name):
            problems.append(f"{host_file}: {name} is not a public name of any root")
    assert not problems, (
        "a host imports these, so each must be host-contract or stable: add it to HOST_CONTRACT or STABLE in "
        f"packages/evals/scripts/public_surface.py and run `{surface.REGENERATE_SNAPSHOT}`:\n  " + "\n  ".join(problems)
    )
