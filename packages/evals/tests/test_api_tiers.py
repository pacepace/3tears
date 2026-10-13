"""Every public name carries a stability tier, and every name a host is taught to import is ``host-contract``.

A public name is ``host-contract`` (it moves only with a ``MIGRATIONS.toml`` record and a deprecation cycle) or
``closure`` (public because a public signature reaches it; it may move in any release, recorded and never shimmed).
``HOST_CONTRACT`` in ``scripts/public_surface.py`` is the one place the first tier is declared, and the snapshot
(``api/surface.json``) records each entry's tier as the generator derives it.

This holds the list's completeness, not its taste: a name the host guide, the tutorial, the README or an example
imports is one a host was told to write against, so a release moving it without the deprecation cycle would break
a host that did exactly what the docs said. What else belongs in the contract is a judgement the list's comments
record, and nothing here second-guesses it.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from pathlib import Path

import pytest

from packages.evals.tests.script_loader import load_script

surface = load_script("public_surface.py")

Entry = surface.SurfaceEntry

#: The package directory, which the docs and examples sit under.
PACKAGE = Path(__file__).resolve().parents[1]

#: What a host author is told to write against, in reading order: the README's first result, the host guide, the
#: tutorial and every example. A topic guide's imports are not here: a name only it imports is ``closure`` until
#: promoted, which is free, where demoting a name from the contract is a break.
HOST_GUIDES: tuple[Path, ...] = (
    PACKAGE / "README.md",
    PACKAGE / "docs" / "adopting-a-host.md",
    PACKAGE / "docs" / "tutorial.md",
    *sorted(p for p in (PACKAGE / "examples").iterdir() if p.suffix in (".py", ".md")),
)

#: One ``from threetears.evals… import …`` statement in prose or a fenced block, parenthesised across lines or not.
_DOC_IMPORT = re.compile(r"^[ \t]*from threetears\.evals(?:\.\w+)* import (?:\([^)]*\)|[^\n]*)", re.M)


def _imported(module: ast.Module) -> set[str]:
    return {
        f"{node.module}.{alias.name}"
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("threetears.evals")
        for alias in node.names
    }


def imported_names(path: Path) -> set[str]:
    """Every ``root.name`` a doc or an example imports from the package.

    An example is parsed whole. A page is not, since its fenced blocks hold elisions (``LaunchHost(..., x=y)``)
    that are not Python: each import statement on it is found by its spelling and parsed on its own.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return _imported(ast.parse(text, filename=str(path)))
    return {name for match in _DOC_IMPORT.finditer(text) for name in _imported(ast.parse(textwrap.dedent(match[0])))}


def test_every_snapshot_entry_carries_a_tier() -> None:
    """Read from the file as written, so an entry with no tier is caught rather than defaulted on the way in."""
    names = json.loads(surface.SNAPSHOT_PATH.read_text(encoding="utf-8"))["names"]
    untiered = {name: entry.get("tier") for name, entry in names.items() if entry.get("tier") not in surface.TIERS}
    assert not untiered, f"entries with no tier, or another: {untiered}; run `{surface.REGENERATE_SNAPSHOT}`"
    assert {entry["tier"] for entry in names.values()} == set(surface.TIERS)


def test_every_name_a_host_is_taught_to_import_is_host_contract() -> None:
    recorded = surface.load_snapshot()
    taught = {name: path for path in HOST_GUIDES for name in sorted(imported_names(path))}
    assert "threetears.evals.quick.run_eval" in taught, "the import reader found nothing, so this test says nothing"
    outside = {
        name: f"{path.relative_to(PACKAGE)} ({recorded[name].tier if name in recorded else 'not public'})"
        for name, path in sorted(taught.items())
        if name not in recorded or recorded[name].tier != "host-contract"
    }
    assert not outside, (
        "A host guide or example imports these, so a host writes against them: add each to HOST_CONTRACT in "
        f"packages/evals/scripts/public_surface.py and run `{surface.REGENERATE_SNAPSHOT}`:\n  "
        + "\n  ".join(f"{name} <- {where}" for name, where in outside.items())
    )


def test_the_import_reader_sees_every_spelling(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        textwrap.dedent(
            """\
            Run it:

            ```python
            from threetears.evals.quick import run_eval
            from threetears.evals.storage import (
                SqliteDocumentStore,  # a file
                DocTypeRoute as Route,
            )
            LaunchHost(eval_host=host, ..., on_run_end=watch)
            ```

            1. In a list:

               ```python
               from threetears.evals.kernel.host import EvalHost
               ```

            Not an import: the threetears.evals.schema module, or `from os import path`.
            """
        ),
        encoding="utf-8",
    )
    assert imported_names(page) == {
        "threetears.evals.quick.run_eval",
        "threetears.evals.storage.SqliteDocumentStore",
        "threetears.evals.storage.DocTypeRoute",
        "threetears.evals.kernel.host.EvalHost",
    }


def test_a_tier_comes_from_the_declared_list() -> None:
    installed = surface.installed_surface()
    assert installed["threetears.evals.kernel.host.EvalHost"].tier == "host-contract"
    assert installed["threetears.evals.quick.ARM_LEVER"].tier == "closure"
    assert {name for name, entry in installed.items() if entry.tier == "host-contract"} == surface.HOST_CONTRACT


# --- the rules, on made-up surfaces ------------------------------------------------------------------------------

A = "threetears.evals.alpha"


def _release() -> object:
    return surface.Release.model_validate({"version": "unreleased"})


def test_a_listed_name_no_root_exports_is_a_finding_regenerating_does_not_fix() -> None:
    """A stale list would read as a promise about a name that is not there."""
    findings = surface.tier_findings({})
    assert len(findings) == len(surface.HOST_CONTRACT)
    assert not any(f.regenerating_fixes for f in findings)
    assert "threetears.evals.kernel.host.EvalHost, which no public root exports" in "\n".join(
        f.problem for f in findings
    )


def test_a_tier_change_is_a_diff_in_review() -> None:
    [finding] = surface.snapshot_findings(
        {f"{A}.f": Entry("function", "f()", "closure")},
        {f"{A}.f": Entry("function", "f()", "host-contract")},
        _release(),
        {f"{A}.f"},
    )
    assert finding.regenerating_fixes
    assert "moved from tier host-contract to closure" in finding.problem


def test_a_host_contract_name_leaving_unrecorded_names_the_deprecation_cycle() -> None:
    [finding] = surface.snapshot_findings({}, {f"{A}.gone": Entry("class", "", "host-contract")}, _release(), None)
    assert not finding.regenerating_fixes
    assert "keeps working beside its new home for one minor release" in finding.problem


def test_a_snapshot_entry_without_a_tier_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "surface.json"
    entry = {"kind": "function", "signature": "f()"}
    path.write_text(json.dumps({"format": surface.SNAPSHOT_FORMAT, "names": {f"{A}.f": entry}}), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"alpha\.f with tier None.*regenerate it"):
        surface.load_snapshot(path)
