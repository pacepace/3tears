"""Every public name carries a stability tier, every name a host is taught to import is ``host-contract``, and every
name any page or example host imports is declared.

A public name is ``host-contract`` (it moves only with a ``MIGRATIONS.toml`` record and a deprecation cycle),
``stable`` (the rest of the declared surface: it may move in any release, recorded and never shimmed) or
``provisional`` (exported, mostly because a declared signature reaches it, and promised nothing). ``HOST_CONTRACT``
and ``STABLE`` in ``scripts/public_surface.py`` are the one place the first two are declared, and the snapshot
(``api/surface.json``) records each entry's tier as the generator derives it.

This holds the lists' completeness, not their taste: a name the host guide, the ladder, the README or an example
imports is one a host was told to write against, so a release moving it without the deprecation cycle would break
a host that did exactly what the docs said; and a name a topic guide or an example host imports is one a reader was
shown, so it may not be provisional. What else belongs in either list is a judgement their comments record, and
nothing here second-guesses it beyond the ceiling on how many names they hold.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from importlib.resources import files
from pathlib import Path

import pytest

import threetears.evals

from packages.evals.tests.script_loader import load_script

surface = load_script("public_surface.py")

Entry = surface.SurfaceEntry

#: The package directory, which the docs and examples sit under.
PACKAGE = Path(__file__).resolve().parents[1]

#: What a host author is told to write against, in reading order: the README's first result, the host guide, the
#: ladder and every example. A topic guide's imports are not here (they are :data:`SHOWN`): a name only it imports
#: is ``stable`` until promoted, which is free, where demoting a name from the contract is a break.
HOST_GUIDES: tuple[Path, ...] = (
    PACKAGE / "README.md",
    PACKAGE / "docs" / "adopting-a-host.md",
    PACKAGE / "docs" / "ladder.md",
    *sorted(p for p in (PACKAGE / "examples").rglob("*") if p.suffix in (".py", ".md")),
)

#: Every page and example host a reader is shown, beyond the host guides: each topic guide and the courier host, the
#: least a product writes (``tests/fixtures/courierhost/``), which the package matrix holds to the roots' ``__all__``
#: as an adopter's own code.
SHOWN: tuple[Path, ...] = (
    *sorted(p for p in (PACKAGE / "docs").glob("*.md") if p.name != "reference.md"),
    *sorted((PACKAGE / "tests" / "fixtures" / "courierhost").rglob("*.py")),
)

#: The toy host (``tests/fixtures/toyhost/``): a host that exercises every shape of the contract, read after the
#: courier host. It reaches past the declared surface on purpose (a variant's identity, a lease's default, a payload
#: elision), so its imports are held to the public roots, not to the declared tiers: a name it alone imports is
#: ``provisional`` unless something a reader is taught to write imports it too.
EXERCISER: tuple[Path, ...] = tuple(sorted((PACKAGE / "tests" / "fixtures" / "toyhost").rglob("*.py")))

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
    assert "threetears.evals.Eval" in taught, "the import reader found nothing, so this test says nothing"
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


def test_every_name_a_page_or_an_example_host_imports_is_declared() -> None:
    recorded = surface.load_snapshot()
    shown = {name: path for path in (*HOST_GUIDES, *SHOWN) for name in sorted(imported_names(path))}
    assert "threetears.evals.kernel.host.EvalHost" in shown, "the import reader found nothing, so this says nothing"
    provisional = {
        name: f"{path.relative_to(PACKAGE)} ({recorded[name].tier if name in recorded else 'not public'})"
        for name, path in sorted(shown.items())
        if name not in recorded or not surface.declared(recorded[name].tier)
    }
    assert not provisional, (
        "A page or an example host imports these, so a reader was shown them: add each to STABLE in "
        f"packages/evals/scripts/public_surface.py and run `{surface.REGENERATE_SNAPSHOT}`:\n  "
        + "\n  ".join(f"{name} <- {where}" for name, where in provisional.items())
    )


def test_every_name_the_toy_host_imports_is_public() -> None:
    """Exported by a root, any tier: the toy host is reference code an adopter reads, so it never reaches below one."""
    recorded = surface.load_snapshot()
    imported = {name: path for path in EXERCISER for name in sorted(imported_names(path))}
    assert "threetears.evals.kernel.host.EvalHost" in imported, "the import reader found nothing, so this says nothing"
    below = {name: str(path.relative_to(PACKAGE)) for name, path in sorted(imported.items()) if name not in recorded}
    assert not below, f"the toy host imports names no public root exports: {below}"


def test_the_declared_surface_is_one_sitting() -> None:
    """The declared surface is what a reader takes in at once; promoting a name past the ceiling demotes another."""
    declared = surface.HOST_CONTRACT | surface.STABLE
    assert len(declared) <= surface.DECLARED_CEILING, (
        f"{len(declared)} declared names, over the ceiling of {surface.DECLARED_CEILING}: demote one to "
        "provisional (drop it from STABLE) before promoting another."
    )


#: One root's line on ``docs/surface.md``: the root, ``threetears.evals.`` dropped, then its names.
_SURFACE_LINE = re.compile(r"^- `([\w.]+)`: (.+)$", re.M)


def test_the_surface_page_lists_exactly_the_declared_names() -> None:
    """``docs/surface.md`` maps the declared surface by rung: every declared name once, and nothing provisional."""
    page = (PACKAGE / "docs" / "surface.md").read_text(encoding="utf-8")
    listed: list[str] = []
    for label, names in _SURFACE_LINE.findall(page):
        root = label if label == "threetears.evals" else f"threetears.evals.{label}"
        listed += [f"{root}.{name}" for name in re.findall(r"`(\w+)`", names)]
    declared = surface.HOST_CONTRACT | surface.STABLE
    twice = sorted({name for name in listed if listed.count(name) > 1})
    assert not twice, f"docs/surface.md lists these more than once: {twice}"
    assert set(listed) == declared, (
        f"docs/surface.md is not the declared surface. Missing: {sorted(declared - set(listed))}. "
        f"Not declared: {sorted(set(listed) - declared)}."
    )


def test_the_import_reader_sees_every_spelling(tmp_path: Path) -> None:
    page = tmp_path / "page.md"
    page.write_text(
        textwrap.dedent(
            """\
            Run it:

            ```python
            from threetears.evals import Eval
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
        "threetears.evals.Eval",
        "threetears.evals.storage.SqliteDocumentStore",
        "threetears.evals.storage.DocTypeRoute",
        "threetears.evals.kernel.host.EvalHost",
    }


def test_a_tier_comes_from_the_declared_list() -> None:
    installed = surface.installed_surface()
    assert installed["threetears.evals.kernel.host.EvalHost"].tier == "host-contract"
    assert installed["threetears.evals.quick.EXIT_FAILED"].tier == "stable"
    assert installed["threetears.evals.kernel.AttemptOutcome"].tier == "provisional"
    assert {name for name, entry in installed.items() if entry.tier == "host-contract"} == surface.HOST_CONTRACT
    assert {name for name, entry in installed.items() if entry.tier == "stable"} == surface.STABLE


def test_the_installed_tier_map_is_the_repository_snapshot() -> None:
    """The wheel ships the tiers (``api/`` is not installed), and what it ships is exactly what the snapshot records."""
    recorded = json.loads(surface.SNAPSHOT_PATH.read_text(encoding="utf-8"))["names"]
    installed = files("threetears.evals").joinpath("surface_tiers.json").read_text(encoding="utf-8")
    assert installed == surface.render_tier_map(surface.load_snapshot()), (
        f"the shipped surface_tiers.json is not the snapshot's tiers: run `{surface.REGENERATE_SNAPSHOT}`"
    )
    assert {name: threetears.evals.surface_tier(name) for name in recorded} == {
        name: entry["tier"] for name, entry in recorded.items()
    }
    assert threetears.evals.declared_names() == surface.HOST_CONTRACT | surface.STABLE


def test_a_host_gates_its_imports_by_tier_against_the_installed_build() -> None:
    assert threetears.evals.surface_tier("threetears.evals.kernel.host.EvalHost") == "host-contract"
    assert threetears.evals.surface_tier("threetears.evals.quick.EXIT_FAILED") == "stable"
    assert threetears.evals.surface_tier("threetears.evals.kernel.AttemptOutcome") == "provisional"
    assert "threetears.evals.surface_tier" in threetears.evals.declared_names()
    with pytest.raises(KeyError, match="not a public name of this build"):
        threetears.evals.surface_tier("threetears.evals.run.runner.grade_goal_checks")


def test_a_provisional_name_records_no_signature() -> None:
    """Its signature is no promise, so a change to it is not a diff the snapshot asks a reviewer to read."""
    installed = surface.installed_surface()
    assert all(entry.signature == "" for entry in installed.values() if entry.tier == "provisional")
    assert installed["threetears.evals.kernel.host.EvalHost"].signature.startswith("EvalHost(")


# --- the rules, on made-up surfaces ------------------------------------------------------------------------------

A = "threetears.evals.alpha"


def _release() -> object:
    return surface.Release.model_validate({"version": "unreleased"})


def test_a_listed_name_no_root_exports_is_a_finding_regenerating_does_not_fix() -> None:
    """A stale list would read as a promise about a name that is not there."""
    findings = surface.tier_findings({})
    assert len(findings) == len(surface.HOST_CONTRACT) + len(surface.STABLE)
    assert not any(f.regenerating_fixes for f in findings)
    assert "threetears.evals.kernel.host.EvalHost, which no public root exports" in "\n".join(
        f.problem for f in findings
    )


def test_a_tier_change_is_a_diff_in_review() -> None:
    [finding] = surface.snapshot_findings(
        {f"{A}.f": Entry("function", "f()", "stable")},
        {f"{A}.f": Entry("function", "f()", "host-contract")},
        _release(),
        {f"{A}.f"},
    )
    assert finding.regenerating_fixes
    assert "moved from tier host-contract to stable" in finding.problem


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
