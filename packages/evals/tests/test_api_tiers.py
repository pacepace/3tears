"""Every public name carries a stability tier, every name a host is taught to import is ``host-contract``, and every
name any page or example host imports is declared.

A public name is ``host-contract`` (it moves only with a ``MIGRATIONS.toml`` record naming its replacement, a single
cut-over with no shim),
``stable`` (the rest of the declared surface: it may move in any release, recorded and never shimmed) or
``provisional`` (exported, mostly because a declared signature reaches it, and promised nothing). ``HOST_CONTRACT``
and ``STABLE`` in ``scripts/public_surface.py`` are the one place the first two are declared, and the snapshot
(``api/surface.json``) records each entry's tier as the generator derives it.

This holds the lists' completeness, not their taste: a name the host guide, the ladder, the README or an example
imports is one a host was told to write against, so a release moving it without a record of its replacement would break
a host that did exactly what the docs said; and a name a topic guide or an example host imports is one a reader was
shown, so it may not be provisional. What else belongs in either list is a judgement their comments record, and
nothing here second-guesses it beyond each kind's ceiling.

Every declared name is also exactly one kind (``scripts/surface_kinds.py``): ``typed`` in an eval, ``bound`` by a
host, ``action`` (behind the catalogue), ``invoked`` by a host's code, ``received`` or for ``tests``. This holds that each has one, and that each is admitted by the
criteria code can check (owner ruling, 2026-11-15): a received shape is reached, two near-duplicate shapes say why
both exist, and a typed name is taught on the rung that imports it or the page recorded for it. No total is held to a
number; the adopter's two goals, a first eval's names and rung 3's, are.
"""

from __future__ import annotations

import ast
import json
import re
import textwrap
from functools import cache
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


#: Each kind's rule and record, and what admits a name (``scripts/surface_kinds.py``).
kinds_of = load_script("surface_kinds.py")


@cache
def declared_kinds() -> object:
    return kinds_of.declared_kinds(surface.load_snapshot())


def test_every_declared_name_has_exactly_one_kind() -> None:
    """By rule, or by an entry no rule contradicts: a name with neither, or with both, is a problem the record names."""
    found = declared_kinds()
    assert not found.problems, "\n  ".join(("the declared names' kinds:", *found.problems))
    declared = {name for name, entry in surface.load_snapshot().items() if surface.declared(entry.tier)}
    assert set(found.of) == declared
    assert set(found.of.values()) == set(kinds_of.KINDS)


def test_a_kind_comes_from_its_rule_or_its_entry() -> None:
    kind = declared_kinds().of
    assert kind["threetears.evals.Eval"] == "typed"
    assert kind["threetears.evals.Result"] == "received"  # what `run()` hands back, a rule: returned, never taken
    assert kind["threetears.evals.quick.Grader"] == "received"
    assert kind["threetears.evals.schema.DocumentStore"] == "bound"  # a protocol a host implements
    # A protocol the engine implements and hands a host's kind, or satisfies with its own storage: no host must
    # implement it, so it is not bound.
    assert kind["threetears.evals.kernel.CellSink"] == "received"
    assert kind["threetears.evals.kernel.RunStore"] == "received"
    assert kind["threetears.evals.ops.runs_list"] == "action"  # an action's name: invoked through the catalogue
    assert kind["threetears.evals.analysis.format_number"] == "invoked"  # a step a host does the engine's way
    assert kind["threetears.evals.Answer"] == "typed"  # taught on a page, not a rung
    assert kind["threetears.evals.schema.EvalRun"] == "received"
    assert kind["threetears.evals.testing.STORE_CONFORMANCE_CASES"] == "tests"
    # Entries, since no rule is certain: a host's dataclass, a function handed over as a value, a launch entry.
    for name in (
        "threetears.evals.kernel.host.EvalHost",
        "threetears.evals.providers.describe_failure",
        "threetears.evals.kernel.default_job_timeout",
        "threetears.evals.run.LaunchableKind",
    ):
        assert kind[name] == "bound" and name in kinds_of.KIND_ENTRIES, name


def test_no_name_takes_a_kind_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Made up: a function that is no action, a class and a shape nothing names have no rule, and each fails saying
    how to decide; a protocol the ledger does not place fails naming the ledger; a protocol entered against the side
    the ledger gives it, and an entry for no declared name, fail too."""
    names = {
        "threetears.evals.alpha.P": surface.SurfaceEntry("protocol", "P{}", "stable"),
        "threetears.evals.alpha.Q": surface.SurfaceEntry("protocol", "Q{}", "stable"),
        "threetears.evals.alpha.f": surface.SurfaceEntry("function", "f()", "stable"),
        "threetears.evals.alpha.C": surface.SurfaceEntry("class", "C()", "stable"),
        "threetears.evals.alpha.M": surface.SurfaceEntry("model", "M()", "stable"),
    }
    monkeypatch.setattr(kinds_of, "protocol_sides", lambda: {"threetears.evals.alpha.P": kinds_of.HOST_IMPLEMENTS})
    monkeypatch.setattr(
        kinds_of, "KIND_ENTRIES", {"threetears.evals.alpha.P": "received", "threetears.evals.z": "bound"}
    )
    problems = kinds_of.declared_kinds(names).problems
    assert problems[0] == (
        "threetears.evals.alpha.P is bound by rule and also entered as received: drop it from KIND_ENTRIES"
    )
    assert problems[1].startswith("threetears.evals.alpha.Q (protocol) has no kind")
    assert "declared_holdbacks.txt as port-host-implements" in problems[1]
    for name, problem in zip(("f", "C", "M"), problems[2:5], strict=True):
        assert problem.startswith(f"threetears.evals.alpha.{name} (")
        assert "has no kind, since no rule decides it: record it in KIND_ENTRIES" in problem
        assert all(f"{kind}, {meaning}" in problem for kind, meaning in kinds_of.KIND_MEANINGS.items())
        assert "asking first whether a newcomer writes it" in problem
    assert problems[5] == "threetears.evals.z is in KIND_ENTRIES but is not declared"
    assert len(problems) == 6


def test_every_declared_name_is_admitted() -> None:
    """The criteria code can check, over the whole declared surface: each failure says what to do."""
    snapshot = surface.load_snapshot()
    declared = {name: entry for name, entry in snapshot.items() if surface.declared(entry.tier)}
    problems = kinds_of.admission_problems(declared, declared_kinds().of)
    assert not problems, "\n  ".join(("names the admission criteria refuse:", *problems))
    # Not vacuous: two near-duplicate pairs are found, and each is admitted only because one names the other.
    assert len(kinds_of.near_duplicate_shapes(declared_kinds().of)) == 2
    assert dict(kinds_of.TYPED) and dict(kinds_of.TYPED_IN_DOCS)


def test_a_name_no_criterion_admits_is_refused_with_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shape nothing reaches, a typed name its page does not teach, and a typed kind entered by hand."""
    snapshot = surface.load_snapshot()
    declared = {name: entry for name, entry in snapshot.items() if surface.declared(entry.tier)}
    kinds = declared_kinds().of  # read before anything is patched: the record is cached for every test
    # Only the shape, and nothing that could reach it.
    lone = kinds_of.admission_problems(declared, {"threetears.evals.schema.EvalRun": "received"})
    assert any(
        problem.startswith("threetears.evals.schema.EvalRun is a received shape that no typed, bound, action, invoked")
        and "demote it" in problem
        for problem in lone
    )
    monkeypatch.setattr(kinds_of, "TYPED_IN_DOCS", {"threetears.evals.Answer": "docs/ladder.md"})
    monkeypatch.setattr(kinds_of, "KIND_ENTRIES", {**kinds_of.KIND_ENTRIES, "threetears.evals.quick.as_host": "typed"})
    problems = kinds_of.admission_problems(declared, kinds)
    assert "threetears.evals.Answer is recorded as taught on docs/ladder.md, which does not name it (criterion 4)" in (
        problems
    )
    assert any(
        problem.startswith("threetears.evals.quick.as_host is entered typed in KIND_ENTRIES") for problem in problems
    )


def test_a_first_eval_needs_two_names_and_rung_three_five() -> None:
    """The adopter's goals: the only numbers the surface is held to."""
    assert kinds_of.typed_through(1) <= kinds_of.FIRST_EVAL_GOAL == 2
    assert kinds_of.typed_through(3) <= kinds_of.THROUGH_RUNG_3_GOAL == 5
    assert set(kinds_of.TYPED.values()) <= set(kinds_of.rung_titles()), "a typed name's rung is not on the ladder"


def test_the_reach_walk_follows_fields_returns_and_aliases() -> None:
    reach = kinds_of.shape_reach(surface.load_snapshot(), declared_kinds().of).by
    assert reach["threetears.evals.schema.IntervalScale"] is not None  # through an alias no root exports
    assert reach["threetears.evals.transports.fastmcp.ToolCall"] is not None  # a Callable alias's parameter
    assert reach["threetears.evals.RootExtra"] == "threetears.evals.ROOT_EXTRAS"  # a constant's annotation
    # A parameter named by a name its module imports only for type checking (get_run's RunStore).
    assert reach["threetears.evals.kernel.RunStore"] is not None
    # A documented return its protocol-typed annotation does not name (Model.generate hands back a Completion).
    assert reach["threetears.evals.providers.Completion"] is not None


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
    assert installed == surface.render_tier_map(surface.load_snapshot(), declared_kinds().of), (
        f"the shipped surface_tiers.json is not the snapshot's tiers and kinds: run `{surface.REGENERATE_SNAPSHOT}`"
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


def test_a_host_contract_name_leaving_unrecorded_asks_for_its_replacement() -> None:
    [finding] = surface.snapshot_findings({}, {f"{A}.gone": Entry("class", "", "host-contract")}, _release(), None)
    assert not finding.regenerating_fixes
    assert "names what replaces it (Port: ...)" in finding.problem


def test_a_snapshot_entry_without_a_tier_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "surface.json"
    entry = {"kind": "function", "signature": "f()"}
    path.write_text(json.dumps({"format": surface.SNAPSHOT_FORMAT, "names": {f"{A}.f": entry}}), encoding="utf-8")
    with pytest.raises(SystemExit, match=r"alpha\.f with tier None.*regenerate it"):
        surface.load_snapshot(path)
