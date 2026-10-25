"""
enforcement: no public API exists only for a test.

``CLAUDE.md`` ("Leading underscores are a stability contract"): *a test reaches an object through
its front door ... Promote a name only when production code needs it too.* A name a module exports
(its ``__all__``, or every public top-level name of a module without one, as
:mod:`threetears.enforcement.release.surface` reads a module's surface) must have a caller in some
package's ``src`` other than the module that defines it -- or name its caller outside this repo in
``_public_api_consumers.txt``, one ``<module path>:<name>  # <who calls it, and where>`` per line.

**What it holds.** Names added to the public surface since the latest release tag, the baseline
:mod:`test_api_growth_requires_a_minor_bump` compares against. A name public at that tag was
released under the rule as a convention, and much of the family's surface is consumed by the hub
and the agent repos, so the gate is a ratchet: each release's new names answer to it, and the
ledger holds those whose callers live elsewhere. Missing tags are a refusal, as for the growth gate.

**What counts as a caller.** A reference to the name -- a bare name, an attribute, or a ``from ...
import`` of it -- in any ``src`` module but those defining it at top level, and not a package
``__init__``'s re-export import. Matching is by name, not by resolved binding, so a common name used
for something else also counts: the gate can miss a test-only name, never invent one. A method of
an exported class is reached through the class and is not held.

**The enforcement package's callers are test shells.** Its domains exist to be called from a
``tests/enforcement`` shell, so a name ``packages/enforcement/src`` exports also counts a reference in
any tests tree outside the enforcement package itself (owner ruling, 2026-10-25). Its own unit tests
do not count: a runner no other package's tests adopt still has no caller.

**The ledger stays honest.** An entry must name a name some module exports, give a reason, and be
one the scan still finds without a caller in this repo; an entry whose name gained one here is
stale and must go.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from threetears.enforcement.common import parse_python_path
from threetears.enforcement.release import release_tags, surface_at, surface_now

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEDGER = _REPO_ROOT / "tests" / "enforcement" / "_public_api_consumers.txt"
_MIN_REASON = 30

#: the package whose public names are a test-suite API: its callers are other packages' test shells.
_ENFORCEMENT_SRC = "packages/enforcement/src/"

#: the enforcement package's own tests, which exercise a domain but do not adopt it.
_ENFORCEMENT_TESTS = "packages/enforcement/tests/"

#: every tests tree whose references count as a caller of an enforcement name.
_TEST_ROOTS = ("tests", "packages/*/tests", "packages/agent/*/tests")

#: every workspace member's source tree, per the root pyproject's workspace globs.
_SOURCE_ROOTS = tuple(
    sorted(
        path.relative_to(_REPO_ROOT).as_posix()
        for pattern in ("packages/*/src", "packages/agent/*/src")
        for path in _REPO_ROOT.glob(pattern)
        if (path.parent / "pyproject.toml").is_file()
    )
)


@dataclass(frozen=True)
class _Scan:
    """what one walk of the workspace's ``src`` saw.

    :ivar exported: (module path, name) for every module-level public name today
    :ivar added: those not public at the baseline tag
    :ivar uses: module path -> every name it references
    :ivar test_uses: test module path -> every name it references, outside the enforcement package's tests
    :ivar defines: module path -> every name it defines at top level
    :ivar baseline: the tag compared against
    """

    exported: frozenset[tuple[str, str]]
    added: frozenset[tuple[str, str]]
    uses: dict[str, frozenset[str]]
    test_uses: dict[str, frozenset[str]]
    defines: dict[str, frozenset[str]]
    baseline: str

    def has_caller(self, module: str, name: str) -> bool:
        """whether some ``src`` module other than a definer of ``name`` references it, or, for an
        enforcement name, some test module outside the enforcement package's own tests.

        :param module: the exporting module
        :ptype module: str
        :param name: the name
        :ptype name: str
        :return: whether a caller exists
        :rtype: bool
        """
        if any(name in used and path != module and name not in self.defines[path] for path, used in self.uses.items()):
            return True
        return module.startswith(_ENFORCEMENT_SRC) and any(name in used for used in self.test_uses.values())


def _names(tree: ast.Module, *, re_exports: bool) -> tuple[frozenset[str], frozenset[str]]:
    """the names a module references, and the names it defines at top level.

    :param tree: the module
    :ptype tree: ast.Module
    :param re_exports: whether a ``from ... import`` counts as a reference (not in a package ``__init__``)
    :ptype re_exports: bool
    :return: references, definitions
    :rtype: tuple[frozenset[str], frozenset[str]]
    """
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.ImportFrom) and re_exports:
            used.update(alias.name for alias in node.names)
    defined: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            defined.add(statement.name)
        elif isinstance(statement, ast.Assign | ast.AnnAssign):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            defined.update(target.id for target in targets if isinstance(target, ast.Name))
    return frozenset(used), frozenset(defined)


@pytest.fixture(scope="module")
def scan() -> _Scan:
    tags = release_tags(_REPO_ROOT)
    if not tags:
        pytest.fail("no vX.Y.Z release tag is visible (`git fetch --tags`); without a baseline nothing is compared")
    baseline = max(tags.items(), key=lambda item: item[1])[0]
    exported: set[tuple[str, str]] = set()
    added: set[tuple[str, str]] = set()
    uses: dict[str, frozenset[str]] = {}
    defines: dict[str, frozenset[str]] = {}
    for source_root in _SOURCE_ROOTS:
        now = surface_now(_REPO_ROOT, source_root)
        before = surface_at(_REPO_ROOT, baseline, source_root)
        for module, names in now.modules.items():
            for name in names:
                if "." in name or " " in name:
                    continue
                exported.add((module, name))
                if name not in before.modules.get(module, set()):
                    added.add((module, name))
        for path in sorted((_REPO_ROOT / source_root).rglob("*.py")):
            relative = path.relative_to(_REPO_ROOT).as_posix()
            uses[relative], defines[relative] = _names(parse_python_path(path), re_exports=path.name != "__init__.py")
    test_uses: dict[str, frozenset[str]] = {}
    for pattern in _TEST_ROOTS:
        for root in _REPO_ROOT.glob(pattern):
            for path in sorted(root.rglob("*.py")):
                relative = path.relative_to(_REPO_ROOT).as_posix()
                if not relative.startswith(_ENFORCEMENT_TESTS):
                    test_uses[relative], _ = _names(parse_python_path(path), re_exports=True)
    return _Scan(frozenset(exported), frozenset(added), uses, test_uses, defines, baseline)


def _ledger() -> dict[tuple[str, str], str]:
    """the ledger's entries: (module path, name) -> the reason given.

    :return: the entries
    :rtype: dict[tuple[str, str], str]
    """
    entries: dict[tuple[str, str], str] = {}
    for line in _LEDGER.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        target, _, reason = line.partition("#")
        module, _, name = target.strip().rpartition(":")
        entries[(module, name)] = reason.strip()
    return entries


class TestPublicApiHasACaller:
    def test_the_scan_covers_the_workspace(self, scan: _Scan) -> None:
        """non-vacuity: the modules read, the surface seen, and a known caller found."""
        assert len(scan.uses) > 900, f"only {len(scan.uses)} src modules read; discovery has collapsed"
        assert len(scan.exported) > 5000, f"only {len(scan.exported)} public names seen; the surface reader collapsed"
        assert scan.has_caller("packages/core/src/threetears/core/backends/protocol.py", "LoadFile")
        # an enforcement runner's caller is another package's test shell: core's tests adopt memory_only_kv
        assert scan.has_caller(
            "packages/enforcement/src/threetears/enforcement/memory_only_kv/runner.py", "run_memory_only_kv_enforcement"
        )

    def test_a_test_caller_counts_for_an_enforcement_name_alone(self) -> None:
        """the test-shell allowance is the enforcement package's, and its own tests never earn it."""
        enforcement = "packages/enforcement/src/threetears/enforcement/demo/runner.py"
        elsewhere = "packages/iam/src/threetears/iam/demo.py"
        adopted = _Scan(
            exported=frozenset(),
            added=frozenset(),
            uses={enforcement: frozenset(), elsewhere: frozenset()},
            test_uses={"packages/iam/tests/test_demo.py": frozenset({"run_demo", "demo_helper"})},
            defines={enforcement: frozenset({"run_demo"}), elsewhere: frozenset({"demo_helper"})},
            baseline="v0.0.0",
        )
        assert adopted.has_caller(enforcement, "run_demo")
        assert not adopted.has_caller(elsewhere, "demo_helper")
        unadopted = _Scan(
            exported=frozenset(),
            added=frozenset(),
            uses={enforcement: frozenset()},
            test_uses={},
            defines={enforcement: frozenset({"run_demo"})},
            baseline="v0.0.0",
        )
        assert not unadopted.has_caller(enforcement, "run_demo")

    def test_every_name_added_since_the_last_release_has_a_caller_or_names_one(self, scan: _Scan) -> None:
        ledger = _ledger()
        missing = sorted(
            f"  {module}:{name}"
            for module, name in scan.added
            if (module, name) not in ledger and not scan.has_caller(module, name)
        )
        assert not missing, (
            f"{len(missing)} name(s) public since {scan.baseline} that nothing in this repo's src calls. A public "
            f"name exists for production code: make it private and test through the front door, or name its "
            f"caller outside this repo in {_LEDGER.name}:\n" + "\n".join(missing)
        )

    def test_every_ledger_entry_is_live_and_reasoned(self, scan: _Scan) -> None:
        problems: list[str] = []
        for (module, name), reason in sorted(_ledger().items()):
            if (module, name) not in scan.exported:
                problems.append(f"  {module}:{name} is not exported; drop the entry")
            elif scan.has_caller(module, name):
                problems.append(f"  {module}:{name} has a caller in this repo now; drop the entry")
            if len(reason) < _MIN_REASON:
                problems.append(f"  {module}:{name} names no caller (a reason of at least {_MIN_REASON} characters)")
        assert not problems, f"{_LEDGER.name}:\n" + "\n".join(problems)
