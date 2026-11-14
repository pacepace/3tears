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

**So are the test-support modules' (``threetears.core.testing``: fakes, the plan check, the calibration
helpers).** They exist for test suites, so a reference in any tests tree counts, except the support's own unit
tests (``packages/core/tests/unit/testing/``): a helper only its own tests call still has no caller (owner
ruling via the 0.95.0 load-path audit, 2026-11-09, extending the enforcement package's).

**The ledger stays honest.** An entry must name a name some module exports, give a reason, and be
one the scan still finds without a caller in this repo; an entry whose name gained one here is
stale and must go.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.enforcement.scan_cache import ScanCache, content_key, files_key
from threetears.enforcement.common import parse_python_path
from threetears.enforcement.release import release_tags, surface_at, surface_now

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEDGER = _REPO_ROOT / "tests" / "enforcement" / "_public_api_consumers.txt"
_MIN_REASON = 30

#: the package whose public names are a test-suite API: its callers are other packages' test shells.
_ENFORCEMENT_SRC = "packages/enforcement/src/"

#: the enforcement package's own tests, which exercise a domain but do not adopt it.
_ENFORCEMENT_TESTS = "packages/enforcement/tests/"

#: the test-support package, whose public names are a test-suite API as the enforcement package's are
_TEST_SUPPORT_SRC = "packages/core/src/threetears/core/testing/"

#: the test-support package's own tests, which exercise a helper but are not its callers
_TEST_SUPPORT_TESTS = "packages/core/tests/unit/testing/"

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

#: modules the release in progress moved, as (module path at the baseline tag, module path now). A name public at
#: the baseline under the mapped old module, with the same name, was released before and is not new; every other
#: name in the new modules answers to the gate as usual. List only the modules holding a moved name that stays
#: public with no caller in this repo, and empty the map once the release that made the move is the baseline:
#: the baseline then holds the new modules, and the pool ships release tags' trees, not the history between them.
#: The 3tears-interfaces consolidation (media-contracts into threetears.interfaces.media) shipped in v0.93.0.
_MOVED_MODULES: tuple[tuple[str, str], ...] = ()


def _public_at_baseline(
    module: str,
    name: str,
    before: dict[str, frozenset[str] | set[str]],
    moves: tuple[tuple[str, str], ...],
) -> bool:
    """whether ``name`` was public at the baseline in ``module``, or in a module ``moves`` maps to it.

    :param module: the exporting module now
    :ptype module: str
    :param name: the name
    :ptype name: str
    :param before: module path -> public names, at the baseline tag, across every source root
    :ptype before: dict[str, frozenset[str] | set[str]]
    :param moves: (module path at the baseline, module path now) pairs
    :ptype moves: tuple[tuple[str, str], ...]
    :return: whether the name was already released there
    :rtype: bool
    """
    if name in before.get(module, set()):
        return True
    return any(new == module and name in before.get(old, set()) for old, new in moves)


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
        if module.startswith(_ENFORCEMENT_SRC):
            return any(name in used for used in self.test_uses.values())
        if module.startswith(_TEST_SUPPORT_SRC):
            return any(
                name in used for path, used in self.test_uses.items() if not path.startswith(_TEST_SUPPORT_TESTS)
            )
        return False


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


def _tree_at(ref: str) -> str:
    """the id of *ref*'s tree: what every file read at *ref* is, so a key for anything read there.

    :param ref: a tag
    :ptype ref: str
    :return: the tree's object id
    :rtype: str
    :raises subprocess.CalledProcessError: when this clone cannot name the ref's tree
    """
    found = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{tree}}"], cwd=_REPO_ROOT, capture_output=True, text=True, check=True
    )
    return found.stdout.strip()


def _names_key(item: tuple[str, bool]) -> str:
    """the content key of one module's names: its source, and whether its imports count.

    :param item: (module path, whether a ``from ... import`` counts as a reference)
    :ptype item: tuple[str, bool]
    :return: the key
    :rtype: str
    """
    relative, re_exports = item
    return content_key("names", str(re_exports), (_REPO_ROOT / relative).read_bytes())


def _module_names(item: tuple[str, bool]) -> tuple[frozenset[str], frozenset[str]]:
    """:func:`_names` of one module.

    :param item: (module path, whether a ``from ... import`` counts as a reference)
    :ptype item: tuple[str, bool]
    :return: references, definitions
    :rtype: tuple[frozenset[str], frozenset[str]]
    """
    relative, re_exports = item
    return _names(parse_python_path(_REPO_ROOT / relative), re_exports=re_exports)


def _surface_now_key(source_root: str) -> str:
    """the content key of a source root's surface in the working tree: every module under it.

    :param source_root: repo-relative source tree
    :ptype source_root: str
    :return: the key
    :rtype: str
    """
    return content_key("surface_now", source_root, files_key((_REPO_ROOT / source_root).rglob("*.py"), _REPO_ROOT))


@pytest.fixture(scope="module")
def scan(scan_cache: ScanCache) -> _Scan:
    tags = release_tags(_REPO_ROOT)
    if not tags:
        pytest.fail("no vX.Y.Z release tag is visible (`git fetch --tags`); without a baseline nothing is compared")
    baseline = max(tags.items(), key=lambda item: item[1])[0]
    baseline_tree = _tree_at(baseline)
    exported: set[tuple[str, str]] = set()
    added: set[tuple[str, str]] = set()
    # a moved module's baseline surface lives under its old source root, which no longer exists in the tree
    moved_roots = {old.partition("/src/")[0] + "/src" for old, _ in _MOVED_MODULES}
    before: dict[str, frozenset[str] | set[str]] = {}
    baseline_surfaces = scan_cache.memoised(
        "public_api_surface_at",
        sorted(set(_SOURCE_ROOTS) | moved_roots),
        lambda source_root: content_key("surface_at", baseline_tree, source_root),
        lambda source_root: surface_at(_REPO_ROOT, baseline, source_root).modules,
    )
    for modules in baseline_surfaces.values():
        before.update(modules)
    surfaces = scan_cache.memoised(
        "public_api_surface_now",
        _SOURCE_ROOTS,
        _surface_now_key,
        lambda source_root: surface_now(_REPO_ROOT, source_root).modules,
    )
    src_modules: list[tuple[str, bool]] = []
    for source_root in _SOURCE_ROOTS:
        for module, names in surfaces[source_root].items():
            for name in names:
                if "." in name or " " in name:
                    continue
                exported.add((module, name))
                if not _public_at_baseline(module, name, before, _MOVED_MODULES):
                    added.add((module, name))
        for path in sorted((_REPO_ROOT / source_root).rglob("*.py")):
            src_modules.append((path.relative_to(_REPO_ROOT).as_posix(), path.name != "__init__.py"))
    test_modules: list[tuple[str, bool]] = []
    for pattern in _TEST_ROOTS:
        for root in _REPO_ROOT.glob(pattern):
            for path in sorted(root.rglob("*.py")):
                relative = path.relative_to(_REPO_ROOT).as_posix()
                if not relative.startswith(_ENFORCEMENT_TESTS):
                    test_modules.append((relative, True))
    names = scan_cache.memoised("public_api_names", [*src_modules, *test_modules], _names_key, _module_names)
    uses: dict[str, frozenset[str]] = {}
    defines: dict[str, frozenset[str]] = {}
    for item in src_modules:
        uses[item[0]], defines[item[0]] = names[item]
    test_uses = {item[0]: names[item][0] for item in test_modules}
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

    def test_a_moved_name_is_not_new_and_a_new_name_in_a_moved_module_still_is(self) -> None:
        """the move map excuses only a name released under the mapped old module, under the same name."""
        old = "packages/old/src/threetears/old/contracts.py"
        new = "packages/new/src/threetears/new/contracts.py"
        before: dict[str, frozenset[str] | set[str]] = {old: frozenset({"Moved"})}
        moves = ((old, new),)
        assert _public_at_baseline(new, "Moved", before, moves)
        assert not _public_at_baseline(new, "BrandNew", before, moves)
        # without the map entry the same move reads as new, and a map for another module excuses nothing
        assert not _public_at_baseline(new, "Moved", before, ())
        assert not _public_at_baseline(new, "Moved", before, ((old, "packages/other/src/threetears/other.py"),))

    def test_every_moved_module_is_a_real_move(self, scan: _Scan) -> None:
        """guard the guard: a map entry with a mistyped path would excuse nothing, silently.

        The old path is gone from the tree and was released: it is a file in the tree of some release tag at or
        before the baseline, by version (still true once a later tag is the baseline); the new path is in the tree.
        Read from the tags' trees alone, not from history, since a testament runner's checkout carries the release
        tags (``.testament.toml``'s ``[git] tags``) as refs to their trees but no commits or ancestry. The map
        empties once its release is the baseline, so an empty map passes.
        """
        tags = release_tags(_REPO_ROOT)
        released = sorted(tag for tag, version in tags.items() if version <= tags[scan.baseline])
        for old, new in _MOVED_MODULES:
            assert not (_REPO_ROOT / old).exists(), f"{old} is still in the tree; it did not move"
            assert (_REPO_ROOT / new).is_file(), f"{new} is not in the tree"
            # one read for every tag: `<tag>:<path>` answers `<sha> blob <size>`, or `<tag>:<path> missing`
            found = subprocess.run(
                ["git", "cat-file", "--batch-check"],
                cwd=_REPO_ROOT,
                input="".join(f"{tag}:{old}\n" for tag in released),
                capture_output=True,
                text=True,
                check=True,
            )
            kinds = [line.split()[1] for line in found.stdout.splitlines() if not line.endswith(" missing")]
            assert "blob" in kinds, f"{old} is in no release tag's tree at or before {scan.baseline}"

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
