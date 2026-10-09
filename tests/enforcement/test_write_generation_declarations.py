"""
enforcement: what a collection may say about its write generation, and what following one owes.

A table's write generation tells a pod that its cache of the table is behind. That only holds
while three things stay true across the family, and each is a rule here:

1. **A declaration is one of three things, and an opt-out says why.** ``write_generation`` in a
   class body is ``WRITE_GENERATION``, ``WRITE_GENERATION_UNDECLARED`` or
   ``NoWriteGeneration(reason="...")`` with a non-empty literal reason. ``BaseCollection`` refuses
   anything else when the class is defined; this walker is for the reader, who should be able to
   find every opt-out and its reason with a search, never behind a variable or a computed string.
2. **A switched-on collection announces no row without its advance.** A class that declares
   ``WRITE_GENERATION`` and publishes a row message from a method of its own, bypassing the write
   paths ``BaseCollection`` advances on, must hand that message the advance (``bump=``). A row
   announced without one is a commit the generation never moved for, and a follower that heard
   every broadcast reads "the generation did not move" as "nothing changed".
3. **Whoever follows a table schedules the pass that judges it.** ``follow_generation`` only
   starts a mark; nothing is ever dropped until ``generation_catchup_tick`` or
   ``follow_generation_key`` compares the mark with the bucket. A bootstrap that follows and
   schedules neither has a registry that counts broadcasts for ever and never acts on one missed.
4. **One table, one line of classes, one declaration.** Every ``BaseCollection`` subclass the
   family defines is enumerated by importing every module, in a process of its own
   (``_collection_census.py``). The classes that name one table must descend from one class that
   names it -- a subclass adding queries is the same class for this purpose, the way
   ``HubGroupMemberCollection`` extends ``GroupMemberCollection`` -- and they must all declare the
   same write generation. Two unrelated classes for a table are two places its declaration can
   disagree, and a follower trusting "the generation did not move" is wrong whenever they do.
   A concrete class whose table cannot be read off the class (it is named per instance) is listed
   in :data:`_TABLES_NAMED_PER_INSTANCE` with why; the registry's own refusal
   (``CollectionRegistry.register``) covers those at run time.

**What rule 4 cannot reach, stated rather than implied.** Classes in other repositories: the hub
and the SDK each define a class for ``playbook_entries`` and ``concepts`` beside
``threetears.agent.knowledge``'s. No check in this repository sees them; the one-class cleanup
before the default flips removes them, and each repository runs this census over its own classes.

Rules 1 to 3 are AST-only; rule 4 imports, in a subprocess.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__: list[str] = []

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: every workspace member's source tree, discovered so a new package is covered the day it exists.
_SRC_GLOBS = ("packages/*/src", "packages/agent/*/src")

_DECLARATION = "write_generation"
_SWITCHED_ON = "WRITE_GENERATION"
_NAMED_DECLARATIONS = frozenset({_SWITCHED_ON, "WRITE_GENERATION_UNDECLARED"})
_OPT_OUT = "NoWriteGeneration"
_PASSES = frozenset({"generation_catchup_tick", "follow_generation_key"})

#: the module that defines ``follow_generation``, and the one whose watcher calls it and IS a pass.
_FOLLOW_DEFINERS = frozenset(
    {
        "packages/core/src/threetears/core/collections/registry.py",
        "packages/epoch/src/threetears/epoch/generation_tick.py",
    }
)


@dataclass(frozen=True)
class Violation:
    """one place a rule is broken."""

    source: str
    lineno: int
    symbol: str
    reason: str


def _source_roots(repo_root: Path) -> list[Path]:
    """every package source root under ``repo_root``.

    :param repo_root: the repository, or a planted tree shaped like one
    :ptype repo_root: Path
    :return: the roots, sorted
    :rtype: list[Path]
    """
    roots: set[Path] = set()
    for pattern in _SRC_GLOBS:
        roots.update(path for path in repo_root.glob(pattern) if path.is_dir())
    return sorted(roots)


def _modules(repo_root: Path) -> list[tuple[str, ast.Module]]:
    """every source module under ``repo_root``, parsed.

    :param repo_root: the repository, or a planted tree shaped like one
    :ptype repo_root: Path
    :return: ``(path relative to the root, tree)`` pairs
    :rtype: list[tuple[str, ast.Module]]
    """
    parsed: list[tuple[str, ast.Module]] = []
    for root in _source_roots(repo_root):
        for path in sorted(root.rglob("*.py")):
            parsed.append((path.relative_to(repo_root).as_posix(), ast.parse(path.read_text(encoding="utf-8"))))
    return parsed


def _declarations(node: ast.ClassDef) -> list[tuple[int, ast.expr]]:
    """the values a class body assigns to ``write_generation``.

    :param node: the class
    :ptype node: ast.ClassDef
    :return: ``(line, value)`` for each assignment that has a value
    :rtype: list[tuple[int, ast.expr]]
    """
    found: list[tuple[int, ast.expr]] = []
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign):
            targets: list[ast.expr] = [statement.target]
        elif isinstance(statement, ast.Assign):
            targets = list(statement.targets)
        else:
            continue
        if statement.value is not None and any(isinstance(t, ast.Name) and t.id == _DECLARATION for t in targets):
            found.append((statement.lineno, statement.value))
    return found


def _name_of(value: ast.expr) -> str | None:
    """the bare or dotted-tail name an expression refers to.

    :param value: the expression
    :ptype value: ast.expr
    :return: ``X`` for ``X`` or ``module.X``, else ``None``
    :rtype: str | None
    """
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def _declaration_problem(value: ast.expr) -> str | None:
    """why a ``write_generation`` value is not one of the three declarations, written out.

    :param value: the assigned expression
    :ptype value: ast.expr
    :return: the problem, or ``None`` when the value is a declaration
    :rtype: str | None
    """
    if _name_of(value) in _NAMED_DECLARATIONS:
        return None
    if isinstance(value, ast.Call) and _name_of(value.func) == _OPT_OUT:
        reasons = [keyword.value for keyword in value.keywords if keyword.arg == "reason"]
        if (
            len(reasons) == 1
            and not value.args
            and isinstance(reasons[0], ast.Constant)
            and isinstance(reasons[0].value, str)
            and reasons[0].value.strip()
        ):
            return None
        return "NoWriteGeneration takes reason=<a non-empty string literal>, written where it is declared"
    return (
        "write_generation is WRITE_GENERATION, WRITE_GENERATION_UNDECLARED or "
        "NoWriteGeneration(reason=...), named directly"
    )


def find_unreadable_declarations(repo_root: Path) -> list[Violation]:
    """every ``write_generation`` assignment that is not one of the three declarations, written out.

    :param repo_root: the repository, or a planted tree shaped like one
    :ptype repo_root: Path
    :return: the violations
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for source, tree in _modules(repo_root):
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for lineno, value in _declarations(node):
                problem = _declaration_problem(value)
                if problem is not None:
                    violations.append(Violation(source, lineno, node.name, problem))
    return violations


def find_rows_announced_without_their_advance(repo_root: Path) -> list[Violation]:
    """every switched-on class that publishes a row message of its own with no advance on it.

    :param repo_root: the repository, or a planted tree shaped like one
    :ptype repo_root: Path
    :return: the violations
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for source, tree in _modules(repo_root):
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(_name_of(value) == _SWITCHED_ON for _, value in _declarations(node)):
                continue
            for call in ast.walk(node):
                if (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "_publish_invalidation"
                    and not any(keyword.arg == "bump" for keyword in call.keywords)
                ):
                    violations.append(
                        Violation(
                            source,
                            call.lineno,
                            node.name,
                            "declares WRITE_GENERATION and publishes a row message with no bump=; the "
                            "generation never moves for that commit",
                        )
                    )
    return violations


def find_followers_without_a_pass(repo_root: Path) -> list[Violation]:
    """every module that follows a table's generation and schedules no pass to judge it.

    :param repo_root: the repository, or a planted tree shaped like one
    :ptype repo_root: Path
    :return: the violations
    :rtype: list[Violation]
    """
    violations: list[Violation] = []
    for source, tree in _modules(repo_root):
        if source in _FOLLOW_DEFINERS:
            continue
        follows = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "follow_generation"
        ]
        if not follows:
            continue
        named = {_name_of(node) for node in ast.walk(tree) if isinstance(node, ast.Name | ast.Attribute)}
        if not named & _PASSES:
            violations.append(
                Violation(
                    source,
                    follows[0].lineno,
                    "follow_generation",
                    "follows a table's write generation and schedules neither generation_catchup_tick "
                    "nor follow_generation_key, so a missed broadcast is never acted on",
                )
            )
    return violations


#: concrete collection classes whose table is named per instance, so no census can read it off
#: the class, with why. The registry refuses two disagreeing collections for one table at run time.
_TABLES_NAMED_PER_INSTANCE: dict[str, str] = {
    "threetears.core.coordination.tables.CoordinationCollection": (
        "the shared base of the four coordination tables; each subclass names its own schema"
    ),
    "threetears.geo.collection.TileCollection": "one table per cache scope, geo_tiles_{scope}",
    "threetears.geo.features.FeatureCache": "one table per cache scope, geo_features_{scope}",
}

_CENSUS = Path(__file__).resolve().parent / "_collection_census.py"


def run_census(roots: list[Path]) -> dict[str, Any]:
    """enumerate every collection class under ``roots``, in a process of its own.

    :param roots: package source roots to import
    :ptype roots: list[Path]
    :return: the census: ``classes`` and ``import_failures``
    :rtype: dict[str, Any]
    """
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(path for path in sys.path if path)}
    completed = subprocess.run(  # noqa: S603 - this interpreter, this repository's own script
        [sys.executable, str(_CENSUS), *(str(root) for root in roots)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    census: dict[str, Any] = json.loads(completed.stdout)
    return census


def find_census_problems(census: dict[str, Any], named_per_instance: dict[str, str]) -> list[str]:
    """every way the census breaks rule 4, or the enumeration itself is incomplete.

    :param census: what :func:`run_census` found
    :ptype census: dict[str, Any]
    :param named_per_instance: concrete classes whose table is named per instance, with why
    :ptype named_per_instance: dict[str, str]
    :return: one line per problem
    :rtype: list[str]
    """
    problems = [f"module did not import, so its classes were not enumerated: {f}" for f in census["import_failures"]]
    classes: list[dict[str, Any]] = census["classes"]
    names = {record["name"] for record in classes}
    has_subclass = {ancestor for record in classes for ancestor in record["ancestors"]}
    by_table: dict[str, list[dict[str, Any]]] = {}
    for record in classes:
        if record["declaration"] == "invalid":
            problems.append(f"{record['name']}: write_generation is not one of the three declarations")
        if record["declaration"] == "opted_out" and not str(record["reason"] or "").strip():
            problems.append(f"{record['name']}: NoWriteGeneration without a reason")
        if record["table"] is not None:
            by_table.setdefault(record["table"], []).append(record)
        elif not record["abstract"] and record["name"] not in has_subclass and record["name"] not in named_per_instance:
            problems.append(
                f"{record['name']}: its table cannot be read off the class; name it in a schema or a "
                f"table_name property, or list it in _TABLES_NAMED_PER_INSTANCE with why"
            )
    for stale in sorted(set(named_per_instance) - names):
        problems.append(f"_TABLES_NAMED_PER_INSTANCE lists {stale}, which the census no longer finds")
    for table, records in sorted(by_table.items()):
        members = {record["name"] for record in records}
        roots = sorted(record["name"] for record in records if not set(record["ancestors"]) & members)
        if len(roots) > 1:
            problems.append(f"table {table!r} is named by unrelated classes: {', '.join(roots)}")
        kinds = sorted({record["declaration"] for record in records})
        if len(kinds) > 1:
            problems.append(
                f"table {table!r} has classes declaring different write generations ({', '.join(kinds)}): "
                f"{', '.join(sorted(members))}"
            )
    return problems


def _assert_clean(violations: list[Violation]) -> None:
    """fail, listing every violation.

    :param violations: what a walker found
    :ptype violations: list[Violation]
    :return: nothing
    :rtype: None
    """
    assert not violations, "\n".join(f"{v.source}:{v.lineno} {v.symbol}: {v.reason}" for v in violations)


class TestTheFamily:
    def test_the_walkers_see_the_workspace(self) -> None:
        assert len(_source_roots(_REPO_ROOT)) > 20
        # the declaration's own default is a class-body assignment these walkers must be reading
        base = "packages/core/src/threetears/core/collections/base.py"
        tree = dict(_modules(_REPO_ROOT))[base]
        defaults = [
            _name_of(value)
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "BaseCollection"
            for _, value in _declarations(node)
        ]
        assert defaults == ["WRITE_GENERATION_UNDECLARED"]

    def test_every_declaration_is_one_of_the_three_and_an_opt_out_says_why(self) -> None:
        _assert_clean(find_unreadable_declarations(_REPO_ROOT))

    def test_no_switched_on_collection_announces_a_row_without_its_advance(self) -> None:
        _assert_clean(find_rows_announced_without_their_advance(_REPO_ROOT))

    def test_whoever_follows_a_table_schedules_the_pass(self) -> None:
        _assert_clean(find_followers_without_a_pass(_REPO_ROOT))

    def test_one_table_is_one_line_of_classes_with_one_declaration(self) -> None:
        census = run_census(_source_roots(_REPO_ROOT))
        # the census reaches the family: the classes this rule exists for are among what it found
        found = {record["name"]: record["table"] for record in census["classes"]}
        assert found["threetears.agent.knowledge.collections.ConceptCollection"] == "concepts"
        assert found["threetears.agent.acl.collections.GroupMemberCollection"] == "group_members"
        problems = find_census_problems(census, _TABLES_NAMED_PER_INSTANCE)
        assert not problems, "\n".join(problems)


def _plant(tmp_path: Path, name: str, source: str) -> Path:
    """write one module into a planted tree shaped like the workspace.

    :param tmp_path: the planted repository root
    :ptype tmp_path: Path
    :param name: the module's file name
    :ptype name: str
    :param source: its source
    :ptype source: str
    :return: the planted repository root
    :rtype: Path
    """
    package = tmp_path / "packages" / "planted" / "src" / "threetears" / "planted"
    package.mkdir(parents=True, exist_ok=True)
    (package / name).write_text(source, encoding="utf-8")
    return tmp_path


class TestTheWalkersBite:
    def test_a_flag_is_not_a_declaration(self, tmp_path: Path) -> None:
        root = _plant(tmp_path, "flag.py", "class Hot(Base):\n    write_generation = False\n")
        assert [v.symbol for v in find_unreadable_declarations(root)] == ["Hot"]

    def test_an_opt_out_with_no_reason_is_flagged(self, tmp_path: Path) -> None:
        root = _plant(tmp_path, "empty.py", 'class Hot(Base):\n    write_generation = NoWriteGeneration(reason="  ")\n')
        assert [v.symbol for v in find_unreadable_declarations(root)] == ["Hot"]

    def test_an_opt_out_whose_reason_is_not_written_there_is_flagged(self, tmp_path: Path) -> None:
        root = _plant(tmp_path, "hidden.py", "class Hot(Base):\n    write_generation = NoWriteGeneration(reason=WHY)\n")
        assert [v.symbol for v in find_unreadable_declarations(root)] == ["Hot"]

    def test_the_three_declarations_pass(self, tmp_path: Path) -> None:
        source = (
            "class A(Base):\n    write_generation = WRITE_GENERATION\n\n\n"
            "class B(Base):\n    write_generation = generation.WRITE_GENERATION_UNDECLARED\n\n\n"
            'class C(Base):\n    write_generation = NoWriteGeneration(reason="append-only; nothing reads a row by key")\n'
        )
        assert find_unreadable_declarations(_plant(tmp_path, "fine.py", source)) == []

    def test_a_switched_on_class_publishing_a_bare_row_is_flagged(self, tmp_path: Path) -> None:
        source = (
            "class Members(Base):\n"
            "    write_generation = WRITE_GENERATION\n\n"
            "    async def remove(self, key):\n"
            "        await self._publish_invalidation(key)\n"
        )
        found = find_rows_announced_without_their_advance(_plant(tmp_path, "bare.py", source))
        assert [(v.symbol, v.lineno) for v in found] == [("Members", 5)]

    def test_a_switched_on_class_publishing_with_its_advance_passes(self, tmp_path: Path) -> None:
        source = (
            "class Members(Base):\n"
            "    write_generation = WRITE_GENERATION\n\n"
            "    async def remove(self, key):\n"
            "        await self._publish_invalidation(key, bump=await self._bump_generation(1))\n"
        )
        assert find_rows_announced_without_their_advance(_plant(tmp_path, "bumped.py", source)) == []

    def test_a_class_that_is_not_switched_on_may_publish_a_bare_row(self, tmp_path: Path) -> None:
        source = "class Rooms(Base):\n    async def leave(self, key):\n        await self._publish_invalidation(key)\n"
        assert find_rows_announced_without_their_advance(_plant(tmp_path, "rooms.py", source)) == []

    def test_a_bootstrap_that_follows_and_schedules_no_pass_is_flagged(self, tmp_path: Path) -> None:
        source = "def wire(registry):\n    registry.follow_generation('role_assignments')\n"
        found = find_followers_without_a_pass(_plant(tmp_path, "bootstrap.py", source))
        assert [v.source for v in found] == ["packages/planted/src/threetears/planted/bootstrap.py"]

    def test_a_bootstrap_that_follows_and_schedules_a_pass_passes(self, tmp_path: Path) -> None:
        source = (
            "from threetears.epoch import generation_catchup_tick\n\n\n"
            "async def wire(registry, reader):\n"
            "    registry.follow_generation('role_assignments')\n"
            "    await generation_catchup_tick(registry, reader)\n"
        )
        assert find_followers_without_a_pass(_plant(tmp_path, "scheduled.py", source)) == []

    def test_two_unrelated_classes_for_one_table_are_flagged(self, tmp_path: Path) -> None:
        root = _plant(
            tmp_path,
            "twice.py",
            _PLANTED_HEAD + _planted("Concepts", "concepts") + _planted("AdminConcepts", "concepts"),
        )
        problems = find_census_problems(run_census(_source_roots(root)), {})
        assert problems == [
            "table 'concepts' is named by unrelated classes: "
            "threetears.planted.twice.AdminConcepts, threetears.planted.twice.Concepts"
        ]

    def test_a_subclass_of_the_tables_class_is_the_same_class(self, tmp_path: Path) -> None:
        source = _PLANTED_HEAD + _planted("Members", "group_members") + "\n\nclass HubMembers(Members):\n    pass\n"
        assert find_census_problems(run_census(_source_roots(_plant(tmp_path, "line.py", source))), {}) == []

    def test_a_subclass_that_declares_otherwise_is_flagged(self, tmp_path: Path) -> None:
        source = (
            _PLANTED_HEAD
            + _planted("Members", "group_members")
            + "\n\nclass HubMembers(Members):\n    write_generation = WRITE_GENERATION\n"
        )
        problems = find_census_problems(run_census(_source_roots(_plant(tmp_path, "split.py", source))), {})
        assert problems == [
            "table 'group_members' has classes declaring different write generations (on, undeclared): "
            "threetears.planted.split.HubMembers, threetears.planted.split.Members"
        ]

    def test_a_concrete_class_whose_table_cannot_be_read_is_flagged_unless_listed(self, tmp_path: Path) -> None:
        source = _PLANTED_HEAD + _planted("Scoped", None)
        census = run_census(_source_roots(_plant(tmp_path, "scoped.py", source)))
        assert [p.split(":")[0] for p in find_census_problems(census, {})] == ["threetears.planted.scoped.Scoped"]
        assert find_census_problems(census, {"threetears.planted.scoped.Scoped": "one per scope"}) == []

    def test_a_module_that_does_not_import_is_flagged(self, tmp_path: Path) -> None:
        root = _plant(tmp_path, "broken.py", "import threetears.planted.nowhere\n")
        problems = find_census_problems(run_census(_source_roots(root)), {})
        assert len(problems) == 1 and problems[0].startswith("module did not import")


#: the head of a planted module of collection classes the census imports for real.
_PLANTED_HEAD = """from typing import Any

from threetears.core.collections import WRITE_GENERATION, BaseCollection
"""


def _planted(name: str, table: str | None) -> str:
    """a concrete planted collection class naming ``table`` (or a table named per instance).

    :param name: the class name
    :ptype name: str
    :param table: the literal table name, or ``None`` for one read off the instance
    :ptype table: str | None
    :return: the class source
    :rtype: str
    """
    answer = repr(table) if table is not None else "self.scoped_table"
    return (
        f"\n\nclass {name}(BaseCollection[Any]):\n"
        f"    @property\n    def table_name(self) -> str:\n        return {answer}\n\n"
        f"    @property\n    def entity_class(self) -> Any:\n        return dict\n\n"
        f"    async def fetch_from_store(self, entity_id: Any) -> Any:\n        return None\n\n"
        f"    async def save_to_store(self, data: Any, original_timestamp: Any = None, *, conn: Any = None) -> int:\n"
        f"        return 0\n\n"
        f"    async def delete_from_store(self, entity_id: Any) -> None:\n        return None\n\n"
        f"    def serialize(self, data: Any) -> bytes:\n        return b''\n\n"
        f"    def deserialize(self, data: bytes) -> Any:\n        return {{}}\n"
    )
