"""Every writer of an existing campaign takes the campaign write lock — derived from the code, not listed.

A campaign document carries no ETag, so each edit of one is a blind read-modify-write, and two that
interleave lose the first one's change (``threetears.evals.contracts.campaign_writes._CAMPAIGN_WRITES``). The lock is taken
by decorating each writer with ``serialized_campaign_write``, and a decorator list fails by omission:
a new archive, rename or membership path written without it brings the lost-update race back with
nothing going red. So the population here is every function in the package source that calls
``save_campaign`` — the act that makes it a writer — and each must be decorated or be the one path
that writes a campaign no other writer can yet hold.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1] / "src"
_DECORATOR = "serialized_campaign_write"

# A writer that saves a campaign that does not exist yet: nothing else can have read it, so there
# is no read-modify-write to lose. Keyed by (path, qualified name) so a same-named function elsewhere
# is not excused by it.
_CREATES: frozenset[tuple[str, str]] = frozenset({("threetears/evals/analysis/campaigns.py", "create_campaign")})


def _decorator_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names: set[str] = set()
    for decorator in fn.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _saves_a_campaign(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``fn``'s own body calls ``save_campaign`` — a nested function's call is its own."""
    pending: list[ast.AST] = list(fn.body)
    while pending:
        node = pending.pop()
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "save_campaign":
            return True
        pending.extend(ast.iter_child_nodes(node))
    return False


def _visit(node: ast.AST, prefix: str, rel: str, found: list[tuple[str, str, bool]]) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            _visit(child, f"{prefix}{child.name}.", rel, found)
        elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
            qualname = f"{prefix}{child.name}"
            if child.name != "save_campaign" and _saves_a_campaign(child):
                found.append((rel, qualname, _DECORATOR in _decorator_names(child)))
            _visit(child, f"{qualname}.", rel, found)


@functools.cache
def _campaign_writers() -> list[tuple[str, str, bool]]:
    """Every function in the package source that calls ``save_campaign``, with whether it is decorated.

    A storage backend's own ``save_campaign`` (and a protocol declaring it) is the write, not a
    writer, and is left out by name. Nested functions are reported under their enclosing chain, so a
    closure that saves is attributed to itself rather than excused by its parent's decorator.
    """
    found: list[tuple[str, str, bool]] = []
    for path in sorted((_ROOT / "threetears").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        # A caller names the method in its source, so a file without the name holds none; skipping its
        # parse narrows no population and keeps the walk off the whole tree's AST.
        if "save_campaign" not in source:
            continue
        rel = path.relative_to(_ROOT).as_posix()
        _visit(ast.parse(source, filename=rel), "", rel, found)
    return found


def test_the_population_is_the_one_the_lock_was_written_for():
    """Positive control: the walk finds the writers known today, so a green below is not an empty set."""
    writers = {(rel, name) for rel, name, _ in _campaign_writers()}

    for expected in (
        ("threetears/evals/analysis/campaigns.py", "remove_runs_from_campaign"),
        ("threetears/evals/run/curation.py", "_detach_run_from_all_campaigns"),
        ("threetears/evals/analysis/campaigns.py", "update_campaign"),
        ("threetears/evals/analysis/campaigns.py", "add_runs_to_campaign"),
        ("threetears/evals/analysis/campaigns.py", "set_campaign_control"),
        *_CREATES,
    ):
        assert expected in writers, (
            f"{expected} no longer saves a campaign — the walk or this list is wrong: {sorted(writers)}"
        )


def test_every_writer_of_an_existing_campaign_holds_the_lock():
    unserialized = [
        f"{rel}::{name}"
        for rel, name, decorated in _campaign_writers()
        if not decorated and (rel, name) not in _CREATES
    ]

    assert unserialized == [], (
        "these save a campaign without @campaign_writes.serialized_campaign_write, so a concurrent writer can "
        f"read the same document and one of the two saves is lost: {unserialized}"
    )


def test_the_create_exemption_still_names_a_create():
    """An exemption whose function started editing an existing campaign would excuse exactly the race."""
    for rel, name in _CREATES:
        assert name.rsplit(".", 1)[-1].startswith("create_"), f"{rel}::{name} is exempted as a create path"
