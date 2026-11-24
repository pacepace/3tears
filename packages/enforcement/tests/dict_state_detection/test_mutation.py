"""tests for the walker refusing an object's dict or set state changed in place after ``__init__``."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.dict_state_detection import (
    DictStateAllowlistEntry,
    DictStateConfig,
    find_mutated_state_violations,
    find_stale_immutable_debts,
    run_dict_state_enforcement,
)

_RATIONALE = "one record a key in flight, a redesign of the fence to agree first"


def _module(tmp_path: Path, source: str) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    module = repo / "src" / "pkg" / "mod.py"
    module.parent.mkdir(parents=True)
    module.write_text(source)
    return repo, repo / "src"


def _found(tmp_path: Path, source: str) -> list[tuple[int, str]]:
    repo, root = _module(tmp_path, source)
    return [(violation.line, violation.symbol) for violation in find_mutated_state_violations((root,), repo)]


#: the copy follower's bookkeeping before 2026-11-23, in small: each change in place the walker refuses
_OLD_FOLLOWER = """\
class Follower:
    def __init__(self, tables):
        self._heard: dict[str, tuple] = {name: () for name in tables}
        self._counts = dict.fromkeys(tables, 0)
        self._whole: dict[str, str] = {}
        self._file_waits: dict[str, dict[str, object]] = {}
        self._dropping: set[str] = set()
        self._partitions = {name: [] for name in tables}

    def hear(self, table, batch):
        self._heard[table] = (*self._heard[table], batch)
        self._counts[table] += 1

    def read_whole(self, table, why):
        self._whole.setdefault(table, why)
        self._partitions[table].clear()

    def wait_for_file(self, table, file, wait):
        waits = self._file_waits.setdefault(table, {})
        waits[file] = wait

    def file_read(self, table, name):
        waits = self._file_waits.get(table, {})
        waits.pop(name, None)
        del self._file_waits[table]

    def forget(self, name):
        self._dropping.update((name,))
        self._dropping |= {name}
"""


def test_the_old_follower_bookkeeping_is_refused_at_each_change(tmp_path: Path) -> None:
    assert _found(tmp_path, _OLD_FOLLOWER) == [
        (11, "Follower._heard"),
        (12, "Follower._counts"),
        (15, "Follower._whole"),
        (16, "Follower._partitions"),
        (19, "Follower._file_waits"),
        (20, "Follower._file_waits"),
        (24, "Follower._file_waits"),
        (25, "Follower._file_waits"),
        (28, "Follower._dropping"),
        (29, "Follower._dropping"),
    ]


def test_immutable_values_replaced_whole_pass(tmp_path: Path) -> None:
    source = """\
from types import MappingProxyType


class Follower:
    def __init__(self, tables):
        self._heard = MappingProxyType({name: () for name in tables})
        self._dropping: frozenset[str] = frozenset()
        self._count = 0

    def hear(self, table, batch):
        self._heard = MappingProxyType({**self._heard, table: (*self._heard[table], batch)})
        self._dropping = self._dropping | {table}
        self._count += 1
        local = {}
        local[table] = batch
        held = self._heard.get(table)
        count = len(held)
        count += 1
"""
    assert _found(tmp_path, source) == []


def test_a_change_in_init_passes_and_one_in_a_closure_does_not(tmp_path: Path) -> None:
    source = """\
class Owner:
    def __init__(self):
        self._tasks: dict[str, object] = {}
        self._tasks["first"] = None

    def start(self, name, task):
        def done():
            self._tasks.pop(name, None)

        return done
"""
    assert _found(tmp_path, source) == [(8, "Owner._tasks")]


def test_an_entry_changed_through_its_container_is_refused(tmp_path: Path) -> None:
    source = """\
class Owner:
    def __init__(self):
        self._by_table: dict[str, list[str]] = {}

    def add(self, table, name):
        self._by_table[table].append(name)
        self._by_table.get(table).append(name)
"""
    assert _found(tmp_path, source) == [(6, "Owner._by_table"), (7, "Owner._by_table")]


def test_the_runner_refuses_a_change_the_allowlist_names_and_tracks_a_named_debt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DICT_STATE_ENFORCEMENT_MODE", "strict")
    repo, root = _module(tmp_path, _OLD_FOLLOWER)
    (repo / "pyproject.toml").write_text('[project]\nname = "synthetic"\n')
    allowed = tuple(
        DictStateAllowlistEntry(file="src/pkg/mod.py", class_name="Follower", attr_name=name, rationale=_RATIONALE)
        for name in ("_whole", "_file_waits")
    )
    config = DictStateConfig(repo_root=repo, src_roots=(root,), allowlist=allowed, immutable_roots=(root,))
    with pytest.raises(pytest.fail.Exception, match="Follower._file_waits"):
        run_dict_state_enforcement(config, walker="detect")
    debts = tuple(
        DictStateAllowlistEntry(file="src/pkg/mod.py", class_name="Follower", attr_name=name, rationale=_RATIONALE)
        for name in ("_heard", "_counts", "_whole", "_partitions", "_file_waits", "_dropping")
    )
    tracked = DictStateConfig(
        repo_root=repo, src_roots=(root,), allowlist=allowed, immutable_roots=(root,), immutable_known_violations=debts
    )
    run_dict_state_enforcement(tracked, walker="detect")


def test_a_debt_paid_is_reported_stale(tmp_path: Path) -> None:
    repo, root = _module(tmp_path, "class Owner:\n    def __init__(self):\n        self._x = frozenset()\n")
    debt = DictStateAllowlistEntry(file="src/pkg/mod.py", class_name="Owner", attr_name="_x", rationale=_RATIONALE)
    stale = find_stale_immutable_debts((root,), (debt,), repo)
    assert [violation.symbol for violation in stale] == ["Owner._x"]
    assert stale[0].category == "dict_state_detection.stale_immutable_debt"
