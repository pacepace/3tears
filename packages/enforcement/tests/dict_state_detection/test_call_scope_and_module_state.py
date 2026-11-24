"""tests for the widened walk's call-scope proof (what it now sees, and what stays unsure) and module-level state."""

from __future__ import annotations

from pathlib import Path

import pytest

from threetears.enforcement.dict_state_detection import (
    DictStateConfig,
    find_mutated_state_violations,
    run_dict_state_enforcement,
)


def _repo(tmp_path: Path, modules: dict[str, str]) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    root = repo / "src"
    for name, source in modules.items():
        module = root / "pkg" / name
        module.parent.mkdir(parents=True, exist_ok=True)
        module.write_text(source)
    return repo, root


def _symbols(tmp_path: Path, source: str, **more: str) -> set[str]:
    repo, root = _repo(tmp_path, {"mod.py": source, **{f"{name}.py": text for name, text in more.items()}})
    return {violation.symbol for violation in find_mutated_state_violations((root,), repo, widened=True)}


#: an accumulator handed to builders picked from a dispatch table, as the hub compiler's emitter does
_DISPATCH = """\
class Emission:
    def __init__(self):
        self._pending = []

def _one(emission, step):
    emission._pending.append(step)

def _two(emission, step):
    emission._pending.append(step)

{table}

def build(steps):
    emission = Emission()
    for step in steps:
        {call}
    return len(emission._pending)
"""


@pytest.mark.parametrize(
    ("table", "call"),
    [
        (
            "",
            "builders = {'a': _one, 'b': _two}\n        builder = builders.get(step)\n        builder(emission, step)",
        ),
        ("_BUILDERS = {'a': _one, 'b': _two}", "_BUILDERS[step](emission, step)"),
        ("_BUILDERS = {'a': _one, 'b': _two}", "builder = _BUILDERS.get(step)\n        builder(emission, step)"),
    ],
    ids=["local-table-get", "module-table-subscript", "module-table-get"],
)
def test_a_value_handed_through_a_dispatch_table_of_functions_is_followed(
    tmp_path: Path, table: str, call: str
) -> None:
    assert _symbols(tmp_path, _DISPATCH.format(table=table, call=call)) == set()


@pytest.mark.parametrize(
    ("table", "call"),
    [
        ("_BUILDERS = {'a': _one, 'b': _two}\n_BUILDERS['c'] = print", "_BUILDERS[step](emission, step)"),
        ("_BUILDERS = {'a': _one, 'b': KEEP.append}\nKEEP = []", "_BUILDERS[step](emission, step)"),
        (
            "_BUILDERS = {'a': _one, 'b': _two}\n\ndef _keeper(emission, step):\n    KEPT.append(emission)\n\n"
            "_BUILDERS = {'a': _keeper}",
            "_BUILDERS[step](emission, step)",
        ),
        (
            "KEPT = []\n\ndef _keeper(emission, step):\n    KEPT.append(emission)\n\n_BUILDERS = {'a': _one, 'b': _keeper}",
            "_BUILDERS[step](emission, step)",
        ),
    ],
    ids=["table-changed", "value-not-a-function", "table-rebound", "one-entry-keeps-it"],
)
def test_an_unprovable_dispatch_table_leaves_the_value_unsure(tmp_path: Path, table: str, call: str) -> None:
    assert "Emission._pending" in _symbols(tmp_path, _DISPATCH.format(table=table, call=call))


def _chain(depth: int) -> str:
    """an accumulator built in ``build`` and handed down ``depth`` calls, the last one filling it."""
    lines = ["class Walk:", "    def __init__(self):", "        self._edges = []", ""]
    for level in range(depth):
        lines += [f"def _f{level}(walk, item):", f"    _f{level + 1}(walk, item)", ""]
    lines += [f"def _f{depth}(walk, item):", "    walk._edges.append(item)", ""]
    lines += [
        "def build(items):",
        "    walk = Walk()",
        "    for item in items:",
        "        _f0(walk, item)",
        "    return 1",
    ]
    return "\n".join(lines) + "\n"


def test_a_value_handed_down_to_the_depth_limit_is_followed_and_one_call_deeper_is_unsure(tmp_path: Path) -> None:
    # build -> _f0 is one call; _f0 .. _fN each one more: 13 hops is the deepest followed (_MAX_DEPTH 14)
    assert _symbols(tmp_path / "deep", _chain(12)) == set()
    assert "Walk._edges" in _symbols(tmp_path / "deeper", _chain(16))


#: a budget filled a page at a time by a nested coroutine that hands it to one awaited worker thread
_THREADED = """\
import asyncio

class Budget:
    def __init__(self):
        self._rows = []

def _take(budget, page):
    budget._rows.extend(page)
    return True

async def fetch(pages, run):
    budget = Budget()

    async def _paged(conn):
        for page in pages:
            {hand_off}

    await run(_paged)
    return len(budget._rows)
"""


def test_a_value_a_nested_coroutine_hands_to_one_awaited_worker_thread_stays_in_the_call(tmp_path: Path) -> None:
    assert _symbols(tmp_path, _THREADED.format(hand_off="await asyncio.to_thread(_take, budget, page)")) == set()


@pytest.mark.parametrize(
    "hand_off",
    [
        "asyncio.to_thread(_take, budget, page)",
        "asyncio.get_running_loop().run_in_executor(None, _take, budget, page)",
        "asyncio.ensure_future(asyncio.to_thread(_take, budget, page))",
    ],
    ids=["thread-not-awaited", "executor", "task"],
)
def test_a_value_handed_to_a_thread_nobody_waits_for_is_unsure(tmp_path: Path, hand_off: str) -> None:
    assert "Budget._rows" in _symbols(tmp_path, _THREADED.format(hand_off=hand_off))


def test_a_closure_kept_beyond_its_call_is_refused(tmp_path: Path) -> None:
    source = _THREADED.format(hand_off="await asyncio.to_thread(_take, budget, page)").replace(
        "    await run(_paged)\n", "    HANDLERS.append(_paged)\n"
    )
    assert "Budget._rows" in _symbols(tmp_path, "HANDLERS = []\n" + source)


#: one stream's parser state, held by a reading built per answer, as brave_answers' readers are
_FIELDS = """\
from dataclasses import dataclass, field

class TagReader:
    def __init__(self):
        self._text = []

    def feed(self, chunk):
        self._text.append(chunk)

@dataclass
class Reading:
    tags: TagReader = field(default_factory=TagReader)

def read(chunks):
    reading = Reading()
    for chunk in chunks:
        reading.tags.feed(chunk)
    return len(reading.tags._text)
"""


def test_a_field_built_by_default_factory_lives_as_long_as_its_call_scoped_holder(tmp_path: Path) -> None:
    assert _symbols(tmp_path, _FIELDS) == set()
    kept = _FIELDS + "\nclass Service:\n    def __init__(self):\n        self.reading = Reading()\n"
    assert "TagReader._text" in _symbols(tmp_path / "kept", kept)


def test_a_same_named_class_in_another_module_does_not_decide_this_one(tmp_path: Path) -> None:
    other = "class Reading:\n    pass\n\nKEPT = []\n\ndef keep():\n    KEPT.append(Reading())\n"
    assert {s for s in _symbols(tmp_path, _FIELDS, other=other) if not s.startswith("(module)")} == set()
    imported = (
        other + "\n" + "from pkg.mod import Reading as Imported\n\ndef keep_theirs():\n    KEPT.append(Imported())\n"
    )
    assert "TagReader._text" in _symbols(tmp_path / "imported", _FIELDS, other=imported.replace("Imported", "Reading"))


#: a value returned back up to the one call that handed it in, while that function's other callers keep theirs
_RETURNED = """\
class Report:
    def __init__(self):
        self._names = []

def _passed(value):
    return value

KEPT = []

def fill(names):
    report = _passed(Report())
    report._names.extend(names)
    return len(report._names)

def keep(thing):
    KEPT.append(_passed(thing))
"""


def test_a_value_returned_goes_back_only_to_the_call_that_handed_it_in(tmp_path: Path) -> None:
    assert {s for s in _symbols(tmp_path, _RETURNED) if not s.startswith("(module)")} == set()
    leaked = _RETURNED.replace("    return len(report._names)\n", "    KEPT.append(_passed(report))\n    return 0\n")
    assert "Report._names" in _symbols(tmp_path / "leaked", leaked)


#: gap (d): module-level state
_MODULE_STATE = """\
from collections import deque
from types import MappingProxyType

_SEEN = set()
_BY_KEY: dict[str, int] = {}
_RECENT = deque(maxlen=8)
_COUNTS = {}
_TABLE = {"a": 1}
_FROZEN = MappingProxyType({"a": 1})
_LIST = []

for _key in ("x", "y"):
    _COUNTS[_key] = 0

def note(key):
    _SEEN.add(key)
    _BY_KEY[key] = 1
    entry = _BY_KEY.setdefault(key, 0)

def recent(item):
    _RECENT.append(item)

def grow():
    global _LIST
    _LIST += [1]

def rebind():
    global _FROZEN
    _FROZEN = MappingProxyType({**_FROZEN, "b": 2})

def shadow(_SEEN):
    _SEEN.add(1)

def local():
    _COUNTS = {}
    _COUNTS["z"] = 1
    return _TABLE["a"]

class Holder:
    def touch(self, key):
        _BY_KEY.pop(key, None)
"""


def test_module_level_containers_changed_in_place_are_refused_and_constants_pass(tmp_path: Path) -> None:
    symbols = _symbols(tmp_path, _MODULE_STATE)
    assert symbols == {"(module)._SEEN", "(module)._BY_KEY", "(module)._RECENT", "(module)._LIST"}


def test_module_level_state_is_widened_only_and_honours_non_serving_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DICT_STATE_ENFORCEMENT_MODE", "strict")
    repo, root = _repo(tmp_path, {"mod.py": _MODULE_STATE})
    assert find_mutated_state_violations((root,), repo, widened=False) == []
    with pytest.raises(pytest.fail.Exception, match=r"\(module\)\._SEEN"):
        run_dict_state_enforcement(
            DictStateConfig(repo_root=repo, src_roots=(root,), immutable_roots=(root,), immutable_widened=True)
        )
    run_dict_state_enforcement(
        DictStateConfig(
            repo_root=repo,
            src_roots=(root,),
            immutable_roots=(root,),
            immutable_widened=True,
            non_serving_roots=(root / "pkg" / "mod.py",),
        )
    )


def test_an_asyncio_queue_is_the_approved_hand_off_and_a_deque_popped_in_place_is_not(tmp_path: Path) -> None:
    queue = """\
import asyncio

class Backlog:
    def __init__(self):
        self._held: asyncio.Queue[int] = asyncio.Queue()
        self.size = 0

    async def put(self, item):
        self._held.put_nowait(item)

    async def get(self):
        return self._held.get_nowait()

def make():
    return Backlog()

KEEP = make()
"""
    assert _symbols(tmp_path, queue) == set()
    popped = (
        queue.replace("asyncio.Queue[int] = asyncio.Queue()", "deque[int] = deque()")
        .replace("self._held.put_nowait(item)", "self._held.append(item)")
        .replace("self._held.get_nowait()", "self._held.popleft()")
    )
    assert _symbols(tmp_path / "deque", "from collections import deque\n" + popped) == {"Backlog._held"}


def test_the_init_walker_lets_a_call_scoped_object_start_with_a_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DICT_STATE_ENFORCEMENT_MODE", "strict")
    source = """\
class Wall:
    def __init__(self, rows):
        self._edges = {}
        for row in rows:
            self._edges.setdefault(row, set()).add(row)

def visible(rows):
    wall = Wall(rows)
    return len(wall._edges)
"""
    repo, root = _repo(tmp_path, {"mod.py": source})
    config = DictStateConfig(repo_root=repo, src_roots=(root,), immutable_roots=(root,), immutable_widened=True)
    run_dict_state_enforcement(config, walker="detect")
    (root / "pkg" / "keeper.py").write_text(
        "from pkg.mod import Wall\n\nclass Service:\n    def __init__(self):\n        self.wall = Wall(())\n"
    )
    with pytest.raises(pytest.fail.Exception, match="Wall._edges"):
        run_dict_state_enforcement(config, walker="detect")


#: the weak containers changed in place, through an object and at module level (nats' ``_BOUNDED`` before 2026-11-24)
_WEAK_IN_PLACE = """\
import weakref
from weakref import WeakValueDictionary

_BOUNDED: weakref.WeakSet[object] = weakref.WeakSet()
_ASKED = weakref.WeakKeyDictionary()

def bound(transport):
    _BOUNDED.add(transport)

def ask(reader, answer):
    _ASKED[reader] = answer

class Ledger:
    def __init__(self):
        self._stores = WeakValueDictionary()
        self._backends: weakref.WeakSet[object] = weakref.WeakSet()

    def join(self, store):
        self._stores[id(store)] = store
        self._backends.add(store)

    def leave(self, store):
        self._stores.pop(id(store), None)
        self._backends.discard(store)

def keep(ledger):
    ledger_ref = Ledger()
    return ledger_ref
"""

#: the approved shape: a frozen holder of weak references, replaced whole
_WEAK_REPLACED = """\
from threetears.observe import FrozenWeakSet, FrozenWeakValues

_BOUNDED: FrozenWeakSet[object] = FrozenWeakSet()

def bound(transport):
    global _BOUNDED
    _BOUNDED = _BOUNDED.with_item(transport)

class Ledger:
    def __init__(self):
        self._stores: FrozenWeakValues[int, object] = FrozenWeakValues()

    def join(self, store):
        self._stores = self._stores.with_entry(id(store), store)

    def leave(self, store):
        self._stores = self._stores.without_key(id(store))
"""


def test_weak_containers_changed_in_place_are_refused(tmp_path: Path) -> None:
    assert _symbols(tmp_path, _WEAK_IN_PLACE) == {
        "(module)._BOUNDED",
        "(module)._ASKED",
        "Ledger._stores",
        "Ledger._backends",
    }


def test_frozen_weak_holders_replaced_whole_pass(tmp_path: Path) -> None:
    assert _symbols(tmp_path, _WEAK_REPLACED) == set()
