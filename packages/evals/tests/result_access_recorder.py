"""Which attributes of an ``EvalResult`` a reader reads: a test-only recorder over hydrated results (W4b P1).

The compact per-result reading (``ResultReading``) carries exactly the attributes the readers access, so that set is
measured, not guessed. :func:`recording_results` serves every result a store hydrates as a **recorded result** while
the block runs, and :func:`recording` collects, for the reads made inside it, each attribute path read and every call
site that read it.

**How a result is recorded.** Every ``EvalResult`` the engine hydrates is read through ``EvalResult.from_dict`` (the
storage's one stored read, ``EvalStorage._hydrate``), so the recorder patches that classmethod: the real stored read
runs, upgrades and strict validation unchanged, and the model it returns has its class swapped, in place, for a
recording subclass of its own class (``RecordedEvalResult``). Each nested model it holds (a rubric score, a usage row,
a goal outcome) is swapped the same way for a recording subclass of its class, which knows the path it sits at
(``rubric_scores[]``, ``transcript_score``, ``usage[]``). Nothing is copied or re-validated, and a recorded result is
an instance of ``EvalResult``, so an ``isinstance`` check, a ``model_copy`` (which keeps the class), a ``model_dump``
and pydantic's serialization read it exactly as they read the result it was: a proxy object would fail all four. A
recorded model compares as the model it records (``==`` against an unrecorded one, or one recorded at another path,
is the two models' own equality, which requires one class) and prints under that model's name.

**A reader that reads results as their readings** (``ResultReading.of``, which the storage builds for each result
it hydrates for a read of readings) is served the recorded result itself in the reading's place: a reading carries
the attributes of its result under the same names, so what a reader reads of it is what it reads of the result, and
it is recorded where the reader reads it. Building a real reading would read every attribute it carries, at the
reading's own module, and record none of the reader's own reads. That a reading answers each reader as its result
does is proved apart (``test_readings_match_results.py`` and the oracle).

**What a path looks like.** A top-level field is its name (``eval_run_id``); a field of a nested model is its owner's
path then its own (``transcript_score.score``); an item of a list is ``[]`` (``rubric_scores[].score``,
``usage[].cost_usd``) and a value of a mapping ``{}``. A method is its name with ``()`` (``judge_scores()``), and the
reads it makes are recorded too, at the model's own module. A read of the whole object is recorded under the name of
the operation: ``model_dump()``, ``model_copy()``, ``to_dict()``, ``__dict__``, ``==``, ``hash()``, ``repr()``, ``iter()``.

**Judge prose** (:data:`PROSE_FIELDS`) is recorded as a truth test, ``bool(judge_reasoning)``, where the read's own
expression only asks whether it is empty (an operand of ``bool()``, ``not``, an ``if`` / ``while`` / conditional
test, a comprehension's filter, or a boolean operator in one of those), found from the call site's bytecode position
and the parsed source; anywhere else it is a read of the text. Each prose read also counts the distinct results whose
prose it read (:class:`ProseReads`).

**What it cannot see**, so a reader of its output knows: ``isinstance(x, EvalResult)`` (a recorded result passes it),
pydantic validators (they run at hydration, before the class is swapped), and reads made inside pydantic itself (a
``model_dump`` is recorded as one whole-object read, not as the fields it serializes).
"""

from __future__ import annotations

import ast
import inspect
import linecache
import weakref
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType, FrameType
from typing import Any
from unittest import mock

import pydantic
from pydantic import BaseModel

import threetears.evals as evals_package
from threetears.evals.kernel.result_reading import ResultReading, SharedValues
from threetears.evals.schema.models import EvalResult

__all__ = [
    "PROSE_FIELDS",
    "WHOLE_OBJECT_READS",
    "ProseReads",
    "RecordingModel",
    "ResultAccess",
    "recording",
    "recording_results",
]

#: The judge-prose fields: a score's reasoning, the judge's top-level reasoning, and the judge's "cannot tell" notes.
PROSE_FIELDS = frozenset({"reasoning", "judge_reasoning", "cannot_tell", "judge_cannot_tell"})

#: The reads of a whole object, which a slim reading cannot serve field by field.
WHOLE_OBJECT_READS = frozenset(
    {
        "model_dump()",
        "model_dump_json()",
        "model_copy()",
        "to_dict()",
        "to_json()",
        "model_fields_set",
        "__dict__",
        "==",
        "hash()",
        "repr()",
        "str()",
        "iter()",
        "copy()",
        "deepcopy()",
    }
)

#: Where the source tree a call site is reported relative to starts.
_SRC = Path(evals_package.__file__).resolve().parent

#: Each call site's name, by code object and line.
_SITES: dict[tuple[CodeType, int], str] = {}

#: Where pydantic's own source lives: a read it makes (a ``model_copy`` copying, a ``model_dump`` serializing) is not a
#: reader's.
_PYDANTIC = str(Path(pydantic.__file__).parent)


@dataclass
class ProseReads:
    """Every read of one prose path at one call site.

    :ivar events: How many reads were made there.
    :ivar results: The ids of the distinct results whose prose was read there.
    """

    events: int = 0
    results: set[str] = field(default_factory=set)


@dataclass
class ResultAccess:
    """What the reads inside one :func:`recording` block read of the recorded results.

    :ivar sites: Each attribute path read, mapped to every call site that read it (``module.py:line function``).
    :ivar prose: Each prose path read, mapped to its call sites and what each read (:class:`ProseReads`).
    """

    sites: dict[str, set[str]] = field(default_factory=dict)
    prose: dict[str, dict[str, ProseReads]] = field(default_factory=dict)

    @property
    def paths(self) -> frozenset[str]:
        """Every attribute path read."""
        return frozenset(self.sites)

    def whole_object_reads(self) -> dict[str, set[str]]:
        """The reads of a whole result or nested model, by path, with their call sites."""
        return {path: sites for path, sites in self.sites.items() if path.rsplit(".", 1)[-1] in WHOLE_OBJECT_READS}

    def merge(self, other: ResultAccess) -> None:
        """Add ``other``'s reads to these."""
        for path, sites in other.sites.items():
            self.sites.setdefault(path, set()).update(sites)
        for path, by_site in other.prose.items():
            mine = self.prose.setdefault(path, {})
            for site, reads in by_site.items():
                held = mine.setdefault(site, ProseReads())
                held.events += reads.events
                held.results |= reads.results


class _Session:
    """The one recording block running: what it read, and the caches its classification fills."""

    access: ResultAccess | None = None


_SESSION = _Session()

#: The result id each recorded object belongs to, by ``id()``, for the objects that carry prose. An entry is dropped
#: when its object is collected (``weakref.finalize``), so an id reused later is never mistaken for it.
_OWNERS: dict[int, str] = {}

#: Each recording class, by the class it records and the path its instances sit at.
_CLASSES: dict[tuple[type[BaseModel], str], type[BaseModel]] = {}

#: Each source file parsed for prose classification: its Attribute nodes by position, and each node's parent.
_PARSED: dict[str, tuple[dict[tuple[int, int, int, int], ast.Attribute], dict[ast.AST, ast.AST]]] = {}

#: Each call site's truth-only verdict, by code object and instruction offset.
_TRUTH: dict[tuple[CodeType, int], bool] = {}


def _joined(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _site(frame: FrameType) -> str:
    """``frame``'s call site: its file relative to the evals source (or its name, outside it), line and function."""
    key = (frame.f_code, frame.f_lineno)
    site = _SITES.get(key)
    if site is None:
        path = Path(frame.f_code.co_filename).resolve()
        shown = str(path.relative_to(_SRC)) if path.is_relative_to(_SRC) else path.name
        site = _SITES[key] = f"{shown}:{frame.f_lineno} {frame.f_code.co_name}"
    return site


def _caller() -> FrameType | None:
    """The frame that made the read, or None when pydantic made it.

    Always two frames above :func:`_note`'s caller's caller (``_caller`` <- ``_note`` <- the recording hook <- the
    reader), found by depth rather than by walking: a frame's attributes are audited, and a test runner's audit hook
    pays for each one on every read.
    """
    frame = inspect.currentframe()
    for _ in range(3):
        if frame is None:
            return None
        frame = frame.f_back
    if frame is None or frame.f_code.co_filename.startswith(_PYDANTIC):
        return None
    return frame


def _parsed(filename: str) -> tuple[dict[tuple[int, int, int, int], ast.Attribute], dict[ast.AST, ast.AST]]:
    """``filename``'s Attribute nodes by their source position, and every node's parent."""
    held = _PARSED.get(filename)
    if held is None:
        source = "".join(linecache.getlines(filename))
        nodes: dict[tuple[int, int, int, int], ast.Attribute] = {}
        parents: dict[ast.AST, ast.AST] = {}
        if source:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    parents[child] = node
                if isinstance(node, ast.Attribute) and node.end_lineno is not None and node.end_col_offset is not None:
                    nodes[(node.lineno, node.end_lineno, node.col_offset, node.end_col_offset)] = node
        held = _PARSED[filename] = (nodes, parents)
    return held


def _in_truth_context(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether ``node``'s value is only asked whether it is truthy where it is read."""
    child: ast.AST = node
    while True:
        parent = parents.get(child)
        if isinstance(parent, ast.Call):
            return isinstance(parent.func, ast.Name) and parent.func.id == "bool" and parent.args == [child]
        if isinstance(parent, ast.UnaryOp):
            return isinstance(parent.op, ast.Not)
        if isinstance(parent, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            return parent.test is child
        if isinstance(parent, ast.comprehension):
            return child in parent.ifs
        if isinstance(parent, ast.BoolOp):
            child = parent
            continue
        return False


def _truth_only(frame: FrameType) -> bool:
    """Whether the read ``frame`` is making is a truth test of the value (see the module docstring)."""
    key = (frame.f_code, frame.f_lasti)
    verdict = _TRUTH.get(key)
    if verdict is None:
        positions = list(frame.f_code.co_positions())
        index = frame.f_lasti // 2
        verdict = False
        if 0 <= index < len(positions):
            lineno, end_lineno, col, end_col = positions[index]
            if None not in (lineno, end_lineno, col, end_col):
                nodes, parents = _parsed(frame.f_code.co_filename)
                node = nodes.get((lineno, end_lineno, col, end_col))  # type: ignore[arg-type]
                verdict = node is not None and _in_truth_context(node, parents)
        _TRUTH[key] = verdict
    return verdict


def _owner_of(model: object) -> str:
    """The id of the result ``model`` belongs to: its own for a result (a copy of one, which no hydration saw, too),
    else the one its hydration recorded, or ``"?"`` for a nested model copied after hydration."""
    if type(model).__recorded_path__ == "":  # type: ignore[attr-defined]
        return str(object.__getattribute__(model, "__dict__")["id"])
    return _OWNERS.get(id(model), "?")


def _note(owner: object, name: str, *, prose: bool = False) -> None:
    """Record that ``name`` was read on ``owner``, a recorded object, if a block is recording."""
    access = _SESSION.access
    if access is None:
        return
    frame = _caller()
    if frame is None:
        return
    site = _site(frame)
    read = _joined(type(owner).__recorded_path__, name)  # type: ignore[attr-defined]
    if prose:
        if _truth_only(frame):
            read = f"bool({read})"
        reads = access.prose.setdefault(read, {}).setdefault(site, ProseReads())
        reads.events += 1
        reads.results.add(_owner_of(owner))
    access.sites.setdefault(read, set()).add(site)


class RecordingModel:
    """The reads every recording class records, mixed in before the class it records (see the module docstring).

    A recording class carries the path its instances sit at (``__recorded_path__``) and the fields of the class it
    records (``__recorded_fields__``); it adds no slot, so a hydrated model's class can be swapped for it in place.
    """

    __slots__ = ()
    __recorded_path__: str
    __recorded_fields__: frozenset[str]
    __recorded_base__: type[BaseModel]

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if name == "__dict__":
            _note(self, "__dict__")
        elif not name.startswith("_"):
            if name in type(self).__recorded_fields__:
                _note(self, name, prose=name in PROSE_FIELDS)
            elif name != "model_fields_set" and callable(value):
                _note(self, f"{name}()")
            else:
                _note(self, name)
        return value

    def __eq__(self, other: object) -> bool:
        _note(self, "==")
        # Compared as the two models' own classes compare them: pydantic's equality, and the core shapes' own, both
        # require one class on both sides, which a recorded model (one class per path) never shares with an
        # unrecorded one or one recorded elsewhere.
        equal: bool = _unrecorded(self) == _unrecorded(other)
        return equal

    def __hash__(self) -> int:
        _note(self, "hash()")
        return super().__hash__()

    def __repr__(self) -> str:
        _note(self, "repr()")
        return super().__repr__()

    def __repr_name__(self) -> str:
        """The recorded class's name, so a recorded model prints as the model it is."""
        return type(self).__recorded_base__.__name__

    def __str__(self) -> str:
        _note(self, "str()")
        return super().__str__()

    def __iter__(self) -> Any:
        _note(self, "iter()")
        return super().__iter__()  # type: ignore[misc]

    def __copy__(self) -> Any:
        _note(self, "copy()")
        return super().__copy__()  # type: ignore[misc]

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Any:
        _note(self, "deepcopy()")
        return super().__deepcopy__(memo)  # type: ignore[misc]


#: The instance state a pydantic model keeps, which a stand-in of the recorded class shares.
_MODEL_STATE = ("__dict__", "__pydantic_fields_set__", "__pydantic_extra__", "__pydantic_private__")


def _unrecorded(value: object) -> object:
    """``value`` itself, or for a recorded model a stand-in of the class it records sharing its state, unrecorded.

    Only its own class changes: a nested model it holds stays recorded, and compares through this same rule.
    """
    if not isinstance(value, RecordingModel):
        return value
    stand_in = object.__new__(type(value).__recorded_base__)
    for name in _MODEL_STATE:
        object.__setattr__(stand_in, name, object.__getattribute__(value, name))
    return stand_in


def _recording_class(base: type[BaseModel], path: str) -> type[BaseModel]:
    """The recording subclass of ``base`` whose instances sit at ``path``."""
    held = _CLASSES.get((base, path))
    if held is None:
        namespace: dict[str, Any] = {
            "__module__": __name__,
            "__qualname__": f"Recorded{base.__name__}",
            "__recorded_path__": path,
            "__recorded_fields__": frozenset(base.model_fields),
            "__recorded_base__": base,
        }
        if base.__hash__ is None:
            namespace["__hash__"] = None
        held = _CLASSES[(base, path)] = type(base)(f"Recorded{base.__name__}", (RecordingModel, base), namespace)
    return held


def _record(model: BaseModel, path: str, owner: str) -> None:
    """Swap ``model``'s class, and each nested model's, for its recording subclass at ``path``, in place."""
    base = type(model)
    if isinstance(model, RecordingModel):
        return
    for name in base.model_fields:
        value = model.__dict__.get(name)
        _record_value(value, _joined(path, name), owner)
    object.__setattr__(model, "__class__", _recording_class(base, path))
    if PROSE_FIELDS & set(base.model_fields):
        _OWNERS[id(model)] = owner
        weakref.finalize(model, _OWNERS.pop, id(model), None)


def _record_value(value: object, path: str, owner: str) -> None:
    if isinstance(value, BaseModel):
        _record(value, path, owner)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, BaseModel):
                _record(item, f"{path}[]", owner)
    elif isinstance(value, dict):
        for item in value.values():
            if isinstance(item, BaseModel):
                _record(item, f"{path}{{}}", owner)


@contextmanager
def recording_results() -> Iterator[None]:
    """Serve every ``EvalResult`` a store hydrates inside the block as a recorded result, and every reading built of
    one as that recorded result itself (see the module docstring).

    The stored read itself is unchanged: the real ``from_dict`` runs first, and its result's class is swapped after.
    """
    stored_read: Callable[[dict[str, Any]], EvalResult] = EvalResult.from_dict

    def recorded_read(data: dict[str, Any]) -> EvalResult:
        result = stored_read(data)
        _record(result, "", result.id)
        return result

    def served_whole(result: EvalResult, shared: SharedValues | None = None) -> EvalResult:
        return result

    with (
        mock.patch.object(EvalResult, "from_dict", recorded_read),
        mock.patch.object(ResultReading, "of", staticmethod(served_whole)),
    ):
        yield


@contextmanager
def recording() -> Iterator[ResultAccess]:
    """Collect the reads of recorded results made inside the block. Blocks do not nest."""
    if _SESSION.access is not None:
        raise RuntimeError("a recording block is already running; they do not nest")
    access = ResultAccess()
    _SESSION.access = access
    try:
        yield access
    finally:
        _SESSION.access = None
