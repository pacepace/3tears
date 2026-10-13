"""Recorded completions: a completion client that records a live run once and replays it exactly, offline.

A suite that exercises code calling a model has three choices for the model: call it every time (paid, slow, and
never the same answer twice), stand a scripted double in for it (free, but what is scored is the double's text, not
a model's), or call it once and keep what it said. :class:`RecordedCompletions` is the third.
:meth:`RecordedCompletions.record` wraps the host's real client and appends every completion it returns to a JSONL
file; :meth:`RecordedCompletions.replay` reads that file and hands the same completions back, in the order they were
made, with no network. The examples run this way: live when the provider's key is set, from their committed
recording when it is not, and ``RECORD=1`` re-records one.

**What a call is matched on.** Each recorded line is keyed by the sha256 of the canonical JSON of the call's
``system``, ``user`` and ``response_format`` and of the ``model`` and ``temperature`` the completion reported, and
carries the call's index under that key: the first identical call is index 0, the next 1, so ``k`` repeats of one
prompt replay as the ``k`` different answers they were. A replaying client knows the prompt it is asked but not the
model behind it, so it finds the key from the models and temperatures the recording holds; one file can hold several
clients' calls (two models answering one prompt), and a client replaying one of them names it
(``replay(path, model=...)``).

**A miss is refused, never served live.** A prompt the recording does not hold, or one asked more times than it was
recorded, raises :class:`RecordingRefused` naming the prompt (its first 80 characters), the call's index, and the
command that re-records it. It is an :class:`~threetears.evals.kernel.host.ApparatusError`: a replay that cannot
serve is a fault of the rig, not of the code under test, so a candidate that raises it has its cell excluded rather
than scored. :meth:`RecordedCompletions.check_covers` asks the same question of a list of prompts before any replay
starts, which is how a test says by name that a prompt changed under its recording.

**What is kept.** All ten :class:`~threetears.evals.schema.CompletionResult` members as the inner client returned
them, and every other attribute its completion type declares (a dataclass field, a Pydantic field, an instance
attribute: a ``served_provider``, say) under ``extra``. A computed property is not a declared attribute and is not
kept. A value JSON cannot hold is refused when it is recorded, not rendered as text. A replayed completion is a
:class:`ReplayedCompletion`; ``cost_usd`` and ``price_source`` are replayed as recorded, so a replayed price is the
historical fact of what that call cost when it was made, not a quote at today's rates.

**The file.** One JSON object per line, appended as each call returns and before its completion is handed back, in
the order calls returned. Recording onto a file that exists appends to it, the new calls numbered after the ones it
holds; a fresh recording removes the file first, and only when asked (``record(..., replace=True)``), so no recording
is ever overwritten by accident. Each line carries the recorder's format version (:attr:`RecordedCompletions.FORMAT_VERSION`)
and the time it was written; a line from a newer format, or one that is not a line this recorder writes, refuses the
whole file by its line number. One process records a file at a time.
"""

from __future__ import annotations

import json
import threading
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal, NamedTuple, Self, get_args

from threetears.evals.kernel.host.apparatus import ApparatusError
from threetears.evals.schema.completion import (
    COMPLETION_RESULT_ATTRIBUTES,
    CompletionClient,
    CompletionResult,
    StopReason,
)
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.testing.completion_conformance import (
    CompletionConformanceFailure,
    check_completion_conformance,
)

__all__ = ["RecordedCompletions", "RecordedPrompt", "RecordingRefused", "ReplayedCompletion"]

#: The recording format this build writes and the newest it reads. A line's ``recorder_version`` above it is refused.
_FORMAT_VERSION = 1

#: How much of a prompt's ``user`` a refusal quotes, so the prompt is recognisable without the message carrying all of it.
_PROMPT_HEAD = 80

#: The top-level members every recorded line carries, and nothing else at this format version.
_LINE_MEMBERS = frozenset({"recorder_version", "recorded_at", "key", "index", "user_head", "result", "extra"})

#: Serialises every append in this process, so two recording clients on one file never interleave a line or give two
#: calls one index.
_APPEND_LOCK = threading.Lock()


class RecordingRefused(ApparatusError):
    """A recording cannot serve what was asked of it: a miss, a missing or unreadable file, a newer format.

    An :class:`~threetears.evals.kernel.host.ApparatusError`, because a replay that cannot serve a call is the rig's
    fault rather than the code under test's: raised out of a candidate, its cell is excluded, not scored. The message
    names what was asked and ends with :attr:`remedy`.

    Attributes:
        path: The recording.
        remedy: What makes a recording that serves it: usually the command that re-records it.
    """

    def __init__(self, path: Path, problem: str, remedy: str) -> None:
        """Name the recording, what it cannot do, and the remedy.

        Args:
            path: The recording.
            problem: What it cannot serve, or why it cannot be read, as a clause.
            remedy: What makes one that serves it.
        """
        self.path = path
        self.remedy = remedy
        super().__init__(f"recording {path}: {problem}; {remedy}")


class RecordedPrompt(NamedTuple):
    """One call a recording is expected to serve, as :meth:`RecordedCompletions.check_covers` is handed it.

    The three arguments of :meth:`~threetears.evals.schema.CompletionClient.generate`, so a test lists the prompts its
    code sends in the form it sends them.

    Attributes:
        system: The system prompt.
        user: The user message.
        response_format: The provider directive, or ``None``.
    """

    system: str
    user: str
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayedCompletion:
    """A completion as it was recorded, replayed: the ten :class:`~threetears.evals.schema.CompletionResult` members.

    Satisfies :class:`~threetears.evals.schema.CompletionResult` structurally. Every member holds what the inner
    client's completion held when it was recorded. ``cost_usd`` and ``price_source`` are a historical fact, what the
    call cost when it was made and who priced it then, not a quote: a replay calls no price source, and a rate that
    moved since is not reflected.

    Attributes:
        content: The completion's text.
        input_tokens: Prompt tokens the provider counted, or ``None``.
        output_tokens: Completion tokens the provider counted, reasoning included, or ``None``.
        cost_usd: What the call cost when it was recorded, or ``None`` when nothing priced it.
        price_source: Who priced it then, or ``None``.
        model: The model the call was attributed to.
        served_model: The model the provider's response named, or ``None``.
        reasoning_tokens: The reasoning share of ``output_tokens``, or ``None``.
        temperature: The temperature the request was sent with, or ``None``.
        stop_reason: Why the completion stopped, in the engine's vocabulary.
        extra: Every other attribute the recorded completion's type declared, by name (a ``served_provider``, say),
            read-only.
    """

    content: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    price_source: str | None
    model: str
    served_model: str | None
    reasoning_tokens: int | None
    temperature: float | None
    stop_reason: StopReason
    extra: Mapping[str, Any]


class RecordedCompletions:
    """A completion client that records the calls an inner client answers, or replays a recording of them.

    Build one with :meth:`record` (every call goes to the inner client and is appended to the recording) or
    :meth:`replay` (every call is served from the recording, with no network). Either satisfies
    :class:`~threetears.evals.schema.CompletionClient`, and ``generate`` is safe to call concurrently on one instance.
    See the module docstring for what a call is matched on and what is kept.

    One replaying client per recorded client: two clients replaying the same model's calls from one file each serve
    the first answer first, so ``k`` repeats split between them replay out of order.

    Attributes:
        FORMAT_VERSION: The recording format this build writes, and the newest it reads.
        path: The recording.
        mode: ``"record"`` or ``"replay"``, so an example can say which it did.
    """

    FORMAT_VERSION: ClassVar[int] = _FORMAT_VERSION

    def __init__(
        self,
        path: Path,
        *,
        inner: CompletionClient | None,
        recorded: Mapping[str, Sequence[ReplayedCompletion]],
        model: str | None = None,
        temperature: float | None | Literal["any"] = "any",
        rerecord: str | None = None,
    ) -> None:
        """Hold a recording's calls, recording onto it through ``inner`` or replaying it without one.

        :meth:`record` and :meth:`replay` build one: they read and check the file, which this does not.

        Args:
            path: The recording.
            inner: The client every call goes to while recording; ``None`` replays.
            recorded: The calls the recording already holds, by key, in index order.
            model: While replaying, the model whose calls this client serves; ``None`` serves any.
            temperature: While replaying, the temperature whose calls this client serves; ``"any"`` serves any.
            rerecord: The command that re-records ``path``, quoted in every refusal; ``None`` names the convention.
        """
        self.path = path
        self.mode: Literal["record", "replay"] = "replay" if inner is None else "record"
        self._inner = inner
        self._recorded = {key: list(calls) for key, calls in recorded.items()}
        self._model = model
        self._temperature = temperature
        self._remedy = _rerecord_remedy(path, rerecord)
        self._served: Counter[str] = Counter()
        self._lock = threading.Lock()
        # The file's size when this client last read or wrote it: a different size on disk means another client in
        # this process appended to it, or it was removed, and the indexes are read again before the next append.
        self._size = _file_size(path)

    @classmethod
    def record(cls, inner: CompletionClient, path: Path, *, replace: bool = False) -> Self:
        """A client that sends every call to ``inner`` and appends what it returns to ``path``.

        The completion goes back to the caller unchanged, after its line is written. A recording that exists is
        appended to, its calls numbered after the ones it holds, and read first: one this build cannot read is refused
        before any call is made. ``inner`` is this client's to close: :meth:`aclose` closes it.

        Args:
            inner: The host's real client.
            path: The recording. Its directory is created when missing.
            replace: Remove ``path`` first and record afresh. Pass it when building the client, before any call;
                without it, an existing recording is never overwritten.

        Returns:
            The recording client.

        Raises:
            RecordingRefused: ``path`` holds a line this build cannot read, or one from a newer format.
        """
        if replace:
            path.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        recorded = _read_recording(path, _rerecord_remedy(path, None)) if path.exists() else {}
        return cls(path, inner=inner, recorded=recorded)

    @classmethod
    def replay(
        cls,
        path: Path,
        *,
        model: str | None = None,
        temperature: float | None | Literal["any"] = "any",
        rerecord: str | None = None,
    ) -> Self:
        """A client that serves every call from the recording at ``path``, by prompt and call index, with no network.

        Args:
            path: The recording.
            model: The model whose recorded calls this client serves, when the recording holds a prompt answered by
                more than one; ``None`` serves whichever the recording holds, and refuses a prompt it holds under two.
            temperature: The same, for a prompt recorded at more than one temperature (``None`` is a call sent with
                none); ``"any"`` serves whichever it holds.
            rerecord: The command that re-records ``path`` (``"RECORD=1 python llm_judge.py"``), quoted in every
                refusal; ``None`` names the ``RECORD=1`` convention.

        Returns:
            The replaying client.

        Raises:
            RecordingRefused: ``path`` does not exist, holds no calls, holds a line this build cannot read, or holds
                one from a newer format.
        """
        recorded = _readable_replay(path, _rerecord_remedy(path, rerecord))
        return cls(path, inner=None, recorded=recorded, model=model, temperature=temperature, rerecord=rerecord)

    @classmethod
    def check_covers(
        cls,
        path: Path,
        prompts: Iterable[RecordedPrompt],
        *,
        model: str | None = None,
        temperature: float | None | Literal["any"] = "any",
        rerecord: str | None = None,
    ) -> None:
        """Refuse, naming every prompt, when the recording at ``path`` no longer covers ``prompts``.

        What a test calls before a replay starts, so a prompt that changed under its recording is named as such
        rather than surfacing as a miss partway through. A prompt listed ``n`` times needs ``n`` recorded calls.
        ``model`` and ``temperature`` select the client's calls as :meth:`replay` does.

        Args:
            path: The recording.
            prompts: Every call the code under test will make.
            model: As :meth:`replay`'s.
            temperature: As :meth:`replay`'s.
            rerecord: As :meth:`replay`'s.

        Raises:
            RecordingRefused: The recording cannot be read (as :meth:`replay`), or holds fewer calls than asked for
                one or more prompts, each named.
        """
        remedy = _rerecord_remedy(path, rerecord)
        recorded = _readable_replay(path, remedy)
        uncovered: list[str] = []
        for prompt, asked in Counter(prompts).items():
            keys = _keys_for(recorded, prompt, model, temperature)
            if len(keys) > 1:
                uncovered.append(f"{_head(prompt.user)} is recorded under {_pairs(keys.values())}")
                continue
            held = sum(len(recorded[key]) for key in keys)
            if held < asked:
                uncovered.append(f"{_head(prompt.user)} is asked {asked} time(s) and recorded {held}")
        if uncovered:
            raise RecordingRefused(
                path, f"the recording no longer covers {len(uncovered)} prompt(s): " + "; ".join(uncovered), remedy
            )

    async def generate(
        self,
        *,
        system: str,
        user: str,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Record or replay one call.

        Args:
            system: The system prompt.
            user: The user message.
            response_format: The provider directive, or ``None``.

        Returns:
            Recording, the inner client's completion, unchanged; replaying, the recorded one as a
            :class:`ReplayedCompletion`.

        Raises:
            RecordingRefused: Replaying, the recording holds no call for this prompt, holds it under more than one
                model or temperature this client could serve, or holds fewer calls for it than this one's index.
            CompletionConformanceFailure: Recording, the inner client's completion lacks a member, or holds a value
                of the wrong type or one JSON cannot hold.
        """
        if self._inner is None:
            return self._serve(system, user, response_format)
        completion = await self._inner.generate(system=system, user=user, response_format=response_format)
        self._append(system, user, response_format, completion)
        return completion

    def _serve(self, system: str, user: str, response_format: dict[str, Any] | None) -> ReplayedCompletion:
        keys = _keys_for(self._recorded, RecordedPrompt(system, user, response_format), self._model, self._temperature)
        if not keys:
            raise RecordingRefused(
                self.path,
                f"no call is recorded for the prompt {_head(user)}{self._selection()} (call 0): the prompt changed since "
                "it was recorded, or the code now asks something it did not ask then",
                self._remedy,
            )
        if len(keys) > 1:
            raise RecordingRefused(
                self.path,
                f"the prompt {_head(user)} is recorded under {_pairs(keys.values())}, and this client serves any of them",
                "name the one it replays: RecordedCompletions.replay(path, model=..., temperature=...)",
            )
        ((key, (model, temperature)),) = keys.items()
        with self._lock:
            index = self._served[key]
            calls = self._recorded[key]
            if index >= len(calls):
                raise RecordingRefused(
                    self.path,
                    f"the prompt {_head(user)} is recorded {len(calls)} time(s) under model {model!r} at temperature "
                    f"{temperature!r}, and this is call {index} (counting from 0): the code asks it more often than it "
                    "did when it was recorded",
                    self._remedy,
                )
            self._served[key] = index + 1
        return calls[index]

    def _selection(self) -> str:
        parts = ([] if self._model is None else [f"model {self._model!r}"]) + (
            [] if self._temperature == "any" else [f"temperature {self._temperature!r}"]
        )
        return f" under {' at '.join(parts)}" if parts else ""

    def _append(
        self, system: str, user: str, response_format: dict[str, Any] | None, completion: CompletionResult
    ) -> None:
        check_completion_conformance(completion)
        members = _recordable_members(completion)
        extra = _declared_extras(completion)
        key = _call_key(system, user, response_format, members["model"], members["temperature"])
        with _APPEND_LOCK, self._lock:
            if _file_size(self.path) != self._size:
                self._recorded = _read_recording(self.path, self._remedy) if self.path.exists() else {}
            calls = self._recorded.setdefault(key, [])
            line = {
                "recorder_version": _FORMAT_VERSION,
                "recorded_at": datetime.now(UTC).isoformat(),
                "key": key,
                "index": len(calls),
                "user_head": user[:_PROMPT_HEAD],
                "result": members,
                "extra": extra,
            }
            with self.path.open("a", encoding="utf-8") as recording:
                recording.write(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n")
            calls.append(_completion_of(members, extra))
            self._size = _file_size(self.path)

    async def aclose(self) -> None:
        """Recording, close the inner client, which this client owns; replaying, nothing: it owns no transport."""
        if self._inner is not None:
            await self._inner.aclose()

    async def __aenter__(self) -> Self:
        """Enter a scope that closes on exit."""
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close on the way out, whether or not the body raised."""
        await self.aclose()


def _rerecord_remedy(path: Path, rerecord: str | None) -> str:
    command = (
        f"`{rerecord}`"
        if rerecord
        else "`RECORD=1 …` set on the command that made it (an example re-records with `RECORD=1 python <example>.py`)"
    )
    return f"re-record {path} with {command}, the provider's key set"


def _head(user: str) -> str:
    return repr(user[:_PROMPT_HEAD] + ("…" if len(user) > _PROMPT_HEAD else ""))


def _pairs(pairs: Iterable[tuple[str, float | None]]) -> str:
    return ", ".join(f"model {model!r} at temperature {temperature!r}" for model, temperature in sorted(pairs, key=str))


def _readable_replay(path: Path, remedy: str) -> dict[str, list[ReplayedCompletion]]:
    """The calls a recording holds, refused when there is none to replay: no file, or a file with no calls."""
    if not path.exists():
        raise RecordingRefused(path, "there is no recording at this path", remedy)
    recorded = _read_recording(path, remedy)
    if not recorded:
        raise RecordingRefused(path, "the recording holds no calls, so it can serve none", remedy)
    return recorded


def _keys_for(
    recorded: Mapping[str, Sequence[ReplayedCompletion]],
    prompt: RecordedPrompt,
    model: str | None,
    temperature: float | None | Literal["any"],
) -> dict[str, tuple[str, float | None]]:
    """The recorded keys ``prompt`` could be served from under ``model`` and ``temperature``, each with its pair.

    A replaying client knows the prompt and not the model behind it, so every (model, temperature) the recording
    holds is tried, narrowed to the ones the client was told it replays.
    """
    pairs = {(call.model, call.temperature) for calls in recorded.values() for call in calls}
    candidates = {
        _call_key(prompt.system, prompt.user, prompt.response_format, held_model, held_temperature): (
            held_model,
            held_temperature,
        )
        for held_model, held_temperature in pairs
        if (model is None or held_model == model) and (temperature == "any" or held_temperature == temperature)
    }
    return {key: pair for key, pair in candidates.items() if key in recorded}


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _call_key(
    system: str, user: str, response_format: dict[str, Any] | None, model: str, temperature: float | None
) -> str:
    """The sha256 a call is recorded and replayed under. Raises ``UnhashableContentError`` for a directive JSON cannot hold."""
    return canonical_digest(
        {
            "system": system,
            "user": user,
            "response_format": response_format,
            "model": model,
            "temperature": temperature,
        }
    )


def _member_problem(name: str, value: Any) -> str | None:
    """What is wrong with one recorded member's value, or ``None`` when it is the protocol's type."""
    number = isinstance(value, int | float) and not isinstance(value, bool)
    count = isinstance(value, int) and not isinstance(value, bool)
    match name:
        case "content" | "model":
            ok = isinstance(value, str)
        case "input_tokens" | "output_tokens" | "reasoning_tokens":
            ok = value is None or count
        case "cost_usd" | "temperature":
            ok = value is None or number
        case "price_source" | "served_model":
            ok = value is None or isinstance(value, str)
        case "stop_reason":
            ok = value in get_args(StopReason)
        case _:  # pragma: no cover — every protocol member is matched above
            ok = False
    return None if ok else f"`{name}` holds {value!r} ({type(value).__name__})"


def _normalised(name: str, value: Any) -> Any:
    """``cost_usd`` and ``temperature`` as floats, so ``0`` and ``0.0`` key and replay alike."""
    return float(value) if name in ("cost_usd", "temperature") and value is not None else value


def _recordable_members(completion: CompletionResult) -> dict[str, Any]:
    members: dict[str, Any] = {}
    for name in COMPLETION_RESULT_ATTRIBUTES:
        value = getattr(completion, name)
        if (problem := _member_problem(name, value)) is not None:
            raise CompletionConformanceFailure(
                f"{type(completion).__name__}: {problem}, not what the CompletionResult protocol declares, so the "
                "recording would not replay it as the call returned it"
            )
        members[name] = _normalised(name, value)
    return members


def _declared_extras(completion: object) -> dict[str, Any]:
    """Every attribute ``completion``'s type declares beyond the protocol's ten, which JSON must be able to hold."""
    if is_dataclass(completion) and not isinstance(completion, type):
        names = [f.name for f in fields(completion)]
    elif isinstance(declared := getattr(type(completion), "model_fields", None), Mapping):
        names = list(declared)  # a Pydantic model's fields, read off the class without importing Pydantic
    else:
        slots = [name for klass in type(completion).__mro__ for name in _slot_names(klass)]
        names = [*getattr(completion, "__dict__", {}), *(name for name in slots if hasattr(completion, name))]
    extras: dict[str, Any] = {}
    for name in dict.fromkeys(names):
        if name.startswith("_") or name in COMPLETION_RESULT_ATTRIBUTES:
            continue
        value = getattr(completion, name)
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as unencodable:
            raise CompletionConformanceFailure(
                f"{type(completion).__name__}: `{name}` holds a value of type {type(value).__name__}, which JSON cannot "
                f"hold ({unencodable}), "
                "so the recording cannot keep it; make the attribute JSON-safe, or keep it off the completion type"
            ) from unencodable
        extras[name] = value
    return extras


def _slot_names(klass: type) -> tuple[str, ...]:
    """The names ``klass`` itself declares in ``__slots__``, which may be one string or an iterable of them."""
    declared = getattr(klass, "__slots__", ())
    return (declared,) if isinstance(declared, str) else tuple(declared)


def _completion_of(members: Mapping[str, Any], extra: Mapping[str, Any]) -> ReplayedCompletion:
    return ReplayedCompletion(**members, extra=MappingProxyType(dict(extra)))


def _read_recording(path: Path, remedy: str) -> dict[str, list[ReplayedCompletion]]:
    """Every call ``path`` holds, by key in index order, each line checked; the first bad line refuses the file."""
    recorded: defaultdict[str, list[ReplayedCompletion]] = defaultdict(list)
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # the newline that ends the last line
    for number, text in enumerate(lines, start=1):
        key, index, completion = _read_line(path, number, text, remedy)
        if index != len(recorded[key]):
            raise _malformed(
                path,
                number,
                f"its call index is {index}, and the lines before it hold {len(recorded[key])} call(s) under its key: "
                "the file was edited, or two processes recorded onto it at once",
                remedy,
            )
        recorded[key].append(completion)
    return dict(recorded)


def _mismatch(expected: frozenset[str], found: Iterable[str]) -> str | None:
    """Which expected members an object lacks and which unknown ones it adds, as a clause, or ``None``."""
    found = set(found)
    clauses = [
        f"{verb} {', '.join(sorted(names))}"
        for verb, names in (("lacks", expected - found), ("adds", found - expected))
        if names
    ]
    return " and ".join(clauses) or None


def _malformed(path: Path, number: int, problem: str, remedy: str) -> RecordingRefused:
    return RecordingRefused(path, f"line {number} is not a recorded call: {problem}", remedy)


def _read_line(path: Path, number: int, text: str, remedy: str) -> tuple[str, int, ReplayedCompletion]:
    try:
        line = json.loads(text)
    except json.JSONDecodeError as undecodable:
        raise _malformed(
            path, number, f"it is not JSON ({undecodable.msg}, column {undecodable.colno})", remedy
        ) from None
    if not isinstance(line, dict):
        raise _malformed(path, number, f"it holds a JSON {type(line).__name__}, not an object", remedy)
    version = line.get("recorder_version")
    if not isinstance(version, int) or isinstance(version, bool):
        raise _malformed(path, number, f"its recorder_version is {version!r}, not a whole number", remedy)
    if version > _FORMAT_VERSION:
        raise RecordingRefused(
            path,
            f"line {number} was written in recording format {version}, newer than this build's {_FORMAT_VERSION}",
            "upgrade 3tears-evals to the build that recorded it, or " + remedy,
        )
    if mismatch := _mismatch(_LINE_MEMBERS, line.keys()):
        raise _malformed(path, number, f"it {mismatch}", remedy)
    key, index, recorded_at, result, extra = (line[name] for name in ("key", "index", "recorded_at", "result", "extra"))
    if not isinstance(key, str) or len(key) != 64:
        raise _malformed(path, number, f"its key {key!r} is not a sha256 hex digest", remedy)
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise _malformed(path, number, f"its index {index!r} is not a call index", remedy)
    if not isinstance(recorded_at, str):
        raise _malformed(path, number, f"its recorded_at {recorded_at!r} is not a time", remedy)
    try:
        datetime.fromisoformat(recorded_at)
    except ValueError:
        raise _malformed(path, number, f"its recorded_at {recorded_at!r} is not an ISO 8601 time", remedy) from None
    if not isinstance(line["user_head"], str):
        raise _malformed(path, number, "its user_head is not text", remedy)
    if not isinstance(extra, dict):
        raise _malformed(path, number, f"its extra is a JSON {type(extra).__name__}, not an object", remedy)
    if not isinstance(result, dict):
        raise _malformed(path, number, f"its result is a JSON {type(result).__name__}, not an object", remedy)
    if mismatch := _mismatch(frozenset(COMPLETION_RESULT_ATTRIBUTES), result.keys()):
        raise _malformed(path, number, f"its result {mismatch}", remedy)
    if problems := [
        problem for name in COMPLETION_RESULT_ATTRIBUTES if (problem := _member_problem(name, result[name]))
    ]:
        raise _malformed(path, number, "its result's " + "; ".join(problems), remedy)
    members = {name: _normalised(name, result[name]) for name in COMPLETION_RESULT_ATTRIBUTES}
    return key, index, _completion_of(members, extra)
