"""Recorded tool turns: a model's replies, kept per session in order, replayed with the tools run live.

A loop that lets a model call tools is replayed differently from a completion. The tools run live on the fixture's
data each replay, so what the loop sends changes whenever a tool's text does, and a recording matched on what was
sent would refuse every replay after such a change (the brittleness that made matched recordings costly to keep).
:class:`RecordedTurns` therefore keeps **the replies alone**: per named session (the analyst's, each editor
round's), in the order they came, and never a byte of what was sent. A replay hands back a session's next reply
whatever it is asked, so the loop runs as recorded for as long as the code asks what it asked when it was recorded.

**The snapshot id is a placeholder.** A loop's replies name the snapshot they read (a tool call's arguments, a
submission), and a fixture's snapshot id moves whenever the fixture is frozen again. Each string of a recorded reply
holds the session's snapshot id as ``{snapshot}``, and a replay writes the snapshot it is given back in its place. A
reply already holding the literal ``{snapshot}`` is refused when it is recorded, so a replay never writes an id where
the model wrote the placeholder's text.

**Drift is a refusal that names its turn.** A session asked one turn more than it holds raises
:class:`~threetears.evals.testing.RecordingRefused` naming the session and the turn, and :meth:`RecordedTurns.unconsumed`
names the sessions a replay left turns of. Either means the code now runs differently from when the recording was
made. The remedy is to the code, or to the recorded reply as data: a turn recording is never recorded again, since a
second paid run would answer differently and hide what changed.

**The file.** One JSON object per line, appended as each turn returns and before its reply is handed back: the
recorder's format version (:attr:`RecordedTurns.FORMAT_VERSION`), when it was written, the session's name, the turn's
index in that session (from 0), and the reply (:meth:`~threetears.evals.schema.TurnReply.to_json`). Recording onto a
file that exists appends to it, each session's turns numbered after the ones it holds; a fresh recording removes the
file first, and only when asked (``record(..., replace=True)``). A line of another format, one whose index is not its
session's next, or one this recorder does not write refuses the whole file by its line number. Costs replay as
recorded, the historical fact of what each turn cost when it was made.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Self

from threetears.evals.schema.turns import TurnToolSpec, ToolTurnClient, TurnMessage, TurnReply
from threetears.evals.testing.recorded_completions import RecordingRefused

__all__ = ["RecordedTurns"]

#: What a recorded reply holds in place of the session's snapshot id.
_SNAPSHOT_PLACEHOLDER = "{snapshot}"

#: The top-level members every recorded line carries, and nothing else at this format version.
_LINE_MEMBERS = frozenset({"recorder_version", "recorded_at", "session", "index", "reply"})

#: What a drifted replay's refusal tells its reader to do.
_DRIFT_REMEDY = (
    "the code now runs differently from when the recording was made: fix the code, or edit the recorded reply as "
    "data (a turn recording is never recorded again)"
)

#: Serialises every append in this process, so two sessions recording onto one file never interleave a line.
_APPEND_LOCK = threading.Lock()


class RecordedTurns:
    """A tool-turn recording: the replies of each named session, replayed in order, or recorded from a live client.

    :meth:`replay` reads a file; :meth:`record` wraps a live :class:`~threetears.evals.schema.ToolTurnClient` and
    appends each reply it returns. Either way :meth:`turn_session` hands out one session's client, which a loop
    sends that session's turns through: a replaying one answers the session's next recorded reply and sends nothing
    (its price ceiling is ``0.0``); a recording one sends to the live client and keeps its reply.
    ``threetears.evals.providers.Model`` builds one from ``turn_recording=``.

    :cvar FORMAT_VERSION: The recording format this build writes, and the only one it reads.
    """

    FORMAT_VERSION: ClassVar[int] = 1

    def __init__(
        self, path: Path, *, inner: ToolTurnClient | None, held: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> None:
        """Hold a read recording; :meth:`replay` and :meth:`record` build one.

        :param path: The recording.
        :ptype path: Path
        :param inner: The live client a recording sends through, or ``None`` to replay.
        :ptype inner: ToolTurnClient | None
        :param held: Session -> its recorded replies' JSON, in order, the snapshot id still a placeholder.
        :ptype held: Mapping[str, Sequence[Mapping[str, Any]]]
        """
        self.path = path
        self._book = _Book(path, inner, {name: [dict(reply) for reply in replies] for name, replies in held.items()})

    @classmethod
    def replay(cls, path: str | os.PathLike[str]) -> Self:
        """Read a recording to replay, offline.

        :param path: The recording.
        :ptype path: str | os.PathLike[str]
        :return: The replay.
        :rtype: RecordedTurns
        :raises RecordingRefused: The file is missing, or holds a line this recorder does not write, named by number.
        """
        recording = Path(path)
        if not recording.is_file():
            raise RecordingRefused(
                recording, "there is no such turn recording", "record it once against the live model"
            )
        return cls(recording, inner=None, held=_read_recording(recording))

    @classmethod
    def record(cls, client: ToolTurnClient, path: str | os.PathLike[str], *, replace: bool = False) -> Self:
        """Record the turns sent through ``client`` onto ``path``.

        :param client: The live client each turn is sent through.
        :ptype client: ToolTurnClient
        :param path: The recording, appended to when it exists.
        :ptype path: str | os.PathLike[str]
        :param replace: Remove the file first, so the recording starts afresh.
        :ptype replace: bool
        :return: The recorder.
        :rtype: RecordedTurns
        :raises RecordingRefused: The file to append to holds a line this recorder does not write.
        """
        recording = Path(path)
        if replace:
            recording.unlink(missing_ok=True)
        held = _read_recording(recording) if recording.is_file() else {}
        return cls(recording, inner=client, held=held)

    def turn_session(self, name: str, *, snapshot: str) -> ToolTurnClient:
        """The client one session's turns go through; asked again for the same name, it carries on where it was.

        :param name: The session's name (``analyst``, ``editor-1``), which its lines are kept under.
        :ptype name: str
        :param snapshot: The snapshot the session reads, held in each reply as ``{snapshot}``.
        :ptype snapshot: str
        :return: The session's client.
        :rtype: ToolTurnClient
        :raises ValueError: An empty name or snapshot id.
        """
        if not name:
            raise ValueError("a turn session needs a name, which its recorded turns are kept under")
        if not snapshot:
            raise ValueError("a turn session needs the snapshot id its replies hold as {snapshot}")
        return _Session(self._book, name, snapshot)

    def unconsumed(self) -> dict[str, int]:
        """The sessions a replay has not served every recorded turn of, and how many each has left.

        :return: Session -> turns left; empty when every recorded turn was served. Always empty while recording.
        :rtype: dict[str, int]
        """
        book = self._book
        if book.inner is not None:
            return {}
        left = {name: len(replies) - book.served.get(name, 0) for name, replies in book.held.items()}
        return {name: count for name, count in sorted(left.items()) if count}


class _Book:
    """What a recording holds and has served, shared by every session it hands out.

    :ivar path: The recording.
    :ivar inner: The live client a recording sends through, or ``None`` while replaying.
    :ivar held: Session -> its replies' JSON, in order, the snapshot id a placeholder.
    :ivar served: Session -> how many of its turns have been served (replaying) or recorded (recording).
    """

    def __init__(self, path: Path, inner: ToolTurnClient | None, held: dict[str, list[dict[str, Any]]]) -> None:
        self.path = path
        self.inner = inner
        self.held = held
        self.served: dict[str, int] = {}

    async def turn(
        self,
        name: str,
        snapshot: str,
        *,
        system: str,
        messages: Sequence[TurnMessage],
        tools: Sequence[TurnToolSpec],
        max_tokens: int,
    ) -> TurnReply:
        """The session's next reply: the recorded one, or the live client's, recorded before it is handed back."""
        if self.inner is None:
            replies = self.held.get(name, [])
            index = self.served.get(name, 0)
            if index >= len(replies):
                holds = f"holds {len(replies)} recorded turns" if replies else "holds no recorded turns"
                raise RecordingRefused(
                    self.path, f"session {name!r} {holds}, and its turn {index + 1} was asked", _DRIFT_REMEDY
                )
            self.served[name] = index + 1
            return TurnReply.from_json(_restored(replies[index], snapshot))
        reply = await self.inner.turn(system=system, messages=messages, tools=tools, max_tokens=max_tokens)
        kept = _held(reply.to_json(), snapshot, self.path)
        with _APPEND_LOCK:
            replies = self.held.setdefault(name, [])
            line = {
                "recorder_version": RecordedTurns.FORMAT_VERSION,
                "recorded_at": datetime.now(UTC).isoformat(),
                "session": name,
                "index": len(replies),
                "reply": kept,
            }
            written = json.dumps(line, ensure_ascii=False, allow_nan=False)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(written + "\n")
            replies.append(kept)
            self.served[name] = len(replies)
        return reply

    def ceiling(
        self, *, system: str, messages: Sequence[TurnMessage], tools: Sequence[TurnToolSpec], max_tokens: int
    ) -> float | None:
        """The live client's ceiling while recording; ``0.0`` while replaying, which sends nothing."""
        if self.inner is None:
            return 0.0
        return self.inner.turn_price_ceiling(system=system, messages=messages, tools=tools, max_tokens=max_tokens)


class _Session:
    """One named session of a recording: a :class:`~threetears.evals.schema.ToolTurnClient`."""

    def __init__(self, book: _Book, name: str, snapshot: str) -> None:
        self._book = book
        self._name = name
        self._snapshot = snapshot

    async def turn(
        self, *, system: str, messages: Sequence[TurnMessage], tools: Sequence[TurnToolSpec], max_tokens: int
    ) -> TurnReply:
        return await self._book.turn(
            self._name, self._snapshot, system=system, messages=messages, tools=tools, max_tokens=max_tokens
        )

    def turn_price_ceiling(
        self, *, system: str, messages: Sequence[TurnMessage], tools: Sequence[TurnToolSpec], max_tokens: int
    ) -> float | None:
        return self._book.ceiling(system=system, messages=messages, tools=tools, max_tokens=max_tokens)


def _held(value: Any, snapshot: str, path: Path) -> Any:  # Any: a JSON value
    """``value`` with every occurrence of ``snapshot`` in its strings written as the placeholder."""
    if isinstance(value, str):
        if _SNAPSHOT_PLACEHOLDER in value:
            raise RecordingRefused(
                path,
                f"a reply holds the literal text {_SNAPSHOT_PLACEHOLDER}, which a replay would read as the snapshot id",
                "record the session with a prompt that does not lead the model to write it",
            )
        return value.replace(snapshot, _SNAPSHOT_PLACEHOLDER)
    if isinstance(value, Mapping):
        return {key: _held(member, snapshot, path) for key, member in value.items()}
    if isinstance(value, list):
        return [_held(member, snapshot, path) for member in value]
    return value


def _restored(value: Any, snapshot: str) -> Any:  # Any: a JSON value
    """``value`` with the placeholder in its strings written back as ``snapshot``."""
    if isinstance(value, str):
        return value.replace(_SNAPSHOT_PLACEHOLDER, snapshot)
    if isinstance(value, Mapping):
        return {key: _restored(member, snapshot) for key, member in value.items()}
    if isinstance(value, list):
        return [_restored(member, snapshot) for member in value]
    return value


def _read_recording(path: Path) -> dict[str, list[dict[str, Any]]]:  # Any: JSON objects
    """Session -> its recorded replies' JSON, in order; any line this recorder does not write refuses the file."""
    held: dict[str, list[dict[str, Any]]] = {}
    remedy = "restore the file from version control; a turn recording is edited as data, never recorded again"
    with path.open(encoding="utf-8") as file:
        for number, text in enumerate(file, start=1):
            if not text.strip():
                continue
            try:
                line = json.loads(text)
            except json.JSONDecodeError as unread:
                raise RecordingRefused(path, f"line {number} is not JSON ({unread.msg})", remedy) from unread
            if not isinstance(line, dict) or set(line) != _LINE_MEMBERS:
                raise RecordingRefused(path, f"line {number} is not a line this recorder writes", remedy)
            version = line["recorder_version"]
            if version != RecordedTurns.FORMAT_VERSION:
                raise RecordingRefused(
                    path,
                    f"line {number} is recorder format {version!r}, and this build reads format "
                    f"{RecordedTurns.FORMAT_VERSION}",
                    remedy,
                )
            name, index, reply = line["session"], line["index"], line["reply"]
            if not isinstance(name, str) or not name or not isinstance(reply, dict):
                raise RecordingRefused(path, f"line {number} names no session or holds no reply", remedy)
            replies = held.setdefault(name, [])
            if index != len(replies):
                raise RecordingRefused(
                    path,
                    f"line {number} is turn index {index!r} of session {name!r}, whose next is {len(replies)}",
                    remedy,
                )
            try:
                TurnReply.from_json(reply)
            except ValueError as unread:
                raise RecordingRefused(path, f"line {number}'s reply cannot be read: {unread}", remedy) from unread
            replies.append(reply)
    return held
