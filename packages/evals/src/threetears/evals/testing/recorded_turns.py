"""Recorded tool turns: a model's replies, kept per session in order, replayed with the tools run live.

A loop that lets a model call tools is replayed differently from a completion. The tools run live on the fixture's
data each replay, so what the loop sends changes whenever a tool's text does, and a recording matched on what was
sent would refuse every replay after such a change (the brittleness that made matched recordings costly to keep).
:class:`RecordedTurns` therefore keeps **the replies alone**: per named session (the analyst's, each editor
round's), in the order they came, and never a byte of what was sent. A replay hands back a session's next reply
whatever it is asked, so the loop runs as recorded for as long as the code asks what it asked when it was recorded.

**Price ceilings replay too.** A loop admits each turn on its price ceiling, and a budget that refused a turn is an
ending a replay must reach as the recording did. So each ceiling a recording session answered is kept against the
turn it bounded (the session's next), the refused turn's included, and a replay answers that turn's recorded ceiling
rather than a price of its own; a session that asks a ceiling for a turn it recorded none for is refused, naming the
turn. A replayed reply's ``cost_usd`` is the recorded one, the historical fact of what the turn cost when it was made.

**The snapshot id is a placeholder.** A loop's replies name the snapshot they read (a tool call's arguments, a
submission), and a fixture's snapshot id moves whenever the fixture is frozen again. Each string of a recorded reply
holds the session's snapshot id as ``{snapshot}``, and a replay writes the snapshot it is given back in its place. A
reply already holding the literal ``{snapshot}`` is refused when it is recorded, so a replay never writes an id where
the model wrote the placeholder's text.

**Drift is a refusal that names its turn, and nothing is recorded twice.** A session asked one turn more than it
holds raises :class:`~threetears.evals.testing.RecordingRefused` naming the session and the turn, and
:meth:`RecordedTurns.unconsumed` names the sessions a replay left replies or ceilings of. Either means the code now
runs differently from when the recording was made, and the remedy is to the code, or to the recorded reply as data: a
turn recording that exists is only ever replayed (:meth:`RecordedTurns.record` refuses one), since a second paid run
would answer differently and hide what changed. The one exception is a recording run that was cut short (interrupted,
or a turn that failed), which leaves a file a replay runs off the end of: that file is deleted and recorded again.

**The file.** One JSON object per line, appended as each event happens: the recorder's format version
(:attr:`RecordedTurns.FORMAT_VERSION`), when it was written, the session's name, the index (from 0) of the turn the
line belongs to, and its ``kind``: a ``ceiling`` line holds the ceiling answered for that turn (``null`` when the
client could not bound it; a later one for the same turn replaces it), and a ``reply`` line holds the turn's reply
(:meth:`~threetears.evals.schema.TurnReply.to_json`), written before the reply is handed back. A line of another
format, one for any turn but its session's next, or one this recorder does not write refuses the whole file by its
line number.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Self

from threetears.evals.schema.turns import ToolTurnClient, TurnMessage, TurnReply, TurnToolSpec
from threetears.evals.testing.recorded_completions import RecordingRefused

__all__ = ["RecordedTurns"]

#: What a recorded reply holds in place of the session's snapshot id.
_SNAPSHOT_PLACEHOLDER = "{snapshot}"

#: The top-level members each kind of line carries, and nothing else at this format version.
_LINE_MEMBERS: Mapping[str, frozenset[str]] = {
    "reply": frozenset({"recorder_version", "recorded_at", "session", "index", "kind", "reply"}),
    "ceiling": frozenset({"recorder_version", "recorded_at", "session", "index", "kind", "ceiling"}),
}

#: What a drifted replay's refusal tells its reader to do: a recording run that was cut short (interrupted, or a turn
#: that failed) is the one case a recording is made again; any other drift is the code's.
_DRIFT_REMEDY = (
    "if the recording run was cut short (interrupted, or a turn failed), delete the file and record it again; "
    "otherwise the code now runs differently from when the recording was made: fix the code, or edit the recorded "
    "reply as data"
)

#: What a refusal to read a recording tells its reader to do.
_UNREADABLE_REMEDY = "restore the file from version control; a turn recording is edited as data, never recorded again"

#: Serialises every append in this process, so two sessions recording onto one file never interleave a line.
_APPEND_LOCK = threading.Lock()


@dataclass(slots=True)
class _SessionLines:
    """One session's lines as read: its replies' JSON in order, and the ceiling recorded for each turn that has one."""

    replies: list[dict[str, Any]] = field(default_factory=list)  # Any: JSON objects
    ceilings: dict[int, float | None] = field(default_factory=dict)


class RecordedTurns:
    """A tool-turn recording: the replies of each named session, replayed in order, or recorded from a live client.

    :meth:`replay` reads a file; :meth:`record` wraps a live :class:`~threetears.evals.schema.ToolTurnClient` and
    keeps each reply it returns and each price ceiling it answers. Either way :meth:`turn_session` hands out one
    session's client, which a loop sends that session's turns through: a replaying one answers the session's next
    recorded reply and the ceiling recorded for that turn, and sends nothing; a recording one sends to the live client
    and keeps what it says. ``threetears.evals.providers.Model`` builds one from ``turn_recording=``.

    :cvar FORMAT_VERSION: The recording format this build writes, and the only one it reads.
    """

    FORMAT_VERSION: ClassVar[int] = 1

    def __init__(self, path: str | os.PathLike[str], *, inner: ToolTurnClient | None) -> None:
        """Open a recording: read it to replay (``inner`` is ``None``), or start one, recorded through ``inner``.

        :param path: The recording.
        :ptype path: str | os.PathLike[str]
        :param inner: The live client a recording sends through, or ``None`` to replay.
        :ptype inner: ToolTurnClient | None
        :raises RecordingRefused: Replaying, the file is missing or holds a line this recorder does not write, named
            by number; recording, the file exists: a turn recording is replayed, never recorded again.
        """
        self.path = Path(path)
        if inner is None:
            if not self.path.is_file():
                raise RecordingRefused(
                    self.path, "there is no such turn recording", "record it once against the live model"
                )
            sessions = _read_recording(self.path)
        else:
            if self.path.exists():
                raise RecordingRefused(
                    self.path,
                    "it exists, and a turn recording is replayed, never recorded again",
                    "replay it (RecordedTurns.replay); delete it first only if its recording run was cut short",
                )
            sessions = {}
        self._book = _Book(self.path, inner, sessions)

    @classmethod
    def replay(cls, path: str | os.PathLike[str]) -> Self:
        """Read a recording to replay, offline.

        :param path: The recording.
        :ptype path: str | os.PathLike[str]
        :return: The replay.
        :rtype: RecordedTurns
        :raises RecordingRefused: The file is missing, or holds a line this recorder does not write, named by number.
        """
        return cls(path, inner=None)

    @classmethod
    def record(cls, client: ToolTurnClient, path: str | os.PathLike[str]) -> Self:
        """Record the turns sent through ``client`` onto ``path``, a file that does not exist yet.

        :param client: The live client each turn is sent through.
        :ptype client: ToolTurnClient
        :param path: The recording to make.
        :ptype path: str | os.PathLike[str]
        :return: The recorder.
        :rtype: RecordedTurns
        :raises RecordingRefused: The file exists: a turn recording is replayed, never recorded again.
        """
        return cls(path, inner=client)

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
        """The sessions a replay left recorded lines of, and how many: replies never served, and ceilings never asked.

        :return: Session -> lines left; empty when every reply was served and every ceiling asked. Always empty while
            recording.
        :rtype: dict[str, int]
        """
        book = self._book
        if book.inner is not None:
            return {}
        left = {
            name: len(lines.replies) - book.served.get(name, 0) + len(set(lines.ceilings) - book.asked.get(name, set()))
            for name, lines in book.sessions.items()
        }
        return {name: count for name, count in sorted(left.items()) if count}


class _Book:
    """What a recording holds and has served, shared by every session it hands out.

    :ivar path: The recording.
    :ivar inner: The live client a recording sends through, or ``None`` while replaying.
    :ivar sessions: Session -> its lines: replies (the snapshot id a placeholder) and recorded ceilings.
    :ivar served: Session -> how many of its turns have been served (replaying) or recorded (recording).
    :ivar asked: Session -> the turns a replay was asked the recorded ceiling of.
    """

    def __init__(self, path: Path, inner: ToolTurnClient | None, sessions: dict[str, _SessionLines]) -> None:
        self.path = path
        self.inner = inner
        self.sessions = sessions
        self.served: dict[str, int] = {}
        self.asked: dict[str, set[int]] = {}

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
        index = self.served.get(name, 0)
        if self.inner is None:
            replies = self.sessions[name].replies if name in self.sessions else []
            if index >= len(replies):
                holds = f"holds {len(replies)} recorded turns" if replies else "holds no recorded turns"
                raise RecordingRefused(
                    self.path, f"session {name!r} {holds}, and its turn {index + 1} was asked", _DRIFT_REMEDY
                )
            self.served[name] = index + 1
            return TurnReply.from_json(_restored(replies[index], snapshot))
        reply = await self.inner.turn(system=system, messages=messages, tools=tools, max_tokens=max_tokens)
        kept = _held(reply.to_json(), snapshot, self.path)
        self._append(name, index, "reply", kept)
        self.sessions.setdefault(name, _SessionLines()).replies.append(kept)
        self.served[name] = index + 1
        return reply

    def ceiling(
        self,
        name: str,
        *,
        system: str,
        messages: Sequence[TurnMessage],
        tools: Sequence[TurnToolSpec],
        max_tokens: int,
    ) -> float | None:
        """The session's next turn's ceiling: the one recorded for it, or the live client's, recorded."""
        index = self.served.get(name, 0)
        if self.inner is None:
            ceilings = self.sessions[name].ceilings if name in self.sessions else {}
            if index not in ceilings:
                raise RecordingRefused(
                    self.path,
                    f"session {name!r} recorded no price ceiling for its turn {index + 1}, and one was asked",
                    _DRIFT_REMEDY,
                )
            self.asked.setdefault(name, set()).add(index)
            return ceilings[index]
        ceiling = self.inner.turn_price_ceiling(system=system, messages=messages, tools=tools, max_tokens=max_tokens)
        self._append(name, index, "ceiling", ceiling)
        self.sessions.setdefault(name, _SessionLines()).ceilings[index] = ceiling
        return ceiling

    def _append(self, name: str, index: int, kind: str, value: object) -> None:
        line = {
            "recorder_version": RecordedTurns.FORMAT_VERSION,
            "recorded_at": datetime.now(UTC).isoformat(),
            "session": name,
            "index": index,
            "kind": kind,
            kind: value,
        }
        written = json.dumps(line, ensure_ascii=False, allow_nan=False)
        with _APPEND_LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(written + "\n")


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
        return self._book.ceiling(self._name, system=system, messages=messages, tools=tools, max_tokens=max_tokens)


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


def _read_recording(path: Path) -> dict[str, _SessionLines]:
    """Session -> its lines as read; any line this recorder does not write refuses the whole file by its number."""
    sessions: dict[str, _SessionLines] = {}
    with path.open(encoding="utf-8") as file:
        for number, text in enumerate(file, start=1):
            if not text.strip():
                continue
            try:
                line = json.loads(text)
            except json.JSONDecodeError as unread:
                raise _unreadable(path, f"line {number} is not JSON ({unread.msg})") from unread
            kind = line.get("kind") if isinstance(line, dict) else None
            if not isinstance(kind, str) or kind not in _LINE_MEMBERS or set(line) != _LINE_MEMBERS[kind]:
                raise _unreadable(path, f"line {number} is not a line this recorder writes")
            version = line["recorder_version"]
            if version != RecordedTurns.FORMAT_VERSION:
                raise _unreadable(
                    path,
                    f"line {number} is recorder format {version!r}, and this build reads format "
                    f"{RecordedTurns.FORMAT_VERSION}",
                )
            name, index = line["session"], line["index"]
            if not isinstance(name, str) or not name:
                raise _unreadable(path, f"line {number} names no session")
            lines = sessions.setdefault(name, _SessionLines())
            if index != len(lines.replies):
                raise _unreadable(
                    path,
                    f"line {number} is for turn index {index!r} of session {name!r}, whose next is "
                    f"{len(lines.replies)}",
                )
            if kind == "ceiling":
                lines.ceilings[index] = _read_ceiling(path, number, line["ceiling"])
                continue
            reply = line["reply"]
            if not isinstance(reply, dict):
                raise _unreadable(path, f"line {number} holds no reply")
            try:
                TurnReply.from_json(reply)
            except ValueError as unread:
                raise _unreadable(path, f"line {number}'s reply cannot be read: {unread}") from unread
            lines.replies.append(reply)
    return sessions


def _read_ceiling(path: Path, number: int, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise _unreadable(path, f"line {number}'s ceiling {value!r} is neither a finite number nor null")
    return float(value)


def _unreadable(path: Path, problem: str) -> RecordingRefused:
    return RecordingRefused(path, problem, _UNREADABLE_REMEDY)
