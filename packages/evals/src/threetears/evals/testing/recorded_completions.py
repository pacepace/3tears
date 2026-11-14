"""Recorded completions: a completion client that records a live run once and replays it exactly, offline.

A suite that exercises code calling a model has three choices for the model: call it every time (paid, slow, and
never the same answer twice), stand a scripted double in for it (free, but what is scored is the double's text, not
a model's), or call it once and keep what it said. :class:`RecordedCompletions` is the third.
:meth:`RecordedCompletions.record` wraps the host's real client and appends every completion it returns to a JSONL
file; :meth:`RecordedCompletions.replay` reads that file and hands the same completions back, with no network. The package's examples run on it: live with a key, recorded with ``RECORD=1``, and otherwise
replayed from the ``recordings/`` beside each rung (``examples/README.md``).

**What a call is matched on.** A recording holds one model's calls at one temperature. Each line is keyed by the
sha256 of the canonical JSON of the call's ``system``, ``user`` and ``response_format``, the model and temperature, and
the request settings the client was built with (a :class:`~threetears.evals.schema.ClientRequestSettings`, or ``None``
for a client built with none), and carries the call's index under that key, so ``k`` repeats of one prompt replay as
the ``k`` answers they were.

**Which recorded call a cell is served.** A run's cells start in an order shuffled by the run's fresh id, several at
once, so serving a prompt's calls in the order they are asked would hand a recorded answer to a different cell on each
replay. A call made inside an eval's cell is recorded with the cell it was made for (format 3): the arm's subject and
contestant identity, the case and the repeat (:class:`~threetears.evals.schema.CellIdentity`'s ``subject_id``,
``variant_key``, ``test_case_id`` and ``k_iteration``), and its ask number: how many times that cell had asked the
prompt before, **counted when the call is asked**, never when its answer lands, so two asks of one prompt by one cell
at once are numbered in the order they were asked, recording and replaying alike. A replay serves a cell the calls
recorded for that very cell, its ``n``-th ask its ``n``-th by ask number, so two arms that send a judge the same
prompt, or a follow-up asked by several repeats, get back what each got when it was recorded, on every replay, in any
order; a cell asking a prompt more often than it was recorded is refused, and is never served another cell's call.
Where no recorded call names the asking cell, repeat ``i`` is served the call of index ``i`` among those that name no
cell (a format-2 line, or a call made outside every cell) when it is left, and otherwise the first of those left,
which is deterministic only while each repeat asks such a prompt once. A cell the recording names nowhere (its arm,
case or eval renamed since) and left no such call stands for the one recorded cell line (a subject, arm and case)
that asked the prompt at its repeat as often and shares the most of its subject, arm and case: it is bound to that line
for the rest of the replay and served its calls, so a renamed arm replays as it was recorded, and so does a case whose
id a fresh store minted anew (each arm's cell stands for its own arm's line, which shares its subject and arm), on
every replay. Where none fits it is refused, saying so; where several fit as well, which is its own cannot be told, and it is refused rather than
handed one of them, in whatever order the cells ask; so is a call outside every cell when only calls made in cells are left. Where the one line that
fits has asked as itself, or another cell already stands for it, it is refused too. So no cell is ever served a call
recorded for a cell that has asked as itself, and only cells whose best fit is one same recorded line turn on order:
the recorded cell itself and a cell named nowhere that fits it best, or two cells named nowhere that fit it best (an
arm added beside a recorded one whose every prompt it shares, which cannot be told from a rename; two cases of one
arm where the recording held one), where whichever asks first takes the line and every later one is refused,
naming it. The
settings never reach ``generate``, so :meth:`RecordedCompletions.record` and
:meth:`RecordedCompletions.replay` are told them. They are matched because they change the answer, so one file can
hold one model under two settings (two efforts), each client replaying its own, and a replay asked under settings a
prompt was not recorded with is refused naming each setting that differs. A second model, or a second temperature,
goes in a file of its own: recording one onto a file is refused.

**It stands in for any client the engine is handed.** A role factory's client is a
:class:`~threetears.evals.schema.BoundCompletionClient`, which names its model (``model_name``) and prices a call
before it is made (``price_ceiling``): an analysis writer and every out-of-run call need both. Recording, both are
the inner client's, read through, so the recorder is a bound client exactly when its inner one is (a plain
:class:`~threetears.evals.schema.CompletionClient` has neither, and reading one off its recorder raises
``AttributeError`` naming it). Replaying, ``model_name`` is the model the recording holds, and ``price_ceiling`` is
``0.0``: a replay sends nothing and costs nothing.

**A miss is refused, never served live.** A prompt the recording does not hold, one it holds only under other request
settings, or one asked more times than it was recorded, raises :class:`RecordingRefused` naming the prompt (its first
80 characters) or the settings that differ, and the command that re-records it. A prompt held under no key names its
own key's first 12 hex digits and where it parts from the recorded prompts: how many open on the same 80 characters (a
line keeps no more, so what changed lies past them), else the nearest recorded key and the character it parts at.
Each call is served once per client:
one whose every recorded call already went to another eval's run in the process (a ``Model`` shared by two evals, or
an eval run twice on one) is refused for that cause, naming the eval, and the remedy is a client per eval, never a
recording made again. It is an
:class:`~threetears.evals.kernel.host.ReplayMiss`: a replay that cannot serve is a fault of the rig, not of the code
under test, and its recordings are stale. So a candidate that meets one never has the cell scored: the cell is recorded
``apparatus_failed`` naming the call, and the run fails with the miss. A judge call that meets one is never read as the
judge's failed answer either: :func:`~threetears.evals.run.run_judge_llm` raises it, so a writer's pairwise pass fails
at the miss, and a run's scoring judge, or a judge kind's judge under test on any one trial, fails the run naming the
call. A page whose text moved under a judge's recording is named, never judged on the calls still found.
:meth:`RecordedCompletions.check_covers` asks the same question of a list of prompts before any replay starts, which
is how a test says by name that a prompt changed under its recording.

**What is kept.** All ten :class:`~threetears.evals.schema.CompletionResult` members as the inner client returned
them; the optional ``sent_settings`` member, what the request actually carried, as
:func:`~threetears.evals.schema.sent_settings_of` reads it (``None`` when the completion has none), so a replayed
completion reports what the recorded call was sent with, not what a replay is asked under; and every other
attribute its completion type declares (a dataclass field, a Pydantic field, an instance attribute: a
``served_provider``, say) under ``extra``, and a kept ``served_provider`` is replayed as the completion's own, so a
replayed call's usage row names the upstream the recorded one did. A computed property is not a declared attribute and
is not kept. A value JSON cannot hold (a non-finite ``cost_usd`` or ``temperature`` included) is refused when it is recorded,
as a :class:`~threetears.evals.testing.CompletionConformanceFailure`, not rendered as text. A replayed completion is a
:class:`ReplayedCompletion`; ``cost_usd`` and ``price_source`` are replayed as recorded, so a replayed price is the
historical fact of what that call cost when it was made, not a quote at today's rates.

**The file.** One JSON object per line, appended as each call returns and before its completion is handed back, in
the order calls returned. Recording onto a file that exists appends to it, the new calls numbered after the ones it
holds; a fresh recording removes the file first, and only when asked (``record(..., replace=True)``), so no recording
is ever overwritten by accident. Each line carries the recorder's format version (:attr:`RecordedCompletions.FORMAT_VERSION`)
and the time it was written. A format-2 line, which noted no cell, is read and served by the fallback above, so a
recording made before format 3 replays as it did, and one recorded onto since holds both. A line from a newer format,
one from format 1 (it kept no ``sent_settings``, so its calls cannot say what they were sent with), or one that is not
a line this recorder writes, refuses the whole file by its line number. One process records a file at a time.
"""

from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal, NamedTuple, Self, get_args

from threetears.evals.kernel.capped_completion import refresh_prices
from threetears.evals.kernel.eval_scope import RunningEval, running_eval
from threetears.evals.kernel.host.apparatus import ReplayMiss
from threetears.evals.schema.completion import (
    COMPLETION_RESULT_ATTRIBUTES,
    OPTIONAL_COMPLETION_ATTRIBUTES,
    CompletionClient,
    CompletionResult,
    StopReason,
    sent_settings_of,
)
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import ClientRequestSettings
from threetears.evals.schema.traces import CellIdentity, current_cell_identity
from threetears.evals.testing.completion_conformance import (
    CompletionConformanceFailure,
    check_completion_conformance,
)

__all__ = ["RecordedCompletions", "RecordedPrompt", "RecordingRefused", "ReplayedCompletion"]

#: The recording format this build writes, and the only one it reads: a line's ``recorder_version`` above it is refused
#: as newer, and one below it as older.
#:
#: - **1** — the first format.
#: - **2** — each line carries ``sent_settings``, what the recorded call's request actually carried. A format-1 line
#:   kept none, so it cannot replay a completion that says what it was sent with, and is refused by name.
#: - **3** — each line carries ``cell``, the cell the call was made for (``null`` outside every cell), so a replay serves
#:   each cell its own call. A format-2 line noted none and is still read, served by the fallback.
_FORMAT_VERSION = 3

#: The oldest format this build reads: a line below it is refused by name.
_OLDEST_FORMAT = 2

#: How much of a prompt's ``user`` a refusal quotes, so the prompt is recognisable without the message carrying all of it.
_PROMPT_HEAD = 80

#: How many hex digits of a call's key a refusal names it by.
_KEY_DIGITS = 12

#: The top-level members every format-2 line carries, and nothing else.
_FORMAT_2_MEMBERS = frozenset(
    {"recorder_version", "recorded_at", "key", "index", "user_head", "settings", "sent_settings", "result", "extra"}
)

#: The top-level members a line carries, and nothing else, by the format it was written in.
_LINE_MEMBERS = {2: _FORMAT_2_MEMBERS, 3: _FORMAT_2_MEMBERS | {"cell"}}

#: The members a format-3 line's ``cell`` object carries, and nothing else.
_CELL_MEMBERS = frozenset({"subject", "variant", "case", "repeat", "ask"})

#: Serialises every append in this process, so two recording clients on one file never interleave a line or give two
#: calls one index.
_APPEND_LOCK = threading.Lock()


class RecordingRefused(ReplayMiss):
    """A recording cannot serve what was asked of it: a miss, a missing or unreadable file, a newer format.

    A :class:`~threetears.evals.kernel.host.ReplayMiss`, because a replay that cannot serve a call is the rig's fault
    rather than the code under test's, and its recordings are stale: wherever it is met, a candidate's call or a judge's,
    its cell is recorded ``apparatus_failed`` naming the call, never scored, and the run fails with it. The message
    names what was asked and ends with :attr:`remedy`.

    :ivar path: The recording.
    :ivar remedy: What makes a recording that serves it: usually the command that re-records it.
    """

    def __init__(self, path: Path, problem: str, remedy: str) -> None:
        """Name the recording, what it cannot do, and the remedy.

        :param path: The recording.
        :ptype path: Path
        :param problem: What it cannot serve, or why it cannot be read, as a clause.
        :ptype problem: str
        :param remedy: What makes one that serves it.
        :ptype remedy: str
        """
        self.path = path
        self.remedy = remedy
        super().__init__(f"recording {path}: {problem}; {remedy}")


class _CellAsk(NamedTuple):
    """Which cell made a call, and which of that cell's asks of the prompt it was: what a format-3 line records.

    :ivar subject: The run's subject (the cell identity's ``subject_id``).
    :ivar variant: The arm's contestant identity (its ``variant_key``), or ``None`` where the cell's identity named none.
    :ivar case: The case (``test_case_id``).
    :ivar repeat: The repeat (``k_iteration``), or ``None`` where the cell's identity named none.
    :ivar ask: How many times this cell had asked the prompt before, counting from 0.
    """

    subject: str
    variant: str | None
    case: str
    repeat: int | None
    ask: int


#: A cell as a recording notes it: :class:`_CellAsk` without the ask.
_CellCoordinates = tuple[str, str | None, str, int | None]

#: A cell less its repeat — its subject, arm and case: every repeat of one case by one arm.
_CellLine = tuple[str, str | None, str]


def _shared(recorded: _CellLine, asking: _CellLine) -> int:
    """How many of a cell line's subject, arm and case a recorded line shares: what a renamed cell is bound by."""
    return sum(1 for held, asked in zip(recorded, asking, strict=True) if held == asked)


def _coordinates(cell: CellIdentity) -> _CellCoordinates:
    """The coordinates of ``cell`` a recording notes: the ones that name the same cell on every run of the eval."""
    return (cell.subject_id, cell.variant_key, cell.test_case_id, cell.k_iteration)


class RecordedPrompt(NamedTuple):
    """One call a recording is expected to serve, as :meth:`RecordedCompletions.check_covers` is handed it.

    The three arguments of :meth:`~threetears.evals.schema.CompletionClient.generate`, so a test lists the prompts its
    code sends in the form it sends them.

    :ivar system: The system prompt.
    :ivar user: The user message.
    :ivar response_format: The provider directive, or ``None``.
    """

    system: str
    user: str
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class ReplayedCompletion:
    """A completion as it was recorded, replayed: the ten :class:`~threetears.evals.schema.CompletionResult` members.

    Satisfies :class:`~threetears.evals.schema.CompletionResult` structurally, and carries its optional
    ``sent_settings`` member. Every member holds what the inner client's completion held when it was recorded.
    ``cost_usd`` and ``price_source`` are a historical fact, what the call cost when it was made and who priced it then,
    not a quote: a replay calls no price source, and a rate that moved since is not reflected.

    :ivar content: The completion's text.
    :ivar input_tokens: Prompt tokens the provider counted, or ``None``.
    :ivar output_tokens: Completion tokens the provider counted, reasoning included, or ``None``.
    :ivar cost_usd: What the call cost when it was recorded, or ``None`` when nothing priced it.
    :ivar price_source: Who priced it then, or ``None``.
    :ivar model: The model the call was attributed to.
    :ivar served_model: The model the provider's response named, or ``None``.
    :ivar reasoning_tokens: The reasoning share of ``output_tokens``, or ``None``.
    :ivar temperature: The temperature the request was sent with, or ``None``.
    :ivar stop_reason: Why the completion stopped, in the engine's vocabulary.
    :ivar sent_settings: What the recorded call's request actually carried, as its completion reported it, or ``None``
        when it reported nothing: never the settings the replay was asked under.
    :ivar extra: Every other attribute the recorded completion's type declared, by name (a ``served_provider``, say),
        read-only. A ``served_provider`` there is read back as :attr:`served_provider` too.
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
    sent_settings: ClientRequestSettings | None
    extra: Mapping[str, Any]

    @property
    def served_provider(self) -> str | None:
        """The upstream the recorded call's reply named as serving it, as ``extra`` keeps it, or ``None`` when it named
        none: so a replayed call's usage row names who served it, as the recorded call's did.

        :return: The provider, or ``None`` when ``extra`` holds no non-empty text under ``served_provider``.
        :rtype: str | None
        """
        said = self.extra.get("served_provider")
        return said if isinstance(said, str) and said else None


#: What a recording holds under one key: the request settings it was recorded under, and its calls in index order.
_Recorded = tuple[ClientRequestSettings | None, list[ReplayedCompletion]]

#: The one model a recording holds calls of, and the one temperature they were sent at.
_Client = tuple[str, float | None]


class _Read(NamedTuple):
    """What a recording holds, as :func:`_read_recording` reads it.

    :ivar recorded: Its calls by key, each with the request settings it was recorded under, in index order.
    :ivar cells: The cell each call names, by key in index order (``None`` for a call that names none).
    :ivar heads: The head of each key's user message, as its lines keep it: what a miss is compared with.
    """

    recorded: dict[str, _Recorded]
    cells: dict[str, list[_CellAsk | None]]
    heads: dict[str, str]


class RecordedCompletions:
    """A completion client that records the calls an inner client answers, or replays a recording of them.

    Build one with :meth:`record` (every call goes to the inner client and is appended to the recording) or
    :meth:`replay` (every call is served from the recording, with no network). Either satisfies
    :class:`~threetears.evals.schema.CompletionClient`, and ``generate`` is safe to call concurrently on one instance.
    A replaying client, and a recording one whose inner client is bound, also satisfies
    :class:`~threetears.evals.schema.BoundCompletionClient` (:attr:`model_name`, :meth:`price_ceiling`), so either
    stands in for an analysis or out-of-run client.
    See the module docstring for what a call is matched on and what is kept.

    One replaying client per recorded client: two clients replaying the same calls from one file each serve every
    call, so a prompt asked through both is served twice. One replaying client per eval, too: it
    serves each recorded call once, so a second eval asking the same prompts is refused, naming the eval that took them. Two clients that differ only in their
    request settings are two recorded clients, and each replays its own calls.

    :ivar FORMAT_VERSION: The recording format this build writes, and the newest it reads.
    :ivar path: The recording.
    :ivar mode: ``"record"`` or ``"replay"``, so an example can say which it did.
    """

    FORMAT_VERSION: ClassVar[int] = _FORMAT_VERSION

    def __init__(
        self,
        path: Path,
        *,
        inner: CompletionClient | None,
        settings: ClientRequestSettings | None,
        recorded: Mapping[str, tuple[ClientRequestSettings | None, Sequence[ReplayedCompletion]]],
        rerecord: str | None = None,
    ) -> None:
        """Hold a recording's calls, recording onto it through ``inner`` or replaying it without one.

        :meth:`record` and :meth:`replay` build one: they read and check the file, which this does not.

        :param path: The recording.
        :ptype path: Path
        :param inner: The client every call goes to while recording; ``None`` replays.
        :ptype inner: CompletionClient | None
        :param settings: The request settings the client is built with (recording ``inner``, or replaying the client
            that was recorded), or ``None`` for one built with none: recorded on every line, and matched on replay.
        :ptype settings: ClientRequestSettings | None
        :param recorded: The calls the recording already holds, by key: the request settings the key was recorded under,
            and its calls in index order.
        :ptype recorded: Mapping[str, tuple[ClientRequestSettings | None, Sequence[ReplayedCompletion]]]
        :param rerecord: The command that re-records ``path``, quoted in every refusal; ``None`` says to re-record it
            with the client it stands in for.
        :ptype rerecord: str | None
        """
        self.path = path
        self.mode: Literal["record", "replay"] = "replay" if inner is None else "record"
        self._inner = inner
        self._settings = settings
        self._recorded = {key: (held, list(calls)) for key, (held, calls) in recorded.items()}
        #: The cell each recorded call names, by key in index order (``None`` for a call that names none). Built empty
        #: here; :meth:`record` and :meth:`replay` set it from the file they read.
        self._cells: dict[str, list[_CellAsk | None]] = {
            key: [None] * len(calls) for key, (_, calls) in recorded.items()
        }
        #: Replaying, the head of each key's user message, as the file keeps it: what a miss names its nearest by. Set by
        #: :meth:`replay`.
        self._heads: dict[str, str] = {}
        #: Replaying, how many times each cell has asked each key, so a cell's second ask of a prompt is served its
        #: second call.
        self._asks: dict[tuple[str, _CellCoordinates], int] = {}
        #: Recording, the ask number each cell's next ask of each prompt (by its digest) takes, given as it is asked.
        self._next_ask: dict[tuple[str, _CellCoordinates], int] = {}
        #: Replaying, each cell line (a cell less its repeat) that matches none the recording names, bound to the
        #: recorded line it stands for: the one line its first unmatched ask could only have been (:meth:`_index_of`).
        self._bound: dict[_CellLine, _CellLine] = {}
        #: Replaying, every recorded line a cell has asked as itself: never bound to a cell that matches none.
        self._live: set[_CellLine] = set()
        self._remedy = _rerecord_remedy(path, rerecord)
        #: Each key's served calls, by index, with the eval run each went to (None outside every one).
        self._taken: dict[str, dict[int, RunningEval | None]] = {}
        self._lock = threading.Lock()
        # The file's size when this client last read or wrote it: a different size on disk means another client in
        # this process appended to it, or it was removed, and the indexes are read again before the next append.
        self._size = _file_size(path)

    @classmethod
    def record(
        cls,
        inner: CompletionClient,
        path: Path,
        *,
        settings: ClientRequestSettings | None,
        replace: bool = False,
    ) -> Self:
        """A client that sends every call to ``inner`` and appends what it returns to ``path``.

        The completion goes back to the caller unchanged, after its line is written. A recording that exists is
        appended to, its calls numbered after the ones it holds, and read first: one this build cannot read is refused
        before any call is made. ``inner`` is this client's to close: :meth:`aclose` closes it.

        ``settings`` is required and has no default, as :func:`~threetears.evals.ops.start_paid_call`'s ``budget_s``
        has none: the settings ``inner`` was built with are not on the client and never reach ``generate``, so the
        recorder cannot read them, and a default would record every client as built with none, which is the one
        mistake a replay under changed settings could then not catch.

        :param inner: The host's real client.
        :ptype inner: CompletionClient
        :param path: The recording. Its directory is created when missing.
        :ptype path: Path
        :param settings: The request settings ``inner`` was built with, the same value its builder applied (for a judge,
            :data:`~threetears.evals.run.JUDGE_REQUEST_SETTINGS`), or ``None`` when it was built with none.
            Every line carries them and every key covers them.
        :ptype settings: ClientRequestSettings | None
        :param replace: Remove ``path`` first and record afresh. Pass it when building the client, before any call;
            without it, an existing recording is never overwritten.
        :ptype replace: bool
        :return: The recording client.
        :rtype: Self
        :raises RecordingRefused: ``path`` holds a line this build cannot read, or one from a newer format.
        """
        if replace:
            path.unlink(missing_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        recorded, cells, _ = _read_recording(path, _rerecord_remedy(path, None)) if path.exists() else ({}, {}, {})
        client = cls(path, inner=inner, settings=settings, recorded=recorded)
        client._cells = cells
        return client

    @classmethod
    def replay(cls, path: Path, *, settings: ClientRequestSettings | None, rerecord: str | None = None) -> Self:
        """A client that serves every call from the recording at ``path``, by prompt and call index, with no network.

        Only calls recorded under ``settings`` are served. ``settings`` is required and has no default, as it is for
        :meth:`record`: a replaying client is asked only a prompt, so the settings the replayed client is built with
        are known to the caller alone, and a recording made under others is an answer the current configuration never
        asked for.

        :param path: The recording.
        :ptype path: Path
        :param settings: The request settings the client this one stands in for is built with, the same value
            :meth:`record` was given for it, or ``None`` for one built with none (which matches only calls recorded
            with none).
        :ptype settings: ClientRequestSettings | None
        :param rerecord: The command that re-records ``path`` (``"RECORD=1 pytest tests/test_judge.py"``), quoted in
            every refusal; ``None`` says to re-record it with the client it stands in for.
        :ptype rerecord: str | None
        :return: The replaying client.
        :rtype: Self
        :raises RecordingRefused: ``path`` does not exist, holds no calls, holds a line this build cannot read, or holds
            one from a newer format.
        """
        recorded, cells, heads = _readable_replay(path, _rerecord_remedy(path, rerecord))
        client = cls(path, inner=None, settings=settings, recorded=recorded, rerecord=rerecord)
        client._cells = cells
        client._heads = heads
        return client

    @classmethod
    def check_covers(
        cls,
        path: Path,
        prompts: Iterable[RecordedPrompt],
        *,
        settings: ClientRequestSettings | None,
        rerecord: str | None = None,
    ) -> None:
        """Refuse, naming every prompt, when the recording at ``path`` no longer covers ``prompts``.

        What a test calls before a replay starts, so a prompt that changed under its recording is named as such
        rather than surfacing as a miss partway through. A prompt listed ``n`` times needs ``n`` recorded calls.
        ``settings`` selects the client's calls as :meth:`replay` does, and a prompt held only under other request
        settings is named with the settings that differ.

        :param path: The recording.
        :ptype path: Path
        :param prompts: Every call the code under test will make.
        :ptype prompts: Iterable[RecordedPrompt]
        :param settings: As :meth:`replay`'s, and required for the same reason.
        :ptype settings: ClientRequestSettings | None
        :param rerecord: As :meth:`replay`'s.
        :ptype rerecord: str | None
        :raises RecordingRefused: The recording cannot be read (as :meth:`replay`), or holds fewer calls than asked for
            one or more prompts, or holds them only under other request settings, each named.
        """
        remedy = _rerecord_remedy(path, rerecord)
        recorded, _, _ = _readable_replay(path, remedy)
        uncovered: list[str] = []
        for prompt, asked in _tally(prompts):
            key, others = _lookup(recorded, prompt, settings)
            if key is None and others:
                uncovered.append(
                    f"{_head(prompt.user)} is recorded only under other request settings "
                    f"({_settings_drift(others, settings)})"
                )
                continue
            held = 0 if key is None else len(recorded[key][1])
            if held < asked:
                uncovered.append(f"{_head(prompt.user)} is asked {asked} time(s) and recorded {held}")
        if uncovered:
            raise RecordingRefused(
                path, f"the recording no longer covers {len(uncovered)} prompt(s): " + "; ".join(uncovered), remedy
            )

    @property
    def model_name(self) -> str:
        """The model this client calls (:attr:`~threetears.evals.schema.BoundCompletionClient.model_name`).

        Recording, the inner client's. Replaying, the one model the recording holds.

        :raises AttributeError: Recording, the inner client names no model: it is a plain ``CompletionClient``, so this
            recorder cannot stand in for a bound one.
        """
        if self._inner is not None:
            named = getattr(self._inner, "model_name", None)
            if not isinstance(named, str):
                raise AttributeError(
                    f"model_name: the recorded client {type(self._inner).__name__} names no model (it is a "
                    "CompletionClient, not a BoundCompletionClient), so its recorder cannot stand in for an analysis "
                    "or out-of-run client; record a client that names its model"
                )
            return named
        model, _ = _client_of(self._recorded)
        return model

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        """The most one :meth:`generate` can cost
        (:meth:`~threetears.evals.schema.BoundCompletionClient.price_ceiling`).

        Recording, the inner client's ceiling, since every call goes to it. Replaying, ``0.0``: a replay sends
        nothing to a provider and costs nothing (the recorded ``cost_usd`` it hands back is a historical fact, not a
        charge).

        :param system: The system message the call would send.
        :ptype system: str
        :param user: The user message the call would send.
        :ptype user: str
        :param response_format: The directive the call would send, or ``None``.
        :ptype response_format: dict[str, Any] | None
        :return: The ceiling in dollars, or ``None`` when the inner client cannot say.
        :rtype: float | None
        :raises AttributeError: Recording, the inner client prices nothing: it is a plain ``CompletionClient``.
        """
        if self._inner is None:
            return 0.0
        ceiling = getattr(self._inner, "price_ceiling", None)
        if ceiling is None:
            raise AttributeError(
                f"price_ceiling: the recorded client {type(self._inner).__name__} prices no call (it is a "
                "CompletionClient, not a BoundCompletionClient), so its recorder cannot stand in for an out-of-run "
                "client; record a client that prices its calls"
            )
        result: float | None = ceiling(system=system, user=user, response_format=response_format)
        return result

    async def generate(
        self,
        *,
        system: str,
        user: str,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Record or replay one call.

        :param system: The system prompt.
        :ptype system: str
        :param user: The user message.
        :ptype user: str
        :param response_format: The provider directive, or ``None``.
        :ptype response_format: dict[str, Any] | None
        :return: Recording, the inner client's completion, unchanged; replaying, the recorded one as a
            :class:`ReplayedCompletion`.
        :rtype: CompletionResult
        :raises RecordingRefused: Replaying, the recording holds no call for this prompt, holds it only under other
            request settings, or has served every call it holds for it (to another eval's run, named as the cause). Recording, the completion names
            another model or temperature than the calls the recording holds.
        :raises CompletionConformanceFailure: Recording, the inner client's completion lacks a member, or holds a value
            of the wrong type or one JSON cannot hold.
        """
        if self._inner is None:
            return self._serve(system, user, response_format)
        tag = self._numbered(system, user, response_format)  # numbered as it is asked, as a replay numbers it
        completion = await self._inner.generate(system=system, user=user, response_format=response_format)
        self._append(system, user, response_format, completion, tag)
        return completion

    def _numbered(self, system: str, user: str, response_format: dict[str, Any] | None) -> _CellAsk | None:
        """The cell a call about to be made is made for, and its ask number: the cell's asks of the prompt so far.

        Numbered when the call is asked, never when its answer lands, so two asks of one prompt by one cell at once
        are numbered in the order they were asked, as a replay numbers them. Numbers rise from past every ask the
        file already holds for the cell under the prompt, so recording onto a file never repeats one. An ask whose
        call fails is not written, and leaves a gap a replay does not see: it serves a cell's asks in number order.
        """
        cell = current_cell_identity()
        if cell is None:
            return None
        coordinates = _coordinates(cell)
        prompt = canonical_digest({"system": system, "user": user, "response_format": response_format})
        with self._lock:
            ask = self._next_ask.get((prompt, coordinates))
            if ask is None:
                ask = 0
                if self._recorded:
                    model, temperature = _client_of(self._recorded)
                    key = _call_key(system, user, response_format, model, temperature, self._settings)
                    held = [tag.ask for tag in self._cells.get(key, []) if tag is not None and tag[:4] == coordinates]
                    ask = max(held, default=-1) + 1
            self._next_ask[(prompt, coordinates)] = ask + 1
        return _CellAsk(*coordinates, ask)

    def _serve(self, system: str, user: str, response_format: dict[str, Any] | None) -> ReplayedCompletion:
        key, others = _lookup(self._recorded, RecordedPrompt(system, user, response_format), self._settings)
        if key is None and others:
            raise RecordingRefused(
                self.path,
                f"the prompt {_head(user)} is recorded only under other request settings "
                f"({_settings_drift(others, self._settings)}), and a recording serves only a client built with the "
                "settings it was made under",
                self._remedy,
            )
        if key is None:
            model, temperature = _client_of(self._recorded)
            asked = _call_key(system, user, response_format, model, temperature, self._settings)
            raise RecordingRefused(
                self.path,
                f"no call is recorded for the prompt {_head(user)} (call 0, key {asked[:_KEY_DIGITS]}): the prompt "
                f"changed since it was recorded, or the code now asks something it did not ask then; "
                f"{_nearest(self._heads, user)}",
                self._remedy,
            )
        cell = current_cell_identity()
        asking = running_eval()
        with self._lock:
            _, calls = self._recorded[key]
            taken = self._taken.setdefault(key, {})
            index = self._index_of(key, cell, user)
            if index is None:
                raise self._spent(user, len(calls), taken.values(), asking)
            taken[index] = asking
        return calls[index]

    def _index_of(self, key: str, cell: CellIdentity | None, user: str) -> int | None:
        """Which of ``key``'s calls the asking ``cell`` is served; ``None`` when every one it could be is served.

        Called under the lock. In order:

        1. **A cell the recording names under the key** is served its own calls, its ``n``-th ask its ``n``-th by
           ask number, and nothing else: one asking more often than it was recorded is refused. A cell the recording
           names under other keys only is served by rule 3 alone, never as another cell.
        2. **A cell line bound to a recorded one** (rule 4) is served that line's call of its repeat, ``n``-th ask
           ``n``-th.
        3. **Otherwise a call that names no cell** (format 2, or made outside every cell): repeat ``i`` is served the
           call of index ``i`` when it names none and is left, and otherwise the first such call left.
        4. **Otherwise the one recorded line it can be**: a cell named nowhere in the recording (its arm, its case or
           its eval renamed since) is bound, for the rest of the replay, to the one recorded line under the key that
           asked at its repeat as often and shares the most of its subject, its arm and its case (so a case reminted
           in a fresh store binds each arm to its own recorded arm). With none sharing any it is refused, saying so;
           with several sharing as many, as ambiguous, whatever order they ask in; and where that one line has asked as
           itself, or is already stood for by another cell, as not its own. A call outside every cell, when only
           calls made inside cells are left, is refused as ambiguous too.

        So a cell is never served a call recorded for a cell that has asked as itself, and which call a cell is
        served turns on no order but in one case: several cells whose best fit is one same recorded line (the recorded
        cell itself and a stand-in, or two stand-ins: an arm added beside a recorded arm whose every prompt it shares,
        rather than one renamed, or two cases of one arm where the recording held one), where the first to ask takes
        it and each later one is refused, naming who did.
        """
        tags = self._cells[key]
        taken = self._taken.get(key, {})
        repeat = cell.k_iteration if cell is not None else None
        coordinates = _coordinates(cell) if cell is not None else None
        ask = 0
        if coordinates is not None:
            ask = self._asks.get((key, coordinates), 0)
            self._asks[(key, coordinates)] = ask + 1
            line = coordinates[:3]
            if line in self._lines():
                self._live.add(line)
                if (bound := next((by for by, to in self._bound.items() if to == line), None)) is not None:
                    raise RecordingRefused(
                        self.path,
                        f"the calls recorded for the cell {_spelled(line)} were served to the cell {_spelled(bound)}, "
                        "which the recording names nowhere and which asked before this one did, the one recorded cell "
                        "it fit: an arm added beside one whose every prompt it shares cannot be told from a rename",
                        self._remedy,
                    )
            own = _ranked(tags, coordinates)
            if own:
                return own[ask] if ask < len(own) and own[ask] not in taken else None
            if (stands_for := self._bound.get(line)) is not None:
                recorded = _ranked(tags, (*stands_for, repeat))
                return recorded[ask] if ask < len(recorded) and recorded[ask] not in taken else None
        free = [at for at in range(len(tags)) if at not in taken]
        unnamed = [at for at in free if tags[at] is None]
        if unnamed:
            return repeat if repeat in unnamed else unnamed[0]
        if coordinates is not None and coordinates[:3] in self._live:
            return None  # a cell the recording names stands for no other, however many calls are left
        if coordinates is None:
            if not free:
                return None
            raise RecordingRefused(
                self.path,
                f"the prompt {_head(user)} is asked outside every cell, and every call recorded for it that is left "
                "was made for a cell: which one this ask stands for cannot be told",
                self._remedy,
            )
        line = coordinates[:3]
        fitting = {
            tag[:3]: _shared(tag[:3], line)
            for tag in tags
            if tag is not None and tag.repeat == repeat and _shared(tag[:3], line) and len(_ranked(tags, tag[:4])) > ask
        }
        # The lines sharing the most of its subject, arm and case: a case reminted in a fresh store leaves each arm's
        # cell its own recorded arm's line, never another arm's that shares only the subject.
        most = max(fitting.values(), default=0)
        candidates = sorted((fits for fits, shared in fitting.items() if shared == most), key=repr)
        if len(candidates) > 1:
            raise RecordingRefused(
                self.path,
                f"the cell {_spelled(line)} is named nowhere in the recording, and the prompt {_head(user)} was asked "
                f"at its repeat by {len(candidates)} recorded cells it could stand for "
                f"({', '.join(_spelled(c) for c in candidates)}): which is its own cannot be told",
                self._remedy,
            )
        if not candidates and not free:
            return None  # every call under the key is served: the refusal names who took them
        if not candidates:
            raise RecordingRefused(
                self.path,
                f"the cell {_spelled(line)} is named nowhere in the recording, and no recorded cell it could stand for "
                f"(one sharing its subject, arm or case) asked the prompt {_head(user)} at its repeat {ask + 1} "
                "time(s): its case, arm and eval all changed since it was recorded, or it asks what it did not ask then",
                self._remedy,
            )
        (stands_for,) = candidates
        holder = next((by for by, to in self._bound.items() if to == stands_for), None)
        if stands_for in self._live or holder is not None:
            raise RecordingRefused(
                self.path,
                f"the cell {_spelled(line)} is named nowhere in the recording, and the one recorded cell it could "
                f"stand for, {_spelled(stands_for)}, is "
                + ("asking as itself" if holder is None else f"already stood for by the cell {_spelled(holder)}")
                + ": its calls are not this cell's",
                self._remedy,
            )
        self._bound[line] = stands_for
        recorded = _ranked(tags, (*stands_for, repeat))
        return recorded[ask] if recorded[ask] not in taken else None

    def _lines(self) -> set[_CellLine]:
        """Every cell line the recording names, under any key."""
        return {tag[:3] for tags in self._cells.values() for tag in tags if tag is not None}

    def _spent(
        self, user: str, recorded: int, takers: Iterable[RunningEval | None], asking: RunningEval | None
    ) -> RecordingRefused:
        """The refusal of a prompt every recorded call of which is served: to another eval's run, or to this code."""
        others = sorted({taker.name for taker in takers if taker is not None and taker is not asking})
        if others:
            return RecordingRefused(
                self.path,
                f"the prompt {_head(user)} is recorded {recorded} time(s), and every one was already served to "
                f"another eval's run in this process ({', '.join(map(repr, others))}): a replay serves each recorded "
                "call once per model, so one model shared by two evals, or by one eval run twice, serves the second "
                "none",
                "build a Model per eval; the recording is whole, so record nothing again",
            )
        return RecordingRefused(
            self.path,
            f"the prompt {_head(user)} is recorded {recorded} time(s), and this is call {recorded} (counting from 0): "
            "the code asks it more often than it did when it was recorded",
            self._remedy,
        )

    def _append(
        self,
        system: str,
        user: str,
        response_format: dict[str, Any] | None,
        completion: CompletionResult,
        tag: _CellAsk | None,
    ) -> None:
        check_completion_conformance(completion)
        members = _recordable_members(completion)
        sent = sent_settings_of(completion)
        extra = _declared_extras(completion)
        key = _call_key(system, user, response_format, members["model"], members["temperature"], self._settings)
        with _APPEND_LOCK, self._lock:
            if _file_size(self.path) != self._size:
                self._recorded, self._cells, _ = (
                    _read_recording(self.path, self._remedy) if self.path.exists() else ({}, {}, {})
                )
            if self._recorded and (held := _client_of(self._recorded)) != (members["model"], members["temperature"]):
                raise RecordingRefused(
                    self.path,
                    f"it holds calls of model {held[0]!r} at temperature {held[1]!r}, and this call was answered by "
                    f"{members['model']!r} at {members['temperature']!r}: a recording holds one model at one temperature",
                    "record this client into a file of its own",
                )
            _, calls = self._recorded.setdefault(key, (self._settings, []))
            tags = self._cells.setdefault(key, [])
            line = {
                "recorder_version": _FORMAT_VERSION,
                "recorded_at": datetime.now(UTC).isoformat(),
                "key": key,
                "index": len(calls),
                "user_head": user[:_PROMPT_HEAD],
                "settings": None if self._settings is None else self._settings.model_dump(mode="json"),
                "sent_settings": None if sent is None else sent.model_dump(mode="json"),
                "result": members,
                "extra": extra,
                "cell": None if tag is None else tag._asdict(),
            }
            with self.path.open("a", encoding="utf-8") as recording:
                recording.write(json.dumps(line, ensure_ascii=False, allow_nan=False) + "\n")
            calls.append(_completion_of(members, sent, extra))
            tags.append(tag)
            self._size = _file_size(self.path)

    async def refresh_prices(self) -> None:
        """Bring the prices :meth:`price_ceiling` reads up to date: recording, the inner client's; replaying, none
        are read."""
        if self._inner is not None:
            await refresh_prices(self._inner)

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


def _ranked(tags: Sequence[_CellAsk | None], coordinates: _CellCoordinates) -> list[int]:
    """The indexes of the calls recorded for the cell at ``coordinates``, in the order it asked them (by ask number)."""
    return [
        at
        for _, at in sorted((tag.ask, at) for at, tag in enumerate(tags) if tag is not None and tag[:4] == coordinates)
    ]


def _spelled(line: _CellLine) -> str:
    """A cell line as a refusal names it."""
    subject, variant, case = line
    return f"(subject {subject!r}, arm {variant!r}, case {case!r})"


def _rerecord_remedy(path: Path, rerecord: str | None) -> str:
    if rerecord:
        return f"re-record {path} with `{rerecord}`, the provider's key set"
    return f"re-record {path} with the client it stands in for (RecordedCompletions.record(..., replace=True))"


def _tally(prompts: Iterable[RecordedPrompt]) -> list[tuple[RecordedPrompt, int]]:
    """Each distinct prompt with how many times it is listed, in first-listed order.

    Counted by the prompt's canonical JSON, not by hashing it: a ``response_format`` is a dict, which does not hash,
    and two directives equal as JSON are the same call however their keys were ordered.

    :param prompts: The prompts, each repeat listed once per repeat.
    :ptype prompts: Iterable[RecordedPrompt]
    :return: ``(prompt, times listed)`` per distinct prompt.
    :rtype: list[tuple[RecordedPrompt, int]]
    """
    tally: dict[str, tuple[RecordedPrompt, int]] = {}
    for prompt in prompts:
        identity = json.dumps([prompt.system, prompt.user, prompt.response_format], sort_keys=True)
        first, listed = tally.get(identity, (prompt, 0))
        tally[identity] = (first, listed + 1)
    return list(tally.values())


def _head(user: str) -> str:
    return repr(user[:_PROMPT_HEAD] + ("…" if len(user) > _PROMPT_HEAD else ""))


def _nearest(heads: Mapping[str, str], user: str) -> str:
    """Where a missed prompt parts from the recorded ones, as a clause: how many recorded keys open on the same text
    (a line keeps the first :data:`_PROMPT_HEAD` characters of its user message, so what changed lies past them, or in
    the system prompt or the reply's format), else the nearest recorded key and the character it parts at.
    """
    head = user[:_PROMPT_HEAD]
    shared = {key: len(os.path.commonprefix([head, held])) for key, held in heads.items()}
    alike = sorted(key for key, held in heads.items() if held == head)
    if alike:
        named = ", ".join(key[:_KEY_DIGITS] for key in alike[:3]) + (", ..." if len(alike) > 3 else "")
        if len(user) < _PROMPT_HEAD:
            # The whole user message is kept and matches: only what no line keeps can have changed.
            return (
                f"{len(alike)} recorded key(s) hold the same user message ({named}), so what changed is the system "
                "prompt or the reply's format"
            )
        return (
            f"{len(alike)} recorded key(s) open on the same {len(head)} characters ({named}), so what changed lies past "
            "them in the user message, or in the system prompt or the reply's format"
        )
    if not shared:
        return "the recording keeps no prompt to compare it with"
    nearest = min(shared, key=lambda key: (-shared[key], key))
    return (
        f"the nearest recorded key, {nearest[:_KEY_DIGITS]}, parts from it at character {shared[nearest] + 1} "
        f"({_head(heads[nearest])})"
    )


def _stated(settings: ClientRequestSettings | None, names: Iterable[str]) -> str:
    if settings is None:
        return "no request settings"
    return ", ".join(f"{name}={getattr(settings, name)!r}" for name in names)


def _drift(
    first: ClientRequestSettings | None,
    second: ClientRequestSettings | None,
    labels: tuple[str, str] = ("recorded with", "asked with"),
) -> str:
    """How two request settings values differ: each field that differs, the first's value against the second's.

    A value against none names every field the value holds, but ``unsent_reasoning_max_tokens`` while it is unset:
    a field that says something only when set, which every recording made before it existed holds unset.
    """
    held = [settings for settings in (first, second) if settings is not None]
    names = [
        name
        for name in ClientRequestSettings.model_fields
        if name != "unsent_reasoning_max_tokens" or any(s.unsent_reasoning_max_tokens is not None for s in held)
    ]
    if first is not None and second is not None:
        names = [name for name in names if getattr(first, name) != getattr(second, name)]
    return f"{labels[0]} {_stated(first, names)}, {labels[1]} {_stated(second, names)}"


def _settings_drift(held: Iterable[ClientRequestSettings | None], asked: ClientRequestSettings | None) -> str:
    """How each request settings value a prompt is recorded under differs from the ones asked, once per value."""
    return "; ".join(dict.fromkeys(_drift(settings, asked) for settings in held))


def _readable_replay(path: Path, remedy: str) -> _Read:
    """The calls a recording holds, the cells they name and the head of each key's prompt, refused when there is none
    to replay: no file, or no calls."""
    if not path.exists():
        raise RecordingRefused(path, "there is no recording at this path", remedy)
    read = _read_recording(path, remedy)
    if not read.recorded:
        raise RecordingRefused(path, "the recording holds no calls, so it can serve none", remedy)
    return read


def _client_of(recorded: Mapping[str, _Recorded]) -> _Client:
    """The one model and temperature a non-empty recording's calls hold (:func:`_read_recording` holds it to one)."""
    first = next(call for _, calls in recorded.values() for call in calls)
    return first.model, first.temperature


def _lookup(
    recorded: Mapping[str, _Recorded], prompt: RecordedPrompt, settings: ClientRequestSettings | None
) -> tuple[str | None, list[ClientRequestSettings | None]]:
    """The key ``prompt`` is recorded under for ``settings``; else ``None`` and every other settings value it is under."""
    model, temperature = _client_of(recorded)

    def key(under: ClientRequestSettings | None) -> str:
        return _call_key(prompt.system, prompt.user, prompt.response_format, model, temperature, under)

    if key(settings) in recorded:
        return key(settings), []
    # Settings are not hashable; their canonical JSON tells two values apart as the key does.
    held = {json.dumps(_settings_member(under), sort_keys=True): under for under, _ in recorded.values()}
    return None, [under for under in held.values() if key(under) in recorded]


def _file_size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _settings_member(settings: ClientRequestSettings | None) -> dict[str, Any] | None:
    """Request settings as a key covers them: the fields that differ from their defaults, or ``None`` for none.

    Defaults are left out so a field ``ClientRequestSettings`` gains later, at a default meaning what was sent before
    it existed, leaves every recorded key where it was; two values holding the same settings key alike either way.
    """
    return None if settings is None else settings.model_dump(mode="json", exclude_defaults=True)


def _call_key(
    system: str,
    user: str,
    response_format: dict[str, Any] | None,
    model: str,
    temperature: float | None,
    settings: ClientRequestSettings | None,
) -> str:
    """The sha256 a call is recorded and replayed under. Raises ``UnhashableContentError`` for a directive JSON cannot hold."""
    return canonical_digest(
        {
            "system": system,
            "user": user,
            "response_format": response_format,
            "model": model,
            "temperature": temperature,
            "settings": _settings_member(settings),
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
            ok = value is None or (number and math.isfinite(value))
        case "price_source" | "served_model":
            ok = value is None or isinstance(value, str)
        case "stop_reason":
            ok = value in get_args(StopReason)
        case _:  # pragma: no cover — every protocol member is matched above
            ok = False
    if ok:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return f"`{name}` holds {value!r}, a number JSON cannot hold"
    return f"`{name}` holds {value!r} ({type(value).__name__})"


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
    """Every attribute ``completion``'s type declares beyond the protocol's ten and its optional members, JSON-safe."""
    if is_dataclass(completion) and not isinstance(completion, type):
        names = [f.name for f in fields(completion)]
    elif isinstance(declared := getattr(type(completion), "model_fields", None), Mapping):
        names = list(declared)  # a Pydantic model's fields, read off the class without importing Pydantic
    else:
        slots = [name for klass in type(completion).__mro__ for name in _slot_names(klass)]
        names = [*getattr(completion, "__dict__", {}), *(name for name in slots if hasattr(completion, name))]
    extras: dict[str, Any] = {}
    for name in dict.fromkeys(names):
        if name.startswith("_") or name in COMPLETION_RESULT_ATTRIBUTES or name in OPTIONAL_COMPLETION_ATTRIBUTES:
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


def _completion_of(
    members: Mapping[str, Any], sent: ClientRequestSettings | None, extra: Mapping[str, Any]
) -> ReplayedCompletion:
    return ReplayedCompletion(**members, sent_settings=sent, extra=MappingProxyType(dict(extra)))


def _read_recording(path: Path, remedy: str) -> _Read:
    """Every call ``path`` holds, by key in index order with its settings, the cell each names, and the head of each
    key's prompt; each line checked, and a bad line refuses all."""
    recorded: dict[str, _Recorded] = {}
    cells: dict[str, list[_CellAsk | None]] = {}
    heads: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # the newline that ends the last line
    client: _Client | None = None
    for number, text in enumerate(lines, start=1):
        key, index, settings, completion, cell, head = _read_line(path, number, text, remedy)
        if client is None:
            client = (completion.model, completion.temperature)
        elif (completion.model, completion.temperature) != client:
            raise _malformed(
                path,
                number,
                f"it holds a call of model {completion.model!r} at temperature {completion.temperature!r}, and the "
                f"lines before it model {client[0]!r} at {client[1]!r}: a recording holds one model at one temperature",
                remedy,
            )
        held, calls = recorded.setdefault(key, (settings, []))
        if index != len(calls):
            raise _malformed(
                path,
                number,
                f"its call index is {index}, and the lines before it hold {len(calls)} call(s) under its key: "
                "the file was edited, or two processes recorded onto it at once",
                remedy,
            )
        if settings != held:
            raise _malformed(
                path,
                number,
                f"its settings are not the ones the lines before it under its key carry "
                f"({_drift(held, settings, ('they carry', 'it carries'))}), and a key covers its settings: the file "
                "was edited",
                remedy,
            )
        calls.append(completion)
        cells.setdefault(key, []).append(cell)
        heads.setdefault(key, head)
    return _Read(recorded, cells, heads)


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


def _read_settings(
    path: Path, number: int, settings: Any, remedy: str, member: str = "settings"
) -> ClientRequestSettings | None:
    """A line's ``settings`` or ``sent_settings``: ``null``, or request settings this build reads exactly, uncoerced."""
    if settings is None:
        return None
    if not isinstance(settings, dict):
        raise _malformed(
            path, number, f"its {member} are a JSON {type(settings).__name__}, not an object or null", remedy
        )
    try:
        return ClientRequestSettings.model_validate(settings, strict=True)
    except ValueError as invalid:  # a pydantic ValidationError is one
        raise _malformed(
            path,
            number,
            f"its {member} are not request settings this build reads ({' '.join(str(invalid).split())})",
            remedy,
        ) from None


def _read_cell(path: Path, number: int, cell: Any, remedy: str) -> _CellAsk | None:
    """A format-3 line's ``cell``: ``null`` (made outside every cell), or the cell and its ask, each member typed."""
    if cell is None:
        return None
    if not isinstance(cell, dict):
        raise _malformed(path, number, f"its cell is a JSON {type(cell).__name__}, not an object or null", remedy)
    if mismatch := _mismatch(_CELL_MEMBERS, cell.keys()):
        raise _malformed(path, number, f"its cell {mismatch}", remedy)

    def whole(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value >= 0

    typed = (
        isinstance(cell["subject"], str)
        and (cell["variant"] is None or isinstance(cell["variant"], str))
        and isinstance(cell["case"], str)
        and (cell["repeat"] is None or whole(cell["repeat"]))
        and whole(cell["ask"])
    )
    if not typed:
        raise _malformed(
            path,
            number,
            f"its cell {cell!r} is not a cell (subject and case text, variant text or null, repeat a count or null, "
            "ask a count)",
            remedy,
        )
    return _CellAsk(cell["subject"], cell["variant"], cell["case"], cell["repeat"], cell["ask"])


def _read_line(
    path: Path, number: int, text: str, remedy: str
) -> tuple[str, int, ClientRequestSettings | None, ReplayedCompletion, _CellAsk | None, str]:
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
    if version < _OLDEST_FORMAT:
        raise RecordingRefused(
            path,
            f"line {number} was written in recording format {version}, older than the oldest this build reads, "
            f"{_OLDEST_FORMAT}: it kept no sent_settings, so its calls cannot say what their requests were sent with",
            remedy,
        )
    if mismatch := _mismatch(_LINE_MEMBERS[version], line.keys()):
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
    settings = _read_settings(path, number, line["settings"], remedy)
    sent = _read_settings(path, number, line["sent_settings"], remedy, "sent_settings")
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
    cell = _read_cell(path, number, line.get("cell"), remedy)
    return key, index, settings, _completion_of(members, sent, extra), cell, line["user_head"]
