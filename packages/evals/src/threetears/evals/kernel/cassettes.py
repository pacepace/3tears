"""The cassette seams a candidate kind supplies, and the cell handle it supplies them to.

A cassette run records what a candidate's tools answered (``cassette_mode='capture'``) or re-serves a
recording in place of running them (``'replay'``). The engine never reaches into a candidate to do
that: it hands each cell's :meth:`~threetears.evals.kernel.candidate_kind.CandidateKind.prepare` a
:class:`CellCassettes`, and the kind calls :meth:`CellCassettes.wire` once with the
:class:`CassetteSeams` its candidate exposes. This module is that vocabulary — what a kind implements
and what it is handed. The engine half that records and replays is
:mod:`threetears.evals.run.cassette_proxy`.

It lives in contracts for the reason :class:`~threetears.evals.kernel.candidate_kind.CellSpanWindow`
does: ``prepare`` is declared here, a kind is implemented on both sides of the run/analysis line, and
contracts is the one package every implementer may import.

Two seams, because ``act()`` is not where every tool answers
------------------------------------------------------------

A synchronous tool returns its result from ``act()``, so wrapping ``act()`` records and replays
everything about the call (:class:`ActionSeam`). A host whose tools run as plain blocking calls inside
its own turn loop declares :class:`SyncActionSeam` instead, and its tools are driven through
``act_sync()``; the two action seams record and replay through one implementation, so a recording made
on either path replays on either. An **asynchronous** one does not: its ``act()``
returns an acknowledgement, the real work runs elsewhere, and the result reaches the candidate later.
Recording ``act()`` there would record the acknowledgement, so the kind reports the work itself
(:class:`DeliverySeam`): capture hands it a :class:`DeliveryRecorder`, replay a :class:`DeliveryReplay`.

Every recording is keyed by what was asked, and by which time it was asked
-------------------------------------------------------------------------

An action is keyed by its tool, action and parameters; a piece of background work by its tool and its
request. Both also carry an **occurrence**: the Nth time this cell asked exactly that. A session that
rolls ``1d20`` twice replays the two answers it rolled, in the order it asked for them, and a
session that sends one scout north and another to the chapel gets each its own report, however the
two raced each other home in the capture. Asking an (N+1)th time when the capture asked N times is a
:class:`CassetteExhausted`; asking something the capture never asked is a :class:`CassetteMiss`.
Neither ever serves a neighbouring answer in its place, and neither ever runs the tool live.

Misses are loud, and they are the rig's
---------------------------------------

Every replay failure is an :class:`~threetears.evals.kernel.host.apparatus.ApparatusError`, so a
kind's tool boundary must re-raise it rather than absorb it into an ordinary failed tool result — the
cell is the measuring rig's failure, not the candidate's.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self, runtime_checkable

from threetears.evals.kernel.host.apparatus import ApparatusError
from threetears.evals.schema.models import CassetteKey

__all__ = [
    "ActionSeam",
    "CassetteCorrupt",
    "CassetteExhausted",
    "CassetteMiss",
    "CassetteMode",
    "CassetteSeams",
    "CellCassettes",
    "DeliveryOutcome",
    "DeliveryRecorder",
    "DeliveryReplay",
    "DeliverySeam",
    "DeliveryTicket",
    "Recordable",
    "ReplayedDelivery",
    "SyncActionSeam",
    "SyncToolLike",
    "SyncToolWrap",
    "ToolLike",
    "ToolWrap",
]


#: A run's cassette mode. ``'off'`` runs every tool live and records nothing, so a cell of such a run
#: is handed no :class:`CellCassettes` at all.
CassetteMode = Literal["capture", "replay", "off"]

#: How a recorded piece of background work ended when its capture cell did: it delivered a payload,
#: it failed, or the cell ended with it still in flight.
DeliveryOutcome = Literal["delivered", "failed", "undelivered"]


# =============================================================================
# Replay failures — every one the rig's, never the candidate's
# =============================================================================


class CassetteMiss(ApparatusError, LookupError):
    """A replay asked for something its corpus never recorded.

    Carries the whole key, so an operator can see exactly what the candidate asked that the capture
    did not. ``LookupError`` is a base too, so an ``except LookupError`` reader still catches it.
    """

    def __init__(self, key: CassetteKey) -> None:
        """Name the key the corpus has no recording for.

        Args:
            key: What the replay looked up.
        """
        self.key = key
        super().__init__(
            f"Cassette miss in replay mode: the corpus never recorded tool={key.tool!r} action={key.action!r} "
            f"params_hash={key.params_hash!r} (corpus={key.corpus_id!r} template_id={key.template_id!r} "
            f"test_case_id={key.test_case_id!r}). Re-capture the template (cassette_mode='capture') to record it."
        )


class CassetteExhausted(ApparatusError, LookupError):
    """A replay asked for something one more time than its capture did.

    The corpus recorded ``key.occurrence`` answers to this exact ask and the candidate is asking for
    another. Serving one of the recorded answers again would hand a second dice roll the first roll's
    number, so it stops the cell instead.
    """

    def __init__(self, key: CassetteKey) -> None:
        """Name the ask and how many answers to it were recorded.

        Args:
            key: The occurrence the replay looked up, one past the last recorded.
        """
        self.key = key
        self.recorded = key.occurrence
        super().__init__(
            f"Cassette replay exhausted: the corpus recorded tool={key.tool!r} action={key.action!r} "
            f"params_hash={key.params_hash!r} {key.occurrence} time(s) and the candidate asked again "
            f"(corpus={key.corpus_id!r} template_id={key.template_id!r} test_case_id={key.test_case_id!r}). "
            "Re-capture the template (cassette_mode='capture') if the candidate now legitimately asks more often."
        )


class CassetteCorrupt(ApparatusError, ValueError):
    """The corpus cannot be used as recorded.

    A recording could not be read back — its stored document no longer loads (written under another
    schema, or carrying a field this build does not know), the store failed to answer, or its payload
    no longer rebuilds as the type its seam declares — or a capture could not clear a case's previous
    recording, which would leave two sessions mixed in one corpus. An
    :class:`~threetears.evals.kernel.host.apparatus.ApparatusError`
    rather than the underlying error, because a ``ValueError`` reaching a kind's tool boundary would
    be absorbed into an ordinary failed tool result and the candidate would be scored on its manner
    toward a broken corpus.
    """


# =============================================================================
# What a cassette stores — a tool's answer, as a type the seam declares
# =============================================================================


@runtime_checkable
class Recordable(Protocol):
    """The two members the cassette layer uses on whatever it records: an action result or a delivered payload.

    Capture stores the value as JSON (:meth:`model_dump`); replay rebuilds one from that JSON
    (:meth:`model_validate`), strictly, so a corrupted recording raises :class:`CassetteCorrupt`
    rather than reaching the candidate as a malformed answer. Nothing else about the value is read —
    its fields belong to the kind that defined them. Any pydantic model satisfies it.

    Replay needs the *class*, since it has only the stored dict to build from, so each seam declares
    its types as values (:attr:`ActionSeam.recorded_tools`, :attr:`DeliverySeam.payload_type`).
    """

    def model_dump(self, *, mode: Literal["json"]) -> dict[str, Any]:
        """The value as a JSON-safe dict — what a cassette's ``response`` stores."""
        ...  # pragma: no cover — protocol

    @classmethod
    def model_validate(cls, obj: Any) -> Self:
        """Rebuild a value from a recorded ``response``, raising ``ValueError`` on a shape that does not fit."""
        ...  # pragma: no cover — protocol


@runtime_checkable
class ToolLike(Protocol):
    """The three members the cassette layer calls on a synchronous tool it wraps.

    Structural, and deliberately narrower than any host's tool type: this is the whole of what the
    engine's proxy *calls*. Everything else a host's tool has is forwarded by the proxy, so the
    candidate's own machinery sees the wrapped tool as the tool.
    """

    @property
    def name(self) -> str:
        """The tool's own name — also the ``tool`` component of a cassette key."""
        ...  # pragma: no cover — protocol

    def can_dispatch(self, action: str) -> bool:
        """Whether ``action`` is a name this tool is known to dispatch.

        The proxy asks before the store does: an undispatchable name is not a recordable event, so
        it goes straight to the tool for that tool's ordinary unknown-action answer. A tool that
        routes any name answers ``True``.
        """
        ...  # pragma: no cover — protocol

    async def act(self, action: str, parameters: dict[str, Any]) -> Recordable:
        """Perform one action and return its result — what capture records and replay serves."""
        ...  # pragma: no cover — protocol


#: What the action seam is handed: maps the candidate's tools, by name, to the tools it should use —
#: each declared recorded tool wrapped for the cassette run, every other one unchanged.
ToolWrap = Callable[[Mapping[str, ToolLike]], dict[str, ToolLike]]


@runtime_checkable
class SyncToolLike(Protocol):
    """The three members the cassette layer calls on a tool that answers as a plain blocking call.

    :class:`ToolLike` with ``act_sync`` in place of the awaitable ``act``: for a host whose tools run
    synchronously inside its own turn loop, where there is no event loop to await on. The method has
    its own name rather than a synchronous ``act`` because a runtime check cannot tell a coroutine
    function from a plain one by name alone, and a proxy answering both must never confuse them.
    """

    @property
    def name(self) -> str:
        """The tool's own name — also the ``tool`` component of a cassette key."""
        ...  # pragma: no cover — protocol

    def can_dispatch(self, action: str) -> bool:
        """Whether ``action`` is a name this tool is known to dispatch (see :meth:`ToolLike.can_dispatch`)."""
        ...  # pragma: no cover — protocol

    def act_sync(self, action: str, parameters: dict[str, Any]) -> Recordable:
        """Perform one action, blocking, and return its result — what capture records and replay serves."""
        ...  # pragma: no cover — protocol


#: What the synchronous action seam is handed: :data:`ToolWrap` over :class:`SyncToolLike` tools.
SyncToolWrap = Callable[[Mapping[str, SyncToolLike]], dict[str, SyncToolLike]]


# =============================================================================
# The delivery seam's engine objects, as the kind sees them
# =============================================================================


class DeliveryTicket(Protocol):
    """One piece of background work being captured, from its start to however it ends.

    Returned by :meth:`DeliveryRecorder.started`. The kind settles it at most once, where the work
    ends; work still unsettled when the cell ends is recorded as ``'undelivered'``, because that is
    what the capture session saw.
    """

    def delivered(self, payload: Recordable) -> None:
        """Record that the work reached the candidate with ``payload``.

        Args:
            payload: The delivered payload, of the seam's declared ``payload_type``.
        """
        ...  # pragma: no cover — protocol

    def failed(self, error: str) -> None:
        """Record that the work ended in an error and delivered nothing.

        Args:
            error: Why it failed, as the kind would report it.
        """
        ...  # pragma: no cover — protocol


class DeliveryRecorder(Protocol):
    """Captures one asynchronous tool's background work for one cell.

    Handed to :meth:`DeliverySeam.arm_capture`. The kind calls :meth:`started` where the work is
    started — where it is ASKED for, not where it lands — and settles the ticket where it ends. The
    request at the start is what a replay is matched on, so two pieces of work that finish out of
    order are still each recorded against the ask that started them.
    """

    def started(self, request: Mapping[str, Any]) -> DeliveryTicket:
        """Begin recording one piece of background work.

        Args:
            request: What the candidate asked the tool for — the parameters a replay must repeat to
                be served this recording. JSON-shaped.

        Returns:
            The ticket to settle when the work ends.
        """
        ...  # pragma: no cover — protocol


@dataclass(frozen=True)
class ReplayedDelivery[P: Recordable]:
    """One recorded piece of background work, served in place of running it.

    Attributes:
        outcome: How the work ended in the capture. A kind delivers ``payload`` for ``'delivered'``,
            reports ``error`` for ``'failed'``, and delivers nothing for ``'undelivered'`` — the
            capture's session ended with this work still in flight, so no payload exists.
        payload: The recorded payload, rebuilt as the seam's ``payload_type``; set exactly when
            ``outcome`` is ``'delivered'``.
        error: Why the captured work failed; set exactly when ``outcome`` is ``'failed'``.
    """

    outcome: DeliveryOutcome
    payload: P | None = None
    error: str | None = None


class DeliveryReplay[P: Recordable](Protocol):
    """Serves one asynchronous tool's recorded work for one cell.

    Handed to :meth:`DeliverySeam.arm_replay`. The kind's tool calls :meth:`next` where it would have
    started live work, and settles what comes back through its own delivery path.
    """

    def next(self, request: Mapping[str, Any]) -> ReplayedDelivery[P]:
        """Take the recording of this request, in place of starting live work.

        Args:
            request: What the candidate asked the tool for this time.

        Returns:
            The recording of the capture's matching ask — the Nth one for the Nth time it is asked.

        Raises:
            CassetteMiss: The capture never asked this.
            CassetteExhausted: The capture asked this fewer times than the candidate now has.
            CassetteCorrupt: The recording could not be read back.
        """
        ...  # pragma: no cover — protocol


# =============================================================================
# The seams a kind supplies
# =============================================================================


@runtime_checkable
class ActionSeam(Protocol):
    """A kind's synchronous tools, as the cassette lane records and replays them."""

    @property
    def recorded_tools(self) -> Mapping[str, type[Recordable]]:
        """The synchronous tools to record and replay, each mapped to the type its ``act()`` returns.

        Replay rebuilds a tool's recordings as its type. Never empty: a candidate with no synchronous
        tool to record has no action seam.
        """
        ...  # pragma: no cover — protocol

    def arm_tools(self, wrap: ToolWrap) -> None:
        """Replace the candidate's tools with ``wrap(tools)``, for the rest of the cell.

        Call ``wrap`` exactly once, with every tool the candidate can call: a declared recorded tool
        missing from the mapping is refused, because in replay it would run live.

        Args:
            wrap: Maps the current tools to the ones the candidate must use from now on.
        """
        ...  # pragma: no cover — protocol


@runtime_checkable
class SyncActionSeam(Protocol):
    """A kind's synchronous tools whose calls block rather than await, as the cassette lane records and replays them.

    :class:`ActionSeam` for a candidate driving its tools through :meth:`SyncToolLike.act_sync`. The
    rules are the action seam's, enforced by the same code: the same keys and occurrences, the same
    misses, exhaustion and corruption, the same refusal of an unwrapped declared tool. A seam declares
    one of the two, never both: the cell would not know which wrap the candidate calls through.
    """

    @property
    def recorded_tools(self) -> Mapping[str, type[Recordable]]:
        """The tools to record and replay, each mapped to the type its ``act_sync()`` returns. Never empty."""
        ...  # pragma: no cover — protocol

    def arm_sync_tools(self, wrap: SyncToolWrap) -> None:
        """Replace the candidate's tools with ``wrap(tools)``, for the rest of the cell.

        Call ``wrap`` exactly once, with every tool the candidate can call; each recorded tool comes
        back as a proxy whose ``act_sync`` captures or replays.

        Args:
            wrap: Maps the current tools to the ones the candidate must use from now on.
        """
        ...  # pragma: no cover — protocol


@runtime_checkable
class DeliverySeam(Protocol):
    """One asynchronous tool of a kind, as the cassette lane records and replays its background work."""

    @property
    def payload_type(self) -> type[Recordable]:
        """The type of what this tool delivers, which replay rebuilds recordings as."""
        ...  # pragma: no cover — protocol

    def arm_capture(self, recorder: DeliveryRecorder) -> None:
        """Run the tool live, reporting each piece of work to ``recorder`` where it starts and where it ends.

        Args:
            recorder: This cell's recorder for this tool.
        """
        ...  # pragma: no cover — protocol

    def arm_replay(self, replay: DeliveryReplay[Any]) -> None:
        """Start no live work: take each piece from ``replay.next(request)`` and settle it the tool's own way.

        A :class:`CassetteMiss`, :class:`CassetteExhausted` or :class:`CassetteCorrupt` from ``next``
        must propagate, never be absorbed into an ordinary tool failure. Every piece of work served
        this way is reported on ``CandidateOutput.async_deliveries`` with ``substituted=True``: the
        engine knows how many it served and refuses a cell that reports fewer, or reports one live.

        Args:
            replay: This cell's replay for this tool.
        """
        ...  # pragma: no cover — protocol


@runtime_checkable
class CassetteSeams(Protocol):
    """What a kind hands :meth:`CellCassettes.wire`: the seams its candidate exposes.

    Either may be absent, but not both — a candidate with no seams is one a cassette run cannot
    record or replay anything of, and wiring it is refused rather than run live under a replay mode.
    """

    @property
    def action_seam(self) -> ActionSeam | SyncActionSeam | None:
        """The synchronous tools to record — awaited (:class:`ActionSeam`) or blocking (:class:`SyncActionSeam`) — or ``None``."""
        ...  # pragma: no cover — protocol

    @property
    def delivery_seams(self) -> Mapping[str, DeliverySeam]:
        """One seam per asynchronous tool, keyed by the tool's name — the ``tool`` its records carry."""
        ...  # pragma: no cover — protocol


class CellCassettes(Protocol):
    """One cell's handle on its run's cassette lane, as the kind driving that cell sees it.

    Handed to :meth:`~threetears.evals.kernel.candidate_kind.CandidateKind.prepare` as
    ``cassettes`` for every cell of a run whose ``cassette_mode`` is ``'capture'`` or ``'replay'``,
    and ``None`` for a run with cassettes off. It is the cell's, like ``span_window``: it already
    knows the run's corpus, the template and the case, so one kind instance drives every cell and
    names none of them.

    A kind handed one must call :meth:`wire` exactly once, before ``prepare`` returns. The engine
    refuses a cell whose kind did not: a candidate a cassette run never wired would run its tools
    live under a replay.
    """

    @property
    def mode(self) -> Literal["capture", "replay"]:
        """Whether this cell records its tools' answers or replays them."""
        ...  # pragma: no cover — protocol

    def wire(self, seams: CassetteSeams) -> None:
        """Arm every seam the candidate exposes, for the rest of the cell.

        Every refusal of the declaration is made before anything is armed, so a refused declaration
        leaves the candidate as the kind built it.

        Args:
            seams: The candidate's seams.

        Raises:
            TypeError: ``seams`` is not a :class:`CassetteSeams`, or its action seam is neither an
                :class:`ActionSeam` nor a :class:`SyncActionSeam`.
            ValueError: This cell was already wired; ``seams`` declares no seam at all; its action
                seam is both an :class:`ActionSeam` and a :class:`SyncActionSeam`, or names no tool;
                a tool is declared on both seams; or the action seam did not apply the wrap exactly
                once over every tool it declared.
            CassetteCorrupt: A capture could not clear the case's previous recording.
        """
        ...  # pragma: no cover — protocol
