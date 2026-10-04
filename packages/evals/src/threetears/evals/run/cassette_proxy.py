"""Cassette layer — tool recording and replay for deterministic, quota-free evals.

A run with ``cassette_mode='capture'`` executes real tools and records what they answered into its
own corpus; a later run with ``'replay'`` names that corpus (``EvalRun.cassette_corpus_id``) and is
served from it without touching a third-party service. Where a template's prompt effectively
dictates what the candidate asks its tools, that gives a pinned corpus of genuine tool output — a
regression lane for a new prompt, a new model or a new judge that costs nothing to re-run.

The kind supplies the seams; the engine hands each cell its lane
----------------------------------------------------------------

The engine never reaches into a candidate. :func:`~threetears.evals.run.runner.execute_run` builds
the run's :class:`CassetteLane` once (:meth:`CassetteLane.for_run`) and hands each cell's ``prepare``
that cell's :class:`CassetteCell`, which already knows the corpus, the template and the case. The
kind calls :meth:`CassetteCell.wire` with the
:class:`~threetears.evals.contracts.cassettes.CassetteSeams` its candidate exposes, and the cell arms
them; the vocabulary a kind implements is :mod:`threetears.evals.contracts.cassettes`.

* **Action seam** — each declared synchronous tool is swapped for a :class:`CassetteProxy`, which
  wraps ``act()``.
* **Delivery seam** — capture hands the kind a recorder it calls where background work STARTS and
  where it ends; replay hands it a replay it calls where it would have started live work.

Every recording is keyed by what was asked and by which time it was asked
--------------------------------------------------------------------------

A recording's :class:`~threetears.evals.contracts.models.CassetteKey` is the corpus, template, case,
tool, action, a digest of the parameters (or of the background work's request) and the
**occurrence** — the Nth time this cell asked exactly that, counted in the order the candidate
asked. A session that rolls ``1d20`` twice replays both of its rolls in order, and two scouts that
raced each other home in the capture are each paired with the request that sent them. Background
work is recorded against its request at the moment it starts and written when it ends — delivered,
failed, or still in flight when the cell ended — so how the work finished is part of what a replay
serves. Replay never falls back to the real tool:

* :class:`~threetears.evals.contracts.cassettes.CassetteMiss` — the corpus never recorded this ask.
* :class:`~threetears.evals.contracts.cassettes.CassetteExhausted` — it recorded this ask fewer times
  than the candidate has now made it.
* :class:`~threetears.evals.contracts.cassettes.CassetteCorrupt` — a recording could not be read
  back: its row no longer loads, the store failed, or its payload no longer rebuilds as the type the
  seam declares; or a capture could not clear the case's previous recording.

Every one is an :class:`~threetears.evals.contracts.host.apparatus.ApparatusError`. A failed write
during capture is logged and leaves a hole at exactly that key, which a replay reports as a miss or
an exhaustion naming it — never as a neighbouring answer served in its place.

One corpus per capture run, one session per case
------------------------------------------------

A corpus is the capture run's own, named by its id, so two runs capturing at once — a launch over
several models starts one concurrent run per model — never write into each other's. Within a run
cells are serial, and wiring a capture cell first clears whatever the corpus already holds for the
case, so the corpus holds exactly one session per case: the last cell of that case to run. Capture
at ``k=1`` for a corpus whose every case is the session you meant.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Literal, Protocol

from threetears.evals.contracts.cassettes import (
    ActionSeam,
    CassetteCorrupt,
    CassetteExhausted,
    CassetteMiss,
    CassetteSeams,
    DeliveryOutcome,
    DeliveryTicket,
    Recordable,
    ReplayedDelivery,
    ToolLike,
)
from threetears.evals.contracts.errors import StorageError
from threetears.evals.contracts.models import AsyncDelivery, CassetteKey, CassetteSeam, EvalCassette
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.models import EvalRun

log = get_logger(__name__)


#: The ``action`` component every delivery-seam record is stored under. Background work is not an
#: action, so its records share one engine-owned name rather than borrowing one of the tool's.
DELIVERY_ACTION = "delivery"


def params_hash(params: Mapping[str, Any]) -> str:
    """Render a params mapping into a deterministic 16-char hex hash.

    Part of every cassette key. Two equivalent mappings produce the same hash regardless of key
    insertion order at any nesting level; lists keep their order, because order in a list is usually
    meaning (a dice pool, a ranked shortlist).

    ``default=str`` renders datetimes, UUIDs and other non-JSON values a model may surface through
    tool params, rather than raising: a raise would fire only at capture time and silently leave a
    gap in the corpus.

    Args:
        params: The action parameters, or the background work's request.

    Returns:
        A 16-char hex prefix of the sha256 of the canonical JSON — 64 bits, ample within one case.
    """
    canonical = json.dumps(dict(params or {}), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _rebuild[R: Recordable](payload_type: type[R], recorded: dict[str, Any], *, key: CassetteKey) -> R:
    """Rebuild one recording as its seam's declared type, a misfit raised as the rig's fault.

    Raises:
        CassetteCorrupt: The recording does not validate as ``payload_type``.
    """
    try:
        return payload_type.model_validate(recorded)
    except ValueError as exc:
        raise CassetteCorrupt(
            f"Cassette recording {key.doc_id!r} does not rebuild as {payload_type.__name__}: {exc}. "
            "The corpus predates a change to that type; re-capture the template."
        ) from exc


# =============================================================================
# The store this layer reads and writes
# =============================================================================


class CassetteStore(Protocol):
    """The storage the cassette layer reads and writes.

    :class:`~threetears.evals.contracts.storage.EvalStorage` satisfies it in production.
    """

    def get_cassette(self, key: CassetteKey, scope_id: str) -> EvalCassette | None:
        """The recording of one key, or ``None``; ``StorageError`` or ``ValueError`` when it cannot be read."""
        ...  # pragma: no cover — protocol

    def save_cassette(self, cassette: EvalCassette) -> None:
        """Persist a recording; ``StorageError`` on failure."""
        ...  # pragma: no cover — protocol

    def delete_case_cassettes(self, *, corpus_id: str, template_id: str, test_case_id: str, scope_id: str) -> int:
        """Delete everything a corpus recorded for one case; ``StorageError`` on failure."""
        ...  # pragma: no cover — protocol


class ReplayReportDefect(ValueError):
    """A kind reported the background work a replay cell served as something it was not.

    Every piece of background work a replay serves is a harness's payload, so the kind's record of it
    must say ``substituted=True`` — and must exist: a replayed delivery missing from
    ``async_deliveries`` would let the cell's production cost be published as though nothing had been
    substituted. The engine knows what it served, so it holds the kind's report to it.
    """


# =============================================================================
# CassetteCell — one cell's lane
# =============================================================================


class CassetteCell:
    """One cell's handle on its run's cassette lane — what a kind's ``prepare`` is handed as ``cassettes``.

    Built by :meth:`CassetteLane.cell` for each cell, already bound to the corpus, the template, the
    case and the candidate model, so a kind names none of them. It counts each ask's occurrences for
    the whole cell, keeps the capture's open background work until the cell ends, and remembers what
    it replayed, so the runner can hold the kind's report to it.
    """

    def __init__(
        self,
        *,
        mode: Literal["capture", "replay"],
        store: CassetteStore,
        scope_id: str,
        corpus_id: str,
        template_id: str,
        test_case_id: str,
        model: str,
    ) -> None:
        """Bind the cell to its run's corpus and its own coordinates.

        Args:
            mode: Whether the cell records or replays.
            store: Where the corpus lives.
            scope_id: The scope the corpus lives in.
            corpus_id: The capture run whose corpus this is.
            template_id: The template the cell runs.
            test_case_id: The cell's case.
            model: The cell's candidate model — capture provenance.
        """
        self._mode: Literal["capture", "replay"] = mode
        self._store = store
        self._scope_id = scope_id
        self._corpus_id = corpus_id
        self._template_id = template_id
        self._test_case_id = test_case_id
        self._model = model
        self._occurrences: Counter[tuple[str, str, str]] = Counter()
        self._open: dict[CassetteKey, None] = {}
        self._served: Counter[str] = Counter()
        self._replayed_tools: frozenset[str] = frozenset()
        self._wired = False
        self._closed = False

    @property
    def mode(self) -> Literal["capture", "replay"]:
        """Whether this cell records its tools' answers or replays them."""
        return self._mode

    @property
    def wired(self) -> bool:
        """Whether the cell's kind has wired its candidate's seams."""
        return self._wired

    # -- keys and IO ------------------------------------------------------------------------------

    def _next_key(self, tool: str, action: str, params: Mapping[str, Any]) -> CassetteKey:
        """The key of this ask's next occurrence in the cell."""
        phash = params_hash(params)
        occurrence = self._occurrences[(tool, action, phash)]
        self._occurrences[(tool, action, phash)] += 1
        return CassetteKey(
            corpus_id=self._corpus_id,
            template_id=self._template_id,
            test_case_id=self._test_case_id,
            tool=tool,
            action=action,
            params_hash=phash,
            occurrence=occurrence,
        )

    def _read(self, key: CassetteKey, *, seam: CassetteSeam) -> EvalCassette:
        """The recording of ``key``, or the rig's failure to have one.

        Raises:
            CassetteMiss: The corpus never recorded this ask.
            CassetteExhausted: It recorded this ask ``key.occurrence`` times.
            CassetteCorrupt: The row could not be read, or was recorded at the other seam.
        """
        try:
            recorded = self._store.get_cassette(key, self._scope_id)
        except (StorageError, ValueError) as exc:
            raise CassetteCorrupt(f"Cassette recording {key.doc_id!r} could not be read back: {exc}") from exc
        if recorded is None:
            raise CassetteMiss(key) if key.occurrence == 0 else CassetteExhausted(key)
        if recorded.seam != seam:
            raise CassetteCorrupt(
                f"Cassette recording {key.doc_id!r} was recorded at the {recorded.seam} seam and is asked for at the "
                f"{seam} seam; re-capture the template."
            )
        return recorded

    def _write(self, cassette: EvalCassette) -> None:
        """Save one recording; a failure is logged and leaves a hole a replay reports by its key."""
        try:
            self._store.save_cassette(cassette)
        except StorageError as exc:
            # Loud but not fatal: the live answer still reaches the candidate, so a transient write
            # failure costs one recording rather than the turn. The key is explicit, so a replay
            # reports the hole as a miss naming it rather than pairing a neighbour into its place.
            log.error(
                "Cassette capture: failed to record %s (%s). This corpus is INCOMPLETE — a replay asking this "
                "will stop the cell; re-capture before replaying it.",
                cassette.id,
                exc,
            )

    # -- wiring -----------------------------------------------------------------------------------

    def wire(self, seams: CassetteSeams) -> None:
        """Arm every seam a cell's kind supplies.

        Every refusal of the declaration is made before anything is armed, so a refused declaration
        leaves the candidate as the kind built it. A capture clears whatever the corpus already holds
        for this case before arming, so the case's recording is this cell's session alone.

        Args:
            seams: The cell candidate's seams.

        Raises:
            TypeError: ``seams`` is not a :class:`~threetears.evals.contracts.cassettes.CassetteSeams`.
            ValueError: The cell was already wired; ``seams`` declares no seam at all; its action seam
                names no tool; a tool is declared on both seams; or the action seam did not apply the
                wrap exactly once over every tool it declared.
            CassetteCorrupt: A capture could not clear the case's previous recording.
        """
        if self._wired:
            raise ValueError(
                "this cell's cassettes are already wired; a kind wires its candidate's seams once per cell"
            )
        if not isinstance(seams, CassetteSeams):
            raise TypeError(
                f"{type(seams).__name__} is not a CassetteSeams: a cassette run wires a candidate only through "
                "the action_seam and delivery_seams its kind supplies."
            )
        action = seams.action_seam
        deliveries = dict(seams.delivery_seams)
        if action is None and not deliveries:
            raise ValueError(
                f"{type(seams).__name__} supplies no cassette seam: nothing of this candidate could be recorded "
                "or replayed, so a replay run would go live. Supply an action seam, a delivery seam, or launch "
                "with cassette_mode='off'."
            )
        recorded: dict[str, type[Recordable]] = {}
        if action is not None:
            recorded = dict(action.recorded_tools)
            if not recorded:
                raise ValueError(
                    "the action seam names no tool to record; a candidate with no synchronous tool to record "
                    "supplies action_seam=None."
                )
            if both := sorted(recorded.keys() & deliveries.keys()):
                raise ValueError(
                    f"{', '.join(repr(name) for name in both)} declared on both the action seam and a delivery "
                    "seam. An asynchronous tool's act() returns an acknowledgement, so recording it as an action "
                    "would replay the acknowledgement and never the delivery; declare it on the delivery seam only."
                )
        if self._mode == "capture":
            try:
                self._store.delete_case_cassettes(
                    corpus_id=self._corpus_id,
                    template_id=self._template_id,
                    test_case_id=self._test_case_id,
                    scope_id=self._scope_id,
                )
            except StorageError as exc:
                raise CassetteCorrupt(
                    f"Cassette capture could not clear corpus {self._corpus_id!r}'s previous recording of case "
                    f"{self._test_case_id!r}, so this cell's session would be mixed into it: {exc}"
                ) from exc
        self._wired = True
        if action is not None:
            self._wire_action_seam(action, recorded)
        for tool, seam in deliveries.items():
            if self._mode == "capture":
                seam.arm_capture(_Recorder(partial(self._start, tool)))
            else:
                seam.arm_replay(_Replay(partial(self._replay, tool, seam.payload_type)))
        if self._mode == "replay":
            self._replayed_tools = frozenset(deliveries)

    def _wire_action_seam(self, seam: ActionSeam, recorded: Mapping[str, type[Recordable]]) -> None:
        """Have the action seam swap each recorded tool for a cassette proxy, and hold it to doing so."""
        applied: list[frozenset[str]] = []

        def wrap(tools: Mapping[str, ToolLike]) -> dict[str, ToolLike]:
            if missing := sorted(recorded.keys() - tools.keys()):
                raise ValueError(
                    f"the action seam declares {', '.join(repr(name) for name in missing)} as recorded, but the "
                    f"candidate's tools are {sorted(tools)}; an unwrapped tool would run live under replay."
                )
            applied.append(frozenset(tools))
            return {
                name: CassetteProxy(tool, act=partial(self._act, tool, recorded[name])) if name in recorded else tool
                for name, tool in tools.items()
            }

        seam.arm_tools(wrap)
        if len(applied) != 1:
            raise ValueError(
                f"the action seam applied the cassette wrap {len(applied)} times; it must apply it exactly once, "
                "or its recorded tools run unwrapped (live, under replay) or wrapped twice."
            )

    # -- what the seams do ------------------------------------------------------------------------

    async def _act(
        self, tool: ToolLike, result_type: type[Recordable], action: str, parameters: dict[str, Any]
    ) -> Recordable:
        """Capture or replay one synchronous call — a :class:`CassetteProxy`'s ``act``."""
        key = self._next_key(tool.name, action, parameters)
        if self._mode == "replay":
            recorded = self._read(key, seam="action")
            return _rebuild(result_type, recorded.response or {}, key=key)
        result = await tool.act(action, parameters)
        self._write(
            EvalCassette.build(
                key,
                scope_id=self._scope_id,
                seam="action",
                captured_model=self._model,
                response=result.model_dump(mode="json"),
            )
        )
        return result

    def _start(self, tool: str, request: Mapping[str, Any]) -> DeliveryTicket:
        """Open the capture of one piece of background work — a recorder's ``started``."""
        key = self._next_key(tool, DELIVERY_ACTION, request)
        self._open[key] = None
        return _Ticket(partial(self._settle, key))

    def _settle(
        self, key: CassetteKey, outcome: DeliveryOutcome, payload: Recordable | None, error: str | None
    ) -> None:
        """Record how one captured piece of background work ended — a ticket's settlement.

        Raises:
            ValueError: The work was already settled.
        """
        if self._closed:
            # The cell has ended and already recorded this work as undelivered, which is what its
            # session saw; a payload arriving after the cell is not part of the session.
            log.warning(
                "Cassette capture: %s settled after its cell ended; recorded as undelivered, as the session saw it",
                key.doc_id,
            )
            return
        if key not in self._open:
            raise ValueError(f"background work {key.doc_id!r} was already settled; a ticket is settled once")
        del self._open[key]
        response = payload.model_dump(mode="json") if payload is not None else None
        self._write(self._delivery(key, outcome, response=response, error=error))

    def _delivery(
        self,
        key: CassetteKey,
        outcome: DeliveryOutcome,
        *,
        response: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> EvalCassette:
        """The recording of one piece of background work."""
        return EvalCassette.build(
            key,
            scope_id=self._scope_id,
            seam="delivery",
            captured_model=self._model,
            outcome=outcome,
            response=response,
            error=error,
        )

    def _replay[P: Recordable](
        self, tool: str, payload_type: type[P], request: Mapping[str, Any]
    ) -> ReplayedDelivery[P]:
        """Serve one piece of background work from the corpus — a replay's ``next``."""
        key = self._next_key(tool, DELIVERY_ACTION, request)
        recorded = self._read(key, seam="delivery")
        self._served[tool] += 1
        if recorded.outcome == "delivered":
            return ReplayedDelivery(
                outcome="delivered", payload=_rebuild(payload_type, recorded.response or {}, key=key)
            )
        if recorded.outcome == "failed":
            return ReplayedDelivery(outcome="failed", error=recorded.error)
        return ReplayedDelivery(outcome="undelivered")

    # -- the cell's end ---------------------------------------------------------------------------

    def close(self) -> None:
        """End the cell: record the capture's background work still in flight as undelivered.

        Called by the runner once the kind's ``invoke`` has returned or been cancelled. Idempotent.
        """
        if self._closed:
            return
        self._closed = True
        for key in self._open:
            self._write(self._delivery(key, "undelivered"))
        self._open.clear()

    def hold(self, deliveries: Sequence[AsyncDelivery] | None, *, complete: bool) -> None:
        """Hold the kind's report of its background work to what this cell replayed.

        Every entry for a tool this cell replayed must be substituted — the tool could not run live
        here — and, on a ``complete`` report, there must be at least one such entry per piece of work
        the cell served. A reading taken when a deadline cut the cell off is held to the first rule
        only: it may legitimately not have caught up.

        Args:
            deliveries: The kind's ``async_deliveries``.
            complete: Whether this is the kind's returned output rather than a mid-cell reading.

        Raises:
            ReplayReportDefect: The report says a replayed piece of work ran live, or leaves one out.
        """
        if not self._replayed_tools:
            return
        entries = list(deliveries or [])
        if live := sorted({e.tool for e in entries if e.tool in self._replayed_tools and not e.substituted}):
            raise ReplayReportDefect(
                f"async_deliveries reports work by {', '.join(map(repr, live))} as live (substituted=False), but "
                "this replay cell armed that tool to replay and it ran no live work; report a replayed "
                "delivery with substituted=True."
            )
        if not complete:
            return
        reported = Counter(e.tool for e in entries if e.tool in self._replayed_tools)
        if short := sorted(tool for tool, served in self._served.items() if reported[tool] < served):
            raise ReplayReportDefect(
                "async_deliveries leaves out replayed work: "
                + ", ".join(f"{tool!r} served {self._served[tool]}, reported {reported[tool]}" for tool in short)
                + ". Report every piece of background work the candidate started, replayed ones included."
            )


# =============================================================================
# What the seams are handed — thin, so every rule lives on the cell
# =============================================================================


class CassetteProxy:
    """A recorded synchronous tool, as the candidate sees it once its cell is wired.

    Composition and ``__getattr__`` delegation: every attribute except the three
    :class:`~threetears.evals.contracts.cassettes.ToolLike` members passes through to the wrapped
    tool, so the candidate's own tool machinery interacts with the proxy exactly as with the tool.
    ``act`` goes to the cell: in capture the tool runs live and its result is recorded under the
    call's key and occurrence (a failed result too — a failure pattern is itself replayable signal);
    in replay the recording of this call's occurrence is served and the tool is never called.
    """

    def __init__(self, wrapped: ToolLike, *, act: Callable[[str, dict[str, Any]], Awaitable[Recordable]]) -> None:
        """Wrap one tool for one cell.

        Args:
            wrapped: The tool.
            act: The cell's capture-or-replay of one call to it.
        """
        self._wrapped = wrapped
        self._act = act

    @property
    def name(self) -> str:
        """The wrapped tool's name — the candidate-facing identity is unchanged."""
        return self._wrapped.name

    def can_dispatch(self, action: str) -> bool:
        """Whether the wrapped tool dispatches ``action`` — forwarded verbatim.

        Declared rather than left to :meth:`__getattr__`: a port member reachable only through
        ``__getattr__`` is invisible to a type checker and to a runtime-checkable ``isinstance``.
        """
        return self._wrapped.can_dispatch(action)

    def __getattr__(self, item: str) -> Any:
        """Delegate every attribute the proxy does not define to the wrapped tool."""
        return getattr(self._wrapped, item)

    async def act(self, action: str, parameters: dict[str, Any]) -> Recordable:
        """Capture or replay this call through the cell's corpus.

        **An action the wrapped tool cannot dispatch never reaches the corpus.** It is not a
        recordable event: a model inventing an action name is ordinary candidate behaviour, and the
        answer to it is the tool's own unknown-action answer — the same one a run without cassettes
        gives.

        Raises:
            CassetteMiss: Replay, and the corpus never recorded this call.
            CassetteExhausted: Replay, and the corpus recorded this call fewer times than it is now made.
            CassetteCorrupt: Replay, and the recording could not be read back.
        """
        if not self._wrapped.can_dispatch(action):
            return await self._wrapped.act(action, parameters)
        return await self._act(action, parameters)


class _Ticket:
    """One captured piece of background work, handed back by ``started`` to be settled once."""

    def __init__(self, settle: Callable[[DeliveryOutcome, Recordable | None, str | None], None]) -> None:
        self._settle = settle

    def delivered(self, payload: Recordable) -> None:
        """Record that the work reached the candidate with ``payload``."""
        self._settle("delivered", payload, None)

    def failed(self, error: str) -> None:
        """Record that the work ended in an error and delivered nothing."""
        self._settle("failed", None, error)


class _Recorder:
    """One asynchronous tool's capture, for one cell."""

    def __init__(self, start: Callable[[Mapping[str, Any]], DeliveryTicket]) -> None:
        self._start = start

    def started(self, request: Mapping[str, Any]) -> DeliveryTicket:
        """Begin recording one piece of background work, against the request that started it."""
        return self._start(request)


class _Replay[P: Recordable]:
    """One asynchronous tool's replay, for one cell."""

    def __init__(self, serve: Callable[[Mapping[str, Any]], ReplayedDelivery[P]]) -> None:
        self._serve = serve

    def next(self, request: Mapping[str, Any]) -> ReplayedDelivery[P]:
        """Take the recording of this request's next occurrence, in place of starting live work."""
        return self._serve(request)


# =============================================================================
# CassetteLane — the per-run object the runner builds and splits into cells
# =============================================================================


@dataclass(frozen=True)
class CassetteLane:
    """One run's cassette lane: its mode, its store, and the corpus it records into or replays.

    Built once per run by the runner (:meth:`for_run`) and split into one :class:`CassetteCell` per
    cell (:meth:`cell`), which the runner hands that cell's ``prepare``. One lane per run is what
    binds a run's whole matrix to one corpus.

    Attributes:
        mode: ``'capture'`` or ``'replay'``; ``'off'`` is refused, since a run with cassettes off has
            no lane.
        store: Where cassettes are read and written.
        scope_id: The scope every record this lane touches lives in.
        corpus_id: The corpus: a capture run's own id, or the capture run a replay names.
    """

    mode: Literal["capture", "replay"]
    store: CassetteStore
    scope_id: str
    corpus_id: str

    def __post_init__(self) -> None:
        """Refuse a lane in a mode that records and replays nothing.

        Raises:
            ValueError: ``mode`` is not ``'capture'`` or ``'replay'``.
        """
        if self.mode not in ("capture", "replay"):
            raise ValueError(
                f"CassetteLane requires mode 'capture' or 'replay', got {self.mode!r}; "
                "a run with cassettes off has no lane (CassetteLane.for_run returns None)."
            )

    @classmethod
    def for_run(cls, run: EvalRun, store: CassetteStore) -> CassetteLane | None:
        """The lane a run's own cassette settings call for, or ``None`` when its cassettes are off.

        Args:
            run: The run: its ``cassette_mode``, its ``scope_id``, and the corpus — its own ``id`` for a
                capture, ``cassette_corpus_id`` for a replay.
            store: Where its cassettes are read and written.

        Returns:
            The lane, or ``None`` for ``cassette_mode='off'``.
        """
        if run.cassette_mode == "off":
            return None
        if run.cassette_mode == "capture":
            return cls(mode="capture", store=store, scope_id=run.scope_id, corpus_id=run.id)
        # EvalRun refuses a replay that names no corpus, so this is the replay's own.
        assert run.cassette_corpus_id is not None
        return cls(mode="replay", store=store, scope_id=run.scope_id, corpus_id=run.cassette_corpus_id)

    def cell(self, *, template_id: str, test_case_id: str, model: str) -> CassetteCell:
        """The handle one cell of this run is given.

        Args:
            template_id: The template the cell runs.
            test_case_id: The cell's case.
            model: The cell's candidate model.

        Returns:
            The cell's handle, unwired.
        """
        return CassetteCell(
            mode=self.mode,
            store=self.store,
            scope_id=self.scope_id,
            corpus_id=self.corpus_id,
            template_id=template_id,
            test_case_id=test_case_id,
            model=model,
        )


__all__ = [
    "DELIVERY_ACTION",
    "CassetteCell",
    "CassetteLane",
    "CassetteProxy",
    "CassetteStore",
    "ReplayReportDefect",
    "params_hash",
]
