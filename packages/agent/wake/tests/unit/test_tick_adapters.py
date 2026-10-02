"""Unit tests for the S-2 wake->scheduled-jobs adapter layer in
:mod:`threetears.agent.wake.tick`.

Since S-2 ``wake_tick_job`` delegates the tick pump to the generic
:func:`threetears.scheduled_jobs.scheduled_tick_job`; this module is the thin
adapter that bridges wake's conversation-native collections + trigger shape to
the generic engine. The end-to-end behavior is proven against real Postgres in
``tests/integration/test_wake_tick_loop.py``; these unit tests pin the DB-free
seams that an integration failure would not localize. Each one runs
``wake_tick_job`` against a stand-in engine that captures what wake hands it --
the schedule store, the fire store, the dispatch route and the config -- and
drives those through the generic surface the real engine uses:

- the payload round-trip (``WakeTrigger`` -> opaque payload -> ``WakeTrigger``),
- the result packing (``WakeDispatchResult`` -> ``JobFireResult.output`` dict),
- the fire store's ``finalize_success`` unpack into wake's typed
  ``output_text`` / ``display_suppressed`` columns (the silent path has no
  integration coverage),
- the parameter-name translation (``partition_key`` <-> ``conversation_id``,
  ``job_id`` <-> ``schedule_id``),
- structural conformance of the adapters to the core Protocols, and
- ``wake_tick_job``'s wiring: the preserved ``"agent_wake_tick"`` lock key, the
  ``nats_client`` pass-through (so the engine's degrade-open actually protects
  wake -- the prod-incident contract), and the yield-duration re-emit.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from uuid_utils import uuid7

from threetears.scheduled_jobs import (
    DueSchedule,
    FireStore,
    JobFireResult,
    JobTrigger,
    ScheduleStore,
)

from threetears.agent.wake import tick as tick_mod
from threetears.agent.wake.collections import WakeFireCollection, WakeScheduleCollection
from threetears.agent.wake.entities import WakeScheduleEntity
from threetears.agent.wake.tick import wake_tick_job
from threetears.agent.wake.types import WakeDispatchResult, WakeTrigger


def _new_uuid() -> UUID:
    return UUID(str(uuid7()))


def _make_schedule_entity(**overrides: Any) -> WakeScheduleEntity:
    """Build a fully-populated WakeScheduleEntity for mapping assertions."""
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    data: dict[str, Any] = {
        "conversation_id": _new_uuid(),
        "schedule_id": _new_uuid(),
        "user_id": _new_uuid(),
        "agent_id": _new_uuid(),
        "skill_id": _new_uuid(),
        "schedule_type": "interval",
        "schedule_config": {"seconds": 60},
        "task_prompt": "check the build",
        "execution_mode": "spawn",
        "status": "active",
        "next_fire_at": now - timedelta(seconds=5),
        "last_fired_at": now - timedelta(minutes=1),
        "name": "nightly check",
        "missed_fire_policy": "catch_up",
        "context_from_schedule_id": _new_uuid(),
        "include_conversation_history": False,
        "date_created": now,
        "date_updated": now,
    }
    data.update(overrides)
    return WakeScheduleEntity(data, is_new=False)


def _empty_schedule_collection(nats_client: Any = None) -> WakeScheduleCollection:
    """A WakeScheduleCollection with no pool (methods are overridden / unused).

    ``nats_client`` is the client its evictions broadcast through: a tick given one refuses a
    schedule collection without one.
    """
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig

    registry = CollectionRegistry()
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return WakeScheduleCollection(registry=registry, config=cfg, nats_client=nats_client)


def _empty_fire_collection() -> WakeFireCollection:
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig

    registry = CollectionRegistry()
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return WakeFireCollection(registry=registry, config=cfg)


class _RecordingScheduleCollection(WakeScheduleCollection):
    """Records the kwargs the adapter forwards; returns canned due rows.

    Subclasses the production collection (fake-parity satisfied by subclass
    declaration) and overrides only the two methods the adapter calls.
    """

    def __init__(self, due: list[WakeScheduleEntity] | None = None) -> None:
        super().__init__(registry=_recording_registry(), config=_recording_config())
        self._due = due or []
        self.claim_calls: list[dict[str, Any]] = []
        self.list_due_calls: list[dict[str, Any]] = []

    async def list_due_for_tick(self, now: datetime, *, limit: int = 200) -> list[WakeScheduleEntity]:
        self.list_due_calls.append({"now": now, "limit": limit})
        return list(self._due)

    async def claim_and_reschedule(
        self,
        *,
        conversation_id: UUID,
        schedule_id: UUID,
        expected_next_fire: datetime,
        computed_next_fire: datetime | None,
        new_status: str,
        now: datetime,
    ) -> bool:
        self.claim_calls.append(
            {
                "conversation_id": conversation_id,
                "schedule_id": schedule_id,
                "expected_next_fire": expected_next_fire,
                "computed_next_fire": computed_next_fire,
                "new_status": new_status,
                "now": now,
            }
        )
        return True


class _RecordingFireCollection(WakeFireCollection):
    """Records the kwargs the adapter forwards to the fire collection."""

    def __init__(self) -> None:
        super().__init__(registry=_recording_registry(), config=_recording_config())
        self.create_calls: list[dict[str, Any]] = []
        self.success_calls: list[dict[str, Any]] = []
        self.failed_calls: list[dict[str, Any]] = []

    async def create_dispatching(
        self,
        *,
        fire_id: UUID,
        schedule_id: UUID | None,
        webhook_subscription_id: UUID | None,
        conversation_id: UUID,
        scheduled_fire_at: datetime | None,
        actual_fired_at: datetime,
        fire_source: str,
        execution_mode: str,
    ) -> None:
        self.create_calls.append(
            {
                "fire_id": fire_id,
                "schedule_id": schedule_id,
                "webhook_subscription_id": webhook_subscription_id,
                "conversation_id": conversation_id,
                "scheduled_fire_at": scheduled_fire_at,
                "actual_fired_at": actual_fired_at,
                "fire_source": fire_source,
                "execution_mode": execution_mode,
            }
        )

    async def finalize_success(
        self,
        conversation_id: UUID,
        fire_id: UUID,
        *,
        status: str = "fired",
        output_text: str | None = None,
        latency_ms: int | None = None,
        display_suppressed: bool = False,
    ) -> None:
        self.success_calls.append(
            {
                "conversation_id": conversation_id,
                "fire_id": fire_id,
                "status": status,
                "output_text": output_text,
                "latency_ms": latency_ms,
                "display_suppressed": display_suppressed,
            }
        )

    async def finalize_failed(
        self,
        conversation_id: UUID,
        fire_id: UUID,
        *,
        error: str,
        latency_ms: int | None = None,
    ) -> None:
        self.failed_calls.append(
            {"conversation_id": conversation_id, "fire_id": fire_id, "error": error, "latency_ms": latency_ms}
        )


def _recording_registry() -> Any:
    from threetears.core.collections.registry import CollectionRegistry

    return CollectionRegistry()


def _recording_config() -> Any:
    from threetears.core.config import DefaultCoreConfig

    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


class _Wiring:
    """what ``wake_tick_job`` handed the scheduled-jobs engine."""

    def __init__(self, captured: dict[str, Any]) -> None:
        self.schedule_store: ScheduleStore = captured["schedule_store"]
        self.fire_store: FireStore = captured["fire_store"]
        self.dispatch_routes: dict[str, Any] = captured["dispatch_routes"]
        self.nats_client: Any = captured["nats_client"]
        self.config: Any = captured["config"]

    async def route(self, job_trigger: JobTrigger, fire_id: UUID) -> JobFireResult:
        """drive wake's one dispatch route as the engine does."""
        result: JobFireResult = await self.dispatch_routes["agent_wake"](job_trigger, fire_id)
        return result


async def _wired(
    monkeypatch: pytest.MonkeyPatch,
    *,
    schedules: WakeScheduleCollection | None = None,
    fires: WakeFireCollection | None = None,
    nats_client: Any = None,
    pool: Any = None,
    callback: Any = None,
) -> _Wiring:
    """run ``wake_tick_job`` against an engine that captures its arguments instead of ticking.

    :param monkeypatch: pytest's monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :param schedules: the schedule collection; an empty one by default
    :ptype schedules: WakeScheduleCollection | None
    :param fires: the fire collection; an empty one by default
    :ptype fires: WakeFireCollection | None
    :param nats_client: the NATS client passed through to the engine
    :ptype nats_client: Any
    :param pool: the pool the wake callback is handed
    :ptype pool: Any
    :param callback: the wake dispatch callback; answers ``fired`` by default
    :ptype callback: Any
    :return: what the engine was handed
    :rtype: _Wiring
    """
    captured: dict[str, Any] = {}

    async def _capturing_engine(
        schedule_store: Any,
        fire_store: Any,
        dispatch_routes: Any,
        *,
        nats_client: Any = None,
        config: Any = None,
    ) -> None:
        captured.update(
            schedule_store=schedule_store,
            fire_store=fire_store,
            dispatch_routes=dispatch_routes,
            nats_client=nats_client,
            config=config,
        )

    async def _fired(_t: WakeTrigger, _f: UUID, _p: Any) -> WakeDispatchResult:
        return WakeDispatchResult(status="fired")

    monkeypatch.setattr(tick_mod, "scheduled_tick_job", _capturing_engine)
    await wake_tick_job(
        pool if pool is not None else object(),
        nats_client,
        callback if callback is not None else _fired,
        schedules=schedules if schedules is not None else _empty_schedule_collection(nats_client),
        fires=fires if fires is not None else _empty_fire_collection(),
    )
    return _Wiring(captured)


async def _due_rows(monkeypatch: pytest.MonkeyPatch, *entities: WakeScheduleEntity) -> list[DueSchedule]:
    """the due rows the engine reads for ``entities``, through wake's schedule store."""
    wiring = await _wired(monkeypatch, schedules=_RecordingScheduleCollection(due=list(entities)))
    return await wiring.schedule_store.list_due_for_tick(
        datetime(2026, 6, 1, 12, 0, tzinfo=UTC), kinds=("agent_wake",), limit=50
    )


def _job_trigger(due: DueSchedule, *, fired_at: datetime, scheduled_fire_at: datetime) -> JobTrigger:
    """the generic envelope the engine builds for one due row."""
    return JobTrigger(
        job_id=due.job_id,
        partition_key=due.partition_key,
        kind=due.kind,
        schedule_type=due.schedule_type,
        fired_at=fired_at,
        scheduled_fire_at=scheduled_fire_at,
        payload=due.payload,
        name=due.name,
    )


async def _rebuilt_trigger(
    monkeypatch: pytest.MonkeyPatch, entity: WakeScheduleEntity, fired_at: datetime
) -> WakeTrigger:
    """the wake trigger the dispatch callback receives for ``entity``'s due row."""
    [due] = await _due_rows(monkeypatch, entity)
    seen: dict[str, Any] = {}

    async def _cb(trigger: WakeTrigger, _f: UUID, _p: Any) -> WakeDispatchResult:
        seen["trigger"] = trigger
        return WakeDispatchResult(status="fired")

    wiring = await _wired(monkeypatch, callback=_cb)
    await wiring.route(
        _job_trigger(due, fired_at=fired_at, scheduled_fire_at=entity.next_fire_at or fired_at), _new_uuid()
    )
    trigger: WakeTrigger = seen["trigger"]
    return trigger


async def _packed(monkeypatch: pytest.MonkeyPatch, result: WakeDispatchResult) -> JobFireResult:
    """the generic result the engine receives when the wake callback answers ``result``."""
    [due] = await _due_rows(monkeypatch, _make_schedule_entity())

    async def _cb(_t: WakeTrigger, _f: UUID, _p: Any) -> WakeDispatchResult:
        return result

    wiring = await _wired(monkeypatch, callback=_cb)
    moment = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    return await wiring.route(_job_trigger(due, fired_at=moment, scheduled_fire_at=moment), _new_uuid())


class TestWakeDueScheduleMapping:
    """A due schedule exposes the entity through the generic surface."""

    async def test_maps_identity_and_scheduling_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entity = _make_schedule_entity()
        [due] = await _due_rows(monkeypatch, entity)
        assert due.partition_key == entity.conversation_id
        assert due.job_id == entity.schedule_id
        assert due.kind == "agent_wake"
        assert due.schedule_type == entity.schedule_type
        assert due.schedule_config == entity.schedule_config
        assert due.missed_fire_policy == entity.missed_fire_policy
        assert due.next_fire_at == entity.next_fire_at
        assert due.last_fired_at == entity.last_fired_at
        assert due.name == entity.name

    async def test_packs_agent_fields_into_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entity = _make_schedule_entity()
        [due] = await _due_rows(monkeypatch, entity)
        payload = due.payload
        assert payload["user_id"] == entity.user_id
        assert payload["agent_id"] == entity.agent_id
        assert payload["skill_id"] == entity.skill_id
        assert payload["execution_mode"] == entity.execution_mode
        assert payload["task_prompt"] == entity.task_prompt
        assert payload["context_from_schedule_id"] == entity.context_from_schedule_id
        assert payload["include_conversation_history"] is False

    async def test_conforms_to_due_schedule_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        [due] = await _due_rows(monkeypatch, _make_schedule_entity())
        assert isinstance(due, DueSchedule)


class TestRebuildWakeTrigger:
    """The dispatch route reconstructs the wake trigger from the generic
    envelope -- the inverse of the due row's payload."""

    async def test_round_trip_through_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entity = _make_schedule_entity()
        fired_at = datetime(2026, 6, 1, 12, 0, 5, tzinfo=UTC)

        trigger = await _rebuilt_trigger(monkeypatch, entity, fired_at)

        assert trigger.schedule_id == entity.schedule_id
        assert trigger.conversation_id == entity.conversation_id
        assert trigger.user_id == entity.user_id
        assert trigger.agent_id == entity.agent_id
        assert trigger.skill_id == entity.skill_id
        assert trigger.execution_mode == entity.execution_mode
        assert trigger.task_prompt == entity.task_prompt
        assert trigger.context_from_schedule_id == entity.context_from_schedule_id
        assert trigger.include_conversation_history is False
        assert trigger.schedule_type == entity.schedule_type
        assert trigger.fired_at == fired_at
        assert trigger.schedule_name == entity.name
        assert trigger.fire_source == "scheduled_tick"

    async def test_handles_absent_optional_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entity = _make_schedule_entity(
            skill_id=None,
            task_prompt=None,
            context_from_schedule_id=None,
            name=None,
            include_conversation_history=True,
        )
        trigger = await _rebuilt_trigger(monkeypatch, entity, datetime(2026, 6, 1, 12, 0, 5, tzinfo=UTC))
        assert trigger.skill_id is None
        assert trigger.task_prompt is None
        assert trigger.context_from_schedule_id is None
        assert trigger.schedule_name is None
        assert trigger.include_conversation_history is True


class TestToJobFireResult:
    """The dispatch route packs wake's typed result into the opaque
    generic result + output dict."""

    async def test_packs_output_and_passes_through_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = await _packed(
            monkeypatch,
            WakeDispatchResult(status="fired", output_text="hello", latency_ms=42, display_suppressed=False),
        )
        assert result.status == "fired"
        assert result.latency_ms == 42
        assert result.output == {"output_text": "hello", "display_suppressed": False}
        assert result.error is None

    async def test_carries_silent_flag_and_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = await _packed(
            monkeypatch,
            WakeDispatchResult(status="fired_silent", output_text="[SILENT] hi", display_suppressed=True),
        )
        assert result.output == {"output_text": "[SILENT] hi", "display_suppressed": True}

        failed = await _packed(monkeypatch, WakeDispatchResult(status="failed", error="boom", latency_ms=3))
        assert failed.status == "failed"
        assert failed.error == "boom"


class TestProtocolConformance:
    """The stores wake hands the engine satisfy the core Protocols structurally."""

    async def test_schedule_store_conforms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert isinstance((await _wired(monkeypatch)).schedule_store, ScheduleStore)

    async def test_fire_store_conforms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        assert isinstance((await _wired(monkeypatch)).fire_store, FireStore)


class TestScheduleStoreTranslation:
    """The schedule store translates the generic param names."""

    async def test_list_due_wraps_entities(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entities = [_make_schedule_entity(), _make_schedule_entity()]
        coll = _RecordingScheduleCollection(due=entities)
        store = (await _wired(monkeypatch, schedules=coll)).schedule_store
        now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
        rows = await store.list_due_for_tick(now, kinds=("agent_wake",), limit=50)
        assert len(rows) == 2
        assert all(isinstance(r, DueSchedule) for r in rows)
        assert rows[0].job_id == entities[0].schedule_id
        assert coll.list_due_calls == [{"now": now, "limit": 50}]

    async def test_claim_maps_partition_and_job_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        coll = _RecordingScheduleCollection()
        store = (await _wired(monkeypatch, schedules=coll)).schedule_store
        pk = _new_uuid()
        job = _new_uuid()
        expected = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
        computed = datetime(2026, 6, 1, 12, 1, tzinfo=UTC)
        now = datetime(2026, 6, 1, 12, 0, 5, tzinfo=UTC)
        ok = await store.claim_and_reschedule(
            partition_key=pk,
            job_id=job,
            expected_next_fire=expected,
            computed_next_fire=computed,
            new_status="active",
            now=now,
        )
        assert ok is True
        assert coll.claim_calls == [
            {
                "conversation_id": pk,
                "schedule_id": job,
                "expected_next_fire": expected,
                "computed_next_fire": computed,
                "new_status": "active",
                "now": now,
            }
        ]


class TestFireStoreTranslation:
    """The fire store maps create + unpacks the opaque output dict."""

    async def test_create_dispatching_maps_ids_and_scheduled_source(self, monkeypatch: pytest.MonkeyPatch) -> None:
        coll = _RecordingFireCollection()
        store = (await _wired(monkeypatch, fires=coll)).fire_store
        fire_id = _new_uuid()
        job = _new_uuid()
        pk = _new_uuid()
        scheduled = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
        fired = datetime(2026, 6, 1, 12, 0, 5, tzinfo=UTC)
        await store.create_dispatching(
            fire_id=fire_id,
            job_id=job,
            partition_key=pk,
            scheduled_fire_at=scheduled,
            actual_fired_at=fired,
        )
        assert len(coll.create_calls) == 1
        call = coll.create_calls[0]
        assert call["schedule_id"] == job
        assert call["conversation_id"] == pk
        assert call["webhook_subscription_id"] is None
        assert call["fire_source"] == "scheduled_tick"
        assert call["scheduled_fire_at"] == scheduled
        assert call["actual_fired_at"] == fired

    async def test_finalize_success_unpacks_output_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        coll = _RecordingFireCollection()
        store = (await _wired(monkeypatch, fires=coll)).fire_store
        pk = _new_uuid()
        fire_id = _new_uuid()
        await store.finalize_success(
            pk,
            fire_id,
            status="fired_silent",
            output={"output_text": "[SILENT] done", "display_suppressed": True},
            latency_ms=7,
        )
        assert coll.success_calls == [
            {
                "conversation_id": pk,
                "fire_id": fire_id,
                "status": "fired_silent",
                "output_text": "[SILENT] done",
                "latency_ms": 7,
                "display_suppressed": True,
            }
        ]

    async def test_finalize_success_tolerates_missing_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        coll = _RecordingFireCollection()
        store = (await _wired(monkeypatch, fires=coll)).fire_store
        await store.finalize_success(_new_uuid(), _new_uuid(), status="yielded", output=None)
        call = coll.success_calls[0]
        assert call["output_text"] is None
        assert call["display_suppressed"] is False

    async def test_finalize_failed_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        coll = _RecordingFireCollection()
        store = (await _wired(monkeypatch, fires=coll)).fire_store
        pk = _new_uuid()
        fire_id = _new_uuid()
        await store.finalize_failed(pk, fire_id, error="kaboom", latency_ms=11)
        assert coll.failed_calls == [{"conversation_id": pk, "fire_id": fire_id, "error": "kaboom", "latency_ms": 11}]


class TestWakeTickJobWiring:
    """``wake_tick_job`` delegates to the core engine with wake's lock key +
    the ``nats_client`` passed through, and the adapter callback bridges the
    consumer's wake-shaped callback."""

    async def test_delegates_with_wake_lock_key_and_nats_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        nats = object()

        wiring = await _wired(monkeypatch, nats_client=nats)

        assert isinstance(wiring.schedule_store, ScheduleStore)
        assert isinstance(wiring.fire_store, FireStore)
        assert wiring.nats_client is nats
        # the lock key predates the generic engine; every running replica contends on this exact
        # string, so a rename would let an old and a new replica tick at once during a deploy.
        assert wiring.config.tick_lock_key == "agent_wake_tick"
        # wake registers exactly one kind; a second entry would mean the pump
        # had silently taken ownership of rows it does not understand.
        assert list(wiring.dispatch_routes) == ["agent_wake"]

    async def test_adapter_callback_bridges_trigger_and_packs_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pool = object()
        seen: dict[str, Any] = {}

        async def _cb(trigger: WakeTrigger, fire_id: UUID, p: Any) -> WakeDispatchResult:
            seen["trigger"] = trigger
            seen["fire_id"] = fire_id
            seen["pool"] = p
            return WakeDispatchResult(status="fired", output_text="ok", latency_ms=9)

        entity = _make_schedule_entity()
        [due] = await _due_rows(monkeypatch, entity)
        wiring = await _wired(monkeypatch, pool=pool, callback=_cb)

        # drive the captured adapter callback with a generic envelope
        fire_id = _new_uuid()
        result = await wiring.route(
            _job_trigger(
                due,
                fired_at=datetime(2026, 6, 1, 12, 0, 5, tzinfo=UTC),
                scheduled_fire_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
            ),
            fire_id,
        )

        # the wake-shaped callback saw a faithfully rebuilt trigger + the pool
        assert seen["fire_id"] == fire_id
        assert seen["pool"] is pool
        assert seen["trigger"].schedule_id == entity.schedule_id
        assert seen["trigger"].conversation_id == entity.conversation_id
        # the result was packed back into the generic shape
        assert isinstance(result, JobFireResult)
        assert result.status == "fired"
        assert result.output == {"output_text": "ok", "display_suppressed": False}
        assert result.latency_ms == 9

    async def test_yielded_fire_reemits_yield_duration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        observed: list[float] = []

        class _RecordingEmitter:
            def observe_yield_duration(self, seconds: float) -> None:
                observed.append(seconds)

        monkeypatch.setattr(tick_mod, "get_wake_emitter", lambda: _RecordingEmitter())

        async def _cb(_t: WakeTrigger, _f: UUID, _p: Any) -> WakeDispatchResult:
            return WakeDispatchResult(status="yielded", latency_ms=2000)

        [due] = await _due_rows(monkeypatch, _make_schedule_entity())
        wiring = await _wired(monkeypatch, callback=_cb)
        await wiring.route(
            _job_trigger(
                due,
                fired_at=datetime(2026, 6, 1, 12, 0, 5, tzinfo=UTC),
                scheduled_fire_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
            ),
            _new_uuid(),
        )

        assert observed == [2.0]
