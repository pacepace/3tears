"""Unit tests for the S-2 wake->scheduled-jobs adapter layer in
:mod:`threetears.agent.wake.tick`.

Since S-2 ``wake_tick_job`` delegates the tick pump to the generic
:func:`threetears.scheduled_jobs.scheduled_tick_job`; this module is the thin
adapter that bridges wake's conversation-native collections + trigger shape to
the generic engine. The end-to-end behavior is proven against real Postgres in
``tests/integration/test_wake_tick_loop.py``; these unit tests pin the DB-free
seams that an integration failure would not localize:

- the payload round-trip (``WakeTrigger`` -> opaque payload -> ``WakeTrigger``),
- the result packing (``WakeDispatchResult`` -> ``JobFireResult.output`` dict),
- the ``_WakeFireStore.finalize_success`` unpack into wake's typed
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
from threetears.agent.wake.tick import (
    _WAKE_TICK_LOCK_KEY,
    _rebuild_wake_trigger,
    _to_job_fire_result,
    _WakeDueSchedule,
    _WakeFireStore,
    _WakeScheduleStore,
    wake_tick_job,
)
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


def _empty_schedule_collection() -> WakeScheduleCollection:
    """A WakeScheduleCollection with no pool (methods are overridden / unused)."""
    from threetears.core.collections.registry import CollectionRegistry
    from threetears.core.config import DefaultCoreConfig

    registry = CollectionRegistry()
    cfg = DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")
    return WakeScheduleCollection(registry=registry, config=cfg)


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


class TestWakeDueScheduleMapping:
    """``_WakeDueSchedule`` exposes the entity through the generic surface."""

    def test_maps_identity_and_scheduling_fields(self) -> None:
        entity = _make_schedule_entity()
        due = _WakeDueSchedule(entity)
        assert due.partition_key == entity.conversation_id
        assert due.job_id == entity.schedule_id
        assert due.kind == "agent_wake"
        assert due.schedule_type == entity.schedule_type
        assert due.schedule_config == entity.schedule_config
        assert due.missed_fire_policy == entity.missed_fire_policy
        assert due.next_fire_at == entity.next_fire_at
        assert due.last_fired_at == entity.last_fired_at
        assert due.name == entity.name

    def test_packs_agent_fields_into_payload(self) -> None:
        entity = _make_schedule_entity()
        payload = _WakeDueSchedule(entity).payload
        assert payload["user_id"] == entity.user_id
        assert payload["agent_id"] == entity.agent_id
        assert payload["skill_id"] == entity.skill_id
        assert payload["execution_mode"] == entity.execution_mode
        assert payload["task_prompt"] == entity.task_prompt
        assert payload["context_from_schedule_id"] == entity.context_from_schedule_id
        assert payload["include_conversation_history"] is False

    def test_conforms_to_due_schedule_protocol(self) -> None:
        assert isinstance(_WakeDueSchedule(_make_schedule_entity()), DueSchedule)


class TestRebuildWakeTrigger:
    """``_rebuild_wake_trigger`` reconstructs the wake trigger from the
    generic envelope -- the inverse of ``_WakeDueSchedule.payload``."""

    def test_round_trip_through_payload(self) -> None:
        entity = _make_schedule_entity()
        due = _WakeDueSchedule(entity)
        fired_at = datetime(2026, 6, 1, 12, 0, 5, tzinfo=UTC)
        job_trigger = JobTrigger(
            job_id=due.job_id,
            partition_key=due.partition_key,
            kind=due.kind,
            schedule_type=due.schedule_type,
            fired_at=fired_at,
            scheduled_fire_at=entity.next_fire_at or fired_at,
            payload=due.payload,
            name=due.name,
        )

        trigger = _rebuild_wake_trigger(job_trigger)

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

    def test_handles_absent_optional_fields(self) -> None:
        entity = _make_schedule_entity(
            skill_id=None,
            task_prompt=None,
            context_from_schedule_id=None,
            name=None,
            include_conversation_history=True,
        )
        due = _WakeDueSchedule(entity)
        job_trigger = JobTrigger(
            job_id=due.job_id,
            partition_key=due.partition_key,
            kind=due.kind,
            schedule_type=due.schedule_type,
            fired_at=datetime(2026, 6, 1, 12, 0, 5, tzinfo=UTC),
            scheduled_fire_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
            payload=due.payload,
            name=due.name,
        )
        trigger = _rebuild_wake_trigger(job_trigger)
        assert trigger.skill_id is None
        assert trigger.task_prompt is None
        assert trigger.context_from_schedule_id is None
        assert trigger.schedule_name is None
        assert trigger.include_conversation_history is True


class TestToJobFireResult:
    """``_to_job_fire_result`` packs wake's typed result into the opaque
    generic result + output dict."""

    def test_packs_output_and_passes_through_status(self) -> None:
        result = _to_job_fire_result(
            WakeDispatchResult(status="fired", output_text="hello", latency_ms=42, display_suppressed=False)
        )
        assert result.status == "fired"
        assert result.latency_ms == 42
        assert result.output == {"output_text": "hello", "display_suppressed": False}
        assert result.error is None

    def test_carries_silent_flag_and_error(self) -> None:
        result = _to_job_fire_result(
            WakeDispatchResult(status="fired_silent", output_text="[SILENT] hi", display_suppressed=True)
        )
        assert result.output == {"output_text": "[SILENT] hi", "display_suppressed": True}

        failed = _to_job_fire_result(WakeDispatchResult(status="failed", error="boom", latency_ms=3))
        assert failed.status == "failed"
        assert failed.error == "boom"


class TestProtocolConformance:
    """The adapter stores satisfy the core Protocols structurally."""

    def test_schedule_store_conforms(self) -> None:
        assert isinstance(_WakeScheduleStore(_empty_schedule_collection()), ScheduleStore)

    def test_fire_store_conforms(self) -> None:
        assert isinstance(_WakeFireStore(_empty_fire_collection()), FireStore)


class TestScheduleStoreTranslation:
    """``_WakeScheduleStore`` translates the generic param names."""

    async def test_list_due_wraps_entities(self) -> None:
        entities = [_make_schedule_entity(), _make_schedule_entity()]
        store = _WakeScheduleStore(_RecordingScheduleCollection(due=entities))
        now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
        rows = await store.list_due_for_tick(now, limit=50)
        assert len(rows) == 2
        assert all(isinstance(r, _WakeDueSchedule) for r in rows)
        assert rows[0].job_id == entities[0].schedule_id

    async def test_claim_maps_partition_and_job_ids(self) -> None:
        coll = _RecordingScheduleCollection()
        store = _WakeScheduleStore(coll)
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
    """``_WakeFireStore`` maps create + unpacks the opaque output dict."""

    async def test_create_dispatching_maps_ids_and_scheduled_source(self) -> None:
        coll = _RecordingFireCollection()
        store = _WakeFireStore(coll)
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

    async def test_finalize_success_unpacks_output_dict(self) -> None:
        coll = _RecordingFireCollection()
        store = _WakeFireStore(coll)
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

    async def test_finalize_success_tolerates_missing_output(self) -> None:
        coll = _RecordingFireCollection()
        store = _WakeFireStore(coll)
        await store.finalize_success(_new_uuid(), _new_uuid(), status="yielded", output=None)
        call = coll.success_calls[0]
        assert call["output_text"] is None
        assert call["display_suppressed"] is False

    async def test_finalize_failed_passthrough(self) -> None:
        coll = _RecordingFireCollection()
        store = _WakeFireStore(coll)
        pk = _new_uuid()
        fire_id = _new_uuid()
        await store.finalize_failed(pk, fire_id, error="kaboom", latency_ms=11)
        assert coll.failed_calls == [{"conversation_id": pk, "fire_id": fire_id, "error": "kaboom", "latency_ms": 11}]


class TestWakeTickJobWiring:
    """``wake_tick_job`` delegates to the core engine with wake's lock key +
    the ``nats_client`` passed through, and the adapter callback bridges the
    consumer's wake-shaped callback."""

    async def test_delegates_with_wake_lock_key_and_nats_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        async def _fake_engine(
            schedule_store: Any,
            fire_store: Any,
            dispatch_callback: Any,
            *,
            nats_client: Any = None,
            config: Any = None,
        ) -> None:
            captured["schedule_store"] = schedule_store
            captured["fire_store"] = fire_store
            captured["dispatch_callback"] = dispatch_callback
            captured["nats_client"] = nats_client
            captured["config"] = config

        monkeypatch.setattr(tick_mod, "scheduled_tick_job", _fake_engine)

        nats = object()

        async def _cb(_t: WakeTrigger, _f: UUID, _p: Any) -> WakeDispatchResult:
            return WakeDispatchResult(status="fired")

        await wake_tick_job(object(), nats, _cb)

        assert isinstance(captured["schedule_store"], _WakeScheduleStore)
        assert isinstance(captured["fire_store"], _WakeFireStore)
        assert captured["nats_client"] is nats
        assert captured["config"].tick_lock_key == _WAKE_TICK_LOCK_KEY == "agent_wake_tick"

    async def test_adapter_callback_bridges_trigger_and_packs_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        async def _fake_engine(
            _ss: Any, _fs: Any, dispatch_callback: Any, *, nats_client: Any = None, config: Any = None
        ) -> None:
            captured["cb"] = dispatch_callback

        monkeypatch.setattr(tick_mod, "scheduled_tick_job", _fake_engine)

        pool = object()
        seen: dict[str, Any] = {}

        async def _cb(trigger: WakeTrigger, fire_id: UUID, p: Any) -> WakeDispatchResult:
            seen["trigger"] = trigger
            seen["fire_id"] = fire_id
            seen["pool"] = p
            return WakeDispatchResult(status="fired", output_text="ok", latency_ms=9)

        await wake_tick_job(pool, None, _cb)

        # drive the captured adapter callback with a generic envelope
        entity = _make_schedule_entity()
        due = _WakeDueSchedule(entity)
        fire_id = _new_uuid()
        job_trigger = JobTrigger(
            job_id=due.job_id,
            partition_key=due.partition_key,
            kind=due.kind,
            schedule_type=due.schedule_type,
            fired_at=datetime(2026, 6, 1, 12, 0, 5, tzinfo=UTC),
            scheduled_fire_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
            payload=due.payload,
            name=due.name,
        )
        result = await captured["cb"](job_trigger, fire_id)

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
        captured: dict[str, Any] = {}

        async def _fake_engine(
            _ss: Any, _fs: Any, dispatch_callback: Any, *, nats_client: Any = None, config: Any = None
        ) -> None:
            captured["cb"] = dispatch_callback

        observed: list[float] = []

        class _RecordingEmitter:
            def observe_yield_duration(self, seconds: float) -> None:
                observed.append(seconds)

        monkeypatch.setattr(tick_mod, "scheduled_tick_job", _fake_engine)
        monkeypatch.setattr(tick_mod, "get_wake_emitter", lambda: _RecordingEmitter())

        async def _cb(_t: WakeTrigger, _f: UUID, _p: Any) -> WakeDispatchResult:
            return WakeDispatchResult(status="yielded", latency_ms=2000)

        await wake_tick_job(object(), None, _cb)

        due = _WakeDueSchedule(_make_schedule_entity())
        job_trigger = JobTrigger(
            job_id=due.job_id,
            partition_key=due.partition_key,
            kind=due.kind,
            schedule_type=due.schedule_type,
            fired_at=datetime(2026, 6, 1, 12, 0, 5, tzinfo=UTC),
            scheduled_fire_at=datetime(2026, 6, 1, 12, 0, tzinfo=UTC),
            payload=due.payload,
            name=due.name,
        )
        await captured["cb"](job_trigger, _new_uuid())

        assert observed == [2.0]
