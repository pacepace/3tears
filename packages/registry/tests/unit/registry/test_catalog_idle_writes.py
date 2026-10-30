"""a pod re-registering what it already registered writes nothing to the catalog bucket.

Every tool pod re-publishes its whole manifest on every heartbeat, and every registry replica hears
every one. The catalog used to write each of the pod's tools to the bucket on each of them, so an
idle platform rewrote its whole catalog every heartbeat interval, once per replica, and logged a
line per tool as it did. The bucket is only ever read to warm-load a starting registry, so a
write that changes nothing a reader uses is pure churn.

What still moves on every heartbeat is liveness: when each copy last registered and when each
definition was last announced. That stays in memory, where routing, discovery and the heartbeat
sweep read it, and the per-pod liveness every replica shares stays in the heartbeat rows. These pin
both halves: the idle writes are gone, and staleness is still judged from the latest heartbeat.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

import pytest

from threetears.agent.tools.server import RegistrationManifest, ToolManifestEntry
from threetears.core.testing.kv import FakeKvBucket
from threetears.nats import IncomingMessage, set_default_namespace
from threetears.registry.catalog import WRITE_FAILURE_THRESHOLD, CopyStatus, ToolCatalog
from threetears.registry.registration import RegistrationHandler

from .copy_entries import announced, definition, endpoint, entry

_POD = "pod-001"
_TOOL = "threetears.calculator@1.0.0"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """a deterministic subject namespace for the handler's subjects.

    :return: nothing
    :rtype: None
    """
    set_default_namespace("test")


async def _bound_catalog() -> tuple[ToolCatalog, FakeKvBucket]:
    """a catalog bound to an empty in-memory bucket.

    :return: the catalog and its bucket
    :rtype: tuple[ToolCatalog, FakeKvBucket]
    """
    bucket = FakeKvBucket(bucket_name="tool_catalog", storage="file", direct=True)
    catalog = ToolCatalog()
    await catalog.load_from_kv(bucket)
    return catalog, bucket


async def _writes(bucket: FakeKvBucket) -> int:
    """the bucket's latest revision: every put and delete advances it by one.

    :param bucket: the bucket
    :ptype bucket: FakeKvBucket
    :return: the highest revision any key holds, 0 for an empty bucket
    :rtype: int
    """
    revisions = [0]
    for key in await bucket.list_keys():
        held = await bucket.get_entry(key=key)
        if held is not None:
            revisions.append(held[1])
    return max(revisions)


class _ProbeAnsweringClient:
    """the slice of the NATS client the registration handler uses, answering every probe ready."""

    def __init__(self) -> None:
        self.replies: list[object] = []

    async def subscribe(self, *, subject: object, cb: object, queue: str | None = None) -> object:
        """accept the subscription.

        :param subject: the subject
        :ptype subject: object
        :param cb: the handler
        :ptype cb: object
        :param queue: the queue group
        :ptype queue: str | None
        :return: a subscription token
        :rtype: object
        """
        del subject, cb, queue
        return object()

    async def unsubscribe(self, sub: object) -> None:
        """drop the subscription.

        :param sub: the subscription token
        :ptype sub: object
        :return: nothing
        :rtype: None
        """
        del sub

    async def request(self, *, subject: object, message: object, response_type: type, timeout: object) -> object:
        """answer a reachability probe: the pod is ready.

        :param subject: the probe subject, ending in the pod id
        :ptype subject: object
        :param message: the probe
        :ptype message: object
        :param response_type: the reply model
        :ptype response_type: type
        :param timeout: the probe timeout
        :ptype timeout: object
        :return: a ready reply
        :rtype: object
        """
        del message, timeout
        pod_id = str(getattr(subject, "path", "")).rsplit(".", 1)[-1]
        return response_type(pod_id=pod_id, ready=True)

    async def publish_reply(self, *, reply_subject: str, message: object) -> None:
        """record the registration reply.

        :param reply_subject: where the reply goes
        :ptype reply_subject: str
        :param message: the reply
        :ptype message: object
        :return: nothing
        :rtype: None
        """
        del reply_subject
        self.replies.append(message)


def _manifest(description: str = "calculator tool") -> IncomingMessage:
    """the manifest a pod re-publishes on every heartbeat.

    :param description: the tool's description
    :ptype description: str
    :return: the message carrying it
    :rtype: IncomingMessage
    """
    manifest = RegistrationManifest(
        pod_id=_POD,
        tools=[
            ToolManifestEntry(
                name="threetears.calculator",
                version="1.0.0",
                description=description,
                input_schema={"type": "object", "properties": {}},
            )
        ],
    )
    return IncomingMessage(
        data=manifest.model_dump_json().encode("utf-8"), reply_subject="reply.subject", subject="test.tools.register"
    )


async def _handler(catalog: ToolCatalog) -> RegistrationHandler:
    """an open-mode registration handler over ``catalog``, started.

    :param catalog: the catalog
    :ptype catalog: ToolCatalog
    :return: the handler
    :rtype: RegistrationHandler
    """
    handler = RegistrationHandler(catalog, namespace="test")
    await handler.start(_ProbeAnsweringClient())  # type: ignore[arg-type]
    return handler


@pytest.mark.asyncio
async def test_repeated_identical_registrations_write_nothing_after_the_first() -> None:
    """the first registration writes the tool (and its promotion); a heartbeat's re-registration writes nothing."""
    catalog, bucket = await _bound_catalog()
    handler = await _handler(catalog)

    await handler.handle_registration(_manifest())
    after_first = await _writes(bucket)
    assert after_first > 0, "the first registration is persisted"

    for _ in range(5):
        await handler.handle_registration(_manifest())

    assert await _writes(bucket) == after_first


@pytest.mark.asyncio
async def test_a_changed_definition_is_written_once() -> None:
    """a pod that starts announcing a different definition writes it once, then goes quiet again."""
    catalog, bucket = await _bound_catalog()
    handler = await _handler(catalog)
    await handler.handle_registration(_manifest())
    before = await _writes(bucket)

    await handler.handle_registration(_manifest("calculator tool, now with powers"))
    changed = await _writes(bucket)
    assert changed == before + 1

    for _ in range(3):
        await handler.handle_registration(_manifest("calculator tool, now with powers"))
    assert await _writes(bucket) == changed

    raw = await bucket.get(key=(await bucket.list_keys())[0])
    assert raw is not None
    persisted = json.loads(raw)
    descriptions = {
        item["definition"]["description"] for copy in persisted["endpoints"] for item in copy["definitions"]
    }
    assert "calculator tool, now with powers" in descriptions


@pytest.mark.asyncio
async def test_a_status_change_is_written() -> None:
    """a copy quarantined and re-registered is a change to the persisted copy, and is written."""
    catalog, bucket = await _bound_catalog()
    handler = await _handler(catalog)
    await handler.handle_registration(_manifest())
    before = await _writes(bucket)

    catalog.mark_pod_endpoints_unavailable(_POD)
    await handler.handle_registration(_manifest())

    assert await _writes(bucket) == before + 1


@pytest.mark.asyncio
async def test_liveness_advances_in_memory_on_a_registration_that_writes_nothing() -> None:
    """an unwritten re-registration still refreshes the copy's liveness, so staleness runs from it."""
    catalog, bucket = await _bound_catalog()
    now = datetime.now(UTC)
    tool = definition()
    early = now - timedelta(seconds=40)
    await catalog.register(entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=tool, first=early)))
    before = await _writes(bucket)

    again = endpoint(_POD, tool_definition=tool, first=early)
    again.definitions = announced(tool, first=early, last=now)
    again.date_last_heartbeat = now
    await catalog.register(entry("threetears.calculator", "1.0.0", again))

    assert await _writes(bucket) == before, "a refreshed liveness alone is not written"
    held = catalog.get(_TOOL)
    assert held is not None
    ttl = timedelta(seconds=45)
    # 70 s after the first announcement, but 30 s after the latest: live, judged from the latest
    assert held.copy_status(_POD, now=now + timedelta(seconds=30), ttl=ttl) is CopyStatus.AVAILABLE
    # and stale once the latest announcement itself is older than the window
    assert held.copy_status(_POD, now=now + timedelta(seconds=46), ttl=ttl) is CopyStatus.UNAVAILABLE
    copy = held.get_endpoint(_POD)
    assert copy is not None
    assert copy.date_last_heartbeat == now


@pytest.mark.asyncio
async def test_a_definition_that_lapses_is_pruned_and_the_prune_is_written(monkeypatch: pytest.MonkeyPatch) -> None:
    """a definition the pod stopped announcing is dropped at its next registration past the window, once."""
    catalog, bucket = await _bound_catalog()
    now = datetime.now(UTC)
    old = definition("the old definition")
    new = definition("the new definition")
    monkeypatch.setenv("THREETEARS_REGISTRY_DEFINITION_TTL", "35")
    await catalog.register(
        entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=old, first=now - timedelta(seconds=30)))
    )
    await catalog.register(entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=new, first=now)))
    held = catalog.get(_TOOL)
    assert held is not None
    copy = held.get_endpoint(_POD)
    assert copy is not None
    assert len(copy.definitions) == 2, "both are live inside the 35 s window"
    before = await _writes(bucket)

    monkeypatch.setenv("THREETEARS_REGISTRY_DEFINITION_TTL", "20")
    await catalog.register(entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=new, first=now)))
    assert list(copy.definitions) == [new.digest]
    assert await _writes(bucket) == before + 1, "the prune changed the copy, so it is written"

    await catalog.register(entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=new, first=now)))
    assert await _writes(bucket) == before + 1


@pytest.mark.asyncio
async def test_a_catalog_loaded_from_the_bucket_is_revived_by_its_pods_next_heartbeat() -> None:
    """a starting registry loads copies unavailable; the pod's next heartbeat and registration serve them again."""
    catalog, bucket = await _bound_catalog()
    first = await _handler(catalog)
    await first.handle_registration(_manifest())
    for _ in range(3):
        await first.handle_registration(_manifest())

    restarted = ToolCatalog()
    await restarted.load_from_kv(bucket)
    loaded = restarted.get(_TOOL)
    assert loaded is not None
    assert loaded.copy_status(_POD) is CopyStatus.UNAVAILABLE, "a loaded copy waits for its pod"

    restarted.mark_pod_endpoints_available(_POD)
    await (await _handler(restarted)).handle_registration(_manifest())

    revived = restarted.get(_TOOL)
    assert revived is not None
    assert revived.copy_status(_POD) is CopyStatus.AVAILABLE
    assert revived.available_to(None)


@pytest.mark.asyncio
async def test_a_registration_that_changes_nothing_logs_nothing_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """the per-heartbeat INFO lines are gone; a registration that changes the catalog still says so."""
    catalog, _bucket = await _bound_catalog()
    handler = await _handler(catalog)
    loggers = ("threetears.registry.catalog", "threetears.registry.registration")

    with caplog.at_level(logging.INFO):
        await handler.handle_registration(_manifest())
    first = [record for record in caplog.records if record.name in loggers and record.levelno >= logging.INFO]
    assert any("registered tool in catalog" in record.getMessage() for record in first)
    assert any("registration completed" in record.getMessage() for record in first)

    caplog.clear()
    with caplog.at_level(logging.INFO):
        for _ in range(5):
            await handler.handle_registration(_manifest())
    repeated = [record for record in caplog.records if record.name in loggers and record.levelno >= logging.INFO]
    assert repeated == [], [record.getMessage() for record in repeated]

    caplog.clear()
    with caplog.at_level(logging.INFO):
        await handler.handle_registration(_manifest("calculator tool, now with powers"))
    changed = [record for record in caplog.records if record.name in loggers and record.levelno >= logging.INFO]
    assert any("registered tool in catalog" in record.getMessage() for record in changed)


@pytest.mark.asyncio
async def test_a_write_that_failed_is_tried_again_on_the_next_registration() -> None:
    """a write that did not land is not remembered as written, whatever was written before it.

    content A lands; the write of B fails, so the bucket still holds A; the copy goes back to A.
    the bucket's A is what this catalog last recorded, but the failed write left the key's state
    unknown, so the registration of A writes again rather than taking the bucket to hold it.
    """
    catalog, bucket = await _bound_catalog()
    first = definition("content A")
    second = definition("content B")
    now = datetime.now(UTC)
    await catalog.register(entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=first, first=now)))
    landed = await _writes(bucket)

    bucket.become_unreachable(RuntimeError("bucket gone"))
    with pytest.raises(RuntimeError):
        await catalog.register(
            entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=second, first=now))
        )
    bucket.become_reachable()
    # the copy now holds both definitions in memory; drop B the way a lapsed definition is dropped
    held = catalog.get(_TOOL)
    assert held is not None
    copy = held.get_endpoint(_POD)
    assert copy is not None
    del copy.definitions[second.digest]

    await catalog.register(entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=first, first=now)))

    assert await _writes(bucket) == landed + 1


@pytest.mark.asyncio
async def test_a_catalog_with_no_bucket_still_tells_a_change_from_a_repeat() -> None:
    """before its bucket is bound a catalog writes nothing, but still reports which registrations changed it."""
    catalog = ToolCatalog()
    now = datetime.now(UTC)
    tool = definition()
    assert await catalog.register(
        entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=tool, first=now))
    )
    assert not await catalog.register(
        entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=tool, first=now))
    )
    assert await catalog.register(
        entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=definition("changed"), first=now))
    )
    await catalog.deregister(_TOOL)
    assert await catalog.register(
        entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=tool, first=now))
    )


@pytest.mark.asyncio
async def test_a_replica_out_of_rotation_for_failed_deletes_comes_back_on_an_idle_heartbeat() -> None:
    """a delete that failed is never retried: its entry is gone from memory. so while the catalog is not
    persisting, a re-registration writes even when nothing changed, and the first one that lands puts the
    replica back in rotation -- on an idle platform that write is the only one coming."""
    catalog, bucket = await _bound_catalog()
    now = datetime.now(UTC)
    tool = definition()
    kept = entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=tool, first=now))
    await catalog.register(kept)
    for index in range(WRITE_FAILURE_THRESHOLD):
        await catalog.register(entry(f"threetears.gone{index}", "1.0.0", endpoint("pod-002")))
    bucket.become_unreachable(RuntimeError("bucket gone"))
    for index in range(WRITE_FAILURE_THRESHOLD):
        assert await catalog.deregister(f"threetears.gone{index}@1.0.0") is False
    assert catalog.persisting is False
    bucket.become_reachable()
    before = await _writes(bucket)

    await catalog.register(entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=tool, first=now)))

    assert await _writes(bucket) == before + 1
    assert catalog.persisting is True
    # back in rotation, an unchanged re-registration is quiet again
    await catalog.register(entry("threetears.calculator", "1.0.0", endpoint(_POD, tool_definition=tool, first=now)))
    assert await _writes(bucket) == before + 1
