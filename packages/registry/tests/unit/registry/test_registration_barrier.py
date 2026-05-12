"""tests for three-phase tool pod registration barrier.

covers the handshake: pod publishes manifest -> registry writes
catalog entry in 'pending' state -> registry probes pod via
request-reply -> on successful round-trip, registry promotes
pending endpoints to 'available'. on probe failure, endpoints
remain pending so the routing path refuses to forward.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.agent.tools.server import RegistrationManifest, ToolManifestEntry
from threetears.nats import IncomingMessage, RequestError, set_default_namespace
from threetears.registry.catalog import ToolCatalog
from threetears.registry.registration import (
    ProbeRequest,
    ProbeResponse,
    RegistrationHandler,
)


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """default namespace so :class:`Subjects` builders are deterministic."""
    set_default_namespace("test")


# -- helpers --


def _make_manifest(
    pod_id: str = "pod-probe-001",
    tools: list[dict[str, Any]] | None = None,
) -> RegistrationManifest:
    """create registration manifest for barrier tests.

    :param pod_id: pod identifier
    :ptype pod_id: str
    :param tools: optional list of tool dicts
    :ptype tools: list[dict[str, Any]] | None
    :return: test registration manifest
    :rtype: RegistrationManifest
    """
    if tools is None:
        tools = [
            {
                "name": "threetears.calculator",
                "version": "1.0.0",
                "description": "calculator tool",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
    tool_entries = [ToolManifestEntry(**t) for t in tools]
    result = RegistrationManifest(pod_id=pod_id, tools=tool_entries)
    return result


def _make_nats_msg(
    data: bytes,
    reply: str | None = "reply.subject",
) -> IncomingMessage:
    """build a wrapper :class:`IncomingMessage` envelope.

    :param data: raw message payload bytes
    :ptype data: bytes
    :param reply: optional reply subject; ``None`` for fire-and-forget
    :ptype reply: str | None
    :return: wrapper-shaped envelope
    :rtype: IncomingMessage
    """
    return IncomingMessage(data=data, reply_subject=reply, subject="aibots.tools.register")


# -- state machine tests --


class TestRegistrationBarrierStateMachine:
    """tests for pending -> available transition driven by probe."""

    @pytest.mark.asyncio
    async def test_fresh_registration_lands_as_pending(self) -> None:
        """newly registered endpoint starts in 'pending' state before probe completes."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(
            catalog,
            namespace="test",
            probe_timeout=1.0,
        )
        nc = AsyncMock()
        nc.request = AsyncMock(side_effect=RequestError("request timed out"))
        await handler.start(nc)

        manifest = _make_manifest()
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        assert len(entry.endpoints) == 1
        assert entry.endpoints[0].status == "pending"

    @pytest.mark.asyncio
    async def test_probe_success_promotes_to_available(self) -> None:
        """successful probe round-trip transitions pending endpoint to available."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test", probe_timeout=1.0)
        nc = AsyncMock()
        # the wrapper's request() returns the parsed response_type
        # instance directly, not a raw nats message with .data
        nc.request = AsyncMock(
            return_value=ProbeResponse(pod_id="pod-probe-001", ready=True),
        )
        await handler.start(nc)

        manifest = _make_manifest()
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        assert entry.endpoints[0].status == "available"

    @pytest.mark.asyncio
    async def test_probe_failure_leaves_endpoint_pending(self) -> None:
        """probe timeout or error leaves endpoint in 'pending' state."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test", probe_timeout=0.5)
        nc = AsyncMock()
        nc.request = AsyncMock(side_effect=RequestError("request timed out"))
        await handler.start(nc)

        manifest = _make_manifest()
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        assert entry.endpoints[0].status == "pending"

    @pytest.mark.asyncio
    async def test_probe_malformed_reply_leaves_endpoint_pending(self) -> None:
        """malformed probe reply leaves endpoint pending.

        the wrapper's :meth:`request` already validates the reply
        bytes against ``response_type=ProbeResponse`` and raises
        :class:`RequestError` on validation failure. simulate that by
        having the AsyncMock raise; the handler catches Exception so
        endpoints stay pending.
        """
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test", probe_timeout=0.5)
        nc = AsyncMock()
        nc.request = AsyncMock(
            side_effect=RequestError("response decode failed: subject=test.tools.probe.pod-probe-001"),
        )
        await handler.start(nc)

        manifest = _make_manifest()
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        assert entry.endpoints[0].status == "pending"

    @pytest.mark.asyncio
    async def test_probe_reply_with_ready_false_leaves_endpoint_pending(self) -> None:
        """ProbeResponse(ready=False) does not promote endpoints.

        a pod explicitly reporting itself not ready (e.g. still warming
        caches) must not flip catalog state; endpoints remain pending
        until a subsequent probe confirms readiness.
        """
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test", probe_timeout=0.5)
        nc = AsyncMock()
        # the wrapper returns the parsed ProbeResponse directly
        nc.request = AsyncMock(
            return_value=ProbeResponse(pod_id="pod-001", ready=False),
        )
        await handler.start(nc)

        manifest = _make_manifest()
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        assert entry.endpoints[0].status == "pending"

    @pytest.mark.asyncio
    async def test_probe_issued_to_pod_specific_subject(self) -> None:
        """probe is issued to {namespace}.tools.probe.{pod_id} subject."""
        set_default_namespace("custom-ns")
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="custom-ns", probe_timeout=1.0)
        nc = AsyncMock()
        # any side_effect that prevents promotion is fine; this test
        # only inspects which subject the wrapper got called on.
        nc.request = AsyncMock(
            side_effect=RequestError("not interested in success path here"),
        )
        await handler.start(nc)

        manifest = _make_manifest(pod_id="pod-subject-check")
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        nc.request.assert_called_once()
        # wrapper request is kw-only with typed Subject
        call_args = nc.request.call_args
        assert call_args.kwargs["subject"].path == "custom-ns.tools.probe.pod-subject-check"

    @pytest.mark.asyncio
    async def test_probe_payload_carries_pod_id(self) -> None:
        """probe message is a ProbeRequest carrying pod_id."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test", probe_timeout=1.0)
        nc = AsyncMock()
        nc.request = AsyncMock(
            side_effect=RequestError("not interested in success path here"),
        )
        await handler.start(nc)

        manifest = _make_manifest(pod_id="pod-payload-001")
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        # wrapper request is kw-only typed; the message kwarg is the
        # raw ProbeRequest instance the handler built
        message = nc.request.call_args.kwargs["message"]
        assert isinstance(message, ProbeRequest)
        assert message.pod_id == "pod-payload-001"

    @pytest.mark.asyncio
    async def test_probe_timeout_applied_to_nats_request(self) -> None:
        """probe timeout from config is passed as a timedelta to the wrapper."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test", probe_timeout=2.5)
        nc = AsyncMock()
        nc.request = AsyncMock(
            side_effect=RequestError("not interested in success path here"),
        )
        await handler.start(nc)

        manifest = _make_manifest()
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        call_args = nc.request.call_args
        # wrapper takes a timedelta; the handler converts the
        # configured float seconds to timedelta(seconds=2.5).
        assert call_args.kwargs["timeout"].total_seconds() == 2.5


# -- multi-tool and multi-pod tests --


class TestRegistrationBarrierMultipleTools:
    """tests for probe promoting all pending endpoints for a pod at once."""

    @pytest.mark.asyncio
    async def test_probe_success_promotes_all_tools_for_pod(self) -> None:
        """single probe round-trip promotes every pending endpoint for that pod."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test", probe_timeout=1.0)
        nc = AsyncMock()
        probe_reply = MagicMock()
        probe_reply.data = ProbeResponse(pod_id="pod-multi").model_dump_json().encode("utf-8")
        nc.request = AsyncMock(return_value=probe_reply)
        await handler.start(nc)

        manifest = _make_manifest(
            pod_id="pod-multi",
            tools=[
                {
                    "name": "threetears.calculator",
                    "version": "1.0.0",
                    "description": "calculator",
                    "input_schema": {"type": "object"},
                },
                {
                    "name": "threetears.dictionary",
                    "version": "1.0.0",
                    "description": "dictionary",
                    "input_schema": {"type": "object"},
                },
            ],
        )
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        calc_entry = catalog.get("threetears.calculator@1.0.0")
        dict_entry = catalog.get("threetears.dictionary@1.0.0")
        assert calc_entry is not None
        assert dict_entry is not None
        assert calc_entry.endpoints[0].status == "available"
        assert dict_entry.endpoints[0].status == "available"

    @pytest.mark.asyncio
    async def test_probe_only_affects_registering_pod(self) -> None:
        """probe for pod-B does not promote pending endpoints for pod-A."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test", probe_timeout=0.5)

        # pod-A registers but probe fails; endpoints remain pending
        nc_a = AsyncMock()
        nc_a.request = AsyncMock(side_effect=TimeoutError("dead pod"))
        await handler.start(nc_a)
        manifest_a = _make_manifest(pod_id="pod-A")
        msg_a = _make_nats_msg(data=manifest_a.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg_a)

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        pod_a_endpoint = entry.get_endpoint("pod-A")
        assert pod_a_endpoint is not None
        assert pod_a_endpoint.status == "pending"

        # pod-B registers and probe succeeds; pod-A must stay pending
        nc_b = AsyncMock()
        probe_reply_b = MagicMock()
        probe_reply_b.data = ProbeResponse(pod_id="pod-B").model_dump_json().encode("utf-8")
        nc_b.request = AsyncMock(return_value=probe_reply_b)
        await handler.start(nc_b)
        manifest_b = _make_manifest(pod_id="pod-B")
        msg_b = _make_nats_msg(data=manifest_b.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg_b)

        entry_after = catalog.get("threetears.calculator@1.0.0")
        assert entry_after is not None
        pod_a_after = entry_after.get_endpoint("pod-A")
        pod_b_after = entry_after.get_endpoint("pod-B")
        assert pod_a_after is not None
        assert pod_b_after is not None
        assert pod_a_after.status == "pending"
        assert pod_b_after.status == "available"


# -- catalog mark_ready tests --


class TestCatalogMarkReady:
    """tests for ToolCatalog.mark_ready promotion method."""

    @pytest.mark.asyncio
    async def test_mark_ready_promotes_only_pending(self) -> None:
        """mark_ready flips only 'pending' endpoints to 'available'."""
        from threetears.registry.catalog import CatalogEntry, ToolEndpoint

        catalog = ToolCatalog()
        entry_pending = CatalogEntry(
            tool_name="tool.warming",
            tool_version="1.0",
            full_name="tool.warming@1.0",
            description="pending",
            input_schema={},
            endpoints=[ToolEndpoint(pod_id="pod-A", status="pending")],
        )
        entry_unavailable = CatalogEntry(
            tool_name="tool.down",
            tool_version="1.0",
            full_name="tool.down@1.0",
            description="unavailable",
            input_schema={},
            endpoints=[ToolEndpoint(pod_id="pod-A", status="unavailable")],
        )
        await catalog.register(entry_pending)
        await catalog.register(entry_unavailable)

        promoted = await catalog.mark_ready("pod-A")
        assert promoted == ["tool.warming@1.0"]

        warming_after = catalog.get("tool.warming@1.0")
        down_after = catalog.get("tool.down@1.0")
        assert warming_after is not None
        assert down_after is not None
        assert warming_after.endpoints[0].status == "available"
        assert down_after.endpoints[0].status == "unavailable"

    @pytest.mark.asyncio
    async def test_mark_ready_returns_empty_for_unknown_pod(self) -> None:
        """mark_ready returns empty list when pod has no pending endpoints."""
        catalog = ToolCatalog()
        promoted = await catalog.mark_ready("nonexistent-pod")
        assert promoted == []

    @pytest.mark.asyncio
    async def test_mark_ready_persists_to_kv(self) -> None:
        """mark_ready writes promoted entry to KV so recovery preserves the transition."""
        import json
        from threetears.registry.catalog import CatalogEntry, ToolEndpoint

        catalog = ToolCatalog()
        kv = AsyncMock()
        kv.keys = AsyncMock(return_value=[])
        kv.put = AsyncMock()
        kv.delete = AsyncMock()
        await catalog.load_from_kv(kv)

        entry = CatalogEntry(
            tool_name="tool.promote",
            tool_version="1.0",
            full_name="tool.promote@1.0",
            description="promotion test",
            input_schema={},
            endpoints=[ToolEndpoint(pod_id="pod-P", status="pending")],
        )
        await catalog.register(entry)
        kv.put.reset_mock()

        promoted = await catalog.mark_ready("pod-P")
        assert promoted == ["tool.promote@1.0"]

        kv.put.assert_called_once()
        call_args = kv.put.call_args
        payload = json.loads(call_args[0][1].decode("utf-8"))
        assert payload["endpoints"][0]["status"] == "available"

    @pytest.mark.asyncio
    async def test_mark_ready_rolls_back_in_memory_on_kv_failure(self) -> None:
        """KV write failure during mark_ready leaves in-memory state untouched.

        atomicity invariant: if any KV write raises, no endpoint has its
        in-memory status flipped, so a retry can safely re-run the whole
        transition without a partially-applied state.
        """
        from threetears.registry.catalog import CatalogEntry, ToolEndpoint

        catalog = ToolCatalog()
        kv = AsyncMock()
        kv.keys = AsyncMock(return_value=[])
        kv.put = AsyncMock()
        kv.delete = AsyncMock()
        await catalog.load_from_kv(kv)

        entry_one = CatalogEntry(
            tool_name="tool.one",
            tool_version="1.0",
            full_name="tool.one@1.0",
            description="one",
            input_schema={},
            endpoints=[ToolEndpoint(pod_id="pod-X", status="pending")],
        )
        entry_two = CatalogEntry(
            tool_name="tool.two",
            tool_version="1.0",
            full_name="tool.two@1.0",
            description="two",
            input_schema={},
            endpoints=[ToolEndpoint(pod_id="pod-X", status="pending")],
        )
        await catalog.register(entry_one)
        await catalog.register(entry_two)
        # arm the failure only after both successful registrations have written
        # baseline state; mark_ready is now the failing path.
        kv.put.side_effect = RuntimeError("kv write failed")

        with pytest.raises(RuntimeError, match="kv write failed"):
            await catalog.mark_ready("pod-X")

        # both endpoints must still be pending; no partial in-memory flip
        one_after = catalog.get("tool.one@1.0")
        two_after = catalog.get("tool.two@1.0")
        assert one_after is not None
        assert two_after is not None
        assert one_after.endpoints[0].status == "pending"
        assert two_after.endpoints[0].status == "pending"

    @pytest.mark.asyncio
    async def test_mark_ready_preserves_date_registered_in_kv(self) -> None:
        """mark_ready KV write retains the original date_registered.

        prevents the KV snapshot from drifting the registration timestamp
        to probe-promotion time on every transition.
        """
        import json
        from datetime import UTC, datetime

        from threetears.registry.catalog import CatalogEntry, ToolEndpoint

        catalog = ToolCatalog()
        kv = AsyncMock()
        kv.keys = AsyncMock(return_value=[])
        kv.put = AsyncMock()
        kv.delete = AsyncMock()
        await catalog.load_from_kv(kv)

        original_date = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        entry = CatalogEntry(
            tool_name="tool.dated",
            tool_version="1.0",
            full_name="tool.dated@1.0",
            description="dated",
            input_schema={},
            endpoints=[ToolEndpoint(pod_id="pod-D", status="pending")],
            date_registered=original_date,
        )
        await catalog.register(entry)
        kv.put.reset_mock()

        await catalog.mark_ready("pod-D")

        kv.put.assert_called_once()
        payload = json.loads(kv.put.call_args[0][1].decode("utf-8"))
        assert payload["date_registered"] == original_date.isoformat()
