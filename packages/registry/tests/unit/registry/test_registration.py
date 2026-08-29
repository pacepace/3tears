"""tests for RegistrationHandler."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.agent.tools.server import RegistrationManifest, ToolManifestEntry
from threetears.nats import IncomingMessage, set_default_namespace
from threetears.registry.auth import ToolPodAuth
from threetears.registry.catalog import CatalogEntry, ToolCatalog, ToolEndpoint
from threetears.registry.registration import (
    RegistrationHandler,
    RegistrationResponse,
)


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """default namespace so :class:`Subjects` builders are deterministic.

    each test that needs a different prefix calls
    :func:`set_default_namespace` directly inside its body; this
    fixture resets to ``test`` so cross-test bleed is impossible.
    """
    set_default_namespace("test")


# -- helpers --


def _make_registry_nc() -> AsyncMock:
    """build an :class:`AsyncMock` NATS wrapper replying to every probe subject.

    probe subjects follow ``{ns}.tools.probe.{pod_id}``; the mock's
    ``request`` method parses pod_id out of the subject (a typed
    :class:`Subject`) and echoes it back in a valid
    :class:`ProbeResponse`. tests never have to wire probe replies
    per pod_id.

    matches the canonical wrapper surface RegistrationHandler depends
    on: kw-only ``request(subject, message, response_type, timeout)``,
    kw-only ``subscribe(subject, cb, queue=None)``, kw-only
    ``publish_reply(reply_subject, message)``,
    ``unsubscribe(sub)``.

    :return: configured AsyncMock NATS wrapper
    :rtype: AsyncMock
    """

    async def _reply(
        *,
        subject: Any,
        message: Any,
        response_type: Any,
        timeout: Any,
    ) -> Any:
        del message, timeout
        pod_id = subject.path.rsplit(".", 1)[-1]
        # the wrapper's request method returns the parsed
        # :class:`response_type` instance; mirror that here so the
        # registration handler receives a typed ProbeResponse.
        return response_type(pod_id=pod_id, ready=True)

    nc = AsyncMock()
    nc.request = AsyncMock(side_effect=_reply)
    return nc


def _make_manifest(
    pod_id: str = "pod-001",
    tools: list[dict[str, Any]] | None = None,
) -> RegistrationManifest:
    """create registration manifest for testing.

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
    subject: str = "3tears.tools.register",
) -> IncomingMessage:
    """build a wrapper :class:`IncomingMessage` envelope.

    :param data: raw message payload bytes
    :ptype data: bytes
    :param reply: optional reply subject; ``None`` for fire-and-forget
    :ptype reply: str | None
    :param subject: concrete subject the message arrived on
    :ptype subject: str
    :return: wrapper-shaped envelope
    :rtype: IncomingMessage
    """
    return IncomingMessage(data=data, reply_subject=reply, subject=subject)


def _make_entry(
    tool_name: str = "threetears.calculator",
    tool_version: str = "1.0.0",
    pod_id: str = "pod-001",
    status: str = "available",
) -> CatalogEntry:
    """create catalog entry with single endpoint for testing.

    :param tool_name: namespaced tool name
    :ptype tool_name: str
    :param tool_version: semver version string
    :ptype tool_version: str
    :param pod_id: pod identifier for endpoint
    :ptype pod_id: str
    :param status: endpoint availability status
    :ptype status: str
    :return: test catalog entry with one endpoint
    :rtype: CatalogEntry
    """
    endpoint = ToolEndpoint(
        pod_id=pod_id,
        status=status,
        in_flight=0,
    )
    result = CatalogEntry(
        tool_name=tool_name,
        tool_version=tool_version,
        full_name=f"{tool_name}@{tool_version}",
        description=f"{tool_name} tool",
        input_schema={"type": "object"},
        endpoints=[endpoint],
    )
    return result


# -- manifest validation tests --


class TestRegistrationHandlerValidation:
    """tests for manifest validation in registration handler."""

    @pytest.mark.asyncio
    async def test_rejects_malformed_json(self) -> None:
        """handler rejects message with invalid JSON payload."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        msg = _make_nats_msg(data=b"not json")
        await handler.handle_registration(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response_data["success"] is False
        assert "malformed" in response_data["error"]

    @pytest.mark.asyncio
    async def test_rejects_empty_pod_id(self) -> None:
        """handler rejects manifest with empty pod_id."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = _make_manifest(pod_id="")
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response_data["success"] is False
        assert "pod_id" in response_data["error"]

    @pytest.mark.asyncio
    async def test_rejects_empty_tools_list(self) -> None:
        """handler rejects manifest with empty tools list."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = RegistrationManifest(pod_id="pod-001", tools=[])
        msg = _make_nats_msg(
            data=manifest.model_dump_json().encode("utf-8"),
        )
        await handler.handle_registration(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response_data["success"] is False
        assert "tools" in response_data["error"]


# -- multi-pod registration tests --


class TestRegistrationHandlerMultiPod:
    """tests for additive multi-pod registration."""

    @pytest.mark.asyncio
    async def test_allows_registration_from_different_pod(self) -> None:
        """handler allows same tool@version from different pod."""
        catalog = ToolCatalog()
        existing = _make_entry(pod_id="pod-OTHER")
        await catalog.register(existing)

        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = _make_manifest(pod_id="pod-NEW")
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response_data["success"] is True
        assert "threetears.calculator@1.0.0" in response_data["registered_tools"]

    @pytest.mark.asyncio
    async def test_allows_reregistration_from_same_pod(self) -> None:
        """handler allows re-registration of tool from same pod."""
        catalog = ToolCatalog()
        existing = _make_entry(pod_id="pod-001")
        await catalog.register(existing)

        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = _make_manifest(pod_id="pod-001")
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response_data["success"] is True
        assert "threetears.calculator@1.0.0" in response_data["registered_tools"]

    @pytest.mark.asyncio
    async def test_second_pod_adds_endpoint(self) -> None:
        """registering from second pod adds endpoint to existing entry."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest_a = _make_manifest(pod_id="pod-A")
        msg_a = _make_nats_msg(data=manifest_a.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg_a)

        manifest_b = _make_manifest(pod_id="pod-B")
        msg_b = _make_nats_msg(data=manifest_b.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg_b)

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        assert len(entry.endpoints) == 2
        pod_ids = {ep.pod_id for ep in entry.endpoints}
        assert pod_ids == {"pod-A", "pod-B"}


# -- successful registration tests --


class TestRegistrationHandlerSuccess:
    """tests for successful tool registration."""

    @pytest.mark.asyncio
    async def test_registers_single_tool(self) -> None:
        """handler registers single tool from manifest."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = _make_manifest()
        msg = _make_nats_msg(data=manifest.model_dump_json().encode("utf-8"))
        await handler.handle_registration(msg)

        nc.publish_reply.assert_called_once()
        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response_data["success"] is True
        assert response_data["pod_id"] == "pod-001"
        assert "threetears.calculator@1.0.0" in response_data["registered_tools"]

        entry = catalog.get("threetears.calculator@1.0.0")
        assert entry is not None
        assert len(entry.endpoints) == 1
        assert entry.endpoints[0].pod_id == "pod-001"
        assert entry.status == "available"

    @pytest.mark.asyncio
    async def test_registers_multiple_tools(self) -> None:
        """handler registers all tools from manifest atomically."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
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

        response_data = json.loads(nc.publish_reply.call_args.kwargs["message"].model_dump_json())
        assert response_data["success"] is True
        assert len(response_data["registered_tools"]) == 2
        assert catalog.get("threetears.calculator@1.0.0") is not None
        assert catalog.get("threetears.dictionary@1.0.0") is not None

    @pytest.mark.asyncio
    async def test_no_reply_when_no_reply_subject(self) -> None:
        """handler does not publish response when no reply subject."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)

        manifest = _make_manifest()
        msg = _make_nats_msg(
            data=manifest.model_dump_json().encode("utf-8"),
            reply=None,
        )
        await handler.handle_registration(msg)

        nc.publish_reply.assert_not_called()
        assert catalog.get("threetears.calculator@1.0.0") is not None


# -- lifecycle tests --


class TestRegistrationHandlerLifecycle:
    """tests for handler start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_subscribes_to_register_subject(self) -> None:
        """start subscribes to {namespace}.tools.register."""
        set_default_namespace("myns")
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="myns")
        nc = _make_registry_nc()
        await handler.start(nc)
        nc.subscribe.assert_called_once()
        # wrapper subscribe is kw-only with typed Subject
        subject_arg = nc.subscribe.call_args.kwargs["subject"]
        assert subject_arg.path == "myns.tools.register"

    @pytest.mark.asyncio
    async def test_stop_unsubscribes(self) -> None:
        """stop unsubscribes from registration subject through the wrapper."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        mock_sub = MagicMock()
        nc.subscribe = AsyncMock(return_value=mock_sub)
        await handler.start(nc)
        await handler.stop()
        # wrapper exposes ``nc.unsubscribe(sub)``; the subscription
        # handle itself is opaque (no public ``.unsubscribe`` method).
        nc.unsubscribe.assert_called_once_with(mock_sub)


# -- wire format tests --


class TestRegistrationResponse:
    """tests for RegistrationResponse model."""

    def test_success_response_serialization(self) -> None:
        """RegistrationResponse serializes success correctly."""
        resp = RegistrationResponse(
            success=True,
            pod_id="pod-001",
            registered_tools=["tool.a@1.0", "tool.b@2.0"],
        )
        data = json.loads(resp.model_dump_json())
        assert data["success"] is True
        assert data["pod_id"] == "pod-001"
        assert len(data["registered_tools"]) == 2
        assert data["error"] is None

    def test_error_response_serialization(self) -> None:
        """RegistrationResponse serializes error correctly."""
        resp = RegistrationResponse(
            success=False,
            pod_id="pod-fail",
            error="conflict detected",
        )
        data = json.loads(resp.model_dump_json())
        assert data["success"] is False
        assert data["error"] == "conflict detected"
        assert data["registered_tools"] == []


# -- per-key-identity authenticator tests (raw-token verify) --


class _RecordingAuthenticator:
    """a ``ToolPodAuthenticator`` double that records the RAW token it was handed.

    admits exactly ``expected_token`` (proving the registry passes the token through UN-hashed --
    a sha256 digest would never match), returning an auth ctx that OWNS ``owned_namespaces``.

    ``provider_nodes`` reports the ownership graph. It defaults to this pod's own
    nodes, rooted, because a pod's nodes are rows in that graph and the two must
    not be able to disagree; a test that needs a node owned by SOMEBODY ELSE
    passes ``other_nodes``.
    """

    def __init__(
        self,
        expected_token: str,
        owned_namespaces: list[str],
        other_nodes: list[str] | None = None,
    ) -> None:
        self._expected = expected_token
        self._owned = owned_namespaces
        self._other = other_nodes or []
        self.seen_tokens: list[str] = []

    async def verify_pod(self, token: str) -> "ToolPodAuth | None":
        self.seen_tokens.append(token)
        result: ToolPodAuth | None = None
        if token == self._expected:
            result = ToolPodAuth(
                pod_entity_id="pod-001",
                name="recording-pod",
                owned_namespaces=self._owned,
            )
        return result

    async def provider_nodes(self) -> tuple[str, ...]:
        from threetears.core.namespaces import build_tool_provider_node_name

        nodes: list[str] = []
        for stem in [*self._owned, *self._other]:
            try:
                nodes.append(build_tool_provider_node_name(stem))
            except ValueError:
                continue
        return tuple(nodes)


def _manifest_with_token(token: str | None, tools: list[dict[str, Any]] | None = None) -> RegistrationManifest:
    """build a manifest carrying ``token`` as its bootstrap_token (the self-minted JWT slot)."""
    base = _make_manifest(tools=tools)
    return RegistrationManifest(pod_id=base.pod_id, tools=base.tools, bootstrap_token=token)


class TestRegistrationHandlerAuthenticator:
    """the authenticator receives the RAW manifest token and gates + filters registration."""

    @pytest.mark.asyncio
    async def test_raw_token_admitted_and_registered(self) -> None:
        """a token the authenticator accepts registers the tools; the RAW token reaches verify_pod."""
        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=["threetears"])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        manifest = _manifest_with_token("the-jwt")
        msg = _make_nats_msg(manifest.model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        # verify_pod saw the RAW token, not a hash of it
        assert auth.seen_tokens == ["the-jwt"]
        reply = nc.publish_reply.call_args.kwargs["message"]
        assert isinstance(reply, RegistrationResponse)
        assert reply.success is True
        assert reply.registered_tools == ["threetears.calculator@1.0.0"]

    @pytest.mark.asyncio
    async def test_invalid_token_denied(self) -> None:
        """a token the authenticator rejects fails registration with an auth error."""
        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=["threetears"])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        manifest = _manifest_with_token("a-forged-token")
        msg = _make_nats_msg(manifest.model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        assert auth.seen_tokens == ["a-forged-token"]
        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is False
        assert reply.error == "invalid bootstrap token"
        assert catalog.get("threetears.calculator@1.0.0") is None

    @pytest.mark.asyncio
    async def test_tokenless_manifest_is_admitted_outside_every_owned_node(self) -> None:
        """the agent-owned in-process pod: still admitted, still never verified.

        It registers over its agent's own NATS connection, which the auth-callout
        already authenticated per-key, so it presents no token and never reaches
        the verifier. What it offers here sits under no provider node anybody
        owns, which is the ordinary case for an agent's own tools.
        """
        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=["threetears"])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        tools = [
            {
                "name": "myagent.summarize",
                "version": "1.0.0",
                "description": "an agent's own tool",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
        manifest = _manifest_with_token(None, tools=tools)
        msg = _make_nats_msg(manifest.model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        assert auth.seen_tokens == []  # tokenless -> never reached the verifier
        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is True
        assert reply.registered_tools == ["myagent.summarize@1.0.0"]

    @pytest.mark.asyncio
    async def test_tokenless_manifest_is_filtered_not_exempt(self) -> None:
        """the path that used to return before any filtering ran.

        A tokenless pod owns no provider node, so a name inside somebody else's
        node is refused -- the whole point of this change. Paired above with the
        name that IS admitted, so this is a filter rather than a blanket refusal.
        """
        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=["threetears"])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        manifest = _manifest_with_token(None)
        msg = _make_nats_msg(manifest.model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        assert auth.seen_tokens == []  # still never verified: it holds no row to verify against
        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is False
        assert "threetears.calculator" in reply.error
        assert catalog.get("threetears.calculator@1.0.0") is None

    @pytest.mark.asyncio
    async def test_tools_filtered_to_owned_namespaces(self) -> None:
        """tools outside the pod's allowed namespaces are dropped; in-namespace tools survive."""
        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=["threetears"])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        tools = [
            {
                "name": "threetears.calculator",
                "version": "1.0.0",
                "description": "allowed",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "acme.secret",
                "version": "1.0.0",
                "description": "outside the allow-list",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
        manifest = _manifest_with_token("the-jwt", tools=tools)
        msg = _make_nats_msg(manifest.model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is True
        assert reply.registered_tools == ["threetears.calculator@1.0.0"]
        assert catalog.get("acme.secret@1.0.0") is None

    @pytest.mark.asyncio
    async def test_a_namespace_does_not_admit_a_prefix_sibling(self) -> None:
        """``threetears`` admits its own children and NOT ``threetearsimposter``.

        the pod's allow-list is a set of name NODES, compared on a
        segment boundary. a raw prefix test admits any name that merely
        begins with the same characters, which is why every value in
        this column used to be written with a trailing dot -- a
        value-level workaround this compares its way past.
        """
        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=["threetears"])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        tools = [
            {
                "name": "threetears.calculator",
                "version": "1.0.0",
                "description": "a real child of the granted node",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "threetearsimposter.exfiltrate",
                "version": "1.0.0",
                "description": "shares the node's characters, not its segment",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
        manifest = _manifest_with_token("the-jwt", tools=tools)
        msg = _make_nats_msg(manifest.model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is True
        assert reply.registered_tools == ["threetears.calculator@1.0.0"]
        assert catalog.get("threetearsimposter.exfiltrate@1.0.0") is None

    @pytest.mark.asyncio
    async def test_the_node_itself_is_admitted(self) -> None:
        """a tool named exactly the granted node registers.

        the containment rule counts a node as containing itself, so a
        pod granted ``acme`` may serve a tool literally called ``acme``.
        """
        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=["acme"])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        tools = [
            {
                "name": "acme",
                "version": "1.0.0",
                "description": "the node itself",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
        manifest = _manifest_with_token("the-jwt", tools=tools)
        msg = _make_nats_msg(manifest.model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is True
        assert reply.registered_tools == ["acme@1.0.0"]


class TestRefusingEveryToolSaysWhichNodeItComparedAgainst:
    """a pod that registers nothing is told what was compared, not just that it failed.

    the refusal reaches the pod author as ``RegistrationResponse.error``,
    and it used to read ``no tools authorized for this pod's namespaces``
    -- true, and indistinguishable from a missing RBAC grant. the two
    values that decide the outcome are the pod's own allow-list nodes and
    the names it offered, so both belong in the sentence.
    """

    @pytest.mark.asyncio
    async def test_a_trailing_separator_node_is_named_in_the_refusal(self) -> None:
        """``evd.`` matches nothing, and the refusal quotes it back."""
        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=["evd."])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        tools = [
            {
                "name": "evd.hello",
                "version": "1.0",
                "description": "a tool under a node written with a trailing dot",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
        manifest = _manifest_with_token("the-jwt", tools=tools)
        msg = _make_nats_msg(manifest.model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is False
        assert "evd." in reply.error
        assert "evd.hello" in reply.error

    @pytest.mark.asyncio
    async def test_a_glob_shaped_node_is_named_in_the_refusal(self) -> None:
        """``evd.*`` is not a node; the refusal says which value failed."""
        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=["evd.*"])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        tools = [
            {
                "name": "evd.hello",
                "version": "1.0",
                "description": "a tool under a node written as a glob",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
        manifest = _manifest_with_token("the-jwt", tools=tools)
        msg = _make_nats_msg(manifest.model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is False
        assert "evd.*" in reply.error
        assert "evd.hello" in reply.error


class TestTheManifestFilterComparesTheMcpName:
    """the manifest filter reads ``tool.name``, never a namespace name.

    the two live in the same conceptual space and are easy to confuse.
    a pod's declaration holds bare mcp-name NODES (``pentest``,
    ``aibots.admin``, ``threetears``); ``tool.name`` is the mcp name a
    pod offers (``pentest.sqlmap``). The canonical
    ``namespaces.name`` -- ``tools.pentest.sqlmap.1-0`` -- is
    a THIRD string, built downstream, and it never reaches this filter.

    That distinction is what decides whether rooting the namespace name
    at ``tools.`` starves registration. It does not: a rooted namespace
    name would match none of these nodes, but the filter is not handed
    one. These tests pin that, because the failure mode if it were
    wrong is total -- ``_authenticate_and_filter`` rejects the WHOLE
    manifest, so the builtin, pentest and admin tool servers would
    register nothing at all, and the only signal is a log line on a
    pod that then sits there healthy and empty.
    """

    @pytest.mark.asyncio
    async def test_a_rooted_namespace_name_is_not_what_the_filter_compares(self) -> None:
        # the A/B: the mcp name is admitted, and the namespace name
        # built FROM it is not. if the filter ever started reading the
        # namespace name, the first assertion would fail and this one
        # would be the reason why.
        from threetears.core.namespaces import build_tool_namespace_name

        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=["pentest"])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        tools = [
            {
                "name": "pentest.sqlmap",
                "version": "1.0",
                "description": "offered under its mcp name",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
        manifest = _manifest_with_token("the-jwt", tools=tools)
        msg = _make_nats_msg(manifest.model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is True
        assert reply.registered_tools == ["pentest.sqlmap@1.0"]
        # and the namespace name this tool will be given downstream is
        # NOT a value any node in the allow-list contains.
        namespace_name = build_tool_namespace_name("pentest.sqlmap", "1.0")
        assert namespace_name == "tools.pentest.sqlmap.1-0"
        assert not namespace_name.startswith("pentest")

    @pytest.mark.asyncio
    async def test_a_manifest_offering_a_rooted_name_is_refused(self) -> None:
        # the inverse guard: a pod that mistakenly offered the built
        # namespace name as its tool name must NOT be admitted by a
        # node naming the provider, or the registry would route a call
        # to a name no dispatcher resolves.
        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=["pentest"])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        tools = [
            {
                "name": "tools.pentest.sqlmap",
                "version": "1.0",
                "description": "a namespace name offered where an mcp name belongs",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]
        manifest = _manifest_with_token("the-jwt", tools=tools)
        msg = _make_nats_msg(manifest.model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is False
        assert "tools.pentest.sqlmap" in reply.error

    @pytest.mark.asyncio
    async def test_every_live_pod_allow_list_still_admits_its_own_tools(self) -> None:
        """the four platform pods' live declarations, exercised.

        read off a running deployment. ``dipp`` is included in the
        trailing-dot spelling it actually carries there, and it admits
        NOTHING -- that is the earlier landing's deliberate visible
        failure, asserted here so this landing cannot be blamed for it.
        """
        live_pods = {
            "admin-tool-server": (["aibots.admin"], "aibots.admin.list_agents", True),
            "builtin-tool-server": (["threetears"], "threetears.calculator", True),
            "pentest-tool-server": (["pentest"], "pentest.sqlmap", True),
            "dipp-tool-server": (["dipp."], "dipp.getthing", False),
        }
        for pod_name, (allowed, offered, expected) in live_pods.items():
            catalog = ToolCatalog()
            auth = _RecordingAuthenticator("the-jwt", owned_namespaces=allowed)
            handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
            nc = _make_registry_nc()
            await handler.start(nc)
            tools = [
                {
                    "name": offered,
                    "version": "1.0",
                    "description": pod_name,
                    "input_schema": {"type": "object", "properties": {}},
                },
            ]
            manifest = _manifest_with_token("the-jwt", tools=tools)
            msg = _make_nats_msg(manifest.model_dump_json().encode("utf-8"))

            await handler.handle_registration(msg)

            reply = nc.publish_reply.call_args.kwargs["message"]
            assert reply.success is expected, f"{pod_name} offering {offered} under {allowed}"


class TestNoRegistrationPathIsUnfiltered:
    """the third path -- a handler built with no authenticator at all.

    It used to ``return`` before any filtering, exactly as the tokenless path did.
    It goes through the one rule now; its answer is permissive only because a
    registry with no host to ask holds no ownership data, and that is stated
    rather than special-cased.
    """

    #: the name this pair is argued over. A SINGLE-OWNER provider, deliberately:
    #: ``threetears`` reads as the natural example and is the one stem that must
    #: never appear here, because it is the FRAMEWORK namespace -- co-hosted by the
    #: shared built-in pod and by every agent's in-process ToolServer -- and using
    #: it as an ownable example is what produced a landing that refused every
    #: agent's builtins. See
    #: ``test_the_framework_namespace_has_no_single_owner.py``.
    _TOOL = [
        {
            "name": "pentest.sqlmap",
            "version": "1.0.0",
            "description": "sqlmap tool",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]

    @pytest.mark.asyncio
    async def test_open_mode_admits_because_the_graph_is_empty(self) -> None:
        """no authenticator, no graph, nothing to enforce -- and no exemption either."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test")
        nc = _make_registry_nc()
        await handler.start(nc)
        msg = _make_nats_msg(_make_manifest(tools=self._TOOL).model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is True
        assert reply.registered_tools == ["pentest.sqlmap@1.0.0"]

    @pytest.mark.asyncio
    async def test_the_same_name_is_refused_once_the_graph_names_an_owner(self) -> None:
        """the A/B that shows the previous test is permissive for its stated reason.

        Same manifest, same tool name. The only difference is that a host is
        present and reports a provider node containing the name, which the
        registering pod does not own.
        """
        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=[], other_nodes=["pentest"])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        msg = _make_nats_msg(_manifest_with_token(None, tools=self._TOOL).model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is False
        assert catalog.get("pentest.sqlmap@1.0.0") is None


class TestAnUnreadableOwnershipGraphRefuses:
    """a read that FAILS must not look like a graph with nothing in it.

    An empty inventory admits every unbound pod. A host that cannot answer
    therefore refuses the registration, which the pod retries on its next
    heartbeat, rather than admitting a manifest nobody filtered.
    """

    class _BrokenDirectory:
        """an authenticator whose graph read raises."""

        async def verify_pod(self, token: str) -> "ToolPodAuth | None":
            del token
            return None

        async def provider_nodes(self) -> tuple[str, ...]:
            raise RuntimeError("the broker is down")

    @pytest.mark.asyncio
    async def test_a_failing_graph_read_refuses_the_registration(self) -> None:
        """refused, and nothing written to the catalog."""
        catalog = ToolCatalog()
        handler = RegistrationHandler(catalog, namespace="test", authenticator=self._BrokenDirectory())
        nc = _make_registry_nc()
        await handler.start(nc)
        msg = _make_nats_msg(_manifest_with_token(None).model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is False
        assert "ownership graph unavailable" in reply.error
        assert catalog.get("threetears.calculator@1.0.0") is None

    @pytest.mark.asyncio
    async def test_a_working_graph_read_admits_the_same_manifest(self) -> None:
        """the admitted twin: the refusal above is the failure, not the manifest."""
        catalog = ToolCatalog()
        auth = _RecordingAuthenticator("the-jwt", owned_namespaces=[])
        handler = RegistrationHandler(catalog, namespace="test", authenticator=auth)
        nc = _make_registry_nc()
        await handler.start(nc)
        msg = _make_nats_msg(_manifest_with_token(None).model_dump_json().encode("utf-8"))

        await handler.handle_registration(msg)

        reply = nc.publish_reply.call_args.kwargs["message"]
        assert reply.success is True
