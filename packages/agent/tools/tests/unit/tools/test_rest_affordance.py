"""unit tests for the inbound REST affordance declaration.

the REST face is the fourth reach a tool may declare. unlike
``face_platform_tool`` / ``face_api`` / ``face_mcp`` it cannot be a bare
boolean: an inbound HTTP address needs a method, a path template, a
binding from URL positions to schema properties, and a cacheability
posture. this module covers the declaration and its intra-tool coherence
checks ONLY -- nothing here serves anything, and nothing here routes. the
checks that need no tool (the method vocabulary, the parameter binding, the
cache posture) are core's ``test_tool_wire_rest``, beside the declaration.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid7

import pytest

from threetears.agent.tools.base_tool import (
    MCPToolDefinition,
    TearsTool,
    ToolResult,
)
from threetears.agent.tools.builtin.http_api_tool import HttpOperationDescriptor
from threetears.agent.tools.http_operation import (
    PathTemplateBinding,
    RestAffordance,
    RestAffordanceError,
)
from threetears.agent.tools.server import RegistrationManifest, ToolServer
from threetears.core.http_cache import CacheClass


class _StubTool(TearsTool):
    """minimal concrete TearsTool whose input schema the tests control."""

    def __init__(
        self,
        *,
        name: str = "test.rest_stub",
        version: str = "1.0",
        properties: dict[str, Any] | None = None,
    ) -> None:
        """capture the identity + input-schema properties for this stub.

        :param name: namespaced tool name
        :ptype name: str
        :param version: semver string
        :ptype version: str
        :param properties: JSON-schema ``properties`` this tool declares
        :ptype properties: dict[str, Any] | None
        :return: nothing
        :rtype: None
        """
        self._name = name
        self._version = version
        self._properties = properties if properties is not None else {}

    async def execute(self, **kwargs: Any) -> ToolResult:
        """no-op execute body.

        :param kwargs: ignored
        :ptype kwargs: Any
        :return: trivial result
        :rtype: ToolResult
        """
        return ToolResult(success=True, content="")

    def mcp_schema(self) -> MCPToolDefinition:
        """return the stub's schema over the configured properties.

        :return: MCP tool definition
        :rtype: MCPToolDefinition
        """
        return MCPToolDefinition(
            name=self._name,
            version=self._version,
            description="stub tool for REST affordance tests",
            input_schema={"type": "object", "properties": self._properties},
        )

    def mcp_name(self) -> str:
        """return the stub's mcp name.

        :return: name
        :rtype: str
        """
        return self._name

    def mcp_version(self) -> str:
        """return the stub's version.

        :return: version
        :rtype: str
        """
        return self._version


_SURVEY_PROPERTIES: dict[str, Any] = {
    "survey_id": {"type": "string"},
    "page": {"type": "integer"},
    "include_drafts": {"type": "boolean"},
}


class _RestReadTool(_StubTool):
    """a coherent GET declaration: one path param, two query params."""

    face_rest = RestAffordance(method="GET", path_template="/surveys/{survey_id}")

    def __init__(self, *, name: str = "test.rest_read") -> None:
        """build the stub over the survey property set.

        :param name: namespaced tool name
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        super().__init__(name=name, properties=_SURVEY_PROPERTIES)


class _IncoherentTemplateTool(_StubTool):
    """a declaration whose template names a property the tool never declares."""

    face_rest = RestAffordance(method="GET", path_template="/surveys/{survey_id}/waves/{wave_id}")

    def __init__(self, *, name: str = "test.rest_incoherent") -> None:
        """build the stub over the survey property set.

        :param name: namespaced tool name
        :ptype name: str
        :return: nothing
        :rtype: None
        """
        super().__init__(name=name, properties=_SURVEY_PROPERTIES)


def _server(nats_client: AsyncMock | None = None) -> ToolServer:
    """build a ToolServer that never opens a connection.

    :param nats_client: the NATS double to inject, or ``None`` for a server that is never connected
    :ptype nats_client: AsyncMock | None
    :return: tool server
    :rtype: ToolServer
    """
    server: ToolServer
    if nats_client is None:
        server = ToolServer(agent_id=uuid7(), customer_id=uuid7(), nats_url="nats://test:4222")
    else:
        server = ToolServer(agent_id=uuid7(), customer_id=uuid7(), nats_client=nats_client)
    return server


class TestSharedStructureWithHttpOperationDescriptor:
    """the inbound declaration is the outbound descriptor's mirror, not a rival."""

    def test_both_derive_from_the_same_path_template_binding(self) -> None:
        """one base owns method + path template + derived placeholders."""
        assert issubclass(RestAffordance, PathTemplateBinding)
        assert issubclass(HttpOperationDescriptor, PathTemplateBinding)

    def test_placeholders_are_derived_identically_on_both_sides(self) -> None:
        """the same template yields the same ``path_params`` either way."""
        inbound = RestAffordance(method="GET", path_template="/a/{x}/b/{y}")
        outbound = HttpOperationDescriptor(
            method="GET",
            path_template="/a/{x}/b/{y}",
            param_schema={},
            credentials_ref=None,
            name="t",
            version="1.0",
            description="d",
        )
        assert inbound.path_params == outbound.path_params == frozenset({"x", "y"})


class TestShortLivedPointer:
    """an unversioned shared read held for seconds: the pointer a client reads to find the version.

    A versioned address is immutable; the small document naming the current version cannot be, so
    it is held only briefly, and a client reads it again on its next poll.
    """

    async def test_the_max_age_round_trips_through_the_manifest(self) -> None:
        """the wire carries it to the hub unchanged."""

        class _PointerTool(_StubTool):
            """a short-lived read."""

            face_rest = RestAffordance(
                method="GET", path_template="/surveys/{survey_id}", cache=CacheClass.AUTHENTICATED, cache_max_age=5
            )

            def __init__(self) -> None:
                """build the stub over the survey property set.

                :return: nothing
                :rtype: None
                """
                super().__init__(name="test.rest_pointer", properties=_SURVEY_PROPERTIES)

        mock_nc = AsyncMock()
        server = _server(mock_nc)
        server.register(_PointerTool())
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        restored = RegistrationManifest.model_validate_json(manifest.model_dump_json())
        declaration = restored.tools[0].face_rest
        assert declaration is not None
        assert declaration.cache_max_age == 5


class TestFaceRestDefaultsOff:
    """every face is surface to defend; this one is off unless authored."""

    def test_base_class_default_is_none(self) -> None:
        """``TearsTool.face_rest`` defaults to ``None``."""
        assert TearsTool.face_rest is None

    def test_subclass_without_override_inherits_none(self) -> None:
        """a tool authored before this shard declares no REST face."""
        assert _StubTool.face_rest is None
        assert _StubTool().face_rest is None

    def test_existing_face_flags_are_untouched_by_a_rest_declaration(self) -> None:
        """declaring REST does not switch on any other reach."""
        assert _RestReadTool.face_platform_tool is True
        assert _RestReadTool.face_api is False
        assert _RestReadTool.face_mcp is False


class TestRegistrationCoherence:
    """intra-tool coherence fails at registration, not at first request."""

    def test_coherent_declaration_registers(self) -> None:
        """a template whose placeholders are all declared properties is fine."""
        server = _server()
        server.register(_RestReadTool())

    def test_template_naming_an_undeclared_property_is_refused(self) -> None:
        """the message names the offending property."""
        server = _server()
        with pytest.raises(RestAffordanceError, match="wave_id"):
            server.register(_IncoherentTemplateTool())

    def test_refusal_names_the_tool(self) -> None:
        """an operator needs to know WHICH tool refused."""
        server = _server()
        with pytest.raises(RestAffordanceError, match="test.rest_incoherent"):
            server.register(_IncoherentTemplateTool())

    def test_tool_without_a_declaration_registers_unchanged(self) -> None:
        """a tool with no REST face is not asked any REST question."""
        server = _server()
        server.register(_StubTool())
        assert server.tools_count == 1


class TestManifestCarriesTheDeclaration:
    """the existing registration manifest carries it -- no second channel."""

    async def test_manifest_entry_carries_the_declaration(self) -> None:
        """``publish_registration`` stamps ``face_rest`` onto the entry."""
        mock_nc = AsyncMock()
        server = _server(mock_nc)
        server.register(_RestReadTool())
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        assert isinstance(manifest, RegistrationManifest)
        entry = manifest.tools[0]
        assert entry.face_rest == _RestReadTool.face_rest

    async def test_manifest_entry_defaults_to_no_declaration(self) -> None:
        """a tool with no REST face lands as ``None``, not as a stub object."""
        mock_nc = AsyncMock()
        server = _server(mock_nc)
        server.register(_StubTool())
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        assert manifest.tools[0].face_rest is None

    async def test_declaration_round_trips_through_json_without_loss(self) -> None:
        """method, template, derived placeholders, cache posture all survive."""
        mock_nc = AsyncMock()
        server = _server(mock_nc)
        server.register(_RestReadTool(name="test.rest_roundtrip"))
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        restored = RegistrationManifest.model_validate_json(manifest.model_dump_json())
        declaration = restored.tools[0].face_rest
        assert declaration is not None
        assert declaration.method == "GET"
        assert declaration.path_template == "/surveys/{survey_id}"
        assert declaration.path_params == frozenset({"survey_id"})
        assert declaration.cache is CacheClass.INHERIT
        assert declaration == _RestReadTool.face_rest

    async def test_narrowed_cache_posture_round_trips(self) -> None:
        """a narrowing declaration is not silently widened by the wire."""

        class _PrivateReadTool(_StubTool):
            """a read narrowed to origin-only."""

            face_rest = RestAffordance(
                method="GET",
                path_template="/surveys/{survey_id}",
                cache=CacheClass.PRIVATE,
            )

            def __init__(self) -> None:
                """build the stub over the survey property set.

                :return: nothing
                :rtype: None
                """
                super().__init__(name="test.rest_private", properties=_SURVEY_PROPERTIES)

        mock_nc = AsyncMock()
        server = _server(mock_nc)
        server.register(_PrivateReadTool())
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        restored = RegistrationManifest.model_validate_json(manifest.model_dump_json())
        declaration = restored.tools[0].face_rest
        assert declaration is not None
        assert declaration.cache is CacheClass.PRIVATE


class TestOutboundToolUnaffected:
    """``HttpApiTool``'s descriptor keeps its shape and its constructor."""

    def test_descriptor_still_accepts_its_seven_authored_fields(self) -> None:
        """the Hub's ``ApiToolPod`` construction site is unchanged."""
        descriptor = HttpOperationDescriptor(
            method="GET",
            path_template="/users/{id}",
            param_schema={"type": "object", "properties": {"id": {"type": "string"}}},
            credentials_ref=None,
            name="example.get_user",
            version="1.0",
            description="get a user by id",
        )
        assert descriptor.method == "GET"
        assert descriptor.path_params == frozenset({"id"})

    def test_descriptor_accepts_any_verb_a_third_party_spec_carries(self) -> None:
        """the OUTBOUND side is not narrowed by the inbound vocabulary."""
        descriptor = HttpOperationDescriptor(
            method="PROPFIND",
            path_template="/dav/{path}",
            param_schema={},
            credentials_ref=None,
            name="dav.propfind",
            version="1.0",
            description="webdav",
        )
        assert descriptor.method == "PROPFIND"


class TestScopeNode:
    """the namespace node whose child tool nodes name the scopes a reader may read at the edge."""

    async def test_the_scope_node_round_trips_through_the_manifest(self) -> None:
        """the wire carries it to the hub unchanged."""

        class _ScopedTool(_StubTool):
            """a short-lived read whose scopes are named under a node."""

            face_rest = RestAffordance(
                method="GET",
                path_template="/surveys/{survey_id}",
                cache=CacheClass.AUTHENTICATED,
                cache_max_age=5,
                scope_node="tools.enr.state",
            )

            def __init__(self) -> None:
                super().__init__(name="test.rest_scoped", properties=_SURVEY_PROPERTIES)

        mock_nc = AsyncMock()
        server = _server(mock_nc)
        server.register(_ScopedTool())
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        wire = manifest.model_dump_json()
        restored = RegistrationManifest.model_validate_json(wire)
        declaration = restored.tools[0].face_rest
        assert declaration is not None
        assert declaration.scope_node == "tools.enr.state"
