"""unit tests for the inbound REST affordance declaration.

the REST face is the fourth reach a tool may declare. unlike
``face_platform_tool`` / ``face_api`` / ``face_mcp`` it cannot be a bare
boolean: an inbound HTTP address needs a method, a path template, a
binding from URL positions to schema properties, and a cacheability
posture. this module covers the declaration and its intra-tool coherence
checks ONLY -- nothing here serves anything, and nothing here routes.
"""

from __future__ import annotations

from dataclasses import fields
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
    QUERY_METHODS,
    HttpMethod,
    ParameterBinding,
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


class _RestWriteTool(_StubTool):
    """a coherent POST declaration: one path param, two body properties."""

    face_rest = RestAffordance(method="POST", path_template="/surveys/{survey_id}")

    def __init__(self, *, name: str = "test.rest_write") -> None:
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


def _server() -> ToolServer:
    """build a ToolServer that never opens a connection.

    :return: tool server
    :rtype: ToolServer
    """
    return ToolServer(
        agent_id=uuid7(),
        customer_id=uuid7(),
        namespace_collection=None,
        nats_url="nats://test:4222",
    )


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

    def test_inbound_carries_no_upstream_credential(self) -> None:
        """an inbound face is authorized per caller, not by an upstream secret."""
        field_names = {f.name for f in fields(RestAffordance)}
        assert "credentials_ref" not in field_names

    def test_inbound_does_not_restate_the_tool_schema(self) -> None:
        """``param_schema`` is the tool's own ``mcp_schema()``; restating it drifts."""
        field_names = {f.name for f in fields(RestAffordance)}
        assert "param_schema" not in field_names


class TestClosedMethodVocabulary:
    """the method is a closed vocabulary, validated at declaration time."""

    def test_vocabulary_covers_every_query_method(self) -> None:
        """a five-verb list would make half of ``QUERY_METHODS`` unreachable."""
        assert QUERY_METHODS <= {member.value for member in HttpMethod}
        assert {member.value for member in HttpMethod} == {
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "HEAD",
            "OPTIONS",
        }

    def test_query_methods_are_the_body_less_four(self) -> None:
        """the binding default reuses the outbound constant unchanged."""
        assert QUERY_METHODS == frozenset({"GET", "DELETE", "HEAD", "OPTIONS"})

    def test_method_outside_the_vocabulary_is_refused(self) -> None:
        """an unknown verb fails at construction, naming the verb."""
        with pytest.raises(RestAffordanceError, match="BREW"):
            RestAffordance(method="BREW", path_template="/pots/{pot_id}")

    def test_method_is_normalized_to_upper_case(self) -> None:
        """a lower-case declaration is the same declaration."""
        assert RestAffordance(method="get", path_template="/x").method == "GET"


class TestParameterBinding:
    """path AND query, because a GET has nowhere else to put its arguments."""

    def test_get_binds_the_remainder_to_query(self) -> None:
        """non-path properties of a GET become query parameters."""
        binding = _RestReadTool.face_rest.bind({"type": "object", "properties": _SURVEY_PROPERTIES})
        assert binding == ParameterBinding(
            path=frozenset({"survey_id"}),
            query=frozenset({"page", "include_drafts"}),
            body=frozenset(),
        )

    def test_post_binds_the_remainder_to_the_body(self) -> None:
        """non-path properties of a POST become body properties."""
        binding = _RestWriteTool.face_rest.bind({"type": "object", "properties": _SURVEY_PROPERTIES})
        assert binding == ParameterBinding(
            path=frozenset({"survey_id"}),
            query=frozenset(),
            body=frozenset({"page", "include_drafts"}),
        )

    def test_every_query_method_binds_to_query(self) -> None:
        """DELETE / HEAD / OPTIONS follow GET, per ``QUERY_METHODS``."""
        for method in ("DELETE", "HEAD", "OPTIONS"):
            affordance = RestAffordance(method=method, path_template="/surveys/{survey_id}")
            binding = affordance.bind({"type": "object", "properties": _SURVEY_PROPERTIES})
            assert binding.query == frozenset({"page", "include_drafts"})
            assert binding.body == frozenset()

    def test_every_body_method_binds_to_body(self) -> None:
        """PUT / PATCH follow POST."""
        for method in ("PUT", "PATCH"):
            affordance = RestAffordance(method=method, path_template="/surveys/{survey_id}")
            binding = affordance.bind({"type": "object", "properties": _SURVEY_PROPERTIES})
            assert binding.body == frozenset({"page", "include_drafts"})
            assert binding.query == frozenset()

    def test_a_schema_with_no_properties_binds_nothing(self) -> None:
        """an argument-free tool is still a legal REST resource."""
        affordance = RestAffordance(method="GET", path_template="/health")
        binding = affordance.bind({"type": "object", "properties": {}})
        assert binding == ParameterBinding(path=frozenset(), query=frozenset(), body=frozenset())


class TestCacheability:
    """derived-and-narrowable, never declared outright."""

    def test_default_is_inherit(self) -> None:
        """a declaration says nothing about cacheability unless it narrows."""
        assert RestAffordance(method="GET", path_template="/x").cache is CacheClass.INHERIT

    def test_resolution_cannot_widen_past_the_resource(self) -> None:
        """declaring PUBLIC over a PRIVATE resource resolves to PRIVATE."""
        affordance = RestAffordance(method="GET", path_template="/x", cache=CacheClass.PUBLIC)
        assert affordance.resolve_cache_class(CacheClass.PRIVATE) is CacheClass.PRIVATE

    def test_resolution_honours_a_narrowing_declaration(self) -> None:
        """declaring PRIVATE over a PUBLIC resource resolves to PRIVATE."""
        affordance = RestAffordance(method="GET", path_template="/x", cache=CacheClass.PRIVATE)
        assert affordance.resolve_cache_class(CacheClass.PUBLIC) is CacheClass.PRIVATE

    def test_inherit_takes_the_resource_class(self) -> None:
        """the default resolves to whatever the resource already is."""
        affordance = RestAffordance(method="GET", path_template="/x")
        assert affordance.resolve_cache_class(CacheClass.AUTHENTICATED) is CacheClass.AUTHENTICATED

    def test_shared_cacheability_on_an_unsafe_method_is_refused(self) -> None:
        """only GET and HEAD have a shared-cache story; a POST has none."""
        with pytest.raises(RestAffordanceError, match="POST"):
            RestAffordance(method="POST", path_template="/x", cache=CacheClass.PUBLIC)

    def test_head_may_declare_shared_cacheability(self) -> None:
        """HEAD is cacheable by definition and shares GET's cache entry."""
        affordance = RestAffordance(method="HEAD", path_template="/x", cache=CacheClass.PUBLIC)
        assert affordance.cache is CacheClass.PUBLIC

    def test_private_on_an_unsafe_method_is_fine(self) -> None:
        """narrowing to origin-only is legal on any method."""
        assert RestAffordance(method="POST", path_template="/x", cache=CacheClass.PRIVATE).cache is CacheClass.PRIVATE

    def test_version_param_must_ride_in_the_path(self) -> None:
        """a shared cache keys on URL; a version outside the URL cannot key it."""
        with pytest.raises(RestAffordanceError, match="build_version"):
            RestAffordance(
                method="GET",
                path_template="/surveys/{survey_id}",
                cache=CacheClass.PUBLIC,
                cache_version_param="build_version",
            )

    def test_version_param_in_the_path_is_accepted(self) -> None:
        """version-in-the-address makes a rebuild a new address, not a purge."""
        affordance = RestAffordance(
            method="GET",
            path_template="/surveys/{survey_id}/v{build_version}",
            cache=CacheClass.PUBLIC,
            cache_version_param="build_version",
        )
        assert affordance.cache_version_param == "build_version"


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
        assert len(server._tools) == 1  # noqa: SLF001


class TestManifestCarriesTheDeclaration:
    """the existing registration manifest carries it -- no second channel."""

    async def test_manifest_entry_carries_the_declaration(self) -> None:
        """``publish_registration`` stamps ``face_rest`` onto the entry."""
        server = _server()
        server.register(_RestReadTool())
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        assert isinstance(manifest, RegistrationManifest)
        entry = manifest.tools[0]
        assert entry.face_rest == _RestReadTool.face_rest

    async def test_manifest_entry_defaults_to_no_declaration(self) -> None:
        """a tool with no REST face lands as ``None``, not as a stub object."""
        server = _server()
        server.register(_StubTool())
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
        await server.publish_registration()
        manifest = mock_nc.publish.await_args.kwargs["message"]
        assert manifest.tools[0].face_rest is None

    async def test_declaration_round_trips_through_json_without_loss(self) -> None:
        """method, template, derived placeholders, cache posture all survive."""
        server = _server()
        server.register(_RestReadTool(name="test.rest_roundtrip"))
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
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

        server = _server()
        server.register(_PrivateReadTool())
        mock_nc = AsyncMock()
        server._nc = mock_nc  # noqa: SLF001
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
