"""unit tests for the inbound REST affordance declaration, on its own.

the declaration and its intra-tool coherence checks that need no tool: the
closed method vocabulary, the path / query / body split, the cache posture and
the short-lived pointer. the checks that need a tool or a ``ToolServer``
(registration coherence, the manifest a pod publishes, the outbound
descriptor it shares a base with) are agent-tools' ``test_rest_affordance``.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import pytest

from threetears.core.http_cache import CacheClass
from threetears.core.tool_wire import (
    QUERY_METHODS,
    HttpMethod,
    ParameterBinding,
    RestAffordance,
    RestAffordanceError,
    ToolManifestEntry,
)

_SURVEY_PROPERTIES: dict[str, Any] = {
    "survey_id": {"type": "string"},
    "page": {"type": "integer"},
    "include_drafts": {"type": "boolean"},
}

#: a coherent GET declaration: one path param, two query params.
_READ = RestAffordance(method="GET", path_template="/surveys/{survey_id}")

#: a coherent POST declaration: one path param, two body properties.
_WRITE = RestAffordance(method="POST", path_template="/surveys/{survey_id}")


class TestInboundDeclarationShape:
    """the inbound declaration carries only what an inbound face needs."""

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
        binding = _READ.bind({"type": "object", "properties": _SURVEY_PROPERTIES})
        assert binding == ParameterBinding(
            path=frozenset({"survey_id"}),
            query=frozenset({"page", "include_drafts"}),
            body=frozenset(),
        )

    def test_post_binds_the_remainder_to_the_body(self) -> None:
        """non-path properties of a POST become body properties."""
        binding = _WRITE.bind({"type": "object", "properties": _SURVEY_PROPERTIES})
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


class TestShortLivedPointer:
    """an unversioned shared read held for seconds: the pointer a client reads to find the version.

    A versioned address is immutable; the small document naming the current version cannot be, so
    it is held only briefly, and a client reads it again on its next poll.
    """

    def test_default_is_no_max_age(self) -> None:
        """a declaration is short-lived only when it says so."""
        assert RestAffordance(method="GET", path_template="/x").cache_max_age is None

    @pytest.mark.parametrize("cache", [CacheClass.INHERIT, CacheClass.AUTHENTICATED, CacheClass.PUBLIC])
    def test_a_shared_unversioned_read_may_be_short_lived(self, cache: CacheClass) -> None:
        """seconds, on a read whose class may reach a shared cache.

        :param cache: the declared class
        :ptype cache: CacheClass
        :return: nothing
        :rtype: None
        """
        affordance = RestAffordance(method="GET", path_template="/x/{scope}/index", cache=cache, cache_max_age=5)
        assert affordance.cache_max_age == 5

    @pytest.mark.parametrize(
        ("inherited", "expected"),
        [(CacheClass.PRIVATE, None), (CacheClass.AUTHENTICATED, 5), (CacheClass.PUBLIC, 5)],
    )
    def test_the_effective_max_age_follows_the_effective_class(
        self, inherited: CacheClass, expected: int | None
    ) -> None:
        """a pointer whose resource resolves origin-only is held nowhere, whatever it declares.

        :param inherited: the resource's own class
        :ptype inherited: CacheClass
        :param expected: the max age a serving side may render
        :ptype expected: int | None
        :return: nothing
        :rtype: None
        """
        affordance = RestAffordance(method="GET", path_template="/x/{scope}/index", cache_max_age=5)
        assert affordance.resolve_cache_max_age(inherited) == expected

    def test_a_versioned_read_is_never_short_lived(self) -> None:
        """an address carrying its version is immutable; a max age on it contradicts that."""
        with pytest.raises(RestAffordanceError, match="immutable"):
            RestAffordance(
                method="GET",
                path_template="/x/{version}",
                cache=CacheClass.AUTHENTICATED,
                cache_version_param="version",
                cache_max_age=5,
            )

    def test_a_private_read_has_no_max_age(self) -> None:
        """origin-only is held nowhere, so a max age on it says something false."""
        with pytest.raises(RestAffordanceError, match="private"):
            RestAffordance(method="GET", path_template="/x", cache=CacheClass.PRIVATE, cache_max_age=5)

    def test_a_write_has_no_max_age(self) -> None:
        """only a verb a shared cache may hold can be held for seconds."""
        with pytest.raises(RestAffordanceError, match="POST"):
            RestAffordance(method="POST", path_template="/x", cache_max_age=5)

    @pytest.mark.parametrize("seconds", [0, -1, 61, 3600])
    def test_the_max_age_is_seconds_not_minutes(self, seconds: int) -> None:
        """from one second to a minute: longer is the unversioned copy a purge would have to fix.

        :param seconds: the declared max age
        :ptype seconds: int
        :return: nothing
        :rtype: None
        """
        with pytest.raises(RestAffordanceError, match="cache_max_age"):
            RestAffordance(method="GET", path_template="/x", cache_max_age=seconds)

    def test_a_declaration_from_before_the_field_reads_as_none(self) -> None:
        """a manifest an older pod sent, with no max age, still validates: the field is additive."""
        entry = {
            "name": "test.old",
            "version": "1.0",
            "description": "",
            "input_schema": {"type": "object", "properties": {}},
            "face_rest": {"method": "GET", "path_template": "/x", "cache": "inherit", "cache_version_param": None},
        }
        declaration = ToolManifestEntry.model_validate(entry).face_rest
        assert declaration is not None
        assert declaration.cache_max_age is None


class TestScopeNode:
    """the namespace node whose child tool nodes name the scopes a reader may read at the edge."""

    def test_none_by_default(self) -> None:
        assert RestAffordance(method="GET", path_template="/x").scope_node is None

    @pytest.mark.parametrize("node", ["", ".", "tools..enr", "tools.enr.", " tools.enr", "tools. enr"])
    def test_a_node_that_is_not_a_namespace_name_is_refused(self, node: str) -> None:
        with pytest.raises(RestAffordanceError, match="scope_node"):
            RestAffordance(method="GET", path_template="/x/{customer}/{scope}/index", cache_max_age=5, scope_node=node)
