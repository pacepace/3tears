"""tests for per-provider structured-output bind kwargs and the dispatcher.

structured output is a provider wire quirk in exactly the same class as
tool-name translation: every provider spells "return json matching this
schema" differently, and nothing above the provider layer should have to
know which spelling applies. these tests pin each provider's emitted
shape plus the dispatcher's fail-loud behavior on an unsupported
provider_type.
"""

from __future__ import annotations

from typing import Any

import pytest
from anthropic import transform_schema

from threetears.models.providers import (
    StructuredOutputError,
    StructuredOutputSchemaError,
    StructuredOutputUnsupportedError,
    structured_output_kwargs,
)
from threetears.models.providers.anthropic import anthropic_structured_output_kwargs
from threetears.models.providers.openai import openai_structured_output_kwargs
from threetears.models.providers.openrouter import openrouter_structured_output_kwargs

_SAMPLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class TestAnthropicStructuredOutputKwargs:
    """tests for ``anthropic_structured_output_kwargs``."""

    def test_emits_output_config_json_schema_format(self) -> None:
        """anthropic emits ``output_config.format`` of type json_schema."""
        kwargs = anthropic_structured_output_kwargs(_SAMPLE_SCHEMA)

        assert set(kwargs) == {"output_config"}
        assert kwargs["output_config"]["format"]["type"] == "json_schema"

    def test_schema_is_passed_through_transform_schema(self) -> None:
        """the bound schema is the ``anthropic.transform_schema`` output.

        a directly-bound ``output_config`` is forwarded verbatim to the
        Anthropic SDK ``messages.create``, so the schema has to be
        pre-transformed (transform_schema enforces the API's json-schema
        limits, e.g. injecting ``additionalProperties: false``).
        """
        expected = transform_schema(_SAMPLE_SCHEMA)

        kwargs = anthropic_structured_output_kwargs(_SAMPLE_SCHEMA)

        assert kwargs["output_config"]["format"]["schema"] == expected
        # non-vacuous: transform_schema must actually have changed the input,
        # otherwise this test would pass against a raw pass-through too.
        assert expected != _SAMPLE_SCHEMA
        assert expected["additionalProperties"] is False

    def test_does_not_emit_openai_or_openrouter_keys(self) -> None:
        """the anthropic shape never leaks ``response_format`` / ``provider``."""
        kwargs = anthropic_structured_output_kwargs(_SAMPLE_SCHEMA, name="answer", strict=True)

        assert "response_format" not in kwargs
        assert "provider" not in kwargs
        assert "extra_body" not in kwargs

    def test_does_not_mutate_caller_schema(self) -> None:
        """the caller's schema dict is not mutated in place."""
        original = dict(_SAMPLE_SCHEMA)

        anthropic_structured_output_kwargs(_SAMPLE_SCHEMA)

        assert _SAMPLE_SCHEMA == original


class TestOpenRouterStructuredOutputKwargs:
    """tests for ``openrouter_structured_output_kwargs``."""

    def test_emits_response_format_and_require_parameters(self) -> None:
        """openrouter emits top-level ``response_format`` + provider routing."""
        kwargs = openrouter_structured_output_kwargs(_SAMPLE_SCHEMA, name="answer", strict=True)

        assert kwargs == {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": _SAMPLE_SCHEMA,
                    "strict": True,
                },
            },
            "provider": {"require_parameters": True},
        }

    def test_strict_false_is_forwarded(self) -> None:
        """``strict=False`` rides through to the json_schema block."""
        kwargs = openrouter_structured_output_kwargs(_SAMPLE_SCHEMA, strict=False)

        assert kwargs["response_format"]["json_schema"]["strict"] is False

    def test_default_name_is_response(self) -> None:
        """the schema name defaults to ``response``."""
        kwargs = openrouter_structured_output_kwargs(_SAMPLE_SCHEMA)

        assert kwargs["response_format"]["json_schema"]["name"] == "response"

    def test_require_parameters_forces_loud_failure(self) -> None:
        """``provider.require_parameters`` is always set.

        it is the only thing that makes an upstream model lacking
        structured-output support reject the request instead of quietly
        returning prose.
        """
        kwargs = openrouter_structured_output_kwargs(_SAMPLE_SCHEMA)

        assert kwargs["provider"]["require_parameters"] is True

    def test_does_not_emit_anthropic_key(self) -> None:
        """the openrouter shape never leaks ``output_config``."""
        kwargs = openrouter_structured_output_kwargs(_SAMPLE_SCHEMA)

        assert "output_config" not in kwargs


class TestOpenAiStructuredOutputKwargs:
    """tests for ``openai_structured_output_kwargs``."""

    def test_response_format_rides_extra_body_not_top_level(self) -> None:
        """the directive rides ``extra_body``, never a top-level key.

        langchain_openai's ``_generate`` diverts to
        ``chat.completions.parse()`` and ``_stream`` pops ``stream`` and
        switches to ``beta.chat.completions.stream()`` whenever
        ``response_format`` is present in the request payload -- both of
        which break the gateway (it needs a raw ``AIMessage`` and a
        working token stream). ``extra_body`` reaches the same wire field
        while keeping the plain ``.create()`` path.
        """
        kwargs = openai_structured_output_kwargs(_SAMPLE_SCHEMA, name="answer", strict=True)

        assert set(kwargs) == {"extra_body"}
        assert "response_format" not in kwargs
        assert kwargs["extra_body"] == {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": _SAMPLE_SCHEMA,
                    "strict": True,
                },
            },
        }

    def test_does_not_emit_openrouter_provider_block(self) -> None:
        """``provider`` routing is OpenRouter-only and must not appear."""
        kwargs = openai_structured_output_kwargs(_SAMPLE_SCHEMA)

        assert "provider" not in kwargs
        assert "provider" not in kwargs["extra_body"]

    def test_does_not_emit_anthropic_key(self) -> None:
        """the openai shape never leaks ``output_config``."""
        kwargs = openai_structured_output_kwargs(_SAMPLE_SCHEMA)

        assert "output_config" not in kwargs

    def test_kwargs_survive_a_langchain_openai_payload_build(self) -> None:
        """the emitted kwargs keep ChatOpenAI on the plain ``.create()`` path.

        behavioral counterpart to the shape assertions above: bind the
        kwargs onto a real ``ChatOpenAI`` and confirm the request payload
        carries no top-level ``response_format`` (the key langchain
        branches on) while still carrying the directive.
        """
        from langchain_core.messages import HumanMessage
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model="gpt-4o", api_key="sk-test")
        kwargs = openai_structured_output_kwargs(_SAMPLE_SCHEMA, name="answer")

        payload = model.bind(**kwargs).kwargs

        assert "response_format" not in payload
        assert payload["extra_body"]["response_format"]["json_schema"]["name"] == "answer"


class TestStructuredOutputKwargsDispatcher:
    """tests for the ``structured_output_kwargs`` provider dispatcher."""

    def test_anthropic_dispatch_matches_provider_function(self) -> None:
        """``anthropic`` routes to the anthropic builder."""
        assert structured_output_kwargs("anthropic", _SAMPLE_SCHEMA, name="answer") == (
            anthropic_structured_output_kwargs(_SAMPLE_SCHEMA, name="answer")
        )

    def test_openrouter_dispatch_matches_provider_function(self) -> None:
        """``openrouter`` routes to the openrouter builder."""
        assert structured_output_kwargs("openrouter", _SAMPLE_SCHEMA, name="answer", strict=False) == (
            openrouter_structured_output_kwargs(_SAMPLE_SCHEMA, name="answer", strict=False)
        )

    def test_openai_dispatch_matches_provider_function(self) -> None:
        """``openai`` routes to the openai builder."""
        assert structured_output_kwargs("openai", _SAMPLE_SCHEMA) == (openai_structured_output_kwargs(_SAMPLE_SCHEMA))

    def test_providers_emit_distinct_shapes(self) -> None:
        """the three providers do not collapse to one shape.

        guards against a dispatcher regression that silently routes every
        provider to the same builder.
        """
        anthropic_kwargs = structured_output_kwargs("anthropic", _SAMPLE_SCHEMA)
        openrouter_kwargs = structured_output_kwargs("openrouter", _SAMPLE_SCHEMA)
        openai_kwargs = structured_output_kwargs("openai", _SAMPLE_SCHEMA)

        assert set(anthropic_kwargs) == {"output_config"}
        assert set(openrouter_kwargs) == {"response_format", "provider"}
        assert set(openai_kwargs) == {"extra_body"}

    def test_unknown_provider_raises_naming_the_provider_type(self) -> None:
        """an unknown provider_type fails loud and names itself.

        silently returning ``{}`` would drop the directive and let the
        model return prose -- the exact failure structured output exists
        to prevent.
        """
        with pytest.raises(StructuredOutputUnsupportedError, match="voyageai"):
            structured_output_kwargs("voyageai", _SAMPLE_SCHEMA)

    def test_unknown_provider_error_is_a_value_error(self) -> None:
        """the raise stays catchable as ``ValueError``.

        matches the package's existing unresolvable-provider idiom in
        ``threetears.models.factory``.
        """
        with pytest.raises(ValueError, match="structured output"):
            structured_output_kwargs("", _SAMPLE_SCHEMA)

    def test_dispatcher_never_returns_empty_kwargs(self) -> None:
        """every supported provider yields a non-empty directive."""
        for provider_type in ("anthropic", "openrouter", "openai"):
            assert structured_output_kwargs(provider_type, _SAMPLE_SCHEMA) != {}


class TestStructuredOutputErrorHierarchy:
    """tests for the shared ``StructuredOutputError`` base.

    the base exists so a caller (the gateway dispatch sites) can catch
    every "the CALLER got it wrong" structured-output failure in one
    clause and classify it as a bad request, rather than letting one kind
    escape into the generic handler and be misread as a provider outage.
    """

    def test_unsupported_error_derives_from_base(self) -> None:
        """the unsupported-provider case is a ``StructuredOutputError``."""
        assert issubclass(StructuredOutputUnsupportedError, StructuredOutputError)

    def test_schema_error_derives_from_base(self) -> None:
        """the bad-schema case is a ``StructuredOutputError``."""
        assert issubclass(StructuredOutputSchemaError, StructuredOutputError)

    def test_base_derives_from_value_error(self) -> None:
        """the base stays catchable as ``ValueError``.

        preserves the package's existing unresolvable-provider idiom in
        ``threetears.models.factory``.
        """
        assert issubclass(StructuredOutputError, ValueError)

    def test_the_two_kinds_are_siblings_not_ancestors(self) -> None:
        """neither concrete error subclasses the other.

        catching one kind must never silently swallow the other -- they
        have different causes and different operator-facing messages.
        """
        assert not issubclass(StructuredOutputSchemaError, StructuredOutputUnsupportedError)
        assert not issubclass(StructuredOutputUnsupportedError, StructuredOutputSchemaError)

    def test_unsupported_provider_is_catchable_as_the_base(self) -> None:
        """the dispatcher's unsupported raise is caught by the base clause."""
        with pytest.raises(StructuredOutputError, match="voyageai"):
            structured_output_kwargs("voyageai", _SAMPLE_SCHEMA)


class TestAnthropicMalformedSchemaHandling:
    """tests that a malformed caller schema fails as a CALLER error.

    ``anthropic.transform_schema`` validates the schema LOCALLY, before
    any API call, and signals rejection with bare builtin exceptions --
    including ``AssertionError`` from ``typing.assert_never``. an
    ``AssertionError`` escaping the provider layer reads to every caller
    as "our code is broken", not "the caller's schema is bad", so a
    consumer classifying errors routes it to an outage/unavailable branch
    and trips its circuit breaker. these tests pin the translation into a
    typed caller-error.
    """

    # one malformed schema per exception type transform_schema raises on
    # caller input, each traced to the raising line in
    # anthropic/lib/_parse/_transform.py.
    _UNRECOGNIZED_TYPE: dict[str, Any] = {
        "type": "definitely_not_a_json_schema_type",
        "properties": {"name": {"type": "banana"}},
    }
    _MISSING_TYPE: dict[str, Any] = {"properties": {"name": {"type": "string"}}}
    _PROPERTIES_NOT_A_MAPPING: dict[str, Any] = {"type": "object", "properties": ["name"]}
    _PROPERTY_VALUE_NOT_A_MAPPING: dict[str, Any] = {"type": "object", "properties": {"name": "string"}}
    _NESTED_UNRECOGNIZED_TYPE: dict[str, Any] = {
        "type": "object",
        "properties": {"outer": {"type": "object", "properties": {"inner": {"type": "banana"}}}},
    }
    _NOT_A_MAPPING_AT_ALL: Any = ["not", "a", "schema"]

    @staticmethod
    def _deeply_nested_schema(depth: int) -> dict[str, Any]:
        """build a validly-typed schema nested ``depth`` levels deep.

        ``transform_schema`` walks nested properties recursively with no
        depth guard, so a sufficiently deep schema exhausts the
        interpreter stack. the depth here clears CPython's default
        1000-frame limit while staying well inside what ``json.loads``
        accepts, i.e. it is reachable from ordinary caller json.

        :param depth: number of nesting levels to build
        :ptype depth: int
        :return: nested json-schema
        :rtype: dict[str, Any]
        """
        schema: dict[str, Any] = {"type": "string"}
        for _ in range(depth):
            schema = {"type": "object", "properties": {"x": schema}}
        return schema

    @pytest.mark.parametrize(
        ("label", "schema"),
        [
            ("unrecognized root type (assert_never -> AssertionError)", _UNRECOGNIZED_TYPE),
            ("no type/anyOf/oneOf/allOf (-> ValueError)", _MISSING_TYPE),
            ("properties is not a mapping (-> AttributeError)", _PROPERTIES_NOT_A_MAPPING),
            ("property value is not a mapping (-> TypeError)", _PROPERTY_VALUE_NOT_A_MAPPING),
            ("unrecognized type nested two levels deep", _NESTED_UNRECOGNIZED_TYPE),
            ("schema is not a mapping at all (-> TypeError)", _NOT_A_MAPPING_AT_ALL),
        ],
    )
    def test_malformed_schema_raises_schema_error(self, label: str, schema: Any) -> None:
        """every malformed-schema shape surfaces as ``StructuredOutputSchemaError``."""
        with pytest.raises(StructuredOutputSchemaError):
            anthropic_structured_output_kwargs(schema)

    def test_pathologically_nested_schema_raises_schema_error(self) -> None:
        """a stack-exhausting schema is a CALLER error, not an outage.

        ``transform_schema`` recurses per nesting level with no depth
        guard, so caller json nested past the interpreter's frame limit
        raises ``RecursionError`` -- which is a ``RuntimeError``, NOT one
        of the value/type rejections, and would otherwise escape the wrap
        into a consumer's generic handler and trip its breaker exactly
        like the bare ``AssertionError`` does.
        """
        schema = self._deeply_nested_schema(1200)

        with pytest.raises(StructuredOutputSchemaError):
            anthropic_structured_output_kwargs(schema)

    def test_pathologically_nested_schema_preserves_the_recursion_cause(self) -> None:
        """the stack-exhaustion cause is chained, not discarded."""
        schema = self._deeply_nested_schema(1200)

        with pytest.raises(StructuredOutputSchemaError) as excinfo:
            anthropic_structured_output_kwargs(schema)

        assert isinstance(excinfo.value.__cause__, RecursionError)
        assert "RecursionError" in str(excinfo.value)

    def test_moderately_nested_schema_still_transforms(self) -> None:
        """nesting BELOW the frame limit is transformed, not rejected.

        non-vacuous guard on the depth tests above: a wrap that rejected
        every nested schema would pass them while breaking legitimate
        callers.
        """
        schema = self._deeply_nested_schema(50)

        kwargs = anthropic_structured_output_kwargs(schema)

        assert kwargs["output_config"]["format"]["schema"]["additionalProperties"] is False

    def test_malformed_schema_does_not_leak_a_bare_assertion_error(self) -> None:
        """the raise is NOT a bare ``AssertionError``.

        this is the whole point: ``AssertionError`` is indistinguishable
        from an internal invariant failure, so consumers cannot classify
        it as a caller error.
        """
        with pytest.raises(StructuredOutputSchemaError) as excinfo:
            anthropic_structured_output_kwargs(self._UNRECOGNIZED_TYPE)

        assert not isinstance(excinfo.value, AssertionError)

    def test_message_names_the_schema_and_the_provider(self) -> None:
        """the message says WHOSE input was rejected and by which provider."""
        with pytest.raises(StructuredOutputSchemaError) as excinfo:
            anthropic_structured_output_kwargs(self._UNRECOGNIZED_TYPE)

        message = str(excinfo.value)
        assert "json schema" in message.lower()
        assert "anthropic" in message.lower()

    def test_message_includes_the_underlying_reason(self) -> None:
        """the underlying rejection reason rides in the message.

        without it an operator sees "bad schema" and has nothing to act
        on; the offending type name is what identifies the fix.
        """
        with pytest.raises(StructuredOutputSchemaError) as excinfo:
            anthropic_structured_output_kwargs(self._UNRECOGNIZED_TYPE)

        assert "definitely_not_a_json_schema_type" in str(excinfo.value)

    def test_underlying_exception_type_is_named_in_the_message(self) -> None:
        """the message names the builtin that ``transform_schema`` raised."""
        with pytest.raises(StructuredOutputSchemaError) as excinfo:
            anthropic_structured_output_kwargs(self._UNRECOGNIZED_TYPE)

        assert "AssertionError" in str(excinfo.value)

    def test_cause_is_preserved(self) -> None:
        """``__cause__`` chains to the original ``transform_schema`` failure.

        the raise uses ``from exc`` so the traceback keeps the line inside
        ``transform_schema`` that actually rejected the schema.
        """
        with pytest.raises(StructuredOutputSchemaError) as excinfo:
            anthropic_structured_output_kwargs(self._UNRECOGNIZED_TYPE)

        cause = excinfo.value.__cause__
        assert isinstance(cause, AssertionError)
        assert "definitely_not_a_json_schema_type" in str(cause)

    def test_cause_is_preserved_for_the_value_error_shape(self) -> None:
        """``__cause__`` is preserved for a non-assertion rejection too."""
        with pytest.raises(StructuredOutputSchemaError) as excinfo:
            anthropic_structured_output_kwargs(self._MISSING_TYPE)

        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_schema_error_is_catchable_as_the_shared_base(self) -> None:
        """consumers can catch bad-schema via ``StructuredOutputError``."""
        with pytest.raises(StructuredOutputError):
            anthropic_structured_output_kwargs(self._UNRECOGNIZED_TYPE)

    def test_schema_error_is_not_the_unsupported_error(self) -> None:
        """a bad schema is NOT reported as an unsupported provider.

        the two have different causes; conflating them makes the
        operator-facing message misreport what went wrong.
        """
        with pytest.raises(StructuredOutputSchemaError) as excinfo:
            anthropic_structured_output_kwargs(self._UNRECOGNIZED_TYPE)

        assert not isinstance(excinfo.value, StructuredOutputUnsupportedError)

    def test_dispatcher_surfaces_the_schema_error(self) -> None:
        """the failure survives dispatch through ``structured_output_kwargs``."""
        with pytest.raises(StructuredOutputSchemaError):
            structured_output_kwargs("anthropic", self._UNRECOGNIZED_TYPE)

    def test_valid_schema_still_transforms(self) -> None:
        """a VALID schema is unaffected by the wrap.

        non-vacuous guard: a wrap that swallowed everything, or one over a
        ``transform_schema`` that had stopped being called, would still
        pass every raises-assertion above.
        """
        kwargs = anthropic_structured_output_kwargs(_SAMPLE_SCHEMA)

        schema = kwargs["output_config"]["format"]["schema"]
        assert schema == transform_schema(_SAMPLE_SCHEMA)
        assert schema["additionalProperties"] is False
        assert schema["properties"]["answer"]["type"] == "string"


class TestVerbatimProvidersValidateSchemasLocally:
    """pins that openrouter/openai reject a malformed schema themselves.

    both builders embed the caller's schema verbatim, and the providers
    do NOT reject an incoherent one: live against openrouter ->
    ``google/gemini-2.5-flash``, ``{"type": "not_a_real_json_type"}``
    came back as the bare string ``"Dublin"`` and ``{"type": "object",
    "properties": "oops"}`` came back as ``{}``, both as SUCCESSFUL
    completions. a caller typo is therefore indistinguishable from a real
    answer unless the schema is checked before it is sent, which is what
    these tests pin.
    """

    _UNRECOGNIZED_TYPE: dict[str, Any] = {
        "type": "definitely_not_a_json_schema_type",
        "properties": {"name": {"type": "banana"}},
    }
    _PROPERTIES_NOT_A_MAPPING: dict[str, Any] = {"type": "object", "properties": "oops"}
    _PROPERTY_VALUE_NOT_A_MAPPING: dict[str, Any] = {"type": "object", "properties": {"name": "string"}}
    _NESTED_UNRECOGNIZED_TYPE: dict[str, Any] = {
        "type": "object",
        "properties": {"outer": {"type": "object", "properties": {"inner": {"type": "banana"}}}},
    }
    _NOT_A_MAPPING_AT_ALL: Any = ["not", "a", "schema"]

    _MALFORMED_SHAPES = [
        ("unrecognized root type", _UNRECOGNIZED_TYPE),
        ("properties is not a mapping", _PROPERTIES_NOT_A_MAPPING),
        ("property value is not a mapping", _PROPERTY_VALUE_NOT_A_MAPPING),
        ("unrecognized type nested two levels deep", _NESTED_UNRECOGNIZED_TYPE),
        ("schema is not a mapping at all", _NOT_A_MAPPING_AT_ALL),
    ]

    @pytest.mark.parametrize(("label", "schema"), _MALFORMED_SHAPES)
    def test_openrouter_rejects_a_malformed_schema(self, label: str, schema: Any) -> None:
        """openrouter refuses to build kwargs around an invalid schema."""
        with pytest.raises(StructuredOutputSchemaError):
            openrouter_structured_output_kwargs(schema)

    @pytest.mark.parametrize(("label", "schema"), _MALFORMED_SHAPES)
    def test_openai_rejects_a_malformed_schema(self, label: str, schema: Any) -> None:
        """openai refuses to build kwargs around an invalid schema."""
        with pytest.raises(StructuredOutputSchemaError):
            openai_structured_output_kwargs(schema)

    def test_openrouter_embeds_a_valid_schema_verbatim(self) -> None:
        """a VALID schema still rides through untouched.

        non-vacuous guard on the rejection tests: a builder that rejected
        everything would pass them while breaking every real caller.
        """
        kwargs = openrouter_structured_output_kwargs(_SAMPLE_SCHEMA)

        assert kwargs["response_format"]["json_schema"]["schema"] == _SAMPLE_SCHEMA

    def test_openai_embeds_a_valid_schema_verbatim(self) -> None:
        """a VALID schema still rides through untouched."""
        kwargs = openai_structured_output_kwargs(_SAMPLE_SCHEMA)

        assert kwargs["extra_body"]["response_format"]["json_schema"]["schema"] == _SAMPLE_SCHEMA

    def test_a_schema_without_a_type_is_accepted(self) -> None:
        """the check validates json-schema VALIDITY, not provider policy.

        ``{"properties": ...}`` with no ``type`` is a legal json-schema
        (Anthropic's own transform rejects it, which is that provider's
        stricter policy, not a property of the schema). pinning this
        keeps the local check from drifting into reimplementing each
        provider's subset rules, where it would start rejecting schemas
        the provider accepts.
        """
        schema: dict[str, Any] = {"properties": {"name": {"type": "string"}}}

        kwargs = openrouter_structured_output_kwargs(schema)

        assert kwargs["response_format"]["json_schema"]["schema"] == schema

    def test_message_names_the_provider_and_the_offending_location(self) -> None:
        """the message says which provider rejected it and where.

        the schema path is what makes the error actionable on a large
        schema; without it an operator sees "invalid schema" and has to
        bisect the document by hand.
        """
        with pytest.raises(StructuredOutputSchemaError) as excinfo:
            openrouter_structured_output_kwargs(self._PROPERTY_VALUE_NOT_A_MAPPING)

        message = str(excinfo.value)
        assert "openrouter" in message.lower()
        assert "properties/name" in message

    def test_message_stays_short_instead_of_re_rendering_the_schema(self) -> None:
        """the message carries the reason, not a dump of the whole schema.

        ``str(SchemaError)`` re-renders the entire offending schema (1.8
        kB for a two-key example); that message rides in an error
        response and in every log record built from it, so the builder
        carries ``.message`` + the path instead.
        """
        with pytest.raises(StructuredOutputSchemaError) as excinfo:
            openrouter_structured_output_kwargs(self._PROPERTY_VALUE_NOT_A_MAPPING)

        assert len(str(excinfo.value)) < 300

    def test_cause_is_preserved(self) -> None:
        """``__cause__`` chains to the underlying jsonschema rejection."""
        from jsonschema.exceptions import SchemaError

        with pytest.raises(StructuredOutputSchemaError) as excinfo:
            openai_structured_output_kwargs(self._UNRECOGNIZED_TYPE)

        assert isinstance(excinfo.value.__cause__, SchemaError)

    def test_pathologically_nested_schema_raises_schema_error(self) -> None:
        """a stack-exhausting schema is a CALLER error, not an outage.

        schema validation descends per nesting level with no depth guard,
        so caller json nested past the interpreter's frame limit raises
        ``RecursionError`` -- a ``RuntimeError``, NOT a ``SchemaError``,
        which would otherwise escape into a consumer's generic handler
        and trip its circuit breaker.
        """
        schema: dict[str, Any] = {"type": "string"}
        for _ in range(1200):
            schema = {"type": "object", "properties": {"x": schema}}

        with pytest.raises(StructuredOutputSchemaError) as excinfo:
            openrouter_structured_output_kwargs(schema)

        assert isinstance(excinfo.value.__cause__, RecursionError)

    def test_rejection_is_catchable_as_the_shared_base(self) -> None:
        """consumers catch it with the same clause as every other kind."""
        with pytest.raises(StructuredOutputError):
            openrouter_structured_output_kwargs(self._UNRECOGNIZED_TYPE)

    def test_dispatcher_surfaces_the_schema_error(self) -> None:
        """the failure survives dispatch through ``structured_output_kwargs``."""
        with pytest.raises(StructuredOutputSchemaError):
            structured_output_kwargs("openrouter", self._UNRECOGNIZED_TYPE)

        with pytest.raises(StructuredOutputSchemaError):
            structured_output_kwargs("openai", self._UNRECOGNIZED_TYPE)
