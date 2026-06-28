"""tests for :class:`CallContext` value type.

covers the three slices spelled out in ``context-task-01``:

1. JSON round-trip — the envelope survives ``model_dump_json`` /
   ``model_validate_json`` across the NATS wire boundary without losing
   identity fields or the ``trace`` escape hatch.
2. ``trace`` merge immutability — helper returns a new instance and
   never mutates the source value.
3. pydantic rejects the removed flat identity fields on
   :class:`CallRequest` with a message that names the field and points
   at :mod:`threetears.agent.tools.context_envelope`.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid7

import pytest
from pydantic import ValidationError

from threetears.agent.tools.context_envelope import CallContext, bind_log_context
from threetears.agent.tools.server import CallRequest
from threetears.observe import clear_context
from threetears.observe.logging import get_context


class TestCallContextRoundTrip:
    """:class:`CallContext` survives JSON serialization unchanged."""

    def test_full_round_trip_preserves_all_fields(self) -> None:
        """every identity field + trace entry survives JSON round-trip."""
        conversation_id = uuid7()
        user_id = uuid7()
        customer_id = uuid7()
        correlation_id = uuid7()
        agent_id = uuid7()
        source = CallContext(
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id,
            correlation_id=correlation_id,
            agent_id=agent_id,
            trace={"request_id": "req-abc", "parent_span": "span-42"},
        )

        wire = source.model_dump_json()
        data = json.loads(wire)
        assert data["conversation_id"] == str(conversation_id)
        assert data["user_id"] == str(user_id)
        assert data["customer_id"] == str(customer_id)
        assert data["correlation_id"] == str(correlation_id)
        assert data["agent_id"] == str(agent_id)
        assert data["trace"] == {"request_id": "req-abc", "parent_span": "span-42"}

        parsed = CallContext.model_validate_json(wire)
        assert parsed.conversation_id == conversation_id
        assert parsed.user_id == user_id
        assert parsed.customer_id == customer_id
        assert parsed.correlation_id == correlation_id
        assert parsed.agent_id == agent_id
        assert parsed.trace == {"request_id": "req-abc", "parent_span": "span-42"}

    def test_all_fields_optional_default_to_none_and_empty_trace(self) -> None:
        """omitted fields default to ``None``; ``trace`` defaults to empty dict."""
        ctx = CallContext()
        assert ctx.conversation_id is None
        assert ctx.user_id is None
        assert ctx.customer_id is None
        assert ctx.correlation_id is None
        assert ctx.agent_id is None
        assert ctx.trace == {}

    def test_uuid_fields_parsed_from_string_wire(self) -> None:
        """string UUIDs on the wire are coerced back to :class:`UUID` instances."""
        conversation_id = uuid7()
        wire = json.dumps({"conversation_id": str(conversation_id)})
        parsed = CallContext.model_validate_json(wire)
        assert isinstance(parsed.conversation_id, UUID)
        assert parsed.conversation_id == conversation_id


class TestCallContextTraceMerge:
    """:meth:`CallContext.with_trace` returns a new instance, never mutates."""

    def test_merge_adds_new_keys(self) -> None:
        """merging adds keys to the returned instance."""
        base = CallContext(trace={"a": "1"})
        merged = base.with_trace({"b": "2"})
        assert merged.trace == {"a": "1", "b": "2"}

    def test_merge_overrides_existing_keys(self) -> None:
        """conflicting keys in the overlay win on the returned instance."""
        base = CallContext(trace={"a": "1", "b": "2"})
        merged = base.with_trace({"b": "updated"})
        assert merged.trace == {"a": "1", "b": "updated"}

    def test_merge_does_not_mutate_source(self) -> None:
        """the source instance's ``trace`` is unchanged after merge."""
        base = CallContext(trace={"a": "1"})
        _ = base.with_trace({"b": "2"})
        assert base.trace == {"a": "1"}

    def test_merge_returns_new_instance(self) -> None:
        """merge returns a distinct object, not the mutated source."""
        base = CallContext(trace={"a": "1"})
        merged = base.with_trace({"b": "2"})
        assert merged is not base

    def test_merge_preserves_identity_fields(self) -> None:
        """identity fields carry over onto the merged instance untouched."""
        conversation_id = uuid7()
        user_id = uuid7()
        base = CallContext(
            conversation_id=conversation_id,
            user_id=user_id,
            trace={"a": "1"},
        )
        merged = base.with_trace({"b": "2"})
        assert merged.conversation_id == conversation_id
        assert merged.user_id == user_id

    def test_merge_empty_overlay_returns_equal_trace(self) -> None:
        """merging an empty overlay yields the same trace contents."""
        base = CallContext(trace={"a": "1"})
        merged = base.with_trace({})
        assert merged.trace == {"a": "1"}
        assert merged is not base


class TestCallRequestRejectsLegacyFlatFields:
    """pydantic surfaces the migration path when callers emit removed fields."""

    def test_conversation_id_flat_field_rejected(self) -> None:
        """a flat ``conversation_id`` produces a validation error naming the field."""
        with pytest.raises(ValidationError) as excinfo:
            CallRequest(
                tool_name="test.stub",
                tool_version="1.0",
                arguments={},
                conversation_id=str(uuid7()),  # type: ignore[call-arg]
            )
        message = str(excinfo.value)
        assert "conversation_id" in message
        assert "CallContext" in message
        assert "threetears.agent.tools.context_envelope" in message

    def test_user_id_flat_field_rejected(self) -> None:
        """a flat ``user_id`` produces a validation error naming the field."""
        with pytest.raises(ValidationError) as excinfo:
            CallRequest(
                tool_name="test.stub",
                tool_version="1.0",
                arguments={},
                user_id=str(uuid7()),  # type: ignore[call-arg]
            )
        message = str(excinfo.value)
        assert "user_id" in message
        assert "CallContext" in message

    def test_customer_id_flat_field_rejected(self) -> None:
        """a flat ``customer_id`` produces a validation error naming the field."""
        with pytest.raises(ValidationError) as excinfo:
            CallRequest(
                tool_name="test.stub",
                tool_version="1.0",
                arguments={},
                customer_id=str(uuid7()),  # type: ignore[call-arg]
            )
        message = str(excinfo.value)
        assert "customer_id" in message
        assert "CallContext" in message

    def test_correlation_id_flat_field_rejected(self) -> None:
        """a flat ``correlation_id`` produces a validation error naming the field.

        correlation_id moved onto :class:`CallContext.correlation_id`
        in context-task-01; the only echo at response time is on
        :class:`CallResponse`, not a second inbound field.
        """
        with pytest.raises(ValidationError) as excinfo:
            CallRequest(
                tool_name="test.stub",
                tool_version="1.0",
                arguments={},
                correlation_id="corr-1",  # type: ignore[call-arg]
            )
        message = str(excinfo.value)
        assert "correlation_id" in message
        assert "CallContext" in message

    def test_call_request_accepts_context_envelope(self) -> None:
        """a :class:`CallContext` nested under ``context`` is accepted."""
        context = CallContext(
            conversation_id=uuid7(),
            user_id=uuid7(),
            correlation_id=uuid7(),
        )
        req = CallRequest(
            tool_name="test.stub",
            tool_version="1.0",
            arguments={},
            context=context,
        )
        assert req.context is not None
        assert req.context.conversation_id == context.conversation_id
        assert req.context.user_id == context.user_id
        assert req.context.correlation_id == context.correlation_id

    def test_call_request_accepts_missing_context(self) -> None:
        """a request without ``context`` still parses (stateless tools)."""
        req = CallRequest(
            tool_name="test.stub",
            tool_version="1.0",
            arguments={},
        )
        assert req.context is None


class TestBindLogContext:
    """:func:`bind_log_context` projects identity fields onto log-tag keys.

    verifies the mapping from :class:`CallContext` UUID fields to the
    canonical tag names (``cid``/``conv``/``user``/``agent``/
    ``customer``) declared in the platform logging contract
    (``docs/guides/logging-contract.md``). each UUID is stringified at
    the binding border so downstream log renderers can emit it without
    re-serialization.
    """

    def teardown_method(self) -> None:
        """reset the ContextVar between tests so state does not leak."""
        clear_context()

    def test_binds_all_identity_fields_as_strings(self) -> None:
        """every populated identity field appears under its canonical tag."""
        conversation_id = uuid7()
        user_id = uuid7()
        customer_id = uuid7()
        correlation_id = uuid7()
        agent_id = uuid7()
        ctx = CallContext(
            conversation_id=conversation_id,
            user_id=user_id,
            customer_id=customer_id,
            correlation_id=correlation_id,
            agent_id=agent_id,
        )

        bind_log_context(ctx)

        bound = get_context()
        assert bound == {
            "cid": str(correlation_id),
            "conv": str(conversation_id),
            "user": str(user_id),
            "agent": str(agent_id),
            "customer": str(customer_id),
        }

    def test_none_fields_are_absent_from_bound_tags(self) -> None:
        """unpopulated identity fields are absent from the bound context.

        :func:`threetears.observe.set_context` pops keys set to ``None``,
        so the bound context carries only the populated tags. downstream
        log formatters render only the tags that are present.
        """
        ctx = CallContext(conversation_id=uuid7())

        bind_log_context(ctx)

        bound = get_context()
        assert bound == {"conv": str(ctx.conversation_id)}

    def test_none_context_clears_all_tags(self) -> None:
        """passing ``None`` resets every canonical tag."""
        bind_log_context(CallContext(conversation_id=uuid7(), user_id=uuid7()))
        bind_log_context(None)

        assert get_context() == {}

    def test_empty_context_clears_all_tags(self) -> None:
        """a :class:`CallContext` with no populated fields leaves no tags set."""
        bind_log_context(CallContext(conversation_id=uuid7()))
        bind_log_context(CallContext())

        assert get_context() == {}


class TestAuthWireFields:
    """v0.13.9 auth C2: receiver-first wire fields, all optional and INERT (off).

    ``identity_token`` (the Hub-issued JWS) rides on :class:`CallContext`, so it travels whole
    through both ``ProxyCallRequest`` and ``CallRequest`` without touching either model.
    ``proxy_assertion`` (the proxy's body-bound assertion to the pod) is a top-level field on
    ``CallRequest`` — which is ``extra='forbid'``, so the receiver must ACCEPT it before any
    sender emits it. Nothing READS these yet; verification + emission land in later chunks.
    """

    def test_call_context_carries_identity_token_round_trip(self) -> None:
        """``identity_token`` survives JSON round-trip on the context envelope."""
        ctx = CallContext(agent_id=uuid7(), identity_token="eyJhbGciOiJFZERTQSJ9.payload.sig")
        wire = ctx.model_dump_json()
        assert json.loads(wire)["identity_token"] == "eyJhbGciOiJFZERTQSJ9.payload.sig"
        parsed = CallContext.model_validate_json(wire)
        assert parsed.identity_token == "eyJhbGciOiJFZERTQSJ9.payload.sig"

    def test_call_context_identity_token_defaults_none(self) -> None:
        """``identity_token`` defaults to ``None`` (the off state)."""
        assert CallContext().identity_token is None

    def test_call_context_without_identity_token_still_parses(self) -> None:
        """backward-compat: an old envelope with no ``identity_token`` deserializes fine."""
        parsed = CallContext.model_validate_json(json.dumps({"agent_id": str(uuid7())}))
        assert parsed.identity_token is None

    def test_with_trace_preserves_identity_token(self) -> None:
        """the ``with_trace`` copy keeps ``identity_token`` (model_copy carries all fields)."""
        base = CallContext(agent_id=uuid7(), identity_token="tok")
        merged = base.with_trace({"a": "1"})
        assert merged.identity_token == "tok"

    def test_call_context_carries_engagement_id_round_trip(self) -> None:
        """``engagement_id`` survives JSON round-trip as a typed UUID on the envelope."""
        engagement_id = uuid7()
        ctx = CallContext(agent_id=uuid7(), engagement_id=engagement_id)
        wire = ctx.model_dump_json()
        assert json.loads(wire)["engagement_id"] == str(engagement_id)
        parsed = CallContext.model_validate_json(wire)
        assert isinstance(parsed.engagement_id, UUID)
        assert parsed.engagement_id == engagement_id

    def test_call_context_engagement_id_defaults_none(self) -> None:
        """``engagement_id`` defaults to ``None`` (calls not bound to an engagement)."""
        assert CallContext().engagement_id is None

    def test_call_context_without_engagement_id_still_parses(self) -> None:
        """backward-compat: an envelope with no ``engagement_id`` deserializes fine."""
        parsed = CallContext.model_validate_json(json.dumps({"agent_id": str(uuid7())}))
        assert parsed.engagement_id is None

    def test_with_trace_preserves_engagement_id(self) -> None:
        """the ``with_trace`` copy keeps ``engagement_id`` (model_copy carries all fields)."""
        engagement_id = uuid7()
        base = CallContext(agent_id=uuid7(), engagement_id=engagement_id)
        merged = base.with_trace({"a": "1"})
        assert merged.engagement_id == engagement_id

    def test_call_request_forwards_engagement_id_in_context(self) -> None:
        """``engagement_id`` nested in ``context`` survives onto ``CallRequest``."""
        engagement_id = uuid7()
        ctx = CallContext(agent_id=uuid7(), engagement_id=engagement_id)
        req = CallRequest(tool_name="test.stub", tool_version="1.0", arguments={}, context=ctx)
        assert req.context is not None
        assert req.context.engagement_id == engagement_id

    def test_call_request_accepts_proxy_assertion(self) -> None:
        """the pod's ``CallRequest`` accepts the new ``proxy_assertion`` field."""
        req = CallRequest(
            tool_name="test.stub", tool_version="1.0", arguments={}, proxy_assertion="assert-blob"
        )
        assert req.proxy_assertion == "assert-blob"

    def test_call_request_proxy_assertion_defaults_none(self) -> None:
        """``proxy_assertion`` defaults to ``None`` (the off state)."""
        req = CallRequest(tool_name="test.stub", tool_version="1.0", arguments={})
        assert req.proxy_assertion is None

    def test_call_request_still_forbids_a_genuinely_unknown_field(self) -> None:
        """extra='forbid' stays intact: only the declared auth field is accepted, not any extra."""
        with pytest.raises(ValidationError):
            CallRequest(
                tool_name="test.stub",
                tool_version="1.0",
                arguments={},
                totally_unknown="x",  # type: ignore[call-arg]
            )

    def test_call_request_forwards_identity_token_in_context(self) -> None:
        """``identity_token`` nested in ``context`` survives onto ``CallRequest``."""
        ctx = CallContext(agent_id=uuid7(), identity_token="eyJ.tok")
        req = CallRequest(tool_name="test.stub", tool_version="1.0", arguments={}, context=ctx)
        assert req.context is not None
        assert req.context.identity_token == "eyJ.tok"

    def test_call_context_carries_user_identity_token_round_trip(self) -> None:
        """the Hub-minted ``user_identity_token`` survives JSON round-trip on the context envelope."""
        ctx = CallContext(agent_id=uuid7(), user_identity_token="eyJhbGciOiJFZERTQSJ9.usr.sig")
        wire = ctx.model_dump_json()
        assert json.loads(wire)["user_identity_token"] == "eyJhbGciOiJFZERTQSJ9.usr.sig"
        parsed = CallContext.model_validate_json(wire)
        assert parsed.user_identity_token == "eyJhbGciOiJFZERTQSJ9.usr.sig"

    def test_call_context_user_identity_token_defaults_none(self) -> None:
        """``user_identity_token`` defaults to ``None`` (agent-initiated, no human in the loop)."""
        assert CallContext().user_identity_token is None

    def test_call_context_without_user_identity_token_still_parses(self) -> None:
        """backward-compat: an envelope with no ``user_identity_token`` deserializes fine."""
        parsed = CallContext.model_validate_json(json.dumps({"agent_id": str(uuid7())}))
        assert parsed.user_identity_token is None

    def test_with_trace_preserves_user_identity_token(self) -> None:
        """the ``with_trace`` copy keeps ``user_identity_token`` (model_copy carries all fields)."""
        base = CallContext(agent_id=uuid7(), user_identity_token="usr")
        merged = base.with_trace({"a": "1"})
        assert merged.user_identity_token == "usr"

    def test_call_request_forwards_user_identity_token_in_context(self) -> None:
        """``user_identity_token`` nested in ``context`` survives onto ``CallRequest``."""
        ctx = CallContext(agent_id=uuid7(), user_identity_token="eyJ.usr")
        req = CallRequest(tool_name="test.stub", tool_version="1.0", arguments={}, context=ctx)
        assert req.context is not None
        assert req.context.user_identity_token == "eyJ.usr"
