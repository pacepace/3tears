"""v0.13.9 auth C2: receiver-first ``pop`` field on :class:`ProxyCallRequest`.

The agent→proxy hop carries a proof-of-possession (``pop``) so a leaked identity token alone
is unusable. :class:`ProxyCallRequest` is ``extra='forbid'``, so the proxy must ACCEPT ``pop``
before any SDK sender emits it — this is the receiver-first half of the rollout. The field is
INERT (off) this chunk: the proxy parses it but nothing verifies it yet. The matching
``proxy_assertion`` / ``identity_token`` fields are covered in the agent-tools envelope tests.
"""

from __future__ import annotations

from uuid import uuid7

import pytest
from pydantic import ValidationError

from threetears.agent.tools.context_envelope import CallContext
from threetears.registry.proxy import ProxyCallRequest


class TestProxyCallRequestAuthFields:
    """``ProxyCallRequest`` accepts ``pop`` while keeping ``extra='forbid'`` intact."""

    def test_accepts_pop(self) -> None:
        req = ProxyCallRequest(tool_name="test.stub", tool_version="1.0", arguments={}, pop="pop-proof")
        assert req.pop == "pop-proof"

    def test_pop_defaults_none(self) -> None:
        req = ProxyCallRequest(tool_name="test.stub", tool_version="1.0", arguments={})
        assert req.pop is None

    def test_pop_is_not_treated_as_a_legacy_flat_field(self) -> None:
        # the legacy-flat-field rejection validator must not snag `pop` -- it is a NEW declared
        # field, not one of the removed flat identity dimensions.
        req = ProxyCallRequest(
            tool_name="test.stub",
            tool_version="1.0",
            arguments={},
            pop="x",
            context=CallContext(agent_id=uuid7()),
        )
        assert req.pop == "x"

    def test_still_forbids_a_genuinely_unknown_field(self) -> None:
        with pytest.raises(ValidationError):
            ProxyCallRequest(
                tool_name="test.stub",
                tool_version="1.0",
                arguments={},
                nonsense="x",  # type: ignore[call-arg]
            )

    def test_forwards_identity_token_in_context(self) -> None:
        ctx = CallContext(agent_id=uuid7(), identity_token="eyJ.tok")
        req = ProxyCallRequest(tool_name="test.stub", tool_version="1.0", arguments={}, context=ctx)
        assert req.context is not None
        assert req.context.identity_token == "eyJ.tok"

    def test_forwards_engagement_id_in_context(self) -> None:
        # engagement_id rides on CallContext (like identity_token), so it travels whole through
        # ProxyCallRequest without touching the proxy model.
        engagement_id = uuid7()
        ctx = CallContext(agent_id=uuid7(), engagement_id=engagement_id)
        req = ProxyCallRequest(tool_name="test.stub", tool_version="1.0", arguments={}, context=ctx)
        assert req.context is not None
        assert req.context.engagement_id == engagement_id

    def test_pop_round_trips_json(self) -> None:
        req = ProxyCallRequest(tool_name="test.stub", tool_version="1.0", arguments={}, pop="p")
        parsed = ProxyCallRequest.model_validate_json(req.model_dump_json())
        assert parsed.pop == "p"
