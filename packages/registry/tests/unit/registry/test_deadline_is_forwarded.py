"""§10.10 / SR-G2: a caller's remaining budget reaches the pod, clamped.

The deadline is the one quantity in the chain nobody downstream can compute.
``ToolManifestEntry.timeout_seconds`` is the tool's declared ceiling and
``ToolServer(max_call_seconds=...)`` is the pod's backstop -- both static, both
already known where they are used. How much patience *this* caller has left is
known only to the caller, and until this change it stopped at the registry.

``CallRequest.deadline_seconds`` (the pod-side accepting half) shipped in
0.24.1. This is the other end of the same wire: ``ProxyCallRequest`` grows the
field so an agent can express it, and the proxy forwards it clamped to its own
wait. No agent populates it yet -- that is hop one's sending half and needs its
own release, for exactly the reason recorded below.

**The rollout property is the load-bearing one here**, and it is tested first
because it is the one that can take a fleet down. A caller that declares no
deadline must produce the bytes the wire carried before this field existed --
not ``"deadline_seconds": null``, but no key at all. That distinction cost three
days of total refusal on the hop below this one: ``extra="forbid"`` treats an
unknown null exactly like an unknown value, and a 0.23.11 pod refused every call
from a 0.24.1 registry over a field no caller ever set. So the guarantee is
tested against a reader that predates the field, the way
``test_forward_is_version_tolerant.py`` does, rather than by asserting a value
was passed.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid7

import pytest
from pydantic import BaseModel, ConfigDict

from threetears.agent.tools.context_envelope import CallContext
from threetears.registry.proxy import (
    ProxyCallRequest,
    _build_internal_payload,
    _forwarded_deadline,
)


def _request(**overrides: Any) -> ProxyCallRequest:
    """A minimal well-formed proxy request."""
    fields: dict[str, Any] = {
        "tool_name": "threetears.web_search",
        "tool_version": "1.0",
        "arguments": {"query": "capybara"},
        "context": CallContext(agent_id=uuid7(), correlation_id=uuid7()),
    }
    fields.update(overrides)
    return ProxyCallRequest(**fields)


def _forwarded(request: ProxyCallRequest, *, effective_timeout: float | None) -> dict[str, Any]:
    """The envelope the pod actually receives, decoded from the wire bytes."""
    payload = _build_internal_payload(request, None, effective_timeout=effective_timeout)
    decoded: dict[str, Any] = json.loads(payload)
    return decoded


class _LaggingPodCallRequest(BaseModel):
    """A pod that predates ``deadline_seconds``, written by hand.

    Deliberately not built by removing a field from the real model: the point is
    to read the wire the way an OLD deployment reads it, and a model derived
    from the current one would inherit whatever the current one learns.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_version: str
    arguments: dict[str, Any]
    context: dict[str, Any] | None = None
    proxy_assertion: str | None = None
    result_subject: str | None = None


class TestACallerThatSaysNothingChangesNothing:
    """The no-deadline path must be byte-compatible with every live pod."""

    def test_the_key_is_absent_not_null(self) -> None:
        envelope = _forwarded(_request(), effective_timeout=30.0)

        assert "deadline_seconds" not in envelope, (
            "an unset deadline reached the wire as an explicit null. A pod predating the "
            "field refuses the WHOLE call on an unknown key, value or null alike -- this is "
            "the exact shape of the three-day cobalt-dev outage."
        )

    def test_a_pod_predating_the_field_still_parses_the_envelope(self) -> None:
        """The guarantee stated as the lagging reader experiences it."""
        envelope = _forwarded(_request(), effective_timeout=30.0)

        parsed = _LaggingPodCallRequest.model_validate(envelope)

        assert parsed.tool_name == "threetears.web_search"

    def test_a_pod_predating_the_field_refuses_when_a_deadline_is_set(self) -> None:
        """The rollout constraint, pinned rather than left as prose.

        This is not a defect -- it is why no agent may be taught to populate the
        field until the fleet carries a pod that accepts it. Pinning it here
        means the constraint is discovered by a test rather than by an outage,
        and it documents the ordering for whoever writes hop one's sender.
        """
        envelope = _forwarded(_request(deadline_seconds=5.0), effective_timeout=30.0)

        assert "deadline_seconds" in envelope
        with pytest.raises(Exception):
            _LaggingPodCallRequest.model_validate(envelope)


class TestTheDeadlineReachesThePod:
    def test_a_declared_deadline_is_forwarded(self) -> None:
        envelope = _forwarded(_request(deadline_seconds=5.0), effective_timeout=30.0)

        assert envelope["deadline_seconds"] == 5.0

    def test_a_caller_may_ask_for_less_than_the_tool_allows(self) -> None:
        """The point of the field: a short-patience caller shortens the call."""
        envelope = _forwarded(_request(deadline_seconds=4.0), effective_timeout=30.0)

        assert envelope["deadline_seconds"] < 30.0

    def test_a_caller_cannot_buy_more_time_than_the_proxy_will_wait(self) -> None:
        """Clamped, because the proxy stops listening at its own timeout.

        Forwarding the larger number would license the pod to work past the
        moment its answer becomes unreadable.
        """
        envelope = _forwarded(_request(deadline_seconds=300.0), effective_timeout=30.0)

        assert envelope["deadline_seconds"] == 30.0


class TestTheClampItself:
    """``_forwarded_deadline`` in isolation, including its boundaries."""

    @pytest.mark.parametrize(
        ("caller", "timeout", "expected"),
        [
            (None, 30.0, None),
            (None, None, None),
            (5.0, 30.0, 5.0),
            (300.0, 30.0, 30.0),
            (30.0, 30.0, 30.0),
            (7.5, None, 7.5),
        ],
    )
    def test_the_clamp_is_the_minimum_of_what_is_known(
        self, caller: float | None, timeout: float | None, expected: float | None
    ) -> None:
        assert _forwarded_deadline(caller, timeout) == expected

    def test_no_caller_deadline_survives_a_missing_timeout(self) -> None:
        """``None`` in both positions must not become ``0`` or an exception.

        A zero deadline would tell the pod it has no time at all, which is a
        refusal dressed as a budget.
        """
        assert _forwarded_deadline(None, None) is None


class TestTheseAssertionsCanFail:
    """Guard: prove the wire assertions distinguish absent from null.

    A test that checked ``envelope.get("deadline_seconds") is None`` would pass
    for BOTH the safe shape and the shape that caused the outage. This shows the
    two are actually told apart.
    """

    def test_absent_and_null_are_not_the_same_assertion(self) -> None:
        absent = _forwarded(_request(), effective_timeout=30.0)
        explicit_null = {**absent, "deadline_seconds": None}

        assert "deadline_seconds" not in absent
        assert "deadline_seconds" in explicit_null
        assert absent.get("deadline_seconds") == explicit_null.get("deadline_seconds")

    def test_a_lagging_reader_refuses_the_null_shape(self) -> None:
        """The regression this file exists to prevent, reproduced deliberately."""
        absent = _forwarded(_request(), effective_timeout=30.0)
        explicit_null = {**absent, "deadline_seconds": None}

        _LaggingPodCallRequest.model_validate(absent)
        with pytest.raises(Exception):
            _LaggingPodCallRequest.model_validate(explicit_null)
