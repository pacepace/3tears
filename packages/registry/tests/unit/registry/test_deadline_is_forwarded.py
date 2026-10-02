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

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from ._dispatch_auth import make_authed_request
from ._forwarding import PROXY_TIMEOUT_SECONDS, forwarded_envelope


async def _forwarded(*, deadline_seconds: float | None = None, tool_timeout: float | None) -> dict[str, Any]:
    """The envelope the pod actually receives for one call routed through the proxy.

    ``tool_timeout`` is the timeout the routed tool declares, which is the proxy's own wait for the
    pod; ``None`` leaves the proxy on its default.
    """
    return await forwarded_envelope(
        make_authed_request(deadline_seconds=deadline_seconds), tool_timeout_seconds=tool_timeout
    )


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

    @pytest.mark.asyncio
    async def test_the_key_is_absent_not_null(self) -> None:
        envelope = await _forwarded(tool_timeout=30.0)

        assert "deadline_seconds" not in envelope, (
            "an unset deadline reached the wire as an explicit null. A pod predating the "
            "field refuses the WHOLE call on an unknown key, value or null alike -- this is "
            "the exact shape of the three-day cobalt-dev outage."
        )

    @pytest.mark.asyncio
    async def test_a_pod_predating_the_field_still_parses_the_envelope(self) -> None:
        """The guarantee stated as the lagging reader experiences it."""
        envelope = await _forwarded(tool_timeout=30.0)

        parsed = _LaggingPodCallRequest.model_validate(envelope)

        assert parsed.tool_name == "threetears.calculator"

    @pytest.mark.asyncio
    async def test_a_pod_predating_the_field_refuses_when_a_deadline_is_set(self) -> None:
        """The rollout constraint, pinned rather than left as prose.

        This is not a defect -- it is why no agent may be taught to populate the
        field until the fleet carries a pod that accepts it. Pinning it here
        means the constraint is discovered by a test rather than by an outage,
        and it documents the ordering for whoever writes hop one's sender.
        """
        envelope = await _forwarded(deadline_seconds=5.0, tool_timeout=30.0)

        assert "deadline_seconds" in envelope
        with pytest.raises(Exception):
            _LaggingPodCallRequest.model_validate(envelope)


class TestTheDeadlineReachesThePod:
    @pytest.mark.asyncio
    async def test_a_declared_deadline_is_forwarded(self) -> None:
        envelope = await _forwarded(deadline_seconds=5.0, tool_timeout=30.0)

        assert envelope["deadline_seconds"] == 5.0

    @pytest.mark.asyncio
    async def test_a_caller_may_ask_for_less_than_the_tool_allows(self) -> None:
        """The point of the field: a short-patience caller shortens the call."""
        envelope = await _forwarded(deadline_seconds=4.0, tool_timeout=30.0)

        assert envelope["deadline_seconds"] < 30.0

    @pytest.mark.asyncio
    async def test_a_caller_cannot_buy_more_time_than_the_proxy_will_wait(self) -> None:
        """Clamped, because the proxy stops listening at its own timeout.

        Forwarding the larger number would license the pod to work past the
        moment its answer becomes unreadable.
        """
        envelope = await _forwarded(deadline_seconds=300.0, tool_timeout=30.0)

        assert envelope["deadline_seconds"] == 30.0


class TestTheClamp:
    """The clamp at its boundaries, as the pod receives it.

    The proxy always resolves its own wait before it forwards -- the routed copy's declared
    timeout, else the proxy default -- so the clamp is exercised against both sources of that
    wait, never against an unresolved one.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("caller", "tool_timeout", "expected"),
        [
            (5.0, 30.0, 5.0),
            (300.0, 30.0, 30.0),
            (30.0, 30.0, 30.0),
            (7.5, None, 7.5),
            (PROXY_TIMEOUT_SECONDS + 100.0, None, PROXY_TIMEOUT_SECONDS),
        ],
    )
    async def test_the_clamp_is_the_minimum_of_the_caller_and_the_proxys_wait(
        self, caller: float, tool_timeout: float | None, expected: float
    ) -> None:
        envelope = await _forwarded(deadline_seconds=caller, tool_timeout=tool_timeout)

        assert envelope["deadline_seconds"] == expected

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool_timeout", [30.0, None])
    async def test_no_caller_deadline_never_becomes_one(self, tool_timeout: float | None) -> None:
        """No caller deadline must not become ``0`` or any other value, whichever wait applies.

        A zero deadline would tell the pod it has no time at all, which is a
        refusal dressed as a budget.
        """
        envelope = await _forwarded(tool_timeout=tool_timeout)

        assert "deadline_seconds" not in envelope


class TestTheseAssertionsCanFail:
    """Guard: prove the wire assertions distinguish absent from null.

    A test that checked ``envelope.get("deadline_seconds") is None`` would pass
    for BOTH the safe shape and the shape that caused the outage. This shows the
    two are actually told apart.
    """

    @pytest.mark.asyncio
    async def test_absent_and_null_are_not_the_same_assertion(self) -> None:
        absent = await _forwarded(tool_timeout=30.0)
        explicit_null = {**absent, "deadline_seconds": None}

        assert "deadline_seconds" not in absent
        assert "deadline_seconds" in explicit_null
        assert absent.get("deadline_seconds") == explicit_null.get("deadline_seconds")

    @pytest.mark.asyncio
    async def test_a_lagging_reader_refuses_the_null_shape(self) -> None:
        """The regression this file exists to prevent, reproduced deliberately."""
        absent = await _forwarded(tool_timeout=30.0)
        explicit_null = {**absent, "deadline_seconds": None}

        _LaggingPodCallRequest.model_validate(absent)
        with pytest.raises(Exception):
            _LaggingPodCallRequest.model_validate(explicit_null)
