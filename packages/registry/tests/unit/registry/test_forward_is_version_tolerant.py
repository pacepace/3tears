"""the registry→pod envelope must not break a pod that predates one of its fields.

:class:`~threetears.agent.tools.server.CallRequest` is ``extra='forbid'``: a pod refuses the
WHOLE call when the envelope carries a field its version does not declare. New fields are
therefore rolled out receiver-first -- the pod ships a release that ACCEPTS the field before
any caller populates it (see ``result_subject``, ``deadline_seconds``).

That rollout order is defeated by serialization alone. ``model_dump_json()`` emits every
declared field, so an optional nobody set still crosses the wire as an explicit null, and a
field can break every lagging pod in the fleet without a single caller ever using it. That is
not hypothetical: a 0.23.11 pod refused every call from a 0.24.1 registry on ``deadline_seconds``
for three days. These pin the property that makes the receiver-first rollout actually hold.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from threetears.nats import SYNC_REPLY_BUDGET_SECONDS

from ._dispatch_auth import make_authed_request
from ._forwarding import forwarded_envelope

#: a declared timeout the reply inbox cannot outlive, so the call takes the durable path and the
#: proxy names a ``result_subject`` in what it forwards.
_DURABLE_TIMEOUT = SYNC_REPLY_BUDGET_SECONDS * 4


class _PodPredatingTheField(BaseModel):
    """a pod's ``CallRequest`` as it looked BEFORE the newest optional was added.

    Deliberately hand-written rather than imported: the point is to model a version this
    workspace no longer contains. ``extra='forbid'`` mirrors the real model, which is what
    turns an unknown key into a refused call rather than an ignored one.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    tool_version: str
    arguments: dict[str, Any]
    context: Any = None
    proxy_assertion: str | None = None
    result_subject: str | None = None


async def _forwarded(*, tool_timeout: float | None = 5.0) -> dict[str, Any]:
    """the envelope a pod receives for one call the proxy forwards, unsigned and with no deadline."""
    return await forwarded_envelope(make_authed_request(arguments={"target": "x"}), tool_timeout_seconds=tool_timeout)


class TestForwardedEnvelopeIsVersionTolerant:
    @pytest.mark.asyncio
    async def test_an_older_pod_can_parse_the_forwarded_envelope(self) -> None:
        """the whole point: a pod predating the newest field still accepts the call."""
        _PodPredatingTheField.model_validate(await _forwarded())

    @pytest.mark.asyncio
    async def test_unset_optionals_are_absent_rather_than_null(self) -> None:
        """an unset optional must not reach the wire at all -- a null key is as fatal to a
        forbidding model as a populated one."""
        wire = await _forwarded()
        assert "deadline_seconds" not in wire
        assert "result_subject" not in wire
        assert "proxy_assertion" not in wire

    @pytest.mark.asyncio
    async def test_the_fields_that_carry_the_call_still_travel(self) -> None:
        """tolerance must not be bought by dropping payload: required fields and a SET
        optional both survive."""
        wire = await _forwarded(tool_timeout=_DURABLE_TIMEOUT)
        assert wire["tool_name"] == "threetears.calculator"
        assert wire["tool_version"] == "1.0.0"
        assert wire["arguments"] == {"target": "x"}
        assert wire["result_subject"].startswith("test.tools.result.pod-1.")
