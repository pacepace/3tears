"""Declared test doubles for the injected ports (SR-O5).

Every double here declares the production protocol it stands in for by
subclassing it, which is what makes protocol drift break loudly instead of
rotting until some downstream test happens to call the missing method. The
family enforces the same rule mechanically for ``Fake<Name>`` classes under
any ``tests/`` tree (``test_fake_protocol_parity``); these live in ``src``
because consumers import them, so the subclass declaration is the whole of
the guarantee and is not optional.

:class:`ScriptedTransport` is named for what it is rather than prefixed
``Fake``: it is not a stub that returns one canned answer but a scripted
sequence, which is what a retry pin and a taxonomy pin both need -- an
attempt that fails followed by one that succeeds.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from threetears.search.contracts import EGRESS_DIRECT, SearchTransport, TransportResponse

__all__ = ["ScriptedTransport", "TransportScript"]


@dataclass(frozen=True, slots=True)
class TransportScript:
    """One scripted exchange: either an answer or a failure.

    Exactly one of ``response`` and ``failure`` is meaningful. A script step
    that raises is how a conformance case drives the taxonomy without a
    network: the adapter's mapping is the thing under test, and the whole
    point of the injected seam is that it can be driven.
    """

    #: HTTP status the step answers with.
    status_code: int = 200
    #: body bytes the step answers with.
    body: bytes = b'{"results": []}'
    #: response headers, lower-cased keys, as the protocol promises.
    headers: Mapping[str, str] = field(default_factory=dict)
    #: wall-clock the step reports having taken.
    elapsed_seconds: float = 0.001
    #: attempts the step reports having made, for the accounting pins.
    attempts: int = 1
    #: exception the step raises instead of answering. A transport may raise
    #: whatever it likes, including this package's typed taxonomy.
    failure: BaseException | None = None


class ScriptedTransport(SearchTransport):  # parity-with: threetears.search.contracts.SearchTransport
    """A transport that answers a prepared sequence, recording what it saw.

    Satisfies the injected seam by subclass declaration as well as by shape,
    so a protocol change breaks this class rather than one of its callers.
    """

    def __init__(self, script: Iterable[TransportScript] = (), *, egress_name: str = EGRESS_DIRECT) -> None:
        """Load the sequence this transport will answer with.

        :param script: the steps, in order. An exhausted script repeats its
            last step, so a test that does not care how many requests were
            made does not have to count them
        :ptype script: Iterable[TransportScript]
        :param egress_name: the exit this transport reports leaving by (D20)
        :ptype egress_name: str
        :raises ValueError: when the script is empty -- a transport with
            nothing to answer would fail in a way that teaches nothing
        """
        steps = list(script)
        if not steps:
            raise ValueError("ScriptedTransport needs at least one step; an empty script answers nothing")
        self._steps = deque(steps)
        self._last = steps[-1]
        self._egress_name = egress_name
        self._calls: list[dict[str, object]] = []

    @property
    def egress_name(self) -> str:
        """Report the configured exit's name.

        :return: the egress name this transport was constructed with
        :rtype: str
        """
        return self._egress_name

    @property
    def calls(self) -> Sequence[Mapping[str, object]]:
        """Every request this transport was asked to make, in order.

        Recorded so a pushdown pin can assert on the wire parameters
        without a network -- the point of a disposition saying ``pushdown``
        is that a parameter actually went out.

        :return: the recorded calls
        :rtype: Sequence[Mapping[str, object]]
        """
        return self._calls

    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str] | None = None,
        json_body: Mapping[str, JsonValue] | None = None,
        timeout_seconds: float | None = None,
    ) -> TransportResponse:
        """Answer the next scripted step.

        :param method: HTTP method, recorded
        :ptype method: str
        :param url: absolute URL, recorded
        :ptype url: str
        :param headers: request headers, recorded
        :ptype headers: Mapping[str, str] | None
        :param params: query parameters, recorded
        :ptype params: Mapping[str, str] | None
        :param json_body: JSON body, recorded
        :ptype json_body: Mapping[str, JsonValue] | None
        :param timeout_seconds: per-call bound, recorded
        :ptype timeout_seconds: float | None
        :return: the scripted response
        :rtype: TransportResponse
        :raises BaseException: whatever the scripted step carries
        """
        self._calls.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers or {}),
                "params": dict(params or {}),
                "json_body": dict(json_body or {}),
                "timeout_seconds": timeout_seconds,
            }
        )
        step = self._steps.popleft() if self._steps else self._last
        if step.failure is not None:
            raise step.failure
        return TransportResponse(
            status_code=step.status_code,
            body=step.body,
            final_url=url,
            egress=self._egress_name,
            elapsed_seconds=step.elapsed_seconds,
            attempts=step.attempts,
            headers=dict(step.headers),
        )
