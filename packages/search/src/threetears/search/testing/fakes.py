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

from threetears.search.contracts import (
    EGRESS_DIRECT,
    BudgetDecision,
    BudgetPort,
    RateLimitDecision,
    RateLimiterPort,
    SearchTransport,
    Spend,
    TransportResponse,
)

__all__ = ["FakeBudgetPort", "FakeRateLimiterPort", "ScriptedTransport", "TransportScript"]


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


class FakeBudgetPort(BudgetPort):  # parity-with: threetears.search.contracts.BudgetPort
    """A budget that answers one fixed decision, recording every call it saw.

    Deliberately not a ledger: a case-by-case simulation of caps and scopes
    already lives in ``test_ports.py``, which pins the port's own contract.
    What a *consumer* of the port -- a Call wiring test, chiefly -- needs
    instead is a double it can hand a canned :class:`BudgetDecision` to and
    then interrogate for what it was asked, so it can assert that Call
    consulted :meth:`check` before the provider call and :meth:`record`
    after, with the estimate/spend and scope tags it expected (SR-D1, SR-D2,
    D4, D5).
    """

    def __init__(self, decision: BudgetDecision | None = None) -> None:
        """Fix the answer every :meth:`check` will give.

        :param decision: the decision to return from every call; defaults to
            an unconditional allow, since a wiring test that cares about a
            refusal configures one explicitly
        :ptype decision: BudgetDecision | None
        """
        self.decision = decision if decision is not None else BudgetDecision(allowed=True)
        #: every ``(estimate, scope_tags)`` pair passed to :meth:`check`, in order.
        self.checks: list[tuple[Spend, tuple[str, ...]]] = []
        #: every ``(spend, scope_tags)`` pair passed to :meth:`record`, in order.
        self.records: list[tuple[Spend, tuple[str, ...]]] = []

    async def check(self, estimate: Spend, *, scope_tags: tuple[str, ...]) -> BudgetDecision:
        """Record the call and return the configured decision.

        :param estimate: what the prospective call is expected to consume
        :ptype estimate: Spend
        :param scope_tags: the scopes this call would debit (SR-D2)
        :ptype scope_tags: tuple[str, ...]
        :return: the decision this fake was constructed with
        :rtype: BudgetDecision
        """
        self.checks.append((estimate, scope_tags))
        return self.decision

    async def record(self, spend: Spend, *, scope_tags: tuple[str, ...]) -> None:
        """Record what the caller reported as spent, and nothing else.

        A refusal a budget answers through :meth:`check` never suppresses a
        later :meth:`record` -- this fake is not the authority that decides
        whether recording should have happened, only a witness to whether it
        did (SR-E3: a failure still reports what it consumed).

        :param spend: what the call consumed, as the calling layer reports it
        :ptype spend: Spend
        :param scope_tags: the scopes to debit (SR-D2)
        :ptype scope_tags: tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        self.records.append((spend, scope_tags))


class FakeRateLimiterPort(RateLimiterPort):  # parity-with: threetears.search.contracts.RateLimiterPort
    """A limiter that answers one fixed decision, recording every key it paced.

    Like :class:`FakeBudgetPort`, this is a witness rather than a simulation
    -- ``test_ports.py`` already pins the pacing behaviour a real limiter
    must have (D8, D20). A Call wiring test wants to assert *that* Call
    called :meth:`acquire` keyed on ``(provider_instance, egress)`` before
    the provider call, and to drive both the granted and the denied path by
    configuring the decision -- not to re-derive token-bucket arithmetic.
    """

    def __init__(self, decision: RateLimitDecision | None = None) -> None:
        """Fix the answer every :meth:`acquire` will give.

        :param decision: the decision to return from every call; defaults to
            an unconditional grant, since a wiring test that cares about a
            denial configures one explicitly
        :ptype decision: RateLimitDecision | None
        """
        self.decision = decision if decision is not None else RateLimitDecision(acquired=True)
        #: every acquisition asked for, as ``(provider_instance, egress, tokens,
        #: max_wait_seconds)``, in order (D8's key is the pair; the last two are
        #: kept too, since a weighted call asks for more than one token -- SR-E4's
        #: pacing analogue).
        self.acquisitions: list[tuple[str, str, float, float]] = []

    async def acquire(
        self,
        *,
        provider_instance: str,
        egress: str,
        tokens: float = 1.0,
        max_wait_seconds: float = 0.0,
    ) -> RateLimitDecision:
        """Record the call and return the configured decision.

        :param provider_instance: the deployment about to be called
        :ptype provider_instance: str
        :param egress: the exit the call will leave by (D20)
        :ptype egress: str
        :param tokens: how much of the key's allowance this call consumes
        :ptype tokens: float
        :param max_wait_seconds: how long the caller will block for permission
        :ptype max_wait_seconds: float
        :return: the decision this fake was constructed with
        :rtype: RateLimitDecision
        """
        self.acquisitions.append((provider_instance, egress, tokens, max_wait_seconds))
        return self.decision
