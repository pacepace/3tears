"""The two injected ports of search-spec.md §3.1: budget and pacing.

``BudgetPort`` (SR-D1, SR-D2, D4, D5) and ``RateLimiterPort`` (D8, D20) are
structural seams (P9): a consumer satisfies them by shape, and this package
imports no implementation of either. What is worth pinning is therefore not
behaviour -- the fakes here have none worth the name -- but the properties
the shapes are supposed to make expressible:

- an estimate and a record are denominated in one type, so the count a cap
  enforces and the count a bill prices cannot drift (SR-E2);
- budget scopes are plural and not interchangeable (SR-D2);
- a budget refusal is local and stays distinguishable from the provider's
  own quota refusal (SR-D3, D5);
- a zero-cost provider is still bounded (SR-D6);
- pacing is keyed on the *pair* ``(provider instance, egress)`` (D8, D20);
- ``core``'s NATS ``TokenBucket`` satisfies the limiter port through a thin
  host-side adapter -- the reason the port has the shape it has.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pytest

from threetears.search.contracts import (
    EGRESS_DIRECT,
    BudgetDecision,
    BudgetPort,
    LocalCapExceeded,
    QuotaExhausted,
    RateLimitDecision,
    RateLimiterPort,
    Spend,
)


class FakeBudget(BudgetPort):  # parity-with: threetears.search.contracts.BudgetPort
    """A per-scope call-count ledger -- SR-D1's "budgets in calls" in ten lines."""

    def __init__(self, call_caps: dict[str, int]) -> None:
        """Hold one call cap per scope tag.

        :param call_caps: cap on ``Spend.calls``, keyed by scope tag
        :ptype call_caps: dict[str, int]
        """
        self.call_caps = call_caps
        self.ledger: dict[str, Spend] = {tag: Spend() for tag in call_caps}
        self.checks: list[tuple[Spend, tuple[str, ...]]] = []

    async def check(self, estimate: Spend, *, scope_tags: tuple[str, ...]) -> BudgetDecision:
        """Refuse when any named scope cannot absorb the estimate.

        :param estimate: what the prospective call is expected to consume
        :ptype estimate: Spend
        :param scope_tags: the scopes this call debits
        :ptype scope_tags: tuple[str, ...]
        :return: the decision, naming the refusing scope on a refusal
        :rtype: BudgetDecision
        """
        self.checks.append((estimate, scope_tags))
        for tag in scope_tags:
            spent = self.ledger.get(tag, Spend())
            if spent.calls + estimate.calls > self.call_caps.get(tag, 0):
                return BudgetDecision(
                    allowed=False,
                    scope=tag,
                    reason=f"{tag} allows {self.call_caps.get(tag, 0)} calls",
                    remediation=f"raise the call cap for {tag}",
                    consumed=spent,
                )
        return BudgetDecision(allowed=True)

    async def record(self, spend: Spend, *, scope_tags: tuple[str, ...]) -> None:
        """Debit the named scopes by what the call actually consumed.

        :param spend: what the call consumed, as the calling layer reports it
        :ptype spend: Spend
        :param scope_tags: the scopes to debit
        :ptype scope_tags: tuple[str, ...]
        :return: nothing
        :rtype: None
        """
        for tag in scope_tags:
            self.ledger[tag] = self.ledger.get(tag, Spend()) + spend


class FakeRateLimiter(RateLimiterPort):  # parity-with: threetears.search.contracts.RateLimiterPort
    """A limiter with one fixed allowance per ``(provider instance, egress)`` key."""

    def __init__(self, allowance: float = 1.0) -> None:
        """Give every key the same starting allowance.

        :param allowance: tokens each key starts with
        :ptype allowance: float
        """
        self.allowance = allowance
        self.remaining: dict[tuple[str, str], float] = {}
        self.keys_seen: list[tuple[str, str]] = []

    async def acquire(
        self,
        *,
        provider_instance: str,
        egress: str,
        tokens: float = 1.0,
        max_wait_seconds: float = 0.0,
    ) -> RateLimitDecision:
        """Spend tokens from the key's allowance, never blocking.

        :param provider_instance: the deployment about to be called
        :ptype provider_instance: str
        :param egress: the exit the call will leave by
        :ptype egress: str
        :param tokens: how much of the allowance this call consumes
        :ptype tokens: float
        :param max_wait_seconds: ignored -- this fake never waits
        :ptype max_wait_seconds: float
        :return: the decision for this key
        :rtype: RateLimitDecision
        """
        key = (provider_instance, egress)
        self.keys_seen.append(key)
        left = self.remaining.get(key, self.allowance)
        if left < tokens:
            return RateLimitDecision(acquired=False, retry_after_seconds=tokens - left)
        self.remaining[key] = left - tokens
        return RateLimitDecision(acquired=True)


@dataclass(frozen=True, slots=True)
class _ClaimResult:
    """Stand-in for ``core``'s ``TokenClaimResult`` -- the three fields it returns."""

    claimed: bool
    tokens_remaining: float
    retry_after_seconds: float


# parity-with: threetears.core.coordination.token_bucket.TokenBucket
class FakeTokenBucket:
    """``core``'s distributed bucket, reduced to the surface an adapter touches.

    Carries the real parity marker on purpose: the claim below is only proof
    that the port is satisfiable by a *thin* adapter for as long as it is
    still the shape ``core`` actually offers.
    """

    def __init__(self, tokens: float = 1.0) -> None:
        """Start every key at the same token count.

        :param tokens: tokens each key begins with
        :ptype tokens: float
        """
        self.tokens: dict[str, float] = {}
        self.start = tokens
        self.claims: list[tuple[str, float, float]] = []

    async def claim(self, key: str = "default", *, tokens: float = 1.0, max_wait_seconds: float = 0.0) -> _ClaimResult:
        """Consume tokens from bucket ``key``, returning the outcome rather than raising.

        :param key: bucket key; independent buckets never interact
        :ptype key: str
        :param tokens: number of tokens this claim consumes
        :ptype tokens: float
        :param max_wait_seconds: seconds the caller will block for
        :ptype max_wait_seconds: float
        :return: the claim outcome
        :rtype: _ClaimResult
        """
        self.claims.append((key, tokens, max_wait_seconds))
        left = self.tokens.get(key, self.start)
        if left < tokens:
            return _ClaimResult(claimed=False, tokens_remaining=left, retry_after_seconds=tokens - left)
        self.tokens[key] = left - tokens
        return _ClaimResult(claimed=True, tokens_remaining=left - tokens, retry_after_seconds=0.0)

    async def refund(self, key: str = "default", *, tokens: float = 1.0) -> float:
        """Put back tokens claimed for work that never happened.

        :param key: bucket key to credit
        :ptype key: str
        :param tokens: tokens to put back
        :ptype tokens: float
        :return: the resulting token count
        :rtype: float
        """
        credited = self.tokens.get(key, self.start) + tokens
        self.tokens[key] = credited
        return credited


class BucketPacer:
    """The thin host-side adapter D8 describes: a key derivation and a rename.

    Deliberately declares no base class -- satisfying
    :class:`RateLimiterPort` by shape is the property under test, and it is
    what lets ``core``'s bucket be the distributed implementation without
    this leaf importing ``threetears.core`` (SR-L7).
    """

    def __init__(self, bucket: FakeTokenBucket) -> None:
        """Wrap one bucket.

        :param bucket: the ``TokenBucket``-shaped primitive to claim from
        :ptype bucket: FakeTokenBucket
        """
        self._bucket = bucket

    async def acquire(
        self,
        *,
        provider_instance: str,
        egress: str,
        tokens: float = 1.0,
        max_wait_seconds: float = 0.0,
    ) -> RateLimitDecision:
        """Claim from the bucket keyed on the D8 pair.

        :param provider_instance: the deployment about to be called
        :ptype provider_instance: str
        :param egress: the exit the call will leave by
        :ptype egress: str
        :param tokens: how much of the key's allowance this call consumes
        :ptype tokens: float
        :param max_wait_seconds: how long the caller will block
        :ptype max_wait_seconds: float
        :return: the claim outcome, in the port's vocabulary
        :rtype: RateLimitDecision
        """
        outcome = await self._bucket.claim(
            f"{provider_instance}|{egress}",
            tokens=tokens,
            max_wait_seconds=max_wait_seconds,
        )
        return RateLimitDecision(acquired=outcome.claimed, retry_after_seconds=outcome.retry_after_seconds)


async def test_budget_port_is_satisfiable_by_shape() -> None:
    """P9: an injected budget needs no base class, only the two methods."""
    budget = FakeBudget({"run:eval-042": 2})
    assert isinstance(budget, BudgetPort)
    assert (await budget.check(Spend(calls=1), scope_tags=("run:eval-042",))).allowed


async def test_the_estimate_and_the_record_are_one_type() -> None:
    """SR-E2: the count a cap enforces and the count a bill prices are one number.

    The port takes :class:`Spend` on both sides, so an estimate cannot be
    denominated in units the eventual record does not use.
    """
    budget = FakeBudget({"run:eval-042": 3})
    estimate = Spend(calls=1, provider_units=Decimal("2"))

    assert (await budget.check(estimate, scope_tags=("run:eval-042",))).allowed
    await budget.record(
        Spend(calls=1, money=Decimal("0.02"), provider_units=Decimal("2")), scope_tags=("run:eval-042",)
    )

    recorded = budget.ledger["run:eval-042"]
    assert recorded.calls == 1
    assert recorded.provider_units == Decimal("2")


async def test_budget_follows_the_bill_not_the_attempts() -> None:
    """D4: what is recorded is the billed spend, not the transport's attempts.

    Three attempts behind one billed call debit the budget once -- the port
    sits below the retry boundary and is handed the spend the calling layer
    reports, never a locally re-derived tally.
    """
    budget = FakeBudget({"run:eval-042": 2})
    billed_after_two_retries = Spend(calls=1, money=Decimal("0.02"), wall_clock_seconds=3.0)

    await budget.record(billed_after_two_retries, scope_tags=("run:eval-042",))

    assert budget.ledger["run:eval-042"].calls == 1


async def test_scopes_are_plural_and_not_interchangeable() -> None:
    """SR-D2: per-run and per-persona are separate authorities, debited together."""
    budget = FakeBudget({"run:eval-042": 1, "persona:capy": 5})
    scopes = ("run:eval-042", "persona:capy")

    assert (await budget.check(Spend(calls=1), scope_tags=scopes)).allowed
    await budget.record(Spend(calls=1), scope_tags=scopes)

    exhausted = await budget.check(Spend(calls=1), scope_tags=scopes)
    assert not exhausted.allowed
    assert exhausted.scope == "run:eval-042", "the refusal must name which scope said no"
    assert (await budget.check(Spend(calls=1), scope_tags=("persona:capy",))).allowed, (
        "the persona scope has four calls left; a refusal in one scope is not a refusal in another"
    )


async def test_naming_no_scope_is_stated_not_omitted() -> None:
    """SR-D2: the tags are a required argument, so ``()`` is a statement."""
    budget = FakeBudget({"run:eval-042": 0})

    decision = await budget.check(Spend(calls=1), scope_tags=())

    assert decision.allowed
    assert budget.checks[-1][1] == ()


async def test_a_refusal_stays_distinguishable_from_provider_quota() -> None:
    """SR-D3/D5: the budget port refuses locally; only a provider exhausts quota."""
    budget = FakeBudget({"run:eval-042": 0})

    decision = await budget.check(Spend(calls=1), scope_tags=("run:eval-042",))
    assert not decision.allowed

    failure = LocalCapExceeded(
        decision.reason or "budget refused the call",
        spend=decision.consumed,
        remediation=decision.remediation,
        scope=decision.scope,
    )
    assert not isinstance(failure, QuotaExhausted)
    assert failure.to_record().scope == "run:eval-042"
    assert failure.to_record().failure_class == "local-cap-exceeded"


async def test_a_refusal_carries_what_the_scope_had_consumed() -> None:
    """SR-E3: a run stopped by a cap still reports what it cost."""
    budget = FakeBudget({"run:eval-042": 1})
    await budget.record(Spend(calls=1, money=Decimal("0.02")), scope_tags=("run:eval-042",))

    decision = await budget.check(Spend(calls=1), scope_tags=("run:eval-042",))

    assert not decision.allowed
    assert decision.consumed == Spend(calls=1, money=Decimal("0.02"))
    assert decision.remediation is not None


async def test_a_free_provider_is_still_bounded() -> None:
    """SR-D6/D6: a self-hosted instance costs nothing, and a call cap still fires.

    A budget keyed only on money never fires for SearXNG; the estimate here
    carries zero money and is refused on ``calls`` alone (SR-D1).
    """
    budget = FakeBudget({"run:eval-042": 1})
    free = Spend(calls=1)
    assert free.money == Decimal("0")

    await budget.record(free, scope_tags=("run:eval-042",))
    decision = await budget.check(free, scope_tags=("run:eval-042",))

    assert not decision.allowed


def test_a_budget_decision_is_a_frozen_seam_value() -> None:
    """Ports are parameters and their answers are facts: no in-place edits."""
    decision = BudgetDecision(allowed=True)

    assert decision.consumed == Spend()
    with pytest.raises((AttributeError, TypeError)):
        decision.allowed = False  # type: ignore[misc]


async def test_rate_limiter_is_satisfiable_by_shape() -> None:
    """P9/D8: an injected limiter needs no base class, only ``acquire``."""
    limiter = FakeRateLimiter()
    assert isinstance(limiter, RateLimiterPort)

    decision = await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)

    assert decision.acquired
    assert decision.retry_after_seconds == 0.0


async def test_pacing_is_keyed_on_the_pair_not_the_provider() -> None:
    """D8/D20: two exits out of one instance are two allowances, not one.

    Collapsing them would pace the wrong thing -- an upstream sees the exit,
    and ``direct`` is a named egress rather than the absence of one.
    """
    limiter = FakeRateLimiter(allowance=1.0)

    assert (await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)).acquired
    assert (await limiter.acquire(provider_instance="searxng-main", egress="warp")).acquired, (
        "a second egress out of the same instance has its own allowance"
    )

    denied = await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)
    assert not denied.acquired
    assert denied.retry_after_seconds > 0.0
    assert limiter.keys_seen == [
        ("searxng-main", EGRESS_DIRECT),
        ("searxng-main", "warp"),
        ("searxng-main", EGRESS_DIRECT),
    ]


async def test_two_instances_of_one_product_are_paced_separately() -> None:
    """SR-N4: two SearXNG deployments are two instances, banned separately."""
    limiter = FakeRateLimiter(allowance=1.0)

    assert (await limiter.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)).acquired
    assert (await limiter.acquire(provider_instance="searxng-spare", egress=EGRESS_DIRECT)).acquired


async def test_a_thin_token_bucket_adapter_satisfies_the_port() -> None:
    """§3.9: ``core``'s NATS ``TokenBucket``, host-injected, where a bus exists.

    The adapter derives the D8 key from the pair, forwards the two knobs
    unchanged, and renames three fields. Nothing else -- if the port needed
    more than that, the distributed implementation would be a reimplementation.
    """
    bucket = FakeTokenBucket(tokens=1.0)
    pacer = BucketPacer(bucket)
    port: RateLimiterPort = pacer
    assert isinstance(pacer, RateLimiterPort)

    granted = await port.acquire(provider_instance="searxng-main", egress="warp", max_wait_seconds=2.5)

    assert granted.acquired
    assert bucket.claims == [("searxng-main|warp", 1.0, 2.5)]


async def test_the_adapter_carries_the_bucket_backoff_through() -> None:
    """A denial is a value, not an exception, on both sides of the adapter.

    ``retry_after_seconds`` is what a caller backs off by, and what a typed
    ``RateLimited`` failure reports if the caller gives up instead.
    """
    bucket = FakeTokenBucket(tokens=1.0)
    port: RateLimiterPort = BucketPacer(bucket)
    await port.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT)

    denied = await port.acquire(provider_instance="searxng-main", egress=EGRESS_DIRECT, tokens=1.0)

    assert not denied.acquired
    assert denied.retry_after_seconds == pytest.approx(1.0)


async def test_weighted_calls_can_cost_more_than_one_token() -> None:
    """SR-E4's weighted units have a pacing analogue: an advanced call costs two."""
    limiter = FakeRateLimiter(allowance=2.0)

    assert (await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT, tokens=2.0)).acquired
    assert not (await limiter.acquire(provider_instance="tavily", egress=EGRESS_DIRECT)).acquired


def test_a_rate_limit_decision_is_a_frozen_seam_value() -> None:
    """The limiter's answer is a fact too, and it never rides a payload."""
    decision = RateLimitDecision(acquired=True)

    assert decision.retry_after_seconds == 0.0
    with pytest.raises((AttributeError, TypeError)):
        decision.acquired = False  # type: ignore[misc]
