"""BudgetPort -- the local refusal authority, consulted before the bill (SR-D1, SR-D2).

Two calls, in this order, around one provider call: ``check(estimate)``
before it and ``record(spend)`` after it. Both belong **below** the
transport's retry boundary, so an attempt that was retried and never billed
never debits a budget -- "budget follows the bill" (D4). Call
(:mod:`threetears.search.call`) is the consult site; the port is handed to
it, never imported by it.

**Why the estimate is a** :class:`~threetears.search.contracts.spend.Spend`.
An estimate and the eventual record must be denominated in the same units
or the cap and the bill drift apart, which is exactly the defect SR-E2
forbids ("the count a cap enforces and the count a bill prices are one
number"). Spend already carries every dimension a budget is expressed in --
calls (SR-D1's "budgets in calls, not only money"), money, weighted
provider units (SR-E4), wall-clock, bytes -- and ``Spend()`` is a zero
spend, so a free self-hosted call is estimable without synthetic pricing
(D6, SR-D6). A dedicated estimate type would be a second vocabulary for
those same five dimensions, free to disagree with the first.

**Why scopes are plural in the signature.** SR-D2: budget scopes are
plural and not interchangeable -- per-persona-per-day, per-invocation,
per-run. The tags travel on the request
(:attr:`~threetears.search.contracts.request.SearchRequest.budget_scope_tags`,
operational and deliberately outside the canonical form) and are passed
through to both methods as a required keyword: a consult that names no
scopes says so with ``()``, and never by omission.

**Why refusal is a returned decision, not a raise.** The port is the
*local* authority of D5 -- it bounds a run's shape, and a refusal here is
always :class:`~threetears.search.contracts.errors.LocalCapExceeded`-shaped,
never :class:`~threetears.search.contracts.errors.QuotaExhausted`, which is
the provider's own refusal about money and can only ever come back from a
provider call (SR-D3 requires the two to stay distinguishable). Returning
:class:`BudgetDecision` keeps that mapping where the rest of the taxonomy
mapping already lives -- in the consuming layer -- and keeps the port
minimal and structurally satisfiable: an implementation is conformant if
its shape type-checks, with no exception contract to honour. It also lets a
refusal carry the facts the typed error needs (which scope said no, what
that scope had consumed, how to fix it) instead of a bare exception a
consumer must re-decorate. The family's own precedent agrees: ``core``'s
``TokenBucket`` never raises for "not enough tokens" either, because a
refusal on a hot path is an expected per-request outcome, not an
exceptional one.

Ports are parameters, never payload (SR-L4, P9): a budget port is injected
at construction and appears in no wire type.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from threetears.search.contracts.spend import Spend

__all__ = ["BudgetDecision", "BudgetPort"]


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """What a budget said about one prospective call.

    A seam value, not a wire type: it crosses the port/consumer seam in
    process and never rides a result payload, so it is a plain frozen
    dataclass rather than a
    :class:`~threetears.search.contracts._base.ContractModel` (the same
    stance :class:`~threetears.search.contracts.transport.TransportResponse`
    takes).

    A refusal carries what
    :class:`~threetears.search.contracts.errors.LocalCapExceeded` needs to
    be honest about *which* authority said no (SR-D2, SR-D3, SR-E3).
    """

    #: whether the call may proceed. False is a local-cap refusal (D5) --
    #: never a statement about the provider's own quota.
    allowed: bool
    #: which budget scope tag refused, when one did (SR-D2). None on an
    #: allow, and on a refusal by a scope the implementation enforces
    #: unconditionally rather than by tag.
    scope: str | None = None
    #: what happened, for the human reading the failure record.
    reason: str | None = None
    #: how to fix it, where the cause is known (raise the per-run cap,
    #: widen the daily allowance) -- carried onto the typed error.
    remediation: str | None = None
    #: what the refusing scope had already consumed. Zero on an allow;
    #: on a refusal it is the spend the typed error carries (SR-E3), so a
    #: run that was stopped by a cap still reports what it cost.
    consumed: Spend = field(default_factory=Spend)


@runtime_checkable
class BudgetPort(Protocol):
    """Structural protocol for the injected budget authority (P9, D5).

    Implementations live with the consumer, which owns the budgets: an
    eval harness enforcing a per-run call cap, an agent enforcing a
    per-persona-per-day allowance, an in-process counter for a one-shot
    script. This package ships none of them and imports none of them.
    """

    async def check(self, estimate: Spend, *, scope_tags: tuple[str, ...]) -> BudgetDecision:
        """Ask whether one prospective call may proceed.

        Consulted immediately before the provider call and below the retry
        boundary (D4). An implementation MUST NOT debit anything here: the
        estimate is a question, and only :meth:`record` answers with a
        charge -- an estimate a run may freely exceed is not an estimate
        (SR-D1), but an estimate that bills is not a check.

        :param estimate: what the call is expected to consume, in the same
            dimensions the eventual record uses (SR-E2). ``Spend(calls=1)``
            is the honest estimate for an unpriced provider (D6)
        :ptype estimate: Spend
        :param scope_tags: the scopes this call debits (SR-D2), as carried
            on the request. Empty means the call named none -- the
            implementation still applies whatever scopes it enforces
            unconditionally
        :ptype scope_tags: tuple[str, ...]
        :return: the decision; a refusal names the scope that said no and
            what it had consumed, for the consumer to raise as
            :class:`~threetears.search.contracts.errors.LocalCapExceeded`
        :rtype: BudgetDecision
        """
        ...

    async def record(self, spend: Spend, *, scope_tags: tuple[str, ...]) -> None:
        """Debit what the call actually consumed.

        Called after the provider answers -- and after a typed failure
        too, which still carries the spend it incurred (SR-E3). Below the
        retry boundary, so retried-but-unbilled attempts contribute
        nothing (D4).

        :param spend: what the call consumed, as reported by the layer
            that made it -- never a locally re-derived tally (SR-E2)
        :ptype spend: Spend
        :param scope_tags: the scopes to debit (SR-D2); the same tags the
            matching :meth:`check` was given
        :ptype scope_tags: tuple[str, ...]
        :return: nothing -- a budget reports its verdict through
            :meth:`check`, not through the recording call
        :rtype: None
        """
        ...
