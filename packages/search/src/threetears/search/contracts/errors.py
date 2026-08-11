"""The typed failure taxonomy -- distinguishable, never merged (SR-J1, D10).

Seven failure classes, each its own exception type, each carrying the
:class:`~threetears.search.contracts.spend.Spend` the failed call still
consumed (SR-E3 -- a run that broke halfway still incurred whatever it
incurred). Errors carry remediation text where the cause is known and
fixable (the SearXNG 403-when-``json``-missing teaching error, the #1
setup failure).

On the seven: SR-J1's source taxonomy lists zero-results among its
distinguishable outcome classes, but SR-J2 pins zero results as a
*success value* -- an empty ``CandidateSet``, never an exception. The
seventh *error* class is therefore the local-cap refusal, which SR-D3 and
D5 require to be distinguishable from the provider's own quota refusal:
local caps bound a run's shape, provider refusal bounds money, and
merging them would hide which authority said no.

Exceptions are for in-process seams only. Nothing raises across the wire
(D10): Bind converts every one of these to a failed ``ToolResult`` with
spend on ``metadata``, via :class:`FailureRecord` -- the JSON-safe
projection that round-trips the whole taxonomy (SR-L4).
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Final

from pydantic import AwareDatetime

from threetears.search.contracts._base import ContractModel
from threetears.search.contracts.spend import Spend

__all__ = [
    "FAILURE_CLASSES",
    "AuthFailed",
    "FailureRecord",
    "LocalCapExceeded",
    "MalformedResponse",
    "QuotaExhausted",
    "RateLimited",
    "SearchFailure",
    "TimedOut",
    "TransportFailed",
]


class SearchFailure(Exception):
    """Base of the typed search failure taxonomy.

    Never raised directly -- one of the seven subclasses names the failure
    class. Every instance carries the spend the failed call consumed
    (SR-E3) and, where the cause is known, remediation text.
    """

    #: wire-stable name of the failure class; each concrete subclass
    #: declares its own.
    failure_class: ClassVar[str] = "search-failure"

    def __init__(
        self,
        message: str,
        *,
        spend: Spend,
        provider_instance: str | None = None,
        remediation: str | None = None,
        egress: str | None = None,
        occurred_at: datetime | None = None,
    ) -> None:
        """Record the failure with what it cost.

        :param message: what happened, for a human reading the record
        :ptype message: str
        :param spend: what the failed call consumed before failing (SR-E3)
        :ptype spend: Spend
        :param provider_instance: which provider instance failed, when one
            was involved
        :ptype provider_instance: str | None
        :param remediation: how to fix it, where the cause is known
        :ptype remediation: str | None
        :param egress: which egress the failing call left by, when the
            transport is known (D8/D20 -- rate and ban budgets key on
            ``(provider instance, egress)``, and pod-resident this record
            is the only fact that survives the wire)
        :ptype egress: str | None
        :param occurred_at: when the failure happened (timezone-aware),
            when the failing site can say
        :ptype occurred_at: datetime | None
        """
        super().__init__(message)
        self.message = message
        self.spend = spend
        self.provider_instance = provider_instance
        self.remediation = remediation
        self.egress = egress
        self.occurred_at = occurred_at

    def to_record(self) -> FailureRecord:
        """Project this failure to its JSON-safe wire record.

        :return: the record Bind puts on a failed result's metadata (D10)
        :rtype: FailureRecord
        """
        return FailureRecord(
            failure_class=self.failure_class,
            message=self.message,
            spend=self.spend,
            provider_instance=self.provider_instance,
            remediation=self.remediation,
            egress=self.egress,
            occurred_at=self.occurred_at,
        )


class RateLimited(SearchFailure):
    """The provider is pacing us: slow down and retry later (429-class)."""

    failure_class: ClassVar[str] = "rate-limited"

    def __init__(
        self,
        message: str,
        *,
        spend: Spend,
        provider_instance: str | None = None,
        remediation: str | None = None,
        egress: str | None = None,
        occurred_at: datetime | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        """Record a rate-limit refusal.

        :param message: what happened
        :ptype message: str
        :param spend: what the call consumed (SR-E3)
        :ptype spend: Spend
        :param provider_instance: which provider instance refused
        :ptype provider_instance: str | None
        :param remediation: how to fix it, where known
        :ptype remediation: str | None
        :param egress: which egress the refused call left by, when known
            (D8 -- pacing keys on it)
        :ptype egress: str | None
        :param occurred_at: when the refusal happened (timezone-aware)
        :ptype occurred_at: datetime | None
        :param retry_after_seconds: the provider's stated backoff, when it
            gave one
        :ptype retry_after_seconds: float | None
        """
        super().__init__(
            message,
            spend=spend,
            provider_instance=provider_instance,
            remediation=remediation,
            egress=egress,
            occurred_at=occurred_at,
        )
        self.retry_after_seconds = retry_after_seconds

    def to_record(self) -> FailureRecord:
        """Project this failure, keeping the provider's stated backoff.

        :return: the JSON-safe wire record
        :rtype: FailureRecord
        """
        return super().to_record().model_copy(update={"retry_after_seconds": self.retry_after_seconds})


class QuotaExhausted(SearchFailure):
    """The provider's own refusal: the paid/allotted quota is spent.

    Distinct from :class:`LocalCapExceeded` by authority (D5): this bounds
    money, and it short-circuits -- a dead search backend is an outage,
    not a per-query warning (SR-D3).
    """

    failure_class: ClassVar[str] = "quota-exhausted"


class AuthFailed(SearchFailure):
    """The provider rejected our credentials or authorization."""

    failure_class: ClassVar[str] = "auth-failed"


class TimedOut(SearchFailure):
    """The call exceeded its deadline. Retry is reasonable (SR-J1)."""

    failure_class: ClassVar[str] = "timed-out"


class TransportFailed(SearchFailure):
    """The request could not complete at the transport: connect failure,
    TLS failure, 5xx-after-retries. Give up rather than retry (SR-J1)."""

    failure_class: ClassVar[str] = "transport-failed"


class MalformedResponse(SearchFailure):
    """The provider answered, but not in the shape its API promises.

    Carries remediation where the malformation is a known misconfiguration
    -- SearXNG answering 403 because ``json`` is missing from its
    configured formats is the canonical teaching case (SR-J1).
    """

    failure_class: ClassVar[str] = "malformed-response"


class LocalCapExceeded(SearchFailure):
    """A locally-configured cap refused the call before the provider saw it.

    The second refusal authority of D5: local caps bound a run's *shape*
    (an overrun is a defect in the run, not a billing event), and SR-D3
    requires this to be distinguishable from :class:`QuotaExhausted`.
    """

    failure_class: ClassVar[str] = "local-cap-exceeded"

    def __init__(
        self,
        message: str,
        *,
        spend: Spend,
        provider_instance: str | None = None,
        remediation: str | None = None,
        egress: str | None = None,
        occurred_at: datetime | None = None,
        scope: str | None = None,
    ) -> None:
        """Record a local-cap refusal.

        :param message: what happened
        :ptype message: str
        :param spend: what the run had consumed in the refusing scope
        :ptype spend: Spend
        :param provider_instance: the provider instance the refused call
            was bound for, when known
        :ptype provider_instance: str | None
        :param remediation: how to fix it, where known
        :ptype remediation: str | None
        :param egress: which egress the refused call would have left by,
            when known
        :ptype egress: str | None
        :param occurred_at: when the refusal happened (timezone-aware)
        :ptype occurred_at: datetime | None
        :param scope: which budget scope tag refused (SR-D2)
        :ptype scope: str | None
        """
        super().__init__(
            message,
            spend=spend,
            provider_instance=provider_instance,
            remediation=remediation,
            egress=egress,
            occurred_at=occurred_at,
        )
        self.scope = scope

    def to_record(self) -> FailureRecord:
        """Project this failure, keeping the refusing scope.

        :return: the JSON-safe wire record
        :rtype: FailureRecord
        """
        return super().to_record().model_copy(update={"scope": self.scope})


#: the seven concrete failure classes, keyed by wire name. The taxonomy is
#: additive within a family minor (D13); readers meeting an unknown class
#: get a loud error from :meth:`FailureRecord.to_failure`, never a silent
#: reclassification.
FAILURE_CLASSES: Final[dict[str, type[SearchFailure]]] = {
    cls.failure_class: cls
    for cls in (
        RateLimited,
        QuotaExhausted,
        AuthFailed,
        TimedOut,
        TransportFailed,
        MalformedResponse,
        LocalCapExceeded,
    )
}


class FailureRecord(ContractModel):
    """JSON-safe projection of one :class:`SearchFailure` (SR-L4, D10).

    This is what crosses a border: Bind renders it onto a failed
    ``ToolResult``'s metadata so spend survives the wire (SR-E3), and it
    rebuilds the typed exception on the far side of any in-process store.
    """

    #: wire name of the failure class (a :data:`FAILURE_CLASSES` key).
    failure_class: str
    #: what happened.
    message: str
    #: what the failed call consumed (SR-E3).
    spend: Spend
    #: which provider instance was involved, when one was.
    provider_instance: str | None = None
    #: how to fix it, where the cause is known.
    remediation: str | None = None
    #: which egress the failing call left by, when the transport is known
    #: (D8/D20). Rate and ban budgets key on ``(provider instance,
    #: egress)``, and a consumer-side pacing or ban tracker reading this
    #: record off ``ToolResult.metadata`` has nothing else to rebuild the
    #: key from -- pod-resident, this record is the only surviving fact.
    egress: str | None = None
    #: when the failure happened (timezone-aware), when the failing site
    #: could say. Provenance on the spend record, per §3.1's
    #: provenance-on-every-spend-record rule (P2, SR-A3).
    occurred_at: AwareDatetime | None = None
    #: provider-stated backoff; only meaningful for ``rate-limited``.
    retry_after_seconds: float | None = None
    #: refusing budget scope; only meaningful for ``local-cap-exceeded``.
    scope: str | None = None

    def to_failure(self) -> SearchFailure:
        """Rebuild the typed exception this record projects.

        :return: the concrete :class:`SearchFailure` subclass instance
        :rtype: SearchFailure
        :raises ValueError: when ``failure_class`` names a class this
            contract version does not know -- refused loudly, never
            best-effort misread (D26's versioned-refusal discipline)
        """
        failure_type = FAILURE_CLASSES.get(self.failure_class)
        if failure_type is None:
            raise ValueError(
                f"unknown failure class {self.failure_class!r}; this reader knows {sorted(FAILURE_CLASSES)}"
            )
        if failure_type is RateLimited:
            return RateLimited(
                self.message,
                spend=self.spend,
                provider_instance=self.provider_instance,
                remediation=self.remediation,
                egress=self.egress,
                occurred_at=self.occurred_at,
                retry_after_seconds=self.retry_after_seconds,
            )
        if failure_type is LocalCapExceeded:
            return LocalCapExceeded(
                self.message,
                spend=self.spend,
                provider_instance=self.provider_instance,
                remediation=self.remediation,
                egress=self.egress,
                occurred_at=self.occurred_at,
                scope=self.scope,
            )
        return failure_type(
            self.message,
            spend=self.spend,
            provider_instance=self.provider_instance,
            remediation=self.remediation,
            egress=self.egress,
            occurred_at=self.occurred_at,
        )
