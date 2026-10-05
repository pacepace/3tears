"""Calls the engine makes outside any run: priced before they are made, and ledgered once they are.

Two engine calls have no run around them: a launch's case generation (an ``llm`` variation axis's
writer, :func:`~threetears.evals.gen.generate_variations`) and the rubric proposer
(:func:`~threetears.evals.gen.propose_draft`). A run's own calls are bounded by its cost cap as
their spend arrives (``EvalRunCostCap``); these two happen before any run exists, or with none
coming, so nothing would bound or record them. This module is what does:

* **Priced before the call.** :meth:`OutOfRunBudget.admit` asks the client what each planned call
  can cost at most (:meth:`~threetears.evals.contracts.provider.PricedCompletion.price_ceiling` — the
  host's answer, since the engine knows neither a model's rates nor the output cap the client was
  built with) and refuses the whole set when the ceilings together would pass the cap, before any
  of them is made. Under an enforced cap a call the client cannot price is refused too: unknown is
  not $0.
* **Ledgered after it.** :meth:`OutOfRunBudget.generate` makes an admitted call and writes one
  :class:`OutOfRunSpend` document for it — what the provider reported, the ceiling it was admitted
  at and the cap it was admitted under — whether the call returned or raised, because a raised
  call can have been billed too.

The ledger is a stored document type of its own (``eval_out_of_run_spend``) in the scope the work
was for, keyed by nothing but its id: one document per call, never rewritten.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, Self

from pydantic import Field, field_validator, model_validator

from threetears.evals.contracts.base import EvalDocumentModel
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.models import EVAL_SCHEMA_VERSION, SchemaVersion, utc_now_iso
from threetears.evals.contracts.provider import StopReason
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.provider import PricedCompletion

log = get_logger(__name__)

#: What an out-of-run call was for: ``variation`` writes a launch's generated cases (an ``llm``
#: variation axis's values), ``proposer`` drafts a rubric for operator review. Each is the
#: :data:`~threetears.evals.contracts.host.CompletionRole` the host built the client in.
OutOfRunPurpose = Literal["variation", "proposer"]

#: How an out-of-run call ended: it returned a completion, or it raised. A raised call is still a
#: ledger row — the provider may have billed it — carrying no usage, since nothing reported any.
OutOfRunOutcome = Literal["completed", "raised"]


class OutOfRunSpend(EvalDocumentModel):
    """One call the engine made outside any run, as it was admitted and as the provider reported it.

    Written by :meth:`OutOfRunBudget.generate` for every call it makes, never rewritten. **Missing is
    not zero**: a token count or a cost the provider did not report is ``None``, as on
    :class:`~threetears.evals.contracts.models.RoleUsage`, and a raised call carries no usage at all.
    """

    doc_type: Literal["eval_out_of_run_spend"] = "eval_out_of_run_spend"
    schema_version: SchemaVersion = EVAL_SCHEMA_VERSION
    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    scope_id: str = Field(min_length=1, description="The scope the work was for, which its ledger lives in.")
    purpose: OutOfRunPurpose
    model: str = Field(min_length=1, description="The model the client named — what the call was priced and asked as.")
    served_model: str | None = Field(
        default=None, description="The model the provider's response named as having answered; None when it named none."
    )
    outcome: OutOfRunOutcome
    failure: str | None = Field(
        default=None,
        description=(
            "For a raised call, the exception's class — never its text, which on some providers is the response "
            "envelope with the account in it. None for a completed call."
        ),
    )
    stop_reason: StopReason | None = Field(
        default=None, description="Why a completed call stopped, normalised; None for a raised call."
    )
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(
        default=None, ge=0.0, description="What the call cost as reported; None when unpriced."
    )
    price_source: str | None = Field(default=None, description="Where cost_usd came from, as the client named it.")
    priced_ceiling_usd: float | None = Field(
        default=None,
        ge=0.0,
        description="The most the client said the call could cost, asked before it was made; None when it could not say.",
    )
    cap_usd: float | None = Field(
        default=None,
        gt=0.0,
        description="The cap the call was admitted under; None when the host enforces no out-of-run cap.",
    )
    template_id: str | None = Field(default=None, description="The template the work was for, when it was for one.")
    subject_id: str | None = Field(default=None, description="The subject the work was for, when it was for one.")
    launch_group_id: str | None = Field(
        default=None, description="The launch a case generation was made for; its runs carry the same group id."
    )
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("doc_type")
    @classmethod
    def check_doc_type(cls, v: str) -> str:
        """Reject documents loaded into the wrong model class."""
        if v != "eval_out_of_run_spend":
            raise ValueError(f"doc_type must be 'eval_out_of_run_spend', got '{v}'")
        return v

    @model_validator(mode="after")
    def _outcome_decides_what_is_recorded(self) -> Self:
        """Refuse a raised call carrying usage or naming no failure, and a completed one carrying a failure."""
        if self.outcome == "raised":
            reported = {
                name: getattr(self, name)
                for name in ("stop_reason", "prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd")
                if getattr(self, name) is not None
            }
            if reported:
                raise ValueError(f"a raised call reported nothing, and this one carries {sorted(reported)}")
            if not self.failure:
                raise ValueError("a raised call names the exception's class in failure")
        elif self.failure is not None:
            raise ValueError("a completed call carries no failure")
        return self


class OutOfRunSpendStore(Protocol):
    """The one write the out-of-run ledger makes.

    Structural, so :class:`~threetears.evals.contracts.storage.EvalStorage` satisfies it by having the
    method. Raises ``StorageError`` on a failed write rather than returning a flag.
    """

    def save_out_of_run_spend(self, spend: OutOfRunSpend, /) -> None:
        """Write one ledger row."""
        ...


@dataclass(frozen=True)
class PlannedCall:
    """One call an out-of-run unit of work means to make: the prompt pair and the directive it sends."""

    system: str
    user: str
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class AdmittedCall:
    """A planned call its budget admitted, at the ceiling it was priced at.

    Only :meth:`OutOfRunBudget.admit` mints one, and only the budget that minted it makes the call
    (:meth:`OutOfRunBudget.generate`), so a call cannot be made without having been priced.
    """

    call: PlannedCall
    purpose: OutOfRunPurpose
    model: str
    ceiling_usd: float | None
    token: str


class RecordedCompletion(NamedTuple):
    """What an admitted call returned, and the ledger row written for it."""

    result: Any
    spend: OutOfRunSpend


@dataclass(eq=False)
class OutOfRunBudget:
    """The cap one out-of-run unit of work is held to, and the ledger its calls are written to.

    Built by whoever starts the work: the launch builds one per generating launch
    (``LaunchRequest.generation_budget``, capped at the host's ``max_out_of_run_cost_usd``), and a host
    proposing a rubric builds one for the proposal. Admissions accumulate — a second admission is
    refused when it and every earlier one together would pass the cap.

    Attributes:
        store: Where every call's ledger row is written.
        scope_id: The scope the work is for, which the rows live in.
        cap_usd: The most the work's calls may be priced at, together; ``None`` when the host enforces
            no out-of-run cap (its enforcement switched off), and then nothing is refused — every call is
            still priced where the client can say, and ledgered.
        template_id: The template the work is for, stamped on every row; ``None`` when it is for none.
        subject_id: The subject the work is for, stamped on every row; ``None`` when it is for none.
        launch_group_id: The launch a case generation is made for, stamped on every row.
    """

    store: OutOfRunSpendStore
    scope_id: str
    cap_usd: float | None
    template_id: str | None = None
    subject_id: str | None = None
    launch_group_id: str | None = None
    _committed_usd: float = field(default=0.0, init=False)
    _tokens: set[str] = field(default_factory=set, init=False)
    _recorded: list[OutOfRunSpend] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Refuse a cap that is not a positive, finite amount.

        Raises:
            ValueError: ``cap_usd`` is not ``None`` and not positive and finite.
        """
        if self.cap_usd is not None and not (math.isfinite(self.cap_usd) and self.cap_usd > 0):
            raise ValueError(f"cap_usd must be a positive amount, or None for no enforced cap; got {self.cap_usd!r}")

    @property
    def committed_usd(self) -> float:
        """The ceilings of every call admitted so far, summed — what the cap has been spent against."""
        return self._committed_usd

    @property
    def recorded(self) -> tuple[OutOfRunSpend, ...]:
        """Every ledger row this budget has written, in the order the calls ended."""
        return tuple(self._recorded)

    def quote(
        self, client: PricedCompletion, purpose: OutOfRunPurpose, calls: Sequence[PlannedCall]
    ) -> list[float | None]:
        """Price ``calls`` on ``client`` against what is left of the cap, refusing as :meth:`admit` would, committing nothing.

        What a pre-flight asks — a battery checking every template's generation before it launches any.

        Args:
            client: The client the calls would be made on.
            purpose: What the calls are for.
            calls: The calls, at least one.

        Returns:
            Each call's ceiling, in order; ``None`` where the client could not say and no cap is enforced.

        Raises:
            ValueError: ``calls`` is empty.
            ValidationFailedError: Under an enforced cap, a call the client cannot price, or ceilings that
                together with what is already committed pass the cap.
        """
        if not calls:
            raise ValueError("an admission prices at least one call")
        ceilings = [
            client.price_ceiling(system=call.system, user=call.user, response_format=call.response_format)
            for call in calls
        ]
        for ceiling in ceilings:
            if ceiling is not None and not (math.isfinite(ceiling) and ceiling >= 0):
                raise ValueError(
                    f"{client.model_name!r} priced a call at {ceiling!r}; a ceiling is a finite amount, 0 or more"
                )
        if self.cap_usd is None:
            return ceilings
        what = f"{len(calls)} {purpose} call(s) on {client.model_name!r}"
        if None in ceilings:
            raise ValidationFailedError(
                f"{what} cannot be priced before they are made: the client cannot bound what a call on "
                f"{client.model_name!r} costs (price_ceiling returned None), and the out-of-run cap ${self.cap_usd:.2f} "
                "is enforced, so an unpriced call is refused rather than made — unknown is not $0. Give the host's "
                "client a rate for that model, or name a model it can price. Nothing was called"
            )
        priced = math.fsum(c for c in ceilings if c is not None)
        if self._committed_usd + priced > self.cap_usd:
            already = f" on top of ${self._committed_usd:.4f} already admitted" if self._committed_usd else ""
            raise ValidationFailedError(
                f"{what} are priced at up to ${priced:.4f}{already}, above the out-of-run cap ${self.cap_usd:.2f}. "
                "Fewer generated cases, a cheaper model or a smaller output cap brings them under; a larger cap raises "
                "it. Nothing was called"
            )
        return ceilings

    def admit(
        self, client: PricedCompletion, purpose: OutOfRunPurpose, calls: Sequence[PlannedCall]
    ) -> list[AdmittedCall]:
        """Price ``calls`` on ``client`` and admit them together, or refuse them together before any is made.

        Args:
            client: The client the calls will be made on.
            purpose: What the calls are for.
            calls: The calls, at least one.

        Returns:
            One :class:`AdmittedCall` per call, in order, each to be made through :meth:`generate`.

        Raises:
            ValueError: ``calls`` is empty.
            ValidationFailedError: See :meth:`quote`.
        """
        ceilings = self.quote(client, purpose, calls)
        self._committed_usd += math.fsum(c for c in ceilings if c is not None)
        admitted = []
        for call, ceiling in zip(calls, ceilings, strict=True):
            token = str(uuid.uuid7())
            self._tokens.add(token)
            admitted.append(
                AdmittedCall(call=call, purpose=purpose, model=client.model_name, ceiling_usd=ceiling, token=token)
            )
        return admitted

    async def generate(self, client: PricedCompletion, admitted: AdmittedCall) -> RecordedCompletion:
        """Make one admitted call and write its ledger row, whether it returns or raises.

        Args:
            client: The client the call was admitted on.
            admitted: What :meth:`admit` returned for the call. Each is made once.

        Returns:
            The completion and its ledger row.

        Raises:
            ValueError: ``admitted`` is not one this budget admitted and has not made, or ``client`` is not
                the model it was priced on.
            StorageError: The ledger row could not be written for a completed call.
            Exception: Whatever the call raised — after its row is written.
        """
        if admitted.token not in self._tokens:
            raise ValueError(
                "this call was not admitted by this budget, or was already made; every out-of-run call is priced by "
                "its own budget's admit() and made once"
            )
        if client.model_name != admitted.model:
            raise ValueError(
                f"the call was priced on {admitted.model!r} and is being made on {client.model_name!r}; make it on the "
                "client it was admitted on"
            )
        self._tokens.discard(admitted.token)
        call = admitted.call
        try:
            result = await client.generate(system=call.system, user=call.user, response_format=call.response_format)
        except BaseException as raised:
            try:
                self._record(admitted, outcome="raised", failure=type(raised).__name__)
            # prawduct:ok-broad-except — a failed ledger write must not replace the call's own failure, which propagates
            except Exception as unrecorded:
                log.error(
                    "eval.out_of_run %s call on %s raised %s and its ledger row could NOT be written: %s",
                    admitted.purpose,
                    admitted.model,
                    type(raised).__name__,
                    unrecorded,
                )
            raise
        spend = self._record(
            admitted,
            outcome="completed",
            served_model=getattr(result, "served_model", None),
            stop_reason=getattr(result, "stop_reason", None),
            prompt_tokens=getattr(result, "input_tokens", None),
            completion_tokens=getattr(result, "output_tokens", None),
            reasoning_tokens=getattr(result, "reasoning_tokens", None),
            cost_usd=getattr(result, "cost_usd", None),
            price_source=getattr(result, "price_source", None),
        )
        return RecordedCompletion(result, spend)

    def _record(self, admitted: AdmittedCall, **reported: Any) -> OutOfRunSpend:
        spend = OutOfRunSpend(
            scope_id=self.scope_id,
            purpose=admitted.purpose,
            model=admitted.model,
            priced_ceiling_usd=admitted.ceiling_usd,
            cap_usd=self.cap_usd,
            template_id=self.template_id,
            subject_id=self.subject_id,
            launch_group_id=self.launch_group_id,
            **reported,
        )
        self.store.save_out_of_run_spend(spend)
        self._recorded.append(spend)
        log.info(
            "eval.out_of_run purpose=%s model=%s outcome=%s cost=%s ceiling=%s cap=%s scope=%s",
            spend.purpose,
            spend.model,
            spend.outcome,
            spend.cost_usd,
            spend.priced_ceiling_usd,
            spend.cap_usd,
            spend.scope_id,
        )
        return spend


__all__ = [
    "AdmittedCall",
    "OutOfRunBudget",
    "OutOfRunOutcome",
    "OutOfRunPurpose",
    "OutOfRunSpend",
    "OutOfRunSpendStore",
    "PlannedCall",
    "RecordedCompletion",
]
