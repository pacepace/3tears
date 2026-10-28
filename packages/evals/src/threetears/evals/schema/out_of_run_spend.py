"""The ledger of calls the engine makes outside any run: one stored document per call, never rewritten.

Five engine calls have no run around them — a launch's case generation, the rubric proposer, an
analysis generation, a judge repeat and a second judge. Each is priced before it is made and
ledgered once it is (:mod:`threetears.evals.kernel.out_of_run`); this module is the ledger's
stored shape, :class:`OutOfRunSpend`, and the store port that writes and reads it.

The ledger is a stored document type of its own (``eval_out_of_run_spend``) in the scope the work
was for, keyed by nothing but its id. It records what the provider reported, the ceiling the call
was admitted at and the cap it was admitted under, whether the call returned or raised; an
attribute the ledger cannot hold (a raw provider stop reason, a negative count) is recorded as
unreadable, never a reason to drop the row of a call that was paid for.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Annotated, Final, Literal, Protocol, Self, get_args

from pydantic import Field, field_validator, model_validator
from threetears.evals.schema.base import CoreDocumentModel, known_unless_newer_minor
from threetears.evals.schema.models import ClientRequestSettings, CoreSchemaVersion, utc_now_iso
from threetears.evals.schema.versioning import CORE_SCHEMA_VERSION
from threetears.evals.schema.completion import StopReason

if TYPE_CHECKING:
    pass


#: What an out-of-run call was for: ``variation`` writes a launch's generated cases (an ``llm``
#: variation axis's values), ``proposer`` drafts a rubric for operator review, ``analysis`` writes a
#: campaign's analysis memo (its first call and the one repair round-trip a refused output buys), ``judge``
#: repeats a finished run's judge scores to measure the judge's agreement with itself
#: (:func:`~threetears.evals.run.judge_repeat_work`), or at two temperatures to measure what temperature does to it
#: (:func:`~threetears.evals.run.judge_temperature_work`), ``second_judge`` asks a judge other than the run's to score
#: a finished run's evidence (:func:`~threetears.evals.run.second_judge_work`) — measurement cost on its own line, never
#: the candidate's. Each but ``second_judge`` is the :data:`~threetears.evals.kernel.host.CompletionRole` the host
#: built the client in; a second judge's client is built in the ``judge`` role.
#:
#: An open vocabulary, its values this build knows registered in :data:`OUT_OF_RUN_PURPOSES`: a value a newer minor
#: added is read from a document that minor wrote, and every reader that branches on it names it rather than folding it
#: into a value it knows (:mod:`~threetears.evals.schema.versioning`).
OutOfRunPurpose = str

#: A value of :data:`OutOfRunPurpose` this build knows, as a type: what a caller names (an action's argument, a
#: filter) and what this build writes, refused outside the set as any ``Literal`` is. A stored field reads
#: the open :data:`OutOfRunPurpose` instead.
KnownOutOfRunPurpose = Literal["variation", "proposer", "analysis", "judge", "second_judge"]

#: The values of :data:`OutOfRunPurpose` this build knows, in the order its readers list them.
OUT_OF_RUN_PURPOSES: Final[tuple[str, ...]] = get_args(KnownOutOfRunPurpose)

#: The type of a stored field carrying a out-of-run purpose: one of :data:`OUT_OF_RUN_PURPOSES`, or, on the stored read
#: of a newer minor's document, a value that minor added.
StoredOutOfRunPurpose = Annotated[OutOfRunPurpose, known_unless_newer_minor("out-of-run purpose", OUT_OF_RUN_PURPOSES)]


#: How an out-of-run call ended: it returned a completion, or it raised. A raised call is still a
#: ledger row — the provider may have billed it — carrying no usage, since nothing reported any.
OutOfRunOutcome = Literal["completed", "raised"]


class OutOfRunSpend(CoreDocumentModel):
    """One call the engine made outside any run, as it was admitted and as the provider reported it.

    Written by :meth:`OutOfRunBudget.generate` for every call it makes, never rewritten. **Missing is
    not zero**: a token count or a cost the provider did not report is ``None``, as on
    :class:`~threetears.evals.schema.models.RoleUsage`, and a raised call carries no usage at all.
    """

    doc_type: Literal["eval_out_of_run_spend"] = "eval_out_of_run_spend"
    schema_version: CoreSchemaVersion = CORE_SCHEMA_VERSION
    id: str = Field(default_factory=lambda: str(uuid.uuid7()))
    scope_id: str = Field(min_length=1, description="The scope the work was for, which its ledger lives in.")
    purpose: StoredOutOfRunPurpose
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
    sent_settings: ClientRequestSettings | None = Field(
        default=None,
        description=(
            "The request settings the call was actually SENT with, as its completion reported them "
            "(``sent_settings_of``), for audit: a client that sent the closest setting a model accepts changes what "
            "the call cost. Nothing pools on it. None = unverified (the client did not report it), and always None "
            "for a raised call, which returned no completion."
        ),
    )
    unreadable: list[str] = Field(
        default_factory=list,
        description=(
            "The attributes a completed call reported that could not be stored as reported — a raw provider stop "
            "reason outside StopReason, a negative count — each recorded as None here. Never a reason to drop the "
            "row: the call was paid for. Empty when every reported attribute was stored."
        ),
    )
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
    campaign_id: str | None = Field(
        default=None, description="The campaign an analysis generation was written for, when it was for one."
    )
    run_id: str | None = Field(
        default=None, description="The finished run a judge repeat re-scored results of, when it was for one."
    )
    calls: int | None = Field(
        default=None,
        ge=1,
        exclude_if=lambda value: value is None,
        description=(
            "The calls this row ledgers together, when it ledgers more than one: a report the report writer wrote "
            "(purpose analysis), one row for every turn it sent, whether its reply came back or not. None = the row is "
            "one call. Left out of the document when unset."
        ),
    )
    unpriced_calls: int | None = Field(
        default=None,
        ge=0,
        exclude_if=lambda value: value is None,
        description=(
            "Of `calls`, those nothing priced: a reply that reported no cost, or a turn in flight when the writing "
            "stopped, whose reply never came and which may have been billed. Set exactly when `calls` is."
        ),
    )
    priced_usd: float | None = Field(
        default=None,
        ge=0.0,
        exclude_if=lambda value: value is None,
        description=(
            "What the priced ones of `calls` reported, together: the priced share, a floor whenever `unpriced_calls` "
            "is not 0, when `cost_usd` is None (the total is unknown, never $0). Set exactly when `calls` is."
        ),
    )
    admitted_by: str | None = Field(
        default=None,
        min_length=1,
        exclude_if=lambda value: value is None,
        description=(
            "What admitted the row's calls when it was not each one's own ceiling under `cap_usd`: a report's "
            "budget, in words ('the report budget of $10.00'), which admitted every turn before it was sent. None = "
            "each call was admitted on `priced_ceiling_usd` under `cap_usd`."
        ),
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
                for name in (
                    "stop_reason",
                    "prompt_tokens",
                    "completion_tokens",
                    "reasoning_tokens",
                    "cost_usd",
                    "sent_settings",
                )
                if getattr(self, name) is not None
            }
            if self.unreadable:
                reported["unreadable"] = self.unreadable
            if reported:
                raise ValueError(f"a raised call reported nothing, and this one carries {sorted(reported)}")
            if not self.failure:
                raise ValueError("a raised call names the exception's class in failure")
        elif self.failure is not None:
            raise ValueError("a completed call carries no failure")
        return self

    @model_validator(mode="after")
    def _a_row_of_several_calls_states_its_shares(self) -> Self:
        """Refuse a row of several calls without its priced and unpriced shares, shares on a row of one, more unpriced
        calls than calls, or a total cost that disagrees with its shares."""
        if self.calls is None:
            if self.unpriced_calls is not None or self.priced_usd is not None:
                raise ValueError("unpriced_calls and priced_usd are a row of several calls' shares; set calls too")
            return self
        if self.unpriced_calls is None or self.priced_usd is None:
            raise ValueError("a row of several calls states its unpriced_calls and its priced_usd")
        if self.unpriced_calls > self.calls:
            raise ValueError(f"{self.unpriced_calls} unpriced calls in a row of {self.calls}")
        if self.unpriced_calls and self.cost_usd is not None:
            raise ValueError("a row with an unpriced call has no total: cost_usd is None, and priced_usd its floor")
        if not self.unpriced_calls and self.cost_usd != self.priced_usd:
            raise ValueError("a row whose every call is priced costs its priced share: cost_usd is priced_usd")
        return self


class OutOfRunSpendStore(Protocol):
    """The one write the out-of-run ledger makes.

    Structural, so :class:`~threetears.evals.kernel.storage.EvalStorage` satisfies it by having the
    method. Raises ``StorageError`` on a failed write rather than returning a flag.
    """

    def save_out_of_run_spend(self, spend: OutOfRunSpend, /) -> None:
        """Write one ledger row."""
        ...


__all__ = [
    "OUT_OF_RUN_PURPOSES",
    "KnownOutOfRunPurpose",
    "OutOfRunOutcome",
    "OutOfRunPurpose",
    "OutOfRunSpend",
    "OutOfRunSpendStore",
    "StoredOutOfRunPurpose",
]
