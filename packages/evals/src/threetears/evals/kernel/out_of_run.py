"""Calls the engine makes outside any run: priced before they are made, and ledgered once they are.

Five engine calls have no run around them: a launch's case generation (an ``llm`` variation axis's
writer, :func:`~threetears.evals.gen.generate_variations`), the rubric proposer
(:func:`~threetears.evals.gen.propose_draft`), an analysis generation
(:func:`~threetears.evals.analysis.run_analysis_generation`), a judge repeat
(:func:`~threetears.evals.run.repeat_judge_scores`) and a second judge
(:func:`~threetears.evals.run.ask_second_judge`). A run's own calls are bounded by its cost
cap as their spend arrives (``EvalRunCostCap``); these happen before any run exists, with none coming, or
after the runs have ended, so nothing would bound or record them. This module is what does:

* **Priced before the call.** :meth:`OutOfRunBudget.admit` asks the client what each planned call
  can cost at most (:meth:`~threetears.evals.schema.completion.PricedCompletion.price_ceiling` — the
  host's answer, since the engine knows neither a model's rates nor the output cap the client was
  built with) and refuses the whole set when the ceilings together would pass the cap, before any
  of them is made. Under an enforced cap a call the client cannot price is refused too: unknown is
  not $0.
* **Ledgered after it.** :meth:`OutOfRunBudget.generate` makes an admitted call and writes one
  :class:`~threetears.evals.schema.out_of_run_spend.OutOfRunSpend` document for it — what the provider reported, the ceiling it was admitted
  at and the cap it was admitted under — whether the call returned or raised, because a raised
  call can have been billed too, and whether or not what the provider reported can be stored as
  reported: an attribute the ledger cannot hold (a raw provider stop reason, a negative count) is
  recorded as unreadable, never a reason to drop the row of a call that was paid for. What the ledger
  reads off a completion is :class:`~threetears.evals.schema.completion.CompletionResult`'s attributes,
  by those names — the one completion protocol the engine reads, whatever the client's shape.

The ledger is a stored document type of its own (``eval_out_of_run_spend``,
:mod:`threetears.evals.schema.out_of_run_spend`) in the scope the work was for, keyed by nothing but
its id: one document per call, never rewritten.

What a case generation's calls ARE is here too (:func:`plan_variation_calls`): the generation that
makes them (:mod:`threetears.evals.gen`) and the battery that prices every template's generation before
it launches any (:mod:`threetears.evals.run`) both read one plan, and the package matrix lets both reach
the kernel and neither reach the other — so a battery cannot price a call its launch would not make.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, NamedTuple

from pydantic import TypeAdapter, ValidationError
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.schema.models import EvalTemplate, EvalTestCase, VariationAxis
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.schema.completion import JSON_OBJECT_RESPONSE_FORMAT
from threetears.observe import get_logger

if TYPE_CHECKING:
    from concurrent.futures import Executor

    from threetears.evals.schema.completion import PricedCompletion

from threetears.evals.schema.out_of_run_spend import OutOfRunPurpose, OutOfRunSpend, OutOfRunSpendStore

log = get_logger(__name__)


@dataclass(frozen=True)
class PlannedCall:
    """One call an out-of-run unit of work means to make: the prompt pair and the directive it sends."""

    system: str
    user: str
    response_format: dict[str, Any] | None = None


@dataclass(frozen=True)
class AdmittedCall:
    """A planned call its budget admitted, at the ceiling it was priced at.

    :meth:`OutOfRunBudget.admit` mints one, and the budget that minted it keeps it, with the client it
    was priced on. :meth:`OutOfRunBudget.generate` makes only an object it minted — the very object, not
    an equal one — on that very client, once. So a copy whose call was altered after pricing
    (``dataclasses.replace``), an admission made by another budget, a second making, and a call made on
    another client of the same model (whose output cap, and so whose price, can differ) are all refused:
    a call cannot be made without having been priced as it is made.
    """

    call: PlannedCall
    purpose: OutOfRunPurpose
    model: str
    ceiling_usd: float | None
    token: str


def existing_axis_values(axis: VariationAxis, existing: Sequence[EvalTestCase]) -> set[str]:
    """The values a template's stored cases already give ``axis`` — what its generation call asks the model to avoid.

    Args:
        axis: The variation axis.
        existing: The template's stored cases in the scope being generated into.

    Returns:
        The non-empty values those cases carry for the axis.
    """
    return {tc.variation_params.get(axis.name, "") for tc in existing if tc.variation_params.get(axis.name)}


def plan_variation_calls(
    template: EvalTemplate, n_variations: int, existing: Sequence[EvalTestCase]
) -> dict[str, PlannedCall]:
    """The one call each of ``template``'s ``llm`` axes makes to generate ``n_variations`` values, built before any is made.

    Each call runs in JSON-object mode and asks for ``{"values": [...]}`` — the object envelope is required
    because ``json_object`` mode forbids a bare top-level array — naming the values the template's stored
    cases already give the axis, so the model avoids them.

    Args:
        template: The template whose ``llm`` axes are written.
        n_variations: How many values each axis is asked for.
        existing: The template's stored cases in the scope being generated into.

    Returns:
        One planned call per ``llm`` axis, keyed by axis name in declaration order; empty when no axis is ``llm``.
    """
    return {
        axis.name: _llm_axis_call(axis, n_variations, existing_axis_values(axis, existing))
        for axis in template.variation_axes
        if axis.generator == "llm"
    }


def _llm_axis_call(axis: VariationAxis, n_variations: int, existing_values: set[str]) -> PlannedCall:
    """The call that asks a model for ``n_variations`` novel values for ``axis``, excluding ``existing_values``."""
    existing_block = (
        "(none — produce any novel values)"
        if not existing_values
        else "\n".join(f"- {v}" for v in sorted(existing_values))
    )
    system_prompt = (
        "You generate values for an eval variation axis. Return ONLY a JSON "
        'object of the form {"values": ["...", "..."]} — a JSON array of strings '
        'under the key "values", with no other keys and no commentary. Each value '
        "must be distinct from every value already shown to you."
    )
    user_prompt = (
        f"Axis name: {axis.name}\n"
        f"Axis description: {axis.description or '(no description)'}\n"
        f"Existing values to exclude:\n{existing_block}\n\n"
        f'Produce {n_variations} new distinct values as a JSON object: {{"values": [...]}}.'
    )
    return PlannedCall(system=system_prompt, user=user_prompt, response_format=JSON_OBJECT_RESPONSE_FORMAT)


class RecordedCompletion(NamedTuple):
    """What an admitted call returned, and the ledger row written for it."""

    result: Any
    spend: OutOfRunSpend


@dataclass(eq=False)
class OutOfRunBudget:
    """The cap one out-of-run unit of work is held to, and the ledger its calls are written to.

    Built by whoever starts the work: the launch builds one per generating launch
    (``LaunchRequest.generation_budget``, capped at the host's ``max_out_of_run_cost_usd``), an analysis
    generation one per generation (:func:`~threetears.evals.ops.analysis_generate`, at the same cap), and a
    host proposing a rubric builds one for the proposal. Admissions accumulate — a second admission is
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
        campaign_id: The campaign an analysis generation is written for, stamped on every row.
        run_id: The finished run a judge repeat re-scores, stamped on every row.
        blocking_executor: Where each ledger write to ``store`` runs, off the event loop — the host's
            ``EvalHost.blocking_executor``, or ``None`` for the loop's default executor. Keyword-only and
            without a default, as the host's own is: a store write made on the loop stalls every
            coroutine in the process, and the default executor is the one a host's liveness probes may
            share, so each construction names where its writes go.
    """

    store: OutOfRunSpendStore
    scope_id: str
    cap_usd: float | None
    template_id: str | None = None
    subject_id: str | None = None
    launch_group_id: str | None = None
    campaign_id: str | None = None
    run_id: str | None = None
    blocking_executor: Executor | None = field(kw_only=True)
    _committed_usd: float = field(default=0.0, init=False)
    #: Every admission not yet made, by token: the object minted and the client it was priced on.
    _admitted: dict[str, tuple[AdmittedCall, PricedCompletion]] = field(default_factory=dict, init=False)
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
            minted = AdmittedCall(
                call=call, purpose=purpose, model=client.model_name, ceiling_usd=ceiling, token=str(uuid.uuid7())
            )
            self._admitted[minted.token] = (minted, client)
            admitted.append(minted)
        return admitted

    async def generate(self, client: PricedCompletion, admitted: AdmittedCall) -> RecordedCompletion:
        """Make one admitted call and write its ledger row, whether it returns or raises.

        Args:
            client: The client the call was admitted on — that object.
            admitted: The object :meth:`admit` returned for the call. Each is made once.

        Returns:
            The completion and its ledger row.

        Raises:
            ValueError: ``admitted`` is not an object this budget admitted and has not made — another budget's,
                one already made, or a copy (its call altered after pricing or not) — or ``client`` is not the
                client it was priced on.
            StorageError: The ledger row could not be written for a completed call.
            Exception: Whatever the call raised — after its row is written.
        """
        minted = self._admitted.get(admitted.token)
        if minted is None or minted[0] is not admitted:
            raise ValueError(
                "this call is not one this budget admitted and has not yet made — another budget's, one already made, "
                "or a copy of an admission (whose call may have been altered after it was priced); every out-of-run "
                "call is priced by its own budget's admit() and made once, as admitted"
            )
        if minted[1] is not client:
            raise ValueError(
                f"the call was priced on one client for {admitted.model!r} and is being made on another (for "
                f"{client.model_name!r}); a call's price is its client's — the output cap it was built with — so it is "
                "made on the client it was admitted on"
            )
        del self._admitted[admitted.token]
        call = admitted.call
        try:
            result = await client.generate(system=call.system, user=call.user, response_format=call.response_format)
        except BaseException as raised:
            try:
                await self._record(admitted, outcome="raised", failure=type(raised).__name__)
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
        # Read by CompletionResult's own attribute names — the protocol every completion the engine reads
        # satisfies — and an attribute a result lacks reads as unreported, never zero.
        reported = {
            field_name: getattr(result, attribute, None) for field_name, attribute in _COMPLETION_ATTRIBUTES.items()
        }
        spend = await self._record(admitted, outcome="completed", **_storable(admitted, reported))
        return RecordedCompletion(result, spend)

    def _row(self, admitted: AdmittedCall) -> dict[str, Any]:
        """What every row of ``admitted`` records whatever the call reported: what it was, and what it was held to."""
        return {
            "scope_id": self.scope_id,
            "purpose": admitted.purpose,
            "model": admitted.model,
            "priced_ceiling_usd": admitted.ceiling_usd,
            "cap_usd": self.cap_usd,
            "template_id": self.template_id,
            "subject_id": self.subject_id,
            "launch_group_id": self.launch_group_id,
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
        }

    async def _record(self, admitted: AdmittedCall, **reported: Any) -> OutOfRunSpend:
        spend = OutOfRunSpend(**self._row(admitted), **reported)
        # The append rides the write onto the worker, so ``recorded`` lists every row the store holds even
        # when the coroutine awaiting the write is cancelled before it reads the outcome.
        await run_blocking(self.blocking_executor, self._write, spend)
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

    def _write(self, spend: OutOfRunSpend) -> None:
        """Write one ledger row and remember it — blocking, so it runs on :attr:`blocking_executor`."""
        self.store.save_out_of_run_spend(spend)
        self._recorded.append(spend)


#: The ledger field each reported attribute of a completion is stored in, keyed by ledger field, valued by the
#: :class:`~threetears.evals.schema.completion.CompletionResult` attribute it is read from.
_COMPLETION_ATTRIBUTES: dict[str, str] = {
    "served_model": "served_model",
    "stop_reason": "stop_reason",
    "prompt_tokens": "input_tokens",
    "completion_tokens": "output_tokens",
    "reasoning_tokens": "reasoning_tokens",
    "cost_usd": "cost_usd",
    "price_source": "price_source",
}


#: Each reported field's own validation, as the row's model declares it — so an attribute is held to exactly
#: what the stored row would hold it to.
def _field_adapter(name: str) -> TypeAdapter[Any]:
    """The validation ``OutOfRunSpend`` applies to its field ``name``: its type and its constraints."""
    declared = OutOfRunSpend.model_fields[name]
    return TypeAdapter(Annotated[declared.annotation, *declared.metadata] if declared.metadata else declared.annotation)


_COMPLETED_FIELD_ADAPTERS: dict[str, TypeAdapter[Any]] = {name: _field_adapter(name) for name in _COMPLETION_ATTRIBUTES}


def _storable(admitted: AdmittedCall, reported: dict[str, Any]) -> dict[str, Any]:
    """``reported`` as a completed row can hold it: each attribute the ledger refuses is recorded None and named.

    Validated one attribute at a time against the row's own field, so one unreadable attribute costs only
    itself and the row of a paid call is always written.

    Args:
        admitted: The call, for the log line.
        reported: The completion's attributes, by ledger field.

    Returns:
        The attributes to store, plus ``unreadable`` naming any recorded None for that reason.
    """
    storable: dict[str, Any] = {}
    unreadable: list[str] = []
    for name, value in reported.items():
        try:
            storable[name] = _COMPLETED_FIELD_ADAPTERS[name].validate_python(value)
        except ValidationError as refused:
            storable[name] = None
            unreadable.append(name)
            log.warning(
                "eval.out_of_run %s call on %s reported %s=%r, which the ledger cannot hold (%s); recorded as unreadable",
                admitted.purpose,
                admitted.model,
                name,
                value,
                refused.errors()[0]["msg"],
            )
    return {**storable, "unreadable": unreadable}


__all__ = [
    "AdmittedCall",
    "OutOfRunBudget",
    "PlannedCall",
    "RecordedCompletion",
    "existing_axis_values",
    "plan_variation_calls",
]
