"""The report planner: one campaign's report input becomes its report plan, in two model calls, each repairable once.

:func:`generate_report_plan` asks the planner for a page in two calls, so neither contract grows with the page:

1. **The structure call** (:func:`first_plan_request`): every choice and no prose
   (:class:`~threetears.evals.kernel.report_plan.PlanStructure`): the mode, picked from the modes code allows; the
   headline kind and the stated confidence; the options rows; the summary charts and the sections, each section and
   explore control by an id the planner chooses; how many Method pairs; and each durable claim's figure keys. Code
   refuses a mode it does not allow and every fault the plan checks find in the choices
   (:func:`~threetears.evals.analysis.plan_assembly.check_structure`).
2. **The prose call** (:func:`prose_plan_request`): the checked structure and the prose keys it created
   (:func:`~threetears.evals.analysis.plan_assembly.prose_slots`) go back with the same report input, and the planner
   returns a flat list of ``{key, text}`` pairs
   (:class:`~threetears.evals.kernel.report_plan.PlanProse`). Code refuses a key missing, repeated or unknown
   (:func:`~threetears.evals.analysis.plan_assembly.read_prose`), assembles the planner's part
   (:func:`~threetears.evals.analysis.plan_assembly.planned_report`), completes it with code's facts and checks the
   whole plan (:func:`~threetears.evals.analysis.report_plan_facts.complete_plan`): a fault now is the prose's, since
   the structure passed the same checks.

Both calls share one system prompt, the resolved ``eval_report_plan`` prompt, followed by what code appends: the
figure-key grammar (:func:`~threetears.evals.analysis.report.figure_keys.figure_key_grammar`), the host's register, and
a short section saying which call this is and what it returns. Their user messages open with the same report input
view and code's facts, so a prompt cache matches the prefix across both.

**One repair per call.** The analysis writer's boundary (:func:`~threetears.evals.analysis.generator.generate_analysis`),
drawn by the same helpers (:mod:`~threetears.evals.analysis.generation_round`): anything a FINISHED call got wrong — a
reply that is not JSON, one off its contract, a mode code does not allow, a fault the checks find, a prose key wrong —
buys that call ONE repair round-trip, which feeds the refusal and the refused reply back. A refused structure is
repaired before any prose is asked for; a refused prose reply repairs only the prose call. A second refusal of either
raises, naming both. A call the provider cut short (an output cap, a content filter) buys none. A generation makes at
most four calls, and every one is on the tally and its cost on the returned :class:`PlanGeneration`.

**The host's limits hold exactly as they do for the writer.** A model the host does not allow to write its analyses
(``HostProfile.analysis_writer_models``) is refused before anything is sent, by the writer's own check
(:func:`~threetears.evals.analysis.generator.refuse_an_unlisted_writer`), unless the call measures writers. Every call,
each repair included, is first put to the caller's ``admit`` with its exact request, which is how a caller holds a
generation to its spend cap (:class:`~threetears.evals.analysis.service.BudgetedGenerator`); a refusal there sends
nothing. The caller's :class:`~threetears.evals.analysis.generator.GenerationTally` is written as each call is sent and
returns, so it is exact however the generation ends.

**Both contracts fit every writer.** Each holds well under :data:`~threetears.evals.kernel.authored.SLOT_BUDGET` free-text
and list slots, the size every strict-schema writer was measured to accept, and a test holds them there; the prose
contract is flat, so a page of many sections adds items, not slots.

**Allowed modes.** Which modes the evidence permits (what survives the gates, what dominates) is code's to compute
(:func:`~threetears.evals.analysis.report_decision.allowed_modes`, carried on the facts); a caller passing no
``allowed_modes`` gets exactly those, and the plan checks refuse a mode outside them whatever a caller passes.

**Closed dependency set.** The planner reads the report input, the resolved prompt, the host's profile and an injected
completion client; it stores nothing. Storing a plan beside its analysis is the caller's.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from threetears.evals.analysis.errors import GenerationError, SoundnessRefusal
from threetears.evals.analysis.generation_round import (
    parse_reply,
    prompt_provenance_clause,
    record_returned_call,
    reject_incomplete_generation,
    repair_user_message,
)
from threetears.evals.analysis.generator import (
    CallAdmission,
    GenerationTally,
    refuse_an_unlisted_writer,
    user_message_digest,
)
from threetears.evals.analysis.numbers import format_number
from threetears.evals.analysis.plan_assembly import check_structure, planned_report, prose_slots, read_prose
from threetears.evals.analysis.plan_prompt import EVAL_REPORT_PLAN_DEFAULT
from threetears.evals.analysis.report.figure_keys import figure_key_grammar
from threetears.evals.analysis.report_input import ReportInput
from threetears.evals.analysis.report_input_build import planner_view
from threetears.evals.analysis.report_plan_facts import PLAN_MODES, PlanFacts, complete_plan, plan_facts
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.host.style import prompt_fragment
from threetears.evals.kernel.prompts.catalogue import REPORT_PLAN_PROMPT
from threetears.evals.kernel.report_plan import (
    PlanProse,
    PlanStructure,
    ReportPlan,
    prose_response_format,
    structure_response_format,
)
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import utc_now_iso

if TYPE_CHECKING:  # the model-provider PORT — a protocol, injected; never constructed here.
    from threetears.evals.schema.completion import CompletionGenerator, CompletionResult

__all__ = [
    "PlanGeneration",
    "first_plan_request",
    "generate_report_plan",
    "plan_user_message",
    "prose_plan_request",
]

log = get_logger(__name__)

#: The heading the report input's view opens each user message under; the prompt names it.
REPORT_INPUT_HEADING = "REPORT INPUT"

#: The heading code's facts follow under; the prompt names it.
FACTS_HEADING = "WHAT CODE HAS ALREADY DECIDED"

#: The heading the checked structure follows under in the prose call.
STRUCTURE_HEADING = "STRUCTURE"

#: The heading the prose keys follow under in the prose call.
PROSE_KEYS_HEADING = "PROSE KEYS"

#: What code appends to the system prompt for the structure call: which call this is and what it returns. Code's,
#: because it describes the contract code sends and reads, as the figure-key grammar is.
STRUCTURE_CALL = """\
THIS CALL: THE STRUCTURE. Return the page's choices only, in the schema sent: the mode, which must be one of \
`allowed_modes` under WHAT CODE HAS ALREADY DECIDED; the headline kind; your confidence form; where each open \
caveat under `caveats` lands, once (`caveat_landings`; none for a caveat code checked); the options rows and roles; \
the summary charts; the evidence sections, each with a short id you choose, unique on the page; the explore controls, \
each with an id; how many Method pairs; and each durable claim's figure keys (paths as FIGURE KEYS spells them, \
without braces or filters) with its confidence. Write no prose: the next call writes every sentence, addressed by the \
ids you give here."""

#: What code appends to the system prompt for the prose call.
PROSE_CALL = """\
THIS CALL: THE PROSE. The structure under STRUCTURE is fixed and checked. Return one {key, text} item for every key \
under PROSE KEYS, each exactly once, and no other key. Each key says what its text holds; one marked optional may be \
empty, every other is written. Write every figure as a key, as FIGURE KEYS says."""

#: Fed back to the planner after a refusal of either call, appended to that call's user message so the long prefix
#: stays byte-identical for a prompt cache. The "do not satisfy it by removing" clause is load-bearing for the reason the
#: writer's is: the cheapest way to satisfy a refusal is to delete what was complained about, which turns a repair into a
#: smaller page; and a refusal is never mended by changing the recommendation to dodge a warning.
_REPAIR_INSTRUCTION = """

YOUR PREVIOUS REPLY WAS REFUSED, AND THIS IS YOUR ONLY CHANCE TO CORRECT IT.

The call finished, so this is not about length. What came back could not be used:

{refusal}

Each fault names the field or key it sits at and why. Emit the WHOLE reply again with every one corrected, following \
each literally.

Keep everything the refusal does not name: reproduce it as it was rather than rewriting the reply around the fix.

Do NOT satisfy the refusal by REMOVING what it complains about, or by changing the recommendation to escape a warning. \
The evidence has not changed: a figure key that named nothing is spelled as the input spells it, a typed number \
becomes a key, a missing key or answer is written, a chart naming an absent id names one the input holds.

THIS IS WHAT YOU SENT, AND IT IS THE ONLY COPY YOU HAVE OF IT:

{previous}"""

_Read = TypeVar("_Read")


class PlanGeneration(BaseModel):
    """A generated report plan and how it was generated: what a caller stores beside it.

    The plan is the one that passed; the cost is every call's, each repair's included, and a repaired call says so, so a
    repaired plan is never mistakable for one accepted first time.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: ReportPlan = Field(description="The completed, checked plan.")
    generator_model: str = Field(description="The model that wrote the prose, as its provider named it.")
    prompt_version: str = Field(
        description="A short hash of the system prompt and both contracts the planner was sent."
    )
    user_message_digest: str = Field(description="The digest of the structure call's first user message.")
    token_cost: float = Field(ge=0, description="Dollars reported across every returned call.")
    repair_attempts: int = Field(ge=0, le=2, description="Calls repaired: at most one per call.")
    structure_refusal: str | None = Field(description="The refusal the structure's repair mended; None for none.")
    prose_refusal: str | None = Field(description="The refusal the prose's repair mended; None for none.")
    generated_at: str = Field(description="When the plan was accepted, ISO-8601.")


def _modes(allowed_modes: Sequence[str] | None, facts: PlanFacts) -> tuple[str, ...]:
    """The modes the planner may pick: the caller's, or those code computed from the input."""
    if allowed_modes is None:
        return facts.allowed_modes
    modes = tuple(allowed_modes)
    if not modes or not set(modes) <= set(PLAN_MODES):
        raise ValueError(f"allowed_modes {list(modes)} must be a non-empty subset of {list(PLAN_MODES)}")
    return modes


def _system(prompt: str, profile: HostProfile, call: str) -> str:
    return f"{prompt}\n\n{figure_key_grammar()}\n\nVOICE: {prompt_fragment(profile.style)}\n\n{call}"


def _compact(node: Any) -> str:
    return json.dumps(node, separators=(",", ":"), ensure_ascii=False)


def _prefix(report_input: ReportInput, facts: PlanFacts, modes: Sequence[str], budget: int | None) -> str:
    """The user message both calls open with: the report input's view, then code's facts."""
    view = {
        "shape": facts.shape,
        "signals": facts.signals.model_dump(mode="json"),
        "spine": facts.spine.as_fact(),
        "allowed_modes": list(modes),
        "masthead": facts.masthead.model_dump(mode="json"),
        "arms": [arm.model_dump(mode="json") for arm in facts.arms],
        "show_arms_key": facts.show_arms_key,
        "options_columns": list(facts.columns),
        "full_results": list(facts.full_results),
        "warnings": {
            arm: [{"position": at, **warning.model_dump(mode="json")} for at, warning in enumerate(raised)]
            for arm, raised in facts.warnings.items()
        },
        "disclosures": {mode: list(implied) for mode, implied in facts.disclosures.items()},
        "caveats": [caveat.model_dump(mode="json") for caveat in facts.caveats],
        "weakest_margin": facts.weakest.model_dump(mode="json") if facts.weakest is not None else None,
        "reader_names": {kind: dict(names) for kind, names in facts.names.items()},
    }
    return f"{REPORT_INPUT_HEADING}\n{planner_view(report_input, budget=budget)}\n\n{FACTS_HEADING}\n{_compact(view)}"


def first_plan_request(
    report_input: ReportInput,
    prompt: str,
    profile: HostProfile,
    *,
    allowed_modes: Sequence[str] | None = None,
    budget: int | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """The exact first request a plan generation sends, the structure call's: system prompt, user message, contract.

    One function, called by :func:`generate_report_plan` and by anything that must predict what it will send (a
    recording keys on it), so the two cannot drift. The user message is the report input's planner view
    (:func:`~threetears.evals.analysis.report_input_build.planner_view`), then code's facts
    (:func:`~threetears.evals.analysis.report_plan_facts.plan_facts`): the shape and signals, the spine the first
    evidence section draws, the allowed modes, the masthead, each arm with its tag, name and colour, whether the arms
    key shows, the options columns and full-results tables, every warning code raises on each arm, numbered as an
    answer is keyed, the disclosures each mode implies, the weakest margin with the keys that name it, and each id's
    reader name.

    :param report_input: The campaign's report input.
    :ptype report_input: ReportInput
    :param prompt: The resolved ``eval_report_plan`` prompt.
    :ptype prompt: str
    :param profile: The host the plan is generated for, whose register the system prompt carries.
    :ptype profile: HostProfile
    :param allowed_modes: The modes the planner may pick; None for the modes code computes from the input.
    :ptype allowed_modes: Sequence[str] | None
    :param budget: The most characters the report input's view may take; None for the view's default.
    :ptype budget: int | None
    :return: ``(system, user, response_format)``.
    :rtype: tuple[str, str, dict[str, Any]]
    :raises RefusedInput: The view is over its budget once every trim is made; nothing is sent.
    :raises ValueError: ``allowed_modes`` is empty or names a mode no plan takes.
    """
    return (
        _system(prompt, profile, STRUCTURE_CALL),
        plan_user_message(report_input, allowed_modes=allowed_modes, budget=budget),
        structure_response_format(),
    )


def plan_user_message(
    report_input: ReportInput, *, allowed_modes: Sequence[str] | None = None, budget: int | None = None
) -> str:
    """The user message a plan generation's structure call opens with: what the planner reads of the evidence.

    The report input's planner view, then code's facts (:func:`first_plan_request` says what they hold). One function,
    called by :func:`first_plan_request` and by the reporter eval, which judges a recorded plan against exactly what its
    planner read (:class:`~threetears.evals.analysis.reporter_kind.ReporterCase`), so the two cannot drift. Its digest
    is what a generation records as ``user_message_digest``.

    :param report_input: The campaign's report input.
    :ptype report_input: ReportInput
    :param allowed_modes: The modes the planner may pick; None for the modes code computes from the input.
    :ptype allowed_modes: Sequence[str] | None
    :param budget: The most characters the report input's view may take; None for the view's default.
    :ptype budget: int | None
    :return: The message.
    :rtype: str
    :raises RefusedInput: The view is over its budget once every trim is made.
    :raises ValueError: ``allowed_modes`` is empty or names a mode no plan takes.
    """
    facts = plan_facts(report_input)
    return _prefix(report_input, facts, _modes(allowed_modes, facts), budget)


def prose_plan_request(
    report_input: ReportInput,
    prompt: str,
    profile: HostProfile,
    structure: PlanStructure,
    *,
    allowed_modes: Sequence[str] | None = None,
    budget: int | None = None,
) -> tuple[str, str, dict[str, Any]]:
    """The exact first request of the prose call, for a checked structure: system prompt, user message, contract.

    The user message is the structure call's, then the structure as checked and every prose key it created, each with
    what it holds and whether it may be empty (:func:`~threetears.evals.analysis.plan_assembly.prose_slots`).

    :param report_input: The campaign's report input.
    :ptype report_input: ReportInput
    :param prompt: The resolved ``eval_report_plan`` prompt.
    :ptype prompt: str
    :param profile: The host the plan is generated for.
    :ptype profile: HostProfile
    :param structure: The structure the prose is for, as checked.
    :ptype structure: PlanStructure
    :param allowed_modes: The modes the planner may pick; None for the modes code computes from the input.
    :ptype allowed_modes: Sequence[str] | None
    :param budget: The most characters the report input's view may take; None for the view's default.
    :ptype budget: int | None
    :return: ``(system, user, response_format)``.
    :rtype: tuple[str, str, dict[str, Any]]
    """
    facts = plan_facts(report_input)
    return _prose_request(
        _prefix(report_input, facts, _modes(allowed_modes, facts), budget), prompt, profile, structure, facts
    )


def _prose_request(
    prefix: str, prompt: str, profile: HostProfile, structure: PlanStructure, facts: PlanFacts
) -> tuple[str, str, dict[str, Any]]:
    keys = [slot.model_dump(mode="json") for slot in prose_slots(structure, facts)]
    user = (
        f"{prefix}\n\n{STRUCTURE_HEADING}\n{_compact(structure.model_dump(mode='json'))}\n\n"
        f"{PROSE_KEYS_HEADING}\n{_compact(keys)}"
    )
    return _system(prompt, profile, PROSE_CALL), user, prose_response_format()


def _prompt_version(prompt: str, profile: HostProfile) -> str:
    """A short hash of what the planner is told: both calls' system prompts and both contracts."""
    told = "\n".join(
        [
            _system(prompt, profile, STRUCTURE_CALL),
            _system(prompt, profile, PROSE_CALL),
            json.dumps(structure_response_format(), sort_keys=True, separators=(",", ":")),
            json.dumps(prose_response_format(), sort_keys=True, separators=(",", ":")),
        ]
    )
    return hashlib.sha256(told.encode("utf-8")).hexdigest()[:12]


async def generate_report_plan(
    report_input: ReportInput,
    *,
    prompt: str,
    model: str,
    client: CompletionGenerator,
    profile: HostProfile,
    allowed_modes: Sequence[str] | None = None,
    tally: GenerationTally | None = None,
    admit: CallAdmission | None = None,
    measuring_writers: bool = False,
    budget: int | None = None,
) -> PlanGeneration:
    """Generate one campaign's report plan from its report input: a structure call and a prose call, each repaired at
    most once (this module's account).

    :param report_input: The campaign's report input.
    :ptype report_input: ReportInput
    :param prompt: The resolved ``eval_report_plan`` system prompt (the caller reads it from the prompt registry, so an
        operator's edits reach here).
    :ptype prompt: str
    :param model: The planner model id, as requested or as the host's client resolved its default; checked against
        the host's allowed writers before anything is sent.
    :ptype model: str
    :param client: The injected completion client, bound to ``model``; the planner never constructs one.
    :ptype client: CompletionGenerator
    :param profile: The host: its allowed writer models and its register.
    :ptype profile: HostProfile
    :param allowed_modes: The modes the planner may pick; the structure's mode must be one. None for the modes code
        computes from the input.
    :ptype allowed_modes: Sequence[str] | None
    :param tally: A caller-owned record of what this generation sent, spent and was refused, exact however it ends.
    :ptype tally: GenerationTally | None
    :param admit: Asked with each call's system prompt, user message and contract BEFORE the call is counted or sent,
        each repair's included; it raises to refuse the call, and nothing is sent. ``None`` admits every call.
    :ptype admit: CallAdmission | None
    :param measuring_writers: True only for a run that measures writer models to learn which belong on the host's
        list, so it is not held to it.
    :ptype measuring_writers: bool
    :param budget: The most characters the report input's view may take; None for the view's default.
    :ptype budget: int | None
    :return: The plan and how it was generated.
    :rtype: PlanGeneration
    :raises GenerationError: ``model`` is not a writer the host allows (nothing is spent), or the provider cut a call
        short (not repaired). The message says what was billed.
    :raises SoundnessRefusal: A call's reply was refused both times (a
        :class:`~threetears.evals.analysis.errors.PlanRefusal` when the second refusal was faults the checks found);
        the message names both refusals and what was billed.
    :raises RefusedInput: The report input's view is over its budget once every trim is made; nothing is sent.
    :raises ValueError: ``allowed_modes`` is empty or names a mode no plan takes.
    """
    tally = tally if tally is not None else GenerationTally()
    if not measuring_writers:
        refuse_an_unlisted_writer(profile, model)
    facts = plan_facts(report_input)
    modes = _modes(allowed_modes, facts)
    prefix = _prefix(report_input, facts, modes, budget)
    version = _prompt_version(prompt, profile)
    tally.prompt_version = version
    campaign = report_input.meta.campaign

    async def call_with_repair(
        stage: str, system: str, user: str, contract: dict[str, Any], read: Callable[[CompletionResult], _Read]
    ) -> tuple[_Read, CompletionResult, str | None]:
        """One call and at most one repair: what it read, the result it read it from, and the refusal it mended."""
        result = await send(stage, "initial", system, user, contract)
        try:
            return read(result), result, None
        except SoundnessRefusal as refused:
            # NOSILENT: the one repair round-trip below is made from here, and the refusal rides on the tally, on the
            # returned generation, and on the summary a second refusal raises with.
            refusal = str(refused)
            tally.refusals.append(refusal)
            log.warning(
                "eval.report_plan %s refused for campaign=%r — attempting ONE repair round-trip "
                "(billed so far: $%.4f): %s",
                stage,
                campaign,
                tally.token_cost,
                refusal,
            )
        except GenerationError as terminal:
            # The subclass is caught first: a refusal of a finished call is repaired, and only a call the provider cut
            # short reaches here. Re-raised as the same object, its message carrying what was billed, so the class the
            # caller branches on survives.
            summary = (
                f"{format_number(tally.token_cost, prefix='$')} was billed and nothing was planned. The {stage} call "
                f"could not be repaired: {terminal}"
            )
            log.warning("eval.report_plan FAILED with no repair possible for campaign=%r — %s", campaign, summary)
            terminal.args = (summary,)
            raise
        previous = result.content or "(the planner returned no content)"
        repair = repair_user_message(user, refusal, previous, instruction=_REPAIR_INSTRUCTION)
        repaired = await send(stage, "repair", system, repair, contract)
        try:
            return read(repaired), repaired, refusal
        except GenerationError as exhausted:
            if isinstance(exhausted, SoundnessRefusal):
                tally.refusals.append(str(exhausted))
            summary = (
                f"{format_number(tally.token_cost, prefix='$')} was billed and nothing was planned. The {stage} call "
                f"was refused twice. First refusal: {refusal}. Second, after the one repair: {exhausted}"
                + prompt_provenance_clause(
                    prompt,
                    default=EVAL_REPORT_PLAN_DEFAULT,
                    prompt_type=REPORT_PLAN_PROMPT,
                    seed_module="plan_prompt.py",
                )
            )
            log.warning(
                "eval.report_plan FAILED after one repair of the %s call for campaign=%r initial_model=%s "
                "repair_model=%s — %s",
                stage,
                campaign,
                result.model or model,
                repaired.model or model,
                summary,
            )
            exhausted.args = (summary,)
            raise

    async def send(stage: str, attempt: str, system: str, user: str, contract: dict[str, Any]) -> CompletionResult:
        if admit is not None:
            admit(system, user, contract)
        tally.calls += 1
        result = await client.generate(system=system, user=user, response_format=contract)
        cost = record_returned_call(tally, result, model)
        log.info(
            "eval report plan %s generated (%s) for campaign=%r: cost=$%.4f prompt_tokens=%s completion_tokens=%s "
            "model=%s stop_reason=%s",
            stage,
            attempt,
            campaign,
            cost,
            "unreported" if result.input_tokens is None else result.input_tokens,
            "unreported" if result.output_tokens is None else result.output_tokens,
            result.model or model,
            result.stop_reason,
        )
        return result

    def read_structure(result: CompletionResult) -> PlanStructure:
        structure = _reply(result, PlanStructure, "structure")
        check_structure(structure, report_input, facts, modes)
        return structure

    structure, _, structure_refusal = await call_with_repair(
        "structure", _system(prompt, profile, STRUCTURE_CALL), prefix, structure_response_format(), read_structure
    )
    system, user, contract = _prose_request(prefix, prompt, profile, structure, facts)
    slots = prose_slots(structure, facts)

    def read_plan(result: CompletionResult) -> ReportPlan:
        prose = read_prose(_reply(result, PlanProse, "prose"), slots)
        return complete_plan(planned_report(structure, prose, facts), report_input, facts)

    plan, produced_by, prose_refusal = await call_with_repair("prose", system, user, contract, read_plan)
    return PlanGeneration(
        plan=plan,
        generator_model=produced_by.model or model,
        prompt_version=version,
        user_message_digest=user_message_digest(prefix),
        token_cost=tally.token_cost,
        repair_attempts=(structure_refusal is not None) + (prose_refusal is not None),
        structure_refusal=structure_refusal,
        prose_refusal=prose_refusal,
        generated_at=utc_now_iso(),
    )


_Reply = TypeVar("_Reply", PlanStructure, PlanProse)


def _reply(result: CompletionResult, model: type[_Reply], stage: str) -> _Reply:
    """A finished call's reply read into its contract's model, or the refusal of it.

    :raises GenerationError: The provider cut the call short (not repaired).
    :raises SoundnessRefusal: The reply is not JSON or is off its contract (repaired once).
    """
    reject_incomplete_generation(result)
    payload = parse_reply(result)
    try:
        return model.model_validate(payload)
    except ValidationError as refused:
        raise SoundnessRefusal(f"the planner's {stage} reply does not match its contract: {refused}") from refused
