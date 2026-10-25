"""The parts every one-call-plus-one-repair document generation shares: the analysis writer's and the report planner's.

Both generators (:func:`~threetears.evals.analysis.generator.generate_analysis` and
:func:`~threetears.evals.analysis.planner.generate_report_plan`) send one request with a strict contract, read the
reply into a typed document, and buy exactly ONE repair round-trip for anything a FINISHED call got wrong. The
boundary is "did the call finish?" (:class:`~threetears.evals.analysis.errors.SoundnessRefusal` says why), and these
helpers are where both generators draw it, so the two cannot draw it differently:

- :func:`reject_incomplete_generation` refuses a call the provider cut short, before anything parses it, with the
  bare class that buys no repair;
- :func:`parse_reply` reads the one JSON object a finished call returned, refusing anything else repairably;
- :func:`repair_user_message` builds the repair call's user message from the original, the refusal and the refused
  output, behind the generator's own instruction;
- :func:`record_returned_call` adds a returned call to the caller's
  :class:`~threetears.evals.analysis.generator.GenerationTally`;
- :func:`prompt_provenance_clause` says, on a twice-refused generation, whether the store-master prompt it ran on is
  the shipped default.

Each generator keeps its own logger and its own words for what it generates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from threetears.evals.analysis.errors import GenerationError, SoundnessRefusal
from threetears.evals.kernel.provider import describe_incomplete_completion, extract_json

if TYPE_CHECKING:
    from threetears.evals.analysis.generator import GenerationTally
    from threetears.evals.schema.completion import CompletionResult

__all__ = [
    "parse_reply",
    "prompt_provenance_clause",
    "record_returned_call",
    "reject_incomplete_generation",
    "repair_user_message",
]


def reject_incomplete_generation(result: CompletionResult) -> None:
    """Refuse a generation the provider itself reported it cut short, before parsing it.

    First in each generator's validation chain, ahead of every parse and every check, and that ordering is the whole
    point. The output cap is a failure mode a SUPPORTED input reaches, not a misuse: a generator's ``model`` may be an
    override for A/B-ing a cheaper model, the cap is fixed and the caller cannot raise it, and a model whose own output
    ceiling sits under what the evidence needs truncates every time: it logs ``finish_reason=length`` and stores
    nothing.

    Placed in the chain rather than at the call sites because a truncated result must be unable to reach the
    soundness checks, not merely unlikely to: a :class:`SoundnessRefusal` raised off a salvaged partial payload buys a
    repair round-trip, which regenerates into the same cap and bills a second full generation to fail the same way.
    One chokepoint covers the initial call and the repair with one check, and a third call site would inherit it.

    **What this does NOT claim.** Nothing here is a timeout or a cancellation — a call that sits unanswered is a
    different failure, upstream of this path — and the case for this check does not rest on one: it rests on the
    truncation being real, on the repair boundary having been an unenforced assumption, and on the caller being owed
    the reason for a generation it was billed for.

    :param result: The completed generator call to check.
    :ptype result: CompletionResult
    :raises GenerationError: The provider reported the call did not finish. Deliberately the BARE class and never
        :class:`SoundnessRefusal` — the call did not finish, so there is nothing to feed back that would change the
        outcome, and the second attempt would run into the identical fixed cap.

        **This is the ONLY thing the bare class means on a call that was made**, and this check is the only place
        that raises it after one. Every other way a generation can fail — output that is not JSON, a missing field,
        any check of the document — is something a finished call DID, and raises the repairable subclass. The
        boundary is therefore "did the call finish", and this function is the single place that answers it.
    """
    incomplete = describe_incomplete_completion(result)
    if incomplete is None:
        return
    # `describe_incomplete_completion` names two remedies — raise the cap, or change model — and both are reachable,
    # but not the way it phrases the first: no ACTION takes the cap. It is the host's configured answer budget, which
    # the host's client builder derives the cap from per client. So this SCOPES the shared advice to where the knob
    # actually is; naming a per-call parameter would send an operator hunting for one that does not exist. The
    # budget's value is not restated — it is the host's, and would drift here. Only ``max_tokens`` gets it:
    # ``truncated`` is cut short by a cause the provider did not name, which is not the cap, so pointing at the budget
    # there would send the operator after the wrong knob.
    remedy = (
        " The output cap is not a parameter of this action: it is the host's configured generator answer "
        "budget. Raise that budget, re-run with a larger-output generator model, or narrow the campaign so "
        "there is less to analyse."
        if result.stop_reason == "max_tokens"
        else ""
    )
    raise GenerationError(f"the generator did not finish — {incomplete}{remedy}")


def parse_reply(result: CompletionResult) -> dict[str, Any]:
    """Extract the single JSON object a generator returned, or raise.

    ``extract_json`` returns a JSON *object* or raises — a bare array or scalar is a ``ValueError`` here, so the object
    shape is guaranteed to the callers.

    **Reaching this function means the call finished normally**, so a failure here really is a malformed payload and
    nothing more. That is not an assumption: a provider-reported truncation or filter was already refused by
    :func:`reject_incomplete_generation`, which each generator runs first for exactly this reason. Reporting only "not
    valid JSON" on a truncation would send a caller to retry or blame the prompt when the fix is the model, while the
    ``stop_reason="max_tokens"`` sat unread on the result.

    :param result: The completed generator call, known to have finished normally.
    :ptype result: CompletionResult
    :return: The parsed object.
    :rtype: dict[str, Any]
    :raises SoundnessRefusal: The content is not a JSON object. Repairable, because the decoder's own message names the
        line and column that broke — an instruction the generator can act on, since the repair is sent the output.
    """
    try:
        return extract_json(result.content or "")
    except ValueError as e:
        raise SoundnessRefusal(f"generator output was not valid JSON: {e}") from e


def repair_user_message(user_message: str, refusal: str, previous: str, *, instruction: str) -> str:
    """Append the refusal AND the refused output to the original user message, behind the generator's instruction.

    ``previous`` is included because the refusal often indexes into it and the model cannot see it otherwise: the
    completion port starts a new conversation per call, so the second call's only context is what this function
    builds. A refusal of a check survives that cold, but a parse refusal ("not valid JSON: Expecting ',' delimiter:
    line 1 column 34") names a coordinate in text the model no longer has. Cheaper to send the output back than to
    spend a second full generation on a hint that points nowhere.

    Appended LAST so the stable prefix stays as long as it can be. That prefix is the system prompt plus the evidence,
    NOT the instruction: the instruction interpolates ``{refusal}`` ahead of ``{previous}``, and the refusal differs
    per call, so nothing from the instruction onward can be byte-identical to anything.

    :param user_message: The original user message, reused verbatim.
    :ptype user_message: str
    :param refusal: The refusal text, which the instruction interpolates.
    :ptype refusal: str
    :param previous: The output that was refused.
    :ptype previous: str
    :param instruction: The generator's repair instruction, a format string holding ``{refusal}`` and ``{previous}``.
    :ptype instruction: str
    :return: The repair call's user message.
    :rtype: str
    """
    return user_message + instruction.format(refusal=refusal, previous=previous)


def record_returned_call(tally: GenerationTally, result: CompletionResult, model: str) -> float:
    """Add one returned call to the tally: counted, its cost added and its model recorded.

    A call whose provider reported no cost adds nothing to the total and is counted as unreported, so the tally does
    not read an unpriced call as a free one.

    :param tally: The caller's tally of the generation.
    :ptype tally: GenerationTally
    :param result: The call that returned.
    :ptype result: CompletionResult
    :param model: The model the generation was asked of, recorded where the provider names none.
    :ptype model: str
    :return: The call's cost in dollars, 0 where the provider reported none.
    :rtype: float
    """
    cost = result.cost_usd if result.cost_usd is not None else 0.0
    tally.returned += 1
    tally.token_cost += cost
    tally.reported_models.append(result.model or model)
    if result.cost_usd is None:
        tally.unreported_cost += 1
    return cost


def prompt_provenance_clause(prompt: str, *, default: str, prompt_type: str, seed_module: str) -> str:
    """Say, on a twice-refused generation, whether the prompt it ran on is the shipped default.

    A generator's prompt is store-master: the shipped default seeds an empty slot once, and after that a rule added to
    it in code reaches a deployment only when an operator pushes it through the host's prompt store. Nothing in a
    deploy does that. So a check can refuse, at full cost and repeatably, a shape the shipped prompt forbids and the
    deployment's prompt has never heard of; the refusal names the rule, and nothing names that the prompt in use
    predates it.

    This is the disclose-only option — report the divergence and leave the judgement to the operator — and
    deliberately no more: an operator's retuned prompt differs from the default by design, so a divergence is a FACT
    to disclose on a failure, never a reason to refuse or to warn on every generation. Compared stripped, because the
    registry round-trip is not guaranteed to preserve a trailing newline.

    :param prompt: The resolved registry prompt, as the generator received it, before code appended anything.
    :ptype prompt: str
    :param default: The shipped default it is compared with.
    :ptype default: str
    :param prompt_type: The registry slot the prompt resolves under.
    :ptype prompt_type: str
    :param seed_module: The file the default ships in, as an operator would look for it.
    :ptype seed_module: str
    :return: A clause to append to the failure summary, beginning with punctuation; empty when the prompt is the
        shipped default.
    :rtype: str
    """
    if prompt.strip() == default.strip():
        return ""
    return (
        f". NOTE: the resolved {prompt_type} prompt is NOT the in-repo default — the store is master, so a rule "
        f"shipped in {seed_module} reaches this deployment only when pushed to the host's prompt store; a refusal "
        "naming a rule the default teaches is the signature of that gap"
    )
