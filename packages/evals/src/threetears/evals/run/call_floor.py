"""A floor under a subject arm's judge and simulator calls, priced before launch.

A launch refuses an arm whose largest call is reserved above its cap (:class:`~threetears.evals.run.CallCeiling`),
since its run could never make that call. The judge kind and the reporter price their largest call exactly: their
prompts are frozen cases. A subject kind's judge and simulator prompts carry a transcript and evidence that only
its run writes, so its largest call cannot be priced before launch — but a FLOOR under it can.

**The floor.** Each call the cell makes is a template-fixed part around a part the cell fills:

- a judge call (:meth:`~threetears.evals.run.JudgeService.floor_requests`) is the dim's instructions and reply
  contract, the template's intent and every heading, around the case's variation, the goal-state outcomes, the
  kind's subject, its case material and its artifact;
- a simulated-user utterance is the actor's policy and intent around the case's variation and the transcript.

Leaving the filled part empty gives the smallest request the cell could send, and every real one is it with text
inserted. Priced on the client the cell's call goes through — the judge's per dim, through the run's judge service
built as the launcher builds it, every parse-retry attempt included, and the simulator role's for the planned
simulator model — that is a reservation the real call cannot come in under.

**Why it is sound.** It rests on one property of the host's client: its ``price_ceiling`` does not fall when text
is added to a prompt. A ceiling priced as the protocol asks — a rate applied to a bound on the prompt's length and
to a fixed output cap, times the client's own re-sends
(:meth:`~threetears.evals.schema.completion.PricedCompletion.price_ceiling`) — has it. Then:

- **the judge:** every judged result makes every dim's call, so each dim's floor is under a call every result
  makes, and the largest of them is under the arm's largest call;
- **the simulated user:** which actor speaks is the conversation's, so the floor is the smallest over the actors,
  and it is counted only when the conversation is bound to make an utterance call (:func:`_utterance_certain`);
  the ``llm_decided`` scheduler's calls are not priced, as none of them is.

**What it cannot see.** The output cap and the client's retry multiplier dominate many ceilings — the incident a
judge-kind arm hit had a $0.58 call under a $0.50 cap — and those are whole in the floor. A ceiling dominated by a
long transcript is not: a floor under the cap admits the arm, and that call is held to the cap as the run
reserves it, as before. A candidate's own calls are the kind's, not the engine's, and are not priced here.
"""

from __future__ import annotations

from dataclasses import dataclass

from threetears.evals.kernel.capped_completion import call_ceiling
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS, JUDGE_MAX_TOKENS
from threetears.evals.run.launch import CallCeiling, LaunchRequest, build_judge_service
from threetears.evals.run.simulator import (
    SIMULATED_USER_RESPONSE_FORMAT,
    SIMULATOR_MAX_TOKENS,
    smallest_utterance_prompts,
)
from threetears.evals.schema.models import ConversationSpec, JudgedArtifact

__all__ = ["price_subject_call_floor"]


@dataclass(frozen=True)
class _Floor:
    """One call's floor, with the words a refusal quotes."""

    usd: float
    call: str
    lowered_by: str


async def price_subject_call_floor(
    host: EvalHost, request: LaunchRequest, *, judged_artifact: JudgedArtifact
) -> CallCeiling | None:
    """A floor under a subject arm's largest judge or simulator call, for its ``LaunchableKind.price_largest_call``.

    What a host composes into the launch entry of each of its own subject kinds — any conversational or
    single-shot kind its launcher builds a judge for with
    :func:`~threetears.evals.run.build_judge_service`:
    ``price_largest_call=lambda request: price_subject_call_floor(eval_host, request, judged_artifact=...)``.
    The estimate and the launch then refuse an arm whose cap cannot afford even the smallest call its cells are
    bound to make, before anything is spent (see the module docstring for the floor and why it is sound).

    Priced on the clients the cells call through, built as a launcher builds them: the judge's from the run's
    judge service over the arm's plan (its pin, the launch's config selection and the kind's declaration, as
    :func:`~threetears.evals.run.build_judge_service` resolves them), and the simulated user's from the host's
    factory in the ``simulator`` role for the planned simulator model. So pass the ``host`` your launcher builds
    both on, and the ``judged_artifact`` your launcher hands ``build_judge_service``.

    :param host: The host the launcher builds the judge and the simulator client on.
    :ptype host: EvalHost
    :param request: The arm, as planned.
    :ptype request: LaunchRequest
    :param judged_artifact: What the kind's judge reads — the declaration its launcher builds its judge with.
    :ptype judged_artifact: JudgedArtifact
    :return: The larger of the judge's and the simulated user's floor, flagged a lower bound
        (:attr:`~threetears.evals.run.CallCeiling.lower_bound`); ``None`` for an arm with no plan, or one whose
        plan names neither a judge nor a simulated user bound to speak, or whose clients cannot price a call.
    :rtype: CallCeiling | None
    :raises ValidationFailedError: The launch's judge-config selection does not fit the template (as
        :func:`~threetears.evals.run.plan_judge` refuses it).
    :raises ValueError: The host supplies no completion clients, or a client priced a call at an amount no
        reservation could hold.
    """
    plan = request.arm_plan
    if plan is None:
        return None
    floors: list[_Floor] = []
    if plan.judge is not None and judged_artifact is not JudgedArtifact.UNJUDGED:
        judge = await _judge_floor(host, request, plan.judge.model, judged_artifact)
        if judge is not None:
            floors.append(judge)
    conversation = request.template.conversation
    if plan.simulator_model is not None and conversation is not None and _utterance_certain(conversation):
        simulator = await _simulator_floor(host, plan.simulator_model, conversation)
        if simulator is not None:
            floors.append(simulator)
    if not floors:
        return None
    floor = max(floors, key=lambda each: each.usd)
    return CallCeiling(usd=floor.usd, call=floor.call, lowered_by=floor.lowered_by, lower_bound=True)


async def _judge_floor(
    host: EvalHost, request: LaunchRequest, judge_model: str, judged_artifact: JudgedArtifact
) -> _Floor | None:
    """The largest of the per-dim judge floors: each is under a call every judged result makes."""
    template = request.template
    run_judge = await run_blocking(
        host.blocking_executor,
        lambda: build_judge_service(
            host, template, judge_model, request.judge_config_ids, judged_artifact=judged_artifact
        ),
    )
    service = run_judge.service
    try:

        def priced() -> list[tuple[float, str]]:
            requests = service.floor_requests(template.rubric, intent=template.intent, judged_artifact=judged_artifact)
            return [(usd, each.dim_id) for each in requests if (usd := service.price_ceiling(each)) is not None]

        floors = await run_blocking(host.blocking_executor, priced)
    finally:
        await service.aclose()
    if not floors:
        return None
    usd, dim_id = max(floors)
    model = run_judge.effective_judges.get(dim_id, judge_model)
    return _Floor(
        usd=usd,
        call=(
            f"the {dim_id!r} judge call every judged case makes ({JUDGE_CALL_ATTEMPTS} request(s): the first and its "
            f"parse retry, each asking up to {JUDGE_MAX_TOKENS} output tokens) on {model!r}"
        ),
        lowered_by=(
            f"a cheaper judge model for {dim_id!r} (the launch's judge_model, or the model its judge config pins); "
            "shorter judge instructions for it (its judge config's prompt); or, on the host, a judge-role client "
            "that re-sends a request fewer times (its retry policy multiplies every request's ceiling) or whose "
            f"price_ceiling prices the {JUDGE_MAX_TOKENS}-token output cap the judge asks for rather than a larger one"
        ),
    )


async def _simulator_floor(host: EvalHost, simulator_model: str, conversation: ConversationSpec) -> _Floor | None:
    """The smallest utterance floor over the actors, or ``None`` when any actor's call cannot be priced."""
    client = host.completion_clients("pricing a simulated user")("simulator", simulator_model)
    try:
        floors: list[tuple[float, str]] = []
        for actor in conversation.actors:
            system, user = smallest_utterance_prompts(actor)
            usd = call_ceiling(client, system=system, user=user, response_format=SIMULATED_USER_RESPONSE_FORMAT)
            if usd is None:
                # The minimum over the actors is unknown when one of them is: that actor might be the one to speak.
                return None
            floors.append((usd, actor.id))
    finally:
        await client.aclose()
    usd, _actor_id = min(floors)
    return _Floor(
        usd=usd,
        call=(
            f"a simulated-user utterance (any of the conversation's {len(floors)} actor(s), each asking up to "
            f"{SIMULATOR_MAX_TOKENS} output tokens) on {simulator_model!r}"
        ),
        lowered_by=(
            "a cheaper simulator model (the launch's simulator_model); shorter actor policies and intents on the "
            "template; or, on the host, a simulator-role client that re-sends a request fewer times (its retry "
            f"policy multiplies every request's ceiling) or whose price_ceiling prices the {SIMULATOR_MAX_TOKENS}-token "
            "output cap the simulated user asks for rather than a larger one"
        ),
    )


def _utterance_certain(conversation: ConversationSpec) -> bool:
    """Whether a conversation that runs as designed makes at least one simulated-user utterance call.

    A round answered by the world makes none, and neither does round one when the first actor's opener fills it
    (its only slot), or when the ``llm_decided`` scheduler may end it after the opener. Any other actors' round
    begins empty, and an empty round's first speaker is an actor — under either scheduler, since a round that
    has delivered nothing cannot be declared done — whose line is an utterance call.

    :param conversation: The template's conversation.
    :ptype conversation: ConversationSpec
    :return: Whether some round up to the turn limit is bound to call for an utterance.
    :rtype: bool
    """
    if not conversation.actors:
        return False
    opener = bool(conversation.actors[0].initial_utterance_template)
    for turn in range(1, conversation.max_turns + 1):
        if conversation.world_round(turn) is not None:
            continue
        if turn > 1 or not opener:
            return True
        if conversation.turn_scheduler == "round_robin" and conversation.max_speakers_per_round > 1:
            return True
    return False
