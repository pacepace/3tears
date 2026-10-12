"""A subject arm whose cap cannot afford even the smallest judge or simulator call it must make is refused first.

A judge launch under a $0.50 cap was admitted, then reserved one call at $0.58 and stopped after paying for its
first case. The judge kind now prices its calls before launch; a subject kind (a persona, a classifier) could not,
because its judge and simulator prompts carry a transcript that only its run writes. But the transcript is not
what dominates such a ceiling — the client's output cap and its retry multiplier are — and those are known before
launch. So a subject kind prices a FLOOR (``price_subject_call_floor``): each call with every part its cell fills
left empty. These pin:

- **The floor is a real call with text removed, never added.** Every judge dim's floor request has the real
  call's system prompt and reply format, and its user prompt is a subsequence of the real one, for conversation
  and document judges, with and without goal checks, variation and a subject. The simulated user's floor is a
  subsequence of every real utterance call of the actor. That is what makes the bound sound, on a client whose
  ceiling does not fall as a prompt grows.
- **An arm whose floor is above its cap is refused** by the estimate (``quote_launch`` and ``launch_estimate``)
  and the launch (``start_run``) alike, word for word, before any launcher runs or any call is made, in words
  that say the figure is a floor; an arm under it is admitted and launched.
- **The simulated user counts only when it is bound to speak**, and is priced on the simulator role's client.

Mutations that turn this file red: the floor keeping the variation's "(none)" or the goal block's "(no goal-state
checks declared)"; ``_largest_call_over_cap`` ignoring ``lower_bound``; the simulator floor taking the largest
actor rather than the smallest; ``_utterance_certain`` returning ``True`` for an opener-only conversation.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.kernel import EvalStorage, ValidationFailedError
from threetears.evals.kernel import withhold_failure_detail
from threetears.evals.ops import LaunchArguments, OpsHost, launch_estimate
from threetears.evals.run import (
    JUDGE_CALL_ATTEMPTS,
    ArmPlan,
    JudgeService,
    KindWiring,
    LaunchHost,
    LaunchRequest,
    TurnDriver,
    build_judge_service,
    launch_run,
    plan_judge,
    price_subject_call_floor,
    quote_launch,
    require_candidate_model,
    start_run,
)
from threetears.evals.run.judge_service import JudgeContext
from threetears.evals.run.simulator import SIMULATED_USER_RESPONSE_FORMAT, smallest_utterance_prompts
from threetears.evals.schema import (
    ActorPolicy,
    ConversationSpec,
    EvalRun,
    EvalTemplate,
    GoalStateOutcome,
    JudgedArtifact,
    JudgeEvidence,
    RubricDim,
    WorldRound,
)
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_judge_config
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND
from packages.evals.tests.fixtures.toyhost.launch import (
    TOYHOST_LAUNCH_SETTINGS,
    plan_toyhost_arm,
    price_toyhost_arm,
    toyhost_launch_host,
)
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template, toyhost_test_cases
from packages.evals.tests.off_loop import off_loop

#: The dim the subject's judge scores.
_DIM = "toy.fidelity"

#: The judge model the arm is scored by.
_JUDGE = "judge-flash"

#: The simulator model the arm's simulated user runs on.
_SIMULATOR = "sim-a"

#: What the host's judge client prices one request at, whatever its prompt: dominated by its output cap and
#: its retries, as the incident's was.
_JUDGE_REQUEST_USD = 0.29

#: What the judge floor is: every attempt the judge's parse retry can make.
_JUDGE_FLOOR_USD = _JUDGE_REQUEST_USD * JUDGE_CALL_ATTEMPTS


class _PricedClient:
    """A client that prices every request at one figure, and records what it priced and what it was sent."""

    def __init__(self, role: str, model: str | None, usd: float | None) -> None:
        self.role = role
        self.model_name = model or "default"
        self.usd = usd
        self.priced: list[tuple[str, str, Any]] = []
        self.calls: list[tuple[str, str]] = []
        self.closed = 0

    def price_ceiling(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> float | None:
        self.priced.append((system, user, response_format))
        return self.usd

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        self.calls.append((system, user))
        raise AssertionError("no call is made by a launch that pricing refuses")

    async def aclose(self) -> None:
        self.closed += 1


def _template(conversation: ConversationSpec | None = None) -> EvalTemplate:
    return toyhost_template().model_copy(
        update={
            "rubric": [RubricDim(name=_DIM, description="the extraction is faithful", scale="ordinal")],
            "conversation": conversation,
        }
    )


class _Subject:
    """The toy host as a judged subject kind, whose entry prices its floor with ``price_subject_call_floor``."""

    def __init__(
        self,
        *,
        judge_usd: float | None = _JUDGE_REQUEST_USD,
        simulator_usd: float | None = None,
        conversation: ConversationSpec | None = None,
        judged_artifact: JudgedArtifact = JudgedArtifact.DOCUMENT,
    ) -> None:
        self.storage = EvalStorage(InMemoryDocumentStore())
        self.template = _template(conversation)
        self.storage.save_template(self.template)
        self.clients: list[_PricedClient] = []
        self.launched: list[LaunchRequest] = []

        def clients(role: str, model: str | None, **_: Any) -> _PricedClient:
            client = _PricedClient(role, model, judge_usd if role == "judge" else simulator_usd)
            self.clients.append(client)
            return client

        toy, _client = toyhost_launch_host(storage=self.storage, clients=clients)  # type: ignore[arg-type]
        (launchable,) = toy.kinds.values()
        eval_host = toy.eval_host
        simulator = _SIMULATOR if conversation is not None else None

        def plan(request: LaunchRequest) -> ArmPlan:
            return replace(
                plan_toyhost_arm(request),
                judge=plan_judge(
                    eval_host, request.template, _JUDGE, request.judge_config_ids, judged_artifact=judged_artifact
                ),
                simulator_model=simulator,
            )

        async def launch(request: LaunchRequest) -> EvalRun:
            self.launched.append(request)
            require_candidate_model(request, None)
            judge = off_loop(
                build_judge_service,
                eval_host,
                request.template,
                _JUDGE,
                request.judge_config_ids,
                judged_artifact=judged_artifact,
            )
            return await launch_run(
                self.host,
                request,
                KindWiring(
                    kind_factory=lambda _cell: None,  # type: ignore[arg-type,return-value]
                    subject=TOYHOST_SUBJECT,
                    test_cases=toyhost_test_cases(request.template),
                    judge=judge,
                    simulator_model=simulator,
                ),
            )

        self.host: LaunchHost = replace(
            toy,
            kinds={
                TOY_EXTRACTOR_KIND: replace(
                    launchable,
                    launch=launch,
                    plan_arm=plan,
                    price_largest_call=lambda request: price_subject_call_floor(
                        eval_host, request, judged_artifact=judged_artifact
                    ),
                    unhonoured_launch_arguments=frozenset({"cassette_mode", "n_variations", "simulator_model"}),
                )
            },
            launch_pricer=price_toyhost_arm,
        )

    def _arguments(self, max_cost_usd: float) -> dict[str, Any]:
        return {
            "template_id": self.template.id,
            "subject_id": TOYHOST_SUBJECT.subject_id,
            "models": [RUN_MODELS[0]],
            "k_runs": 1,
            "max_cost_usd": max_cost_usd,
        }

    async def estimate(self, max_cost_usd: float) -> Any:
        [arm] = (await quote_launch(self.host, **self._arguments(max_cost_usd), scope_id=TOYHOST_SCOPE)).arms
        return arm

    async def ops_estimate(self, max_cost_usd: float) -> Any:
        arguments = LaunchArguments(**self._arguments(max_cost_usd))
        [arm] = (await launch_estimate(OpsHost(launch=self.host), arguments, TOYHOST_SCOPE)).arms
        return arm

    async def launch(self, max_cost_usd: float) -> list[EvalRun]:
        runs = await start_run(self.host, **self._arguments(max_cost_usd), scope_id=TOYHOST_SCOPE)
        async with asyncio.timeout(10):
            while any(self.host.job_manager.is_active(run.id) for run in runs):
                await asyncio.sleep(0.01)
        return runs

    def calls(self) -> list[tuple[str, str]]:
        return [call for client in self.clients for call in client.calls]

    def priced_by(self, role: str) -> list[tuple[str, str, Any]]:
        return [each for client in self.clients if client.role == role for each in client.priced]


# =============================================================================
# The floor is the real call with the cell's parts left out
# =============================================================================


def _is_subsequence(short: str, long: str) -> bool:
    remaining = iter(long)
    return all(char in remaining for char in short)


def _service(configs: dict[str, Any] | None = None) -> JudgeService:
    return JudgeService(
        client_factory=lambda _model, _temperature: None,
        configs=configs,
        failure_describer=withhold_failure_detail,
    )


_RUBRIC = [
    RubricDim(name=_DIM, description="the extraction is faithful", scale="ordinal"),
    RubricDim(name="toy.tone", description="it reads well", scale="pass_fail", scoring_guide={"pass": "it does"}),
]


@pytest.mark.parametrize("judged_artifact", [JudgedArtifact.TRANSCRIPT, JudgedArtifact.DOCUMENT])
@pytest.mark.parametrize("variation", [{}, {"topic": "raids", "mood": "tired"}])
@pytest.mark.parametrize("goals", [[], [GoalStateOutcome(expression="a == 1", passed=True, detail="it was")]])
@pytest.mark.parametrize("subject", [None, "the bot, as its players see it"])
@pytest.mark.parametrize("configured", [False, True])
def test_every_judge_calls_floor_is_the_real_call_with_text_removed(
    judged_artifact: JudgedArtifact,
    variation: dict[str, str],
    goals: list[GoalStateOutcome],
    subject: str | None,
    configured: bool,
) -> None:
    configs = (
        {_DIM: make_judge_config(scope_id=TOYHOST_SCOPE, rubric_dim_id=_DIM, prompt_template="Judge it strictly.")}
        if configured
        else None
    )
    service = _service(configs=configs)
    intent = "Answer the member's question."
    context = JudgeContext(
        case_id="c-1",
        intent=intent,
        variation=variation,
        goal_outcomes=goals,
        judged_artifact=judged_artifact,
        judge_evidence=JudgeEvidence(subject=subject, case_material="the member asked", artifact="the bot replied"),
    )
    real = [
        *(
            [service.transcript_request(context), service.outcome_request(context)]
            if judged_artifact is JudgedArtifact.TRANSCRIPT
            else []
        ),
        *[service.dimension_request(dim, context) for dim in _RUBRIC],
    ]
    floors = service.floor_requests(_RUBRIC, intent=intent, judged_artifact=judged_artifact)

    assert [floor.dim_id for floor in floors] == [request.dim_id for request in real]
    for floor, request in zip(floors, real, strict=True):
        assert floor.system_prompt == request.system_prompt, "the instructions and reply contract are fixed"
        assert floor.response_format == request.response_format
        assert floor.config == request.config
        assert _is_subsequence(floor.user_prompt, request.user_prompt), (
            f"{floor.dim_id}: the floor sends text the real call does not:\n{floor.user_prompt!r}\n"
            f"vs\n{request.user_prompt!r}"
        )
        assert len(floor.user_prompt) < len(request.user_prompt)
        assert intent in floor.user_prompt, "the template's intent is fixed, so the floor carries it"


def test_an_unjudged_kind_has_no_judge_floor() -> None:
    with pytest.raises(ValueError, match="no judge to price"):
        _service().floor_requests(_RUBRIC, intent="x", judged_artifact=JudgedArtifact.UNJUDGED)


class _Recording:
    """A simulator client recording every request it is sent, answering one line."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str, Any]] = []

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> Any:
        self.sent.append((system, user, response_format))
        return type("Reply", (), {"content": '{"utterance": "hello", "done": false}'})()


@pytest.mark.parametrize("variation", [{}, {"topic": "raids"}])
@pytest.mark.parametrize("transcript", [[], [("__candidate__", "hi there"), ("alice", "and you")]])
async def test_every_utterance_calls_floor_is_the_real_call_with_text_removed(
    variation: dict[str, str], transcript: list[tuple[str, str]]
) -> None:
    actor = ActorPolicy(id="alice", policy="Curt, impatient.", intent="Get the raid time.")
    driver = TurnDriver(
        conversation=ConversationSpec(actors=[actor], max_turns=2), variation=variation, transcript=list(transcript)
    )
    llm = _Recording()

    await driver.next_user_turn(llm, actor)  # type: ignore[arg-type]

    [(system, user, response_format)] = llm.sent
    floor_system, floor_user = smallest_utterance_prompts(actor)
    assert response_format == SIMULATED_USER_RESPONSE_FORMAT
    assert _is_subsequence(floor_system, system) and len(floor_system) < len(system)
    assert _is_subsequence(floor_user, user) and len(floor_user) < len(user)
    assert actor.policy in floor_system and actor.intent in floor_user


# =============================================================================
# The estimate and the launch refuse an arm whose floor is above its cap, before anything is spent
# =============================================================================


async def test_a_cap_below_the_judges_floor_is_refused_by_the_estimate_and_the_launch_with_nothing_spent() -> None:
    subject = off_loop(_Subject)

    arm = await subject.estimate(0.5)
    assert arm.outcome == "refused"
    ceiling = arm.plan.call_ceiling
    assert (ceiling.usd, ceiling.lower_bound) == (pytest.approx(_JUDGE_FLOOR_USD), True)
    assert f"the '{_DIM}' judge call every judged case makes" in ceiling.call and repr(_JUDGE) in ceiling.call
    assert "even the smallest such call this arm could make" in arm.refusal
    assert "is reserved at $0.5800 before it is sent, more than the whole cap" in arm.refusal
    assert "Refused before any launcher ran" in arm.refusal
    assert arm.remedy.startswith("name max_cost_usd above $0.58 — a floor, not the cap the arm needs")
    assert "a cheaper judge model" in arm.remedy, "the remedy names what lowers the floor, not only the cap"
    assert "of at least" not in arm.remedy, "a floor never names a cap as if it were enough"

    reported = await subject.ops_estimate(0.5)
    assert (reported.outcome, reported.refusal, reported.remedy) == ("refused", arm.refusal, arm.remedy)
    assert reported.call_ceiling_usd == pytest.approx(_JUDGE_FLOOR_USD)
    assert reported.call_ceiling_lower_bound is True

    with pytest.raises(ValidationFailedError) as refused:
        await subject.launch(0.5)
    assert refused.value.message == arm.refusal, "the estimate refuses word for word what the launch refuses"
    assert refused.value.remedy == arm.remedy

    assert subject.launched == [], "no launcher ran"
    assert subject.calls() == [], "no call was made"
    assert subject.priced_by("judge"), "the judge's floor was priced on the judge role's client"
    assert all(client.closed == 1 for client in subject.clients), "every client built to price was released"
    assert off_loop(subject.storage.query_eval_runs, TOYHOST_SCOPE) == []
    assert subject.host.job_manager.admitted_count == 0


async def test_a_cap_above_the_floor_is_admitted_and_launched() -> None:
    subject = off_loop(_Subject)

    arm = await subject.estimate(0.6)
    assert (arm.outcome, arm.refusal) == ("admitted", None)
    assert arm.plan.call_ceiling.lower_bound is True

    runs = await subject.launch(0.6)
    assert len(runs) == 1 and len(subject.launched) == 1, "admitted and launched"


async def test_a_floor_the_host_cannot_raise_a_cap_to_names_the_hosts_operator() -> None:
    subject = off_loop(_Subject, judge_usd=TOYHOST_LAUNCH_SETTINGS.max_cost_usd)

    arm = await subject.estimate(0.5)
    assert arm.outcome == "refused"
    assert arm.remedy.startswith("even the smallest such call needs a cap above $")
    assert "which a launch cannot raise" in arm.remedy


async def test_a_judge_client_that_cannot_price_leaves_the_arm_to_its_other_rules() -> None:
    subject = off_loop(_Subject, judge_usd=None)

    arm = await subject.estimate(0.5)
    assert arm.plan.call_ceiling is None
    assert arm.outcome == "admitted"


async def test_a_dims_floor_is_priced_on_the_model_its_config_pins() -> None:
    subject = off_loop(_Subject)
    off_loop(
        subject.storage.save_judge_config, make_judge_config(scope_id=TOYHOST_SCOPE, rubric_dim_id=_DIM, model="dear")
    )

    arm = await subject.estimate(0.5)
    assert "'dear'" in arm.plan.call_ceiling.call
    assert {client.model_name for client in subject.clients if client.role == "judge"} == {"dear"}


# =============================================================================
# The simulated user, when it is bound to speak
# =============================================================================

_ACTORS = [
    ActorPolicy(id="alice", policy="Curt.", intent="Get the raid time."),
    ActorPolicy(id="bob", policy="Chatty, at length, about everything.", intent="Make friends."),
]


async def test_a_cap_below_the_simulated_users_floor_is_refused_naming_the_simulator() -> None:
    subject = off_loop(
        _Subject,
        judge_usd=0.01,
        simulator_usd=0.7,
        conversation=off_loop(ConversationSpec, actors=_ACTORS, max_turns=3),
    )

    arm = await subject.estimate(0.5)
    assert arm.outcome == "refused"
    ceiling = arm.plan.call_ceiling
    assert (ceiling.usd, ceiling.lower_bound) == (pytest.approx(0.7), True)
    assert "a simulated-user utterance" in ceiling.call and repr(_SIMULATOR) in ceiling.call
    assert "a cheaper simulator model" in arm.remedy
    priced = subject.priced_by("simulator")
    assert [user for _system, user, _format in priced] == [smallest_utterance_prompts(a)[1] for a in _ACTORS]
    assert all(response_format == SIMULATED_USER_RESPONSE_FORMAT for _s, _u, response_format in priced)

    with pytest.raises(ValidationFailedError) as refused:
        await subject.launch(0.5)
    assert refused.value.message == arm.refusal
    assert subject.launched == [] and subject.calls() == []


async def test_the_larger_floor_of_the_judge_and_the_simulated_user_is_the_arms() -> None:
    subject = off_loop(
        _Subject, simulator_usd=0.1, conversation=off_loop(ConversationSpec, actors=_ACTORS, max_turns=3)
    )

    arm = await subject.estimate(0.5)
    assert arm.plan.call_ceiling.usd == pytest.approx(_JUDGE_FLOOR_USD)
    assert "judge call" in arm.plan.call_ceiling.call


async def test_the_simulated_users_floor_is_unknown_when_one_actors_call_is() -> None:
    subject = off_loop(
        _Subject,
        judge_usd=None,
        simulator_usd=None,
        conversation=off_loop(ConversationSpec, actors=_ACTORS, max_turns=3),
    )

    arm = await subject.estimate(0.5)
    assert arm.plan.call_ceiling is None


@pytest.mark.parametrize(
    "conversation",
    [
        # The opener fills round one's only slot, and the conversation ends after it.
        ConversationSpec(
            actors=[_ACTORS[0].model_copy(update={"initial_utterance_template": "hi"})],
            max_turns=1,
            max_speakers_per_round=1,
        ),
        # The scheduler may end round one after the opener.
        ConversationSpec(
            actors=[_ACTORS[0].model_copy(update={"initial_utterance_template": "hi"}), _ACTORS[1]],
            max_turns=1,
            max_speakers_per_round=2,
            turn_scheduler="llm_decided",
        ),
    ],
    ids=["opener-only", "scheduler-may-end-the-round"],
)
async def test_a_conversation_not_bound_to_call_for_an_utterance_prices_no_simulator_floor(
    conversation: ConversationSpec,
) -> None:
    subject = off_loop(_Subject, judge_usd=0.01, simulator_usd=0.7, conversation=conversation)

    arm = await subject.estimate(0.5)
    assert subject.priced_by("simulator") == []
    assert "judge call" in arm.plan.call_ceiling.call


async def test_a_world_round_makes_no_utterance_but_the_round_after_it_does() -> None:
    world_only = ConversationSpec(actors=_ACTORS, max_turns=1, world_rounds=[WorldRound(turn=1, dimension="toy.inbox")])
    assert await _simulator_priced(world_only) is False
    world_first = world_only.model_copy(update={"max_turns": 2})
    assert await _simulator_priced(world_first) is True


async def _simulator_priced(conversation: ConversationSpec) -> bool:
    subject = off_loop(_Subject, judge_usd=0.01, simulator_usd=0.7, conversation=conversation)
    plan = ArmPlan(
        case_count=1,
        candidate_model=RUN_MODELS[0],
        judge=None,
        simulator_model=_SIMULATOR,
    )
    request = await _request(subject, plan)
    ceiling = await price_subject_call_floor(subject.host.eval_host, request, judged_artifact=JudgedArtifact.TRANSCRIPT)
    return ceiling is not None


async def _request(subject: _Subject, plan: ArmPlan) -> LaunchRequest:
    """A planned request for ``subject``'s template, as the estimate hands its kind's pricer one."""
    seen: list[LaunchRequest] = []
    launchable = subject.host.kinds[TOY_EXTRACTOR_KIND]

    async def capture(request: LaunchRequest) -> None:
        seen.append(request)
        return None

    host = replace(
        subject.host,
        kinds={TOY_EXTRACTOR_KIND: replace(launchable, plan_arm=lambda _request: plan, price_largest_call=capture)},
    )
    await quote_launch(
        host,
        template_id=subject.template.id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
        k_runs=1,
        max_cost_usd=0.5,
        scope_id=TOYHOST_SCOPE,
    )
    [request] = seen
    return replace(request, template=subject.template)
