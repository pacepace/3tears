"""The simulated user's prompts are content-addressed on the run, beside how it was asked (engine item 2).

``simulator_prompt_digest`` digests what a conversation's simulated user is told, as a run sends it: every actor's
utterance call and, under the ``llm_decided`` scheduler, the scheduler's system prompt, each with the parts a cell
fills (the case's scenario parameters, the transcript so far) left empty. Contracts held here:

- **stable for one prompt**: one conversation digests alike every time, and so does an equal one built apart;
- **it moves with the prompt**: an actor's policy or intent, the actors' order and the scheduler each move it;
- **the launch stamps it** beside ``simulator_asked_settings`` for a run that pins a simulated user over a conversation;
- **unset on a kind with no simulator**: a run simulating nobody carries none, and dumps with no key.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.host import with_registry
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND, ToyExtractorKind
from packages.evals.tests.fixtures.toyhost.launch import plan_toyhost_arm, toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template, toyhost_test_cases
from packages.evals.tests.off_loop import off_loop
from threetears.evals.kernel import EvalStorage
from threetears.evals.run import (
    SIMULATOR_REQUEST_SETTINGS,
    ArmPlan,
    KindWiring,
    LaunchHost,
    LaunchRequest,
    launch_run,
    start_run,
)
from threetears.evals.run.simulator import simulator_prompt_digest
from threetears.evals.schema import ActorPolicy, ConversationSpec, EvalRun
from threetears.evals.storage import InMemoryDocumentStore

#: The simulated user's model.
_SIMULATOR = "sim-a"

_GUILD = ActorPolicy(id="guildmate", policy="Terse, a little impatient.", intent="Find out when the raid starts.")
_NEWCOMER = ActorPolicy(id="newcomer", policy="Polite and unsure.", intent="Ask how to join the raid.")


def _conversation(*actors: ActorPolicy, scheduler: str = "round_robin") -> ConversationSpec:
    return ConversationSpec.model_validate(
        {"actors": [actor.model_dump() for actor in actors], "max_turns": 2, "turn_scheduler": scheduler}
    )


# --- stable for one prompt, moving with it -----------------------------------------------------------------------


def test_one_conversation_digests_alike_every_time() -> None:
    first = simulator_prompt_digest(_conversation(_GUILD, _NEWCOMER))
    assert first == simulator_prompt_digest(_conversation(_GUILD, _NEWCOMER))
    assert len(first) == 64 and int(first, 16) >= 0


def test_the_digest_moves_with_a_policy_an_intent_the_actors_order_and_the_scheduler() -> None:
    held = simulator_prompt_digest(_conversation(_GUILD, _NEWCOMER))
    moved = {
        "policy": _conversation(_GUILD.model_copy(update={"policy": "Warm and patient."}), _NEWCOMER),
        "intent": _conversation(_GUILD.model_copy(update={"intent": "Complain about loot."}), _NEWCOMER),
        "order": _conversation(_NEWCOMER, _GUILD),
        "scheduler": _conversation(_GUILD, _NEWCOMER, scheduler="llm_decided"),
    }
    digests = {what: simulator_prompt_digest(conversation) for what, conversation in moved.items()}
    assert held not in digests.values(), digests
    assert len(set(digests.values())) == len(digests)


# --- the launch stamps it ------------------------------------------------------------------------------------------


def _host(conversation: ConversationSpec | None) -> tuple[LaunchHost, EvalStorage]:
    """The toy host launching its extractor over a template with ``conversation``, a simulated user pinned where
    the template states one."""
    storage = EvalStorage(InMemoryDocumentStore())
    template = toyhost_template().model_copy(update={"conversation": conversation})
    storage.save_template(template)
    toy, client = toyhost_launch_host(storage=storage)
    (launchable,) = toy.kinds.values()
    eval_host = toy.eval_host
    simulator = _SIMULATOR if conversation is not None else None
    holder: list[LaunchHost] = []

    def plan(request: LaunchRequest) -> ArmPlan:
        return replace(plan_toyhost_arm(request), simulator_model=simulator)

    async def launch(request: LaunchRequest) -> EvalRun:
        world = eval_host.profile.world
        assert world is not None, "the toy host declares a world"
        kind = ToyExtractorKind(client=client, world=world, goal_checks=tuple(request.template.goal_state_checks))
        return await launch_run(
            holder[0],
            request,
            KindWiring(
                kind_factory=lambda _cell: kind,
                subject=TOYHOST_SUBJECT,
                test_cases=toyhost_test_cases(request.template),
                simulator_model=simulator,
            ),
        )

    host = replace(
        toy,
        eval_host=with_registry(
            eval_host,
            {
                TOY_EXTRACTOR_KIND: replace(
                    launchable,
                    launch=launch,
                    plan_arm=plan,
                    unhonoured_launch_arguments=frozenset({"cassette_mode", "n_variations", "simulator_model"}),
                )
            },
        ),
    )
    holder.append(host)
    return host, storage


async def _launched(conversation: ConversationSpec | None) -> EvalRun:
    host, _storage = off_loop(_host, conversation)
    (run,) = await start_run(
        host,
        template_id=toyhost_template().id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
    )
    async with asyncio.timeout(10):
        while host.job_manager.is_active(run.id):
            await asyncio.sleep(0.01)
    return run


async def test_a_run_that_pins_a_simulated_user_carries_its_prompts_digest_beside_its_asked_settings() -> None:
    conversation = _conversation(_GUILD, _NEWCOMER)

    run = await _launched(conversation)

    assert run.simulator_model == _SIMULATOR
    assert run.simulator_asked_settings == SIMULATOR_REQUEST_SETTINGS
    assert run.simulator_prompt_digest == simulator_prompt_digest(conversation)


async def test_a_run_that_simulates_nobody_carries_no_digest_and_no_key() -> None:
    run = await _launched(None)

    assert (run.simulator_model, run.simulator_prompt_digest) == (None, None)
    assert "simulator_prompt_digest" not in run.to_dict()
