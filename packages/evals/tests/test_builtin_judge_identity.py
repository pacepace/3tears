"""The built-in judge prompt is an instrument with an identity, and it tells the judge where its own recall stands.

A dimension no :class:`~threetears.evals.schema.models.JudgeConfig` scores is judged with the engine's own
wording. Every reader that keys a judge — agreement, judge profiles, drift, evidence tiers — keys it by the
config that asked, so a built-in prompt recorded as ``None`` made every built-in wording one judge: a change
to that wording silently changed the instrument while every stored figure still applied to it. So:

* a built-in call records ``builtin:<sha12>`` where a config's id stands, a fingerprint of the wording sent;
* two results judged by the same wording are one judge and pool; a different wording is a different judge;
* a result stored before the identity was recorded (``None``) is a built-in prompt of unknown wording and
  pools with no ``builtin:`` judge;
* a dim's own description and guide are its criterion, not its judge, and leave the identity alone;
* a configured judge records its config's id exactly as before.

And the built-in prompt of every axis tells the judge that what the candidate looked up outranks the judge's own
recall (Discodon #1895) — in the built-in instructions, never in the suffix an operator's configured prompt gets.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis.agreement import judge_agreement, judge_key
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.run.judge_service import (
    JudgeContext,
    JudgeOutcome,
    JudgeService,
    builtin_judge_config_ids,
    fold_judge_outcomes,
    judge_config_ids_for,
)
from threetears.evals.schema.models import (
    BUILTIN_JUDGE_CONFIG_PREFIX,
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    GoalStateOutcome,
    JudgeConfig,
    JudgedArtifact,
    JudgeEvidence,
    RubricDim,
    builtin_judge_config_id,
    configured_judge_config_id,
    is_builtin_judge_config_id,
    judge_config_of,
)
from packages.evals.tests.factories import make_calibration_rating, make_eval_result

_TONE = RubricDim(name="conversation.tone", description="warm and on topic", scale="ordinal")
_PASS = RubricDim(name="conversation.booked", description="the booking was made", scale="pass_fail")

#: The rule every built-in axis is told, in words a test can find whatever the sentence around them.
_RULE_WORDS = ("outrank your own recall", "cannot verify it from memory", "rather than asserting fabrication")


class _Client:
    """Answers every call with a 4 (or a pass) for the dim it was asked, and records each system prompt."""

    def __init__(self) -> None:
        self.systems: list[str] = []

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        self.systems.append(system)
        match = re.search(r'the single key "(.+?)"', system)
        dim = match.group(1) if match else "?"
        answer: Any = "pass" if '"pass"' in system else 4
        return SimpleNamespace(
            temperature=0.0,
            served_model="judge-a",
            stop_reason="end_turn",
            content=json.dumps({"reasoning": "r", "criteria_scores": {dim: answer}}),
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            model="judge-a",
        )


def _service(client: _Client, configs: dict[str, JudgeConfig] | None = None) -> JudgeService:
    return JudgeService(
        client_factory=lambda _model, _temperature: client, configs=configs, failure_describer=withhold_failure_detail
    )


def _context(
    judged_artifact: JudgedArtifact = JudgedArtifact.TRANSCRIPT, artifact: str = "Candidate: on it"
) -> JudgeContext:
    return JudgeContext(
        case_id="tc-1",
        intent="book me a table",
        variation={},
        goal_outcomes=[GoalStateOutcome(expression="state.booked == True", passed=True)],
        judged_artifact=judged_artifact,
        judge_evidence=JudgeEvidence(subject="Mara", case_material="one table left", artifact=artifact),
    )


async def _judged(service: JudgeService, context: JudgeContext) -> list[tuple[str, JudgeOutcome]]:
    return [
        (TRANSCRIPT_DIM_ID, await service.score_transcript(context)),
        (OUTCOME_DIM_ID, await service.score_outcome(context)),
        (_TONE.name, await service.score_dimension(_TONE, context)),
    ]


# --- the identity --------------------------------------------------------------------------------


async def test_a_dim_no_config_scores_records_its_built_in_prompt_where_a_config_id_stands() -> None:
    outcomes = await _judged(_service(_Client()), _context())

    folded = fold_judge_outcomes(outcomes)

    assert set(folded.config_ids) == {TRANSCRIPT_DIM_ID, OUTCOME_DIM_ID, _TONE.name}
    assert all(is_builtin_judge_config_id(config_id) for config_id in folded.config_ids.values())
    assert folded.config_ids == builtin_judge_config_ids([_TONE], judged_artifact=JudgedArtifact.TRANSCRIPT)


async def test_an_axis_s_built_in_identity_is_a_fingerprint_of_the_prompt_it_was_sent() -> None:
    """So any change to the axis's wording — a word, a character — is a different judge."""
    client = _Client()
    service = _service(client)

    transcript = await service.score_transcript(_context())
    outcome = await service.score_outcome(_context())

    assert transcript.config_id == builtin_judge_config_id(client.systems[0])
    assert outcome.config_id == builtin_judge_config_id(client.systems[1])
    assert builtin_judge_config_id(client.systems[0] + " ") != transcript.config_id
    assert builtin_judge_config_id(client.systems[0].replace("DECISION", "decision")) != transcript.config_id


def test_a_built_in_identity_is_the_prefix_and_twelve_hex_characters() -> None:
    config_id = builtin_judge_config_id("Score it.")
    assert re.fullmatch(rf"{re.escape(BUILTIN_JUDGE_CONFIG_PREFIX)}[0-9a-f]{{12}}", config_id)
    assert not is_builtin_judge_config_id(None)
    assert not is_builtin_judge_config_id("0199a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b")


#: The built-in prompts' identities as this build words them. **When this goes red the built-in wording changed**,
#: and with it the judge: every stored judge profile, agreement figure and drift reading measured on the old wording
#: stops applying to scores judged with the new one, as it would for a new config. That is what a wording change
#: costs; re-pin only when it is meant, and say so in the release's notes.
_PINNED: dict[tuple[JudgedArtifact, str], str] = {
    (JudgedArtifact.TRANSCRIPT, TRANSCRIPT_DIM_ID): "builtin:f886726e63d2",
    (JudgedArtifact.TRANSCRIPT, OUTCOME_DIM_ID): "builtin:5fd0040e2c22",
    (JudgedArtifact.TRANSCRIPT, _TONE.name): "builtin:501c023ca46f",
    (JudgedArtifact.TRANSCRIPT, _PASS.name): "builtin:6a15a72a00e0",
    (JudgedArtifact.DOCUMENT, _TONE.name): "builtin:acec207171e0",
    (JudgedArtifact.DOCUMENT, _PASS.name): "builtin:c2362452be2f",
}


@pytest.mark.parametrize("judged_artifact", [JudgedArtifact.TRANSCRIPT, JudgedArtifact.DOCUMENT])
def test_the_built_in_prompts_identities_are_pinned_so_a_wording_change_is_seen(
    judged_artifact: JudgedArtifact,
) -> None:
    ids = builtin_judge_config_ids([_TONE, _PASS], judged_artifact=judged_artifact)
    pinned = {dim: config_id for (artifact, dim), config_id in _PINNED.items() if artifact is judged_artifact}
    assert ids == pinned, "the built-in judge wording changed: a new judge (see _PINNED)"


def test_a_dim_s_own_wording_is_its_criterion_and_leaves_its_judge_alone() -> None:
    """A dim's description and guide are what is judged, not who judges — a stored profile keys the criterion apart."""
    plain = RubricDim(name="conversation.warmth", description="warm", scale="ordinal")
    guided = RubricDim(
        name="conversation.clarity",
        description="clear and short",
        scale="ordinal",
        scoring_guide={"1": "muddled", "5": "crisp"},
    )

    ids = builtin_judge_config_ids([plain, guided, _PASS], judged_artifact=JudgedArtifact.TRANSCRIPT)
    document = builtin_judge_config_ids([plain], judged_artifact=JudgedArtifact.DOCUMENT)

    assert ids[plain.name] == ids[guided.name]
    assert ids[plain.name] != ids[_PASS.name], "another scale is asked in other words"
    assert document[plain.name] != ids[plain.name], "a document is asked in other words"


def test_an_unjudged_kind_has_no_built_in_prompt() -> None:
    with pytest.raises(ValueError, match="unjudged kind"):
        builtin_judge_config_ids([_TONE], judged_artifact=JudgedArtifact.UNJUDGED)


async def test_a_configured_judge_records_its_config_with_the_reply_format_and_its_siblings_the_built_in_prompt() -> (
    None
):
    config = JudgeConfig(
        scope_id="uni-1", name="tone-v2", rubric_dim_id=_TONE.name, prompt_template="CUSTOM tone instructions"
    )
    service = _service(_Client(), {_TONE.name: config})

    outcomes = await _judged(service, _context())

    folded = fold_judge_outcomes(outcomes)
    recorded = folded.config_ids[_TONE.name]
    assert recorded.startswith(f"{config.id}@") and judge_config_of(recorded) == config.id
    assert not is_builtin_judge_config_id(recorded)
    assert is_builtin_judge_config_id(folded.config_ids[TRANSCRIPT_DIM_ID])
    assert service.judge_config_ids([_TONE], judged_artifact=JudgedArtifact.TRANSCRIPT) == folded.config_ids


def test_a_configured_judge_s_identity_moves_with_the_reply_format_appended_to_its_prompt() -> None:
    """The suffix the engine appends to an operator's prompt is part of what the judge read."""
    assert configured_judge_config_id("cfg", "Return JSON.") != configured_judge_config_id("cfg", "Return JSON. ")
    assert configured_judge_config_id("cfg", "Return JSON.") != configured_judge_config_id("cfg-2", "Return JSON.")
    assert judge_config_of(configured_judge_config_id("cfg", "Return JSON.")) == "cfg"
    # A bare config id names a config, not the prompt that was sent, so it is no judge's recorded identity; a
    # built-in prompt or nothing names no config.
    with pytest.raises(ValueError, match="not a recorded judge identity"):
        judge_config_of("cfg")
    assert judge_config_of(builtin_judge_config_id("Score it.")) is None
    assert judge_config_of(None) is None


#: A configured judge's reply-format fingerprints as this build words the suffix. **When this goes red the suffix
#: changed**, and with it every configured judge: profiles and agreement measured under the old suffix stop applying,
#: though no config changed. Re-pin only when it is meant, and say so in the release's notes.
_PINNED_FORMATS: dict[tuple[JudgedArtifact, str], str] = {
    (JudgedArtifact.TRANSCRIPT, TRANSCRIPT_DIM_ID): "cfg@48f825101269",
    (JudgedArtifact.TRANSCRIPT, OUTCOME_DIM_ID): "cfg@48f825101269",
    (JudgedArtifact.TRANSCRIPT, _TONE.name): "cfg@48f825101269",
    (JudgedArtifact.TRANSCRIPT, _PASS.name): "cfg@77f5d3e70b47",
    (JudgedArtifact.DOCUMENT, _TONE.name): "cfg@c31b0cac0f26",
    (JudgedArtifact.DOCUMENT, _PASS.name): "cfg@469f6f5d4bdb",
}


@pytest.mark.parametrize("judged_artifact", [JudgedArtifact.TRANSCRIPT, JudgedArtifact.DOCUMENT])
def test_the_reply_format_s_fingerprints_are_pinned_so_a_suffix_change_is_seen(judged_artifact: JudgedArtifact) -> None:
    every_dim = {TRANSCRIPT_DIM_ID: "cfg", OUTCOME_DIM_ID: "cfg", _TONE.name: "cfg", _PASS.name: "cfg"}
    ids = judge_config_ids_for([_TONE, _PASS], every_dim, judged_artifact=judged_artifact)
    pinned = {dim: config_id for (artifact, dim), config_id in _PINNED_FORMATS.items() if artifact is judged_artifact}
    assert ids == pinned, "the reply-format suffix changed: every configured judge is a new judge (see _PINNED_FORMATS)"


def test_a_judge_config_may_not_take_an_id_spelled_as_a_configured_judge_s_identity() -> None:
    with pytest.raises(ValidationError, match="a configured judge's recorded"):
        JudgeConfig(
            id="tone@48f825101269",
            scope_id="uni-1",
            name="impostor",
            rubric_dim_id=TRANSCRIPT_DIM_ID,
            prompt_template="Score it.",
        )


def test_a_judge_config_may_not_take_an_id_spelled_as_a_built_in_prompt_s() -> None:
    with pytest.raises(ValidationError, match="names the judge's built-in prompts"):
        JudgeConfig(
            id="builtin:f886726e63d2",
            scope_id="uni-1",
            name="impostor",
            rubric_dim_id=TRANSCRIPT_DIM_ID,
            prompt_template="Score it.",
        )


# --- pooling -------------------------------------------------------------------------------------


async def _scored_result(result_id: str, run_id: str, artifact: str) -> Any:
    """A result judged on tone by the built-in prompt, carrying what the judge phase records."""
    outcome = await _service(_Client()).score_dimension(_TONE, _context(artifact=artifact))
    assert outcome.score is not None
    return make_eval_result(
        id=result_id,
        eval_run_id=run_id,
        rubric_scores=[outcome.score],
        judge_config_ids=fold_judge_outcomes([(_TONE.name, outcome)]).config_ids,
    )


def _ratings(*result_ids: str) -> list[Any]:
    return [make_calibration_rating(result_id=result_id, rubric_dim=_TONE.name, score=4) for result_id in result_ids]


async def test_two_runs_judged_by_the_same_built_in_wording_are_one_judge() -> None:
    first = await _scored_result("r-1", "run-1", "Candidate: a table at nine")
    second = await _scored_result("r-2", "run-2", "Candidate: no tables left")

    (dimension,) = judge_agreement(_ratings("r-1", "r-2"), [first, second]).dimensions

    assert judge_key(first, _TONE.name) == judge_key(second, _TONE.name)
    assert dimension.n == 2
    assert is_builtin_judge_config_id(dimension.judge_config_id)


async def test_a_result_stored_before_built_in_prompts_were_identified_is_another_judge() -> None:
    """A pre-evaladopt.19 result recorded nothing for a built-in dim: an older wording, never the current one."""
    current = await _scored_result("r-1", "run-1", "Candidate: a table at nine")
    stored_before = (await _scored_result("r-old", "run-old", "Candidate: tomorrow, then")).model_copy(
        update={"judge_config_ids": {}}
    )

    agreement = judge_agreement(_ratings("r-1", "r-old"), [current, stored_before])

    assert judge_key(stored_before, _TONE.name) != judge_key(current, _TONE.name)
    assert sorted((d.judge_config_id or "", d.n) for d in agreement.dimensions) == [
        ("", 1),
        (current.judge_config_ids[_TONE.name], 1),
    ]


async def test_a_result_judged_under_another_built_in_wording_is_another_judge() -> None:
    current = await _scored_result("r-1", "run-1", "Candidate: a table at nine")
    reworded = (await _scored_result("r-2", "run-2", "Candidate: tomorrow, then")).model_copy(
        update={"judge_config_ids": {_TONE.name: builtin_judge_config_id("an older wording")}}
    )

    agreement = judge_agreement(_ratings("r-1", "r-2"), [current, reworded])

    assert len(agreement.dimensions) == 2


# --- where the judge's recall stands (Discodon #1895) ----------------------------------------------


@pytest.mark.parametrize("judged_artifact", [JudgedArtifact.TRANSCRIPT, JudgedArtifact.DOCUMENT])
async def test_every_built_in_axis_is_told_sourced_facts_outrank_the_judge_s_recall(
    judged_artifact: JudgedArtifact,
) -> None:
    client = _Client()
    service = _service(client)
    context = _context(judged_artifact)

    if judged_artifact is JudgedArtifact.TRANSCRIPT:
        await service.score_transcript(context)
        await service.score_outcome(context)
    await service.score_dimension(_TONE, context)
    await service.score_dimension(_PASS, context)

    assert len(client.systems) == (4 if judged_artifact is JudgedArtifact.TRANSCRIPT else 2)
    for system in client.systems:
        for words in _RULE_WORDS:
            assert words in system


async def test_an_operator_s_configured_prompt_is_sent_without_the_built_in_rule() -> None:
    """The rule lives in the built-in instructions: the suffix a configured prompt gets is unchanged by it."""
    client = _Client()
    config = JudgeConfig(scope_id="uni-1", name="o", rubric_dim_id=OUTCOME_DIM_ID, prompt_template="OPERATOR prompt")

    await _service(client, {OUTCOME_DIM_ID: config}).score_outcome(_context())

    (system,) = client.systems
    assert system.startswith("OPERATOR prompt")
    assert not any(words in system for words in _RULE_WORDS)
