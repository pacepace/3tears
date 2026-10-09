"""Unit tests for the single-dim judge service.

Exercises the stateless judge contract directly, apart from the runner: single-dim scoring, dual-score axis separation
(the transcript judge never sees the goal-state outcomes; the outcome judge
does), versioned-config honoring (prompt_template + model + temperature), and
failure handling.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from threetears.evals.contracts.models import (
    DEFAULT_JUDGE_TEMPERATURE,
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    GoalStateOutcome,
    JudgeConfig,
    JudgedArtifact,
    JudgeEvidence,
    RubricDim,
)
from threetears.evals.contracts.provider import INCOMPLETE_STOP_REASONS, withhold_failure_detail
from threetears.evals.run.judge import CANNOT_TELL, run_judge_llm
from threetears.evals.run.judge_service import JudgeContext, JudgeOutcome, JudgeService, fold_judge_outcomes
from packages.evals.tests.llm_client_fakes import ReleasableClientMixin

# =============================================================================
# Fakes
# =============================================================================


def _dim_of(system: str) -> str:
    """Extract the dim id the judge prompt asked to be scored."""
    match = re.search(r'the single key "(.+?)"', system)
    return match.group(1) if match else "?"


class _CapturingClient:
    """Records the prompts it received and returns a fixed score for the asked dim."""

    def __init__(self, *, score: int = 4, reasoning: str = "ok", cost: float = 0.002, fail: bool = False):
        self.score = score
        self.reasoning = reasoning
        self.cost = cost
        self.fail = fail
        self.systems: list[str] = []
        self.users: list[str] = []
        self.response_formats: list[Any] = []

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        self.systems.append(system)
        self.users.append(user)
        self.response_formats.append(response_format)
        if self.fail:
            content = "not json at all"
        else:
            content = json.dumps({"reasoning": self.reasoning, "criteria_scores": {_dim_of(system): self.score}})
        return SimpleNamespace(
            served_model=None,
            stop_reason="end_turn",
            content=content,
            input_tokens=10,
            output_tokens=5,
            cost_usd=self.cost,
            model="judge-fake",
        )


class _PromptSensitiveClient:
    """Scores low when the instructions say STRICT, high otherwise.

    Lets a test prove that two different ``prompt_template`` values produce two
    different scores — the JudgeConfig-versioning property.
    """

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        score = 2 if "STRICT" in system else 5
        content = json.dumps({"reasoning": "r", "criteria_scores": {_dim_of(system): score}})
        return SimpleNamespace(
            served_model=None,
            stop_reason="end_turn",
            content=content,
            input_tokens=1,
            output_tokens=1,
            cost_usd=0.0,
            model="x",
        )


def _recording_factory(client: Any):
    """Return ``(factory, calls)`` — factory hands back ``client`` and logs its args."""
    calls: list[tuple[str | None, float | None]] = []

    def factory(model: str | None, temperature: float | None) -> Any:
        calls.append((model, temperature))
        return client

    return factory, calls


def _context(
    *,
    goal_passed: bool = True,
    judged_artifact: JudgedArtifact = JudgedArtifact.TRANSCRIPT,
    evidence: JudgeEvidence | None = None,
) -> JudgeContext:
    return JudgeContext(
        case_id="tc-1",
        intent="book me a table",
        variation={"cuisine": "thai"},
        goal_outcomes=[
            GoalStateOutcome(expression="call_count('bookings.reserve') >= 1", passed=goal_passed, detail="reserved 1")
        ],
        judged_artifact=judged_artifact,
        judge_evidence=evidence
        or JudgeEvidence(
            subject="Mara, a terse concierge.",
            case_material="The restaurant has one table left, at 21:00.",
            artifact="Actor (asker): hi\nCandidate: on it",
        ),
    )


# =============================================================================
# Single-dim scoring
# =============================================================================


async def test_score_dimension_returns_single_rubric_score():
    client = _CapturingClient(score=4, reasoning="decent tone")
    factory, _calls = _recording_factory(client)
    service = JudgeService(client_factory=factory, failure_describer=withhold_failure_detail)

    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="sounds like the subject", scale="ordinal"), _context()
    )

    assert outcome.error is None
    assert outcome.score is not None
    assert outcome.score.dim == "conversation.tone"
    assert outcome.score.score == 4
    assert outcome.score.reasoning == "decent tone"
    assert outcome.config_id is None  # no operator config → built-in default
    assert outcome.usage is not None and outcome.usage.cost_usd == 0.002
    # Exactly one LLM call — single-dim discipline.
    assert len(client.systems) == 1
    # The judge requests JSON mode so scores parse deterministically.
    assert client.response_formats == [{"type": "json_object"}]


@pytest.mark.parametrize("off_scale", [9, 0, -1, 3.5, "4", None, True])
async def test_an_off_scale_score_is_refused_not_clamped(off_scale):
    """A score that is not an integer on 1-5 fails the dim through the parse-failure path.

    Clamping 9 to 5, or rounding 3.5 up to 4, would store a score the judge never gave.
    """
    client = _CapturingClient(score=off_scale)
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.d", description="x", scale="ordinal"), _context()
    )
    assert outcome.score is None
    assert outcome.error is not None and "Failed to parse" in outcome.error
    # The bounded retry was spent on it, and the spend is still reported.
    assert len(client.systems) == 2
    assert outcome.usage is not None and outcome.usage.calls == 2


@pytest.mark.parametrize("on_scale", [1, 3, 5, 4.0])
async def test_an_on_scale_score_is_stored_as_given(on_scale):
    """Every integer on the scale lands unchanged — the positive case beside the refusal."""
    service = JudgeService(
        client_factory=lambda m, t: _CapturingClient(score=on_scale), failure_describer=withhold_failure_detail
    )
    outcome = await service.score_dimension(
        RubricDim(name="conversation.d", description="x", scale="ordinal"), _context()
    )
    assert outcome.error is None
    assert outcome.score is not None
    assert outcome.score.score == int(on_scale)


# =============================================================================
# "Can't tell" — a structured answer, neither a score nor a failure
# =============================================================================


@pytest.mark.parametrize(("answer", "scored"), [(CANNOT_TELL, False), (4, True)])
async def test_a_cannot_tell_answer_leaves_the_dim_unscored_and_a_score_scores(answer, scored):
    """One fixture, both answers: the judge's reply decides, not the harness."""
    client = _CapturingClient(score=answer, reasoning="the transcript never reaches the moment this dim is about")
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)

    outcome = await service.score_dimension(
        RubricDim(name="conversation.d", description="x", scale="ordinal"), _context()
    )

    assert outcome.error is None
    if scored:
        assert outcome.score is not None and outcome.score.score == 4
        assert outcome.cannot_tell is None
    else:
        assert outcome.score is None
        assert outcome.cannot_tell == "the transcript never reaches the moment this dim is about"
    # Answered first time: a "can't tell" is not a parse failure, so it buys no retry.
    assert len(client.systems) == 1


async def test_the_judge_prompt_offers_the_answer():
    client = _CapturingClient()
    await JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail).score_dimension(
        RubricDim(name="conversation.d", description="x", scale="ordinal"), _context()
    )
    assert f'"{CANNOT_TELL}"' in client.systems[0]


async def test_a_caller_that_did_not_offer_the_answer_refuses_it():
    """A composite scorer and a quality scorer never offer it, so for them it is a reply that broke the protocol."""
    client = _CapturingClient(score=CANNOT_TELL)
    result = await run_judge_llm(
        client=client,
        system_prompt="s",
        user_prompt="u",
        criteria_dicts=[{"name": "?", "weight": 1.0}],
        label="J",
        case_id="c",
    )
    assert result is not None and "Failed to parse" in result["error"]


def test_a_cannot_tell_is_folded_apart_from_the_errors():
    folded = fold_judge_outcomes(
        [
            ("told", JudgeOutcome(score=None, cannot_tell="no evidence")),
            ("broke", JudgeOutcome(score=None, error="Failed to parse")),
        ]
    )
    assert folded.cannot_tell == {"told": "no evidence"}
    assert folded.errors == [("broke", "Failed to parse")]


# =============================================================================
# Dual-score axis separation
# =============================================================================


async def test_transcript_axis_excludes_goal_outcomes_outcome_includes_them():
    """The transcript judge must not see the objective goal-state checks; the outcome judge must."""
    client = _CapturingClient()
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    ctx = _context()

    transcript = await service.score_transcript(ctx)
    outcome = await service.score_outcome(ctx)

    assert transcript.score is not None and transcript.score.dim == TRANSCRIPT_DIM_ID
    assert outcome.score is not None and outcome.score.dim == OUTCOME_DIM_ID

    transcript_user = client.users[0]
    outcome_user = client.users[1]
    # Axis separation: only the outcome judge sees the goal-state outcomes.
    assert "goal-state outcomes" not in transcript_user.lower()
    assert "call_count" not in transcript_user
    assert "goal-state outcomes" in outcome_user.lower()
    assert "call_count" in outcome_user
    # Both see the transcript itself.
    assert "Transcript" in transcript_user and "Transcript" in outcome_user


# =============================================================================
# The kind's evidence is placed in the prompt whole, never read
# =============================================================================


async def test_the_kinds_subject_reaches_the_prompt_whole_under_its_own_heading():
    """What the judge is told about the candidate is the kind's rendering, untrimmed and unparsed.

    The engine used to render one host's subject record itself — excerpting its prose, capping its
    lists. Every one of those choices was the host's to make, so the kind now hands over the text
    and the engine's only job is to place it.
    """
    client = _CapturingClient()
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    subject = "Mara, a terse concierge.\n\n  Never apologises twice.\n" + "x" * 9000
    ctx = _context(evidence=JudgeEvidence(subject=subject, case_material="one table left", artifact="hi"))

    await service.score_dimension(RubricDim(name="conversation.tone", description="d", scale="ordinal"), ctx)

    (user,) = client.users
    assert f"# Candidate under test\n{subject}\n" in user


async def test_a_kind_that_renders_no_subject_gets_no_subject_heading():
    client = _CapturingClient()
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    ctx = _context(evidence=JudgeEvidence(case_material="one table left", artifact="hi"))

    await service.score_dimension(RubricDim(name="conversation.tone", description="d", scale="ordinal"), ctx)

    (user,) = client.users
    assert "# Candidate under test" not in user


async def test_facts_the_candidates_interlocutors_never_saw_reach_every_axis():
    """A game master's hidden trap is case material: the judge needs it, the players never had it.

    Which facts a judge sees is the kind's visibility rule, so the case material reaches every
    axis — the transcript axis included, which is scored on what the candidate knew.
    """
    client = _CapturingClient()
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    hidden = "GM ONLY: the third flagstone is a pressure-plate trap (DC 15 to spot)."
    transcript = "Player (Ivo): I step forward.\nPlayer (Ivo, whispered to GM): I check the floor first.\nGM: Roll it."
    ctx = _context(evidence=JudgeEvidence(subject="The GM.", case_material=hidden, artifact=transcript))

    await service.score_transcript(ctx)
    await service.score_outcome(ctx)
    await service.score_dimension(RubricDim(name="conversation.fairness", description="d", scale="ordinal"), ctx)

    assert len(client.users) == 3
    for user in client.users:
        assert hidden in user
        # The artifact is the prompt's last section, placed exactly as the kind rendered it.
        assert user.endswith(f"# Transcript\n{transcript}")


async def test_a_document_is_placed_under_its_own_heading_and_a_transcript_under_its_own():
    """The declaration picks the heading; the evidence is the same three strings either way."""
    evidence = JudgeEvidence(case_material="SOURCE", artifact="ARTIFACT")
    for declared, heading, absent in (
        (JudgedArtifact.TRANSCRIPT, "# Transcript\nARTIFACT", "# Output under review"),
        (JudgedArtifact.DOCUMENT, "# Output under review\nARTIFACT", "# Transcript"),
    ):
        client = _CapturingClient()
        service = JudgeService(client_factory=lambda m, t, c=client: c, failure_describer=withhold_failure_detail)
        await service.score_dimension(
            RubricDim(name="conversation.faithful", description="d", scale="ordinal"),
            _context(judged_artifact=declared, evidence=evidence),
        )
        (user,) = client.users
        assert user.endswith(heading), declared
        assert absent not in user, declared
        assert "# Case material (the evidence the output must be judged against)\nSOURCE\n" in user, declared


def test_no_judge_context_is_built_for_an_unjudged_kind():
    with pytest.raises(ValueError, match="unjudged"):
        _context(judged_artifact=JudgedArtifact.UNJUDGED)


@pytest.mark.parametrize(
    "fields",
    [
        {"case_material": "", "artifact": "a"},
        {"case_material": "c", "artifact": ""},
        {"case_material": "c"},
        {"artifact": "a"},
        {"case_material": "c", "artifact": "a", "transcript": []},
    ],
    ids=["empty-case-material", "empty-artifact", "no-artifact", "no-case-material", "an-undeclared-field"],
)
def test_judge_evidence_refuses_a_judge_nothing_to_read(fields):
    """Both texts a judge reads are required and non-empty; nothing else rides along."""
    with pytest.raises(ValueError):
        JudgeEvidence(**fields)


async def test_default_axes_use_builtin_instructions_and_default_client():
    client = _CapturingClient()
    factory, calls = _recording_factory(client)
    service = JudgeService(client_factory=factory, failure_describer=withhold_failure_detail)

    await service.score_transcript(_context())
    await service.score_outcome(_context())

    assert "DECISION QUALITY" in client.systems[0]  # transcript built-in
    assert "ACHIEVED THE USER'S INTENT" in client.systems[1]  # outcome built-in
    # Both axes default config → same (None, DEFAULT_JUDGE_TEMPERATURE) client key → factory called once. Never
    # the provider's default: a dim with no config is sampled at the temperature a config defaults to (#633).
    assert calls == [(None, DEFAULT_JUDGE_TEMPERATURE)]
    assert DEFAULT_JUDGE_TEMPERATURE == JudgeConfig.model_fields["temperature"].default == 0.0


# =============================================================================
# Versioned JudgeConfig
# =============================================================================


async def test_config_prompt_template_and_model_and_temperature_honored():
    client = _CapturingClient()
    factory, calls = _recording_factory(client)
    config = JudgeConfig(
        scope_id="uni-1",
        name="tone-v2",
        rubric_dim_id="conversation.tone",
        prompt_template="CUSTOM tone instructions",
        model="cfg-model",
        temperature=0.3,
    )
    service = JudgeService(
        client_factory=factory, configs={"conversation.tone": config}, failure_describer=withhold_failure_detail
    )

    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.config_id == config.id
    assert client.systems[0].startswith("CUSTOM tone instructions")
    # The config's model + temperature select the client.
    assert calls == [("cfg-model", 0.3)]


async def test_two_configs_for_same_dim_produce_different_scores():
    """JudgeConfig versioning: different prompt_template ⇒ different score."""
    strict = JudgeConfig(
        scope_id="uni-1",
        name="strict",
        rubric_dim_id="conversation.tone",
        prompt_template="STRICT grading. Penalize hard.",
    )
    lenient = JudgeConfig(
        scope_id="uni-1", name="lenient", rubric_dim_id="conversation.tone", prompt_template="Be generous."
    )
    dim = RubricDim(name="conversation.tone", description="d", scale="ordinal")

    strict_out = await JudgeService(
        client_factory=lambda m, t: _PromptSensitiveClient(),
        configs={"conversation.tone": strict},
        failure_describer=withhold_failure_detail,
    ).score_dimension(dim, _context())
    lenient_out = await JudgeService(
        client_factory=lambda m, t: _PromptSensitiveClient(),
        configs={"conversation.tone": lenient},
        failure_describer=withhold_failure_detail,
    ).score_dimension(dim, _context())

    assert strict_out.score is not None and lenient_out.score is not None
    assert strict_out.score.score == 2
    assert lenient_out.score.score == 5
    assert strict_out.score.score != lenient_out.score.score


async def test_axis_config_overrides_only_the_matching_axis():
    """A transcript-axis config doesn't bleed into the outcome axis."""
    client = _CapturingClient()
    config = JudgeConfig(
        scope_id="uni-1", name="t", rubric_dim_id=TRANSCRIPT_DIM_ID, prompt_template="TRANSCRIPT OVERRIDE"
    )
    service = JudgeService(
        client_factory=lambda m, t: client,
        configs={TRANSCRIPT_DIM_ID: config},
        failure_describer=withhold_failure_detail,
    )

    transcript = await service.score_transcript(_context())
    outcome = await service.score_outcome(_context())

    assert transcript.config_id == config.id
    assert client.systems[0].startswith("TRANSCRIPT OVERRIDE")
    assert outcome.config_id is None
    assert "ACHIEVED THE USER'S INTENT" in client.systems[1]  # outcome still built-in


# =============================================================================
# Failure handling
# =============================================================================


async def test_judge_failure_returns_error_and_no_score():
    service = JudgeService(
        client_factory=lambda m, t: _CapturingClient(fail=True), failure_describer=withhold_failure_detail
    )
    outcome = await service.score_outcome(_context())
    assert outcome.score is None
    assert outcome.error is not None


@pytest.mark.parametrize("reasoning", [None, {"why": "it hedges"}, ["it", "hedges"], 3])
@pytest.mark.parametrize("answer", [4, CANNOT_TELL], ids=["scored", "cannot-tell"])
async def test_a_reply_whose_reasoning_is_not_a_string_is_a_parse_failure_that_keeps_its_spend(reasoning, answer):
    """A malformed ``reasoning`` is the judge breaking its protocol, refused where every other break is.

    Refused later it did worse: a scored reply raised building the score, after the service had
    the spend, so the dim was recorded as an error with its usage dropped; a can't-tell carried a
    non-string reason onto the result, which raised outside every handler and failed the run.
    """
    client = _CapturingClient(score=answer, reasoning=reasoning, cost=0.002)
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)

    outcome = await service.score_dimension(
        RubricDim(name="conversation.d", description="x", scale="ordinal"), _context()
    )

    assert outcome.score is None and outcome.cannot_tell is None
    assert outcome.error is not None and "Failed to parse" in outcome.error
    # Both attempts were paid for, and both are on the outcome.
    assert len(client.systems) == 2
    assert outcome.usage is not None and outcome.usage.calls == 2
    assert outcome.usage.cost_usd == pytest.approx(0.004)


async def test_a_reply_with_no_reasoning_at_all_is_a_parse_failure():
    class _NoReasoningClient:
        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            return SimpleNamespace(
                served_model=None,
                stop_reason="end_turn",
                content=json.dumps({"criteria_scores": {_dim_of(system): 4}}),
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.001,
                model="x",
            )

    service = JudgeService(client_factory=lambda m, t: _NoReasoningClient(), failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.d", description="x", scale="ordinal"), _context()
    )
    assert outcome.score is None
    assert outcome.error is not None and "Failed to parse" in outcome.error
    assert outcome.usage is not None and outcome.usage.calls == 2


async def test_score_omitted_by_judge_is_an_error():
    class _WrongKeyClient:
        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            # Returns a score under the wrong key — the asked dim is missing.
            content = json.dumps({"reasoning": "r", "criteria_scores": {"some_other_dim": 4}})
            return SimpleNamespace(
                served_model=None,
                stop_reason="end_turn",
                content=content,
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                model="x",
            )

    service = JudgeService(client_factory=lambda m, t: _WrongKeyClient(), failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )
    assert outcome.score is None
    assert outcome.error is not None


async def test_score_dimension_parses_real_code_fenced_judge_output():
    """A real judge model wraps its JSON in a markdown code fence.

    Every other parse test here feeds the fake client a bare ``json.dumps(...)``,
    so no test exercised the ACTUAL on-the-wire shape a real judge emits. This
    locks that shape (captured from a live judge call): a ```json fenced object, a bare ``` fenced object, and a
    prose-then-fence response each resolve end-to-end to the right score rather
    than silently dropping the dim. (It asserts the observable contract, not any
    single ``extract_json`` strategy — the fenced shapes here also parse via the
    ``raw_decode``-from-first-brace fallback, so this does not isolate the
    fence-strip path specifically.)
    """
    inner = '{\n  "reasoning": "Calibrated — the candidate flagged the thin item.",\n  "criteria_scores": {\n    "conversation.honesty": 5\n  }\n}'
    real_shapes = [
        f"```json\n{inner}\n```",  # the exact shape a live judge emits
        f"```\n{inner}\n```",  # bare fence, no language tag
        f"Here is my assessment:\n\n```json\n{inner}\n```\n",  # prose then fence
    ]
    for content in real_shapes:

        class _FencedClient:
            async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
                # Waived at the site: the class is constructed and fully consumed inside
                # this same iteration, so the late binding B023 warns about is never observed.
                return SimpleNamespace(
                    served_model=None,
                    stop_reason="end_turn",
                    content=content,  # noqa: B023
                    input_tokens=8,
                    output_tokens=4,
                    cost_usd=0.001,
                    model="judge-fake",
                )

        service = JudgeService(client_factory=lambda m, t: _FencedClient(), failure_describer=withhold_failure_detail)
        outcome = await service.score_dimension(
            RubricDim(name="conversation.honesty", description="d", scale="ordinal"), _context()
        )

        assert outcome.error is None, f"fenced shape should parse, got error for {content!r}"
        assert outcome.score is not None
        assert outcome.score.score == 5
        assert "Calibrated" in outcome.score.reasoning


async def test_judge_retries_on_parse_failure_then_succeeds():
    """A single unparseable judge response is retried, not silently dropped."""

    class _FailOnceClient:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            self.calls += 1
            if self.calls == 1:
                content = "not json at all"  # first attempt unparseable
            else:
                content = json.dumps({"reasoning": "ok", "criteria_scores": {_dim_of(system): 4}})
            return SimpleNamespace(
                served_model=None,
                stop_reason="end_turn",
                content=content,
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                model="x",
            )

    client = _FailOnceClient()
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.score is not None  # recovered — the dim was NOT dropped
    assert outcome.score.score == 4
    assert client.calls == 2  # one retry


async def test_judge_parse_failure_gives_up_after_bounded_retries():
    """A persistently-unparseable judge fails after the bounded retries (no infinite loop)."""

    class _AlwaysFailClient:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            self.calls += 1
            return SimpleNamespace(
                served_model=None,
                stop_reason="end_turn",
                content="not json",
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                model="x",
            )

    client = _AlwaysFailClient()
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.score is None
    assert outcome.error is not None
    assert client.calls == 2  # original + _JUDGE_PARSE_RETRIES(1) = 2 attempts, then gives up


async def test_judge_llm_exception_is_hard_error_not_retried():
    """A raising generate() (network/SDK) is a hard error — returned immediately, not retried."""

    class _RaisingClient:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            self.calls += 1
            raise RuntimeError("network down")

    client = _RaisingClient()
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.score is None
    assert outcome.error is not None
    assert client.calls == 1  # hard error — not retried (only parse-failures retry)


# =============================================================================
# A reply the provider cut short — refused after one call, never re-bought
# =============================================================================


class _StopReasonClient:
    """Answers every call with one fixed reply, stamped with the given ``stop_reason``.

    The token figures are the shape a production run recorded: a reasoning judge spent its
    whole output cap, nearly all of it reasoning.
    """

    def __init__(
        self, *, stop_reason: str, content: str = "not json", output_tokens: int = 4096, reasoning: int | None = 3900
    ):
        self.stop_reason = stop_reason
        self.content = content
        self.output_tokens = output_tokens
        self.reasoning = reasoning
        self.calls = 0

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        self.calls += 1
        return SimpleNamespace(
            served_model=None,
            stop_reason=self.stop_reason,
            content=self.content,
            input_tokens=60000,
            output_tokens=self.output_tokens,
            reasoning_tokens=self.reasoning,
            cost_usd=0.25,
            model="judge-fake",
        )


@pytest.mark.parametrize(
    ("stop_reason", "calls", "names"),
    [
        # The same unparseable content, both directions on one fixture: only the
        # provider's own verdict separates a re-askable reply from a refused request.
        ("end_turn", 2, "Failed to parse"),
        ("max_tokens", 1, "cut short by the provider"),
    ],
)
async def test_a_finished_reply_is_retried_and_a_truncated_one_is_not(stop_reason, calls, names):
    client = _StopReasonClient(stop_reason=stop_reason)
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.score is None
    assert client.calls == calls
    assert names in outcome.error
    # The spend is reported whichever way it ended, and counts every call made.
    assert outcome.usage is not None
    assert outcome.usage.calls == calls
    assert outcome.usage.output_tokens == 4096 * calls


async def test_a_truncated_reply_names_the_cap_and_the_reasoning_share_not_a_parse_failure():
    """The error names the cause and the tokens behind it, where it used to say "Failed to parse"."""
    client = _StopReasonClient(stop_reason="max_tokens", content="", output_tokens=4096, reasoning=3900)
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert "Failed to parse" not in outcome.error
    assert "TRUNCATED at the output cap" in outcome.error
    assert "4096 output token(s), of which 3900 were reasoning" in outcome.error
    assert "not retried" in outcome.error
    assert outcome.usage.reasoning_tokens == 3900


async def test_a_truncated_reply_is_refused_even_when_its_content_parses():
    """Checked before parsing: salvage must not turn a reply the provider cut short into a score.

    ``extract_json`` recovers partial payloads, so without the ordering this reply would
    be scored as if it had finished.
    """
    content = json.dumps({"reasoning": "cut", "criteria_scores": {"conversation.tone": 4}})
    client = _StopReasonClient(stop_reason="max_tokens", content=content)
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.score is None
    assert client.calls == 1
    assert "cut short by the provider" in outcome.error


@pytest.mark.parametrize("stop_reason", sorted(INCOMPLETE_STOP_REASONS))
async def test_every_cut_short_reason_the_port_names_is_refused_after_one_call(stop_reason):
    """The refusal reads the port's vocabulary, not one member of it.

    A content filter refuses the same request the same way, and a reply with no choices
    is the filter's other shape. Re-asking either buys the identical refusal.
    """
    client = _StopReasonClient(stop_reason=stop_reason)
    result = await run_judge_llm(
        client=client,
        system_prompt="s",
        user_prompt="u",
        criteria_dicts=[{"name": "conversation.tone", "weight": 1.0}],
        label="Judge",
        case_id="tc-1",
    )

    assert client.calls == 1
    assert "cut short by the provider" in result["error"]
    assert result["judge_usage"].calls == 1


# =============================================================================
# Judge usage capture — the judge RoleUsage row's source
# =============================================================================


async def test_scored_dim_carries_the_calls_usage():
    client = _CapturingClient(cost=0.002)
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.usage is not None
    assert outcome.usage.model == "judge-fake"
    assert outcome.usage.input_tokens == 10
    assert outcome.usage.output_tokens == 5
    assert outcome.usage.cost_usd == 0.002


async def test_judge_usage_accumulates_across_parse_retries():
    """A retried dim spent tokens twice — the usage row must show both attempts."""

    class _FailOnceClient:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            self.calls += 1
            content = (
                "not json"
                if self.calls == 1
                else json.dumps({"reasoning": "ok", "criteria_scores": {_dim_of(system): 4}})
            )
            return SimpleNamespace(
                served_model=None,
                stop_reason="end_turn",
                content=content,
                input_tokens=7,
                output_tokens=3,
                cost_usd=0.001,
                model="x",
            )

    service = JudgeService(client_factory=lambda m, t: _FailOnceClient(), failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.usage is not None
    assert outcome.usage.input_tokens == 14
    assert outcome.usage.output_tokens == 6
    assert outcome.usage.cost_usd == 0.002


async def test_unscorable_dim_still_reports_the_tokens_it_burned():
    """Giving up after retries doesn't refund the spend — it must still be accounted."""

    class _AlwaysFailClient:
        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            return SimpleNamespace(
                served_model=None,
                stop_reason="end_turn",
                content="not json",
                input_tokens=7,
                output_tokens=3,
                cost_usd=0.001,
                model="x",
            )

    service = JudgeService(client_factory=lambda m, t: _AlwaysFailClient(), failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.score is None
    assert outcome.usage is not None
    assert outcome.usage.input_tokens == 14  # both attempts
    assert outcome.usage.cost_usd == 0.002


async def test_dim_omitted_by_the_judge_still_reports_usage():
    """An omitted criterion key is a parse failure, so it retries — and BOTH
    attempts' spend must be accounted, not just the last."""

    class _WrongKeyClient:
        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            content = json.dumps({"reasoning": "r", "criteria_scores": {"some-other-dim": 3}})
            return SimpleNamespace(
                served_model=None,
                stop_reason="end_turn",
                content=content,
                input_tokens=9,
                output_tokens=4,
                cost_usd=0.003,
                model="x",
            )

    service = JudgeService(client_factory=lambda m, t: _WrongKeyClient(), failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.score is None
    assert outcome.usage is not None
    assert outcome.usage.input_tokens == 18
    assert outcome.usage.cost_usd == pytest.approx(0.006)


async def test_judge_reasoning_tokens_stay_none_when_unreported():
    """The judge client reporting no reasoning split is UNKNOWN, never a zero."""
    service = JudgeService(client_factory=lambda m, t: _CapturingClient(), failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.usage is not None
    assert outcome.usage.reasoning_tokens is None


async def test_judge_reasoning_tokens_sum_when_reported():
    class _ReasoningClient:
        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            content = json.dumps({"reasoning": "r", "criteria_scores": {_dim_of(system): 4}})
            return SimpleNamespace(
                served_model=None,
                stop_reason="end_turn",
                content=content,
                input_tokens=1,
                output_tokens=50,
                reasoning_tokens=30,
                cost_usd=0.0,
                model="x",
            )

    service = JudgeService(client_factory=lambda m, t: _ReasoningClient(), failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.usage is not None
    assert outcome.usage.reasoning_tokens == 30


async def test_judge_cost_unreported_stays_none_not_zero():
    """A judge client that reports no cost must not be recorded as having cost nothing."""

    class _NoCostClient:
        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            content = json.dumps({"reasoning": "r", "criteria_scores": {_dim_of(system): 4}})
            return SimpleNamespace(
                served_model=None,
                stop_reason="end_turn",
                content=content,
                input_tokens=1,
                output_tokens=1,
                cost_usd=None,
                model="x",
            )

    service = JudgeService(client_factory=lambda m, t: _NoCostClient(), failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.usage is not None
    assert outcome.usage.cost_usd is None


async def test_a_dim_one_of_whose_attempts_went_unpriced_costs_unknown_not_its_priced_part():
    """A priced retry beside an unpriced one is not a known cost: the sum would read as the whole."""

    class _PricedThenUnpricedClient:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            self.calls += 1
            content = (
                "not json"
                if self.calls == 1
                else json.dumps({"reasoning": "r", "criteria_scores": {_dim_of(system): 4}})
            )
            return SimpleNamespace(
                served_model=None,
                stop_reason="end_turn",
                content=content,
                input_tokens=7,
                output_tokens=3,
                cost_usd=0.001 if self.calls == 1 else None,
                model="x",
            )

    service = JudgeService(
        client_factory=lambda m, t: _PricedThenUnpricedClient(), failure_describer=withhold_failure_detail
    )
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.score is not None
    assert outcome.usage is not None and outcome.usage.calls == 2
    assert outcome.usage.cost_usd is None
    assert fold_judge_outcomes([("conversation.tone", outcome)]).cost_usd is None


async def test_judge_llm_exception_reports_prior_attempt_spend():
    """A raise mid-loop must not discard what earlier attempts already spent."""

    class _FailThenRaiseClient:
        def __init__(self) -> None:
            self.calls = 0

        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    served_model=None,
                    stop_reason="end_turn",
                    content="not json",
                    input_tokens=11,
                    output_tokens=2,
                    cost_usd=0.004,
                    model="x",
                )
            raise RuntimeError("network down")

    service = JudgeService(
        client_factory=lambda m, t: _FailThenRaiseClient(), failure_describer=withhold_failure_detail
    )
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.score is None
    assert outcome.usage is not None
    assert outcome.usage.input_tokens == 11
    assert outcome.usage.cost_usd == 0.004


async def test_judge_whose_first_call_raises_reports_no_usage_at_all():
    """Nothing was observed, so there is nothing to report — not a zero-token observation.

    The earlier shape returned prompt=0/completion=0/calls=0, which reads as "this judge
    ran and used nothing" rather than "this judge never got a response."
    """

    class _RaisingClient:
        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            raise RuntimeError("network down")

    service = JudgeService(client_factory=lambda m, t: _RaisingClient(), failure_describer=withhold_failure_detail)
    outcome = await service.score_dimension(
        RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context()
    )

    assert outcome.score is None
    assert outcome.error is not None
    assert outcome.usage is None


async def test_judge_result_carries_no_rival_composite():
    """The judge returns per-dimension scores only — each caller composites itself.

    A second, unread ``composite_criteria`` in the result invited a reader to
    mistake it for the canonical composite (``judge_service`` composites across
    its single-dim calls, a composite scorer via ``compute_composite``). It was deleted; this
    pins its absence so it cannot creep back.
    """
    from threetears.evals.run.judge import run_judge_llm

    class _Client:
        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            content = json.dumps({"reasoning": "ok", "criteria_scores": {"a": 4, "b": 2}})
            return SimpleNamespace(
                served_model=None,
                stop_reason="end_turn",
                content=content,
                input_tokens=1,
                output_tokens=1,
                cost_usd=0.0,
                model="x",
            )

    result = await run_judge_llm(
        client=_Client(),
        system_prompt="s",
        user_prompt="u",
        criteria_dicts=[{"name": "a", "weight": 0.7}, {"name": "b", "weight": 0.3}],
        label="Judge",
        case_id="c1",
    )

    assert set(result["criteria_scores"]) == {"a", "b"}
    assert "composite_criteria" not in result


# =============================================================================
# Client release
# =============================================================================


class _ReleasableJudgeClient(ReleasableClientMixin, _CapturingClient):
    """A judge client that scores AND counts its releases."""


async def test_aclose_releases_every_cached_client():
    """The cache is per ``(model, temperature)``; teardown must reach all of them.

    ``_client_for`` cannot release its own mint — the next dim asking for the same
    pair is about to reuse it — so the release is the service's, and it has to
    cover every entry rather than the last one built.
    """
    built: list[_ReleasableJudgeClient] = []

    def factory(model: str | None, temperature: float | None) -> Any:
        client = _ReleasableJudgeClient()
        built.append(client)
        return client

    configs = {
        "conversation.tone": JudgeConfig(
            scope_id="uni-1",
            name="tone-v1",
            rubric_dim_id="conversation.tone",
            prompt_template="score tone",
            model="m-a",
            temperature=0.0,
        ),
        "conversation.taste": JudgeConfig(
            scope_id="uni-1",
            name="taste-v1",
            rubric_dim_id="conversation.taste",
            prompt_template="score taste",
            model="m-b",
            temperature=0.2,
        ),
    }
    service = JudgeService(client_factory=factory, configs=configs, failure_describer=withhold_failure_detail)
    await service.score_dimension(RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context())
    await service.score_dimension(RubricDim(name="conversation.taste", description="d", scale="ordinal"), _context())
    assert len(built) == 2, "two distinct (model, temperature) pairs must not share a client"

    await service.aclose()

    assert [c.aclose_calls for c in built] == [1, 1]


async def test_aclose_is_idempotent():
    """A run teardown plus an explicit close must not double-close a transport."""
    client = _ReleasableJudgeClient()
    service = JudgeService(client_factory=lambda m, t: client, failure_describer=withhold_failure_detail)
    await service.score_dimension(RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context())

    await service.aclose()
    await service.aclose()

    assert client.aclose_calls == 1


async def test_one_refusing_client_does_not_strand_the_rest():
    """A teardown that gives up halfway leaks the remainder silently."""

    class _Refuses(_ReleasableJudgeClient):
        async def aclose(self) -> None:
            raise RuntimeError("transport refused to close")

    refuser, good = _Refuses(), _ReleasableJudgeClient()
    clients = iter([refuser, good])
    configs = {
        "conversation.tone": JudgeConfig(
            scope_id="uni-1",
            name="tone-v1",
            rubric_dim_id="conversation.tone",
            prompt_template="p",
            model="m-a",
            temperature=0.0,
        ),
        "conversation.taste": JudgeConfig(
            scope_id="uni-1",
            name="taste-v1",
            rubric_dim_id="conversation.taste",
            prompt_template="p",
            model="m-b",
            temperature=0.2,
        ),
    }
    service = JudgeService(
        client_factory=lambda m, t: next(clients), configs=configs, failure_describer=withhold_failure_detail
    )
    await service.score_dimension(RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context())
    await service.score_dimension(RubricDim(name="conversation.taste", description="d", scale="ordinal"), _context())

    with pytest.raises(RuntimeError, match="transport refused to close"):
        await service.aclose()

    assert good.aclose_calls == 1


async def test_the_scoped_form_never_replaces_the_bodys_exception():
    """``__aexit__`` logs a failed release; the run's own error is the one to report."""

    class _Refuses(_ReleasableJudgeClient):
        async def aclose(self) -> None:
            raise RuntimeError("transport refused to close")

    service = JudgeService(client_factory=lambda m, t: _Refuses(), failure_describer=withhold_failure_detail)
    await service.score_dimension(RubricDim(name="conversation.tone", description="d", scale="ordinal"), _context())

    with pytest.raises(ValueError, match="the run failed"):
        async with service:
            raise ValueError("the run failed")


async def test_a_judge_reporting_no_token_counts_reads_as_unmeasured():
    """Missing is not zero on the judge's usage either: an unreported count stays None."""

    class _Silent:
        async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
            content = json.dumps({"reasoning": "ok", "criteria_scores": {_dim_of(system): 4}})
            return SimpleNamespace(
                served_model=None, stop_reason="end_turn", content=content, cost_usd=None, model="judge-fake"
            )

    outcome = await JudgeService(
        client_factory=lambda m, t: _Silent(), failure_describer=withhold_failure_detail
    ).score_dimension(RubricDim(name="conversation.d", description="x", scale="ordinal"), _context())

    assert outcome.score is not None and outcome.score.score == 4
    assert outcome.usage is not None
    assert outcome.usage.input_tokens is None
    assert outcome.usage.output_tokens is None
    # The positive case is test_score_dimension_returns_single_rubric_score's reporting client.
