"""Host-neutral eval model factories for the package's own suite.

Every helper here constructs ``threetears.evals`` models and nothing else: no host adapter and no
host profile. A run whose subject, opaque payload and world placements are derived through a
host's own adapter belongs to that host's factories, not here.

Two consequences for a test that reads a run built here:

- ``subject_snapshot`` defaults to :data:`DEFAULT_SUBJECT`, an engine-shaped subject with no
  payload, and ``host_payload`` defaults to ``{}``. A test about a particular host's subject passes
  its own, as the toy host does.
- ``world_placements`` defaults to ``{}``, the recording that the run placed no dimensions (a host
  that declares no world). ``None`` means NOT RECORDED (a run its host assembled without the
  launch); pass it explicitly when that is the case under test.
- ``candidate_kind`` defaults to :data:`DEFAULT_KIND` and ``rubric_scales`` to every scale the
  run's dims need — ``{}`` unless a test names them — because a run states both. A host's launch derives placements through its
  launch context, and a test about that derivation builds the run through the launch.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.kernel.authored import AuthoredAnalysis
from threetears.evals.kernel.campaign import (
    EvalAnalysis,
    EvalAnalysisAttempt,
    EvalCampaign,
    EvalInsight,
    GenerationProvenance,
)
from threetears.evals.kernel.declaration import CampaignDesign, ControlDeclaration, SweptAxis
from threetears.evals.schema import SubjectSnapshot
from threetears.evals.kernel.host.profile import CANDIDATE_MODEL_LEVER
from threetears.evals.schema.values import SweepableValue
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.kernel.identity import IDENTITY_VERSION
from threetears.evals.schema.models import (
    DEFAULT_JUDGE_TEMPERATURE,
    OUTCOME_DIM_ID,
    TRANSCRIPT_DIM_ID,
    ActorPolicy,
    CalibrationRating,
    CatalogRubricDim,
    ClientRequestSettings,
    ConversationSpec,
    EvalResult,
    EvalRun,
    EvalTemplate,
    EvalTestCase,
    EvalTrace,
    GoalStateOutcome,
    JudgeConfig,
    RubricDim,
    RubricScore,
    WorldSeed,
    VariationAxis,
    eval_trace_doc_id,
)
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.schema.store_port import omit_paths
from threetears.evals.kernel.surface import DecisionSurface
from threetears.evals.kernel.usage_capture import blended_cost_roles
from threetears.evals.storage import InMemoryDocumentStore

__all__ = [
    "DEFAULT_KIND",
    "DEFAULT_SUBJECT",
    "RECORDED_REQUEST_SETTINGS",
    "as_listed",
    "fixture_variant_key",
    "load_listed",
    "make_analysis",
    "make_analysis_attempt",
    "make_campaign",
    "make_eval_result",
    "make_eval_run",
    "make_eval_trace",
    "make_insight",
    "make_judge_config",
    "make_rubric_dim",
    "make_scored_result",
    "make_subject",
    "make_template",
    "make_test_case",
    "memory_storage",
    "minimal_declaration",
    "result_capture_defaults",
]

#: The subject a run built here carries unless the test supplies one: engine-shaped, no
#: components, no payload, and ``state={}`` -- the recording that it carried nothing in, not the
#: absence of a state reader. Fixed ``captured_at`` so two default runs agree on it.
DEFAULT_SUBJECT = SubjectSnapshot(
    subject_id="subject-1",
    subject_label="Test subject",
    state={},
    captured_at="2026-07-20T00:00:00+00:00",
)

#: The candidate kind a factory-built template, run or result names unless the test supplies one.
DEFAULT_KIND = "test-kind"


def make_subject(
    subject_id: str = "subject-1", subject_label: str = "Test subject", **overrides: Any
) -> SubjectSnapshot:
    """A subject with the given identity and no components, recording that it carried no state.

    The engine-shaped stand-in for "a run of this subject": a host's adapter would also fill
    ``components`` with the subject's variant coordinates, and a test about those passes them.

    Args:
        subject_id: The subject's stable id.
        subject_label: Its display label.
        **overrides: Any other ``SubjectSnapshot`` field.

    Returns:
        The subject.
    """
    fields: dict[str, Any] = dict(
        subject_id=subject_id, subject_label=subject_label, state={}, captured_at=DEFAULT_SUBJECT.captured_at
    )
    fields.update(overrides)
    return SubjectSnapshot(**fields)


#: A recorded request-settings stamp, for a fixture run that had a judge or a simulated user.
#: Only its being RECORDED matters here: the launch stamps one whenever the role ran, and a run
#: carrying the role's model without it reads as having recorded none. Tests about the stamp's
#: VALUE compare against the production constants directly.
RECORDED_REQUEST_SETTINGS = ClientRequestSettings(max_tokens=4096, reasoning_max_tokens=None)


def make_template(**overrides: Any) -> EvalTemplate:
    """Create an ``EvalTemplate`` with sensible test defaults."""
    defaults: dict[str, Any] = dict(
        scope_id="uni-1",
        candidate_kind=DEFAULT_KIND,
        name="category_bundle",
        description="A shopper asks for a bundle of items from two unrelated categories.",
        intent="Probe the subject's response to unusual category pairings.",
        tools_required=["shop", "chat"],
        variation_axes=[
            VariationAxis(name="category_pair", generator="llm"),
            VariationAxis(name="tone", generator="enum", values=["casual", "cocky"]),
        ],
        conversation=ConversationSpec(
            actors=[ActorPolicy(id="shopper", policy="curious tester", intent="probe the subject")],
        ),
        world_seed=WorldSeed(
            namespaces={
                "shop": {
                    "catalog": [],
                    "cart": [],
                    "history": [],
                },
                "chat": {"messages": []},
            }
        ),
        goal_state_checks=["state.shop.cart.length >= 1"],
        rubric=[RubricDim(name="conversation.tone", description="Sounds like the subject", scale="ordinal")],
    )
    defaults.update(overrides)
    return EvalTemplate(**defaults)


def make_test_case(**overrides: Any) -> EvalTestCase:
    """Create an ``EvalTestCase`` with sensible test defaults."""
    defaults: dict[str, Any] = dict(
        scope_id="uni-1",
        template_id="tpl-1",
        variation_params={"category_pair": "garden + stationery", "tone": "casual"},
    )
    defaults.update(overrides)
    return EvalTestCase(**defaults)


def make_eval_run(**overrides: Any) -> EvalRun:
    """Create an ``EvalRun`` with sensible test defaults.

    A run given a ``judge_model`` also gets recorded per-dim attribution and a recorded
    judge-config set unless the caller states its own: the launch stamps ``judge_model``,
    ``effective_judges``, ``effective_judges_source`` and ``judge_config_ids`` in one construction,
    so a run carrying the pin without them is a shape production cannot produce. Pass
    ``effective_judges=None`` or ``judge_config_ids=None`` explicitly to model a run whose writer
    recorded no attribution. A run given a ``judge_model`` or ``simulator_model`` likewise gets that role's recorded
    request settings (a judged one, the judge temperature too), and a replay run names a corpus — a run cannot replay without naming the capture it serves.

    See the module docstring for the subject, payload and world-placement defaults.
    """
    defaults: dict[str, Any] = dict(
        scope_id="uni-1",
        template_id="tpl-1",
        candidate_kind=DEFAULT_KIND,
        subject_snapshot=DEFAULT_SUBJECT,
        host_payload={},
        candidate_model="sonnet",
        k_runs=1,
        test_case_ids=["tc-1"],
        rubric_scales={},
        world_placements={},
        # What the launch path stamps; a witnessed run is the one a test names.
        apparatus_provenance="commissioned",
    )
    defaults.update(overrides)
    if defaults.get("judge_model") is not None and "effective_judges" not in overrides:
        pin = defaults["judge_model"]
        defaults["effective_judges"] = {OUTCOME_DIM_ID: pin, TRANSCRIPT_DIM_ID: pin}
        defaults.setdefault("effective_judges_source", "recorded")
    if defaults.get("judge_model") is not None and "judge_config_ids" not in overrides:
        defaults["judge_config_ids"] = {}
        defaults.setdefault("judge_config_provenance", {})
    if defaults.get("judge_model") is not None and "judge_request_settings" not in overrides:
        defaults["judge_request_settings"] = RECORDED_REQUEST_SETTINGS
    if defaults.get("judge_model") is not None and "judge_temperature" not in overrides:
        defaults["judge_temperature"] = DEFAULT_JUDGE_TEMPERATURE
    if defaults.get("simulator_model") is not None and "simulator_request_settings" not in overrides:
        defaults["simulator_request_settings"] = RECORDED_REQUEST_SETTINGS
    if defaults.get("cassette_mode") == "replay" and "cassette_corpus_id" not in overrides:
        defaults["cassette_corpus_id"] = "run-capture"
    return EvalRun(**defaults)


def fixture_variant_key(model: str) -> str:
    """The variant key a fixture result at ``model`` carries when its test names none.

    The runner stamps one key per run, and a run is one candidate model, so a fixture keys its
    results by their model: results at two models are two contestants, as they would be in
    production, and results at one model pool. A test about variants names its own keys.
    """
    return canonical_digest({"fixture_candidate_model": model})


def result_capture_defaults(model: str = "sonnet") -> dict[str, Any]:
    """The capture fields every runner exit writes, as a completed cell with nothing measured writes them.

    A result is only ever produced by the runner, which states each of these on every exit, so a
    fixture result carries them too: ``termination="completed"``, the run's cost composition, no
    per-role rows and so a cost of ``0.0`` (the sum over none, as the runner derives it), no
    covariates, phase timings or host measures, and the variant key of a run at ``model``
    (:func:`fixture_variant_key`). A test about one of them names its own.
    """
    return dict(
        candidate_kind=DEFAULT_KIND,
        termination="completed",
        cost_usd=0.0,
        cost_roles=list(blended_cost_roles(None)),
        usage=[],
        covariates={},
        phase_timings={},
        host_measures={},
        variant_key=fixture_variant_key(model),
        identity_version=IDENTITY_VERSION,
    )


def make_eval_result(**overrides: Any) -> EvalResult:
    """Create an ``EvalResult`` with sensible test defaults.

    Carries no trace: the payload lives in a sibling :class:`EvalTrace`. Use
    :func:`make_eval_trace` for the payload, and set ``has_trace=True`` here when the test needs a
    surface to know one exists. The capture fields default as :func:`result_capture_defaults` says.
    """
    defaults: dict[str, Any] = {
        **result_capture_defaults(overrides.get("model", "sonnet")),
        "scope_id": "uni-1",
        "eval_run_id": "run-1",
        "test_case_id": "tc-1",
        "model": "sonnet",
        "k_iteration": 1,
        "goal_state_outcomes": [GoalStateOutcome(expression="state.shop.cart.length >= 1", passed=True)],
        "rubric_scores": [RubricScore(dim="conversation.tone", score=4, scale="ordinal")],
        "cost_usd": 0.01,
    }
    defaults.update(overrides)
    return EvalResult(**defaults)


def make_scored_result(
    test_case_id: str = "tc1",
    model: str = "m1",
    run_id: str = "r1",
    k: int = 1,
    goal_passes: tuple[bool, ...] = (True,),
    rubric_scores: tuple[tuple[str, int], ...] = (),
    candidate_error: str | None = None,
    infra_error: str | None = None,
    judge_error: str | None = None,
) -> EvalResult:
    """An ``EvalResult`` shaped by what the scoring functions actually read.

    Takes the *scoring inputs* as parameters -- how many goal states passed, which rubric dims
    scored what, and which of the three error slots is filled -- because that is the whole
    vocabulary a :mod:`threetears.evals.kernel.scoring` test varies. Deliberately does NOT route
    through :func:`make_eval_result`, whose non-zero ``cost_usd`` default would be a cost nobody
    asked for under a result built to make a point about pass^k.

    Args:
        test_case_id: The case this result measured.
        model: The model under test.
        run_id: The run it belongs to.
        k: The iteration index within the run.
        goal_passes: One flag per goal-state expression, in order.
        rubric_scores: ``(dim, score)`` pairs the judge returned.
        candidate_error: Set when the subject failed -- a scored outcome, not an exclusion.
        infra_error: Set when the harness failed -- excluded from scoring.
        judge_error: Set when the judge failed -- its scores are dropped.

    Returns:
        The result, carrying nothing the scoring functions do not read.
    """
    return EvalResult(
        scope_id="u",
        eval_run_id=run_id,
        test_case_id=test_case_id,
        model=model,
        k_iteration=k,
        goal_state_outcomes=[GoalStateOutcome(expression=f"g{i}", passed=p) for i, p in enumerate(goal_passes)],
        rubric_scores=[RubricScore(dim=n, score=s, scale="ordinal") for n, s in rubric_scores],
        candidate_error=candidate_error,
        infra_error=infra_error,
        judge_error=judge_error,
        **result_capture_defaults(model),
    )


def make_eval_trace(**overrides: Any) -> EvalTrace:
    """Create an ``EvalTrace`` whose ids line up with :func:`make_eval_result`'s defaults."""
    result_id = overrides.pop("result_id", "res-1")
    defaults: dict[str, Any] = dict(
        id=eval_trace_doc_id(result_id),
        scope_id="uni-1",
        result_id=result_id,
        eval_run_id="run-1",
        trace=[{"turn": 1, "candidate": "..."}],
    )
    defaults.update(overrides)
    return EvalTrace(**defaults)


def minimal_declaration(*, control: str | None = None, axis_id: str | None = None) -> CampaignDesign:
    """The smallest valid ``CampaignDesign``, optionally carrying a control variant key.

    Takes a key VERBATIM. Where the point is a control an observation actually carries, pass the
    key its run resolves (:func:`~threetears.evals.analysis.variant_key_of_run`) -- a hand-written
    key joins nothing.

    Args:
        control: The declared control variant key, or None.
        axis_id: The axis to declare. Defaults to the candidate model, the one lever every host
            declares.

    Returns:
        The declaration.
    """
    return CampaignDesign(
        axes=[SweptAxis(axis_id=axis_id or CANDIDATE_MODEL_LEVER, values=[SweepableValue.of("glm", display="GLM")])],
        control=control,
        held_fixed=ControlDeclaration(stimulus="controlled", apparatus="commissioned"),
    )


def as_listed(runs: Any, elide_payload: frozenset[str] = frozenset()) -> list[EvalRun]:
    """What a storage LISTING returns for ``runs``: each payload without the elided paths, marked.

    Args:
        runs: The runs the stand-in would otherwise return.
        elide_payload: The host's listing elisions, as ``EvalService.list_runs`` passes them.

    Returns:
        The listed copies; the originals are untouched.
    """
    if not elide_payload:
        return list(runs)
    listed = []
    for run in runs:
        copy = run.model_copy(update={"host_payload": omit_paths(run.host_payload, sorted(elide_payload))})
        copy.note_elided_payload(elide_payload)
        listed.append(copy)
    return listed


def load_listed(
    load_one: Any, run_ids: Any, scope_id: str, elide_payload: frozenset[str] = frozenset()
) -> list[EvalRun]:
    """What a storage BATCH read returns: every id ``load_one`` resolves, listed as :func:`as_listed` lists.

    Args:
        load_one: The stand-in's own ``(run_id, scope_id) -> EvalRun | None`` lookup.
        run_ids: The ids the batch asks for; an unresolved one is skipped, as the store skips it.
        scope_id: The scope to read in.
        elide_payload: The host's listing elisions, as the batch reader passes them.

    Returns:
        The listed copies of the runs that resolved.
    """
    return as_listed([run for run_id in run_ids if (run := load_one(run_id, scope_id)) is not None], elide_payload)


def make_judge_config(**overrides: Any) -> JudgeConfig:
    """Create a ``JudgeConfig`` in the default scope, bound to one namespaced dim."""
    fields: dict[str, Any] = dict(
        scope_id="uni-1", name="tone-v1", rubric_dim_id="conversation.tone", prompt_template="Score the tone."
    )
    fields.update(overrides)
    return JudgeConfig(**fields)


def make_rubric_dim(**overrides: Any) -> CatalogRubricDim:
    """Create a ``CatalogRubricDim`` in the default scope."""
    fields: dict[str, Any] = dict(
        scope_id="uni-1",
        key="conversation.tone",
        dim=RubricDim(name="conversation.tone", description="Sounds like itself", scale="ordinal"),
    )
    fields.update(overrides)
    return CatalogRubricDim(**fields)


def make_campaign(**overrides: Any) -> EvalCampaign:
    """Create an ``EvalCampaign`` in the default scope, holding no runs."""
    fields: dict[str, Any] = dict(
        scope_id="uni-1", name="campaign", subject_id="subject-1", behavior="conversation", created_by="test:fixture"
    )
    fields.update(overrides)
    return EvalCampaign(**fields)


def make_analysis(**overrides: Any) -> EvalAnalysis:
    """Create an ``EvalAnalysis`` in the default scope with an empty authored document."""
    fields: dict[str, Any] = dict(
        scope_id="uni-1",
        campaign_id="campaign-1",
        subject_id="subject-1",
        subject_kind="",
        behavior="conversation",
        generation=GenerationProvenance(
            prompt_id="eval_analysis_gen",
            prompt_version="v1",
            generator_model="writer-a",
            bundle_fingerprint="sha256:abc",
            generated_at="2026-08-22T00:00:00+00:00",
            token_cost=0.0,
            bundle_assembled_at="2026-01-01T00:00:00+00:00",
            repair_attempts=0,
            repaired_refusal=None,
            cell_model_version=1,
            user_message_digest="sha256:message",
        ),
        document=AuthoredAnalysis(headline="h", summary="", findings=[], decisions=[], questions=[], next=[]),
        decision_surface=DecisionSurface(),
    )
    fields.update(overrides)
    return EvalAnalysis(**fields)


def make_analysis_attempt(**overrides: Any) -> EvalAnalysisAttempt:
    """Create a stored ``EvalAnalysisAttempt`` in the default scope."""
    fields: dict[str, Any] = dict(
        scope_id="uni-1",
        campaign_id="campaign-1",
        outcome="stored",
        analysis_id="analysis-1",
        generator_model="writer-a",
        prompt_id="eval_analysis_gen",
        bundle_fingerprint="sha256:abc",
        started_at="2026-08-22T00:00:00+00:00",
    )
    fields.update(overrides)
    return EvalAnalysisAttempt(**fields)


def make_calibration_rating(**overrides: Any) -> CalibrationRating:
    """Create a ``CalibrationRating``: one person's 4 for ``conversation.tone`` on ``make_eval_result``'s result."""
    defaults: dict[str, Any] = {
        "scope_id": "uni-1",
        "run_id": "run-1",
        "result_id": "r-1",
        "rubric_dim": "conversation.tone",
        "rater": "host",
        "rater_kind": "person",
        "scale": "ordinal",
        "score": 4,
        "reason": "warm and on topic",
    }
    defaults.update(overrides)
    return CalibrationRating(**defaults)


def make_insight(**overrides: Any) -> EvalInsight:
    """Create an ``EvalInsight`` in the default scope."""
    fields: dict[str, Any] = dict(
        scope_id="uni-1", subject_id="subject-1", subject_kind="", statement="observed", confidence="medium"
    )
    fields.update(overrides)
    return EvalInsight(**fields)


def memory_storage() -> tuple[EvalStorage, InMemoryDocumentStore]:
    """An ``EvalStorage`` over a fresh in-memory reference store, and the store itself."""
    store = InMemoryDocumentStore()
    return EvalStorage(store), store
