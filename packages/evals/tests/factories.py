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

from collections.abc import Sequence
from typing import Any

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, TelemetryRollup
from threetears.evals.analysis.lenses.comparison_sets import ComparisonSetsResult
from threetears.evals.analysis.lenses.frontier import FrontierResult
from threetears.evals.kernel.authored import AuthoredAnalysis
from threetears.evals.kernel.campaign_upgrade import UPGRADED_CONTROL_LABEL
from threetears.evals.kernel.campaign import (
    EvalAnalysis,
    EvalAnalysisAttempt,
    EvalCampaign,
    EvalInsight,
    GenerationProvenance,
)
from threetears.evals.kernel.declaration import (
    AspectDesign,
    CampaignArm,
    CampaignDesign,
    ControlDeclaration,
    DecisionClause,
    DecisionRule,
    Question,
    RateDrop,
    SweptAxis,
)
from threetears.evals.schema import SubjectSnapshot
from threetears.evals.kernel.host.profile import CANDIDATE_MODEL_LEVER
from threetears.evals.schema.values import SweepableValue
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.kernel.identity import IDENTITY_VERSION
from threetears.evals.schema.models import (
    GOAL_CHECK_PROOF_RULES,
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
    JudgeDeclaration,
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
    "CURRENT_BUNDLE_SHAPE",
    "DEFAULT_KIND",
    "DEFAULT_SUBJECT",
    "RECORDED_REQUEST_SETTINGS",
    "as_listed",
    "declared",
    "fixture_variant_key",
    "load_listed",
    "empty_bundle",
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

#: The bundle-shape version a bundle built now carries (``AnalysisContextBundle.schema_version``): the value a test
#: that records "an analysis generated from a bundle" writes as ``bundle_schema_version``, so it describes the shape
#: this build writes and moves with it. A test of an older shape names its own number instead.
CURRENT_BUNDLE_SHAPE: int = AnalysisContextBundle.model_fields["schema_version"].default

#: The candidate kind a factory-built template, run or result names unless the test supplies one.
DEFAULT_KIND = "test-kind"


def make_subject(
    subject_id: str = "subject-1", subject_label: str = "Test subject", **overrides: Any
) -> SubjectSnapshot:
    """A subject with the given identity and no components, recording that it carried no state.

    The engine-shaped stand-in for "a run of this subject": a host's adapter would also fill
    ``components`` with the subject's variant coordinates, and a test about those passes them.

    :param subject_id: The subject's stable id.
    :ptype subject_id: str
    :param subject_label: Its display label.
    :ptype subject_label: str
    :param **overrides: Any other ``SubjectSnapshot`` field.
    :ptype overrides: Any
    :return: The subject.
    :rtype: SubjectSnapshot
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
        rubric=[declared(RubricDim(name="conversation.tone", description="Sounds like the subject", scale="ordinal"))],
    )
    defaults.update(overrides)
    return EvalTemplate(**defaults)


def declared(dim: RubricDim, judge: JudgeDeclaration = "builtin") -> RubricDim:
    """``dim`` declaring its judge, as a template a run launches from must (``RubricDim.judge``).

    :param dim: A rubric dimension, declared or not.
    :ptype dim: RubricDim
    :param judge: The judge it declares: the built-in prompt unless the test authors a config for it.
    :ptype judge: JudgeDeclaration
    :return: A copy declaring ``judge``.
    :rtype: RubricDim
    """
    return dim.model_copy(update={"judge": judge})


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
    if defaults.get("judge_model") is not None and "judge_asked_settings" not in overrides:
        defaults["judge_asked_settings"] = RECORDED_REQUEST_SETTINGS
    if defaults.get("judge_model") is not None and "judge_temperature" not in overrides:
        defaults["judge_temperature"] = DEFAULT_JUDGE_TEMPERATURE
    if defaults.get("goal_check_proofs") is not None and "goal_check_proof_rules" not in overrides:
        defaults["goal_check_proof_rules"] = GOAL_CHECK_PROOF_RULES
    if defaults.get("simulator_model") is not None and "simulator_asked_settings" not in overrides:
        defaults["simulator_asked_settings"] = RECORDED_REQUEST_SETTINGS
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
        "rubric_scores": [
            RubricScore(dim="conversation.tone", score=4, scale="ordinal", axis="capability", judge_temperature=0.0)
        ],
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

    :param test_case_id: The case this result measured.
    :ptype test_case_id: str
    :param model: The model under test.
    :ptype model: str
    :param run_id: The run it belongs to.
    :ptype run_id: str
    :param k: The iteration index within the run.
    :ptype k: int
    :param goal_passes: One flag per goal-state expression, in order.
    :ptype goal_passes: tuple[bool, ...]
    :param rubric_scores: ``(dim, score)`` pairs the judge returned.
    :ptype rubric_scores: tuple[tuple[str, int], ...]
    :param candidate_error: Set when the subject failed -- a scored outcome, not an exclusion.
    :ptype candidate_error: str | None
    :param infra_error: Set when the harness failed -- excluded from scoring.
    :ptype infra_error: str | None
    :param judge_error: Set when the judge failed -- its scores are dropped.
    :ptype judge_error: str | None
    :return: The result, carrying nothing the scoring functions do not read.
    :rtype: EvalResult
    """
    return EvalResult(
        scope_id="u",
        eval_run_id=run_id,
        test_case_id=test_case_id,
        model=model,
        k_iteration=k,
        goal_state_outcomes=[GoalStateOutcome(expression=f"g{i}", passed=p) for i, p in enumerate(goal_passes)],
        rubric_scores=[
            RubricScore(dim=n, score=s, scale="ordinal", axis="capability", judge_temperature=0.0)
            for n, s in rubric_scores
        ],
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


def minimal_declaration(
    *,
    control: str | None = None,
    control_levers: dict[str, SweepableValue] | None = None,
    control_identity_version: int | None = None,
    axis_id: str | None = None,
) -> CampaignDesign:
    """The smallest valid ``CampaignDesign``, optionally carrying a control variant key.

    Takes a key VERBATIM. Where the point is a control an observation actually carries, pass the
    key its run resolves (:func:`~threetears.evals.analysis.variant_key_of_run`) -- a hand-written
    key joins nothing.

    :param control: The declared control variant key, or None.
    :ptype control: str | None
    :param control_levers: The control's pre-image, as ``control_from_run_id`` records it, or None for a
        typed key.
    :ptype control_levers: dict[str, SweepableValue] | None
    :param control_identity_version: The predicate the key was computed under; None lets the
        declaration date it with this build's, as it dates any key typed without one.
    :ptype control_identity_version: int | None
    :param axis_id: The axis to declare. Defaults to the candidate model, the one lever every host
        declares.
    :ptype axis_id: str | None
    :return: The declaration.
    :rtype: CampaignDesign
    """
    return CampaignDesign(
        axes=[SweptAxis(axis_id=axis_id or CANDIDATE_MODEL_LEVER, values=[SweepableValue.of("glm", display="GLM")])],
        control=control,
        control_levers=control_levers,
        control_identity_version=control_identity_version,
        held_fixed=ControlDeclaration(stimulus="controlled", apparatus="commissioned"),
    )


def as_listed(runs: Any, elide_payload: frozenset[str] = frozenset()) -> list[EvalRun]:
    """What a storage LISTING returns for ``runs``: each payload without the elided paths, marked.

    :param runs: The runs the stand-in would otherwise return.
    :ptype runs: Any
    :param elide_payload: The host's listing elisions, as ``EvalService.list_runs`` passes them.
    :ptype elide_payload: frozenset[str]
    :return: The listed copies; the originals are untouched.
    :rtype: list[EvalRun]
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

    :param load_one: The stand-in's own ``(run_id, scope_id) -> EvalRun | None`` lookup.
    :ptype load_one: Any
    :param run_ids: The ids the batch asks for; an unresolved one is skipped, as the store skips it.
    :ptype run_ids: Any
    :param scope_id: The scope to read in.
    :ptype scope_id: str
    :param elide_payload: The host's listing elisions, as the batch reader passes them.
    :ptype elide_payload: frozenset[str]
    :return: The listed copies of the runs that resolved.
    :rtype: list[EvalRun]
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


def minimal_aspect_design(*, axis_id: str | None = None, **overrides: Any) -> AspectDesign:
    """The smallest valid ``AspectDesign``: one axis at one level, its stimulus controlled and its rig commissioned.

    :param axis_id: The axis to declare. Defaults to the candidate model, the one lever every host declares.
    :ptype axis_id: str | None
    :param overrides: Any other ``AspectDesign`` field.
    :ptype overrides: Any
    :return: The design.
    :rtype: AspectDesign
    """
    fields: dict[str, Any] = dict(
        axes=[SweptAxis(axis_id=axis_id or CANDIDATE_MODEL_LEVER, values=[SweepableValue.of("glm", display="GLM")])],
        held_fixed=ControlDeclaration(stimulus="controlled", apparatus="commissioned"),
    )
    fields.update(overrides)
    return AspectDesign(**fields)


#: The ``design`` :func:`make_aspect_campaign` declares when a test names none: :func:`minimal_aspect_design`'s.
_MINIMAL_DESIGN = object()


def make_aspect_campaign(
    *,
    label: str = "conversation",
    template_id: str | None = None,
    run_ids: Sequence[str] = (),
    design: AspectDesign | None | object = _MINIMAL_DESIGN,
    question: str = "",
    control: CampaignArm | None = None,
    margin: float | None = None,
    floors: Sequence[DecisionClause] = (),
    **overrides: Any,
) -> EvalCampaign:
    """A stored campaign of ONE aspect, built from what the aspect declares, whatever shape the campaign is stored in.

    A per-aspect test builds its campaign here and reads it through ``aspect_view``, so it never depends on how a
    campaign holds its aspects. While a campaign is stored as one template, one bag of runs and one design, the
    aspect is that campaign read through its upgrade, which bounds what this can build:

    - ``label`` is the campaign's ``behavior``, which the one aspect is labelled with.
    - ``margin`` is derived, never stored: an upgraded aspect's is the ``max_points`` of its floor reading every pass
      rate. So ``margin`` must equal that floor's, and be None exactly when ``floors`` has no ``RateDrop`` naming no
      reading; anything else is refused rather than built as some other margin.
    - ``floors`` name each kind and reading once: the stored shape would merge two into one floor.
    - ``floors`` need a ``question``, and the design's ``merit_axes`` are the question's: None exactly when there is
      no question, a list (empty for an unscoped question) when there is one.
    - ``design`` None is an exploratory aspect, which declares no question, control or floor.
    - ``control`` is the control arm labelled ``UPGRADED_CONTROL_LABEL``: the stored shape names a control by key.

    :param label: The aspect's label.
    :ptype label: str
    :param template_id: The template its runs were gathered on.
    :ptype template_id: str | None
    :param run_ids: Its runs.
    :ptype run_ids: Sequence[str]
    :param design: Its design: :func:`minimal_aspect_design` when omitted, None for an exploratory aspect.
    :ptype design: AspectDesign | None
    :param question: The campaign's question; empty asks none.
    :ptype question: str
    :param control: The campaign's control arm; None declares none.
    :ptype control: CampaignArm | None
    :param margin: The aspect's margin, as above.
    :ptype margin: float | None
    :param floors: The clauses of the floors on the aspect, in order.
    :ptype floors: Sequence[DecisionClause]
    :param overrides: Any other campaign field but ``behavior``, ``template_id``, ``run_ids`` and ``declared_design``.
    :ptype overrides: Any
    :return: The campaign.
    :rtype: EvalCampaign
    :raises ValueError: The arguments declare what this stored shape cannot hold.
    """
    aspect_design = minimal_aspect_design() if design is _MINIMAL_DESIGN else design
    fields: dict[str, Any] = dict(behavior=label, template_id=template_id, run_ids=list(run_ids))
    if aspect_design is None:
        if question or control is not None or floors or margin is not None:
            raise ValueError("an exploratory aspect declares no question, control, margin or floor")
        return make_campaign(**fields, **overrides)
    if not isinstance(aspect_design, AspectDesign):
        raise TypeError(f"design is an AspectDesign or None, not {type(aspect_design).__name__}")
    readings = [(clause.kind, clause.reading if isinstance(clause, RateDrop) else None) for clause in floors]
    if len(set(readings)) != len(readings):
        raise ValueError("floors name one kind and reading twice, which a stored campaign merges into one floor")
    derived = next(
        (clause.max_points for clause in floors if isinstance(clause, RateDrop) and clause.reading is None), None
    )
    if margin != derived:
        raise ValueError(
            f"margin={margin} cannot be built: a stored campaign's aspect reads its margin off its floor that reads "
            f"every pass rate, which here gives {derived}; pass margin={derived}, or a RateDrop(max_points={margin}) "
            "floor naming no reading (a campaign of aspects stores the two apart)"
        )
    if not question and floors:
        raise ValueError("a stored campaign keeps floors on a question; pass the question")
    if (aspect_design.merit_axes is None) != (not question):
        raise ValueError(
            "a design's merit_axes are its question's: None with no question, a list with one (empty for an unscoped "
            f"question); this one has {aspect_design.merit_axes!r} and question={question!r}"
        )
    questions = (
        [
            Question(
                text=question,
                decision=DecisionRule(clauses=tuple(floors)) if floors else None,
                merit_axes=list(aspect_design.merit_axes or []),
            )
        ]
        if question
        else []
    )
    if control is not None and (control.label, control.role) != (UPGRADED_CONTROL_LABEL, "control"):
        raise ValueError(
            f"a stored campaign names its control by key alone, so it reads back as role 'control' labelled "
            f"{UPGRADED_CONTROL_LABEL!r}; pass that arm"
        )
    shared = {name: getattr(aspect_design, name) for name in AspectDesign.model_fields if name != "merit_axes"}
    fields["declared_design"] = CampaignDesign(
        **shared,
        questions=questions,
        control=control.variant_key if control is not None else None,
        control_levers=control.levers if control is not None else None,
        control_identity_version=control.identity_version if control is not None else None,
    )
    return make_campaign(**fields, **overrides)


def empty_bundle(**overrides: Any) -> AnalysisContextBundle:
    """An evidence bundle over no runs: the frozen bundle an analysis of nothing carries."""
    fields: dict[str, Any] = dict(
        campaign_id="campaign-1",
        subject_id="subject-1",
        behavior="conversation",
        scope_id="uni-1",
        comparison=ComparisonSetsResult(),
        frontier=FrontierResult(),
        telemetry=TelemetryRollup(
            n_runs=0,
            n_results=0,
            n_errors=0,
            total_cost_usd=0.0,
            incomplete_cost_usd=0.0,
            unattributed_cost_usd=0.0,
        ),
        host_declarations_digest="sha256:declarations",
    )
    fields.update(overrides)
    return AnalysisContextBundle(**fields)


def make_analysis(**overrides: Any) -> EvalAnalysis:
    """Create an ``EvalAnalysis`` in the default scope with an empty authored document over an empty bundle."""
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
            bundle_schema_version=CURRENT_BUNDLE_SHAPE,
            host_declarations_digest="sha256:declarations",
        ),
        document=AuthoredAnalysis(findings=[], decisions=[], questions=[], next=[]),
        decision_surface=DecisionSurface(),
        bundle=empty_bundle(campaign_id=overrides.get("campaign_id", "campaign-1")).to_dict(),
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
        error=None,
        generator_model="writer-a",
        prompt_id="eval_analysis_gen",
        bundle_fingerprint="sha256:abc",
        started_at="2026-08-22T00:00:00+00:00",
    )
    fields.update(overrides)
    return EvalAnalysisAttempt(**fields)


def make_calibration_rating(**overrides: Any) -> CalibrationRating:
    """Create a ``CalibrationRating``: one person's 4 for ``conversation.tone`` on ``make_eval_result``'s result.

    It carries no rated content (``rated_withheld`` says so) unless ``rated`` is given.
    """
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
        "rated_withheld": "a factory rating of no stored result",
    }
    defaults.update(overrides)
    if "rated" in overrides and "rated_withheld" not in overrides:
        defaults.pop("rated_withheld")
    return CalibrationRating(**defaults)


def make_insight(**overrides: Any) -> EvalInsight:
    """Create an ``EvalInsight`` in the default scope."""
    fields: dict[str, Any] = dict(
        scope_id="uni-1",
        subject_id="subject-1",
        subject_kind="",
        statement="observed",
        confidence="medium",
        invalidation_trigger="retired when its analysis is archived, or replaced by a later analysis restating it",
    )
    fields.update(overrides)
    return EvalInsight(**fields)


def memory_storage() -> tuple[EvalStorage, InMemoryDocumentStore]:
    """An ``EvalStorage`` over a fresh in-memory reference store, and the store itself."""
    store = InMemoryDocumentStore()
    return EvalStorage(store), store
