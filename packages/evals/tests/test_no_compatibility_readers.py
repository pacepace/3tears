"""The fields every writer sets are required, and the reads that used to fill their absence are gone.

A regenerable document is refused under any schema version but this build's
(``REGENERABLE_SCHEMA_VERSION``), and a core document is upgraded to this build's core version before
it is validated (``CORE_UPGRADERS``), so no document this build can read lacks a field this build's
writers set — after the upgrade. Each "this field predates…" fallback was therefore a branch for a
state nothing can produce, and each was deleted with its field made required; a field a future core
version requires is filled, or named as not recorded, by that version's upgrader, never by a reader. These tests pin the refusal side of that: a
document — or a construction — missing one of those fields is refused by name, rather than read as
an older document would once have been read.

Beside them sit the refusals the same work added where a ``None`` was a guess:

- a rubric-dim name is namespaced on the MODEL, so a bare one is unrepresentable — in a template
  rubric, a judge config's binding, a calibration rating and a proposer's draft alike;
- a reporter case pins a recorded memo, its writer's message and that message's digest together;
- a run's recorded lever map is dated by the identity version that recorded it;
- an unjudged run is a recorded level in its context key, not an unrecorded judge.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from threetears.evals.analysis.generator import user_message_digest
from threetears.evals.analysis.reporter_kind import (
    LabelCriterion,
    ReporterCase,
    ReporterLabel,
    judge_case_material,
)
from threetears.evals.kernel import (
    IDENTITY_VERSION,
    EvalStorage,
    OutOfRunBudget,
    resolve_context_identity,
    resolve_variant_identity,
)
from threetears.evals.schema import EvalResult, JudgeConfig, RubricDim
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.schema import SweepableValue
from threetears.evals.schema.models import CalibrationRating, JudgeRescore, RubricScore, RunCompleteness
from threetears.evals.kernel.surface import DecisionSurface, JudgedDimensionFacts, JudgedReading
from threetears.evals.gen import propose_draft
from threetears.evals.gen.prompts.boundary_gen import EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT
from threetears.evals.gen.prompts.proposer import EVAL_PROPOSER_TEMPLATE_DEFAULT
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import (
    make_analysis,
    make_campaign,
    make_eval_result,
    make_eval_run,
    make_template,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.llm_client_fakes import ReleasableClientMixin


def _without(model: BaseModel, field: str) -> dict[str, Any]:
    """``model`` as its constructor arguments, ``field`` left out."""
    data = model.model_dump()
    data.pop(field)
    return data


_REQUIRED: list[Any] = [
    *(
        pytest.param(make_eval_run, field, id=f"EvalRun.{field}")
        for field in ("candidate_kind", "k_runs", "rubric_scales")
    ),
    *(
        pytest.param(make_eval_result, field, id=f"EvalResult.{field}")
        for field in (
            "termination",
            "cost_roles",
            "usage",
            "covariates",
            "phase_timings",
            "host_measures",
            "variant_key",
            "identity_version",
        )
    ),
    pytest.param(make_template, "candidate_kind", id="EvalTemplate.candidate_kind"),
    pytest.param(make_campaign, "created_by", id="EvalCampaign.created_by"),
    pytest.param(make_analysis, "decision_surface", id="EvalAnalysis.decision_surface"),
    *(
        pytest.param(lambda: make_analysis().generation, field, id=f"GenerationProvenance.{field}")
        for field in (
            "bundle_assembled_at",
            "repair_attempts",
            "repaired_refusal",
            "cell_model_version",
            "user_message_digest",
        )
    ),
    pytest.param(
        lambda: RunCompleteness(
            expected_cells=1, produced_cells=1, persisted_cells=1, infra_excluded_cells=0, counted_from="run_loop"
        ),
        "counted_from",
        id="RunCompleteness.counted_from",
    ),
]


@pytest.mark.parametrize(("build", "field"), _REQUIRED)
def test_a_field_every_writer_sets_is_required(build: Callable[[], BaseModel], field: str) -> None:
    """Built whole, the model constructs; with the one field left out, it is refused by name."""
    whole = build()
    type(whole)(**whole.model_dump())  # the control: the dump reconstructs
    with pytest.raises(ValidationError, match=field):
        type(whole)(**_without(whole, field))


def test_a_campaign_with_a_blank_author_is_refused() -> None:
    """Every campaign has an author: the server records the creating surface, and a blank names nobody."""
    with pytest.raises(ValidationError, match="created_by"):
        make_campaign(created_by="")


# --- rubric-dim names are namespaced on the model ---------------------------------------------------


def _rescore(**fields: Any) -> JudgeRescore:
    """One re-judge record, with ``fields`` over a minimal valid one."""
    whole: dict[str, Any] = {
        "dims": ["triage.tone"],
        "prior_judge_error": "dim 'triage.tone': 429",
        "judge_model": "j",
        "cost_usd": 0.0,
    }
    return JudgeRescore(**{**whole, **fields})


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda name: RubricDim(name=name, description="d", scale="ordinal"), id="RubricDim.name"),
        pytest.param(
            lambda name: JudgeConfig(scope_id="s", name="c", rubric_dim_id=name, prompt_template="p"),
            id="JudgeConfig.rubric_dim_id",
        ),
        pytest.param(
            lambda name: CalibrationRating(
                scope_id="s",
                run_id="run-1",
                result_id="r-1",
                rubric_dim=name,
                rater="host",
                rater_kind="person",
                scale="ordinal",
                score=3,
                reason="why",
            ),
            id="CalibrationRating.rubric_dim",
        ),
        # Every place a judged dim's id is STORED, not only where one is authored: a score, a
        # re-judge's record, the run's per-dim maps, a result's per-dim maps, the decision surface,
        # and a reporter case's label. A bare name stored in any of them would bind, pool or
        # compare as the global name the authored guard exists to refuse.
        pytest.param(lambda name: RubricScore(dim=name, score=3, scale="ordinal"), id="RubricScore.dim"),
        pytest.param(lambda name: _rescore(dims=[name]), id="JudgeRescore.dims"),
        pytest.param(lambda name: _rescore(scores={name: 4}), id="JudgeRescore.scores"),
        pytest.param(lambda name: _rescore(errors={name: "429"}), id="JudgeRescore.errors"),
        pytest.param(lambda name: _rescore(cannot_tell={name: "no evidence"}), id="JudgeRescore.cannot_tell"),
        pytest.param(lambda name: _rescore(judge_config_ids={name: "cfg-1"}), id="JudgeRescore.judge_config_ids"),
        pytest.param(lambda name: make_eval_result(judge_config_ids={name: "cfg-1"}), id="EvalResult.judge_config_ids"),
        pytest.param(
            lambda name: make_eval_result(judge_cannot_tell={name: "no evidence"}), id="EvalResult.judge_cannot_tell"
        ),
        pytest.param(lambda name: make_eval_run(rubric_scales={name: "ordinal"}), id="EvalRun.rubric_scales"),
        pytest.param(
            lambda name: make_eval_run(judge_model="j", effective_judges={name: "j"}), id="EvalRun.effective_judges"
        ),
        pytest.param(
            lambda name: make_eval_run(
                judge_model="j", judge_config_ids={name: "cfg-1"}, judge_config_provenance={name: "chosen"}
            ),
            id="EvalRun.judge_config_ids",
        ),
        pytest.param(
            lambda name: make_eval_run(
                judge_model="j", judge_config_ids={"triage.tone": "cfg-1"}, judge_config_provenance={name: "chosen"}
            ),
            id="EvalRun.judge_config_provenance",
        ),
        pytest.param(
            lambda name: JudgedReading(dimension=name, n=0, n_independent=0, evidence_tier="undetermined"),
            id="JudgedReading.dimension",
        ),
        pytest.param(
            lambda name: DecisionSurface(dimensions={name: JudgedDimensionFacts(higher_is_better=True)}),
            id="DecisionSurface.dimensions",
        ),
        pytest.param(
            lambda name: ReporterLabel(dimension=name, direction="high", quote="q", criterion=_CRITERION),
            id="ReporterLabel.dimension",
        ),
    ],
)
@pytest.mark.parametrize("bare", ["character", ".character", "triage.", "a.b.c"])
def test_a_bare_dim_name_is_unrepresentable(build: Callable[[str], BaseModel], bare: str) -> None:
    with pytest.raises(ValidationError, match="must be namespaced"):
        build(bare)
    build("triage.character")  # the control: a namespaced name constructs
    build("__transcript__")  # and so does a reserved dual-score axis id


def test_a_stored_template_carrying_a_bare_dim_is_refused_on_read() -> None:
    """The model is the gate, so a stored document is held to it on read as on write."""
    document = make_template().to_dict()
    document["rubric"][0]["name"] = "tone"
    with pytest.raises(ValidationError, match="must be namespaced"):
        type(make_template()).from_dict(document)


class _ScriptedClient(ReleasableClientMixin):
    """Answers a proposal with canned content, priced at a cent."""

    model_name = "proposer-model"

    def __init__(self, content: str) -> None:
        self.content = content

    def price_ceiling(self, *, system: str, user: str, response_format: Any = None) -> float | None:
        return 0.01

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        return SimpleNamespace(content=self.content)


def _budget() -> OutOfRunBudget:
    return OutOfRunBudget(
        EvalStorage(InMemoryDocumentStore()), scope_id="proposals", cap_usd=1.0, blocking_executor=None
    )


def _draft(name: str, *, scale: str | None = "pass_fail") -> str:
    dim = {"name": name, "description": "declines without stonewalling"}
    if scale is not None:
        dim["scale"] = scale
    return json.dumps(
        {
            "template": {"name": "T", "intent": "i", "rubric": [dim], "variation_axes": []},
            "reused_dim_keys": [],
            "new_dim_suggestions": [{"key": "graceful_decline", "dim": dim, "axis": "boundary"}],
        }
    )


@pytest.mark.parametrize("axis", ["capability", "boundary"])
async def test_a_proposer_draft_naming_a_bare_dim_is_refused_naming_it(axis: str) -> None:
    """The draft is validated into the model, so a bare name the model wrote fails the whole draft."""
    with pytest.raises(ValidationFailedError, match="draft validation(.|\n)*'graceful_decline' must be namespaced"):
        await propose_draft(
            _ScriptedClient(_draft("graceful_decline")),
            budget=_budget(),
            axis=axis,
            subject_id="s",
            system_prompt="",
            subject_feed="F",
            catalog_feed="C",
        )
    proposal, _spend = await propose_draft(
        _ScriptedClient(_draft("boundary.graceful_decline")),
        budget=_budget(),
        axis=axis,
        subject_id="s",
        system_prompt="",
        subject_feed="F",
        catalog_feed="C",
    )
    assert [dim.name for dim in proposal.template.rubric] == ["boundary.graceful_decline"]


@pytest.mark.parametrize("axis", ["capability", "boundary"])
async def test_a_proposer_draft_dimension_with_no_scale_is_refused(axis: str) -> None:
    """A criterion states its scale; a draft that leaves one out is not read as 1-5."""
    with pytest.raises(ValidationFailedError, match="draft validation(.|\n)*scale"):
        await propose_draft(
            _ScriptedClient(_draft("boundary.graceful_decline", scale=None)),
            budget=_budget(),
            axis=axis,
            subject_id="s",
            system_prompt="",
            subject_feed="F",
            catalog_feed="C",
        )


@pytest.mark.parametrize("seed", [EVAL_PROPOSER_TEMPLATE_DEFAULT, EVAL_BOUNDARY_GEN_TEMPLATE_DEFAULT])
def test_each_proposer_prompt_teaches_the_rules_its_draft_is_refused_by(seed: Any) -> None:
    """A refusal that discards the paid call has to be taught in the prompt, not left to be discovered."""
    text = "".join(section.content_template for section in seed.sections)
    assert "DIMENSION NAMES ARE NAMESPACED" in text
    assert "`<context>.<dim>`" in text
    assert "REJECTED whole and discarded" in text
    assert "stored name is bare" not in text, "a catalog dim cannot be stored bare any more"
    assert "EVERY DIMENSION STATES ITS `scale`" in text
    assert text.count('"scale": "') >= 2, "every rubric example in the JSON shape states its scale"


# --- a reporter case pins a memo, its writer's message and the digest together ---------------------


_CRITERION = LabelCriterion(description="as stated", scale="ordinal", scoring_guide={})


def _reporter_case(**fields: Any) -> ReporterCase:
    defaults: dict[str, Any] = {
        "bundle": {"campaign_id": "c"},
        "bundle_fingerprint": "fp",
        "bundle_assembled_at": "2026-01-01T00:00:00+00:00",
        "recorded_analysis_id": "a-1",
        "recorded_memo": "the memo",
        "writer_message": "the message",
        "recorded_writer_message_digest": user_message_digest("the message"),
    }
    defaults.update(fields)
    return ReporterCase(**defaults)


@pytest.mark.parametrize(
    "missing",
    [
        pytest.param({"writer_message": None, "recorded_writer_message_digest": None}, id="memo-alone"),
        pytest.param({"recorded_writer_message_digest": None}, id="no-digest"),
        pytest.param({"recorded_memo": None, "recorded_analysis_id": None}, id="message-without-a-memo"),
        pytest.param({"writer_message": None}, id="digest-without-a-message"),
    ],
)
def test_a_reporter_case_holding_part_of_a_recorded_memo_is_refused(missing: dict[str, Any]) -> None:
    _reporter_case()  # the control: all three
    _reporter_case(
        recorded_analysis_id=None, recorded_memo=None, writer_message=None, recorded_writer_message_digest=None
    )  # and none of them
    with pytest.raises(ValidationError, match="together or not at all"):
        _reporter_case(**missing)


def test_a_reporter_case_storing_an_unstamped_label_is_refused() -> None:
    stamped = ReporterLabel(dimension="memo.accuracy", direction="high", quote="q", criterion=_CRITERION)
    _reporter_case(labels=[stamped])
    with pytest.raises(ValidationError, match="carry none"):
        _reporter_case(labels=[stamped.model_copy(update={"criterion": None})])


def test_a_case_pinning_no_memo_has_no_recorded_memo_to_judge() -> None:
    bare = _reporter_case(
        recorded_analysis_id=None, recorded_memo=None, writer_message=None, recorded_writer_message_digest=None
    )
    with pytest.raises(ValueError, match="pins no recorded memo"):
        judge_case_material(bare, generated_over=None)
    assert "the message" in judge_case_material(_reporter_case(), generated_over=None)


# --- a recorded lever map carries the version that recorded it -------------------------------------


def test_a_recorded_lever_map_with_no_identity_version_is_refused() -> None:
    profile = toyhost_profile()
    levers = {"model": SweepableValue.of("m", display="m")}
    dated = make_eval_run(candidate_model="m", variant_levers=levers, identity_version=IDENTITY_VERSION)
    assert resolve_variant_identity(run=dated, profile=profile).levers == levers
    undated = dated.model_copy(update={"identity_version": None})
    with pytest.raises(ValueError, match="cannot be dated"):
        resolve_variant_identity(run=undated, profile=profile)


# --- an unjudged run's context composes without a judge --------------------------------------------


#: A host that HAS a judge and a simulated user: the toy host with its inapplicability declarations
#: cleared, so a blank judge is not declared away by the host.
_JUDGE_CAPABLE = toyhost_profile(every_seat=True)


def test_an_unjudged_run_composes_its_roles_without_a_judge() -> None:
    """The runner refuses a judged run that names no judge, so a run naming none was not judged.

    That is a recorded level, and its roles component composes over what the run did pin. The
    control is a judged run whose attribution went unrecorded: that one is still partial, because
    which model scored each dim is genuinely lost.
    """
    unjudged = make_eval_run(simulator_model="sim/m")
    identity = resolve_context_identity(unjudged, _JUDGE_CAPABLE)
    assert "roles" not in identity.missing_components
    assert identity.context_components.roles is not None

    unattributed = make_eval_run(simulator_model="sim/m", judge_model="judge/m", effective_judges=None)
    assert "roles" in resolve_context_identity(unattributed, _JUDGE_CAPABLE).missing_components


def test_a_result_with_no_variant_key_is_refused() -> None:
    """Every run resolves a variant, so an empty key is a broken writer, refused on write and on read."""
    with pytest.raises(ValidationError, match="variant_key"):
        make_eval_result(variant_key="")
    stored = make_eval_result().to_dict()
    stored["variant_key"] = ""
    with pytest.raises(ValidationError, match="variant_key"):
        EvalResult.from_dict(stored)
