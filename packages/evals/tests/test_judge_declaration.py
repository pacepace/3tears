"""A judged dimension declares its judge, and nothing scores it with a judge it did not declare.

Before core v15.1 a template dimension with no active judge config was scored by the engine's built-in prompt, and
the run recorded ``builtin:<sha>`` with nothing saying a configured judge had been meant: a host whose configs were
never authored, or archived, was judged by the engine's wording while it believed its own was in use. Now a dimension
registered apart from its judge says which judge scores it (``RubricDim.judge``: ``"builtin"`` or ``"configured"``),
registration (create, seed, an update writing the rubric) refuses one that does not or that the scope contradicts,
and every launch refuses the same before any spend, for a template stored before the declaration or a scope that
moved since. The quick path's ``Judge`` is the declaration, so an eval's author states nothing new.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import pytest

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals import Eval, Judge
from threetears.evals.quick import as_host
from threetears.evals.run import build_judge_service, create_template, get_template, plan_judge, update_template
from threetears.evals.run.definition_seed import SeedCorpus, seed_eval_definitions
from threetears.evals.schema import JudgedArtifact, RubricDim, StopReason
from threetears.evals.schema.models import EvalTemplate, fingerprint_criterion
from packages.evals.tests.factories import declared, make_judge_config, memory_storage
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.run import TOY_EXTRACTOR_KIND, toyhost_template
from packages.evals.tests.off_loop import off_loop

TONE = "dj.tone"
WARMTH = "dj.warmth"
PACE = "dj.pace"

_FIXTURE = Path(__file__).parent / "fixtures" / "durable_documents" / "v15" / "eval_template.json"


def _admit(_definition: object) -> None:
    """A host check with nothing to refuse: the rule under test is the engine's."""


_HOOKS: dict[str, Any] = {
    "require_known_tools_allowed": _admit,
    "refuse_undeclared_world_seed": _admit,
    "refuse_undeliverable_template": _admit,
}


def _dim(name: str = TONE, judge: Any = None) -> RubricDim:
    return RubricDim(name=name, description="Sounds like the station's voice.", scale="ordinal", judge=judge)


def _definition(*dims: RubricDim, name: str = "probe") -> dict[str, Any]:
    return {
        "name": name,
        "intent": "Probe the station's voice.",
        "candidate_kind": TOY_EXTRACTOR_KIND,
        "rubric": [dim.model_dump(mode="json") for dim in dims],
    }


def _stored(*dims: RubricDim) -> tuple[Any, EvalTemplate]:
    """A host whose store holds a template with ``dims``, saved past admission as a template stored before .26 is."""
    storage, _ = memory_storage()
    host = toyhost_host(storage=storage, clients=lambda role, model, **_: object())
    template = toyhost_template().model_copy(update={"rubric": list(dims)})
    storage.save_template(template)
    return host, template


def _plan(host: Any, template: EvalTemplate, selection: Mapping[str, str] | None = None) -> Any:
    return plan_judge(host, template, "judge-a", selection, judged_artifact=JudgedArtifact.DOCUMENT)


# --- the repro: a stored template's undeclared dimension is refused before any spend ------------------------------


def test_a_stored_undeclared_dimension_is_refused_at_launch_where_it_was_scored_by_the_builtin_prompt() -> None:
    """The production defect: a judged dim with no config resolved to ``builtin:<sha>`` and nothing said so."""
    host, template = _stored(_dim())

    with pytest.raises(ValidationFailedError, match=r"'dj\.tone' is judged but declares no judge") as refused:
        _plan(host, template)
    assert 'judge: "builtin"' in refused.value.remedy and 'judge: "configured"' in refused.value.remedy
    with pytest.raises(ValidationFailedError, match=r"'dj\.tone' is judged but declares no judge"):
        build_judge_service(host, template, "judge-a", judged_artifact=JudgedArtifact.DOCUMENT)


def test_a_declared_builtin_dimension_plans_the_builtin_judge() -> None:
    """The positive control: declared, the same template plans as it always did."""
    host, template = _stored(_dim(judge="builtin"))

    assert _plan(host, template).config_ids == {}


# --- registration refuses what launch would --------------------------------------------------------------------


def test_create_names_every_undeclared_dimension_in_one_refusal() -> None:
    host = toyhost_host(storage=memory_storage()[0])

    with pytest.raises(ValidationFailedError) as refused:
        create_template(host, _definition(_dim(TONE), _dim(WARMTH), _dim(PACE)), scope_id=TOYHOST_SCOPE, **_HOOKS)

    for name in (TONE, WARMTH, PACE):
        assert f"'{name}' is judged but declares no judge" in str(refused.value)
    assert host.storage.query_templates(TOYHOST_SCOPE) == []


def test_the_seed_refuses_an_undeclared_dimension_and_writes_nothing() -> None:
    storage, _ = memory_storage()
    host = toyhost_host(storage=storage)
    template = EvalTemplate(**_definition(_dim()), scope_id=TOYHOST_SCOPE)

    with pytest.raises(ValidationFailedError, match=r"'dj\.tone' is judged but declares no judge"):
        seed_eval_definitions(host, SeedCorpus(scope_id=TOYHOST_SCOPE, templates=(template,)), **_HOOKS)
    assert storage.query_templates(TOYHOST_SCOPE) == []


def test_the_seed_honours_a_configured_dimension_by_the_config_it_seeds() -> None:
    storage, _ = memory_storage()
    host = toyhost_host(storage=storage)
    template = EvalTemplate(**_definition(_dim(judge="configured")), scope_id=TOYHOST_SCOPE)
    config = make_judge_config(scope_id=TOYHOST_SCOPE, rubric_dim_id=TONE)

    outcome = seed_eval_definitions(
        host, SeedCorpus(scope_id=TOYHOST_SCOPE, templates=(template,), judge_configs=(config,)), **_HOOKS
    )

    assert outcome.created["eval_template"] == 1 and outcome.created["judge_config"] == 1


def test_an_update_writing_the_rubric_is_held_to_the_declaration_and_one_that_does_not_is_not() -> None:
    host, template = _stored(_dim())

    with pytest.raises(ValidationFailedError, match="declares no judge"):
        update_template(host, template.id, TOYHOST_SCOPE, {"rubric": [_dim().model_dump(mode="json")]}, **_HOOKS)
    renamed = update_template(host, template.id, TOYHOST_SCOPE, {"description": "still undeclared"}, **_HOOKS)
    assert renamed.rubric[0].judge is None, "a write that does not author the rubric leaves it editable"

    declared_now = update_template(
        host, template.id, TOYHOST_SCOPE, {"rubric": [_dim(judge="builtin").model_dump(mode="json")]}, **_HOOKS
    )
    assert declared_now.rubric[0].judge == "builtin"
    assert _plan(host, declared_now).config_ids == {}


# --- configured with no config; builtin with one ----------------------------------------------------------------


def test_configured_with_no_config_is_refused_at_create() -> None:
    host = toyhost_host(storage=memory_storage()[0])

    with pytest.raises(ValidationFailedError, match=r"declares judge: \"configured\", and scope .* holds no active"):
        create_template(host, _definition(_dim(judge="configured")), scope_id=TOYHOST_SCOPE, **_HOOKS)


def test_configured_whose_config_was_archived_after_create_is_refused_at_launch() -> None:
    host = toyhost_host(storage=memory_storage()[0])
    config = make_judge_config(scope_id=TOYHOST_SCOPE, rubric_dim_id=TONE)
    host.storage.save_judge_config(config)
    template = create_template(host, _definition(_dim(judge="configured")), scope_id=TOYHOST_SCOPE, **_HOOKS)
    assert _plan(host, template).config_ids == {TONE: config.id}

    host.storage.save_judge_config(config.model_copy(update={"archived": True}))

    with pytest.raises(ValidationFailedError, match="holds no active judge config"):
        _plan(host, template)


def test_builtin_beside_an_active_config_is_refused_at_launch() -> None:
    """The reverse of the defect: the config would be silently ignored."""
    host, template = _stored(_dim(judge="builtin"))
    config = make_judge_config(scope_id=TOYHOST_SCOPE, rubric_dim_id=TONE)
    host.storage.save_judge_config(config)

    with pytest.raises(ValidationFailedError, match=rf"declares judge: \"builtin\", but judge config '{config.id}'"):
        _plan(host, template)


def test_a_selected_config_scores_only_a_dimension_declared_configured() -> None:
    host, template = _stored(_dim(judge="builtin"))
    config = make_judge_config(scope_id=TOYHOST_SCOPE, rubric_dim_id=TONE, archived=True)
    host.storage.save_judge_config(config)

    with pytest.raises(ValidationFailedError, match="was selected for it by this launch"):
        _plan(host, template, {TONE: config.id})
    configured = template.model_copy(update={"rubric": [_dim(judge="configured")]})
    assert _plan(host, configured, {TONE: config.id}).config_ids == {TONE: config.id}


# --- the quick path declares for itself -------------------------------------------------------------------------


@dataclass(frozen=True)
class _Reply:
    content: str
    input_tokens: int | None = 10
    output_tokens: int | None = 10
    cost_usd: float | None = None
    price_source: str | None = None
    model: str = "scripted/judge"
    served_model: str | None = "scripted/judge"
    reasoning_tokens: int | None = None
    stop_reason: StopReason = "end_turn"
    temperature: float | None = None


# parity-with: threetears.evals.schema.CompletionClient
class _FakeJudge:
    """A judge scoring every dimension it is asked 4, or pass on a pass/fail one."""

    model_name = "scripted/judge"

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        assert response_format is not None
        scores = response_format["json_schema"]["schema"]["properties"]["criteria_scores"]
        (name,) = scores["required"]
        answer = "4" if "4" in scores["properties"][name]["enum"] else "pass"
        return _Reply(json.dumps({"reasoning": "fine", "criteria_scores": {name: answer}}))

    async def aclose(self) -> None:
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


async def _answer(case: Mapping[str, Any]) -> str:
    """Answer the question."""
    return "an answer"


async def test_a_quick_judge_s_dict_rubric_runs_and_its_stored_dimensions_say_builtin() -> None:
    judge = Judge(_FakeJudge(), {"helpful": "It helps."})
    ev = Eval([{"q": "x"}], _answer, [judge], k=1)
    host = as_host([ev], store=":memory:")

    summary = (await ev.run(host=host, scope="declared-quick")).summary

    assert summary.n_scored == 1
    template = off_loop(get_template, host, summary.template_id or "", "declared-quick")
    assert [(dim.name, dim.judge) for dim in template.rubric] == [("answer.helpful", "builtin")]


def test_a_rubric_dim_handed_to_a_quick_judge_without_a_judge_is_declared_builtin() -> None:
    """No new required field on the quick path: the Judge declares what the caller did not."""
    judge = Judge(_FakeJudge(), [_dim()])

    assert [dim.judge for dim in judge.dims] == ["builtin"]


async def test_a_quick_judge_s_configured_dimension_with_no_config_is_refused_before_anything_runs() -> None:
    judge = Judge(_FakeJudge(), [_dim(judge="configured")])
    ev = Eval([{"q": "x"}], _answer, [judge], k=1)
    host = as_host([ev], store=":memory:")

    with pytest.raises(ValueError, match="holds no active judge config"):
        await ev.run(host=host, scope="declared-quick")
    assert off_loop(host.storage.query_templates, "declared-quick") == []


# --- the declaration is not the criterion -----------------------------------------------------------------------


def test_the_judge_declaration_leaves_the_criterion_fingerprint_where_the_v15_fixture_has_it() -> None:
    """A label keyed by the criterion keeps pairing across the declaration: who judges is not what is asked."""
    stored = EvalTemplate.from_dict(json.loads(_FIXTURE.read_text(encoding="utf-8")))
    (undeclared,) = stored.rubric
    assert undeclared.judge is None, "a v15.0 document reads with no declaration"

    for judge in ("builtin", "configured"):
        assert fingerprint_criterion(declared(undeclared, judge)) == fingerprint_criterion(undeclared)
