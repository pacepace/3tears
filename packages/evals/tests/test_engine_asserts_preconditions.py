"""The engine asserts a template's preconditions at t=0 for every cell, so no host can forget to.

A precondition says what a template presumes about the world before the subject's first turn. When only a
host's kind asserted it, a kind that forgot ran the subject in a world that never held the presumption and
scored what followed as the subject's failure: a wrong number, filed against the wrong party. The runner now
asserts it itself, after ``prepare`` returns and before ``invoke`` — against the world its session read back
the moment the seed settled — and excludes the cell as the rig's (``precondition_failed``), never scores it.

The kinds below never assert anything, which is exactly the host the change is for.
"""

from __future__ import annotations

from typing import Any

from threetears.evals.kernel import CandidateOutput, CandidatePreparationFailed, CellSink, WorldSession
from threetears.evals.schema import EvalTemplate, EvalTestCase, JudgedArtifact, Precondition, WorldSeed
from threetears.evals.kernel.identity import IDENTITY_VERSION, DerivedVariantIdentity, compute_variant_key
from threetears.evals.schema.values import SweepableValue
from threetears.evals.kernel.result_condition import ResultOutcome, classify_result
from threetears.evals.run import assert_preconditions
from threetears.evals.run.runner import RunnerOptions, run_one_result
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

_KIND = "presuming-probe"
_MODEL_ONLY = {"model": SweepableValue.of("test/model", display="test/model")}
_CARRIERS = ("page_reader", "console")
_GERMAN = Precondition(expression='state.document_language == "de"', presumes="the page under test is in German")


class _ForgetfulKind:
    """Seeds its cell's world (or not) and runs, never asserting the template's presumptions."""

    judged_artifact = JudgedArtifact.UNJUDGED

    def __init__(self, *, seeds: bool = True, asserts: bool = False) -> None:
        self._seeds = seeds
        self._asserts = asserts
        self.invoked = False

    async def prepare(self, *, world: WorldSession | None, world_seed: WorldSeed, **_: Any) -> WorldSession | None:
        if self._seeds and world is not None:
            await world.seed(world_seed, attached=_CARRIERS)
            if self._asserts:
                # The host-called assertion this change keeps working: the kind asks first, over the same t=0 world.
                seeded = await world.end_state() if world.seeded_state is None else world.seeded_state
                if failed := assert_preconditions(
                    self.template,
                    EvalTestCase(template_id=self.template.id, scope_id="u"),
                    seeded,
                    world=world.registry,
                ):
                    raise CandidatePreparationFailed(
                        "precondition: asked by the kind", termination="precondition_failed", preconditions=failed
                    )
        return world

    template: EvalTemplate

    async def invoke(self, instance: Any, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        self.invoked = True
        return CandidateOutput(output=[{"read": True}])


def _template(language: str, *, preconditions: list[Precondition]) -> EvalTemplate:
    return EvalTemplate(
        scope_id="uni-1",
        name="presumes",
        intent="read the page",
        candidate_kind=_KIND,
        world_seed=WorldSeed(namespaces={"page_reader": {"document_language": language}}),
        preconditions=preconditions,
    )


async def _cell(kind: _ForgetfulKind, template: EvalTemplate) -> Any:
    kind.template = template
    return await run_one_result(
        toyhost_host(profile=toyhost_profile()),
        template=template,
        test_case=EvalTestCase(template_id=template.id, scope_id="u"),
        subject_id="subj-1",
        model="test/model",
        k_iteration=1,
        eval_run_id="run-1",
        scope_id="u",
        judge_service=None,
        options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: kind}),
        variant=DerivedVariantIdentity(
            variant_key=compute_variant_key(_MODEL_ONLY), identity_version=IDENTITY_VERSION, levers=_MODEL_ONLY
        ),
    )


async def test_a_presumption_the_world_does_not_hold_excludes_the_cell_though_the_kind_never_asked() -> None:
    kind = _ForgetfulKind()

    result, _ = await _cell(kind, _template("en", preconditions=[_GERMAN]))

    assert not kind.invoked, "the subject ran in a world that never held what the probe presumes"
    assert result.termination == "precondition_failed"
    assert classify_result(result) is ResultOutcome.INFRA_EXCLUDE, "excluded as the rig's, never scored"
    (outcome,) = result.precondition_outcomes
    assert (outcome.presumes, outcome.held) == (_GERMAN.presumes, False)
    assert result.runner_error is not None and "the page under test is in German" in result.runner_error


async def test_a_presumption_the_world_holds_runs_the_cell() -> None:
    kind = _ForgetfulKind()

    result, _ = await _cell(kind, _template("de", preconditions=[_GERMAN]))

    assert kind.invoked
    assert result.termination == "completed"
    assert result.precondition_outcomes == []


async def test_a_kind_that_asserts_them_itself_gets_one_answer_either_way() -> None:
    held, _ = await _cell(_ForgetfulKind(asserts=True), _template("de", preconditions=[_GERMAN]))
    refused, _ = await _cell(_ForgetfulKind(asserts=True), _template("en", preconditions=[_GERMAN]))

    assert held.termination == "completed", "the engine's second asking agrees with the kind's"
    assert refused.termination == "precondition_failed"
    assert refused.runner_error == "precondition: asked by the kind", "the kind's own refusal stands, once"


async def test_a_kind_that_never_seeds_through_its_session_is_excluded_with_the_reason() -> None:
    kind = _ForgetfulKind(seeds=False)

    result, _ = await _cell(kind, _template("de", preconditions=[_GERMAN]))

    assert not kind.invoked
    assert result.termination == "precondition_failed"
    assert result.runner_error is not None and "never seeded through the engine's world session" in result.runner_error


async def test_a_template_with_no_preconditions_reads_nothing_at_t0() -> None:
    kind = _ForgetfulKind()
    seen: list[WorldSession] = []
    original = kind.prepare

    async def prepare(**arguments: Any) -> Any:
        seen.append(arguments["world"])
        return await original(**arguments)

    kind.prepare = prepare  # type: ignore[method-assign]
    result, _ = await _cell(kind, _template("en", preconditions=[]))

    assert result.termination == "completed"
    assert seen[0].seeded_state is None, "no presumption to assert, so no t=0 read is paid for"
