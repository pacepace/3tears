"""The entity extractor: the model step that lists the entities prose names, on a scripted client.

Every behaviour is driven through a client that answers from a script and records what it was sent, so nothing is
spent and each reply is chosen: the list comes back in the reply's order, every completion is billed to the ``judge``
ledger, an unreadable reply is asked again twice and then refused, a cut-short one is refused at once, blank text
makes no call, and a client that names no model is refused before any call. A refusal is a fault of the measuring
rig, an ``ApparatusError``: driven through the real runner, the cell whose extraction failed is excluded as
``apparatus_failed`` and never scored, and the run goes on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any

import pytest

from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.kind import (
    DOCUMENT_PARAM,
    TOY_EXTRACTOR_KIND,
    TOYHOST_EXTRACTION_FAMILY,
    ScriptedExtractionClient,
    ToyExtractorInstance,
    ToyExtractorKind,
)
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import (
    RUN_K,
    RUN_MODELS,
    toyhost_run,
    toyhost_template,
    toyhost_test_cases,
)
from threetears.evals.kernel import (
    EXTRACTED_SET_FAMILY,
    CandidateOutput,
    CellSink,
    EntityExtractionFailed,
    EntityExtractor,
    NamedEntity,
    RoleUsageLedger,
    SetGrader,
)
from threetears.evals.kernel.host import ApparatusError, MeasureRegistry
from threetears.evals.kernel.result_condition import ResultOutcome, classify_result
from threetears.evals.run import RunnerOptions, execute_run, run_blocking
from threetears.evals.schema import EvalTestCase
from threetears.evals.schema.reference import ReferenceKey

_MODEL = "scripted/extractor"

_TEXT = "Tonight we open with Band A, then Band C from Leeds, and close on a deep cut by Band E."


@dataclass(frozen=True)
class _Reply:
    """One completion, shaped as the engine's completion port describes it."""

    content: str
    stop_reason: str = "end_turn"
    model: str = _MODEL
    served_model: str | None = _MODEL
    input_tokens: int | None = 200
    output_tokens: int | None = 40
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    price_source: str | None = "scripted"
    temperature: float | None = None


# parity-with: threetears.evals.schema.CompletionClient
@dataclass
class _FakeScriptedClient:
    """Answers each call with the next scripted reply, and records every prompt pair and response format sent."""

    replies: list[_Reply]
    model_name: str = _MODEL
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        self.calls.append({"system": system, "user": user, "response_format": response_format})
        return self.replies[len(self.calls) - 1]

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> _FakeScriptedClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


# parity-with: threetears.evals.schema.CompletionClient
class _FakeUnnamedClient:
    """A client that names no model."""

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        raise AssertionError("an unnamed extractor is refused before any call")

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> _FakeUnnamedClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


def _listing(*entities: dict[str, Any]) -> _Reply:
    return _Reply(content=json.dumps({"entities": list(entities)}))


def _extractor(client: _FakeScriptedClient, *, fields: tuple[str, ...] = ("origin",)) -> EntityExtractor:
    return EntityExtractor(client=client, entity_kind="bands", fields=fields)


async def test_the_extractor_returns_the_entities_in_the_replys_order_with_the_fields_stated() -> None:
    client = _FakeScriptedClient(
        [
            _listing(
                {"name": "Band A", "fields": {"origin": None}},
                {"name": "Band C", "fields": {"origin": "Leeds"}},
                {"name": "Band E", "fields": {"origin": " "}},
            )
        ]
    )

    named = await _extractor(client).extract(_TEXT, ledger=RoleUsageLedger(role="judge"))

    assert named == (
        NamedEntity(name="Band A"),
        NamedEntity(name="Band C", fields={"origin": "Leeds"}),
        NamedEntity(name="Band E"),
    )
    (call,) = client.calls
    assert _TEXT in call["user"]
    assert "bands" in call["user"]
    schema = call["response_format"]["json_schema"]
    assert schema["strict"] is True
    entity = schema["schema"]["properties"]["entities"]["items"]
    assert entity["required"] == ["name", "fields"]
    assert entity["properties"]["fields"]["required"] == ["origin"]


async def test_a_grader_bound_to_the_extractor_lists_without_the_key_and_grades_under_its_own_formula() -> None:
    key = ReferenceKey.model_validate({"core": [{"name": "Band A"}], "acceptable": [{"name": "Band C"}]})
    client = _FakeScriptedClient([_listing({"name": "Band A"}, {"name": "Band E"})])
    extractor = _extractor(client, fields=())
    grader = SetGrader("picks", reader_label="Picks", extractor=extractor)

    grade = await grader.grade_text(_TEXT, key, ledger=RoleUsageLedger(role="judge"))

    (call,) = client.calls
    assert "Band C" not in call["system"] + call["user"].replace(_TEXT, "")
    assert (grade.found, grade.unverified_names) == (("Band A",), ("Band E",))
    assert grade.extractor_identity == extractor.identity
    assert {descriptor.formula for descriptor in grader.descriptors()} == {grade.formula}


async def test_a_grader_grades_blank_prose_as_an_empty_set_with_no_call() -> None:
    key = ReferenceKey.model_validate({"core": [{"name": "Band A"}]})
    client = _FakeScriptedClient([])
    grader = SetGrader("picks", reader_label="Picks", extractor=_extractor(client))

    grade = await grader.grade_text("  ", key, ledger=RoleUsageLedger(role="judge"))

    assert client.calls == []
    assert grade.host_measures() == {
        "picks_recall": 0.0,
        "picks_core_recall": 0.0,
        "picks_named": 0.0,
        "picks_unverified": 0.0,
    }


async def test_every_completion_is_recorded_on_the_judge_ledger() -> None:
    client = _FakeScriptedClient([_Reply(content="not json"), _listing({"name": "Band A", "fields": {"origin": None}})])
    ledger = RoleUsageLedger(role="judge")

    await _extractor(client).extract(_TEXT, ledger=ledger)

    (row,) = ledger.rows()
    assert row.role == "judge"
    assert row.model == _MODEL
    assert row.call_count == 2
    assert row.cost_usd == pytest.approx(0.002)


async def test_an_unreadable_reply_is_asked_twice_more_then_refused() -> None:
    unreadable = [
        _Reply(content="not json"),
        _listing({"name": "Band A"}),  # the declared field is missing
        _listing({"name": "Band A", "fields": {"origin": None, "decade": "1970s"}}),  # a field nobody asked for
    ]
    client = _FakeScriptedClient(unreadable)
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(EntityExtractionFailed, match="did not parse in 3 attempts"):
        await _extractor(client).extract(_TEXT, ledger=ledger)

    assert len(client.calls) == 3
    (row,) = ledger.rows()
    assert row.call_count == 3


async def test_a_reply_cut_short_is_refused_at_once_and_still_billed() -> None:
    client = _FakeScriptedClient(
        [_Reply(content='{"entities": [{"name": "Band A", "fields": {"origin": null}}]}', stop_reason="max_tokens")]
    )
    ledger = RoleUsageLedger(role="judge")

    with pytest.raises(EntityExtractionFailed, match="cut short"):
        await _extractor(client).extract(_TEXT, ledger=ledger)

    assert len(client.calls) == 1
    (row,) = ledger.rows()
    assert row.call_count == 1


@pytest.mark.parametrize("text", ["", "   \n\t"])
async def test_blank_text_names_nothing_and_makes_no_call(text: str) -> None:
    client = _FakeScriptedClient([])
    ledger = RoleUsageLedger(role="judge")

    assert await _extractor(client).extract(text, ledger=ledger) == ()
    assert client.calls == []
    assert ledger.rows() == []


def test_a_client_that_names_no_model_is_refused() -> None:
    with pytest.raises(ValueError, match="names its model"):
        EntityExtractor(client=_FakeUnnamedClient(), entity_kind="bands")


@pytest.mark.parametrize(
    ("entity_kind", "fields"),
    [(" ", ()), ("bands", ("origin", "origin")), ("bands", ("",))],
)
def test_a_blank_kind_or_a_blank_or_repeated_field_is_refused(entity_kind: str, fields: tuple[str, ...]) -> None:
    with pytest.raises(ValueError, match="entity_kind|fields"):
        EntityExtractor(client=_FakeScriptedClient([]), entity_kind=entity_kind, fields=fields)


async def test_a_ledger_of_another_role_is_refused() -> None:
    client = _FakeScriptedClient([])

    with pytest.raises(ValueError, match="'judge' role"):
        await _extractor(client).extract(_TEXT, ledger=RoleUsageLedger(role="candidate"))
    assert client.calls == []


def test_the_identity_names_the_model_and_moves_with_how_the_list_is_made() -> None:
    client = _FakeScriptedClient([])
    base = _extractor(client)

    assert base.identity.startswith(f"{_MODEL}|extract:")
    assert len(base.identity.rsplit(":", 1)[1]) == 12
    assert base.identity == _extractor(_FakeScriptedClient([])).identity
    others = {
        EntityExtractor(client=client, entity_kind="cities", fields=("origin",)).identity,
        _extractor(client, fields=()).identity,
        _extractor(client, fields=("origin", "decade")).identity,
        _extractor(_FakeScriptedClient([], model_name="scripted/other")).identity,
    }
    assert base.identity not in others
    assert len(others) == 4


def test_a_failed_extraction_is_a_fault_of_the_rig() -> None:
    assert issubclass(EntityExtractionFailed, ApparatusError)


# =============================================================================
# Through the runner: a cell whose extraction fails is excluded as apparatus_failed, never scored
# =============================================================================

#: The document whose answer the extractor's reply is cut short on.
_CUT_DOCUMENT = "doc-02"

_RUN_KEY = ReferenceKey.model_validate({"core": [{"name": "Band A"}], "acceptable": [{"name": "Band C"}]})


# parity-with: threetears.evals.schema.CompletionClient
@dataclass
class _FakeDocumentClient:
    """Lists Band A and Band E for every answer, and is cut short on the answer about :data:`_CUT_DOCUMENT`."""

    model_name: str = _MODEL
    calls: int = 0

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        self.calls += 1
        listing = json.dumps({"entities": [{"name": "Band A"}, {"name": "Band E"}]})
        return _Reply(content=listing, stop_reason="max_tokens" if _CUT_DOCUMENT in user else "end_turn")

    async def aclose(self) -> None:
        """Nothing to release."""

    async def __aenter__(self) -> _FakeDocumentClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.aclose()


class _ProseGradingKind(ToyExtractorKind):
    """The toy extractor, whose every cell also grades a prose answer about its document, inside ``invoke``."""

    def __init__(self, *, grader: SetGrader, **toy: Any) -> None:
        super().__init__(**toy)
        self._grader = grader

    async def invoke(self, instance: ToyExtractorInstance, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        output = await super().invoke(instance, test_case, sink)
        ledger = RoleUsageLedger(role="judge")

        def so_far() -> CandidateOutput:
            usage = [*output.telemetry.usage, *ledger.rows()]
            return output.model_copy(update={"telemetry": output.telemetry.model_copy(update={"usage": usage})})

        # Registered before the extraction, so a cell it fails keeps the spend it billed.
        sink.report_progress(so_far)
        answer = f"On {test_case.variation_params[DOCUMENT_PARAM]}: Band A, then a deep cut by Band E."
        grade = await self._grader.grade_text(answer, _RUN_KEY, ledger=ledger)
        graded = so_far()
        return graded.model_copy(update={"host_measures": {**output.host_measures, **grade.host_measures()}})


async def test_a_cell_whose_extraction_fails_is_excluded_as_apparatus_failed_and_never_scored() -> None:
    client = _FakeDocumentClient()
    grader = SetGrader("picks", reader_label="Picks", extractor=EntityExtractor(client=client, entity_kind="bands"))
    profile = replace(
        toyhost_profile(),
        measures=MeasureRegistry(
            (*TOYHOST_MEASURES, *grader.descriptors()),
            families=(TOYHOST_EXTRACTION_FAMILY, EXTRACTED_SET_FAMILY),
        ),
    )
    host = toyhost_host(profile=profile)
    world = host.profile.world
    assert world is not None, "the toy profile registers a world"
    template = toyhost_template()
    kind = _ProseGradingKind(
        grader=grader,
        client=ScriptedExtractionClient(),
        world=world,
        goal_checks=tuple(template.goal_state_checks),
    )
    test_cases = toyhost_test_cases(template)
    assert _CUT_DOCUMENT in {case.variation_params[DOCUMENT_PARAM] for case in test_cases}
    run = toyhost_run(model=RUN_MODELS[0], template=template, kind=kind, world=world)
    await run_blocking(host.blocking_executor, host.storage.save_eval_run, run)

    await execute_run(
        host,
        run=run,
        template=template,
        test_cases=test_cases,
        judge_service=None,
        options=RunnerOptions(candidate_kinds={TOY_EXTRACTOR_KIND: lambda _cell: kind}),
    )

    results = await run_blocking(host.blocking_executor, host.storage.query_eval_results_by_run, run.id, run.scope_id)
    assert len(results) == len(test_cases) * RUN_K, "the failed extraction ended its cells, not the run"
    case_of = {case.id: case.variation_params[DOCUMENT_PARAM] for case in test_cases}
    faulted = [result for result in results if case_of[result.test_case_id] == _CUT_DOCUMENT]
    graded = [result for result in results if case_of[result.test_case_id] != _CUT_DOCUMENT]
    assert len(faulted) == RUN_K
    for result in faulted:
        assert result.termination == "apparatus_failed"
        assert result.infra_error is not None
        assert result.infra_error.startswith("apparatus: EntityExtractionFailed: ")
        assert result.candidate_error is None
        assert classify_result(result) is ResultOutcome.INFRA_EXCLUDE
        assert not any(name.startswith("picks_") for name in result.host_measures), "a failed extraction scores nothing"
        (judged,) = [row for row in result.usage if row.role == "judge"]
        assert judged.call_count == 1, "the cut-short call was billed and kept on the excluded cell"
    for result in graded:
        assert result.termination == "completed"
        assert result.host_measures["picks_recall"] == pytest.approx(0.5)
        assert "picks_verified_share" in result.host_measures
