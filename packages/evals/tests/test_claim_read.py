"""A claim read: drawing a claim instrument's graded claims by stratum, and reading its calibration off people's answers.

The draw is stratified by the instrument's own answer, plans each stratum by the misses it is expected to hold, takes
at most two claims a cell, and leaves out (and counts) what no check of the instrument may read: a grade under another
definition, stored evidence other than what was graded, a grade a recording's searches decided (``replayed``), a
restatement. An item shows a claim as the instrument read it and nothing it concluded. The reading weights every
answered item by its stratum's population over its answered items, so the population's 2x2, kappa, recall, precision
and rate bias are the whole campaign's, not the draw's; the floor counts cells, not claims; a lean across arms is
``not_met`` with ``instrument_leans``; a read whose every claim was replayed reads not available. The cells are built by
hand (``claim_read_support.py``); the text is invented: a reading room's almanac.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from packages.evals.tests.claim_read_support import CRITERION, FINGERPRINT, graded_cell
from threetears.evals.analysis.claim_read import (
    EXCLUDED_OTHER_EVIDENCE,
    EXCLUDED_REDEFINED,
    EXCLUDED_REPLAYED,
    EXCLUDED_RESTATEMENT,
    GradedCell,
    allocate_draws,
    draw_claim_read,
    read_claim_calibration,
)
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.schema import Segment
from threetears.evals.schema.claim_read import (
    ClaimAnswer,
    ClaimRead,
    ClaimReadArm,
    ClaimVerdict,
    ClaimView,
    claim_view_digest,
)
from threetears.evals.storage import InMemoryDocumentStore

_ARMS = [ClaimReadArm(key="control", run_ids=["run-a"]), ClaimReadArm(key="candidate", run_ids=["run-b"])]


def _draw(cells: list[GradedCell], **options: object) -> ClaimRead:
    fields: dict[str, object] = {
        "criterion": CRITERION,
        "criterion_fingerprint": FINGERPRINT,
        "questions": ["true"],
        "scope_id": "u",
        "created_by": "agent:test",
        "created_at": "2026-11-22T00:00:00+00:00",
    }
    fields.update(options)
    arms = fields.pop("arms", _ARMS)
    return draw_claim_read(cells, arms, **fields)  # type: ignore[arg-type]


def _answered(read: ClaimRead, fails: set[str], *, rater: str = "ana", made_by: str = "person") -> ClaimRead:
    """The read with one rater's answer to every item: fail where the item is in ``fails``, else pass."""
    answers = [
        ClaimAnswer(
            item=item.item,
            rater=rater,
            made_by=made_by,  # type: ignore[arg-type]
            verdicts=[ClaimVerdict(question="true", score=0 if item.item in fails else 1)],
        )
        for item in read.items
    ]
    return ClaimRead.model_validate({**read.model_dump(), "answers": [a.model_dump() for a in answers]})


# --- allocation ------------------------------------------------------------------------------------------------------


def test_a_plan_gives_every_stratum_its_floor_then_spreads_the_rest_by_expected_misses() -> None:
    population = {"contradicted": 50, "true_ungrounded": 0, "unverifiable": 100, "supported": 1000}

    plan = allocate_draws(population, 200)  # type: ignore[arg-type]

    assert sum(plan.values()) == 200 and plan["true_ungrounded"] == 0
    assert all(plan[s] >= 20 for s in ("contradicted", "unverifiable", "supported"))
    # Neyman over the miss prior: the large, sparse supported stratum takes most, then unverifiable; both are drawn
    # well past their share of an even split between the instrument's fails and passes.
    assert plan["supported"] > plan["unverifiable"] > plan["contradicted"]


def test_a_plan_never_draws_past_a_stratum_and_never_past_the_population() -> None:
    population = {"contradicted": 3, "true_ungrounded": 1, "unverifiable": 4, "supported": 30}

    assert allocate_draws(population, 500) == population  # type: ignore[arg-type]
    small = allocate_draws(population, 10)  # type: ignore[arg-type]
    assert sum(small.values()) == 10 and all(small[s] <= population[s] for s in population)  # type: ignore[index]


def test_a_plan_refuses_a_negative_count() -> None:
    with pytest.raises(ValueError, match="no negative count"):
        allocate_draws({"supported": 3}, -1)  # type: ignore[dict-item]


# --- the draw --------------------------------------------------------------------------------------------------------


def _campaign() -> list[GradedCell]:
    cells = []
    for arm, run in (("a", "run-a"), ("b", "run-b")):
        for at in range(12):
            verdicts = ["supported", "supported", "unverifiable", "contradicted", "true_ungrounded", "supported"]
            cells.append(graded_cell(run, f"{arm}{at}", verdicts[at % 3 :], case_id=f"case-{at}"))
    return cells


def test_a_draw_takes_at_most_two_claims_a_cell_and_counts_each_stratum() -> None:
    read = _draw(_campaign(), planned_items=40)

    per_cell: dict[str, int] = {}
    for item in read.items:
        per_cell[item.result_id] = per_cell.get(item.result_id, 0) + 1
    assert max(per_cell.values()) <= 2
    assert {s.stratum: s.drawn for s in read.draw.strata} == {
        s: sum(1 for item in read.items if item.stratum == s) for s in ("contradicted", "true_ungrounded", "unverifiable", "supported")
    }
    assert sum(s.population for s in read.draw.strata) == sum(len(c.grade.claims) for c in _campaign())
    assert all(s.drawn <= s.planned for s in read.draw.strata)


def test_a_draw_reads_the_same_whatever_order_its_cells_come_in() -> None:
    cells = _campaign()

    assert _draw(cells, planned_items=30) == _draw(list(reversed(cells)), planned_items=30)
    assert _draw(cells, planned_items=30, seed=1).items != _draw(cells, planned_items=30, seed=2).items


def test_what_no_check_may_read_is_left_out_and_counted() -> None:
    redefined = graded_cell("run-a", "x1", ["supported", "supported"], fingerprint="b" * 64)
    replayed = graded_cell("run-a", "x2", ["true_ungrounded", "supported", "supported"], replayed=True)
    restated = graded_cell("run-a", "x3", ["supported", "supported"], restates=[None, 0])
    moved = graded_cell("run-a", "x4", ["supported"])
    moved = moved._replace(
        evidence=moved.evidence.model_copy(update={"segments": (Segment(text="Other text altogether.", unit=0),)})
    )
    elsewhere = graded_cell("run-z", "x5", ["supported"])

    read = _draw([redefined, replayed, restated, moved, elsewhere])

    assert read.draw.excluded == {
        EXCLUDED_OTHER_EVIDENCE: 1,
        EXCLUDED_REDEFINED: 2,
        EXCLUDED_REPLAYED: 3,
        EXCLUDED_RESTATEMENT: 1,
    }
    assert [(item.result_id, item.claim) for item in read.items] == [("x3", 0)]


def test_an_item_shows_the_claim_as_the_instrument_read_it_and_nothing_it_concluded() -> None:
    cell = graded_cell("run-a", "c1", ["true_ungrounded"])

    (item,) = _draw([cell]).items

    assert set(ClaimView.model_fields) == {"claim", "excerpt", "start", "end", "segment", "evidence"}
    assert [(e.ref, e.provenance) for e in item.view.evidence] == [("S1", "case"), ("S2", "run"), ("W1", "web")]
    assert item.view.evidence[-1].text == cell.grade.web[0].windows[0].text
    assert item.view_digest == claim_view_digest(item.view)
    assert item.verdict == "true_ungrounded"


# --- the stored shape ------------------------------------------------------------------------------------------------


def test_a_read_refuses_items_that_do_not_add_up_to_its_draw() -> None:
    read = _draw(_campaign(), planned_items=20)
    document = read.model_dump()
    document["items"][0]["stratum"] = "contradicted" if document["items"][0]["stratum"] != "contradicted" else "supported"
    with pytest.raises(ValidationError, match="its instrument answer gives"):
        ClaimRead.model_validate(document)
    document = read.model_dump()
    document["items"] = document["items"][1:]
    with pytest.raises(ValidationError, match="drew"):
        ClaimRead.model_validate(document)


def test_a_read_refuses_an_answer_to_a_question_it_does_not_ask_and_a_bool_score() -> None:
    read = _draw(_campaign(), planned_items=5)
    item = read.items[0].item
    with pytest.raises(ValidationError, match="which the read does not ask"):
        ClaimRead.model_validate(
            {**read.model_dump(), "answers": [{"item": item, "rater": "ana", "verdicts": [{"question": "grounded", "score": 1}]}]}
        )
    with pytest.raises(ValidationError):
        ClaimVerdict(question="true", score=True)  # type: ignore[arg-type]


def test_a_claim_read_is_stored_and_read_back_whole() -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    read = _answered(_draw(_campaign(), planned_items=10, campaign_id="c-1"), set())

    storage.save_claim_read(read, create_only=True)

    loaded, etag = storage.load_claim_read_with_etag(read.id, "u")
    assert loaded == read and etag is not None
    assert storage.query_claim_reads("u", campaign_id="c-1") == [read]
    assert storage.query_claim_reads("u", campaign_id="c-2") == []


# --- the reading -----------------------------------------------------------------------------------------------------


def _weighted_read() -> tuple[ClaimRead, set[str]]:
    """Ten contradicted and ninety supported claims, five and ten drawn: people fail four of the five and one of the
    ten, so every weighted figure can be worked by hand (weights 2 and 9)."""
    cells = [graded_cell("run-a", f"c{at}", ["contradicted", "contradicted"]) for at in range(5)]
    cells += [graded_cell("run-a", f"s{at}", ["supported", "supported"]) for at in range(45)]
    read = _draw(cells, arms=[_ARMS[0]], planned={"contradicted": 5, "supported": 10})
    contradicted = sorted(item.item for item in read.items if item.stratum == "contradicted")
    supported = sorted(item.item for item in read.items if item.stratum == "supported")
    return read, {*contradicted[:4], supported[0]}


def test_the_population_figures_are_weighted_by_stratum_and_carry_an_interval() -> None:
    read, fails = _weighted_read()

    (reading,) = read_claim_calibration(_answered(read, fails)).questions

    assert reading.table is not None
    assert [round(share, 6) for share in reading.table] == [0.08, 0.02, 0.09, 0.81]
    assert reading.recall.value == pytest.approx(8 / 17)
    assert reading.precision.value == pytest.approx(0.8)
    assert reading.bias.value == pytest.approx(10 / 17)
    expected = 0.1 * 0.17 + 0.9 * 0.83
    assert reading.kappa.value == pytest.approx((0.89 - expected) / (1 - expected))
    for estimate in (reading.kappa, reading.recall, reading.precision):
        assert estimate.lower is not None and estimate.upper is not None
        assert estimate.lower <= estimate.value <= estimate.upper  # type: ignore[operator]
    assert {s.stratum: (s.items, s.agreed) for s in reading.strata}["contradicted"] == (5, 0.8)


def test_the_floor_counts_cells_and_not_claims() -> None:
    cells = [graded_cell("run-a", f"c{at}", ["contradicted", "supported"]) for at in range(12)]
    read = _draw(cells, arms=[_ARMS[0]], planned_items=24)
    fails = {item.item for item in read.items if item.stratum == "contradicted"}

    (reading,) = read_claim_calibration(_answered(read, fails)).questions

    assert reading.items == 24 and reading.cells == 12
    assert reading.kappa.value == pytest.approx(1.0)
    assert reading.check is not None and reading.check.result == "insufficient"


def test_a_lean_across_arms_reads_not_met_instrument_leans() -> None:
    cells = []
    for arm, run in (("a", "run-a"), ("b", "run-b")):
        for at in range(40):
            cells.append(graded_cell(run, f"{arm}{at}", ["contradicted", "supported"] if arm == "a" else ["unverifiable", "supported"]))
    read = _draw(cells, planned_items=160)
    # People fail every claim the instrument contradicted (arm a) and every claim it could not settle (arm b): the
    # instrument catches arm a's false claims and none of arm b's.
    fails = {item.item for item in read.items if item.stratum in ("contradicted", "unverifiable")}

    (reading,) = read_claim_calibration(_answered(read, fails)).questions

    assert reading.lean_p is not None and reading.lean_p <= 0.05
    assert reading.check is not None
    assert (reading.check.result, reading.check.reason) == ("not_met", "instrument_leans")
    assert {arm.arm: arm.recall.value for arm in reading.arms} == {"control": 1.0, "candidate": 0.0}


def test_arms_the_instrument_reads_alike_show_no_lean() -> None:
    cells = [
        graded_cell(run, f"{arm}{at}", ["contradicted", "supported", "unverifiable"])
        for arm, run in (("a", "run-a"), ("b", "run-b"))
        for at in range(40)
    ]
    read = _draw(cells, planned_items=160)
    contradicted = sorted(item.item for item in read.items if item.stratum == "contradicted")
    fails = set(contradicted[::2])

    (reading,) = read_claim_calibration(_answered(read, fails)).questions

    assert reading.lean_p is not None and reading.lean_p > 0.05


def test_a_read_whose_every_claim_was_replayed_reads_not_available_replayed() -> None:
    cells = [graded_cell("run-a", f"r{at}", ["true_ungrounded", "supported"], replayed=True) for at in range(3)]

    (reading,) = read_claim_calibration(_draw(cells)).questions

    assert reading.check is not None and reading.check.state is not None
    assert (reading.check.state.state, reading.check.state.reason) == ("not_available", "replayed")


def test_a_models_answers_are_counted_and_read_in_no_figure() -> None:
    read, fails = _weighted_read()
    by_model = _answered(read, fails, rater="reviewer/model", made_by="model")

    calibration = read_claim_calibration(by_model)

    assert calibration.model_answers == len(read.items)
    assert calibration.questions[0].items == 0 and calibration.questions[0].kappa.value is None


def test_an_earlier_reads_answer_on_a_byte_identical_view_stands_for_an_unanswered_item() -> None:
    read, fails = _weighted_read()
    earlier = _answered(read, fails).model_copy(update={"id": "claimread-earlier"})

    calibration = read_claim_calibration(read, earlier=[earlier])

    assert calibration.borrowed == len(read.items)
    assert calibration.questions[0].recall.value == pytest.approx(8 / 17)


def test_is_claim_reads_peoples_share_and_checks_no_instrument() -> None:
    read, _ = _weighted_read()
    read = ClaimRead.model_validate({**read.model_dump(), "questions": ["true", "is_claim"]})
    answers = [
        ClaimAnswer(item=item.item, rater="ana", verdicts=[ClaimVerdict(question="is_claim", score=int(at % 5 != 0))])
        for at, item in enumerate(read.items)
    ]

    calibration = read_claim_calibration(read.model_copy(update={"answers": answers}))

    is_claim = calibration.questions[1]
    assert is_claim.question == "is_claim" and is_claim.check is None and is_claim.lean_p is None
    assert is_claim.failed.value is not None and 0.0 < is_claim.failed.value < 1.0
    assert not math.isnan(is_claim.failed.value)
