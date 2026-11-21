"""A claim criterion's readings (#1287): the ``claims`` family, every reading computed from a tally, and the built
controls, each moving only its own readings.

- **The family.** ``claims`` is an engine family a judge grades: its readings are never a check and never code-graded,
  and a host may not declare a family of that name.
- **Readings on read.** :func:`claim_readings` computes every reading from a stored tally; a reading whose population
  leaves the cell out is absent, never zero; a lost cell carries only ``_lost``. Three rank: false claims per unit and
  the share of units holding one (lower is better), and the grounded share (higher is better, only with
  ``grounded_in``). Every other reading is a diagnostic. Each description states its population and its evidence.
- **The tally from a grade** (:func:`claim_tally`) is where a claim's tags become counts: session claims count in no
  world reading, hedged and attributed claims are graded and kept out of the asserted counts, a sourced statement is
  asserted, and only a document of a provenance ``grounded_in`` names grounds a claim.
- **Built controls**, scripted grades as a verifier would write them, each against one baseline cell, with every
  other reading asserted unchanged.
- **A criterion name a described measure holds is refused** where a template is admitted and where it launches.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from packages.evals.tests.factories import memory_storage
from packages.evals.tests.fixtures.toyhost.host import TOYHOST_TEMPLATE_CHECKS
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_EXTRACTION_FAMILY, TOYHOST_MEASURES, toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.ops_support import TOYHOST_SCOPE, TOYHOST_SUBJECT
from threetears.evals.kernel import ENGINE_FAMILIES, MeasureFamily, MetricDescriptor, ValidationFailedError
from threetears.evals.kernel.claims.readings import (
    claim_reading_names,
    claim_reading_ranks,
    claim_readings,
    claim_tally,
    describe_claim_reading,
)
from threetears.evals.kernel.host import MeasureRegistrationError, MeasureRegistry
from threetears.evals.kernel.metrics import CLAIMS_FAMILY, CODE_GRADED_FAMILIES, is_code_graded, reader_name_defects
from threetears.evals.run import admit_template, quote_launch
from threetears.evals.run.authoring import update_template
from threetears.evals.schema import ClaimCriterion, PageBudget, RepairBudget, Segment
from threetears.evals.schema.models import (
    ClaimCall,
    ClaimCitation,
    ClaimEvidence,
    ClaimGrade,
    ClaimJudgeCounts,
    ClaimTally,
    EvidenceDocument,
    EvidenceRef,
    GradedClaim,
    ReferenceFact,
    TextSpan,
    WebCheckRecord,
    WebWindow,
)

_C = "guide.claims"
_SHA = "b" * 64

# --- a scripted cell: a city guide's two answers, invented ----------------------------------------------------------

#: The calls a scripted grade made, by place: extraction, the truth batch, the ground batch, a web check.
_EXTRACT, _TRUTH, _GROUND, _WEB = range(4)
_CALLS = (
    ClaimCall(role="extract", served_model="judge-m", judge_temperature=0.0),
    ClaimCall(role="truth", served_model="judge-m", judge_temperature=0.0),
    ClaimCall(role="ground", served_model="judge-m", judge_temperature=0.0),
    ClaimCall(role="web", served_model="judge-m", judge_temperature=0.0),
)

#: The evidence a grade numbered: a key fact and a case passage (the same for every arm), and the arm's own page.
_EVIDENCE = (
    EvidenceRef(
        ref="K1",
        provenance="case",
        entity="Varne bridge",
        fact=ReferenceFact(claim="The Varne bridge opened in 1904.", source_url="https://example.org/varne"),
    ),
    EvidenceRef(ref="S1", provenance="case", document="case-notes", span=TextSpan(start=0, end=40)),
    EvidenceRef(ref="S2", provenance="run", document="fetched-page", span=TextSpan(start=0, end=40)),
)
_CASE_DOCUMENT = EvidenceDocument(
    id="case-notes", provenance="case", title="Case notes", text="The Varne bridge opened in 1904 and spans 210 m."
)
_RUN_DOCUMENT = EvidenceDocument(
    id="fetched-page",
    provenance="run",
    title="Port Lerin guide",
    url="https://example.org/lerin",
    text="The Varne bridge opened in 1904 and spans 210 m.",
)

_SEGMENTS = (
    Segment(text="Here is what I found.", unit=None),
    Segment(text="The Varne bridge opened in 1904.", unit=0),
    Segment(text="It spans 210 m.", unit=0),
    Segment(text="The ferry runs hourly.", unit=1),
    Segment(text="The market opens at dawn.", unit=1),
)

_SUPPORTS_CASE = (ClaimCitation(ref="K1", stance="supports"),)
_CONTRADICTS_CASE = (ClaimCitation(ref="K1", stance="contradicts"),)
_GROUNDS_RUN = (ClaimCitation(ref="S2", stance="supports"),)
_GROUNDS_CASE = (ClaimCitation(ref="S1", stance="supports"),)


def _claim(segment: int, **changes: Any) -> GradedClaim:
    """A world claim in ``segment``: asserted, specific, supported by case evidence and grounded by the arm's page."""
    text = _SEGMENTS[segment].text
    fields: dict[str, Any] = {
        "segment": segment,
        "unit": _SEGMENTS[segment].unit,
        "claim": text,
        "excerpt": text,
        "start": 0,
        "end": len(text),
        "scope": "world",
        "specific": True,
        "hedged": False,
        "attributed": False,
        "truth_citations": _SUPPORTS_CASE,
        "ground_citations": _GROUNDS_RUN,
        "verdict": "supported",
        "graded_against": "case",
        "grounded": True,
        "decided_by": _TRUTH,
        **changes,
    }
    return GradedClaim(**fields)


def _false(segment: int, **changes: Any) -> GradedClaim:
    """A claim the case's key fact contradicts, which the arm's page does not support."""
    fields: dict[str, Any] = {
        "truth_citations": _CONTRADICTS_CASE,
        "ground_citations": (),
        "verdict": "contradicted",
        "grounded": False,
        **changes,
    }
    return _claim(segment, **fields)


def _session(segment: int = 0) -> GradedClaim:
    text = _SEGMENTS[segment].text
    return GradedClaim(
        segment=segment,
        unit=_SEGMENTS[segment].unit,
        claim="The answer follows.",
        excerpt=text,
        start=0,
        end=len(text),
        scope="session",
        specific=False,
        hedged=False,
        attributed=False,
    )


def _web_record(place: int, outcome: str) -> WebCheckRecord:
    window = WebWindow(url="https://example.org/a", text="The market opens at dawn every day.")
    return WebCheckRecord(
        claim=place,
        queries=("market opening time",),
        urls=(window.url,),
        windows=(window,),
        outcome=outcome,  # type: ignore[arg-type]
        deciding_url=window.url,
        quote="The market opens at dawn every day.",
        value_span=TextSpan(start=17, end=21),
    )


#: The baseline cell: four asserted specific claims, each supported by the case and grounded by the arm's page, two
#: to a unit, and one session claim in the preamble.
_BASELINE = (_session(), _claim(1), _claim(2), _claim(3), _claim(4))


def _readings(
    claims: tuple[GradedClaim, ...] = _BASELINE,
    *,
    grounded_in: tuple[str, ...] | None = ("run",),
    documents: tuple[EvidenceDocument, ...] | None = (_CASE_DOCUMENT, _RUN_DOCUMENT),
    web: tuple[WebCheckRecord, ...] = (),
    session: dict[str, Any] | None = None,
    calls: tuple[ClaimCall, ...] = _CALLS,
    ground_loss: Any = None,
) -> dict[str, float]:
    """The cell's readings, as the judge phase would store its tally and the ranking path read it."""
    tally = claim_tally(
        _grade(claims, web=web, calls=calls),
        _evidence(documents, session),
        grounded_in=grounded_in,
        ground_loss=ground_loss,
    )
    return claim_readings(_C, tally, grounded=grounded_in is not None)


def _grade(
    claims: tuple[GradedClaim, ...], *, web: tuple[WebCheckRecord, ...] = (), calls: tuple[ClaimCall, ...] = _CALLS
) -> ClaimGrade:
    return ClaimGrade(
        criterion_fingerprint=_SHA,
        claim_evidence_digest=_SHA,
        calls=calls,
        evidence=_EVIDENCE,
        claims=claims,
        web=web,
    )


def _evidence(documents: tuple[EvidenceDocument, ...] | None, session: dict[str, Any] | None = None) -> ClaimEvidence:
    return ClaimEvidence(segmentation="guide-v1", segments=_SEGMENTS, documents=documents, session=session)


def _swap(place: int, claim: GradedClaim) -> tuple[GradedClaim, ...]:
    """The baseline with the claim at ``place`` replaced."""
    return (*_BASELINE[:place], claim, *_BASELINE[place + 1 :])


def _moved(before: dict[str, float], after: dict[str, float]) -> set[str]:
    """The readings that changed, appeared or disappeared, by suffix."""
    return {name.removeprefix(f"{_C}_") for name in before.keys() | after.keys() if before.get(name) != after.get(name)}


_BASE = _readings()


def test_the_baseline_reads_no_false_claim_and_every_specific_claim_grounded() -> None:
    assert _BASE == {
        f"{_C}_{suffix}": value
        for suffix, value in {
            "false_per_unit": 0.0,
            "false_unit_share": 0.0,
            "grounded_share": 1.0,
            "false": 0.0,
            "false_hedged": 0.0,
            "false_attributed": 0.0,
            "false_share": 0.0,
            "claims": 4.0,
            "units": 2.0,
            "hedged_share": 0.0,
            "unverifiable": 0.0,
            "true_ungrounded": 0.0,
            "web_settled_share": 0.0,
            "evidence_conflicts": 0.0,
            "web_checked": 0.0,
            "web_skipped": 0.0,
            "session_claims": 1.0,
            "ground_lost": 0.0,
            "lost": 0.0,
        }.items()
    }
    assert list(_BASE) == list(claim_reading_names(_C, grounded=True))


# --- the built controls: each moves only its own readings ----------------------------------------------------------


def test_a_planted_lie_the_case_contradicts_moves_the_truth_readings_alone() -> None:
    """A canon contradiction with the web off: the claim's truth flips, and its page never supported it."""
    lie = _readings(_swap(1, _false(1)))

    assert _moved(_BASE, lie) == {"false_per_unit", "false_unit_share", "false", "false_share", "grounded_share"}
    assert (lie[f"{_C}_false_per_unit"], lie[f"{_C}_false_unit_share"], lie[f"{_C}_grounded_share"]) == (0.5, 0.5, 0.75)


def test_a_lie_its_own_page_supports_is_false_and_grounded() -> None:
    """False but grounded: R1 catches it, R2 does not move, and neither reading is a composite of the other."""
    lie = _readings(_swap(1, _false(1, ground_citations=_GROUNDS_RUN, grounded=True)))

    assert _moved(_BASE, lie) == {"false_per_unit", "false_unit_share", "false", "false_share"}


def test_a_hedged_lie_is_graded_and_moves_no_asserted_reading() -> None:
    hedged = _readings(_swap(1, _false(1, hedged=True)))

    assert _moved(_BASE, hedged) == {"false_hedged", "hedged_share"}
    assert hedged[f"{_C}_false_hedged"] == 1.0


def test_an_attributed_lie_is_graded_and_moves_no_asserted_reading() -> None:
    """A rumour the text reports without vouching for it."""
    attributed = _readings(_swap(1, _false(1, attributed=True)))

    assert _moved(_BASE, attributed) == {"false_attributed"}


def test_a_sourced_assertion_counts_as_asserted_and_its_lie_moves_the_truth_readings() -> None:
    """'According to the guidebook, X' is the text's own statement backed by a source: asserted, not attributed. The
    readings follow the tag, never the words, so a sourced lie reads exactly as a bare one."""
    sourced = _false(1, claim="According to the guidebook, the Varne bridge opened in 1911.")

    assert _readings(_swap(1, sourced)) == _readings(_swap(1, _false(1)))
    assert "false_per_unit" in _moved(_BASE, _readings(_swap(1, sourced)))


def test_a_bland_set_has_no_grounded_share_and_reads_no_better_on_truth() -> None:
    """General statements a source would not settle: out of R2's population, so the cell carries no R2 at all."""
    bland = _readings((_session(), *(_claim(s, specific=False) for s in (1, 2, 3, 4))))

    assert _moved(_BASE, bland) == {"grounded_share"}
    assert f"{_C}_grounded_share" not in bland


def test_a_repeated_lie_counts_each_time_it_is_heard() -> None:
    """A restatement takes its original's verdict and counts as heard: R1 doubles, and both units hold a lie."""
    once = _readings(_swap(1, _false(1)))
    repeated = _readings((*_swap(1, _false(1))[:3], _false(3, restates=1), _BASELINE[4]))

    assert _moved(once, repeated) == {"false_per_unit", "false_unit_share", "false", "false_share", "grounded_share"}
    assert (repeated[f"{_C}_false_per_unit"], repeated[f"{_C}_false_unit_share"]) == (1.0, 1.0)


def test_an_invented_vivid_detail_no_evidence_covers_is_unverifiable_and_ungrounded() -> None:
    vivid = _readings(
        _swap(
            4,
            _claim(
                4, truth_citations=(), ground_citations=(), verdict="unverifiable", graded_against=None, grounded=False
            ),
        )
    )

    assert _moved(_BASE, vivid) == {"unverifiable", "grounded_share"}


def test_a_doubted_detail_the_web_confirms_is_true_and_still_ungrounded() -> None:
    """True but confirmed only after the fact: no truth reading moves, R2 falls, and the web diagnostics count it."""
    confirmed = _claim(
        4,
        truth_citations=(),
        ground_citations=(),
        grounded=False,
        suspect=True,
        nominated_by="judge",
        web="checked",
        verdict="true_ungrounded",
        graded_against="web",
        decided_by=_WEB,
    )
    readings = _readings(_swap(4, confirmed), web=(_web_record(4, "true"),))

    assert _moved(_BASE, readings) == {"true_ungrounded", "web_checked", "web_settled_share", "grounded_share"}


def test_a_doubted_detail_the_web_contradicts_moves_truth_and_the_web_diagnostics() -> None:
    refuted = _claim(
        4,
        truth_citations=(),
        ground_citations=(),
        grounded=False,
        suspect=True,
        nominated_by="judge",
        web="checked",
        verdict="contradicted",
        graded_against="web",
        decided_by=_WEB,
    )
    readings = _readings(_swap(4, refuted), web=(_web_record(4, "false"),))

    assert _moved(_BASE, readings) == {
        "false_per_unit",
        "false_unit_share",
        "false",
        "false_share",
        "web_checked",
        "web_settled_share",
        "grounded_share",
    }


def test_a_doubted_detail_past_the_web_budget_is_counted_as_skipped() -> None:
    skipped = _claim(
        4,
        truth_citations=(),
        ground_citations=(),
        grounded=False,
        suspect=True,
        nominated_by="judge",
        web="skipped_cap",
        verdict="unverifiable",
        graded_against=None,
    )

    assert _moved(_BASE, _readings(_swap(4, skipped))) == {"web_skipped", "unverifiable", "grounded_share"}


def test_case_evidence_on_both_sides_reads_supported_and_is_counted() -> None:
    conflicted = _claim(1, truth_citations=(*_SUPPORTS_CASE, *_CONTRADICTS_CASE), evidence_conflict=True)

    assert _moved(_BASE, _readings(_swap(1, conflicted))) == {"evidence_conflicts"}


def test_one_lie_in_each_of_two_units_against_two_in_one_unit() -> None:
    """R1 reads the two alike; R1b, which counts a unit once, does not."""
    spread = _readings((_session(), _false(1), _claim(2), _false(3), _claim(4)))
    packed = _readings((_session(), _false(1), _false(2), _claim(3), _claim(4)))

    assert spread[f"{_C}_false_per_unit"] == packed[f"{_C}_false_per_unit"] == 1.0
    assert (spread[f"{_C}_false_unit_share"], packed[f"{_C}_false_unit_share"]) == (1.0, 0.5)
    assert _moved(spread, packed) == {"false_unit_share"}


def test_a_lie_outside_every_unit_counts_per_unit_and_in_no_unit_s_share() -> None:
    """The preamble is no unit: its lie is in R1's numerator, and R1b, which counts units, does not see it."""
    preamble = _readings((_false(0), *_BASELINE[1:]))
    added = _readings((*_BASELINE, _false(0)))

    assert preamble[f"{_C}_false_unit_share"] == 0.0
    assert added[f"{_C}_false_per_unit"] == 0.5
    assert added[f"{_C}_false_unit_share"] == 0.0


def test_a_shuffled_session_moves_no_world_reading() -> None:
    """Claims about the session itself, out of order against its record: R1 and R2 stay where they were, the session
    claims are counted, and the record a host's derived reading checks them against never reaches a world reading."""
    in_order = _readings(session={"order": ["first", "second", "third"]})
    shuffled = _readings(
        (_session(), _session(1), _session(3), *_BASELINE[1:]), session={"order": ["third", "first", "second"]}
    )

    assert in_order == _BASE
    assert _moved(_BASE, shuffled) == {"session_claims"}
    assert shuffled[f"{_C}_session_claims"] == 3.0


def test_no_research_reads_a_grounded_share_of_zero_and_full_research_one() -> None:
    """The same claims and verdicts, two arms: one given the page, one given nothing. Only R2 moves."""
    ungiven = tuple(
        claim if claim.scope == "session" else claim.model_copy(update={"ground_citations": (), "grounded": False})
        for claim in _BASELINE
    )
    no_research = _readings(ungiven, documents=(_CASE_DOCUMENT,))
    nothing_at_all = _readings(ungiven, documents=())

    assert _BASE[f"{_C}_grounded_share"] == 1.0
    assert no_research[f"{_C}_grounded_share"] == nothing_at_all[f"{_C}_grounded_share"] == 0.0
    assert _moved(_BASE, no_research) == {"grounded_share"}


def _ungrounded_by_none(claims: tuple[GradedClaim, ...] = _BASELINE) -> tuple[GradedClaim, ...]:
    """The claims with no groundedness at all, as a verifier writes them when it asks none or loses the step."""
    return tuple(
        claim if claim.scope == "session" else claim.model_copy(update={"ground_citations": (), "grounded": None})
        for claim in claims
    )


def test_no_grounded_in_gives_no_groundedness_reading() -> None:
    readings = _readings(_ungrounded_by_none(), grounded_in=None)

    assert not {f"{_C}_grounded_share", f"{_C}_ground_lost"} & set(readings)
    assert f"{_C}_grounded_share" not in claim_reading_names(_C, grounded=False)
    assert f"{_C}_grounded_share" not in _readings(_ungrounded_by_none(), grounded_in=None, documents=None)


def test_a_criterion_with_grounded_in_refuses_a_cell_whose_documents_are_undeclared() -> None:
    """The verifier refuses the cell too: groundedness reads what the candidate was given, empty when it was nothing."""
    with pytest.raises(ValueError, match="grounded_in reads the documents the candidate was given"):
        _readings(documents=None)


def test_a_lost_ground_step_gives_no_grounded_share_and_every_truth_reading_stands() -> None:
    """The arm's own pages failed to read, so groundedness is lost alone: no R2 (never 0, the no-research reading),
    the loss counted, and the truth readings exactly where they were, a planted lie among them."""
    lost = _readings(_ungrounded_by_none(), ground_loss="unreadable")
    lie_lost = _readings(_ungrounded_by_none(_swap(1, _false(1))), ground_loss="repair_budget_spent")
    lie = _readings(_swap(1, _false(1)))

    assert _moved(_BASE, lost) == {"grounded_share", "ground_lost"}
    assert f"{_C}_grounded_share" not in lost and lost[f"{_C}_ground_lost"] == 1.0
    assert _moved(lie, lie_lost) == {"grounded_share", "ground_lost"}


def test_groundedness_agrees_with_whether_it_was_asked_and_whether_its_step_was_lost() -> None:
    with pytest.raises(ValueError, match="asked its ground step, which was not lost; a lost step is the tally's"):
        _readings(_ungrounded_by_none())
    with pytest.raises(ValueError, match="lost its ground step \\('unreadable'\\)"):
        _readings(ground_loss="unreadable")
    with pytest.raises(ValueError, match="and the criterion asks no groundedness"):
        _readings(_ungrounded_by_none(), grounded_in=None, ground_loss="unreadable")
    with pytest.raises(ValueError, match="asks no groundedness"):
        _readings(
            tuple(
                c if c.scope == "session" else c.model_copy(update={"ground_citations": (), "grounded": False})
                for c in _BASELINE
            ),
            grounded_in=None,
        )


def test_a_tally_that_lost_its_ground_step_counts_no_grounded_claim_and_stores_the_loss_only_when_set() -> None:
    counts = claim_tally(_grade(_BASELINE), _evidence((_CASE_DOCUMENT, _RUN_DOCUMENT)), grounded_in=("run",)).by_judge
    with pytest.raises(ValueError, match="lost its ground step \\('cut_short'\\), which grounds none"):
        ClaimTally(
            criterion_fingerprint=_SHA,
            documents_declared=True,
            units=2,
            units_with_false=0,
            session_claims=1,
            by_judge=counts,
            ground_loss="cut_short",
        )
    lost = claim_tally(
        _grade(_ungrounded_by_none()),
        _evidence((_CASE_DOCUMENT, _RUN_DOCUMENT)),
        grounded_in=("run",),
        ground_loss="unreadable",
    )

    assert ClaimTally.model_validate(lost.model_dump(mode="json")) == lost
    assert "ground_loss" not in claim_tally(
        _grade(_BASELINE), _evidence((_CASE_DOCUMENT, _RUN_DOCUMENT)), grounded_in=("run",)
    ).model_dump(mode="json")


def test_a_key_fact_never_grounds_a_claim_even_under_grounded_in_case() -> None:
    """A summary is grounded in the source it was shown; the case's key facts are never shown to the candidate."""
    by_source = tuple(
        claim if claim.scope == "session" else claim.model_copy(update={"ground_citations": _GROUNDS_CASE})
        for claim in _BASELINE
    )
    by_key = (*by_source[:1], _claim(1, ground_citations=(ClaimCitation(ref="K1", stance="supports"),)), *by_source[2:])

    assert _readings(by_source, grounded_in=("case",))[f"{_C}_grounded_share"] == 1.0
    with pytest.raises(ValueError, match="cites key fact K1 as its ground, and a key fact is no document"):
        _readings(by_key, grounded_in=("case",))


def test_a_claim_s_unit_is_its_segment_s() -> None:
    """R1b counts units off the segments, so a claim naming another unit than its segment's is refused."""
    with pytest.raises(ValueError, match="claim 1 names unit 1, and its segment 1 is unit 0"):
        _readings(_swap(1, _false(1, unit=1)))


def test_case_support_grounds_a_claim_only_where_grounded_in_names_case() -> None:
    """A summary's source is case evidence the candidate was shown: under grounded_in case its support grounds. Under
    grounded_in run, case support decides truth and grounds nothing, and a ground citation of it is refused."""
    by_source = tuple(
        claim if claim.scope == "session" else claim.model_copy(update={"ground_citations": _GROUNDS_CASE})
        for claim in _BASELINE
    )
    case_only = tuple(
        claim if claim.scope == "session" else claim.model_copy(update={"ground_citations": (), "grounded": False})
        for claim in _BASELINE
    )

    assert _readings(by_source, grounded_in=("case",))[f"{_C}_grounded_share"] == 1.0
    assert _readings(case_only, grounded_in=("run",))[f"{_C}_grounded_share"] == 0.0
    with pytest.raises(ValueError, match="cites case evidence as its ground, and the criterion is grounded in run"):
        _readings(by_source, grounded_in=("run",))
    with pytest.raises(ValueError, match="grounded in nothing"):
        _readings(_BASELINE, grounded_in=None)


def test_only_asserted_specific_world_claims_are_in_the_grounded_share() -> None:
    """A hedged, an attributed and a session claim, each grounded by the page or not, leave R2 where it was."""
    ungrounded_hedge = _claim(2, hedged=True, ground_citations=(), grounded=False)
    ungrounded_rumour = _claim(4, attributed=True, ground_citations=(), grounded=False)

    readings = _readings((_session(), _claim(1), ungrounded_hedge, _claim(3), ungrounded_rumour, _session(2)))

    assert readings[f"{_C}_grounded_share"] == 1.0


def test_the_arm_s_own_page_never_decides_truth() -> None:
    """Run pages are never shown to the truth step, so a truth citation of one is a grade that cannot be."""
    with pytest.raises(ValueError, match="cites run evidence for its truth"):
        _readings(_swap(1, _claim(1, truth_citations=(*_SUPPORTS_CASE, ClaimCitation(ref="S2", stance="contradicts")))))


def test_counts_from_two_judge_keys_are_summed_in_one_tally() -> None:
    """A fallback-served retry is one more judge key: the tally holds an entry for each, and every reading sums them."""
    fallback = (*_CALLS, ClaimCall(role="repair", served_model="judge-fallback", judge_temperature=0.0))
    readings = _readings(_swap(1, _false(1, decided_by=4)), calls=fallback)
    tally = claim_tally(
        _grade(_swap(1, _false(1, decided_by=4)), calls=fallback),
        _evidence((_CASE_DOCUMENT, _RUN_DOCUMENT)),
        grounded_in=("run",),
    )

    assert sorted(entry.served_model or "" for entry in tally.by_judge) == ["judge-fallback", "judge-m"]
    assert readings == _readings(_swap(1, _false(1)))


def test_a_lost_cell_carries_only_its_loss() -> None:
    assert claim_readings(_C, "deadline", grounded=True) == {f"{_C}_lost": 1.0}
    assert _BASE[f"{_C}_lost"] == 0.0


def test_a_cell_with_no_unit_and_no_claim_carries_no_rate() -> None:
    tally = ClaimTally(
        criterion_fingerprint=_SHA, documents_declared=True, units=0, units_with_false=0, session_claims=0
    )
    readings = claim_readings(_C, tally, grounded=True)

    assert not {"false_per_unit", "false_unit_share", "grounded_share", "false_share", "hedged_share"} & {
        name.removeprefix(f"{_C}_") for name in readings
    }
    assert readings[f"{_C}_claims"] == 0.0


def test_readings_are_computed_from_a_stored_tally_alone() -> None:
    """What a result stores is enough: a tally read back from its document gives the readings its grade gave."""
    tally = claim_tally(_grade(_swap(1, _false(1))), _evidence((_CASE_DOCUMENT, _RUN_DOCUMENT)), grounded_in=("run",))
    stored = ClaimTally.model_validate(tally.model_dump(mode="json"))

    assert claim_readings(_C, stored, grounded=True) == _readings(_swap(1, _false(1)))
    assert isinstance(stored.by_judge[0], ClaimJudgeCounts)


# --- descriptors -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", claim_reading_names(_C, grounded=True))
def test_every_reading_is_described_with_its_population_and_evidence(name: str) -> None:
    descriptor = describe_claim_reading(name, criterion=_C, unit="answer")

    assert descriptor.family == CLAIMS_FAMILY
    assert descriptor.transferability_class == "judge_mediated"
    assert "Read over " in descriptor.description and "Evidence: " in descriptor.description
    assert descriptor.diagnostic is not claim_reading_ranks(name, criterion=_C)
    assert not descriptor.is_check
    assert "segue" not in descriptor.description.lower()


def test_three_readings_rank_and_say_which_end_is_better() -> None:
    ranked = {
        name.removeprefix(f"{_C}_"): describe_claim_reading(name, criterion=_C)
        for name in claim_reading_names(_C, grounded=True)
        if claim_reading_ranks(name, criterion=_C)
    }

    assert {suffix: d.higher_is_better for suffix, d in ranked.items()} == {
        "false_per_unit": False,
        "false_unit_share": False,
        "grounded_share": True,
    }
    assert ranked["false_per_unit"].value_range is None and ranked["false_per_unit"].nonnegative
    assert "t-path" in ranked["false_per_unit"].description
    assert ranked["false_unit_share"].value_range == ranked["grounded_share"].value_range == (0.0, 1.0)
    assert "reads 0" in ranked["grounded_share"].description


def test_a_unit_s_noun_is_only_in_the_reader_name() -> None:
    answer = describe_claim_reading(f"{_C}_false_per_unit", criterion=_C, unit="answer")
    entry = describe_claim_reading(f"{_C}_false_per_unit", criterion=_C, unit="entry")

    assert (answer.reader_name, entry.reader_name) == (
        f"False claims per answer ({_C})",
        f"False claims per entry ({_C})",
    )
    assert answer.model_dump(exclude={"reader_name"}) == entry.model_dump(exclude={"reader_name"})


def test_two_criteria_s_readings_are_told_apart_by_a_reader() -> None:
    names = [*claim_reading_names("guide.claims", grounded=True), *claim_reading_names("guide.notes", grounded=False)]
    descriptors = [describe_claim_reading(n, criterion=n.split("_")[0]) for n in names]

    assert reader_name_defects(descriptors) == []


def test_a_name_that_is_no_reading_of_its_criterion_is_refused() -> None:
    with pytest.raises(ValueError, match="is no reading of claim criterion 'guide.claims'"):
        describe_claim_reading("guide.claims_score", criterion=_C)
    with pytest.raises(ValueError, match="is no reading"):
        describe_claim_reading("guide.notes_false", criterion=_C)


# --- the family ------------------------------------------------------------------------------------------------------


def test_claims_is_an_engine_family_a_judge_grades() -> None:
    assert ENGINE_FAMILIES[CLAIMS_FAMILY].graded_by == "judge"
    assert CLAIMS_FAMILY not in CODE_GRADED_FAMILIES
    descriptor = describe_claim_reading(f"{_C}_false_per_unit", criterion=_C)
    assert not is_code_graded(descriptor, MeasureRegistry([]))


def test_a_host_may_not_declare_a_family_named_claims() -> None:
    with pytest.raises(MeasureRegistrationError, match="family claims is one of the engine's own"):
        MeasureRegistry([], families=(MeasureFamily(name="claims", graded_by="code", description="x"),))


# --- a criterion's names against the host's measures ------------------------------------------------------------------


def _criterion(name: str = "rules.claims") -> ClaimCriterion:
    return ClaimCriterion(
        name=name,
        grounded_in=("run",),
        pages=PageBudget(per_document_chars=8000, total_chars=24000, window_chars=1200),
        repairs=RepairBudget(max_items_per_cell=10),
        judge="builtin",
    )


def _colliding_profile(measure: str) -> Any:
    collider = MetricDescriptor(
        name=measure,
        reader_name="A host's own count",
        data_type="numeric",
        family="mechanical",
        transferability_class="mechanical",
        attribution_scope="end_to_end",
        description="A count the host takes, for the test.",
        higher_is_better=False,
        nonnegative=True,
    )
    return dataclasses.replace(
        toyhost_profile(),
        measures=MeasureRegistry((*TOYHOST_MEASURES, collider), families=(TOYHOST_EXTRACTION_FAMILY,)),
    )


@pytest.mark.parametrize("measure", ["rules.claims", "rules.claims_false", "rules.claims_anything"])
def test_a_criterion_whose_name_a_host_measure_holds_is_refused_where_it_is_admitted(measure: str) -> None:
    template = toyhost_template().model_copy(update={"claim_criteria": [_criterion()]})

    with pytest.raises(ValidationFailedError, match=f"claim criterion 'rules.claims' owns .*'{measure}'"):
        admit_template(
            template,
            profile=_colliding_profile(measure),
            checks=TOYHOST_TEMPLATE_CHECKS,
            active_judge_config=lambda _: None,
        )
    assert (
        admit_template(
            template, profile=toyhost_profile(), checks=TOYHOST_TEMPLATE_CHECKS, active_judge_config=lambda _: None
        ).claim_criteria
        == template.claim_criteria
    )


def test_a_measure_that_only_begins_like_a_criterion_is_its_own() -> None:
    template = toyhost_template().model_copy(update={"claim_criteria": [_criterion("rules.claim")]})

    admit_template(
        template,
        profile=_colliding_profile("rules.claims_false"),
        checks=TOYHOST_TEMPLATE_CHECKS,
        active_judge_config=lambda _: None,
    )


async def test_a_launch_refuses_a_criterion_a_measure_the_host_added_since_now_holds() -> None:
    storage, store = memory_storage()
    host, _ = toyhost_launch_host(storage=storage, profile=_colliding_profile("rules.claims_false"))
    template = toyhost_template().model_copy(update={"claim_criteria": [_criterion()]})
    store.upsert(template.to_dict())

    with pytest.raises(ValidationFailedError, match="host describes the measure 'rules.claims_false'"):
        await quote_launch(
            host,
            template_id=template.id,
            subject_id=TOYHOST_SUBJECT.subject_id,
            models=[RUN_MODELS[0]],
            scope_id=TOYHOST_SCOPE,
        )


def test_an_update_writing_a_criterion_a_host_measure_holds_is_refused_and_one_leaving_it_is_not() -> None:
    """Scoped to the write, as the update door's other gates are: a stored template whose criterion a measure the
    host added since now holds stays editable in its other fields (its launch refuses it)."""
    storage, store = memory_storage()
    host, _ = toyhost_launch_host(storage=storage, profile=_colliding_profile("rules.claims_false"))
    template = toyhost_template()
    store.upsert(template.model_copy(update={"claim_criteria": [_criterion("rules.other")]}).to_dict())

    with pytest.raises(ValidationFailedError, match="host describes the measure 'rules.claims_false'"):
        update_template(
            host.eval_host,
            template.id,
            template.scope_id,
            {"claim_criteria": [_criterion().model_dump(mode="json")]},
        )
    store.upsert(template.model_copy(update={"claim_criteria": [_criterion()]}).to_dict())
    edited = update_template(host.eval_host, template.id, template.scope_id, {"intent": "Read the form carefully."})
    assert edited.intent == "Read the form carefully."
