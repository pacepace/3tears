"""The stored shapes of claim grading (#1287, C1): what a template, a trace and a result may carry, and what they refuse.

- **What carries none of it stores, digests and fingerprints exactly as before.** Pinned first, on the build before
  the shapes existed: a template's stored document and both its digests, a judge evidence's fingerprint, and a
  result's and a trace's stored documents, each less the minor stamp every minor moves.
- **What carries it round-trips**, is stored under its own keys, and moves the template's digests as measured content,
  while a unit's noun is presentation and moves neither.
- **Each shape refuses what it cannot mean**, by name: units not numbered 0..n-1, a provenance, scope, verdict or loss
  this build does not know, a session claim graded for truth, a verdict its evidence or web check does not give, a
  criterion name another name holds, and every size cap (documents, windows, the whole claims payload).
- **A cell with claim evidence and no judge evidence is a valid trace**: a claims-only cell of an unjudged kind.
- **A launch refuses a template a newer build stored with a field this build does not know**, naming it, and a kind
  that never renders claim evidence may declare it ignores the criteria.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from packages.evals.tests.clock_support import FIXED_AT
from packages.evals.tests.factories import make_eval_result, make_eval_trace, make_template, memory_storage
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.ops_support import TOYHOST_SCOPE, TOYHOST_SUBJECT
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.run import quote_launch
from threetears.evals.run.definition_digests import measured_template_digest, template_digest
from threetears.evals.run.launch import TEMPLATE_FIELDS
from threetears.evals.schema import CLAIM_LOSSES, ClaimCriterion, PageBudget, RepairBudget, WebBudget
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.schema.models import (
    CLAIM_DOCUMENT_BYTES,
    CLAIM_TRACE_BYTES,
    CLAIM_WINDOW_BYTES,
    TEMPLATE_PRESENTATION_FIELDS,
    ClaimCall,
    ClaimCitation,
    ClaimEvidence,
    ClaimGrade,
    ClaimJudgeCounts,
    ClaimTally,
    EvalResult,
    EvalTemplate,
    EvalTrace,
    EvidenceDocument,
    EvidenceRef,
    GradedClaim,
    JudgedArtifact,
    JudgeEvidence,
    ReferenceFact,
    Segment,
    TextSpan,
    WebCheckRecord,
    WebWindow,
    fingerprint_judged_output,
)
from threetears.evals.schema.versioning import CORE_SCHEMA_MINOR, DURABLE_SCHEMA_MINOR

#: Taken at ``d1406de56``, the build before claim grading's stored shapes: what a template, an evidence, a result and
#: a trace carrying none of them stored, digested and fingerprinted as. Each must hold after the shapes land.
BEFORE_CLAIMS: dict[str, str] = {
    "template document": "521be221c821e4609d21a35d806fb2152e6f99e2c996a5142de54b935b739153",
    "template_digest": "5792093b03b534d9c03a7bd9324530fce57078062b3fd96083e3242ac2b90c68",
    "measured_template_digest": "c06edf75f254582dfe92d90f1ab694bff90966f147f77fd3e7d96c286a42bfc6",
    "toy template document": "167c4f4759493f4d672fde7b4d6199eeab06593b77e7b89a191300ca79751aa4",
    "toy template_digest": "c7a5125e050278830af521c5e4ff3ec3733dc4107f91b2f161d87692dfd67c0b",
    "toy measured_template_digest": "012be363cc55443558089e29a4048801c77ac185c3027508314c6b81e514fd9b",
    "fingerprint_judged_output": "98b3e32cccb43e7bc6a5e8aa1494fb6da47857dc83784bed7f427c2429e8e95c",
    "trace document": "81b795bc7eb75d906e9e5ab45fd8f6d6cb558192b0f877ed308bab7bcef6485b",
    "result document": "08eb3555a8aa6a350bd2b3484d20fafd9d85dcc8c01d054692378a2571cf3235",
}


def _template() -> EvalTemplate:
    """A template with a rubric, fixed in every field a digest or document reads."""
    return make_template(id="tpl-claims-pin", created_at=FIXED_AT, updated_at=FIXED_AT)


def _evidence() -> JudgeEvidence:
    return JudgeEvidence(
        subject="A research assistant.", case_material="  The source, as given.\n", artifact="The answer.\n"
    )


def _trace() -> EvalTrace:
    return make_eval_trace(
        judge_evidence=_evidence(),
        judged_artifact=JudgedArtifact.DOCUMENT,
    )


def _result() -> EvalResult:
    return make_eval_result(id="res-1", scored_at=FIXED_AT)


def _stored(document: dict[str, object]) -> str:
    """The digest of a stored document less its minor stamp, which every minor moves on every document by design."""
    return canonical_digest({key: value for key, value in document.items() if key != "schema_minor"})


def _taken() -> dict[str, str]:
    template, toy = _template(), toyhost_template()
    return {
        "template document": _stored(template.to_dict()),
        "template_digest": template_digest(template),
        "measured_template_digest": measured_template_digest(template),
        "toy template document": _stored(toy.to_dict()),
        "toy template_digest": template_digest(toy),
        "toy measured_template_digest": measured_template_digest(toy),
        "fingerprint_judged_output": fingerprint_judged_output(_evidence()),
        "trace document": _stored(_trace().to_dict()),
        "result document": _stored(_result().to_dict()),
    }


def test_what_carries_no_claim_shape_stores_digests_and_fingerprints_as_before() -> None:
    assert _taken() == BEFORE_CLAIMS


# --- the shapes, built ---------------------------------------------------------------------------------------------

_SHA = "a" * 64


def _criterion(**changes: Any) -> ClaimCriterion:
    fields: dict[str, Any] = {
        "name": "rules.claims",
        "guidance": "A rule's number is a specific claim.",
        "grounded_in": ("run",),
        "web": WebBudget(
            max_queries_per_claim=2, max_checked_per_cell=5, results_per_query=3, page_chars=4000, window_chars=800
        ),
        "pages": PageBudget(per_document_chars=8000, total_chars=24000, window_chars=1200),
        "repairs": RepairBudget(max_items_per_cell=10),
        "judge": "builtin",
        **changes,
    }
    return ClaimCriterion(**fields)


def _dims(*names: str) -> list[Any]:
    base = make_template().rubric[0]
    return [base.model_copy(update={"name": name}) for name in names]


def _evidence_of_claims(**changes: Any) -> ClaimEvidence:
    fields: dict[str, Any] = {
        "segmentation": "answer-v1",
        "segments": (
            Segment(text="A short summary.", unit=None),
            Segment(text="Grappling ends when the grappler is incapacitated.  ", unit=0, tags={"entity": "grapple"}),
        ),
        "documents": (
            EvidenceDocument(
                id="srd-grapple",
                provenance="run",
                title="Grappled",
                url="https://example.org/srd/grappled",
                text="The condition ends if the grappler is incapacitated.",
            ),
        ),
        "session": {"rolls": [14]},
        **changes,
    }
    return ClaimEvidence(**fields)


def _world_claim(**changes: Any) -> GradedClaim:
    excerpt = "Grappling ends when the grappler is incapacitated."
    fields: dict[str, Any] = {
        "segment": 1,
        "unit": 0,
        "claim": "A grapple ends when the grappler is incapacitated.",
        "excerpt": excerpt,
        "start": 0,
        "end": len(excerpt),
        "scope": "world",
        "specific": True,
        "hedged": False,
        "attributed": False,
        "truth_citations": (
            ClaimCitation(
                ref="S1", stance="supports", quote="The condition ends", value_span=TextSpan(start=4, end=13)
            ),
        ),
        "ground_citations": (ClaimCitation(ref="S1", stance="supports"),),
        "verdict": "supported",
        "graded_against": "case",
        "grounded": True,
        "decided_by": 1,
        **changes,
    }
    return GradedClaim(**fields)


def _session_claim(**changes: Any) -> GradedClaim:
    fields: dict[str, Any] = {
        "segment": 0,
        "unit": None,
        "claim": "This is a summary.",
        "excerpt": "A short summary",
        "start": 0,
        "end": 15,
        "scope": "session",
        "specific": False,
        "hedged": False,
        "attributed": False,
        **changes,
    }
    return GradedClaim(**fields)


def _grade(**changes: Any) -> ClaimGrade:
    fields: dict[str, Any] = {
        "criterion_fingerprint": _SHA,
        "claim_evidence_digest": _SHA,
        "calls": (ClaimCall(role="extract", served_model="judge-m", judge_temperature=0.0), ClaimCall(role="truth")),
        "evidence": (
            EvidenceRef(
                ref="K1",
                provenance="case",
                entity="Grappled",
                fact=ReferenceFact(claim="It ends.", source_url="https://example.org"),
            ),
            EvidenceRef(ref="S1", provenance="run", document="srd-grapple", span=TextSpan(start=0, end=52)),
        ),
        "claims": (_session_claim(), _world_claim()),
        **changes,
    }
    return ClaimGrade(**fields)


def _counts(**changes: Any) -> ClaimJudgeCounts:
    fields: dict[str, Any] = {
        "served_model": "judge-m",
        "judge_temperature": 0.0,
        "judge_sent_settings": None,
        **dict.fromkeys(
            (
                "asserted_contradicted",
                "hedged_contradicted",
                "attributed_contradicted",
                "unverifiable",
                "true_ungrounded",
                "web_decided",
                "evidence_conflicts",
                "web_checked",
                "web_skipped",
            ),
            0,
        ),
        "claims": 1,
        "asserted": 1,
        "asserted_specific": 1,
        "grounded_specific": 1,
        **changes,
    }
    return ClaimJudgeCounts(**fields)


def _tally(**changes: Any) -> ClaimTally:
    fields: dict[str, Any] = {
        "criterion_fingerprint": _SHA,
        "documents_declared": True,
        "units": 1,
        "units_with_false": 0,
        "session_claims": 1,
        "by_judge": (_counts(),),
        **changes,
    }
    return ClaimTally(**fields)


def _claims_trace(**changes: Any) -> EvalTrace:
    fields: dict[str, Any] = {
        "claim_evidence": _evidence_of_claims(),
        "claim_grades": {"rules.claims": _grade()},
        **changes,
    }
    return make_eval_trace(**fields)


# --- what carries it ---------------------------------------------------------------------------------------------


def test_a_template_a_trace_and_a_result_carrying_claim_shapes_round_trip_under_their_own_keys() -> None:
    template = _template().model_copy(
        update={"claim_criteria": [_criterion()], "claim_unit_names": {"rules.claims": "answer"}}
    )
    template = EvalTemplate.model_validate(template.model_dump())
    trace = _claims_trace()
    result = make_eval_result(
        id="res-1",
        scored_at=FIXED_AT,
        claim_tallies={"rules.claims": _tally()},
        claim_failures={"rules.other": "deadline"},
    )

    for document in (template, trace, result):
        assert type(document).from_dict(document.to_dict()) == document
    assert {"claim_criteria", "claim_unit_names"} <= set(template.to_dict())
    assert {"claim_evidence", "claim_grades"} <= set(trace.to_dict())
    assert {"claim_tallies", "claim_failures"} <= set(result.to_dict())
    assert trace.to_dict()["claim_evidence"]["segments"][1]["text"].endswith("  "), "segment text is kept as written"


def test_criteria_are_measured_content_and_a_unit_noun_is_presentation() -> None:
    plain = _template()
    graded = plain.model_copy(update={"claim_criteria": [_criterion()]})
    named = graded.model_copy(update={"claim_unit_names": {"rules.claims": "answer"}})

    assert measured_template_digest(graded) != measured_template_digest(plain)
    assert template_digest(graded) != template_digest(plain)
    assert (measured_template_digest(named), template_digest(named)) == (
        measured_template_digest(graded),
        template_digest(graded),
    )
    assert "claim_unit_names" in TEMPLATE_PRESENTATION_FIELDS


# --- the criterion -----------------------------------------------------------------------------------------------


def test_grounded_in_is_a_set_of_the_two_provenances() -> None:
    assert _criterion(grounded_in=("run", "case", "run")).grounded_in == ("case", "run")
    assert _criterion(grounded_in=None).grounded_in is None
    with pytest.raises(ValidationError, match="grounded_in names at least one provenance"):
        _criterion(grounded_in=())
    with pytest.raises(ValidationError, match="literal_error"):
        _criterion(grounded_in=("web",))


def test_a_criterion_is_namespaced_and_judged_by_the_built_in_prompts() -> None:
    with pytest.raises(ValidationError, match="namespaced"):
        _criterion(name="claims")
    with pytest.raises(ValidationError, match="declare judge 'builtin'"):
        _criterion(judge="configured")


@pytest.mark.parametrize(
    ("budget", "refusal"),
    [
        (
            dict(per_document_chars=100, total_chars=50, window_chars=10),
            "window_chars <= per_document_chars <= total_chars",
        ),
        (
            dict(per_document_chars=100, total_chars=500, window_chars=200),
            "window_chars <= per_document_chars <= total_chars",
        ),
    ],
)
def test_a_page_budget_nests(budget: dict[str, int], refusal: str) -> None:
    with pytest.raises(ValidationError, match=refusal):
        PageBudget(**budget)


def test_a_web_budget_whose_windows_could_outgrow_their_share_of_the_trace_is_refused_by_the_cap_s_name() -> None:
    fits = CLAIM_WINDOW_BYTES // 4
    WebBudget(max_queries_per_claim=1, max_checked_per_cell=1, results_per_query=1, page_chars=fits, window_chars=1)
    with pytest.raises(ValidationError, match="CLAIM_WINDOW_BYTES"):
        WebBudget(
            max_queries_per_claim=1, max_checked_per_cell=1, results_per_query=1, page_chars=fits + 1, window_chars=1
        )
    with pytest.raises(ValidationError, match="window_chars"):
        WebBudget(max_queries_per_claim=1, max_checked_per_cell=1, results_per_query=1, page_chars=10, window_chars=11)


@pytest.mark.parametrize(
    ("changes", "refusal"),
    [
        pytest.param({"claim_criteria": [_criterion(), _criterion()]}, "more than once", id="two-criteria-one-name"),
        pytest.param(
            {"claim_criteria": [_criterion(name="conversation.tone")]}, "owns its name", id="a-dimension's-name"
        ),
        pytest.param(
            {"claim_criteria": [_criterion(name="conversation.tone")], "rubric": _dims("conversation.tone_x")},
            "conversation.tone_x",
            id="a-dimension-named-like-a-reading",
        ),
        pytest.param(
            {"claim_criteria": [_criterion(), _criterion(name="rules.claims_web")]},
            "rules.claims_web",
            id="another-criterion's-readings",
        ),
        pytest.param(
            {"claim_criteria": [_criterion()], "claim_unit_names": {"rules.other": "stop"}},
            "not a claim criterion",
            id="a-noun-for-no-criterion",
        ),
        pytest.param(
            {"claim_criteria": [_criterion()], "claim_unit_names": {"rules.claims": "x" * 33}},
            "holds at most 32",
            id="a-long-noun",
        ),
    ],
)
def test_a_criterion_owns_its_name_and_every_reading_name_in_the_template(
    changes: dict[str, Any], refusal: str
) -> None:
    with pytest.raises(ValidationError, match=refusal):
        make_template(**changes)


def test_a_name_that_only_begins_like_a_criterion_s_is_its_own() -> None:
    """The control for the reading-name refusal: only a name continuing the criterion's with ``_`` is its readings'."""
    template = make_template(claim_criteria=[_criterion(name="conversation.to")])

    assert [dim.name for dim in template.rubric] == ["conversation.tone"]


# --- the evidence ------------------------------------------------------------------------------------------------


def test_units_are_numbered_from_zero_each_at_least_once() -> None:
    repeated = (Segment(text="a", unit=0), Segment(text="b", unit=0), Segment(text="c", unit=1), Segment(text="d"))
    assert _evidence_of_claims(segments=repeated).segments == repeated
    with pytest.raises(ValidationError, match="numbered 0..n-1"):
        _evidence_of_claims(segments=(Segment(text="a", unit=1),))
    with pytest.raises(ValidationError, match="numbered 0..n-1"):
        _evidence_of_claims(segments=(Segment(text="a", unit=0), Segment(text="b", unit=2)))
    with pytest.raises(ValidationError, match="greater_than_equal"):
        Segment(text="a", unit=-1)
    with pytest.raises(ValidationError, match="blank"):
        Segment(text="   ")


def test_documents_are_case_or_run_unique_by_id_and_capped_by_name() -> None:
    document = _evidence_of_claims().documents[0]  # type: ignore[index]
    assert _evidence_of_claims(documents=()).documents == ()
    assert _evidence_of_claims(documents=None).documents is None
    with pytest.raises(ValidationError, match="literal_error"):
        EvidenceDocument(id="x", provenance="web", title="t", text="t")
    with pytest.raises(ValidationError, match="more than once"):
        _evidence_of_claims(documents=(document, document))
    big = document.model_copy(update={"text": "x" * CLAIM_DOCUMENT_BYTES})
    with pytest.raises(ValidationError, match="CLAIM_DOCUMENT_BYTES"):
        _evidence_of_claims(documents=(big,))


def test_a_trace_of_claim_evidence_and_no_judge_evidence_is_a_claims_only_cell() -> None:
    trace = make_eval_trace(claim_evidence=_evidence_of_claims())

    assert trace.judge_evidence is None and trace.judged_artifact is None
    assert EvalTrace.from_dict(trace.to_dict()) == trace


def test_a_trace_holds_grades_only_of_the_evidence_it_holds() -> None:
    with pytest.raises(ValidationError, match="only beside the claim evidence"):
        make_eval_trace(claim_grades={"rules.claims": _grade()})
    with pytest.raises(ValidationError, match="not its segment's text"):
        _claims_trace(claim_grades={"rules.claims": _grade(claims=(_session_claim(excerpt="A short summarx"),))})
    with pytest.raises(ValidationError, match="reads unit 0, and its segment is in unit None"):
        _claims_trace(claim_grades={"rules.claims": _grade(claims=(_session_claim(unit=0),))})
    with pytest.raises(ValidationError, match="names segment 9"):
        _claims_trace(claim_grades={"rules.claims": _grade(claims=(_session_claim(segment=9),))})


def test_a_trace_s_claims_payload_over_its_cap_is_refused_by_name() -> None:
    piece = "x" * (CLAIM_DOCUMENT_BYTES // 2 - 1000)
    documents = tuple(
        EvidenceDocument(id=f"d{i}", provenance="case", title="t", text=piece[: len(piece) // 2]) for i in range(2)
    )
    evidence = _evidence_of_claims(documents=documents)
    windows = tuple(WebWindow(url=f"https://example.org/{i}", text="y" * 400_000) for i in range(7))
    record = WebCheckRecord(claim=1, queries=("q",), windows=windows, outcome="inconclusive")
    nominated = _world_claim(
        truth_citations=(),
        verdict="unverifiable",
        graded_against=None,
        suspect=True,
        nominated_by="judge",
        web="checked",
    )
    grade = _grade(claims=(_session_claim(), nominated), web=(record,))
    with pytest.raises(ValidationError, match="CLAIM_TRACE_BYTES"):
        make_eval_trace(claim_evidence=evidence, claim_grades={"rules.claims": grade})
    assert CLAIM_TRACE_BYTES == 3 * 1024 * 1024


# --- the grade ---------------------------------------------------------------------------------------------------


def test_scope_is_world_or_session_and_a_session_claim_is_graded_by_no_truth_step() -> None:
    with pytest.raises(ValidationError, match="literal_error"):
        _session_claim(scope="meta")
    with pytest.raises(ValidationError, match="graded by no truth, ground or web step"):
        _session_claim(verdict="supported", graded_against="case", decided_by=0)
    with pytest.raises(ValidationError, match="graded by no truth, ground or web step"):
        _session_claim(web="skipped_cap")


@pytest.mark.parametrize(
    ("changes", "refusal"),
    [
        pytest.param({"verdict": None}, "has a verdict and the call", id="no-verdict"),
        pytest.param({"decided_by": None}, "has a verdict and the call", id="no-decider"),
        pytest.param({"verdict": "supported", "graded_against": "web"}, "graded_against", id="supported-by-the-web"),
        pytest.param({"verdict": "true_ungrounded", "graded_against": "case"}, "graded_against", id="true-by-the-case"),
        pytest.param(
            {"verdict": "unverifiable", "graded_against": "case"}, "graded_against", id="unverifiable-decided"
        ),
        pytest.param({"truth_citations": ()}, "cites none of", id="case-without-a-citation"),
        pytest.param({"verdict": "contradicted", "evidence_conflict": True}, "conflict reads supported", id="conflict"),
        pytest.param({"web": "skipped_cap"}, "nobody nominated", id="web-unnominated"),
        pytest.param({"nominated_by": "judge"}, "did not suspect", id="nominated-unsuspected"),
        pytest.param({"nominated_by": "run", "suspect": True}, "literal_error", id="run-pages-never-nominate"),
        pytest.param(
            {"ground_citations": (ClaimCitation(ref="S1", stance="contradicts"),)},
            "only supports",
            id="ground-contradicts",
        ),
        pytest.param({"ground_citations": ()}, "cites no document", id="grounded-uncited"),
        pytest.param({"end": 10}, "excerpt runs", id="offsets"),
        pytest.param({"verdict": "false"}, "literal_error", id="unknown-verdict"),
    ],
)
def test_a_world_claim_s_verdict_is_what_its_evidence_gives(changes: dict[str, Any], refusal: str) -> None:
    with pytest.raises(ValidationError, match=refusal):
        _world_claim(**changes)


def test_a_citation_s_spans_lie_inside_its_quote_and_its_quote_is_one_sentence_long() -> None:
    with pytest.raises(ValidationError, match="quotes nothing"):
        ClaimCitation(ref="S1", stance="supports", value_span=TextSpan(start=0, end=1))
    with pytest.raises(ValidationError, match="past its"):
        ClaimCitation(ref="S1", stance="supports", quote="short", value_span=TextSpan(start=0, end=9))
    with pytest.raises(ValidationError, match="string_too_long"):
        ClaimCitation(ref="S1", stance="supports", quote="q" * 301)
    with pytest.raises(ValidationError, match="ends past where it starts"):
        TextSpan(start=3, end=3)


def test_an_evidence_ref_is_a_passage_or_a_case_key_fact() -> None:
    fact = ReferenceFact(claim="c", source_url="https://example.org")
    with pytest.raises(ValidationError, match="a passage .document and span. or a key fact"):
        EvidenceRef(ref="S1", provenance="run", document="d")
    with pytest.raises(ValidationError, match="a passage .document and span. or a key fact"):
        EvidenceRef(ref="S1", provenance="run", document="d", span=TextSpan(start=0, end=1), entity="e", fact=fact)
    with pytest.raises(ValidationError, match="is case evidence"):
        EvidenceRef(ref="K1", provenance="run", entity="e", fact=fact)


def test_a_grade_s_claims_cite_restate_and_check_only_what_it_holds() -> None:
    with pytest.raises(ValidationError, match="numbers 'S1' more than once"):
        _grade(evidence=(*_grade().evidence, _grade().evidence[1]))
    with pytest.raises(ValidationError, match="cites 'S9'"):
        _grade(claims=(_world_claim(truth_citations=(ClaimCitation(ref="S9", stance="supports"),)),))
    with pytest.raises(ValidationError, match="decided by call 5"):
        _grade(claims=(_world_claim(decided_by=5),))
    with pytest.raises(ValidationError, match="no earlier claim of its scope"):
        _grade(claims=(_session_claim(), _world_claim(restates=0)))
    with pytest.raises(ValidationError, match="no earlier claim of its scope"):
        _grade(claims=(_world_claim(restates=0),))
    restated = _grade(claims=(_world_claim(), _world_claim(restates=0)))
    assert restated.claims[1].restates == 0


def _nominated(**changes: Any) -> GradedClaim:
    fields: dict[str, Any] = {
        "truth_citations": (),
        "suspect": True,
        "nominated_by": "judge",
        "web": "checked",
        "verdict": "contradicted",
        "graded_against": "web",
        **changes,
    }
    return _world_claim(**fields)


def _record(**changes: Any) -> WebCheckRecord:
    fields: dict[str, Any] = {
        "claim": 0,
        "queries": ("grappled condition ends",),
        "urls": ("https://example.org/a",),
        "windows": (WebWindow(url="https://example.org/a", text="It ends when the grappler is incapacitated."),),
        "outcome": "false",
        "deciding_url": "https://example.org/a",
        "quote": "It ends when the grappler is incapacitated.",
        "value_span": TextSpan(start=13, end=43),
        **changes,
    }
    return WebCheckRecord(**fields)


def test_a_web_check_decides_by_a_quote_from_a_window_it_read_and_fixes_the_claim_s_verdict() -> None:
    assert _grade(claims=(_nominated(),), web=(_record(),)).web[0].outcome == "false"
    assert _grade(claims=(_nominated(verdict="true_ungrounded"),), web=(_record(outcome="true"),))
    inconclusive = _record(outcome="inconclusive", deciding_url=None, quote=None, value_span=None)
    assert _grade(claims=(_nominated(verdict="unverifiable", graded_against=None),), web=(inconclusive,))
    with pytest.raises(ValidationError, match="found 'true', and the claim reads 'contradicted'"):
        _grade(claims=(_nominated(),), web=(_record(outcome="true"),))
    with pytest.raises(ValidationError, match="names its deciding url"):
        _record(quote=None)
    with pytest.raises(ValidationError, match="no window it read"):
        _record(deciding_url="https://example.org/elsewhere")
    with pytest.raises(ValidationError, match="names no deciding page"):
        _record(outcome="inconclusive")
    with pytest.raises(ValidationError, match="literal_error"):
        _record(outcome="maybe")
    with pytest.raises(ValidationError, match="its claims read checked"):
        _grade(claims=(_nominated(),))
    with pytest.raises(ValidationError, match="more than once"):
        _grade(claims=(_nominated(),), web=(_record(), _record()))


# --- the result --------------------------------------------------------------------------------------------------


def test_a_tally_s_counts_nest_and_hold_one_entry_per_judge() -> None:
    with pytest.raises(ValidationError, match="units with a false claim, of 1 units"):
        _tally(units_with_false=2)
    with pytest.raises(ValidationError, match="one entry per judge key"):
        _tally(by_judge=(_counts(), _counts()))
    assert _tally(by_judge=(_counts(), _counts(judge_temperature=1.0))).by_judge[1].judge_temperature == 1.0
    with pytest.raises(ValidationError, match="do not nest: asserted"):
        _counts(asserted=2)
    with pytest.raises(ValidationError, match="grounded_specific"):
        _counts(grounded_specific=2)


def test_a_criterion_is_tallied_or_lost_never_both_and_a_loss_is_named() -> None:
    assert set(CLAIM_LOSSES) == {
        "cut_short",
        "unreadable",
        "repair_budget_spent",
        "port_error",
        "cost_cap",
        "deadline",
        "evidence_over_cap",
        "trace_lost",
    }
    with pytest.raises(ValidationError, match="both tallied and lost"):
        make_eval_result(claim_tallies={"rules.claims": _tally()}, claim_failures={"rules.claims": "trace_lost"})
    with pytest.raises(ValidationError, match="'gremlins' is not a claim loss this build knows"):
        make_eval_result(claim_failures={"rules.claims": "gremlins"})


def test_a_loss_a_newer_build_added_is_read_from_its_document_and_written_back() -> None:
    document = {
        **make_eval_result(id="res-1", scored_at=FIXED_AT).to_dict(),
        "schema_minor": CORE_SCHEMA_MINOR + 1,
        "claim_failures": {"rules.claims": "gremlins"},
    }

    assert EvalResult.from_dict(document).to_dict() == document


# --- the launch --------------------------------------------------------------------------------------------------


async def test_a_launch_refuses_a_template_a_newer_build_stored_with_a_field_this_build_does_not_know() -> None:
    storage, store = memory_storage()
    host, _ = toyhost_launch_host(storage=storage)
    template = toyhost_template()
    stored = {
        **template.to_dict(),
        "schema_minor": DURABLE_SCHEMA_MINOR + 1,
        "claim_criteria_v2": [{"name": "rules.claims"}],
    }
    store.upsert(stored)
    read = EvalTemplate.from_dict(stored)

    assert read.kept_field_paths() == ("claim_criteria_v2",)
    with pytest.raises(ValidationFailedError, match="states claim_criteria_v2, which this build does not know"):
        await quote_launch(
            host,
            template_id=template.id,
            subject_id=TOYHOST_SUBJECT.subject_id,
            models=[RUN_MODELS[0]],
            scope_id=TOYHOST_SCOPE,
        )


def test_a_kept_field_is_named_at_its_depth() -> None:
    document = _template().to_dict()
    stored = {**document, "schema_minor": DURABLE_SCHEMA_MINOR + 1}
    stored["rubric"] = [{**document["rubric"][0], "calibrated_by": "a newer build"}]

    assert EvalTemplate.from_dict(stored).kept_field_paths() == ("rubric[0].calibrated_by",)
    assert _template().kept_field_paths() == ()


def test_claim_criteria_is_a_template_field_a_kind_may_declare_it_ignores() -> None:
    assert "claim_criteria" in TEMPLATE_FIELDS


def test_the_stored_sizes_are_in_bytes_of_utf8_json() -> None:
    """The caps read what a store holds: a segment of four-byte characters counts four bytes each."""
    document = EvidenceDocument(id="d", provenance="case", title="t", text="\U0001f600" * (CLAIM_DOCUMENT_BYTES // 4))
    with pytest.raises(ValidationError, match="CLAIM_DOCUMENT_BYTES"):
        _evidence_of_claims(documents=(document,))
    assert len(json.dumps("\U0001f600", ensure_ascii=False).encode()) == 6
