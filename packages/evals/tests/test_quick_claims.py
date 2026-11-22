"""``Claims`` on the quick path: an eval graded by the claims in its answers, through the engine's own claims grader.

Every run here goes through ``Eval(...).run`` over a scripted grader that answers each call by the reply schema it is
sent and reads its answer off the prompt, as a model would: it lists each segment as one claim, and finds a claim
supported where a shown passage holds the same sentence, contradicted where a shown sentence on the same subject says
otherwise, and doubts a claim nothing shown covers. So every verdict below is a function of what the quick path
rendered and the engine forwarded, and nothing is spent.

An eval graded by ``Claims`` alone is judged: its template declares the criterion, its kind seats the judge, and
each answer's claims are graded on the grader's model. The case's own documents decide truth, read off a case field or
a callable; what the candidate read rides beside its answer on ``Answer.documents`` and grounds its claims, never
decides their truth, and is never cut for room the case's documents need. An answer is one unit unless a cut says
otherwise. A doubted claim is checked on the web through the search port given, and a page reads every claim reading
and its caveats as the engine ranks them. Readings of your own (``readers``) read each cell's frozen claims and the
run's own record (``session``), are declared on the host the eval builds and rank beside the criterion's own, and
``Result.claim_cells()`` reads every cell's readings and verdicts back. The text is invented: a town's bridges and
towers.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.off_loop import off_loop
from threetears.evals import Answer, Eval, Judge, grader
from threetears.evals.quick import Claims, Document, as_host
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown, report_markdown
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel import ValidationFailedError
from threetears.evals.kernel.host import ClaimView
from threetears.evals.kernel.claims.readings import claim_readings
from threetears.evals.quick.one_call import JUDGED_CALLABLE_KIND
from threetears.evals.run import get_result_trace, list_results, list_runs
from threetears.evals.run.authoring import get_template
from threetears.evals.schema import Segment, WebBudget, WebPage, WebSearchResult
from threetears.evals.schema.models import CLAIM_DOCUMENT_BYTES

MODEL = "scripted/claims"
MEMORY = ":memory:"

#: A sentence the grader lists as one claim, and the passages it checks one against.
_SEGMENT = re.compile(r"Segment (\d+):\n(.*?)(?=\n\nSegment \d+:|\n\n# |\Z)", re.S)
_PASSAGE = re.compile(r"^([SW]\d+)\. From [^\n]*:\n(.*?)(?=\n\n[SKW]\d+\. |\n\n# |\Z)", re.S | re.M)
_CLAIM = re.compile(r"^Claim (\d+), [^:]*: (.*)$", re.M)
_WEB_CLAIM = re.compile(r"^The claim: (.*)$", re.M)
_SENTENCE = re.compile(r"[^.\s][^.]*\.")


@dataclass(frozen=True)
class _Reply:
    """One completion, in the attribute names the engine reads."""

    content: str
    stop_reason: str = "end_turn"
    model: str = MODEL
    served_model: str | None = MODEL
    input_tokens: int | None = 100
    output_tokens: int | None = 20
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    price_source: str | None = "scripted"
    temperature: float | None = None


def _cited(claim: str, passages: list[tuple[str, str]], *, contradicts: bool) -> dict[str, Any] | None:
    """The first shown sentence on the claim's subject (its first three words): a support where it says the same, a
    contradiction (when asked for) where it says otherwise, quoting its subject and its last word as the value."""
    subject = " ".join(claim.split()[:3])
    for ref, text in passages:
        for sentence in _SENTENCE.findall(text):
            same = sentence.rstrip(".") == claim.rstrip(".")
            if sentence.startswith(subject) and (same or contradicts):
                words = sentence.rstrip(".").split()
                stance = "supports" if same else "contradicts"
                return {"ref": ref, "stance": stance, "quote": sentence, "subject": subject, "value": words[-1]}
    return None


# parity-with: threetears.evals.schema.CompletionClient
@dataclass
class _FakeClaimsGrader:
    """Answers each claims call from what its prompt shows, and a rubric dimension a pass; records each schema asked."""

    model_name: str = MODEL
    asked: list[str] = field(default_factory=list)

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        name = (response_format or {}).get("json_schema", {}).get("name", "")
        self.asked.append(name)
        if name == "judge_reply":
            (key,) = re.findall(r'single key "([^"]+)"', system)
            return _Reply(json.dumps({"reasoning": "reads well", "criteria_scores": {key: "pass"}}))
        if name == "claim_listing":
            listed = [
                {
                    "segment": int(index),
                    "claim": text.strip(),
                    "excerpt": text.strip().rstrip("."),
                    "specific": True,
                    "hedged": False,
                    "attributed": False,
                    "scope": "world",
                    "restates": None,
                }
                for index, text in _SEGMENT.findall(user)
            ]
            return _Reply(json.dumps({"claims": listed}))
        passages = _PASSAGE.findall(user)
        if name == "claim_truth":
            truth = []
            for index, claim in _CLAIM.findall(user):
                cited = _cited(claim, passages, contradicts=True)
                truth.append({"claim": int(index), "citations": [cited] if cited else [], "suspect": cited is None})
            return _Reply(json.dumps({"claims": truth}))
        if name == "claim_ground":
            ground = []
            for index, claim in _CLAIM.findall(user):
                cited = _cited(claim, passages, contradicts=False)
                if cited is not None:
                    del cited["stance"]
                ground.append({"claim": int(index), "citations": [cited] if cited else []})
            return _Reply(json.dumps({"claims": ground}))
        if name == "claim_web_queries":
            (claim,) = _WEB_CLAIM.findall(user)
            return _Reply(json.dumps({"queries": [claim]}))
        if name == "claim_web_verdict":
            (claim,) = _WEB_CLAIM.findall(user)
            cited = _cited(claim, passages, contradicts=True)
            if cited is None:
                return _Reply(
                    json.dumps({"outcome": "inconclusive", "ref": None, "quote": None, "subject": None, "value": None})
                )
            outcome = "true" if cited.pop("stance") == "supports" else "false"
            return _Reply(json.dumps({"outcome": outcome, **cited}))
        raise AssertionError(f"the scripted grader was asked {name!r}, which no test here scripts")

    async def aclose(self) -> None:
        """Nothing to release."""


# parity-with: threetears.evals.schema.WebSearch
@dataclass
class _FakeSearch:
    """Answers every query with its one page, and records each query."""

    page: WebPage
    identity: str = "scripted-search"
    queries: list[str] = field(default_factory=list)

    async def search(self, query: str) -> WebSearchResult:
        self.queries.append(query)
        return WebSearchResult(pages=(self.page,), usd=0.0)

    def price_ceiling(self) -> float | None:
        return 0.0


#: A search port no test here searches: what a grader built only to be refused is handed.
_SEARCH = _FakeSearch(WebPage(url="https://example.org/none", title="None", text="Nothing here."))

#: Each case's source, which decides truth for every arm: a bridge's year, a tower's height.
CASES = [
    {"id": "bridge", "town": "Wexley", "source": "The Wexley bridge opened in 1932. It crosses the river once."},
    {"id": "tower", "town": "Harlow", "source": "The Harlow tower stands 90 metres tall. It is built of brick."},
]
TRUE = {"Wexley": "The Wexley bridge opened in 1932.", "Harlow": "The Harlow tower stands 90 metres tall."}
FALSE = {"Wexley": "The Wexley bridge opened in 1951.", "Harlow": "The Harlow tower stands 90 metres tall."}


async def careful(case: dict[str, Any]) -> str:
    """Say one fact about the town's landmark."""
    return TRUE[case["town"]]


async def careless(case: dict[str, Any]) -> str:
    """Say one fact about the town's landmark."""
    return FALSE[case["town"]]


def _results(result: Any, arm: str | None = None) -> list[Any]:
    """The arm's stored results (the one arm's, with none named), in case order."""
    summary = result.summary if arm is None else result.arms[arm]
    stored = off_loop(list_results, result.host.storage, summary.run_id, result.scope_id)
    # A case's stored id ends in its place in the case list.
    return sorted(stored, key=lambda cell: int(cell.test_case_id.rsplit("-", 1)[1]))


async def test_an_eval_graded_by_claims_alone_is_judged_on_the_claims_model() -> None:
    """With no Judge and no function grader, each answer's claims are graded all the same, on the grader's model."""
    client = _FakeClaimsGrader()
    result = await Eval(CASES, careless, [Claims(client, case_documents="source")], k=1).run(store=MEMORY)
    summary = result.summary

    assert (summary.status, summary.n_scored, summary.n_excluded) == ("completed", 2, 0)
    assert client.asked == ["claim_listing", "claim_truth"] * 2, "no rubric call, one listing and one truth call a cell"
    template = off_loop(get_template, result.host, summary.template_id or "", result.scope_id)
    assert template.candidate_kind == JUDGED_CALLABLE_KIND and template.rubric == []
    (criterion,) = template.claim_criteria
    assert (criterion.name, criterion.grounded_in, criterion.web) == ("answer.claims", None, None)
    assert template.claim_unit_names == {"answer.claims": "answer"}
    (run,) = off_loop(list_runs, result.host, result.scope_id)
    assert run.judge_model == MODEL, "the claims grader's model is the run's judge pin"

    bridge, tower = _results(result)
    assert claim_readings("answer.claims", bridge.claim_tallies["answer.claims"])["answer.claims_false_per_unit"] == 1
    assert claim_readings("answer.claims", tower.claim_tallies["answer.claims"])["answer.claims_false_per_unit"] == 0
    evidence = off_loop(get_result_trace, result.host.storage, bridge).claim_evidence
    assert evidence is not None
    assert evidence.segments == (Segment(text="The Wexley bridge opened in 1951.", unit=0),), "one answer, one unit"
    ((source),) = evidence.documents or ()
    assert (source.provenance, source.text, source.title) == ("case", CASES[0]["source"], "case document 1")


async def test_a_comparison_ranks_its_arms_on_false_claims_and_the_page_reads_them() -> None:
    cases = [
        {"town": f"Town{index}", "source": f"The Town{index} bridge opened in {1900 + index}. It is old."}
        for index in range(12)
    ]

    async def recalls(case: dict[str, Any]) -> str:
        return case["source"].split(" It ")[0]

    async def guesses(case: dict[str, Any]) -> str:
        index = int(case["town"].removeprefix("Town"))
        return f"The {case['town']} bridge opened in {1900 + index + index % 2 * 7}."

    result = await Eval(
        cases,
        {"recalls": recalls, "guesses": guesses},
        [Claims(_FakeClaimsGrader(), case_documents="source")],
        k=1,
        name="bridges",
    ).run(store=MEMORY)

    printed = result.render()
    assert "False claims per answer (answer.claims, judged): guesses worse than recalls (lower is better)" in printed
    page = report_markdown(result.report)
    assert "| False claims per answer (answer.claims, judged) | 0 | 0.50 (0.17 to 0.83) | +0.50 |" in page
    assert "| Share of answers with a false claim (answer.claims, judged) | 0% | 50% (17 to 83) | +50 pts |" in page
    assert "Readings not shown" not in page, "every claim reading is carried by both arms"


async def test_what_the_candidate_read_grounds_its_claims_and_reading_nothing_reads_zero() -> None:
    """``Answer.documents`` is the candidate's own: it grounds a claim, and an arm that read none reads a share of 0."""
    notes = {town: f"{text} Noted from the town archive." for town, text in TRUE.items()}

    async def researched(case: dict[str, Any]) -> Answer:
        return Answer(TRUE[case["town"]], documents=[Document(notes[case["town"]], title="archive notes")])

    async def recalled(case: dict[str, Any]) -> Answer:
        return Answer(TRUE[case["town"]], documents=())

    claims = Claims(_FakeClaimsGrader(), grounded_in="run")
    result = await Eval(CASES, {"researched": researched, "recalled": recalled}, [claims], k=1, name="grounding").run(
        store=MEMORY
    )

    for arm, share in (("researched", 1.0), ("recalled", 0.0)):
        for cell in _results(result, arm):
            assert (
                claim_readings("answer.claims", cell.claim_tallies["answer.claims"])["answer.claims_grounded_share"]
                == share
            )
    first = _results(result, "researched")[0]
    evidence = off_loop(get_result_trace, result.host.storage, first).claim_evidence
    assert evidence is not None
    ((read),) = evidence.documents or ()
    assert (read.provenance, read.title) == ("run", "archive notes")
    stored = off_loop(get_result_trace, result.host.storage, first)
    assert "archive" not in json.dumps(stored.trace), "the answer is stored as it was given, its documents apart"
    unchecked = (
        "False claims on answer.claims were not checked: no case evidence or web search decides a claim's truth, so "
        "they read 0 whatever the answers say; its grounded share reads what the answers say."
    )
    assert unchecked in result.render()
    assert unchecked in report_markdown(result.report) and unchecked in campaign_report_markdown(result.page)


async def test_an_answer_that_says_nothing_of_what_it_read_fails_a_run_grounded_in_it() -> None:
    result = await Eval(CASES, careful, [Claims(_FakeClaimsGrader(), grounded_in="run")], k=1).run(store=MEMORY)

    assert result.summary.status == "failed"
    assert "return Answer(value, documents=[...]), with documents=() where it read none" in result.render()


async def test_a_cut_makes_each_entry_a_unit_and_names_it() -> None:
    """A structured answer cut into entries: each entry a unit, a page calling it an entry; Segment values carry a
    piece in no unit and the host's own tags."""

    async def scout(case: dict[str, Any]) -> dict[str, Any]:
        return {"summary": "Two landmarks.", "entries": [TRUE["Wexley"], FALSE["Harlow"]]}

    def entries(case: dict[str, Any], answer: dict[str, Any]) -> list[str]:
        return list(answer["entries"])

    def tagged(case: dict[str, Any], answer: dict[str, Any]) -> list[Segment]:
        return [
            Segment(text=answer["summary"]),
            *(Segment(text=text, unit=at, tags={"entry": str(at)}) for at, text in enumerate(answer["entries"])),
        ]

    source = " ".join(TRUE.values())
    cases = [{"id": "landmarks", "source": source}]
    by_text = await Eval(
        cases, scout, [Claims(_FakeClaimsGrader(), unit="entry", segments=entries, case_documents="source")], k=1
    ).run(store=MEMORY)
    (cell,) = _results(by_text)
    tally = cell.claim_tallies["answer.claims"]
    assert (tally.units, tally.units_with_false) == (2, 0), "the second entry agrees with its source"
    template = off_loop(get_template, by_text.host, by_text.summary.template_id or "", by_text.scope_id)
    assert template.claim_unit_names == {"answer.claims": "entry"}
    evidence = off_loop(get_result_trace, by_text.host.storage, cell).claim_evidence
    assert evidence is not None and evidence.segmentation.endswith("entries")
    assert [segment.unit for segment in evidence.segments] == [0, 1]

    by_segment = await Eval(
        cases,
        scout,
        [Claims(_FakeClaimsGrader(), segments=tagged, segmentation="scout-v1", case_documents="source")],
        k=1,
    ).run(store=MEMORY)
    (cell,) = _results(by_segment)
    evidence = off_loop(get_result_trace, by_segment.host.storage, cell).claim_evidence
    assert evidence is not None and evidence.segmentation == "scout-v1"
    assert [(segment.unit, segment.tags) for segment in evidence.segments] == [
        (None, {}),
        (0, {"entry": "0"}),
        (1, {"entry": "1"}),
    ]
    assert cell.claim_tallies["answer.claims"].units == 2, "the summary's claims count, outside every unit"


async def test_a_doubted_claim_is_checked_on_the_search_given_and_a_skipped_one_is_said() -> None:
    """No case evidence covers the claims, so the grader doubts them, and the web decides what its budget reaches."""
    page = WebPage(url="https://example.org/wexley", title="Wexley", text="The Wexley bridge opened in 1932.")
    search = _FakeSearch(page)

    async def two_claims(case: dict[str, Any]) -> list[str]:
        return [FALSE["Wexley"], "The Wexley market opens on Sundays."]

    def each(case: dict[str, Any], answer: list[str]) -> list[str]:
        return answer

    budget = WebBudget(
        max_queries_per_claim=1, max_checked_per_cell=1, results_per_query=1, page_chars=400, window_chars=200
    )
    cases = [{"id": f"wexley-{at}", "town": "Wexley", "at": at} for at in range(2)]
    result = await Eval(
        cases, two_claims, [Claims(_FakeClaimsGrader(), segments=each, search=search, web=budget)], k=1
    ).run(store=MEMORY)

    assert search.queries == [FALSE["Wexley"]] * 2, "one search a cell, for the first doubted claim"
    for cell in _results(result):
        grade = off_loop(get_result_trace, result.host.storage, cell).claim_grades["answer.claims"]
        first, second = grade.claims
        assert (first.verdict, first.graded_against, first.web) == ("contradicted", "web", "checked")
        assert (second.verdict, second.web) == ("unverifiable", "skipped_cap")
    page_text = report_markdown(result.report)
    assert "| False claims per answer (answer.claims, judged) | 0.50 ± 0 | 2 |" in page_text
    caveats = {caveat.id: caveat.cls for caveat in report_input(campaign_bundle(result.report.bundle)).caveat_list}
    assert caveats["claims_web_skipped:answer.claims"] == "B", "a claim the web budget skipped is said"
    assert (
        "The web budget of answer.claims skipped claims in 2 cells of two_claims: each reads unverifiable, never false, "
        "so its false claims there may be undercounted."
    ) in page_text, "and the page says so under its scoreboard"
    assert caveats["judge_uncalibrated"] == "C", "a claims grader checked against nothing yet is said"


async def test_a_judge_and_claims_grade_one_eval_on_one_model() -> None:
    client = _FakeClaimsGrader()
    judge = Judge(client, {"clear": "The answer is clear."})
    result = await Eval(CASES, careful, [judge, Claims(client, case_documents="source")], k=1).run(store=MEMORY)

    assert result.summary.status == "completed"
    assert sorted(set(client.asked)) == ["claim_listing", "claim_truth", "judge_reply"]
    for cell in _results(result):
        assert [score.dim for score in cell.rubric_scores] == ["answer.clear"]
        assert claim_readings("answer.claims", cell.claim_tallies["answer.claims"])["answer.claims_false_per_unit"] == 0


async def test_a_case_missing_its_evidence_field_is_left_out_by_name() -> None:
    cases = [CASES[0], {"id": "unsourced", "town": "Harlow"}]
    result = await Eval(cases, careful, [Claims(_FakeClaimsGrader(), case_documents="source")], k=1).run(store=MEMORY)

    assert (result.summary.n_scored, result.summary.n_excluded) == (1, 1)
    assert "reads the case's evidence from 'source', which this case lacks" in result.render()


def test_the_case_s_documents_are_never_cut_for_what_an_arm_read() -> None:
    """Each provenance has its own half of a cell's room: an arm that read a great deal cuts only its own pages, so
    truth reads the same case documents whatever an arm read."""
    claims = Claims(_FakeClaimsGrader(), case_documents="source", grounded_in="run")
    source = "The Wexley bridge opened in 1932. " * 2000
    read = [Document("A page about the bridge. " * 20000, title=f"page {at}") for at in range(3)]

    evidence = claims.evidence({"source": source}, "The Wexley bridge opened in 1932.", read)

    case, *pages = evidence.documents or ()
    assert (case.provenance, case.text) == ("case", source)
    assert [page.provenance for page in pages] == ["run"] * 3
    assert all("(cut to its first" in page.title and len(page.text) < len(read[0].text) for page in pages)
    held = sum(len(json.dumps(document.model_dump(mode="json")).encode()) for document in evidence.documents or ())
    assert held <= CLAIM_DOCUMENT_BYTES


def test_a_case_s_documents_read_from_a_field_or_a_callable() -> None:
    claims = Claims(_FakeClaimsGrader(), case_documents="sources")
    listed = {"sources": ["First text.", {"text": "Second text.", "title": "rules", "url": "https://example.org/r"}]}
    first, second = claims.evidence(listed, "An answer.", None).documents or ()
    assert (first.title, second.title, second.url) == ("case document 1", "rules", "https://example.org/r")

    rendered = Claims(_FakeClaimsGrader(), case_documents=lambda case: [Document(case["text"], title="given")])
    (only,) = rendered.evidence({"text": "Its text."}, "An answer.", None).documents or ()
    assert (only.provenance, only.title, only.text) == ("case", "given", "Its text.")
    assert rendered.evidence({"text": "Its text."}, "An answer.", None).segments == (
        Segment(text="An answer.", unit=0),
    )
    assert Claims(_FakeClaimsGrader(), grounded_in="run").evidence({}, "An answer.", None).documents is None, (
        "nothing declared"
    )


@pytest.mark.parametrize(
    ("build", "said"),
    [
        pytest.param(lambda: Claims(object()), "Claims(model=...) takes a model id", id="no-model"),
        pytest.param(
            lambda: Claims(
                _FakeClaimsGrader(),
                web=WebBudget(
                    max_queries_per_claim=1, max_checked_per_cell=1, results_per_query=1, page_chars=10, window_chars=10
                ),
            ),
            "no search was given to check on",
            id="web-without-search",
        ),
        pytest.param(lambda: Claims(_FakeClaimsGrader(), grounded_in="web"), "not 'web'", id="grounded-in-web"),
        pytest.param(lambda: Claims(_FakeClaimsGrader(), case_documents=3), "names the case's field", id="documents"),
        pytest.param(
            lambda: Eval(
                CASES,
                careful,
                [
                    Claims(_FakeClaimsGrader(), case_documents="source"),
                    Claims(_FakeClaimsGrader(), "other", search=_SEARCH),
                ],
            ),
            "graded by one Claims",
            id="two-claims",
        ),
        pytest.param(
            lambda: Eval(
                CASES,
                careful,
                [
                    Judge(_FakeClaimsGrader(model_name="other/judge"), {"clear": "Clear."}),
                    Claims(_FakeClaimsGrader(), case_documents="source"),
                ],
            ),
            "give both the same model",
            id="two-models",
        ),
        pytest.param(lambda: grader(Claims(_FakeClaimsGrader(), search=_SEARCH)), "list it bare", id="wrapped"),
        pytest.param(
            lambda: Claims(_FakeClaimsGrader()),
            "give it case_documents= (the case's own evidence, which decides truth), search= (a web search port, which "
            "checks a doubted claim), or grounded_in='run'",
            id="nothing-to-check-against",
        ),
        pytest.param(
            lambda: Claims(_FakeClaimsGrader(), grounded_in="case", search=_SEARCH),
            "Claims(grounded_in='case') reads the case's own documents, and no case_documents= names them",
            id="grounded-in-case-with-no-case-documents",
        ),
    ],
)
def test_a_claims_grader_that_could_not_grade_as_declared_is_refused_where_it_is_written(build: Any, said: str) -> None:
    with pytest.raises(ValueError, match=re.escape(said)):
        build()


def test_what_a_candidate_read_is_held_as_documents() -> None:
    assert Answer("An answer.", documents=["Its source."]).documents == (Document("Its source."),)
    assert Answer("An answer.").documents is None
    with pytest.raises(ValueError, match="a list of what the candidate read"):
        Answer("An answer.", documents="Its source.")
    with pytest.raises(ValueError, match="Document values or their text"):
        Answer("An answer.", documents=[3])
    with pytest.raises(ValueError, match="a non-blank string"):
        Document("  ")


def test_documents_whose_titles_alone_overflow_their_room_leave_the_cell_out_by_name() -> None:
    """The cut ends: where titles and addresses alone fill the room, no cut of the texts could, and it says so."""
    claims = Claims(_FakeClaimsGrader(), grounded_in="run")
    read = [Document("A page.", title="t" * (CLAIM_DOCUMENT_BYTES // 2))]

    with pytest.raises(ValueError, match=r"take \d+ bytes beside their text, each cut to one character"):
        claims.evidence({}, "An answer.", read)


async def test_renaming_the_unit_keeps_the_case_set_and_its_stamp() -> None:
    """What a page calls a unit is presentation: renamed, the case set and the template's updated_at stand."""
    first = Eval(CASES, careful, [Claims(_FakeClaimsGrader(), case_documents="source")], k=1)
    host = as_host([first], store=MEMORY)
    before = (await first.run(host=host)).summary.template_id or ""
    stored = off_loop(get_template, host, before, first.name)

    renamed = Eval(CASES, careful, [Claims(_FakeClaimsGrader(), unit="reply", case_documents="source")], k=1)
    after = (await renamed.run(host=host)).summary.template_id

    assert after == before
    again = off_loop(get_template, host, before, first.name)
    assert again.claim_unit_names == {"answer.claims": "reply"}
    assert again.updated_at == stored.updated_at, "no run's reading of the template moved"


def returns(view: ClaimView) -> float:
    """World claims naming a town the session record says the story has left."""
    left = (view.session or {}).get("left", ())
    return float(sum(1 for claim in view.claims if claim.scope == "world" and any(town in claim.text for town in left)))


#: The cases, each with the run's own record: the story has left Harlow.
PLAYED = [{**case, "played": {"left": ["Harlow"]}} for case in CASES]


async def test_a_reading_of_your_own_reads_the_session_record_and_each_cell_reads_back() -> None:
    """A reading declared on Claims is the host's derived measure, read off each cell's frozen view of its claims and
    session record; the result reads every cell's readings and verdicts back, arm by arm."""
    claims = Claims(
        _FakeClaimsGrader(), case_documents="source", session="played", readers={"returns": (returns, "lower")}
    )
    result = await Eval(PLAYED, {"careful": careful, "careless": careless}, [claims], k=1, name="returns").run(
        store=MEMORY
    )

    descriptor = result.host.profile.measures.get("answer.claims_returns")
    assert descriptor is not None
    assert (descriptor.derived_from, descriptor.family, descriptor.higher_is_better, descriptor.nonnegative) == (
        "answer.claims",
        "claims",
        False,
        True,
    )
    assert descriptor.description == "World claims naming a town the session record says the story has left."
    cells = off_loop(result.claim_cells)
    assert [(cell.arm, cell.case, cell.repeat) for cell in cells] == [
        ("careful", "bridge", 1),
        ("careful", "tower", 1),
        ("careless", "bridge", 1),
        ("careless", "tower", 1),
    ]
    assert [cell.readings["answer.claims_returns"] for cell in cells] == [0, 1, 0, 1], "only the tower names Harlow"
    assert [cell.readings["answer.claims_false_per_unit"] for cell in cells] == [0, 0, 1, 0]
    assert all(cell.view is not None and cell.lost is None and cell.unread is None for cell in cells)
    verdicts = [[claim.verdict for claim in cell.view.claims] for cell in cells if cell.view is not None]
    assert verdicts == [["supported"], ["supported"], ["contradicted"], ["supported"]]
    first = cells[0].view
    assert first is not None and first.session == {"left": ("Harlow",)}, "the record, deep-frozen"
    assert [cell.case for cell in off_loop(result.claim_cells, "careless")] == ["bridge", "tower"]
    stored = off_loop(get_result_trace, result.host.storage, _results(result, "careful")[1]).claim_evidence
    assert stored is not None and stored.session == {"left": ["Harlow"]}, "stored as written"
    assert any(row["measure_id"] == "answer.claims_returns" for row in result.contrasts()), "it ranks the arms"


def test_the_session_record_reads_off_a_field_or_the_case_and_its_answer() -> None:
    said = Claims(_FakeClaimsGrader(), case_documents="source", session=lambda case, answer: {"said": answer})
    assert said.evidence({"source": "A text."}, "An answer.", None).session == {"said": "An answer."}
    none = Claims(_FakeClaimsGrader(), case_documents="source", session=lambda case, answer: None)
    assert none.evidence({"source": "A text."}, "An answer.", None).session is None, "a cell with no record"
    unset = Claims(_FakeClaimsGrader(), case_documents="source")
    assert unset.evidence({"source": "A text."}, "An answer.", None).session is None
    field_read = Claims(_FakeClaimsGrader(), grounded_in="run", session="played")
    assert field_read.evidence({"played": {"rolls": [14]}}, "An answer.", ()).session == {"rolls": [14]}

    with pytest.raises(ValueError, match="reads the session record from 'played', which this case lacks"):
        field_read.evidence({}, "An answer.", ())
    listed = Claims(_FakeClaimsGrader(), grounded_in="run", session=lambda case, answer: ["not", "an", "object"])
    with pytest.raises(ValueError, match="it is a JSON object"):
        listed.evidence({}, "An answer.", ())
    raising = Claims(_FakeClaimsGrader(), grounded_in="run", session=lambda case, answer: case["absent"])
    with pytest.raises(ValueError, match="the claims grader's session raised KeyError"):
        raising.evidence({}, "An answer.", ())


async def test_a_case_whose_session_record_cannot_be_read_is_left_out_by_name() -> None:
    cases = [PLAYED[0], CASES[1]]
    claims = Claims(_FakeClaimsGrader(), case_documents="source", session="played")
    result = await Eval(cases, careful, [claims], k=1).run(store=MEMORY)

    assert (result.summary.n_scored, result.summary.n_excluded) == (1, 1)
    assert "reads the session record from 'played', which this case lacks" in result.render()


async def test_a_reading_that_fails_on_a_synthetic_cell_is_refused_before_anything_runs() -> None:
    """Launch reads each reading on a cell with no claim and no session record first: one assuming a record is
    refused, naming the reading, before any answer is graded."""
    client = _FakeClaimsGrader()
    assuming = Claims(
        client, case_documents="source", readers={"left": (lambda view: len(view.session["left"]), "lower")}
    )

    with pytest.raises(ValidationFailedError, match="the reading 'answer.claims_left'"):
        await Eval(PLAYED, careful, [assuming], k=1).run(store=MEMORY)
    assert client.asked == [], "nothing was graded"


async def test_a_host_must_read_the_readings_claims_declares() -> None:
    bare = Eval(PLAYED, careful, [Claims(_FakeClaimsGrader(), case_documents="source")], k=1)
    host = as_host([bare], store=MEMORY)
    reading = Claims(_FakeClaimsGrader(), case_documents="source", readers={"returns": (returns, "lower")})

    with pytest.raises(ValueError, match="does not read the readings Claims 'answer.claims' declares"):
        await Eval(PLAYED, careful, [reading], k=1).run(host=host)
    with pytest.raises(ValueError, match="two Claims named 'answer.claims' declare different readings of their own"):
        as_host([bare, Eval(PLAYED, careful, [reading], k=1)], store=MEMORY)


# parity-with: threetears.evals.schema.CompletionClient
@dataclass
class _FakeUnreadableGrader:
    """Answers every call with a reply no schema holds."""

    model_name: str = MODEL

    async def generate(self, *, system: str, user: str, response_format: dict[str, Any] | None = None) -> _Reply:
        return _Reply("not json")

    async def aclose(self) -> None:
        """Nothing to release."""


async def test_claim_cells_says_a_lost_cell_and_refuses_an_eval_no_claims_graded() -> None:
    lost = await Eval(CASES[:1], careful, [Claims(_FakeUnreadableGrader(), case_documents="source")], k=1).run(
        store=MEMORY
    )
    (cell,) = off_loop(lost.claim_cells)
    assert cell.lost is not None and cell.view is None
    assert dict(cell.readings) == {"answer.claims_lost": 1.0}

    def exact(case: dict[str, Any], answer: str) -> bool:
        """Whether the answer is the case's first sentence."""
        return answer == TRUE[case["town"]]

    unclaimed = await Eval(CASES, careful, [exact], k=1).run(store=MEMORY)
    with pytest.raises(ValueError, match="graded under no claim criterion"):
        off_loop(unclaimed.claim_cells)
    with pytest.raises(ValueError, match="no arm"):
        off_loop(unclaimed.claim_cells, "absent")


@pytest.mark.parametrize(
    ("readers", "said"),
    [
        pytest.param({"two words": (returns, "lower")}, "names each reading by an identifier", id="name"),
        pytest.param({"false_per_unit": (returns, "lower")}, "one of the criterion's own readings", id="taken"),
        pytest.param({"returns": returns}, "is (read, better) or (read, better, (low, high))", id="bare"),
        pytest.param({"returns": ("returns", "lower")}, "reads with a function of a cell's view", id="not-callable"),
        pytest.param({"returns": (returns, "down")}, "'lower' or 'higher', not 'down'", id="better-end"),
        pytest.param({"returns": (returns, "lower", (1, 0))}, "by two numbers, low first", id="bounds"),
        pytest.param([returns], "maps each reading's name to (read, better)", id="not-a-mapping"),
    ],
)
def test_a_reading_of_your_own_that_could_not_be_read_is_refused_where_it_is_written(readers: Any, said: str) -> None:
    with pytest.raises(ValueError, match=re.escape(said)):
        Claims(_FakeClaimsGrader(), case_documents="source", readers=readers)


def test_a_bounded_reading_declares_its_range_and_a_session_is_a_field_or_a_callable() -> None:
    share = Claims(_FakeClaimsGrader(), case_documents="source", readers={"share": (returns, "higher", (0, 1))})
    (descriptor,) = share.derived
    assert (descriptor.name, descriptor.value_range, descriptor.nonnegative, descriptor.higher_is_better) == (
        "answer.claims_share",
        (0.0, 1.0),
        False,
        True,
    )
    with pytest.raises(ValueError, match="names the case's field holding the run's own record"):
        Claims(_FakeClaimsGrader(), case_documents="source", session=3)
