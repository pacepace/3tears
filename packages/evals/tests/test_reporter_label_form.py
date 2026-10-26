"""A reporter label's form is its dimension's scale, and it may quote the memo passage it is about.

The owner's ruling (2026-10-24): a pass/fail dimension takes a ``pass``/``fail`` verdict and an ordinal one keeps the
five directions; no mapping between them is added. Pinned here:

- a label holds a direction or a verdict, exactly one, and an excerpt that is not blank; a direction label quoting no
  passage is stored exactly as v16.1 stored it, and a verdict label carries no direction key;
- a verdict on a pass/fail score agrees or disagrees, a direction on one reads no agreement, and a verdict on an
  ordinal score reads none either; the calibration read carries the verdict and the excerpt beside the quote;
- the freeze refuses a direction on a pass/fail dimension, a verdict on an ordinal one, both, neither, an excerpt the
  recorded memo does not hold, and an excerpt with no memo, writing nothing; an excerpt differing from the memo only in
  its runs of whitespace, or carrying a line break at its end, is accepted;
- the excerpt is no part of the verdict, but it is never dropped: re-freezing the same verdicts with another excerpt
  is refused, naming the stored excerpt and the case to supersede, and superseding that case stores the correction;
  re-freezing the same verdicts and excerpts returns the stored case.

Every refusal is structural (a form, a substring), never a reading of what the words say.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.analysis import get_analysis, render_memo_as_written
from threetears.evals.analysis.reporter_bank import label_agrees, read_calibration
from threetears.evals.analysis.reporter_kind import (
    REPORTER_CASE_KEY,
    LabelCriterion,
    ReporterCase,
    ReporterLabel,
    reporter_case_of,
    reporter_case_payload,
)
from threetears.evals.analysis.generator import user_message_digest
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.identity import IDENTITY_VERSION
from threetears.evals.ops import analysis_generate
from threetears.evals.schema.models import EvalResult, EvalTestCase, RubricScore
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, OpsFixture, settled
from packages.evals.tests.reporter_case_support import (
    ANSWERS,
    GROUNDED,
    freeze_corpus_campaign,
    reporter_fixture,
    scaled_reporter_template,
)

_ORDINAL = LabelCriterion(description="Every claim traces to the evidence.", scale="ordinal")
_PASS_FAIL = LabelCriterion(description="The summary answers the question asked.", scale="pass_fail")


def _score(dim: str, scale: str, score: int) -> RubricScore:
    return RubricScore(dim=dim, scale=scale, score=score, axis="capability", judge_temperature=0.0)  # type: ignore[arg-type]


# --- the label ------------------------------------------------------------------------------------------------------


class TestTheLabel:
    def test_a_label_holds_a_direction_or_a_verdict(self) -> None:
        assert ReporterLabel(dimension=ANSWERS, verdict="fail", quote="it answers another question").verdict == "fail"
        assert ReporterLabel(dimension=GROUNDED, direction="mid", quote="mostly").direction == "mid"

    @pytest.mark.parametrize("form", [{"direction": "high", "verdict": "pass"}, {}], ids=["both", "neither"])
    def test_both_or_neither_is_refused(self, form: dict[str, Any]) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            ReporterLabel(dimension=ANSWERS, quote="reads fine", **form)

    def test_a_verdict_outside_pass_or_fail_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ReporterLabel.model_validate({"dimension": ANSWERS, "verdict": "mid", "quote": "reads fine"})

    def test_a_blank_excerpt_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="memo_excerpt"):
            ReporterLabel(dimension=ANSWERS, verdict="pass", quote="reads fine", memo_excerpt=" \n ")


class TestTheStoredPayload:
    def _labels(self, *labels: ReporterLabel) -> list[dict[str, Any]]:
        message = "message for memo-1"
        case = ReporterCase(
            bundle={"campaign_id": "camp-widgets"},
            bundle_fingerprint="fp-1",
            bundle_assembled_at="2026-01-01T00:00:00+00:00",
            recorded_analysis_id="memo-1",
            recorded_memo="memo of memo-1",
            writer_message=message,
            recorded_writer_message_digest=user_message_digest(message),
            labels=list(labels),
        )
        stored: list[dict[str, Any]] = reporter_case_payload(case)[REPORTER_CASE_KEY]["labels"]
        return stored

    def test_a_direction_label_quoting_no_passage_carries_exactly_the_v16_1_keys(self) -> None:
        (stored,) = self._labels(ReporterLabel(dimension=GROUNDED, direction="high", quote="q", criterion=_ORDINAL))
        assert list(stored) == ["dimension", "direction", "quote", "note", "criterion"]

    def test_a_verdict_label_carries_its_verdict_and_excerpt_and_no_direction(self) -> None:
        (stored,) = self._labels(
            ReporterLabel(dimension=ANSWERS, verdict="pass", quote="q", memo_excerpt="memo of", criterion=_PASS_FAIL)
        )
        assert list(stored) == ["dimension", "verdict", "quote", "memo_excerpt", "note", "criterion"]
        assert (stored["verdict"], stored["memo_excerpt"]) == ("pass", "memo of")


# --- agreement ------------------------------------------------------------------------------------------------------


class TestAgreement:
    @pytest.mark.parametrize(
        ("verdict", "score", "agrees"), [("pass", 1, True), ("pass", 0, False), ("fail", 0, True), ("fail", 1, False)]
    )
    def test_a_verdict_agrees_with_the_pass_fail_score_it_names(self, verdict: str, score: int, agrees: bool) -> None:
        label = ReporterLabel(dimension=ANSWERS, verdict=verdict, quote="q")  # type: ignore[arg-type]
        assert label_agrees(label, _score(ANSWERS, "pass_fail", score)) is agrees

    def test_a_label_of_the_other_form_reads_no_agreement(self) -> None:
        """A direction stored on a pass/fail dimension, and a verdict on an ordinal one: no mapping is ratified."""
        direction = ReporterLabel(dimension=ANSWERS, direction="high", quote="q")
        verdict = ReporterLabel(dimension=GROUNDED, verdict="pass", quote="q")
        assert label_agrees(direction, _score(ANSWERS, "pass_fail", 1)) is None
        assert label_agrees(verdict, _score(GROUNDED, "ordinal", 5)) is None

    def test_the_calibration_read_sets_each_verdict_and_excerpt_beside_the_judges_score(self) -> None:
        message = "message for memo-1"
        case = ReporterCase(
            bundle={"campaign_id": "camp-widgets"},
            bundle_fingerprint="fp-1",
            bundle_assembled_at="2026-01-01T00:00:00+00:00",
            recorded_analysis_id="memo-1",
            recorded_memo="The summary answers it.",
            writer_message=message,
            recorded_writer_message_digest=user_message_digest(message),
            labels=[
                ReporterLabel(
                    dimension=ANSWERS,
                    verdict="fail",
                    quote="it answers another question",
                    memo_excerpt="answers it",
                    criterion=_PASS_FAIL,
                ),
                ReporterLabel(dimension=GROUNDED, direction="high", quote="every number checks", criterion=_ORDINAL),
            ],
        )
        result = EvalResult(
            id="r-1",
            scope_id=TOYHOST_SCOPE,
            eval_run_id="run-1",
            test_case_id="c-1",
            model="writer-a",
            k_iteration=1,
            rubric_scores=[_score(ANSWERS, "pass_fail", 0), _score(GROUNDED, "ordinal", 4)],
            termination="completed",
            cost_usd=0.0,
            cost_roles=["candidate", "inner_agent", "judge", "simulator"],
            usage=[],
            covariates={},
            phase_timings={},
            host_measures={},
            variant_key="vk-1",
            identity_version=IDENTITY_VERSION,
        )

        read = read_calibration(
            run_id="run-1",
            template_id="tpl",
            status="completed",
            dimensions=[GROUNDED, ANSWERS],
            cases=[("c-1", case)],
            results=[result],
            superseded_by={},
            archived={},
            live_criteria=None,
            judge_model="judge-x",
            effective_judges=None,
            ratings=[],
        )

        answers, grounded = read.cases[0].cells[0].labelled
        assert (answers.verdict, answers.direction, answers.memo_excerpt) == ("fail", None, "answers it")
        assert (answers.score_label, answers.agrees) == ("fail", True)
        assert (grounded.direction, grounded.verdict, grounded.agrees) == ("high", None, True)


# --- the freeze -----------------------------------------------------------------------------------------------------


async def _memo(fixture: OpsFixture) -> tuple[str, str]:
    """Generate the corpus campaign's analysis, and return its id and its memo as the judge reads it."""
    (job,) = (await analysis_generate(fixture.host, fixture.campaign.id, TOYHOST_SCOPE)).jobs
    analysis_id = (await settled(fixture.host, job.job_id)).analysis_id
    assert analysis_id is not None
    analysis = off_loop(get_analysis, fixture.host.eval_host.storage, analysis_id, TOYHOST_SCOPE)
    return analysis_id, render_memo_as_written(analysis)


def _scaled_fixture() -> OpsFixture:
    fixture = reporter_fixture()
    fixture.host.eval_host.storage.save_template(scaled_reporter_template())
    return fixture


def _stored(fixture: OpsFixture) -> list[EvalTestCase]:
    return fixture.host.eval_host.storage.query_test_cases(TOYHOST_SCOPE, template_id=scaled_reporter_template().id)


def _freeze(fixture: OpsFixture, analysis_id: str | None, labels: list[dict[str, Any]], **more: Any) -> Any:
    fields: dict[str, Any] = {"template_id": scaled_reporter_template().id, "labels": labels, **more}
    if analysis_id is not None:
        fields["recorded_analysis_id"] = analysis_id
    return freeze_corpus_campaign(fixture, **fields)


def _passage(memo: str) -> str:
    """A passage of the memo spanning a line break, its every run of whitespace respelled as another run."""
    at = memo.index("\n\n")
    return re.sub(r"\s+", " \n\t ", memo[max(at - 12, 0) : at + 24])


async def test_a_verdict_on_a_pass_fail_dimension_and_a_direction_on_an_ordinal_one_are_frozen() -> None:
    fixture = off_loop(_scaled_fixture)
    analysis_id, memo = await _memo(fixture)
    excerpt = _passage(memo)
    labels = [
        {"dimension": ANSWERS, "verdict": "fail", "quote": "it answers another question", "memo_excerpt": excerpt},
        {"dimension": GROUNDED, "direction": "high", "quote": "every number traces to a run"},
    ]

    receipt = off_loop(_freeze, fixture, analysis_id, labels)

    (stored,) = off_loop(_stored, fixture)
    answers, grounded = stored.host_payload[REPORTER_CASE_KEY]["labels"]
    assert "direction" not in answers and (answers["verdict"], answers["memo_excerpt"]) == ("fail", excerpt)
    assert "verdict" not in grounded and "memo_excerpt" not in grounded and grounded["direction"] == "high"
    assert [(label.verdict, label.memo_excerpt) for label in receipt.labels] == [("fail", excerpt), (None, None)]


@pytest.mark.parametrize(
    ("labels", "refused"),
    [
        ([{"dimension": ANSWERS, "direction": "high", "quote": "q"}], "pass_fail"),
        ([{"dimension": GROUNDED, "verdict": "pass", "quote": "q"}], "ordinal"),
        ([{"dimension": ANSWERS, "direction": "high", "verdict": "pass", "quote": "q"}], "exactly one"),
        ([{"dimension": ANSWERS, "quote": "q"}], "exactly one"),
        (
            [{"dimension": ANSWERS, "verdict": "pass", "quote": "q", "memo_excerpt": "no memo holds this ✓"}],
            "is not a passage of the recorded memo",
        ),
    ],
    ids=["direction-on-pass-fail", "verdict-on-ordinal", "both", "neither", "excerpt-not-in-the-memo"],
)
async def test_a_label_of_the_wrong_form_or_an_excerpt_the_memo_lacks_is_refused(
    labels: list[dict[str, Any]], refused: str
) -> None:
    fixture = off_loop(_scaled_fixture)
    analysis_id, _memo_text = await _memo(fixture)

    with pytest.raises(ValidationFailedError, match=refused):
        off_loop(_freeze, fixture, analysis_id, labels)

    assert off_loop(_stored, fixture) == [], "a refused freeze writes nothing"


def test_an_excerpt_with_no_recorded_memo_is_refused() -> None:
    fixture = _scaled_fixture()

    with pytest.raises(ValidationFailedError, match="pins none"):
        _freeze(fixture, None, [{"dimension": ANSWERS, "verdict": "pass", "quote": "q", "memo_excerpt": "a passage"}])

    assert _stored(fixture) == []


async def test_an_excerpt_carrying_a_line_break_at_either_end_is_found() -> None:
    fixture = off_loop(_scaled_fixture)
    analysis_id, memo = await _memo(fixture)
    excerpt = "\n" + memo[:20].strip() + "\n"

    receipt = off_loop(
        _freeze,
        fixture,
        analysis_id,
        [{"dimension": ANSWERS, "verdict": "pass", "quote": "q", "memo_excerpt": excerpt}],
    )

    assert [label.memo_excerpt for label in receipt.labels] == [excerpt], "the excerpt is stored as given"


async def test_re_freezing_the_same_verdicts_and_excerpts_returns_the_stored_case() -> None:
    fixture = off_loop(_scaled_fixture)
    analysis_id, memo = await _memo(fixture)
    labels = [{"dimension": ANSWERS, "verdict": "pass", "quote": "it answers the question", "memo_excerpt": memo[:20]}]

    stored = off_loop(_freeze, fixture, analysis_id, labels)
    again = off_loop(_freeze, fixture, analysis_id, labels)

    assert again.test_case_id == stored.test_case_id
    assert len(off_loop(_stored, fixture)) == 1


async def test_another_excerpt_on_the_same_verdicts_is_refused_and_superseding_stores_it() -> None:
    """The excerpt is no part of the verdict, so it leaves case identity alone; but a freeze reporting a case that
    quotes another passage than it was sent would drop the correction while claiming success."""
    fixture = off_loop(_scaled_fixture)
    analysis_id, memo = await _memo(fixture)
    first, corrected = _passage(memo), memo[:20]
    label = {"dimension": ANSWERS, "verdict": "pass", "quote": "it answers the question"}
    stored = off_loop(_freeze, fixture, analysis_id, [{**label, "memo_excerpt": first}])

    with pytest.raises(ValidationFailedError, match="quoting other memo passages") as refused:
        off_loop(_freeze, fixture, analysis_id, [{**label, "memo_excerpt": corrected}])

    assert repr(first) in str(refused.value), "the refusal names the stored excerpt"
    assert f"supersedes=[{stored.test_case_id!r}]" in (refused.value.remedy or "")
    assert len(off_loop(_stored, fixture)) == 1, "a refused freeze writes nothing"

    revised = off_loop(
        _freeze, fixture, analysis_id, [{**label, "memo_excerpt": corrected}], supersedes=[stored.test_case_id]
    )

    assert revised.test_case_id != stored.test_case_id and revised.supersedes == [stored.test_case_id]
    assert [label.memo_excerpt for label in revised.labels] == [corrected]
    by_id = {case.id: reporter_case_of(case) for case in off_loop(_stored, fixture)}
    first_case = by_id[stored.test_case_id]
    assert first_case is not None and [label.memo_excerpt for label in first_case.labels] == [first], (
        "the superseded case is left as it was"
    )
