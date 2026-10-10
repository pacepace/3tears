"""The judge temperature comparison (#633): a run's borderline cases re-judged at the pinned temperature and at the
provider's default, read side by side — the measurement the temperature policy is to rest on.

Every cell here runs through the real runner and the real storage, so the evidence the comparison re-judges is the
one the runner wrote, and the stored scores it selects on are the ones the judge phase stored.

What is pinned:

* a judge deterministic at temperature 0 and noisy at the provider's default shows exactly that: zero variance,
  no unstable case and full self-agreement on the pinned side; variance, unstable cases and lost agreement on the
  default side;
* the pinned side's calls are requested at ``DEFAULT_JUDGE_TEMPERATURE`` and the default side's are sent none,
  whatever a dim's config asks;
* only borderline dims are re-judged by default (a stored score at an end of the scale is skipped unless a recorded
  repeat disagreed on it); ``all`` takes every scored dim;
* a judge whose stored scores all record a model that refuses a temperature is refused before anything is spent; a
  client that drops the temperature at call time, or reports none, makes the comparison NOT comparable, its
  off-setting scores counted and left out;
* every call is priced and admitted against the out-of-run cap before the first is made, each one a ledger row under
  ``judge`` stamped with the run; a cent under the cap nothing is called; the estimate answers what it would do;
* nothing is written to the results;
* the command line runs it in one command, under a cap it names.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink, CellSpanWindow, VariantConfig
from threetears.evals.kernel.cassettes import CellCassettes
from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.identity import resolve_variant_identity
from threetears.evals.kernel.provider import withhold_failure_detail
from threetears.evals.quick import run_cli
from threetears.evals.run import (
    borderline_dims,
    compare_judge_temperatures,
    estimate_judge_temperature_comparison,
)
from threetears.evals.run.judge import JUDGE_CALL_ATTEMPTS, JUDGE_REQUEST_SETTINGS
from threetears.evals.run.judge_service import JudgeService
from threetears.evals.run.launch import build_judge_service
from threetears.evals.run.runner import RunnerOptions, run_one_result
from threetears.evals.schema.models import (
    DEFAULT_JUDGE_TEMPERATURE,
    MODEL_DEFAULT_TEMPERATURE,
    EvalResult,
    EvalTemplate,
    EvalTestCase,
    JudgedArtifact,
    JudgeEvidence,
    JudgeRepeat,
    RepeatedScore,
    RubricDim,
    RubricScore,
)
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.fixtures.toyhost.host import toyhost_host

_SCOPE = "scope-temperature"
_KIND = "temperature-probe"
_JUDGE = "judge/scripted"
_DIM = "doc.faithful"
_CEILING = 0.01

_EVIDENCE = JudgeEvidence(
    subject="The game master. Rules as written.",
    case_material="GM ONLY: the third flagstone is a trap.",
    artifact="Player: I check the floor.\nGM: Roll Perception.",
)

#: A client's completions carry no ``temperature`` attribute when it is built with this — a client that reports none.
_UNREPORTED = object()


@dataclass(frozen=True)
class _Completion:
    content: str
    temperature: float | None
    stop_reason: str = "end_turn"
    input_tokens: int = 10
    output_tokens: int = 5
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    model: str = _JUDGE
    served_model: str | None = _JUDGE


@dataclass(frozen=True)
class _SilentCompletion:
    """A completion from a client that does not say what temperature it sent."""

    content: str
    stop_reason: str = "end_turn"
    input_tokens: int = 10
    output_tokens: int = 5
    reasoning_tokens: int | None = None
    cost_usd: float | None = 0.001
    model: str = _JUDGE
    served_model: str | None = _JUDGE


@dataclass
class _Judge:
    """The judge behind every client: deterministic at temperature 0, noisy when sent none.

    A case's score is read off the ``party=`` value its prompt carries (2 + party % 4, so parties 0-3 score 2, 3, 4
    and 5). Sent no temperature, each call moves that score by -1, 0 or +1 in turn — a judge whose borderline answers
    wander when it samples at the provider's default.
    """

    #: What every client sends whatever it is asked: ``None`` a model refusing a temperature (sends none and says
    #: so), ``_UNREPORTED`` a client that reports nothing; unset, each client sends what it was built for.
    sends: object = "as_asked"
    calls: list[tuple[float | None, str]] = field(default_factory=list)
    requested: list[float | None] = field(default_factory=list)
    _noise: int = 0

    def client(self, temperature: float | None) -> _Client:
        self.requested.append(temperature)
        return _Client(self, temperature)

    def answer(self, sent: object, system: str, user: str) -> Any:
        dim = re.search(r'the single key "(.+?)"', system).group(1)  # type: ignore[union-attr]
        party = int(re.search(r"party: p(\d+)", user).group(1))  # type: ignore[union-attr]
        score = 2 + party % 4
        if sent is None or sent is _UNREPORTED:
            self._noise += 1
            score = max(1, min(5, score + (self._noise % 3) - 1))
        self.calls.append((None if sent is _UNREPORTED else sent, user))  # type: ignore[arg-type]
        content = json.dumps({"reasoning": "read it", "criteria_scores": {dim: score}})
        if sent is _UNREPORTED:
            return _SilentCompletion(content=content)
        return _Completion(content=content, temperature=sent)  # type: ignore[arg-type]


@dataclass
class _Client:
    """One judge client, built for one temperature, priced at :data:`_CEILING` a call."""

    judge: _Judge
    temperature: float | None
    model_name: str = _JUDGE

    def price_ceiling(self, *, system: str, user: str, response_format: Any = None) -> float | None:
        return _CEILING

    async def generate(self, *, system: str, user: str, response_format: Any = None) -> Any:
        sent = self.temperature if self.judge.sends == "as_asked" else self.judge.sends
        return self.judge.answer(sent, system, user)

    async def aclose(self) -> None:
        """Nothing to release."""


class _Kind:
    """A kind rendering :data:`_EVIDENCE` for its one output, as a document."""

    judged_artifact = JudgedArtifact.DOCUMENT

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: CellSpanWindow,
        cassettes: CellCassettes | None,
        world: Any,
    ) -> None:
        """Nothing to build."""

    async def invoke(self, instance: None, test_case: EvalTestCase, sink: CellSink) -> CandidateOutput:
        """Hand back one output and the evidence a judge reads of it."""
        return CandidateOutput(output=[{"rendered": "elsewhere"}], judge_evidence=_EVIDENCE)


async def _judged_run(judge: _Judge, *, cases: int = 4) -> tuple[EvalHost, str]:
    """Run ``cases`` judged cells through the real runner, stored as a finished run would be."""
    host = toyhost_host(clients=lambda role, model, *, temperature=None: judge.client(temperature))
    template = EvalTemplate(
        scope_id=_SCOPE,
        name="temperature",
        intent="run a fair encounter",
        candidate_kind=_KIND,
        rubric=[RubricDim(name=_DIM, description="the GM rules what the dice say", scale="ordinal")],
    )
    host.storage.save_template(template)
    test_cases = [
        EvalTestCase(template_id=template.id, scope_id=_SCOPE, variation_params={"party": f"p{index}"})
        for index in range(cases)
    ]
    for case in test_cases:
        host.storage.save_test_case(case)
    built = build_judge_service(host, template, _JUDGE, judged_artifact=JudgedArtifact.DOCUMENT)
    run = make_eval_run(
        scope_id=_SCOPE,
        template_id=template.id,
        candidate_kind=_KIND,
        candidate_model="candidate/m",
        test_case_ids=[case.id for case in test_cases],
        judge_model=_JUDGE,
        status="completed",
        effective_judges=built.effective_judges,
        judge_request_settings=JUDGE_REQUEST_SETTINGS,
    )
    host.storage.save_eval_run(run)
    for case in test_cases:
        outcome = await run_one_result(
            host,
            template=template,
            test_case=case,
            subject_id="subject-1",
            model=run.candidate_model,
            k_iteration=1,
            eval_run_id=run.id,
            scope_id=_SCOPE,
            judge_service=JudgeService(
                client_factory=lambda _model, temperature: judge.client(temperature),
                failure_describer=withhold_failure_detail,
            ),
            judge_model=_JUDGE,
            options=RunnerOptions(candidate_kinds={_KIND: lambda _cell: _Kind()}),
            variant=resolve_variant_identity(run=run, profile=host.profile),
        )
        host.storage.save_eval_result(outcome.result, outcome.trace)
    judge.calls.clear()
    judge.requested.clear()
    return host, run.id


def _results(host: EvalHost, run_id: str) -> list[EvalResult]:
    return sorted(host.storage.query_eval_results_by_run(run_id, _SCOPE), key=lambda result: result.id)


def _party(result: EvalResult) -> int:
    score = result.judge_score(_DIM)
    assert score is not None
    return score.score - 2


class TestTheComparisonShowsWhatTemperatureDoes:
    async def test_a_judge_deterministic_at_zero_and_noisy_at_default_reads_so_side_by_side(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)

        comparison = await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=None, repeats=4)

        assert comparison.comparable and comparison.incomparable is None
        # Parties 0-2 stored 2, 3 and 4 (borderline); party 3 stored a 5, an end of the scale.
        assert (comparison.results, comparison.cases) == (3, 3)
        (row,) = comparison.dimensions
        assert row.rubric_dim == _DIM
        assert (row.pinned.cases, row.pinned.mean_variance, row.pinned.unstable_cases) == (3, 0.0, 0)
        assert row.pinned.exact_agreement == 1.0
        assert row.provider_default.cases == 3
        assert row.provider_default.mean_variance is not None and row.provider_default.mean_variance > 0
        assert row.provider_default.unstable_cases == 3
        assert row.provider_default.exact_agreement is not None and row.provider_default.exact_agreement < 1.0
        pinned, default = comparison.settings
        assert (pinned.setting, pinned.requested, pinned.recorded) == ("pinned", DEFAULT_JUDGE_TEMPERATURE, ["0"])
        assert (default.setting, default.requested, default.recorded) == (
            "provider_default",
            None,
            [MODEL_DEFAULT_TEMPERATURE],
        )
        assert all(len(case.scores) == 4 for read in comparison.settings for case in read.cases)
        rendered = comparison.render()
        assert "temperature 0" in rendered and "provider default" in rendered and "NOT COMPARABLE" not in rendered

    async def test_each_side_is_sent_its_own_temperature(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)

        await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=None, repeats=2)

        assert set(judge.requested) == {DEFAULT_JUDGE_TEMPERATURE, None}
        sent = [temperature for temperature, _ in judge.calls]
        assert sent.count(DEFAULT_JUDGE_TEMPERATURE) == sent.count(None) == 3 * 2

    async def test_nothing_is_written_to_the_results(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)
        before = _results(host, run_id)

        await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=None, repeats=2)

        assert _results(host, run_id) == before


class TestWhichCasesAreReJudged:
    async def test_a_stored_score_at_an_end_of_the_scale_is_skipped_and_named(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)
        (top,) = [result for result in _results(host, run_id) if _party(result) == 3]

        comparison = await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=None, repeats=2)

        assert [skip.result_id for skip in comparison.skipped] == [top.id]
        assert "borderline" in comparison.skipped[0].reason

    async def test_a_recorded_repeat_that_disagreed_makes_an_end_of_scale_score_borderline(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)
        (top,) = [result for result in _results(host, run_id) if _party(result) == 3]
        assert borderline_dims(top) == set()
        disagreed = top.model_copy(
            update={
                "judge_repeats": [
                    JudgeRepeat(
                        judge_model=_JUDGE,
                        scores=[
                            RepeatedScore(
                                dim=_DIM,
                                scale="ordinal",
                                first_score=5,
                                first_served_model=_JUDGE,
                                first_judge_config_id=None,
                                repeat=RubricScore(dim=_DIM, scale="ordinal", score=4, served_model=_JUDGE),
                            )
                        ],
                    )
                ]
            }
        )
        assert borderline_dims(disagreed) == {_DIM}
        _, etag = host.storage.load_eval_result_with_etag(top.id, _SCOPE)
        host.storage.replace_eval_result(disagreed, if_match=etag)

        comparison = await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=None, repeats=2)

        assert comparison.skipped == [] and comparison.cases == 4

    async def test_all_takes_every_scored_dim(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)

        comparison = await compare_judge_temperatures(
            host, run_id, _SCOPE, out_of_run_cap_usd=None, repeats=2, selection="all"
        )

        assert (comparison.cases, comparison.skipped) == (4, [])

    async def test_a_run_with_no_borderline_case_is_refused_naming_the_way_out(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)
        (top,) = [result for result in _results(host, run_id) if _party(result) == 3]

        with pytest.raises(ValidationFailedError, match="selection='all'"):
            await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=None, result_ids=[top.id])
        assert judge.calls == []

    async def test_fewer_than_two_repeats_is_refused(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)

        with pytest.raises(ValidationFailedError, match="repeats 1"):
            await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=None, repeats=1)
        assert judge.calls == []


class TestAClientThatIgnoresTemperature:
    async def test_a_judge_whose_stored_scores_record_no_temperature_sent_is_refused_before_any_spend(self):
        judge = _Judge(sends=None)
        host, run_id = await _judged_run(judge)
        assert {result.judge_score(_DIM).judge_temperature for result in _results(host, run_id)} == {  # type: ignore[union-attr]
            MODEL_DEFAULT_TEMPERATURE
        }

        with pytest.raises(ValidationFailedError, match="refuses one"):
            await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=1.0)
        assert judge.calls == []
        assert host.storage.query_out_of_run_spend(_SCOPE, purpose="judge") == []

    async def test_a_client_that_drops_the_temperature_at_call_time_makes_the_comparison_not_comparable(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)
        judge.sends = None  # from here the client sends none whatever it is asked, and says so

        comparison = await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=None, repeats=2)

        assert not comparison.comparable
        assert comparison.incomparable is not None and "pinned" in comparison.incomparable
        pinned, default = comparison.settings
        assert pinned.off_setting == 3 * 2 and default.off_setting == 0
        assert pinned.recorded == [MODEL_DEFAULT_TEMPERATURE]
        assert all(case.scores == [] for case in pinned.cases), "an off-setting score is never read as the setting's"
        (row,) = comparison.dimensions
        assert (row.pinned.cases, row.pinned.mean_variance) == (0, None)
        assert "NOT COMPARABLE" in comparison.render()

    async def test_a_client_that_reports_no_temperature_makes_the_comparison_not_comparable(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)
        judge.sends = _UNREPORTED

        comparison = await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=None, repeats=2)

        assert not comparison.comparable
        assert [read.recorded for read in comparison.settings] == [["unrecorded"], ["unrecorded"]]
        assert [read.off_setting for read in comparison.settings] == [6, 6]


class TestEveryCallIsMetered:
    async def test_every_call_is_a_ledger_row_under_judge_stamped_with_the_run(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)

        comparison = await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=1.0, repeats=3)

        rows = host.storage.query_out_of_run_spend(_SCOPE, purpose="judge")
        assert len(rows) == len(judge.calls) == comparison.calls_made == 2 * 3 * 3
        assert {(row.run_id, row.purpose, row.outcome, row.priced_ceiling_usd) for row in rows} == {
            (run_id, "judge", "completed", _CEILING)
        }
        assert comparison.cost_usd == pytest.approx(sum(row.cost_usd or 0 for row in rows))
        assert comparison.cap_usd == 1.0

    async def test_the_estimate_prices_every_attempt_of_both_sides_and_calls_nothing(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)

        estimate = await estimate_judge_temperature_comparison(host, run_id, _SCOPE, out_of_run_cap_usd=1.0, repeats=3)

        assert judge.calls == []
        assert (estimate.results, estimate.cases) == (3, 3)
        assert estimate.max_calls == 2 * 3 * 3 * JUDGE_CALL_ATTEMPTS
        assert estimate.ceiling_usd == pytest.approx(estimate.max_calls * _CEILING)
        assert estimate.would_start and estimate.refusal is None
        assert "would start" in estimate.render()

    async def test_a_cent_under_the_cap_nothing_is_called_and_nothing_is_ledgered(self):
        judge = _Judge()
        host, run_id = await _judged_run(judge)
        total = 2 * 3 * 2 * JUDGE_CALL_ATTEMPTS * _CEILING

        estimate = await estimate_judge_temperature_comparison(
            host, run_id, _SCOPE, out_of_run_cap_usd=total - 0.01, repeats=2
        )
        assert not estimate.would_start and estimate.refusal is not None
        with pytest.raises(ValidationFailedError):
            await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=total - 0.01, repeats=2)

        assert judge.calls == []
        assert host.storage.query_out_of_run_spend(_SCOPE, purpose="judge") == []
        comparison = await compare_judge_temperatures(host, run_id, _SCOPE, out_of_run_cap_usd=total, repeats=2)
        assert comparison.calls_made == 2 * 3 * 2


class TestTheCommandLine:
    async def test_one_command_prices_it_then_runs_it_under_the_cap_it_names(self, capsys: pytest.CaptureFixture[str]):
        judge = _Judge()
        host, run_id = await _judged_run(judge)
        argv = ["judge-temperature", run_id, "--scope", _SCOPE, "--max-cost-usd", "1", "--repeats", "2"]

        assert await _cli([*argv, "--estimate"], host) == 0
        assert judge.calls == []
        assert "would start" in capsys.readouterr().out

        assert await _cli([*argv, "--json"], host) == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["comparable"] is True and printed["calls_made"] == 2 * 3 * 2
        assert len(host.storage.query_out_of_run_spend(_SCOPE, purpose="judge")) == 2 * 3 * 2

    async def test_a_cap_is_named_or_waived_out_loud(self, capsys: pytest.CaptureFixture[str]):
        judge = _Judge()
        host, run_id = await _judged_run(judge)

        with pytest.raises(SystemExit) as no_cap:
            await _cli(["judge-temperature", run_id, "--scope", _SCOPE], host)
        assert no_cap.value.code == 2
        assert await _cli(["judge-temperature", run_id, "--scope", _SCOPE, "--max-cost-usd", "0"], host) == 2
        assert judge.calls == []


async def _cli(argv: list[str], host: EvalHost) -> int:
    """Run the command line in a worker thread — it runs its own event loop, which cannot nest in the test's."""
    import asyncio

    return await asyncio.to_thread(run_cli, argv, host_factory=lambda: host)
