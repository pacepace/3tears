"""A completion whose provider omitted its token counts: unknown at every reader, never zero.

``CompletionResult.input_tokens`` / ``output_tokens`` are ``int | None``, and ``None`` means the
provider did not report the count. The package's rule for spend — unpriced is unknown, never $0 —
holds for tokens too, so an omitted count must reach every usage row and every rollup as unknown.

The end-to-end drive is the toy host's judged run with a judge that reports its price and omits its
counts: the counts cross the judge's attempt accumulator, the judge's ``CallUsage``, the role
ledger, the stored result, and the analysis bundle's token rollup. The candidate's own client still
reports, so the rollup's sums are the candidate's alone and say so through
``n_results_tokens_unreported``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from threetears.evals.analysis import TokenRollup
from threetears.evals.analysis.generator import GenerationTally, generate_analysis
from threetears.evals.schema.completion import StopReason
from threetears.evals.kernel.provider import describe_incomplete_completion
from threetears.evals.schema.models import EvalResult, RoleUsage, utc_now_iso
from packages.evals.tests.bundle_support import one_batch_bundle
from packages.evals.tests.factories import make_eval_result
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_bundle
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.judge import (
    TOY_JUDGE_COST_USD,
    TOY_JUDGE_MODEL,
    ScriptedJudgeClient,
    toyhost_judge_service,
    toyhost_judged_template,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import ToyhostRunPath, execute_toyhost_run, toyhost_run_bundle
from packages.evals.tests.toyhost_memo import MODEL, PROMPT, PROMPT_ID, FixturedClient, memo_payload


async def _judged_drive(*, reports_token_counts: bool) -> ToyhostRunPath:
    return await execute_toyhost_run(
        host=toyhost_host(),
        template=toyhost_judged_template(),
        judge_service=toyhost_judge_service(ScriptedJudgeClient(reports_token_counts=reports_token_counts)),
        judge_model=TOY_JUDGE_MODEL,
    )


async def test_an_omitted_count_lands_on_the_judge_row_as_unknown_beside_a_known_price() -> None:
    path = await _judged_drive(reports_token_counts=False)
    assert path.results, "the judged drive produced no cells"
    for result in path.results:
        (judge_row,) = [row for row in result.usage if row.role == "judge"]
        assert judge_row.prompt_tokens is None
        assert judge_row.completion_tokens is None
        # The price was reported, so it is known: an omitted count says nothing about the dollars.
        assert judge_row.cost_usd is not None and abs(judge_row.cost_usd - TOY_JUDGE_COST_USD) < 1e-12
        (candidate_row,) = [row for row in result.usage if row.role == "candidate"]
        assert candidate_row.prompt_tokens is not None and candidate_row.completion_tokens is not None


async def test_the_bundle_rollup_sums_only_what_was_reported_and_counts_what_was_not() -> None:
    silent = await _judged_drive(reports_token_counts=False)
    reporting = await _judged_drive(reports_token_counts=True)

    silent_tokens = toyhost_run_bundle(silent).telemetry.tokens
    reporting_tokens = toyhost_run_bundle(reporting).telemetry.tokens
    assert silent_tokens is not None and reporting_tokens is not None

    candidate_prompt = sum(
        row.prompt_tokens or 0 for result in silent.results for row in result.usage if row.role == "candidate"
    )
    judge_prompt = sum(
        row.prompt_tokens or 0 for result in reporting.results for row in result.usage if row.role == "judge"
    )
    assert judge_prompt > 0, "the reporting judge must contribute counts, or the comparison below is vacuous"
    # The silent run's sum is the candidate's alone — the judge's unknown counts added nothing.
    assert silent_tokens.prompt_tokens == candidate_prompt
    assert reporting_tokens.prompt_tokens == candidate_prompt + judge_prompt
    # And the partial sum discloses itself, on every result whose judge omitted its counts.
    assert silent_tokens.n_results_tokens_unreported == len(silent.results)
    assert reporting_tokens.n_results_tokens_unreported == 0


def _token_rollup(results: list[EvalResult]) -> TokenRollup | None:
    """The bundle's token rollup over one batch of ``results`` — the telemetry a generation reads."""
    return one_batch_bundle(results, profile=toyhost_profile()).telemetry.tokens


def test_a_rollup_with_no_reported_count_is_unknown_not_zero() -> None:
    silent_row = RoleUsage(role="judge", model="j", cost_usd=0.01, price_source="p", call_count=1)
    results = [make_eval_result(usage=[silent_row]), make_eval_result(usage=[silent_row])]
    tokens = _token_rollup(results)
    assert tokens is not None
    assert (tokens.prompt_tokens, tokens.completion_tokens, tokens.reasoning_tokens) == (None, None, None)
    assert tokens.n_results_with_usage == 2
    assert tokens.n_results_tokens_unreported == 2


def test_an_external_row_is_not_a_token_count_left_unreported() -> None:
    external = RoleUsage(role="external", provider="search", provider_unit="credits", provider_units=3, call_count=1)
    candidate = RoleUsage(role="candidate", model="m", prompt_tokens=10, completion_tokens=5, call_count=1)
    tokens = _token_rollup([make_eval_result(usage=[candidate, external])])
    assert tokens is not None
    assert (tokens.prompt_tokens, tokens.completion_tokens) == (10, 5)
    assert tokens.n_results_tokens_unreported == 0


@dataclass(frozen=True)
class _Completion:
    content: str
    input_tokens: int | None
    output_tokens: int | None
    cost_usd: float | None
    price_source: str | None
    model: str
    served_model: str | None
    reasoning_tokens: int | None
    stop_reason: StopReason


def _silent(stop_reason: StopReason) -> _Completion:
    return _Completion(
        content="",
        input_tokens=None,
        output_tokens=None,
        cost_usd=0.02,
        price_source="p",
        model="m",
        served_model=None,
        reasoning_tokens=None,
        stop_reason=stop_reason,
    )


def test_a_truncation_with_no_reported_count_says_so_rather_than_printing_none() -> None:
    description = describe_incomplete_completion(_silent("max_tokens"))
    assert description is not None
    assert "output token count unreported" in description
    assert "None" not in description


async def test_the_generation_log_names_an_unreported_count(caplog) -> None:
    """Driven through the generator: the provider returns a well-formed memo and omits both counts."""
    bundle = toyhost_bundle(profile=toyhost_profile())
    client = FixturedClient(json.dumps(memo_payload(bundle)))
    client.completion.input_tokens = None
    client.completion.output_tokens = None

    caplog.set_level(logging.INFO, logger="threetears.evals.analysis.generator")
    tally = GenerationTally()
    await generate_analysis(
        bundle,
        prompt=PROMPT,
        model=MODEL,
        client=client,
        prompt_id=PROMPT_ID,
        bundle_assembled_at=utc_now_iso(),
        tally=tally,
        profile=toyhost_profile(),
    )
    (record,) = [r for r in caplog.records if "eval analysis generated" in r.getMessage()]
    assert "prompt_tokens=unreported completion_tokens=unreported" in record.getMessage()
    assert tally.returned == 1
