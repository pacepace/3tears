"""The bundle's growing lists are bounded and say what they left out; the insight ledger does not grow on a restatement.

Three lists grow with the campaign or the ledger rather than with what a reader can act on, and each
rode whole into the paid analysis prompt: the subject's prior insights (#614), the refused merges and
the next experiments (#572). Each is capped with what it dropped counted beside it, as
``divergences_omitted`` is for the divergences. And a regenerated analysis replaces the live insights
it restates instead of adding a row per regeneration (#614).
"""

from __future__ import annotations

import json
from typing import Any

from threetears.evals.analysis import prepare_analysis_generation, run_analysis_generation
from threetears.evals.analysis.bundle import assemble_context_bundle, superseding_insights
from threetears.evals.analysis.generator import build_user_message
from threetears.evals.kernel.campaign import EvalCampaign, EvalInsight
from threetears.evals.kernel.host import EvalHost
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    ToyhostStorage,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import PROMPT_ID, FixturedClient, memo_payload

_CLAIM = "A wider chunk is slower per document on this corpus."


def _insight(index: int, *, statement: str | None = None, source: str = "") -> EvalInsight:
    return EvalInsight(
        scope_id=TOYHOST_SCOPE,
        id=f"i-{index:03d}",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        statement=statement if statement is not None else f"claim number {index}",
        confidence="medium",
        observed_at=f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}+00:00",
        source_analysis_id=source,
    )


def _bundle_over(insights: list[EvalInsight]) -> Any:
    campaign, storage = toyhost_campaign(insights=insights)
    return assemble_context_bundle(campaign, storage=storage, profile=toyhost_profile())


# =============================================================================
# prior_insights (#614)
# =============================================================================


class TestPriorInsightsAreBounded:
    def test_a_ledger_longer_than_the_cap_yields_a_capped_list_and_counts_the_rest(self) -> None:
        ledger = [_insight(index) for index in range(40)]

        bundle = _bundle_over(ledger)

        assert 0 < len(bundle.prior_insights) < len(ledger)
        assert bundle.prior_insights_omitted == len(ledger) - len(bundle.prior_insights)

    def test_the_newest_are_kept_newest_first(self) -> None:
        ledger = [_insight(index) for index in range(40)]

        carried = [insight.id for insight in _bundle_over(ledger).prior_insights]

        assert carried == [f"i-{index:03d}" for index in range(39, 39 - len(carried), -1)]

    def test_the_selection_does_not_depend_on_the_ledgers_order(self) -> None:
        ledger = [_insight(index) for index in range(40)]

        assert _bundle_over(ledger).fingerprint() == _bundle_over(list(reversed(ledger))).fingerprint()

    def test_a_short_ledger_is_carried_whole_with_nothing_omitted(self) -> None:
        bundle = _bundle_over([_insight(index) for index in range(3)])

        assert len(bundle.prior_insights) == 3
        assert bundle.prior_insights_omitted == 0

    def test_one_claim_stated_twice_is_carried_once_and_the_older_counted(self) -> None:
        older = _insight(1, statement=_CLAIM)
        newer = _insight(2, statement="  a wider chunk is SLOWER per document on this corpus ")

        bundle = _bundle_over([older, newer])

        assert [insight.id for insight in bundle.prior_insights] == [newer.id]
        assert bundle.prior_insights_omitted == 1

    def test_a_retracted_insight_is_not_counted_as_omitted(self) -> None:
        retracted = _insight(1, source="an-archived")
        campaign, storage = toyhost_campaign(insights=[retracted, _insight(2)])

        class _Archived(ToyhostStorage):
            def analysis_archived(self, analysis_id: str, scope_id: str) -> bool | None:
                return True if analysis_id == "an-archived" else None

        archived = _Archived([], {})
        archived.__dict__.update(storage.__dict__)
        bundle = assemble_context_bundle(campaign, storage=archived, profile=toyhost_profile())

        assert bundle.retracted_insights == {retracted.id: "an-archived"}
        assert bundle.prior_insights_omitted == 0


# =============================================================================
# Supersession (#614)
# =============================================================================


class TestARestatedInsightReplacesTheOneItRestates:
    def test_a_restatement_takes_the_live_insights_id(self) -> None:
        live = _insight(1, statement=_CLAIM, source="an-earlier")
        minted = _insight(99, statement=_CLAIM.upper(), source="the-new")

        (written,) = superseding_insights([minted], [live], lambda _source: False)

        assert written.id == live.id
        assert written.source_analysis_id == "the-new"

    def test_a_new_claim_keeps_its_own_id(self) -> None:
        minted = _insight(99, statement="something nobody said before")

        (written,) = superseding_insights([minted], [_insight(1, statement=_CLAIM)], lambda _source: False)

        assert written.id == minted.id

    def test_a_retracted_insight_is_never_replaced(self) -> None:
        withdrawn = _insight(1, statement=_CLAIM, source="an-archived")
        minted = _insight(99, statement=_CLAIM)

        (written,) = superseding_insights([minted], [withdrawn], lambda source: source == "an-archived")

        assert written.id == minted.id, "restating a withdrawn claim mints it afresh"

    def test_one_claim_stated_twice_by_one_generation_is_written_once(self) -> None:
        written = superseding_insights(
            [_insight(98, statement=_CLAIM), _insight(99, statement=_CLAIM + ".")], [], lambda _source: None
        )

        assert [insight.id for insight in written] == ["i-098"]


def _corpus_host() -> tuple[EvalHost, str, str]:
    host = toyhost_host(clients=lambda role, model, *, temperature=None: FixturedClient(""))
    campaign, corpus = toyhost_campaign()
    for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
        host.storage.save_eval_run(run)
        for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
            host.storage.save_eval_result(result)
    host.storage.save_campaign(campaign)
    return host, campaign.id, campaign.scope_id


async def _prompt() -> str:
    return "Write the analysis."


async def _generate_once(host: EvalHost, campaign_id: str, scope_id: str) -> Any:
    prepared = await prepare_analysis_generation(
        host, campaign_id, scope_id, model=None, resolve_prompt=_prompt, out_of_run_cap_usd=None
    )
    payload = memo_payload(prepared.bundle)
    payload["findings"][0]["durable"] = _CLAIM
    prepared.client.completion.content = json.dumps(payload)  # type: ignore[attr-defined]
    return await run_analysis_generation(host, prepared, prompt_id=PROMPT_ID, max_output_tokens=4000)


class TestRegeneratingLeavesTheLedgerItsSize:
    async def test_generating_twice_on_an_unchanged_campaign_writes_one_insight(self) -> None:
        host, campaign_id, scope_id = _corpus_host()
        subject = TOYHOST_SUBJECT.subject_id

        first, _ = await _generate_once(host, campaign_id, scope_id)
        after_first = host.storage.query_insights(scope_id, subject_id=subject)
        second, _ = await _generate_once(host, campaign_id, scope_id)
        after_second = host.storage.query_insights(scope_id, subject_id=subject)

        assert len(after_first) == len(after_second) == 1
        (insight,) = after_second
        assert insight.id == after_first[0].id
        assert insight.source_analysis_id == second.id != first.id

    async def test_the_trigger_names_what_retires_the_insight(self) -> None:
        host, campaign_id, scope_id = _corpus_host()

        analysis, (insight,) = await _generate_once(host, campaign_id, scope_id)

        assert analysis.id in insight.invalidation_trigger
        assert "archived" in insight.invalidation_trigger
        assert "same claim" in insight.invalidation_trigger

    async def test_archiving_the_minting_analysis_lets_a_restatement_mint_afresh(self) -> None:
        host, campaign_id, scope_id = _corpus_host()
        first, (minted,) = await _generate_once(host, campaign_id, scope_id)
        host.storage.save_analysis(first.model_copy(update={"archived": True}))

        _second, (again,) = await _generate_once(host, campaign_id, scope_id)

        assert again.id != minted.id
        assert len(host.storage.query_insights(scope_id, subject_id=TOYHOST_SUBJECT.subject_id)) == 2


class TestTheWriterIsNotBilledForTheEnginesBookkeeping:
    def test_neither_the_trigger_nor_the_declarations_digest_reaches_the_writer(self) -> None:
        bundle = _bundle_over([_insight(1).model_copy(update={"invalidation_trigger": "Retracted if X."})])

        message = build_user_message(bundle)

        assert "invalidation_trigger" not in message
        assert "host_declarations_digest" not in message
        assert "prior_insights_omitted" in message


# =============================================================================
# refused_merges and next_experiments (#572)
# =============================================================================

#: Enough variants that one refusal and one recording per variant pass both caps.
_LEVELS = tuple(range(128, 128 * 13, 128))


def _many_half_recorded_arms() -> Any:
    """Twelve chunk widths, each measured once with its grader recorded and once without.

    Each variant is two cells kept apart only by the unrecorded grader: one refused merge and one
    recording that would pool them, per variant.
    """
    profile = toyhost_profile()
    runs, results = [], {}
    for level in _LEVELS:
        for grader in ("g-1", ""):
            run = toyhost_batch(
                chunk_tokens=level,
                retriever_top_k=3,
                extraction_schema="v1",
                ocr_engine_version="tess-5.3.1",
                reviewer_pool="pool-a",
                grader_version=grader,
            )
            runs.append(run)
            results[run.id] = toyhost_measurements(
                run, profile=profile, cost_usd=0.02, total_ms=900.0 + level, field_accuracy=0.8
            )
    base, _storage = toyhost_campaign()
    campaign = EvalCampaign(
        id="b0e1c6a2-5a2d-4c84-9c55-3f0a7e2c1d10",
        scope_id=TOYHOST_SCOPE,
        name="many widths, half the graders unrecorded",
        subject_id=base.subject_id,
        subject_kind=base.subject_kind,
        behavior=base.behavior,
        template_id="",
        run_ids=[run.id for run in runs],
        created_by="test:fixture",
    )
    return assemble_context_bundle(campaign, storage=ToyhostStorage(runs, results), profile=profile)


class TestRefusedMergesAndNextExperimentsAreBounded:
    def test_both_lists_are_capped_with_nonzero_omitted_counts(self) -> None:
        bundle = _many_half_recorded_arms()

        assert bundle.refused_merges_omitted > 0
        assert bundle.next_experiments_omitted > 0
        assert len(bundle.refused_merges) + bundle.refused_merges_omitted == len(_LEVELS)
        assert len(bundle.next_experiments) + bundle.next_experiments_omitted == len(_LEVELS)

    def test_the_kept_entries_are_the_heaviest_and_stay_sorted(self) -> None:
        bundle = _many_half_recorded_arms()

        kept = [entry.variant_key for entry in bundle.next_experiments]
        assert kept == sorted(kept)
        merges = [(merge.variant_key, merge.apparatus_class_ids) for merge in bundle.refused_merges]
        assert merges == sorted(merges)

    def test_the_selection_is_deterministic(self) -> None:
        assert _many_half_recorded_arms().fingerprint() == _many_half_recorded_arms().fingerprint()

    def test_an_ordinary_campaign_omits_nothing(self) -> None:
        bundle = _bundle_over([])

        assert (bundle.refused_merges_omitted, bundle.next_experiments_omitted) == (0, 0)
