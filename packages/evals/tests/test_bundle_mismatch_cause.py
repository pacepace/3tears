"""A re-assembly that no longer reproduces its generation says which of three things moved, or that it cannot say (#575).

A host's apparatus sweepable is a dimension of every apparatus class, so a host editing its own
declarations moves every fingerprint over unchanged runs. The package's versions cannot record that,
so the bundle carries a digest derived from the declarations, the generation's provenance records it
beside the bundle ``schema_version`` it ran over, and :class:`BundleInspection` attributes a mismatch to
the package's shape, the host's declarations or the evidence — and to ``cannot_say`` for an analysis
stored before either was recorded.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from threetears.evals.analysis import (
    inspect_analysis_bundle,
    prepare_analysis_generation,
    run_analysis_generation,
)
from threetears.evals.analysis.bundle import host_declarations_digest
from threetears.evals.contracts.campaign import EvalAnalysis
from threetears.evals.contracts.host import EvalHost
from threetears.evals.contracts.host.profile import HostProfile
from threetears.evals.contracts.host.sweepables import Sweepable
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.toyhost_memo import PROMPT_ID, FixturedClient, memo_payload


def _with_a_lamp(profile: HostProfile, *, name: str = "bench_lamp") -> HostProfile:
    """The toy host, declaring one more apparatus sweepable — a constant every run reads alike."""
    lamp = Sweepable(
        name=name,
        role="apparatus",
        read=lambda _run, _results: "on",
        reader_prose="whether the bench lamp over the scanner was lit",
        confounds="a different lamp changes the scan the extractor reads",
    )
    # Seated on every kind that names its seats, so the extractor's observations carry it.
    kinds = tuple(kind if kind.seats is None else replace(kind, seats=kind.seats | {name}) for kind in profile.kinds)
    return replace(profile, host_sweepables=profile.host_sweepables.extend([lamp]), kinds=kinds)


def _corpus_host(profile: HostProfile | None = None) -> tuple[EvalHost, str, str]:
    host = toyhost_host(profile=profile, clients=lambda role, model, *, temperature=None: FixturedClient(""))
    campaign, corpus = toyhost_campaign()
    for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
        host.storage.save_eval_run(run)
        for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
            host.storage.save_eval_result(result)
    host.storage.save_campaign(campaign)
    return host, campaign.id, campaign.scope_id


async def _prompt() -> str:
    return "Write the analysis."


async def _generated(host: EvalHost, campaign_id: str, scope_id: str) -> EvalAnalysis:
    prepared = await prepare_analysis_generation(
        host, campaign_id, scope_id, model=None, resolve_prompt=_prompt, out_of_run_cap_usd=None
    )
    prepared.client.completion.content = json.dumps(memo_payload(prepared.bundle))  # type: ignore[attr-defined]
    analysis, _insights = await run_analysis_generation(host, prepared, prompt_id=PROMPT_ID, max_output_tokens=4000)
    return analysis


def _restored(host: EvalHost, analysis: EvalAnalysis, **generation: Any) -> None:
    """Store the analysis again with its provenance altered — what an older build would have written."""
    host.storage.save_analysis(
        analysis.model_copy(update={"generation": analysis.generation.model_copy(update=generation)})
    )


class TestTheDigestIsDerivedFromTheDeclarations:
    def test_the_same_declarations_digest_alike(self) -> None:
        assert host_declarations_digest(toyhost_profile()) == host_declarations_digest(toyhost_profile())

    def test_adding_an_apparatus_sweepable_moves_it(self) -> None:
        base = toyhost_profile()

        assert host_declarations_digest(_with_a_lamp(base)) != host_declarations_digest(base)

    def test_renaming_one_moves_it(self) -> None:
        base = toyhost_profile()

        assert host_declarations_digest(_with_a_lamp(base)) != host_declarations_digest(
            _with_a_lamp(base, name="desk_lamp")
        )

    async def test_the_generation_records_it_and_the_bundle_version(self) -> None:
        host, campaign_id, scope_id = _corpus_host()

        analysis = await _generated(host, campaign_id, scope_id)

        assert analysis.generation.host_declarations_digest == host_declarations_digest(host.profile)
        inspection = inspect_analysis_bundle(host, analysis.id, scope_id)
        assert analysis.generation.bundle_schema_version == inspection.bundle.schema_version
        assert inspection.reproduces_generation is True
        assert inspection.mismatch_cause is None


class TestAMismatchNamesWhatMoved:
    async def test_a_host_adding_an_apparatus_sweepable_reads_as_host_declarations(self) -> None:
        host, campaign_id, scope_id = _corpus_host()
        analysis = await _generated(host, campaign_id, scope_id)

        relit = toyhost_host(profile=_with_a_lamp(host.profile), storage=host.storage)
        inspection = inspect_analysis_bundle(relit, analysis.id, scope_id)

        assert inspection.reproduces_generation is False
        assert inspection.mismatch_cause == "host_declarations"
        assert inspection.recorded_host_declarations_digest != inspection.bundle.host_declarations_digest

    async def test_a_schema_version_bump_alone_reads_as_package_shape(self) -> None:
        host, campaign_id, scope_id = _corpus_host()
        analysis = await _generated(host, campaign_id, scope_id)
        current = inspect_analysis_bundle(host, analysis.id, scope_id).bundle
        # What the same evidence digested to under the version before: only the shape version differs.
        older = current.model_copy(update={"schema_version": current.schema_version - 1})
        _restored(host, analysis, bundle_schema_version=older.schema_version, bundle_fingerprint=older.fingerprint())

        inspection = inspect_analysis_bundle(host, analysis.id, scope_id)

        assert inspection.reproduces_generation is False
        assert inspection.mismatch_cause == "package_shape"

    async def test_a_shape_bump_wins_over_a_declarations_change(self) -> None:
        host, campaign_id, scope_id = _corpus_host()
        analysis = await _generated(host, campaign_id, scope_id)
        _restored(host, analysis, bundle_schema_version=1)

        relit = toyhost_host(profile=_with_a_lamp(host.profile), storage=host.storage)

        assert inspect_analysis_bundle(relit, analysis.id, scope_id).mismatch_cause == "package_shape"

    async def test_unchanged_versions_over_moved_runs_read_as_evidence(self) -> None:
        host, campaign_id, scope_id = _corpus_host()
        analysis = await _generated(host, campaign_id, scope_id)
        run = host.storage.load_eval_run(analysis.observation_refs[0], scope_id)
        assert run is not None
        host.storage.save_eval_run(run.model_copy(update={"archived": True}))

        inspection = inspect_analysis_bundle(host, analysis.id, scope_id)

        assert inspection.reproduces_generation is False
        assert inspection.mismatch_cause == "evidence"

    async def test_a_provenance_stored_without_the_fields_cannot_say(self) -> None:
        host, campaign_id, scope_id = _corpus_host()
        analysis = await _generated(host, campaign_id, scope_id)
        _restored(host, analysis, bundle_schema_version=None, host_declarations_digest=None)

        relit = toyhost_host(profile=_with_a_lamp(host.profile), storage=host.storage)
        inspection = inspect_analysis_bundle(relit, analysis.id, scope_id)

        assert inspection.reproduces_generation is False
        assert inspection.mismatch_cause == "cannot_say"

    async def test_one_missing_field_is_enough_to_say_nothing(self) -> None:
        host, campaign_id, scope_id = _corpus_host()
        analysis = await _generated(host, campaign_id, scope_id)
        _restored(host, analysis, host_declarations_digest=None)

        relit = toyhost_host(profile=_with_a_lamp(host.profile), storage=host.storage)

        assert inspect_analysis_bundle(relit, analysis.id, scope_id).mismatch_cause == "cannot_say"

    def test_a_stored_provenance_without_the_fields_still_loads(self) -> None:
        from threetears.evals.contracts.campaign import GenerationProvenance

        stored = {
            "prompt_id": "p",
            "prompt_version": "v",
            "generator_model": "m",
            "bundle_fingerprint": "0" * 64,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "bundle_assembled_at": "2026-01-01T00:00:00+00:00",
            "token_cost": 0.0,
            "repair_attempts": 0,
            "repaired_refusal": None,
            "cell_model_version": 11,
            "user_message_digest": "d",
        }

        provenance = GenerationProvenance.model_validate(stored)

        assert (provenance.bundle_schema_version, provenance.host_declarations_digest) == (None, None)
