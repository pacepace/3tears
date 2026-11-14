"""A re-assembly that no longer reproduces its generation says which of three things moved, or that it cannot say (#575).

A host's apparatus sweepable is a dimension of every apparatus class, so a host editing its own
declarations moves every fingerprint over unchanged runs. The package's versions cannot record that,
so the bundle carries a digest derived from the declarations, the generation's provenance records it
beside the bundle ``schema_version`` it ran over, and :class:`BundleInspection` attributes a mismatch to
the package's shape, the host's declarations or the evidence — and to ``cannot_say`` for an analysis
stored before either was recorded.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from threetears.evals.analysis import assemble_campaign_bundle, inspect_analysis_bundle
from threetears.evals.analysis.bundle.schema import host_declarations_digest
from threetears.evals.kernel.campaign import EvalAnalysis
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.host.sweepables import Sweepable, SweepableRegistry
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.writer_host_support import doc_analysis_over


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
    host = toyhost_host(profile=profile)
    campaign, corpus = toyhost_campaign()
    for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
        host.storage.save_eval_run(run)
        for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
            host.storage.save_eval_result(result)
    host.storage.save_campaign(campaign)
    return host, campaign.id, campaign.scope_id


async def _generated(host: EvalHost, campaign_id: str, scope_id: str) -> EvalAnalysis:
    """The campaign's analysis as a generation stores it: the report writer's, over the bundle the host assembles."""
    bundle = off_loop(assemble_campaign_bundle, host, campaign_id, scope_id)
    analysis, _insights = doc_analysis_over(bundle)
    off_loop(host.storage.save_analysis, analysis)
    return analysis


def _restored(host: EvalHost, analysis: EvalAnalysis, **generation: Any) -> None:
    """Store the analysis again with its provenance altered — what an older build would have written.

    A recorded ``bundle_fingerprint`` is the aspect's: each aspect's bundle is fingerprinted where it is frozen.
    """
    update: dict[str, Any] = {"generation": analysis.generation.model_copy(update=generation)}
    if "bundle_fingerprint" in generation:
        (aspect,) = analysis.aspects
        update["aspects"] = [aspect.model_copy(update={"bundle_fingerprint": generation["bundle_fingerprint"]})]
    host.storage.save_analysis(analysis.model_copy(update=update))


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

    def test_taking_an_apparatus_sweepable_out_of_the_rig_moves_it(self) -> None:
        """A declaration that stops splitting cells moves every apparatus class, so it moves the digest too."""
        base = _with_a_lamp(toyhost_profile())
        (lamp,) = [d for d in base.host_sweepables.declarations if d.name == "bench_lamp"]
        waived = replace(lamp, confounds=None, no_rig_coordinate="the lamp lights the bench, never the scan")
        out_of_rig = replace(
            base,
            host_sweepables=SweepableRegistry(
                [waived if d.name == "bench_lamp" else d for d in base.host_sweepables.declarations],
                roles=base.host_sweepables.roles,
            ),
        )

        assert host_declarations_digest(out_of_rig) != host_declarations_digest(base)

    async def test_the_generation_records_it_and_the_bundle_version(self) -> None:
        host, campaign_id, scope_id = off_loop(_corpus_host)

        analysis = await _generated(host, campaign_id, scope_id)

        assert analysis.generation.host_declarations_digest == host_declarations_digest(host.profile)
        inspection = off_loop(inspect_analysis_bundle, host, analysis.id, scope_id)
        assert analysis.generation.bundle_schema_version == inspection.bundle.schema_version
        assert inspection.reproduces_generation is True
        assert inspection.mismatch_cause is None


class TestAMismatchNamesWhatMoved:
    async def test_a_host_adding_an_apparatus_sweepable_reads_as_host_declarations(self) -> None:
        host, campaign_id, scope_id = off_loop(_corpus_host)
        analysis = await _generated(host, campaign_id, scope_id)

        relit = toyhost_host(profile=_with_a_lamp(host.profile), storage=host.storage)
        inspection = off_loop(inspect_analysis_bundle, relit, analysis.id, scope_id)

        assert inspection.reproduces_generation is False
        assert inspection.mismatch_cause == "host_declarations"
        assert inspection.recorded_host_declarations_digest != inspection.bundle.host_declarations_digest

    async def test_a_schema_version_bump_alone_reads_as_package_shape(self) -> None:
        host, campaign_id, scope_id = off_loop(_corpus_host)
        analysis = await _generated(host, campaign_id, scope_id)
        current = off_loop(inspect_analysis_bundle, host, analysis.id, scope_id).bundle
        # What the same evidence digested to under the version before: only the shape version differs. Recorded as
        # `v2:`, the fingerprint that hashes the bundle's shape; a `v3:` record reads no shape
        # (`test_evidence_fingerprint.py`).
        older = current.model_copy(update={"schema_version": current.schema_version - 1})
        off_loop(
            _restored,
            host,
            analysis,
            bundle_schema_version=older.schema_version,
            bundle_fingerprint=older.fingerprint(like="v2:"),
        )

        inspection = off_loop(inspect_analysis_bundle, host, analysis.id, scope_id)

        assert inspection.reproduces_generation is False
        assert inspection.mismatch_cause == "package_shape"

    async def test_a_shape_bump_wins_over_a_declarations_change(self) -> None:
        host, campaign_id, scope_id = off_loop(_corpus_host)
        analysis = await _generated(host, campaign_id, scope_id)
        # A `v2:` record, the fingerprint that hashes the bundle's shape: against a `v3:` one no shape is a cause.
        current = off_loop(inspect_analysis_bundle, host, analysis.id, scope_id).bundle
        off_loop(_restored, host, analysis, bundle_schema_version=1, bundle_fingerprint=current.fingerprint(like="v2:"))

        relit = toyhost_host(profile=_with_a_lamp(host.profile), storage=host.storage)

        assert off_loop(inspect_analysis_bundle, relit, analysis.id, scope_id).mismatch_cause == "package_shape"

    async def test_unchanged_versions_over_moved_runs_read_as_evidence(self) -> None:
        host, campaign_id, scope_id = off_loop(_corpus_host)
        analysis = await _generated(host, campaign_id, scope_id)
        run = off_loop(host.storage.load_eval_run, analysis.observation_refs[0], scope_id)
        assert run is not None
        off_loop(host.storage.save_eval_run, run.model_copy(update={"archived": True}))

        inspection = off_loop(inspect_analysis_bundle, host, analysis.id, scope_id)

        assert inspection.reproduces_generation is False
        assert inspection.mismatch_cause == "evidence"
