"""A bundle's fingerprint is over the evidence it was assembled from, never over a statistic computed from it (#1062).

Version 2 hashed the bundle's own document, so a better formula, another platform's maths library or a library update
read as "the evidence moved" and what was stored against the old fingerprint was regenerated. Version 3 hashes the
manifest of what the assembly read (:mod:`threetears.evals.analysis.bundle.evidence`). What is held here:

- **computing a statistic differently moves no fingerprint**: the small-sample factor swapped for the earlier one, and
  every computed float of a stored bundle moved, leave ``v3:`` where it was;
- **changing one piece of evidence moves it**, and the manifest names which kind moved and no other;
- **key order and a store's round trip move nothing**: the same corpus read back from a store that reorders every
  document's keys as ``jsonb`` does and hands whole floats back as integers;
- **numbers are rendered one way on every platform** (:func:`~threetears.evals.schema.hashing.input_json`);
- **what was stored under version 2 still reads**: it is compared the version 2 way, is never rewritten to version
  3, and a fingerprint's version alone is never reported as moved evidence;
- **the four hot-path campaigns' ``v2:`` fingerprints are reproduced** by this build through ``like=``, pinned as the
  build before it recorded them (``fixtures/hot_path_oracle/fingerprints_v2.json``); their ``v3:`` fingerprints are
  pinned, one value for every platform, by ``test_hot_path_oracle.py``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest

from threetears.evals.analysis import assemble_campaign_bundle
from threetears.evals.analysis import stats
from threetears.evals.analysis.bundle.evidence import (
    RESULT_EVIDENCE,
    RESULT_NOT_EVIDENCE,
    EvidenceManifest,
    manifest_fingerprint,
    result_evidence,
)
from threetears.evals.analysis.bundle.schema import (
    FINGERPRINT_PREFIX,
    AnalysisContextBundle,
    BundleInspection,
    evidence_digest,
)
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.kernel.campaign import EvalCampaign
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.result_reading import ResultReading
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.schema.hashing import input_digest, input_json
from threetears.evals.run.ratings import rate_result
from threetears.evals.schema.models import EvalResult, EvalTestCase
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.test_hot_path_oracle import WITH_BUNDLES, subject
from packages.evals.tests.test_result_reading_fields import campaign as reading_fields_campaign

FIXTURES = Path(__file__).parent / "fixtures"

#: Every component digest of a manifest: what "which kind of evidence moved" is read off.
COMPONENTS = ("declaration", "host_declarations", "runs", "results", "cases", "ratings", "judge_profiles", "insights")


def _corpus_host(storage: EvalStorage | None = None) -> tuple[EvalHost, EvalCampaign]:
    """The toy host holding its corpus campaign, in ``storage`` when one is given."""
    host = toyhost_host(storage=storage) if storage is not None else toyhost_host()
    campaign, corpus = toyhost_campaign()
    for run in corpus.load_eval_runs(campaign.run_ids, campaign.scope_id):
        host.storage.save_eval_run(run)
        for result in corpus.query_eval_results_by_run(run.id, campaign.scope_id):
            host.storage.save_eval_result(result)
    host.storage.save_campaign(campaign)
    return host, campaign


def _aspect(host: EvalHost, campaign: EvalCampaign) -> AnalysisContextBundle:
    (aspect,) = assemble_campaign_bundle(host, campaign.id, campaign.scope_id).aspects
    return aspect


def _manifest(bundle: AnalysisContextBundle) -> EvidenceManifest:
    assert bundle.evidence is not None, "a bundle assembled by this build carries its evidence manifest"
    return bundle.evidence


def _moved(before: AnalysisContextBundle, after: AnalysisContextBundle) -> set[str]:
    """Which components of the manifest differ between two assemblies."""
    was, now = _manifest(before), _manifest(after)
    return {name for name in COMPONENTS if getattr(was, name) != getattr(now, name)}


def _first_result(host: EvalHost, campaign: EvalCampaign) -> EvalResult:
    run_id = sorted(campaign.run_ids)[0]
    return sorted(host.storage.query_eval_results_by_run(run_id, campaign.scope_id), key=lambda r: r.id)[0]


# =============================================================================
# The fingerprint is the manifest's
# =============================================================================


def test_a_bundle_s_fingerprint_is_version_3_and_the_digest_of_its_manifest_alone() -> None:
    host, campaign = _corpus_host()
    bundle = _aspect(host, campaign)

    fingerprint = bundle.fingerprint()

    assert fingerprint.startswith("v3:") and FINGERPRINT_PREFIX == "v3:"
    assert len(fingerprint) == 3 + 64
    assert fingerprint == manifest_fingerprint(_manifest(bundle).to_dict())
    assert fingerprint == evidence_digest(bundle.to_dict())
    assert fingerprint == evidence_digest({"evidence": _manifest(bundle).to_dict()}), "nothing but the manifest is read"


def test_the_same_evidence_assembled_twice_fingerprints_alike() -> None:
    first = _aspect(*_corpus_host())
    second = _aspect(*_corpus_host())

    assert first.fingerprint() == second.fingerprint()
    assert _manifest(first) == _manifest(second)


def test_the_manifest_counts_what_was_read() -> None:
    host, campaign = _corpus_host()
    manifest = _manifest(_aspect(host, campaign))
    results = [
        result
        for run_id in campaign.run_ids
        for result in host.storage.query_eval_results_by_run(run_id, campaign.scope_id)
    ]

    assert manifest.n_runs == len(campaign.run_ids)
    assert manifest.n_results == len(results)
    stored_cases = host.storage.load_case_strata(sorted({result.test_case_id for result in results}), campaign.scope_id)
    assert manifest.n_cases == len(stored_cases)


# =============================================================================
# Computing a statistic differently moves no fingerprint
# =============================================================================


def test_swapping_the_small_sample_factor_moves_statistics_and_no_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The formula perturbation: Hedges' g computed with a different correction is a different bundle document, a
    different ``v2:`` pre-image, and the same ``v3:`` fingerprint."""
    read = subject("scaled")
    (exact,) = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id).aspects
    monkeypatch.setattr(stats, "hedges_j", lambda df: None if df < 2 else 1.0 - 3.0 / (4.0 * df - 1.0))
    (approximate,) = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id).aspects

    def effects(bundle: AnalysisContextBundle) -> list[float | None]:
        return [c.hedges_g for family in bundle.multiple_comparisons.families for c in family.comparisons]

    assert effects(exact) != effects(approximate), "the swapped factor must move a computed statistic to test anything"
    assert exact.fingerprint() == approximate.fingerprint()
    assert _manifest(exact) == _manifest(approximate)


def _perturbed(node: Any) -> Any:
    """``node`` with every float, at any depth, moved by a part in a thousand: every statistic computed another way."""
    if isinstance(node, bool):
        return node
    if isinstance(node, float):
        return node * 1.001 + 1e-9
    if isinstance(node, dict):
        return {key: _perturbed(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_perturbed(value) for value in node]
    return node


def test_every_computed_field_of_a_bundle_perturbed_moves_no_fingerprint() -> None:
    host, campaign = _corpus_host()
    bundle = _aspect(host, campaign)
    document = bundle.to_dict()
    manifest = document.pop("evidence")

    perturbed = {**_perturbed(document), "evidence": manifest}

    assert perturbed != {**document, "evidence": manifest}, "the perturbation must move the document"
    assert evidence_digest(perturbed) == bundle.fingerprint()
    # The version 2 pre-image is the document itself, so it moves: that is the defect version 3 removes.
    assert evidence_digest(perturbed, like="v2:") != bundle.fingerprint(like="v2:")


def test_the_bundle_s_shape_version_and_build_are_no_part_of_it() -> None:
    host, campaign = _corpus_host()
    bundle = _aspect(host, campaign)

    moved = bundle.model_copy(update={"schema_version": bundle.schema_version + 1, "package_version": "9.9.9"})

    assert moved.fingerprint() == bundle.fingerprint()


# =============================================================================
# Changing one piece of evidence moves it
# =============================================================================


def test_one_result_s_recorded_value_changed_moves_the_fingerprint_and_only_the_results() -> None:
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)
    result = _first_result(host, campaign)
    (measure, value), *_ = result.host_measures.items()
    assert isinstance(value, float)

    host.storage.save_eval_result(
        result.model_copy(update={"host_measures": {**result.host_measures, measure: value / 2}})
    )
    after = _aspect(host, campaign)

    assert after.fingerprint() != before.fingerprint()
    assert _moved(before, after) == {"results"}


def test_one_result_s_score_changed_moves_the_fingerprint() -> None:
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)
    result = _first_result(host, campaign)
    score, *rest = result.rubric_scores

    rescored = score.model_copy(update={"score": score.score - 1})
    host.storage.save_eval_result(result.model_copy(update={"rubric_scores": [rescored, *rest]}))
    after = _aspect(host, campaign)

    assert after.fingerprint() != before.fingerprint()
    assert _moved(before, after) == {"results"}


def test_a_result_deleted_moves_the_fingerprint() -> None:
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)
    result = _first_result(host, campaign)

    assert host.storage.delete_eval_result(result.id, campaign.scope_id)
    after = _aspect(host, campaign)

    assert after.fingerprint() != before.fingerprint()
    assert "results" in _moved(before, after)
    assert _manifest(after).n_results == _manifest(before).n_results - 1


def test_a_member_run_archived_moves_the_fingerprint() -> None:
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)
    run = host.storage.load_eval_run(sorted(campaign.run_ids)[0], campaign.scope_id)
    assert run is not None

    host.storage.save_eval_run(run.model_copy(update={"archived": True}))
    after = _aspect(host, campaign)

    assert after.fingerprint() != before.fingerprint()
    assert "runs" in _moved(before, after)


def test_a_run_s_status_changed_moves_the_fingerprint_and_only_the_runs() -> None:
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)
    run = host.storage.load_eval_run(sorted(campaign.run_ids)[0], campaign.scope_id)
    assert run is not None and run.status == "completed"

    host.storage.save_eval_run(run.model_copy(update={"status": "cancelled"}))
    after = _aspect(host, campaign)

    assert after.fingerprint() != before.fingerprint()
    assert _moved(before, after) == {"runs"}


def test_the_declaration_edited_moves_the_fingerprint_and_only_the_declaration() -> None:
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)
    (aspect,) = campaign.aspects
    assert aspect.design is not None

    asked_later = aspect.model_copy(update={"asked_at": "2031-01-01T00:00:00+00:00"})
    host.storage.save_campaign(campaign.model_copy(update={"aspects": [asked_later]}))
    after = _aspect(host, campaign)

    assert after.fingerprint() != before.fingerprint()
    assert _moved(before, after) == {"declaration"}


def test_a_case_filed_under_another_stratum_moves_the_fingerprint_and_only_the_cases() -> None:
    host, campaign = _corpus_host()
    result = _first_result(host, campaign)
    case = EvalTestCase(id=result.test_case_id, scope_id=campaign.scope_id, template_id="tpl-invoices")
    host.storage.save_test_case(case.model_copy(update={"stratum": "scanned"}))
    before = _aspect(host, campaign)
    assert _manifest(before).n_cases == 1

    host.storage.save_test_case(case.model_copy(update={"stratum": "typed"}))
    after = _aspect(host, campaign)
    host.storage.save_test_case(case.model_copy(update={"stratum": "typed", "name": "The typed one"}))
    named = _aspect(host, campaign)

    assert after.fingerprint() != before.fingerprint()
    assert _moved(before, after) == {"cases"}
    assert named.fingerprint() == after.fingerprint(), "what a reader calls a case is presentation"


def test_a_person_s_rating_added_or_changed_moves_the_fingerprint_and_only_the_ratings() -> None:
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)
    result = _first_result(host, campaign)
    score = result.rubric_scores[0]

    def rate(given: int, reason: str = "read the scan beside the extraction") -> None:
        rate_result(
            host.storage,
            result_id=result.id,
            scope_id=campaign.scope_id,
            rubric_dim=score.dim,
            rater="ana",
            rater_kind="person",
            score=given,
            reason=reason,
            rated_at="2026-03-15T00:00:00+00:00",
        )

    rate(score.score)
    rated = _aspect(host, campaign)
    rate(score.score - 1)
    rerated = _aspect(host, campaign)
    rate(score.score - 1, "other words")
    reworded = _aspect(host, campaign)

    assert _moved(before, rated) == {"ratings"} and _manifest(rated).n_ratings == 1
    assert _moved(rated, rerated) == {"ratings"}
    assert len({before.fingerprint(), rated.fingerprint(), rerated.fingerprint()}) == 3
    assert reworded.fingerprint() == rerated.fingerprint(), "a rater's prose is no part of what they gave"


def test_renaming_the_campaign_moves_no_fingerprint() -> None:
    """A name is presentation, never evidence (owner ruling, 2026-11-11)."""
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)

    host.storage.save_campaign(campaign.model_copy(update={"name": "Another name entirely", "description": "x"}))
    after = _aspect(host, campaign)

    assert after.fingerprint() == before.fingerprint()


def test_where_a_cost_came_from_moves_no_fingerprint_and_the_cost_does() -> None:
    """Pricing stays out (owner rulings, 2026-10-28 and 2026-11-14, #1011): a usage row's price source and serving
    provider are no part of a result's evidence; the cost it recorded is."""
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)
    result = _first_result(host, campaign)
    row, *rest = result.usage

    priced = row.model_copy(update={"price_source": "another-price-list", "served_provider": "another-provider"})
    host.storage.save_eval_result(result.model_copy(update={"usage": [priced, *rest]}))
    repriced = _aspect(host, campaign)
    assert repriced.fingerprint() == before.fingerprint()

    assert row.cost_usd is not None
    dearer = priced.model_copy(update={"cost_usd": row.cost_usd * 2})
    host.storage.save_eval_result(result.model_copy(update={"usage": [dearer, *rest]}))
    assert _aspect(host, campaign).fingerprint() != before.fingerprint()


def test_every_attribute_of_a_reading_is_decided_as_evidence_or_not() -> None:
    """A new attribute of a result's reading is taken into the fingerprint, or left out with a reason, on purpose."""
    attributes = {field.name for field in dataclasses.fields(ResultReading)}

    assert set(RESULT_EVIDENCE) | set(RESULT_NOT_EVIDENCE) == attributes
    assert not set(RESULT_EVIDENCE) & set(RESULT_NOT_EVIDENCE)
    assert len(RESULT_EVIDENCE) == len(set(RESULT_EVIDENCE))


@pytest.mark.parametrize("name", ["enriched", "toyhost", "courierhost"])
def test_a_result_and_its_reading_are_the_same_evidence(name: str) -> None:
    """A store hands the assembly each result as its reading; a view that is the stored result itself digests alike.
    The enriched campaign fills every record list a result holds (re-judges, repeats, second judgings, deliveries,
    preconditions, usage, latency), so every slim record is compared."""
    read = reading_fields_campaign(name)
    results = [
        result
        for run in read.host.storage.query_eval_runs(read.scope_id)
        for result in read.host.storage.query_eval_results_by_run(run.id, read.scope_id)
    ]
    assert results

    for result in results:
        assert input_json(result_evidence(result)) == input_json(result_evidence(ResultReading.of(result))), result.id
    if name == "enriched":
        filled = {attribute for result in results for attribute in RESULT_EVIDENCE if getattr(result, attribute)}
        assert {
            "judge_rescores",
            "judge_repeats",
            "judge_seconds",
            "async_deliveries",
            "precondition_outcomes",
        } <= filled


# =============================================================================
# Key order and a store's round trip move nothing
# =============================================================================


def _as_jsonb(node: Any) -> Any:
    """``node`` as a ``jsonb`` column hands it back: each object's keys by length and then bytes, whatever order
    they were written in, and a whole float as the integer it equals (a store that does not keep a number's
    scale)."""
    if isinstance(node, dict):
        return {key: _as_jsonb(node[key]) for key in sorted(node, key=lambda key: (len(key), key))}
    if isinstance(node, list):
        return [_as_jsonb(value) for value in node]
    if isinstance(node, float) and node.is_integer():
        return int(node)
    return node


class JsonbLikeStore(InMemoryDocumentStore):
    """A document store that keeps each document as ``jsonb`` would: through JSON, its keys reordered."""

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        super().upsert(_as_jsonb(json.loads(json.dumps(document))), if_match=if_match, create_only=create_only)


def test_a_store_that_reorders_keys_and_round_trips_documents_moves_nothing() -> None:
    plain_store, reordering_store = InMemoryDocumentStore(), JsonbLikeStore()
    plain_host, campaign = _corpus_host(EvalStorage(plain_store))
    reordering_host, _ = _corpus_host(EvalStorage(reordering_store))
    result_id = _first_result(plain_host, campaign).id
    as_written = plain_store.get(result_id, campaign.scope_id)
    as_reordered = reordering_store.get(result_id, campaign.scope_id)
    assert as_written is not None and as_reordered is not None
    assert list(as_written) != list(as_reordered), "the reordering store must hand keys back in another order"
    assert json.dumps(as_written) != json.dumps(as_reordered)

    plain, reordered = _aspect(plain_host, campaign), _aspect(reordering_host, campaign)

    assert _manifest(plain) == _manifest(reordered)
    assert plain.fingerprint() == reordered.fingerprint()


def test_a_stored_bundle_s_fingerprint_survives_a_reordering_round_trip() -> None:
    bundle = _aspect(*_corpus_host())

    read_back = _as_jsonb(json.loads(json.dumps(bundle.to_dict())))

    assert list(read_back) != list(bundle.to_dict())
    assert evidence_digest(read_back) == bundle.fingerprint()
    assert AnalysisContextBundle.from_dict(read_back).fingerprint() == bundle.fingerprint()


# =============================================================================
# How stored values are rendered
# =============================================================================


def test_key_order_is_not_read_at_any_depth_and_list_order_is() -> None:
    assert input_json({"b": {"y": 1, "x": 2}, "a": [3, 4]}) == input_json({"a": [3, 4], "b": {"x": 2, "y": 1}})
    assert input_json({"a": [3, 4]}) != input_json({"a": [4, 3]})
    assert input_json({"b": {"y": 1, "x": 2}, "a": [3, 4]}) == '{"a":[3,4],"b":{"x":2,"y":1}}'


def test_null_and_absent_are_one_and_a_null_in_a_list_stays() -> None:
    assert input_json({"a": 1, "b": None, "c": {"d": None}}) == input_json({"a": 1, "c": {}})
    assert input_json([None, 1]) == "[null,1]"


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (1, "1"),
        (1.0, "1"),
        (-0.0, "0"),
        (0.0, "0"),
        (2400.0, "2400"),
        (-3.0, "-3"),
        (0.1, "0.1"),
        (0.71, "0.71"),
        (0.1 + 0.2, "0.30000000000000004"),
        (1 / 3, "0.3333333333333333"),
        (21.28666989160534, "21.28666989160534"),
        (5e-324, "5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
        (1e22, "1e+22"),
        (2.0**53, "9007199254740992.0"),
        (2**53 + 1, "9007199254740993"),
        (True, "true"),
        (False, "false"),
    ],
)
def test_a_number_is_rendered_one_way(value: Any, rendered: str) -> None:
    """A whole float below 2**53 as its integer; any other float as the shortest string that parses back to it,
    which Python computes itself; a bool never as a number. Pinned as text, so a platform that rendered one another
    way would fail here."""
    assert input_json(value) == rendered
    if isinstance(value, float) and not value.is_integer():
        assert float(input_json(value)) == value


def test_a_whole_number_hashes_alike_as_an_integer_and_as_a_float() -> None:
    assert input_digest({"n": 3, "cost": 2.0}) == input_digest({"n": 3.0, "cost": 2})
    assert input_digest({"passed": True}) != input_digest({"passed": 1})


# =============================================================================
# What was stored under version 2 still reads
# =============================================================================


def _pinned_v2() -> dict[str, dict[str, list[str]]]:
    pinned: dict[str, dict[str, list[str]]] = json.loads(
        (FIXTURES / "hot_path_oracle" / "fingerprints_v2.json").read_text()
    )
    return pinned


@pytest.mark.parametrize("name", WITH_BUNDLES)
def test_a_fingerprint_stored_under_version_2_is_reproduced_through_like(name: str) -> None:
    """Each hot-path campaign's ``v2:`` fingerprints, as the build before version 3 recorded them, are what this
    build answers when asked in their version: across the manifest joining the bundle and the exact small-sample
    factor replacing the earlier one. ``scaled_two_aspects``' ``tpl-support`` holds two, the pool's and the aarch64
    dev container's on glibc 2.39: a version 2 fingerprint reads the platform's ``lgamma`` there, which is why it was
    replaced, and what a platform stored is what that platform reproduces."""
    read = subject(name)
    pinned = _pinned_v2()[name]

    aspects = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id).aspects

    assert sorted(aspect.aspect for aspect in aspects) == sorted(pinned)
    for aspect in aspects:
        stored = pinned[aspect.aspect]
        assert aspect.fingerprint(like=stored[0]) in stored, (name, aspect.aspect)
        assert aspect.fingerprint(like=stored[0]).startswith("v2:")
        assert aspect.fingerprint().startswith("v3:")


def _without_key(node: Any, key: str) -> Any:
    if isinstance(node, dict):
        return {name: _without_key(value, key) for name, value in node.items() if name != key}
    if isinstance(node, list):
        return [_without_key(value, key) for value in node]
    return node


def test_a_field_the_bundle_gained_since_version_2_is_no_part_of_a_version_2_fingerprint() -> None:
    """The version 2 pre-image is frozen: ``RunSummary.ended_because`` (#1173) joined the bundle after the last
    ``v2:`` fingerprint a host stored, so a ``v2:`` record over a run that did not complete still matches. The same
    holds for the manifest itself and for any top-level field the bundle gains."""
    host, campaign = _corpus_host()
    run = host.storage.load_eval_run(sorted(campaign.run_ids)[0], campaign.scope_id)
    assert run is not None
    host.storage.save_eval_run(run.model_copy(update={"status": "cancelled", "cancellation_reason": "stopped by hand"}))
    document = _aspect(host, campaign).to_dict()
    assert "ended_because" in json.dumps(document), "the cancelled run's summary must say why it ended"

    as_stored_then = _without_key(document, "ended_because")

    assert evidence_digest(document, like="v2:") == evidence_digest(as_stored_then, like="v2:")
    grown = {**document, "a_field_added_later": {"anything": 1.5}}
    assert evidence_digest(grown, like="v2:") == evidence_digest(document, like="v2:")
    assert evidence_digest(grown) == evidence_digest(document)


def _stored_under_v2(bundle: AnalysisContextBundle) -> dict[str, Any]:
    """The bundle's document as a build before version 3 stored it: no manifest."""
    document = bundle.to_dict()
    del document["evidence"]
    return document


def test_a_bundle_stored_before_the_manifest_keeps_its_version_2_fingerprint_and_is_never_backfilled() -> None:
    bundle = _aspect(*_corpus_host())
    stored = _stored_under_v2(bundle)
    recorded = evidence_digest(stored)

    read_back = AnalysisContextBundle.from_dict(stored)

    assert recorded.startswith("v2:")
    assert read_back.evidence is None
    assert read_back.fingerprint() == recorded
    assert read_back.fingerprint(like="v3:") == recorded, "asked for version 3, a bundle with no manifest answers v2"
    assert "evidence" not in read_back.to_dict(), "a read writes no manifest into what was stored"


def test_a_frame_stored_under_version_2_still_holds_its_bundles() -> None:
    """A stored analysis's frame names each aspect's bundle by the fingerprint it was stored with; put back together
    with the stored bundles, and with the same evidence assembled today, it is not refused."""
    host, campaign = _corpus_host()
    assembled = assemble_campaign_bundle(host, campaign.id, campaign.scope_id)
    stored_bundles = [AnalysisContextBundle.from_dict(_stored_under_v2(aspect)) for aspect in assembled.aspects]
    frame = {
        **assembled.frame(),
        "aspects": [{"aspect": b.aspect, "fingerprint": b.fingerprint()} for b in stored_bundles],
    }
    assert all(entry["fingerprint"].startswith("v2:") for entry in frame["aspects"])

    assert CampaignBundle.from_frame(frame, stored_bundles).labels == assembled.labels
    assert CampaignBundle.from_frame(frame, assembled.aspects).labels == assembled.labels


def test_a_version_2_record_over_unmoved_evidence_reproduces_and_names_no_cause() -> None:
    """The staleness verdict: an analysis that recorded ``v2:`` is compared the version 2 way, so the version changing
    is never what reads as moved."""
    host, campaign = _corpus_host()
    bundle = _aspect(host, campaign)
    recorded = evidence_digest(_stored_under_v2(bundle))
    assert recorded.startswith("v2:") and bundle.fingerprint().startswith("v3:")

    inspected = BundleInspection.over(
        _aspect(host, campaign),
        analysis_id="a",
        recorded_fingerprint=recorded,
        recorded_schema_version=bundle.schema_version,
        recorded_host_declarations_digest=bundle.host_declarations_digest,
    )

    assert inspected.reproduces_generation is True
    assert inspected.mismatch_cause is None
    assert inspected.fingerprint == recorded, "the fingerprint beside a record is in the record's version"


def test_a_version_2_record_over_moved_evidence_reads_as_evidence() -> None:
    host, campaign = _corpus_host()
    bundle = _aspect(host, campaign)
    recorded = evidence_digest(_stored_under_v2(bundle))
    assert host.storage.delete_eval_result(_first_result(host, campaign).id, campaign.scope_id)

    inspected = BundleInspection.over(
        _aspect(host, campaign),
        analysis_id="a",
        recorded_fingerprint=recorded,
        recorded_schema_version=bundle.schema_version,
        recorded_host_declarations_digest=bundle.host_declarations_digest,
    )

    assert (inspected.reproduces_generation, inspected.mismatch_cause) == (False, "evidence")


def test_a_version_3_record_reproduces_whatever_the_bundle_s_shape_version_is() -> None:
    """Against a ``v3:`` record the package's shape is never a cause: a shape bump alone reproduces."""
    host, campaign = _corpus_host()
    bundle = _aspect(host, campaign)

    inspected = BundleInspection.over(
        bundle,
        analysis_id="a",
        recorded_fingerprint=bundle.fingerprint(),
        recorded_schema_version=bundle.schema_version - 1,
        recorded_host_declarations_digest=bundle.host_declarations_digest,
    )

    assert (inspected.reproduces_generation, inspected.mismatch_cause) == (True, None)


def test_a_version_3_record_over_moved_evidence_reads_as_evidence_never_as_package_shape() -> None:
    host, campaign = _corpus_host()
    bundle = _aspect(host, campaign)
    assert host.storage.delete_eval_result(_first_result(host, campaign).id, campaign.scope_id)

    inspected = BundleInspection.over(
        _aspect(host, campaign),
        analysis_id="a",
        recorded_fingerprint=bundle.fingerprint(),
        recorded_schema_version=bundle.schema_version - 1,
        recorded_host_declarations_digest=bundle.host_declarations_digest,
    )

    assert (inspected.reproduces_generation, inspected.mismatch_cause) == (False, "evidence")


def test_a_record_of_a_version_this_build_does_not_compute_is_never_reported_as_moved_evidence() -> None:
    host, campaign = _corpus_host()
    bundle = _aspect(host, campaign)

    inspected = BundleInspection.over(
        bundle,
        analysis_id="a",
        recorded_fingerprint="v9:" + "0" * 64,
        recorded_schema_version=bundle.schema_version,
        recorded_host_declarations_digest=bundle.host_declarations_digest,
    )

    assert (inspected.reproduces_generation, inspected.mismatch_cause) == (False, "package_shape")


def test_the_stored_version_2_analysis_among_the_fixtures_still_verifies_each_frozen_bundle() -> None:
    """``inputs/toyhost_doc.json`` holds an analysis a build before version 3 stored: each aspect's frozen bundle
    still digests to the ``v2:`` fingerprint stored beside it, and its frame still holds its bundles."""
    documents = json.loads((FIXTURES / "hot_path_oracle" / "inputs" / "toyhost_doc.json").read_text())
    analyses = [document for document in documents if document.get("doc_type") == "eval_analysis"]
    assert analyses, "the capture holds a stored analysis"

    for analysis in analyses:
        bundles = []
        for aspect in analysis["aspects"]:
            assert aspect["bundle_fingerprint"].startswith("v2:")
            assert evidence_digest(aspect["bundle"]) == aspect["bundle_fingerprint"]
            bundles.append(AnalysisContextBundle.from_dict(aspect["bundle"]))
        CampaignBundle.from_frame(analysis["campaign_bundle"], bundles)
