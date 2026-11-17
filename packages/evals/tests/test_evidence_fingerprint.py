"""A bundle's fingerprint is over the evidence it was assembled from, never over a statistic computed from it (#1062).

Version 2 hashed the bundle's own document, so a better formula, another platform's maths library or a library update
read as "the evidence moved" and what was stored against the old fingerprint was regenerated. Version 3 hashes the
manifest of what the assembly read (:mod:`threetears.evals.analysis.bundle.evidence`). The rule it is held to:
**anything that can change a verdict or a printed number is evidence**, and leaving evidence out is the dangerous
direction. What is held here:

- **computing a statistic differently moves no fingerprint**, the aspect's or the campaign's;
- **changing one piece of evidence moves it**, and the manifest names which kind moved and no other: a result, a
  run, the declaration (its question, margin, bars), the host's criteria (its pass threshold, a registered bar, a
  measure's declaration), a case, a rating, a prior insight;
- **a name, a label, a description or a rationale moves nothing**;
- **a field a record gains later moves nothing**, whatever its default, until it is named as evidence;
- **key order and a store's round trip move nothing**: the same corpus read back from a store that reorders every
  document's keys as ``jsonb`` does and hands whole floats back as integers;
- **numbers are rendered one way on every platform** (:func:`~threetears.evals.schema.hashing.input_json`);
- **what was stored under an earlier version still reads**: it is compared in its own version, is never rewritten,
  and a fingerprint's version alone is never reported as moved evidence;
- **the four hot-path campaigns' ``v2:`` fingerprints are reproduced** by this build through ``like=``, pinned as the
  build before it recorded them (``fixtures/hot_path_oracle/fingerprints_v2.json``); their ``v3:`` fingerprints are
  pinned, one value for every platform, by ``test_hot_path_oracle.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

import pytest

from threetears.evals.analysis.service import assemble_campaign_bundle
from threetears.evals.analysis import stats
from threetears.evals.analysis.bundle.evidence import (
    EvidenceManifest,
    ResultsDigest,
    evidence_manifest,
    manifest_fingerprint,
)
from threetears.evals.analysis.bundle.schema import (
    FINGERPRINT_PREFIX,
    AnalysisContextBundle,
    BundleInspection,
    evidence_digest,
)
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.aspects import AspectDeclaration, aspect_view
from threetears.evals.kernel.campaign import EvalCampaign, EvalInsight
from threetears.evals.kernel.declaration import BarOverride
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.host.bars import BarRegistry, PassThreshold
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.result_reading import ResultReading, ScoreReading
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.schema.hashing import UnhashableContentError, input_digest, input_json, round_figures
from threetears.evals.run.ratings import rate_result
from threetears.evals.schema.models import (
    EvalResult,
    EvalRun,
    EvalTestCase,
    MeasureAttestation,
    MeasureRemeasure,
    RunCompleteness,
)
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_campaign
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.test_hot_path_oracle import WITH_BUNDLES, subject
from packages.evals.tests.test_result_reading_fields import campaign as reading_fields_campaign

FIXTURES = Path(__file__).parent / "fixtures"

#: Every component digest of a manifest: what "which kind of evidence moved" is read off.
COMPONENTS = (
    "declaration",
    "criteria",
    "host_declarations",
    "runs",
    "results",
    "cases",
    "ratings",
    "judge_profiles",
    "insights",
)


def _corpus_host(
    storage: EvalStorage | None = None, *, profile: HostProfile | None = None
) -> tuple[EvalHost, EvalCampaign]:
    """The toy host holding its corpus campaign, in ``storage`` when one is given, under ``profile`` when one is."""
    host = toyhost_host(storage=storage, profile=profile)
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


def result_evidence(result: Any) -> str:
    """One result's evidence, as the assembly digests it: the digest of the results it read, were this the only one."""
    read = ResultsDigest()
    read.add(result)
    return read.digest()


def run_evidence(run: EvalRun) -> str:
    """One member run's evidence, as the assembly digests it: the manifest's ``runs`` over this run alone."""
    campaign, _ = toyhost_campaign()
    return evidence_manifest(
        aspect_view(campaign),
        profile=toyhost_profile(),
        host_declarations="",
        catalog={},
        runs=[run],
        archived_run_ids=[],
        unresolved_run_ids=[],
        results=ResultsDigest(),
        cases=[],
        ratings=[],
        tiers=[],
        insights=[],
        retracted_insights={},
    ).runs


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
        assert result_evidence(result) == result_evidence(ResultReading.of(result)), result.id
    if name == "enriched":
        lists = ("judge_rescores", "judge_repeats", "judge_seconds", "async_deliveries", "precondition_outcomes")
        assert all(any(getattr(result, attribute) for result in results) for attribute in lists)
        for attribute in lists:
            filled = next(result for result in results if getattr(result, attribute))
            emptied = filled.model_copy(update={attribute: None if attribute == "async_deliveries" else []})
            assert result_evidence(emptied) != result_evidence(filled), f"{attribute} is evidence"


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


def test_a_null_is_a_value_in_a_map_as_stored() -> None:
    """In a map a host writes, a key's presence can carry meaning: a recorded null and a key never written differ."""
    assert input_json({"a": 1, "b": None}) != input_json({"a": 1})
    assert input_json({"a": 1, "b": None}) == '{"a":1,"b":null}'
    assert input_json([None, 1]) == "[null,1]"


def test_an_unset_field_of_a_record_and_one_never_written_are_one_and_a_null_in_a_host_s_map_is_kept() -> None:
    """The engine reads a record's stored null and a field never written as one value (both hydrate to ``None``), so
    evidence leaves an unset field out. A free-form map a host or a kind writes is taken as stored, its nulls kept."""
    host, campaign = _corpus_host()
    run = host.storage.load_eval_run(sorted(campaign.run_ids)[0], campaign.scope_id)
    assert run is not None and run.cassette_corpus_id is None and run.case_set is None
    stored = run.to_dict()
    written = {**stored, "cassette_corpus_id": None, "case_set": None}
    never = {key: value for key, value in stored.items() if key not in ("cassette_corpus_id", "case_set")}

    assert run_evidence(EvalRun.from_dict(written)) == run_evidence(EvalRun.from_dict(never)) == run_evidence(run)
    with_null = run_evidence(run.model_copy(update={"overlays": {"greeting": None}}))
    without = run_evidence(run.model_copy(update={"overlays": {}}))
    assert with_null != without
    no_overlays = {key: value for key, value in stored.items() if key != "overlays"}
    assert without == run_evidence(EvalRun.from_dict(no_overlays)), "an empty map is unset"


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
        (1.5e-05, "1.5e-05"),
        (2.0**53, "9007199254740992"),
        (2.0**53 + 2.0, "9007199254740994"),
        (2**53 + 1, "9007199254740993"),
        (1e22, "10000000000000000000000"),
        (-1e22, "-10000000000000000000000"),
        (True, "true"),
        (False, "false"),
        (math.nan, "NaN"),
        (math.inf, "Infinity"),
        (-math.inf, "-Infinity"),
        ("naïve café", '"na\\u00efve caf\\u00e9"'),
    ],
)
def test_a_value_is_rendered_one_way(value: Any, rendered: str) -> None:
    """A whole float as the integer it equals, at any magnitude; any other finite float as the shortest string that
    parses back to it, which Python computes itself; NaN and the infinities as their fixed tokens; a bool never as a
    number; a string with every non-ASCII character escaped. Pinned as text, so a platform that rendered one
    another way would fail here."""
    assert sys.float_repr_style == "short", "Python's own shortest round-trip repr, not the C library's"
    assert input_json(value) == rendered
    if isinstance(value, float) and math.isfinite(value) and not value.is_integer():
        assert float(input_json(value)) == value


@pytest.mark.parametrize("whole", [0.0, 1.0, 2400.0, 2.0**53, 2.0**60, 1e22, 1.7976931348623157e308, -1e300])
def test_a_whole_float_hashes_as_the_integer_a_jsonb_store_hands_back(whole: float) -> None:
    """A ``jsonb`` column holds a number as an exact decimal and hands a whole one back as an integer, whatever it was
    written as and however large. Rendered here as that integer, the value written and the value read back are one."""
    handed_back = int(whole)
    assert float(handed_back) == whole

    assert input_json(whole) == input_json(handed_back) == str(handed_back)
    assert input_digest({"n": whole}) == input_digest({"n": handed_back})


def test_a_whole_number_hashes_alike_as_an_integer_and_as_a_float_and_a_bool_is_no_number() -> None:
    assert input_digest({"n": 3, "cost": 2.0}) == input_digest({"n": 3.0, "cost": 2})
    assert input_digest({"passed": True}) != input_digest({"passed": 1})


@pytest.mark.parametrize(
    "unencodable",
    [datetime(2026, 11, 14, tzinfo=UTC), Decimal("0.1"), {1, 2}, object(), b"bytes"],
    ids=["datetime", "decimal", "set", "object", "bytes"],
)
def test_a_value_json_cannot_encode_is_refused_never_rendered_by_its_repr(unencodable: Any) -> None:
    with pytest.raises(UnhashableContentError):
        input_json({"held": unencodable})


def test_two_unicode_spellings_of_one_text_are_two_values() -> None:
    """Strings are not Unicode-normalised: composed and decomposed forms differ, as they differ in the store."""
    assert input_json("é") != input_json("é")


# =============================================================================
# What was stored under version 2 still reads
# =============================================================================


def _pinned_v2() -> dict[str, dict[str, list[str]]]:
    pinned: dict[str, dict[str, list[str]]] = json.loads(
        (FIXTURES / "hot_path_oracle" / "fingerprints_v2.json").read_text()
    )
    return pinned


#: The campaigns whose evidence a version 2 fingerprint was stored over and is still what it was. ``courierhost`` is
#: left out, and its entry in ``fingerprints_v2.json`` kept as the build before version 3 stored it, never re-taken:
#: its fixture's upgrade step now attests the definitions its runs ran under (#1187), so its evidence moved, and no
#: earlier build ever recorded a version 2 fingerprint over the evidence it holds now.
_V2_UNMOVED: Final = tuple(name for name in WITH_BUNDLES if name != "courierhost")


@pytest.mark.parametrize("name", _V2_UNMOVED)
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


def test_a_field_the_bundle_gains_after_version_2_is_no_part_of_a_version_2_fingerprint() -> None:
    """The version 2 pre-image is frozen to the bundle's fields as the last build that stored ``v2:`` held them: the
    manifest, and any top-level field the bundle gains, leave a ``v2:`` fingerprint where it was."""
    document = _aspect(*_corpus_host()).to_dict()

    grown = {**document, "a_field_added_later": {"anything": 1.5}}

    assert evidence_digest(grown, like="v2:") == evidence_digest(document, like="v2:")
    assert evidence_digest(_stored_under_v2_document(document)) == evidence_digest(document, like="v2:")
    # A field added inside a frozen value (#1187: a frontier whose contestants' definitions differ) is left out too.
    apart = {**document, "frontier": {**document["frontier"], "definitions_apart": True}}
    assert evidence_digest(apart, like="v2:") == evidence_digest(document, like="v2:")
    assert evidence_digest(grown) == evidence_digest(document)


def _stored_under_v2_document(document: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "evidence"}


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


# =============================================================================
# Criteria are evidence: what a verdict is decided against
# =============================================================================


def _under(profile: HostProfile) -> AnalysisContextBundle:
    return _aspect(*_corpus_host(profile=profile))


def _with_bars(profile: HostProfile, **changed: Any) -> HostProfile:
    """The profile, its first registered bar changed."""
    first, *rest = profile.bars.bars
    return dataclasses.replace(
        profile,
        bars=BarRegistry([dataclasses.replace(first, **changed), *rest], pass_thresholds=profile.bars.pass_thresholds),
    )


def test_a_registered_bar_s_threshold_changed_moves_the_fingerprint_and_only_the_criteria() -> None:
    """A host's registered bar decides a verdict (`bar_adjudications`) with no campaign naming it. Version 2 caught a
    change to one through the statistics; version 3 holds it as evidence."""
    base = toyhost_profile()
    first = base.bars.bars[0]
    before = _under(base)

    after = _under(_with_bars(base, threshold=first.threshold - 0.25))

    assert [a.model_dump() for a in after.bar_adjudications] != [b.model_dump() for b in before.bar_adjudications]
    assert after.fingerprint() != before.fingerprint()
    assert _moved(before, after) == {"criteria"}


def test_a_registered_bar_s_rationale_reworded_moves_nothing() -> None:
    base = toyhost_profile()

    assert _under(_with_bars(base, rationale="said another way")).fingerprint() == _under(base).fingerprint()


def test_the_pass_threshold_changed_moves_the_fingerprint_and_only_the_criteria() -> None:
    """The pass threshold decides whether every judged trial passed."""
    base = toyhost_profile()
    behavior = _corpus_host()[1].behavior
    before = _under(base)
    raised = dataclasses.replace(
        base,
        bars=BarRegistry(
            base.bars.bars,
            pass_thresholds=[
                PassThreshold(behavior=behavior, threshold=base.bars.pass_threshold(behavior) + 1, rationale="stricter")
            ],
        ),
    )

    after = _under(raised)
    reworded = _under(
        dataclasses.replace(
            base,
            bars=BarRegistry(
                base.bars.bars,
                pass_thresholds=[
                    PassThreshold(
                        behavior=behavior, threshold=base.bars.pass_threshold(behavior) + 1, rationale="other words"
                    )
                ],
            ),
        )
    )

    assert after.fingerprint() != before.fingerprint()
    assert _moved(before, after) == {"criteria"}
    assert reworded.fingerprint() == after.fingerprint(), "a threshold's rationale is prose"


def _with_measure(profile: HostProfile, name: str, **changed: Any) -> HostProfile:
    """The profile, one host measure's declaration changed."""
    descriptors = [
        descriptor.model_copy(update=changed) if descriptor.name == name else descriptor
        for descriptor in (profile.measures.get(each) for each in profile.measures.names)
        if descriptor is not None
    ]
    return dataclasses.replace(profile, measures=MeasureRegistry(descriptors, families=profile.measures.families))


def test_a_measure_s_declaration_changed_moves_the_fingerprint_and_its_words_do_not() -> None:
    """What a measure declares decides how it is read and compared: its direction, its range, its materiality
    threshold. Its description and what a reader calls it are words."""
    base = toyhost_profile()
    before = _under(base)
    measure = base.measures.get("field_accuracy")
    assert measure is not None and "field_accuracy" in before.measure_catalog

    stricter = _under(_with_measure(base, "field_accuracy", materiality_threshold=0.4321))
    reworded = _under(
        _with_measure(base, "field_accuracy", description="Said another way.", reader_name="Fields right")
    )

    assert before.measure_catalog["field_accuracy"].materiality_threshold != 0.4321
    assert stricter.measure_catalog["field_accuracy"].materiality_threshold == 0.4321
    assert _moved(before, stricter) == {"criteria"}
    assert reworded.measure_catalog["field_accuracy"].reader_name == "Fields right"
    assert reworded.fingerprint() == before.fingerprint()


def _redeclared(campaign: EvalCampaign, **aspect: Any) -> EvalCampaign:
    (declared,) = campaign.aspects
    return campaign.model_copy(update={"aspects": [declared.model_copy(update=aspect)]})


def test_the_campaign_s_own_criteria_changed_move_the_fingerprint_and_only_the_declaration() -> None:
    """The question, a bar the campaign declares and the repetitions it intends are what its verdicts are decided
    against."""
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)
    (aspect,) = campaign.aspects
    assert aspect.design is not None and aspect.design.bars

    host.storage.save_campaign(campaign.model_copy(update={"question": "is the narrower chunk good enough?"}))
    asked = _aspect(host, campaign)
    first, *rest = aspect.design.bars
    tighter = BarOverride(measure_id=first.measure_id, threshold=first.threshold / 2, direction=first.direction)
    design = aspect.design.model_copy(update={"bars": [tighter, *rest]})
    host.storage.save_campaign(_redeclared(campaign, design=design))
    barred = _aspect(host, campaign)
    repeated = aspect.design.model_copy(update={"intended_repetitions": (aspect.design.intended_repetitions or 1) + 4})
    host.storage.save_campaign(_redeclared(campaign, design=repeated))
    planned = _aspect(host, campaign)

    for changed in (asked, barred, planned):
        assert changed.fingerprint() != before.fingerprint()
        assert _moved(before, changed) == {"declaration"}
    assert len({before.fingerprint(), asked.fingerprint(), barred.fingerprint(), planned.fingerprint()}) == 4


def test_a_prior_insight_added_or_restated_moves_the_fingerprint_and_only_the_insights() -> None:
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)

    def insight(statement: str) -> EvalInsight:
        return EvalInsight(
            id="insight-1",
            scope_id=campaign.scope_id,
            subject_id=campaign.subject_id,
            subject_kind=campaign.subject_kind,
            statement=statement,
            scope="the wide chunk",
            confidence="medium",
            observed_at="2026-01-01T00:00:00+00:00",
            invalidation_trigger="a new retriever",
        )

    host.storage.save_insight(insight("The wide chunk costs more than it returns."))
    held = _aspect(host, campaign)
    host.storage.save_insight(insight("The wide chunk returns more than it costs."))
    restated = _aspect(host, campaign)

    assert _moved(before, held) == {"insights"} and _manifest(held).n_insights == 1
    assert _moved(held, restated) == {"insights"}


# =============================================================================
# A field a record gains later moves nothing
# =============================================================================


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ScoreReadingAfterALaterBuild(ScoreReading):
    """A slim score as a later build might carry it: one more field, whose default is not null."""

    added_later: bool = False
    counted_later: int = 0


class CompletenessAfterALaterBuild(RunCompleteness):
    """A run's completeness as a later build might store it: one more field, whose default is not null."""

    added_later: bool = False
    listed_later: list[str] = []


class RunAfterALaterBuild(EvalRun):
    """A run as a later build might store it: one more top-level field, whose default is not null, and a
    completeness that gained fields."""

    added_later: str = "a default"
    completeness: CompletenessAfterALaterBuild | None = None


class DeclarationAfterALaterBuild(AspectDeclaration):
    """An aspect's declaration as a later build might hold it: one more field, whose default is not null."""

    added_later: bool = False


def test_a_field_added_to_a_nested_record_with_a_default_moves_no_stored_fingerprint() -> None:
    """Records are built from named fields, at every depth, so a later field is in no fingerprint until it is named:
    a slim score's, a run's completeness's, a run's own. (Each record's fields are all decided, as evidence or not,
    when the evidence module is imported, so the later build cannot ship without deciding.)"""
    host, campaign = _corpus_host()
    result = _first_result(host, campaign)
    reading = ResultReading.of(result)
    later_scores = tuple(
        ScoreReadingAfterALaterBuild(**{f.name: getattr(score, f.name) for f in dataclasses.fields(ScoreReading)})
        for score in reading.rubric_scores
    )
    assert later_scores

    assert result_evidence(reading.model_copy(update={"rubric_scores": later_scores})) == result_evidence(reading)

    run = host.storage.load_eval_run(result.eval_run_id, campaign.scope_id)
    assert run is not None
    counted = run.model_copy(
        update={
            "completeness": RunCompleteness(
                expected_cells=4, produced_cells=4, persisted_cells=4, infra_excluded_cells=0, counted_from="run_loop"
            )
        }
    )
    assert counted.completeness is not None
    later = RunAfterALaterBuild.model_validate(
        {
            **counted.to_dict(),
            "completeness": CompletenessAfterALaterBuild(**counted.completeness.model_dump()).model_dump(),
        }
    )
    assert later.added_later == "a default" and "added_later" in later.to_dict()["completeness"]
    assert run_evidence(later) == run_evidence(counted)
    assert run_evidence(counted) != run_evidence(run), "the completeness itself is evidence"


def test_a_field_added_to_the_declaration_with_a_default_moves_no_stored_fingerprint() -> None:
    """The declaration is rendered without any field that holds its declared default, so a field a later build adds
    to it is in no fingerprint until a campaign sets it."""
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)
    assert before.declared is not None
    later = DeclarationAfterALaterBuild.model_validate(before.declared.to_dict())

    def digest(declared: AspectDeclaration) -> str:
        document = declared.model_dump(mode="json", exclude_defaults=True)
        return input_digest(document)

    assert "added_later" in later.to_dict()
    assert digest(later) == digest(before.declared)
    assert digest(later.model_copy(update={"added_later": True})) != digest(before.declared)


# =============================================================================
# The campaign's fingerprint means what an aspect's does
# =============================================================================


def _nudge(node: Any) -> Any:
    """``node`` with every float moved in its ninth figure, but the aggregate's weights, which must sum to one."""
    if isinstance(node, bool):
        return node
    if isinstance(node, float):
        return node * (1 - 1e-9)
    if isinstance(node, dict):
        return {key: value if key == "weights" else _nudge(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_nudge(value) for value in node]
    return node


def test_a_campaign_s_fingerprint_is_version_4_over_its_declaration_and_its_aspects_evidence() -> None:
    host, campaign = _corpus_host()
    bundle = assemble_campaign_bundle(host, campaign.id, campaign.scope_id)

    fingerprint = bundle.fingerprint()

    assert fingerprint.startswith("v4:") and len(fingerprint) == 3 + 64
    assert assemble_campaign_bundle(host, campaign.id, campaign.scope_id).fingerprint() == fingerprint
    assert CampaignBundle.from_dict(bundle.to_dict()).fingerprint() == fingerprint


def test_every_computed_reading_of_a_campaign_perturbed_moves_no_campaign_fingerprint() -> None:
    """The campaign's readings across aspects, with their intervals and reasons, are computed: no part of ``v4:``.
    They are the whole of what moved a ``v3:`` campaign fingerprint, which hashed them."""
    host, campaign = _corpus_host()
    bundle = assemble_campaign_bundle(host, campaign.id, campaign.scope_id)
    frame = bundle.frame()

    perturbed = CampaignBundle.from_frame({**frame, "readings": _nudge(frame["readings"])}, bundle.aspects)

    assert perturbed.frame() != frame, "the frame's readings carry floats to perturb"
    assert perturbed.fingerprint() == bundle.fingerprint()
    assert perturbed.fingerprint(like="v3:") != bundle.fingerprint(like="v3:")


def test_swapping_the_small_sample_factor_moves_no_campaign_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    read = subject("scaled_two_aspects")
    exact = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id)
    monkeypatch.setattr(stats, "hedges_j", lambda df: None if df < 2 else 1.0 - 3.0 / (4.0 * df - 1.0))
    approximate = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id)

    assert [a.to_dict() for a in exact.aspects] != [a.to_dict() for a in approximate.aspects]
    assert exact.fingerprint() == approximate.fingerprint()
    assert [a.fingerprint() for a in exact.aspects] == [a.fingerprint() for a in approximate.aspects]


def test_evidence_changed_in_an_aspect_or_the_declaration_moves_the_campaign_fingerprint_and_a_name_does_not() -> None:
    host, campaign = _corpus_host()
    before = assemble_campaign_bundle(host, campaign.id, campaign.scope_id).fingerprint()

    host.storage.save_campaign(campaign.model_copy(update={"name": "Another name entirely"}))
    renamed = assemble_campaign_bundle(host, campaign.id, campaign.scope_id).fingerprint()
    host.storage.save_campaign(campaign.model_copy(update={"question": "is the narrower chunk good enough?"}))
    asked = assemble_campaign_bundle(host, campaign.id, campaign.scope_id).fingerprint()
    host.storage.save_campaign(campaign)
    assert host.storage.delete_eval_result(_first_result(host, campaign).id, campaign.scope_id)
    fewer = assemble_campaign_bundle(host, campaign.id, campaign.scope_id).fingerprint()

    assert renamed == before
    assert len({before, asked, fewer}) == 3


def test_a_campaign_bundle_frozen_before_the_manifest_keeps_its_version_3_fingerprint() -> None:
    """A stored ``v3:`` campaign fingerprint was taken over the frame, each aspect named by its ``v2:`` fingerprint.
    A bundle frozen then answers it whatever is asked, and the same evidence assembled today answers it through
    ``like``: neither is rewritten, and the version alone moves nothing."""
    host, campaign = _corpus_host()
    assembled = assemble_campaign_bundle(host, campaign.id, campaign.scope_id)
    document = assembled.to_dict()
    frozen_then = CampaignBundle.from_dict(
        {
            **document,
            "aspects": [{k: v for k, v in aspect.items() if k != "evidence"} for aspect in document["aspects"]],
        }
    )

    stored = frozen_then.fingerprint()

    assert stored.startswith("v3:")
    assert frozen_then.fingerprint(like="v4:") == stored
    assert assembled.fingerprint(like=stored) == stored
    assert assembled.fingerprint().startswith("v4:")


# =============================================================================
# The small-sample factor put back, where the two factors differ
# =============================================================================


def _factor_as_version_2_took_it(df: int) -> float:
    """Hedges' factor in its textbook form, which every ``v2:`` fingerprint was taken with."""
    return math.exp(math.lgamma(df / 2.0) - math.lgamma((df - 1) / 2.0)) / math.sqrt(df / 2.0)


def test_a_version_2_record_at_high_degrees_of_freedom_is_matched_only_with_the_earlier_factor_put_back() -> None:
    """At a few thousand degrees of freedom the exact factor and the earlier one differ past the twelfth figure, so
    a ``v2:`` fingerprint stored then is matched by the same evidence assembled now only because the version 2
    pre-image puts the earlier factor back; without that it reads as moved. The hot-path fixtures sit near 128
    degrees of freedom, where twelve-figure rounding hides the difference, so this is held apart."""
    effect = 0.3141592653589793
    exact, df = next(
        (factor, df)
        for df in range(3000, 9000)
        if (factor := stats.hedges_j(df)) is not None
        and round_figures(effect * factor) != round_figures(effect * _factor_as_version_2_took_it(df))
    )
    read = subject("scaled")
    (bundle,) = assemble_campaign_bundle(read.host, read.campaign_id, read.scope_id).aspects
    document = bundle.to_dict()
    family = document["multiple_comparisons"]["families"][0]
    comparison = next(c for c in family["comparisons"] if c["hedges_g"] is not None and c["test"] == "paired")

    def at_high_df(g: float) -> dict[str, Any]:
        """The bundle's document, one paired comparison read over ``df + 1`` cases with this effect size."""
        moved = {
            **comparison,
            "hedges_g": g,
            "control": {**comparison["control"], "n_cases": df + 1},
            "contrast": {**comparison["contrast"], "n_cases": df + 1},
        }
        comparisons = [moved if c is comparison else c for c in family["comparisons"]]
        families = [{**family, "comparisons": comparisons}, *document["multiple_comparisons"]["families"][1:]]
        return {**document, "multiple_comparisons": {**document["multiple_comparisons"], "families": families}}

    assembled_now = at_high_df(effect * exact)
    stored_then = {k: v for k, v in at_high_df(effect * _factor_as_version_2_took_it(df)).items() if k != "evidence"}
    recorded = evidence_digest(stored_then)
    assert recorded.startswith("v2:")

    assert evidence_digest(assembled_now, like=recorded) == recorded
    # Without the put-back (a document carrying no manifest is read as stored, its g untouched) it does not match.
    without_put_back = {k: v for k, v in assembled_now.items() if k != "evidence"}
    assert evidence_digest(without_put_back) != recorded


# =============================================================================
# What a run's measures counted is evidence (#1160)
# =============================================================================


def _attested(run: EvalRun, digest: str, *, note: str | None = None) -> EvalRun:
    """``run`` as a host's attestation leaves it: its measures' definitions stated, and the entry that says so."""
    definitions = {"field_accuracy": digest}
    return run.model_copy(
        update={
            "measure_definitions": definitions,
            "measure_attestation": MeasureAttestation(
                attested_at="2026-11-14T00:00:00+00:00", attested_by="ops", definitions=definitions, note=note
            ),
        }
    )


def test_a_run_that_recorded_no_measure_definitions_hashes_as_it_did_before_they_existed() -> None:
    """A run launched before definitions were recorded carries none of the three fields, and its evidence names none:
    the document as it was stored then, the three keys absent, and the run as this build holds it (the definitions
    null, the re-measures an empty list) are one evidence."""
    host, campaign = _corpus_host()
    recorded = host.storage.load_eval_run(sorted(campaign.run_ids)[0], campaign.scope_id)
    assert recorded is not None and recorded.measure_definitions
    assert recorded.measure_attestation is None and not recorded.measure_remeasures
    unrecorded = recorded.model_copy(update={"measure_definitions": None})

    as_stored_then = {
        key: value
        for key, value in unrecorded.to_dict().items()
        if key not in ("measure_definitions", "measure_attestation", "measure_remeasures")
    }
    assert EvalRun.from_dict(as_stored_then).measure_remeasures == []

    assert run_evidence(EvalRun.from_dict(as_stored_then)) == run_evidence(unrecorded)
    assert run_evidence(unrecorded) != run_evidence(recorded)


def test_a_run_attested_or_re_measured_under_another_definition_is_other_evidence() -> None:
    host, campaign = _corpus_host()
    run = host.storage.load_eval_run(sorted(campaign.run_ids)[0], campaign.scope_id)
    assert run is not None
    one, other = _attested(run, "a" * 64), _attested(run, "b" * 64)
    remeasured = one.model_copy(
        update={
            "measure_definitions": {"field_accuracy": "b" * 64},
            "measure_remeasures": [
                MeasureRemeasure(
                    remeasured_at="2026-11-14T01:00:00+00:00",
                    completed_at="2026-11-14T01:05:00+00:00",
                    was={"field_accuracy": "a" * 64},
                    now={"field_accuracy": "b" * 64},
                    reread=["field_accuracy"],
                    results_reread=36,
                )
            ],
        }
    )

    digests = [run_evidence(each) for each in (run, one, other, remeasured)]

    assert len(set(digests)) == 4
    assert run_evidence(_attested(run, "a" * 64, note="said another way")) == run_evidence(one), "a note is prose"


def test_every_member_attested_moves_the_fingerprint_and_only_the_runs() -> None:
    host, campaign = _corpus_host()
    before = _aspect(host, campaign)

    def attest_all(digest: str) -> AnalysisContextBundle:
        for run_id in campaign.run_ids:
            run = host.storage.load_eval_run(run_id, campaign.scope_id)
            assert run is not None
            host.storage.save_eval_run(_attested(run, digest))
        return _aspect(host, campaign)

    one = attest_all("a" * 64)
    other = attest_all("b" * 64)

    assert _moved(before, one) == {"runs"}
    assert _moved(one, other) == {"runs"}
    assert len({before.fingerprint(), one.fingerprint(), other.fingerprint()}) == 3
