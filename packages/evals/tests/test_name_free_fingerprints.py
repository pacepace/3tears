"""Names are presentation, never evidence: renaming a lever moves no fingerprint, and every page prints the new name.

Owner ruling, 2026-11-11. The bundle an analysis is frozen over carries ids and structured facts only (bundle shape
66), and every reader name is applied when a page is rendered:

- a derived arm's label is spelled from its levers' ids read as words (``chunk tokens 1024tok, retriever top k 3``),
  and a page prints it with the levers' stored names (``Chunk size 1024tok, retrieval depth 3``);
- the lockstep-group sentence, the factor-pair scan's completeness and each co-varying pair are worded at render from
  the facts the bundle holds, every factor by its reader name.

So renaming a lever, by the name a campaign stores on its declared axis or the one its host declares, moves no aspect
bundle fingerprint, no evidence digest and no campaign bundle fingerprint, and the page, built now or rebuilt from a
stored analysis, prints the name.

Mutations that turn this file red: a derived arm labelled with its levers' reader names; a page printing a derived
arm's frozen label instead of naming its levers; a factor sentence stored in the bundle, or worded with lever ids.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest

from threetears.evals.analysis import AnalysisContextBundle, assemble_context_bundle
from threetears.evals.analysis.bundle.schema import evidence_digest
from threetears.evals.analysis.campaign_bundle import CampaignBundle
from threetears.evals.analysis.campaigns import set_campaign_control
from threetears.evals.analysis.report.serialize_md import campaign_report_markdown
from threetears.evals.analysis.report.words import printed_arm_labels
from threetears.evals.analysis.service import analysis_report, assemble_campaign_bundle, campaign_report
from threetears.evals.kernel.campaign import CampaignDeclaration, EvalCampaign
from threetears.evals.kernel.host import EvalHost, HostProfile, SweepableRegistry
from threetears.evals.kernel.storage import EvalStorage
from packages.evals.tests.campaign_bundle_support import campaign_bundle
from packages.evals.tests.factories import analysed_aspect, make_analysis, make_campaign, memory_storage
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_aspect_design, toyhost_campaign_of_one_aspect
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile

#: Each run's chunk size and retriever depth: the control first, then two candidates code derives and labels.
_RUNS = ((256, 3), (1024, 3), (1024, 5))

#: What the candidates' labels are frozen as: each lever by its id read as words.
_FROZEN = ["chunk tokens 1024tok, retriever top k 3", "chunk tokens 1024tok, retriever top k 5"]

#: What a page prints for them once the levers are renamed.
_RENAMED = ["Chunk size 1024tok, retrieval depth 3", "Chunk size 1024tok, retrieval depth 5"]


def _host_renaming(profile: HostProfile, names: Mapping[str, str]) -> HostProfile:
    """``profile`` with its host declaring ``names`` for those levers, and nothing else moved."""
    host = profile.host_sweepables
    return replace(
        profile,
        host_sweepables=SweepableRegistry(
            [
                replace(declared, reader_name=names[declared.name]) if declared.name in names else declared
                for declared in host.declarations
            ],
            roles=host.roles,
        ),
    )


def _batches(host: EvalHost, *, again: bool = False) -> list[str]:
    """One toy batch per :data:`_RUNS` entry, stored with its results; their run ids. ``again`` runs each once more,
    under a run id of its own."""
    batches = [
        toyhost_batch(
            chunk_tokens=chunk, retriever_top_k=depth, extraction_schema="v1", ocr_engine_version="tess-5.3.1"
        )
        for chunk, depth in _RUNS
    ]
    if again:
        batches = [batch.model_copy(update={"id": str(uuid.uuid5(uuid.NAMESPACE_URL, batch.id))}) for batch in batches]
    for batch in batches:
        host.storage.save_eval_run(batch)
        for result in toyhost_measurements(
            batch, profile=host.profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8, layout_fidelity=3
        ):
            host.storage.save_eval_result(result)
    return [batch.id for batch in batches]


def _armed_campaign(*, second_aspect_names: str | None = None) -> tuple[EvalHost, EvalCampaign]:
    """The toy host holding a campaign over :data:`_RUNS`, its control named from the first run, so its other two arms
    are derived and labelled by the two levers they differ on. With ``second_aspect_names``, a second aspect over runs
    of its own declares ``chunk_tokens`` again and stores that name for it, where the first stores none."""
    host = toyhost_host(profile=toyhost_profile())
    runs = _batches(host)
    campaign = toyhost_campaign_of_one_aspect(
        id="5f0c2a71-9e4b-4d3a-8c1f-2b7e6d5a4c30",
        scope_id=TOYHOST_SCOPE,
        name="name-free fingerprints",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=runs,
        design=toyhost_aspect_design(),
        created_by="test:fixture",
    )
    if second_aspect_names is not None:
        (first,) = campaign.aspects
        assert first.design is not None
        axes = [axis.model_copy(update={"reader_name": second_aspect_names}) for axis in first.design.axes]
        second = first.model_copy(
            update={
                "label": "second_look",
                "run_ids": _batches(host, again=True),
                "design": first.design.model_copy(update={"axes": axes}),
            }
        )
        campaign = campaign.model_copy(update={"aspects": [first, second]})
    host.storage.save_campaign(campaign)
    campaign = set_campaign_control(
        host.storage, campaign.id, campaign.scope_id, runs[0], set_by="t", profile=host.profile
    )
    return host, campaign


def _renamed(host: EvalHost, campaign: EvalCampaign) -> EvalHost:
    """Rename both levers the derived arms are spelled by: ``chunk_tokens`` by the name the campaign stores on its
    declared axis, ``retriever_top_k`` by the one its host declares. Returns the host declaring the new name."""
    aspects = []
    for aspect in campaign.aspects:
        assert aspect.design is not None
        axes = [
            axis.model_copy(update={"reader_name": "chunk size"}) if axis.axis_id == "chunk_tokens" else axis
            for axis in aspect.design.axes
        ]
        aspects.append(aspect.model_copy(update={"design": aspect.design.model_copy(update={"axes": axes})}))
    host.storage.save_campaign(campaign.model_copy(update={"aspects": aspects}))
    return replace(host, profile=_host_renaming(host.profile, {"retriever_top_k": "retrieval depth"}))


def _identities(bundle: CampaignBundle) -> dict[str, Any]:
    """Every fingerprint and digest a campaign bundle is identified by."""
    return {
        "campaign": bundle.fingerprint(),
        "aspects": [aspect.fingerprint() for aspect in bundle.aspects],
        "evidence": [evidence_digest(aspect.to_dict()) for aspect in bundle.aspects],
    }


def _derived(bundle: CampaignBundle) -> list[str]:
    return [arm.variant_key for arm in bundle.arms if arm.role == "candidate"]


def test_renaming_a_lever_moves_no_fingerprint_or_digest_and_the_page_prints_the_new_name() -> None:
    host, campaign = _armed_campaign()
    before = assemble_campaign_bundle(host, campaign.id, campaign.scope_id)
    renamed = _renamed(host, campaign)

    after = assemble_campaign_bundle(renamed, campaign.id, campaign.scope_id)

    assert (before.lever_names["chunk_tokens"], before.lever_names["retriever_top_k"]) == (
        "chunk tokens",
        "retriever top k",
    )
    assert (after.lever_names["chunk_tokens"], after.lever_names["retriever_top_k"]) == (
        "chunk size",
        "retrieval depth",
    )
    assert _identities(after) == _identities(before)
    # The derived arms are frozen by their levers' ids, whatever the levers are called.
    assert [arm.label for arm in after.arms if arm.role == "candidate"] == _FROZEN
    assert [arm.label for arm in before.arms] == [arm.label for arm in after.arms]
    printed = printed_arm_labels(after)
    assert [printed[key] for key in _derived(after)] == _RENAMED
    page = campaign_report_markdown(campaign_report(renamed, campaign.id, campaign.scope_id))
    for name in _RENAMED:
        assert name in page
    assert "Chunk tokens 1024tok" not in page and "retriever top k 3" not in page


def test_before_any_rename_a_derived_arm_prints_as_it_did() -> None:
    host, campaign = _armed_campaign()
    bundle = assemble_campaign_bundle(host, campaign.id, campaign.scope_id)

    printed = printed_arm_labels(bundle)

    assert [printed[key] for key in _derived(bundle)] == [name[:1].upper() + name[1:] for name in _FROZEN]


# --- one campaign names a lever one way --------------------------------------------------------------------------


def test_one_campaign_names_a_lever_one_way_whichever_aspect_declares_its_name() -> None:
    # The first aspect stores no name for chunk_tokens (the host's is `chunk tokens`); the second stores `chunk size`.
    # Every aspect, the derived arms and the page name it by the one rule: the first declared axis storing a name.
    host, campaign = _armed_campaign(second_aspect_names="chunk size")
    assert [aspect.design.axes[0].reader_name for aspect in campaign.aspects if aspect.design] == [None, "chunk size"]

    bundle = assemble_campaign_bundle(host, campaign.id, campaign.scope_id)

    assert [aspect.lever_names["chunk_tokens"] for aspect in bundle.aspects] == ["chunk size", "chunk size"]
    assert bundle.lever_names["chunk_tokens"] == "chunk size"
    printed = printed_arm_labels(bundle)
    assert [printed[key] for key in _derived(bundle)] == [
        "Chunk size 1024tok, retriever top k 3",
        "Chunk size 1024tok, retriever top k 5",
    ]
    page = campaign_report_markdown(campaign_report(host, campaign.id, campaign.scope_id))
    assert "Chunk size 1024tok" in page
    assert "chunk tokens" not in page.lower(), "a surface named the lever by the host's name, not the campaign's"


def test_a_campaign_bundle_whose_aspects_name_one_lever_apart_is_refused() -> None:
    host, campaign = _armed_campaign(second_aspect_names="chunk size")
    bundle = assemble_campaign_bundle(host, campaign.id, campaign.scope_id)
    first, second = bundle.aspects
    apart = second.model_copy(update={"lever_names": {**second.lever_names, "chunk_tokens": "chunk width"}})

    with pytest.raises(ValueError, match=r"name lever\(s\) \['chunk_tokens'\] apart"):
        CampaignBundle.model_validate({**bundle.model_dump(exclude={"aspects"}), "aspects": [first, apart]})


# --- the factor sentences are worded at render time ------------------------------------------------------------------

#: The toy host's three levers, renamed by its declarations.
_NAMES = {"chunk_tokens": "chunk size", "retriever_top_k": "retrieval depth", "extraction_schema": "field schema"}


def _factored(batches: Sequence[dict[str, Any]], profile: HostProfile) -> tuple[AnalysisContextBundle, EvalStorage]:
    """A campaign of one toy batch per entry under ``profile``, every batch recording each value it names."""
    storage, _ = memory_storage()
    runs = []
    for values in batches:
        run = toyhost_batch(grader_version="grade-2.1.0", **values)
        storage.save_eval_run(run)
        for result in toyhost_measurements(run, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8):
            storage.save_eval_result(result)
        runs.append(run)
    campaign = make_campaign(
        id="factors",
        scope_id=TOYHOST_SCOPE,
        name="factors",
        subject_id=runs[0].subject_snapshot.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        run_ids=[run.id for run in runs],
        created_by="test:fixture",
    )
    storage.save_campaign(campaign)
    return assemble_context_bundle(campaign, storage=storage, profile=profile), storage


#: Three levers and an apparatus dimension in lockstep, low on two runs and high on two.
_LOW = {"chunk_tokens": 256, "retriever_top_k": 3, "extraction_schema": "v1", "ocr_engine_version": "tess-5.3.1"}
_HIGH = {"chunk_tokens": 512, "retriever_top_k": 5, "extraction_schema": "v2", "ocr_engine_version": "tess-5.4.0"}
_LOCKSTEP = [
    {**_LOW, "reviewer_pool": "pool-a"},
    {**_LOW, "reviewer_pool": "pool-b"},
    {**_HIGH, "reviewer_pool": "pool-a"},
]

#: chunk_tokens nested in retriever_top_k: a co-varying pair outside any group.
_NESTED = [
    {"chunk_tokens": 256, "retriever_top_k": 3, "reviewer_pool": "pool-a"},
    {"chunk_tokens": 256, "retriever_top_k": 3, "reviewer_pool": "pool-b"},
    {"chunk_tokens": 512, "retriever_top_k": 5, "reviewer_pool": "pool-a"},
    {"chunk_tokens": 1024, "retriever_top_k": 5, "reviewer_pool": "pool-a"},
]

#: The sentences a page says of each, every lever by its renamed name.
_ALIASED = (
    "Chunk size, field schema, ocr engine version and retrieval depth move together across all 3 runs, splitting them "
    "into the same 2 groups; no comparison separates them, so a difference across them belongs to all 4 at once."
)
_PAIR = "Chunk size and retrieval depth co-vary; 3 of their 6 combinations never ran"
_COMPLETENESS = "3 factor pair(s) examined over the 3 factors that varied; 1 co-vary"


def _stored_page(bundle: AnalysisContextBundle, storage: EvalStorage) -> str:
    """The page of an analysis frozen over ``bundle``, rebuilt from the store."""
    campaign = storage.load_campaign(bundle.campaign_id, bundle.scope_id)
    assert campaign is not None
    frozen = campaign_bundle(bundle)
    analysis = make_analysis(
        campaign_id=bundle.campaign_id,
        scope_id=bundle.scope_id,
        behavior=bundle.behavior,
        aspects=[analysed_aspect(bundle)],
        campaign_bundle=frozen.frame(),
        design_snapshot=CampaignDeclaration.of(campaign),
    )
    storage.save_analysis(analysis)
    return campaign_report_markdown(analysis_report(storage, analysis.id, bundle.scope_id))


def test_the_factor_sentences_name_each_lever_by_its_reader_name_and_move_no_fingerprint() -> None:
    profile = toyhost_profile()
    renamed = _host_renaming(profile, _NAMES)

    for batches, said in ((_LOCKSTEP, [_ALIASED]), (_NESTED, [_PAIR, _COMPLETENESS])):
        plain, _ = _factored(batches, profile)
        named, storage = _factored(batches, renamed)

        assert named.fingerprint() == plain.fingerprint()
        assert evidence_digest(named.to_dict()) == evidence_digest(plain.to_dict())
        # Facts by id in the bundle, never a sentence.
        stored = named.to_dict()
        assert all("sentence" not in group for group in stored["aliased_factors"])
        assert stored["factor_pairs"] is not None and "completeness" not in stored["factor_pairs"]
        page = _stored_page(named, storage)
        for sentence in said:
            assert sentence in page, sentence
        # Never by its id, as the bundle worded these before names left the evidence.
        assert "chunk_tokens and retriever_top_k co-vary" not in page
        assert "chunk_tokens, extraction_schema" not in page
