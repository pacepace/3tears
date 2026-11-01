"""A report writer campaign the promotion read decides on, scripted over the toy host with no model.

A control and a candidate arm measured over the toy host's twelve documents, each document stored as a writer case its
mint put on one side of the split, every run naming a case set and scored by one judge on one rubric. What each arm
measures is given, so what the engine's corrected verdicts say, and the read's decision, follow from it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from packages.evals.tests.fixtures.toyhost.campaign import (
    TOYHOST_NARROW,
    TOYHOST_WIDE,
    toyhost_aspect_design,
    toyhost_campaign_of_one_aspect,
)
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_DOCUMENTS,
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import TOYHOST_MEASURES, toyhost_profile
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.analysis.campaigns import set_campaign_control
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.host.measures import MeasureRegistry
from threetears.evals.ops.report_writer.writer_contract import WRITER_KIND_MEASURES
from threetears.evals.ops.report_writer.writer_kind import (
    WriterCase,
    WriterCaseOrigin,
    WriterSide,
    writer_case_payload,
)
from threetears.evals.schema.models import CaseSetRef, EvalTestCase

#: The held-out set every run names unless a test says otherwise.
HELD_OUT = CaseSetRef(name="writer-held-out", version=1)

#: The scoring judge every run names unless a test says otherwise.
SCORING_JUDGE = "google/gemini-flash"

#: What the control measures on each document.
CONTROL: dict[str, Any] = {"cost_usd": 0.02, "total_ms": 900.0, "field_accuracy": 0.78, "layout_fidelity": None}

#: A candidate shown better than the control on every reading: cheaper, faster, more accurate.
BETTER: dict[str, Any] = {"cost_usd": 0.01, "total_ms": 600.0, "field_accuracy": 0.99, "layout_fidelity": None}

#: The candidate the read is asked about: the arm addressed from the wide batch, by its label.
CANDIDATE = "1024tok"

#: What a writer run lands on ``host_measures`` for each document, by the writer's measure names: the arm's own.
WriterMeasures = Mapping[str, Mapping[str, float | bool]]


def promotion_campaign(
    candidate: dict[str, Any],
    control: dict[str, Any] = CONTROL,
    *,
    side: WriterSide = "held_out",
    case_sets: tuple[CaseSetRef | None, CaseSetRef | None] = (HELD_OUT, HELD_OUT),
    judges: tuple[str, str] = (SCORING_JUDGE, SCORING_JUDGE),
    writer: tuple[WriterMeasures, WriterMeasures] | None = None,
) -> tuple[EvalHost, str]:
    """The toy host holding a campaign of a control and a candidate, its writer cases on ``side``; and its id.

    :param candidate: What the candidate arm measures on each document.
    :ptype candidate: dict[str, Any]
    :param control: What the control arm measures on each document.
    :ptype control: dict[str, Any]
    :param side: The side of the split every case was minted on.
    :ptype side: WriterSide
    :param case_sets: The case set each run names, control first.
    :ptype case_sets: tuple[CaseSetRef | None, CaseSetRef | None]
    :param judges: The scoring judge each run names, control first.
    :ptype judges: tuple[str, str]
    :param writer: The writer's measures each arm lands on each document, control first; none when ``None``. The
        profile declares the writer's measures either way.
    :ptype writer: tuple[WriterMeasures, WriterMeasures] | None
    :return: The host and the campaign's id.
    :rtype: tuple[EvalHost, str]
    """
    profile = toyhost_profile()
    profile = replace(
        profile,
        measures=MeasureRegistry((*TOYHOST_MEASURES, *WRITER_KIND_MEASURES), families=profile.measures.families),
    )
    host = toyhost_host(profile=profile)
    run_ids = []
    for level, values, case_set, judge, landed in zip(
        (TOYHOST_NARROW, TOYHOST_WIDE), (control, candidate), case_sets, judges, writer or ({}, {}), strict=True
    ):
        batch = toyhost_batch(
            chunk_tokens=level, retriever_top_k=3, extraction_schema="v1", ocr_engine_version="tess-5.3.1"
        ).model_copy(
            update={
                "case_set": case_set,
                "judge_model": judge,
                "judge_temperature": 0.0,
                "template_digest": "a" * 64,
            }
        )
        host.storage.save_eval_run(batch)
        for result in toyhost_measurements(batch, profile=profile, **values):
            if result.test_case_id in landed:
                host_measures = {**result.host_measures, **landed[result.test_case_id]}
                result = result.model_copy(update={"host_measures": host_measures})
            host.storage.save_eval_result(result)
        run_ids.append(batch.id)
    for document in TOYHOST_DOCUMENTS:
        origin = WriterCaseOrigin(
            host_id="toyhost",
            minted_by="test",
            source="snapshot",
            campaign_id=None,
            analysis_id=None,
            group=document,
            side=side,
            case_set=HELD_OUT.name if side == "held_out" else "writer-dev",
            held_out_fraction=0.3,
            skill=load_skill().digest,
        )
        host.storage.save_test_case(
            EvalTestCase(
                id=document,
                scope_id=TOYHOST_SCOPE,
                template_id="writer-template",
                host_payload=writer_case_payload(
                    WriterCase(
                        snapshot={"id": document, "campaign_id": None, "report_input_digest": document},
                        origin=origin,
                    )
                ),
            )
        )
    campaign = toyhost_campaign_of_one_aspect(
        id="5f1c2e3d-4b5a-4c6d-8e7f-90a1b2c3d4e5",
        scope_id=TOYHOST_SCOPE,
        name="writer promotion",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=run_ids,
        design=toyhost_aspect_design(bars=[]),
        created_by="test:fixture",
    )
    host.storage.save_campaign(campaign)
    set_campaign_control(host.storage, campaign.id, TOYHOST_SCOPE, run_ids[0], set_by="test", profile=profile)
    return host, campaign.id
