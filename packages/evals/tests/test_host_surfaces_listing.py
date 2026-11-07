"""A host's surfaces print a campaign the way its report reads (presentation build, chunk 6; Discodon, 2026-11-07).

Contracts held here (``campaigns_list``'s ``CampaignLine.report``, ``arm_names``, the report document's markdown and
text):

- **the headline**: a report document its editor passed lists its own headline, every figure key written out as its
  page prints it; a draft lists code's verdict. Code's warnings and its line on when criteria were set are listed in
  ``caveats`` beside the headline, never inside it, and ``analyses_list``'s verdict is still those lines then that
  verdict (Q7);
- **no cut**: no listing line holds a cut of a prompt or of the question. An arm whose level is too long to print is
  named by its campaign label, else by its lever and a letter (``prompt A``); the empty level reads ``no <lever>``
  (``no system prompt``), never ``(none)``; the question prints whole, in its own field, and in no other line but
  ``question_short``, the one field documented as cut (Q10);
- **no raw key**: no markdown, text or listing output of the ``toyhost_doc`` capture holds a figure key in braces, the
  analyst's own caveat labels included.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from packages.evals.tests.factories import analysed_aspect, make_analysis
from packages.evals.tests.fixtures.toyhost.campaign import toyhost_aspect_design, toyhost_campaign_of_one_aspect
from packages.evals.tests.fixtures.toyhost.corpus import (
    TOYHOST_SCOPE,
    TOYHOST_SUBJECT,
    toyhost_batch,
    toyhost_measurements,
)
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.hot_path_support import campaign_id_of, toyhost_from

from threetears.evals.analysis import assemble_campaign_bundle
from threetears.evals.analysis.arms import ELISION, arm_names
from threetears.evals.analysis.report.doc_page.text import above_appendix_text, doc_page_text
from threetears.evals.analysis.report_doc_store import read_report_doc
from threetears.evals.analysis.style import arm_name
from threetears.evals.kernel.campaign import VariantIndexEntry
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.identity import compute_variant_key
from threetears.evals.ops import CampaignDefinition, CampaignLine, campaign_create, campaigns_list, report_read
from threetears.evals.ops.analysis import analyses_list
from threetears.evals.schema import SweepableValue

#: The stored campaign whose report ``analysis_generate`` wrote: the hot path oracle's ``toyhost_doc`` capture.
_CAPTURE = Path(__file__).parent / "fixtures" / "hot_path_oracle" / "inputs" / "toyhost_doc.json"

#: A figure key in braces, as no surface but the JSON form may print one.
_RAW_KEY = re.compile(r"\{[a-z_.|]+\}")

#: The captured document's headline, which holds no key.
_HEADLINE = "The two token budgets read alike on invoice extraction."

#: Code's verdict over the captured campaign: its question was declared and not tested.
_CODE_VERDICT = "Declared question: not tested."


def _captured() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = json.loads(_CAPTURE.read_text())
    return documents


def _report_doc(documents: list[dict[str, Any]]) -> dict[str, Any]:
    (stored,) = [document for document in documents if document.get("doc_type") == "eval_report_doc"]
    return stored


def _edited(edit: Callable[[dict[str, Any]], None]) -> list[dict[str, Any]]:
    """The capture with its stored report document edited in place by ``edit``."""
    documents = copy.deepcopy(_captured())
    edit(_report_doc(documents))
    return documents


def _with_keys(stored: dict[str, Any]) -> None:
    """The document's headline and an analyst caveat's label citing the snapshot's figures by key."""
    doc = stored["doc"]
    doc["headline"] = "Over {cases} cases the two token budgets read alike."
    doc["confidence"]["caveats"].append(
        {"id": "analyst:few-cases", "label": "Only {cases} cases", "material_if": "only if more cases split them"}
    )


def _as_draft(stored: dict[str, Any]) -> None:
    stored["ending"] = "stored_with_findings"


def _listed(documents: list[dict[str, Any]]) -> tuple[EvalHost, CampaignLine]:
    host = toyhost_from(documents)
    (line,) = campaigns_list(host, TOYHOST_SCOPE).campaigns
    assert line.id == campaign_id_of(documents)
    return host, line


def _lines(line: CampaignLine) -> Iterator[str]:
    """Every line a listing prints of a campaign but the two question fields."""
    yield from (line.title, line.verdict_short or "", line.outcome_short or "")
    if line.report is not None:
        yield from (line.report.headline, *line.report.caveats, line.report.why_regenerate or "")


# --- the headline --------------------------------------------------------------------------------------------------


def test_a_passed_document_lists_its_own_headline_with_code_s_caveats_beside_it() -> None:
    host, line = _listed(_captured())
    assert line.report is not None and not line.report.draft

    assert line.report.headline == _HEADLINE
    assert line.report.caveats == [
        "Adopting 1024tok overrides the bar it misses on extract_invoice_fields.",
        "Adopting 256tok overrides the bar it misses on extract_invoice_fields.",
    ]
    (analysed,) = analyses_list(host, line.id, TOYHOST_SCOPE).analyses
    assert analysed.verdict == " ".join([*line.report.caveats, _CODE_VERDICT]), "the analysis line is still code's"


def test_a_draft_lists_code_s_verdict_never_the_writer_s_headline() -> None:
    _, line = _listed(_edited(_as_draft))
    assert line.report is not None and line.report.draft

    assert line.report.headline == _CODE_VERDICT
    assert len(line.report.caveats) == 2


def test_a_headline_citing_figures_is_listed_as_its_page_prints_it() -> None:
    documents = _edited(_with_keys)
    host, line = _listed(documents)
    page = report_read(host, line.id, TOYHOST_SCOPE, format="markdown").body
    assert line.report is not None

    assert line.report.headline == "Over 12 cases the two token budgets read alike."
    assert f"# {line.report.headline}" in page.splitlines(), "the listing and the page write the key alike"


# --- no cut of a prompt or the question ---------------------------------------------------------------------------

#: Two system prompts too long to print whole, which part only in their middle.
_PROMPTS = tuple(
    f"Read every invoice line {times} and quote the vendor's own words for each total you extract today"
    for times in ("twice", "once")
)


def _batch(label: str, instructions: str | None) -> Any:
    run = toyhost_batch(
        chunk_tokens=256,
        retriever_top_k=3,
        extraction_schema="v1",
        ocr_engine_version="tess-5.3.1",
        reviewer_pool="pool-a",
        batch_label=label,
    )
    return run if instructions is None else run.model_copy(update={"overlays": {"instructions": instructions}})


def _prompt_campaign(*, labelled: bool) -> tuple[EvalHost, CampaignLine]:
    """A campaign whose arms differ by their extractor instructions alone: none, and each of :data:`_PROMPTS`; with
    ``labelled``, the first long prompt's arm declares the label ``read twice``. Its report is stored code-only."""
    profile = toyhost_profile()
    host = toyhost_host(profile=profile)
    runs = [_batch("plain", None), _batch("twice", _PROMPTS[0]), _batch("once", _PROMPTS[1])]
    for run in runs:
        host.storage.save_eval_run(run)
        for result in toyhost_measurements(
            run, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8, layout_fidelity=3
        ):
            host.storage.save_eval_result(result)
    question = f"Which of these instructions keeps the totals right? {_PROMPTS[0]}"
    campaign = toyhost_campaign_of_one_aspect(
        id="7b0e2c55-5d0f-4c8e-9a3c-1f2e3d4c5b6c",
        scope_id=TOYHOST_SCOPE,
        name="instructions",
        subject_id=TOYHOST_SUBJECT.subject_id,
        subject_kind="extractor_config",
        behavior="extract_invoice_fields",
        template_id="",
        run_ids=[run.id for run in runs],
        design=toyhost_aspect_design(),
        question=question,
        created_by="test:fixture",
    )
    host.storage.save_campaign(campaign)
    if labelled:
        definition = CampaignDefinition(
            name="instructions, labelled",
            subject_id=TOYHOST_SUBJECT.subject_id,
            behavior="extract_invoice_fields",
            question=question,
            arms=[
                {"label": "plain", "role": "control", "from_run_ids": [runs[0].id]},
                {"label": "read twice", "from_run_ids": [runs[1].id]},
            ],
            aspects=[
                {
                    "label": "extract_invoice_fields",
                    "run_ids": [run.id for run in runs],
                    "design": toyhost_aspect_design().model_dump(mode="json"),
                }
            ],
        )
        campaign_id = campaign_create(host, definition, TOYHOST_SCOPE, created_by="test:fixture").id
    else:
        campaign_id = campaign.id
    bundle = assemble_campaign_bundle(host, campaign_id, TOYHOST_SCOPE)
    host.storage.save_analysis(
        make_analysis(
            campaign_id=campaign_id,
            scope_id=TOYHOST_SCOPE,
            aspects=[analysed_aspect(aspect) for aspect in bundle.aspects],
            campaign_bundle=bundle.frame(),
            observation_refs=bundle.run_ids,
        )
    )
    (line,) = [one for one in campaigns_list(host, TOYHOST_SCOPE).campaigns if one.id == campaign_id]
    return host, line


def test_no_listing_line_holds_a_cut_of_a_prompt_or_of_the_question() -> None:
    _, line = _prompt_campaign(labelled=False)
    assert line.report is not None and line.report.caveats, "the arms are named in the caveats"

    for printed in _lines(line):
        assert ELISION not in printed, printed
        assert "(none)" not in printed and "(empty)" not in printed, printed
        assert not any(prompt[:20] in printed for prompt in _PROMPTS), printed
    named = " ".join(line.report.caveats)
    assert all(arm in named for arm in ("no extractor instructions", "extractor instructions A")), named
    assert line.question.endswith(_PROMPTS[0]), "the question prints whole, in its own field"


def test_an_arm_a_person_labelled_is_listed_by_its_label() -> None:
    _, line = _prompt_campaign(labelled=True)
    assert line.report is not None

    for printed in _lines(line):
        assert ELISION not in printed, printed
    assert (
        line.report.headline
        == "Declared question: nothing separated extractor instructions A or read twice from plain."
    )
    assert "Adopting extractor instructions A overrides the bar it misses on extract_invoice_fields." in (
        line.report.caveats
    ), "the arm no one labelled wears one letter in the headline and its caveats"


def _entry(**levers: SweepableValue) -> VariantIndexEntry:
    return VariantIndexEntry(variant_key=compute_variant_key(levers), levers=levers)


def test_an_empty_level_reads_no_system_prompt_and_a_long_one_its_lever_and_a_letter() -> None:
    empty, first, second = index = [
        _entry(system_prompt=SweepableValue.of(None, display="(none)")),
        _entry(system_prompt=SweepableValue.of(_PROMPTS[0], display=_PROMPTS[0])),
        _entry(system_prompt=SweepableValue.of(_PROMPTS[1], display=_PROMPTS[1])),
    ]
    named = {key: arm_name(name).label for key, name in arm_names(index).items()}
    assert named == {
        empty.variant_key: "no system prompt",
        first.variant_key: "system prompt B",
        second.variant_key: "system prompt A",
    }, "lettered by the levels' text, so an index's order moves no letter"
    assert arm_names(list(reversed(index))) == arm_names(index)

    blank, long = index = [
        _entry(prompt=SweepableValue.of("", display="(empty)")),
        _entry(prompt=SweepableValue.of(_PROMPTS[0], display=_PROMPTS[0][:30] + ELISION)),
    ]
    named = {key: arm_name(name).label for key, name in arm_names(index).items()}
    assert named == {blank.variant_key: "no prompt", long.variant_key: "prompt A"}, "a host's own cut is a cut too"


# --- no raw figure key ---------------------------------------------------------------------------------------------


def test_no_markdown_text_or_listing_output_of_the_toyhost_doc_capture_holds_a_raw_figure_key() -> None:
    for documents in (_captured(), _edited(_with_keys)):
        host, line = _listed(documents)
        stored = _report_doc(documents)
        page = read_report_doc(host.storage, str(stored["id"]), TOYHOST_SCOPE).page
        outputs = {
            "markdown": report_read(host, line.id, TOYHOST_SCOPE, format="markdown").body,
            "markdown, every trial": report_read(host, line.id, TOYHOST_SCOPE, format="markdown", trials="all").body,
            "text": doc_page_text(page),
            "text above the appendix": above_appendix_text(page),
            "listing": "\n".join(_lines(line)),
        }
        for surface, output in outputs.items():
            assert not _RAW_KEY.findall(output), (surface, _RAW_KEY.findall(output))
    assert "- Only 12 cases. only if more cases split them" in doc_page_text(page), "an analyst's label is written out"
