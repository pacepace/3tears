"""A host's surfaces print a campaign the way its report reads (presentation build, chunk 6; Discodon, 2026-11-07).

Contracts held here (``campaigns_list``'s ``CampaignLine.report``, the reader name of an arm, the report document's
markdown and text):

- **the headline**: a report document its editor passed lists its own headline, every figure key written out as its
  page prints it; a draft lists code's verdict. Code's warnings and its line on when criteria were set are listed in
  ``caveats`` beside the headline, never inside it, and ``analyses_list``'s verdict is those lines then that verdict
  (Q7). A document whose snapshot is gone, or that no longer reads, lists code's verdict and says to regenerate;
- **no cut**: no listing line holds a cut of a prompt or of the question. At render time only, an arm whose level is
  too long to print is named by its campaign label, else by its lever alone (``Extractor instructions``), one name on
  every surface; where two arms would then read alike, each surface adds the tag it prints the arm by (``Extractor
  instructions (B)``), the document page and the listing alike, which no figure key appends twice; the empty level of
  a text lever reads ``no <lever>``, never ``(none)``, and an empty number lever keeps its placeholder; the question
  prints whole, in its own field (Q10);
- **frozen evidence never moves**: the labels a bundle freezes, and its fingerprint, are the ones the build before this
  change froze for the same campaign;
- **no raw key**: no markdown, text or listing output of the ``toyhost_doc`` capture holds a figure key in braces, the
  analyst's own caveat labels included, and a strict build refuses a label whose key does not resolve.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

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

from threetears.evals.analysis import assemble_campaign_bundle, campaign_report, report_markdown, serialize_report
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.arms import ELISION
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.campaign.answer_blocks import NO_LONGER_READS
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.text import above_appendix_text, doc_page_text
from threetears.evals.analysis.report.figure_keys import render, written_arm
from threetears.evals.analysis.report.words import (
    campaign_arm_tags,
    input_arm_tags,
    reader_arm_labels,
    reader_arm_name,
    told_apart,
)
from threetears.evals.analysis.report_arms import fact_arms
from threetears.evals.analysis.report_doc_store import read_report_doc
from threetears.evals.analysis.report_input_build import report_input
from threetears.evals.kernel.host import EvalHost
from threetears.evals.kernel.report_doc import ReportDoc
from threetears.evals.ops import CampaignDefinition, CampaignLine, campaign_create, campaigns_list, report_read
from threetears.evals.ops.analysis import analyses_list

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


def _without(doc_type: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``documents`` without the one of ``doc_type``."""
    return [document for document in documents if document.get("doc_type") != doc_type]


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


def test_a_headline_whose_snapshot_is_gone_lists_code_s_verdict_and_says_to_regenerate() -> None:
    _, line = _listed(_without("eval_report_snapshot", _edited(_with_keys)))
    assert line.report is not None

    assert line.report.headline == _CODE_VERDICT, "never the headline's braces"
    assert line.report_state == "needs_regeneration"
    assert line.report.why_regenerate is not None and line.report.why_regenerate.startswith(NO_LONGER_READS)


def test_a_document_that_no_longer_reads_lists_code_s_verdict_and_says_to_regenerate() -> None:
    _, line = _listed(_without("eval_report_doc", _captured()))
    assert line.report is not None

    assert line.report.headline == _CODE_VERDICT
    assert line.report_state == "needs_regeneration" and line.report.why_regenerate is not None


# --- no cut of a prompt or the question ---------------------------------------------------------------------------

#: Two prompts too long to print whole, which part only in their middle.
_PROMPTS = tuple(
    f"Read every invoice line {times} and quote the vendor's own words for each total you extract today"
    for times in ("twice", "once")
)

#: A campaign's id, as each campaign below is stored.
_PROMPTS_CAMPAIGN = "7b0e2c55-5d0f-4c8e-9a3c-1f2e3d4c5b6c"


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


#: Each arm's batch label and extractor instructions: two long prompts and none (an empty level, which the engine
#: displays ``(none)``).
_THREE_ARMS: tuple[tuple[str, str | None], ...] = (("twice", _PROMPTS[0]), ("once", _PROMPTS[1]), ("plain", None))

#: One long prompt and none.
_TWO_ARMS: tuple[tuple[str, str | None], ...] = (("twice", _PROMPTS[0]), ("plain", None))


def _prompt_host(
    question: str = "", arms: tuple[tuple[str, str | None], ...] = _THREE_ARMS
) -> tuple[EvalHost, list[Any]]:
    """A host holding a campaign whose arms differ by their extractor instructions alone (:data:`_THREE_ARMS` by
    default)."""
    profile = toyhost_profile()
    host = toyhost_host(profile=profile)
    runs = [_batch(label, instructions) for label, instructions in arms]
    for run in runs:
        host.storage.save_eval_run(run)
        for result in toyhost_measurements(
            run, profile=profile, cost_usd=0.02, total_ms=900.0, field_accuracy=0.8, layout_fidelity=3
        ):
            host.storage.save_eval_result(result)
    host.storage.save_campaign(
        toyhost_campaign_of_one_aspect(
            id=_PROMPTS_CAMPAIGN,
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
    )
    return host, runs


def _analysed(host: EvalHost, campaign_id: str) -> CampaignLine:
    """``campaign_id``'s line once a code-only analysis is stored over its bundle."""
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
    return line


#: What the build before render-time naming (0d801dcc0) froze for :func:`_prompt_host`'s campaign: its arms' labels
#: and its fingerprint, taken on that build. Render-time naming leaves both as they were.
_FROZEN_LABELS = [
    "Read every invoice line twice an…endor's own wo…",
    "Read every invoice line once and…ndor's own wor…",
    "(none)",
]
_FROZEN_FINGERPRINT = "v3:2742f149003a50d9f5bfb1af67dff6fdd763a4ec893af71a5aa0e1b5960649f7"


def test_the_bundle_freezes_the_labels_and_fingerprint_it_froze_before() -> None:
    host, _ = _prompt_host()
    bundle = assemble_campaign_bundle(host, _PROMPTS_CAMPAIGN, TOYHOST_SCOPE)

    assert [arm.label for arm in bundle.arms] == _FROZEN_LABELS
    assert bundle.fingerprint() == _FROZEN_FINGERPRINT


def test_no_listing_line_holds_a_cut_of_a_prompt_or_of_the_question() -> None:
    question = f"Which of these instructions keeps the totals right? {_PROMPTS[0]}"
    host, _ = _prompt_host(question)
    line = _analysed(host, _PROMPTS_CAMPAIGN)
    assert line.report is not None and line.report.caveats, "the arms are named in the caveats"

    for printed in _lines(line):
        assert ELISION not in printed, printed
        assert "(none)" not in printed, printed
        assert not any(prompt[:20] in printed for prompt in _PROMPTS), printed
    assert line.title == (
        "Extractor instructions (A) vs Extractor instructions (B) vs no extractor instructions · "
        "extract_invoice_fields (instructions)"
    )
    assert line.question == question, "the question prints whole, in its own field"


def _bare(name: str) -> str:
    """A name without the tag that tells it apart."""
    return re.sub(r" \([A-Z]+\)$", "", name)


@pytest.mark.parametrize("arms", [_TWO_ARMS, _THREE_ARMS], ids=["two arms", "three arms"])
def test_an_arm_reads_one_name_on_every_page_and_one_tag_on_one_listing_line(
    arms: tuple[tuple[str, str | None], ...],
) -> None:
    """One name per arm everywhere: its lever, never a cut, and ``no <lever>`` for none. Where two arms would read
    alike, each surface tells them apart by the tag it prints the arm by: the code page and its aspect page by their
    charts' tag, and the document page and the listing (whose headline is the document's) by the document's, so
    one listing line, headline and caveats, never names an arm two ways."""
    host, _ = _prompt_host(arms=arms)
    bundle = assemble_campaign_bundle(host, _PROMPTS_CAMPAIGN, TOYHOST_SCOPE)
    code_tags = campaign_arm_tags(bundle.aspects, bundle.readings.control)
    code = reader_arm_labels(bundle, code_tags)
    listed_tags = input_arm_tags(bundle)
    listed = reader_arm_labels(bundle, listed_tags)
    report = campaign_report(host, _PROMPTS_CAMPAIGN, TOYHOST_SCOPE)
    page, aspect = serialize_report(report, "markdown"), report_markdown(report.aspects[0].report)
    built = report_input(bundle)
    doc = {arm.id: arm for arm in fact_arms(built)}
    line = _analysed(host, _PROMPTS_CAMPAIGN)
    assert line.report is not None
    caveats = " ".join(line.report.caveats)

    keys = [entry.variant_key for entry in bundle.aspects[0].variant_index]
    assert len(keys) == len(arms)
    for key in keys:
        held = doc[bundle.label_of(key)]
        assert _bare(code[key]) == _bare(listed[key]) == _bare(held.name), "one name, its tag aside"
        assert code[key] in page and code[key] in aspect, code[key]
        assert (held.name, held.tag) == (listed[key], listed_tags[key]), "the listing names it as the document does"
        assert f"Adopting {listed[key]} overrides" in caveats, (listed[key], caveats)
        assert render(f"{{arm:{held.id}}}", built, {arm.id: arm.name for arm in doc.values()}) == written_arm(
            held.name, held.tag
        )
        for name, tags in ((code[key], code_tags), (listed[key], listed_tags)):
            assert name == _bare(name) or name.endswith(f" ({tags[key]})"), name
    assert {_bare(name) for name in listed.values()} == {"Extractor instructions", "no extractor instructions"}
    told = [name for name in listed.values() if name != _bare(name)]
    assert len(told) == (0 if arms == _TWO_ARMS else 2), "only arms that would read alike carry a tag"
    for printed in (page, aspect, caveats, line.title):
        assert ELISION not in printed and "(none)" not in printed


def test_a_name_told_apart_by_its_tag_is_written_with_its_tag_once() -> None:
    host, _ = _prompt_host()
    built = report_input(assemble_campaign_bundle(host, _PROMPTS_CAMPAIGN, TOYHOST_SCOPE))
    arms = fact_arms(built)
    written = render(" vs ".join(f"{{arm:{arm.id}}}" for arm in built.arms), built, {arm.id: arm.name for arm in arms})

    assert [arm.id for arm in built.arms] == _FROZEN_LABELS, "the report input keeps the frozen names"
    assert written == "Extractor instructions (A) vs Extractor instructions (B) vs no extractor instructions (C)"


def test_an_arm_a_person_labelled_is_listed_by_its_label() -> None:
    host, runs = _prompt_host()
    definition = CampaignDefinition(
        name="instructions, labelled",
        subject_id=TOYHOST_SUBJECT.subject_id,
        behavior="extract_invoice_fields",
        question="Which instructions keep the totals right?",
        arms=[
            {"label": "plain", "role": "control", "from_run_ids": [runs[2].id]},
            {"label": "read twice", "from_run_ids": [runs[0].id]},
        ],
        aspects=[
            {
                "label": "extract_invoice_fields",
                "run_ids": [run.id for run in runs],
                "design": toyhost_aspect_design().model_dump(mode="json"),
            }
        ],
    )
    line = _analysed(host, campaign_create(host, definition, TOYHOST_SCOPE, created_by="test:fixture").id)
    assert line.report is not None

    for printed in _lines(line):
        assert ELISION not in printed, printed
    assert (
        line.report.headline == "Declared question: nothing separated Extractor instructions or read twice from plain."
    )
    assert "Adopting Extractor instructions overrides the bar it misses on extract_invoice_fields." in (
        line.report.caveats
    ), "the arm no one labelled reads one name in the headline and its caveats"


def test_an_empty_text_level_reads_no_lever_and_an_empty_number_keeps_its_placeholder() -> None:
    text = frozenset({"system_prompt"})

    assert reader_arm_name("(none)", {"system_prompt": "(none)"}, text=text) == "no system prompt"
    assert reader_arm_name("(none)", {"temperature": "(none)"}, text=text) == "(none)"
    shown = _PROMPTS[0][:59] + ELISION  # as the engine displays a long level, which a name then cuts again
    cut = _FROZEN_LABELS[0]
    assert reader_arm_name(cut, {"system_prompt": shown}, text=text) == "System prompt"
    assert reader_arm_name(f"{cut} (arm 0123456789ab)", {"system_prompt": shown}, text=text) == "System prompt", (
        "a tag, never a digest, tells two cut alike apart"
    )
    assert reader_arm_name("v2", {"system_prompt": shown}, text=text) == "v2", "a given name is as given"
    assert told_apart(
        {"a": "System prompt", "b": "System prompt", "c": "v1"}, {"a": "A", "b": "B", "c": "P"}, "ab"
    ) == {
        "a": "System prompt (A)",
        "b": "System prompt (B)",
        "c": "v1",
    }


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


def test_a_strict_build_refuses_an_analyst_label_whose_key_does_not_resolve() -> None:
    documents = _edited(_with_keys)
    host = toyhost_from(documents)
    stored = _report_doc(documents)
    snapshot = read_report_doc(host.storage, str(stored["id"]), TOYHOST_SCOPE).snapshot
    doc = copy.deepcopy(stored["doc"])
    doc["confidence"]["caveats"][-1]["label"] = "Only {no.such_key} cases"
    unresolved = ReportDoc.model_validate(doc)

    with pytest.raises(UnresolvableReference):
        build(unresolved, frozen_input(snapshot), strict=True)
    assert "Only — cases" in doc_page_text(build(unresolved, frozen_input(snapshot), strict=False))
