"""A host's surfaces print a campaign the way its report reads (Discodon, 2026-11-07).

Contracts held here (``campaigns_list``'s ``CampaignLine.report``, the reader name of an arm, the report document's
markdown and text):

- **the headline**: a report document its editor passed lists its own headline, every figure key written out as its
  page prints it; a draft lists code's verdict. Code's warnings and its line on when criteria were set are listed in
  ``caveats`` beside the headline, never inside it, and ``analyses_list``'s verdict is those lines then that verdict
  (owner ruling). A document whose snapshot is gone, or that no longer reads, lists code's verdict and says to regenerate;
- **no cut**: no listing line holds a cut of a prompt or of the question. At render time only, an arm whose level is
  too long to print is named by its campaign label, else by its lever alone (``Extractor instructions``), one name on
  every surface; where two arms would then read alike, each surface adds the tag it prints the arm by (``Extractor
  instructions (B)``), the document page and the listing alike, which no figure key appends twice; the empty level of
  a text lever reads ``no <lever>``, never ``(none)``, and an empty number lever keeps its placeholder; the question
  prints whole, in its own field (owner ruling);
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

from threetears.evals.analysis import (
    CampaignBundle,
    assemble_campaign_bundle,
    campaign_report,
    report_markdown,
    serialize_report,
)
from threetears.evals.analysis.analyst.snapshot import frozen_input
from threetears.evals.analysis.arms import ELISION
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.campaign.answer_blocks import NO_LONGER_READS
from threetears.evals.analysis.report.doc_page.build import build
from threetears.evals.analysis.report.doc_page.text import above_appendix_text, doc_page_text
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.figure_keys import render, written_arm
from threetears.evals.analysis.report.words import (
    campaign_arm_tags,
    input_arm_tags,
    printed_arm_labels,
    reader_arm_labels,
    reader_arm_name,
    told_apart,
)
from threetears.evals.analysis.report.fact_arms import fact_arms
from threetears.evals.analysis.report.doc_page.store import read_report_doc
from threetears.evals.analysis.report.input_build import report_input
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
_CODE_VERDICT = "The declared question is untested."


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


def test_a_draft_lists_its_own_headline_marked_a_draft_with_code_s_caveats_beside_it() -> None:
    # Re-pinned when the written page came to open on its question: a draft lists "Draft: <its headline>" (DRAFT_HEADLINE), where it
    # listed code's verdict; code's caveats stay beside it.
    _, line = _listed(_edited(_as_draft))
    assert line.report is not None and line.report.draft

    assert line.report.headline == f"Draft: {_HEADLINE}"
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


def _batch(label: str, instructions: str | None, chunk_tokens: int = 256) -> Any:
    run = toyhost_batch(
        chunk_tokens=chunk_tokens,
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
#: and its fingerprint, taken on that build. Render-time naming leaves both as they were. The fingerprint was re-pinned
#: at bundle shape 66: the shape-65 bundle the old pin was taken over, moved by
#: ``fixtures/report_inputs/migrate_bundle_shape_66.py``, computes this one and is this build's bundle leaf for leaf,
#: less ``lever_names``. It was re-pinned at campaign bundle shape 4 the same way as ``test_declared_names``'s: this
#: build's bundle, read as the build before it fingerprinted a shape-3 frame, computes the old pin.
_FROZEN_LABELS = [
    "Read every invoice line twice an…endor's own wo…",
    "Read every invoice line once and…ndor's own wor…",
    "(none)",
]
_FROZEN_FINGERPRINT = "v3:8c3c29f48b94b1c5366f8a54e55e77c5947f601b41720fb899aa7e04b54bd678"


def test_the_bundle_freezes_the_labels_and_fingerprint_it_froze_before() -> None:
    host, _ = _prompt_host()
    bundle = assemble_campaign_bundle(host, _PROMPTS_CAMPAIGN, TOYHOST_SCOPE)

    assert [arm.label for arm in bundle.arms] == _FROZEN_LABELS
    assert _before_the_manifest(bundle).fingerprint() == _FROZEN_FINGERPRINT


def _before_the_manifest(bundle: CampaignBundle) -> CampaignBundle:
    """``bundle`` as a build before fingerprint version 3 held it: no aspect carries an evidence manifest, so each is
    named in the frame by its ``v2:`` fingerprint and the campaign's fingerprint is the one that build took. The
    manifest and the aspects' ``v3:`` names are all that version 3 moved (#1062)."""
    document = bundle.to_dict()
    aspects = [{key: value for key, value in aspect.items() if key != "evidence"} for aspect in document["aspects"]]
    return CampaignBundle.from_dict({**document, "aspects": aspects})


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


#: Two prompts that keep a ``, `` once a name cuts them, each beside its own chunk size: arms spelled from two levers.
_COMMA_ARMS: tuple[tuple[str, int, str], ...] = (
    ("comma-a", 256, "Read every invoice, line by line, twice and quote the vendor's own words for each total"),
    ("comma-b", 512, "Read every invoice, line by line, once and quote the vendor's own words for each total"),
)


def test_a_derived_arm_whose_cut_level_keeps_a_comma_is_spelled_and_printed_by_its_levers_names() -> None:
    """A derived arm's spelling was read back off its name's text, split on ``, ``: a cut prompt that kept a comma
    left it unspelled, and its frozen label printed raw, lever ids and all, on every page. It is spelled from its
    levels as data now, and the campaign page and the listing name it by its levers' names."""
    host = _levered_host(_COMMA_ARMS)
    bundle = assemble_campaign_bundle(host, _PROMPTS_CAMPAIGN, TOYHOST_SCOPE)

    derived = [arm for arm in bundle.arms if arm.role == "candidate"]
    assert len(derived) == 2
    for arm in derived:
        spelling = bundle.arm_spellings[arm.variant_key]
        assert len(spelling.levers) == 2 and spelling.label() == arm.label
        assert any(", " in level for level in spelling.levels), "the cut prompt keeps its comma, as data"
        assert "=" not in arm.label, arm.label
    named = reader_arm_labels(bundle, input_arm_tags(bundle))
    page = serialize_report(campaign_report(host, _PROMPTS_CAMPAIGN, TOYHOST_SCOPE), "markdown")
    line = _analysed(host, _PROMPTS_CAMPAIGN)
    assert line.report is not None
    for arm in derived:
        name = named[arm.variant_key]
        assert "=" not in name and "chunk_tokens" not in name, name
        assert _bare(name) in page, name
    for printed in _lines(line):
        assert "chunk_tokens" not in printed and "instructions=" not in printed, printed


#: Two prompts the host displays alike (they part only past where its display cuts them), each at one chunk size, and a
#: third arm at another: the first two are spelled from the same two levers alike, told apart by their digests alone.
_SHARED_HEAD = "Read every invoice line and quote, for each total, the words the vendor printed beside it"
_TOLD_BY_DIGEST: tuple[tuple[str, int, str], ...] = (
    ("today", 512, f"{_SHARED_HEAD}, today"),
    ("always", 512, f"{_SHARED_HEAD}, always"),
    ("small", 256, f"{_SHARED_HEAD}, today"),
)


def _levered_host(arms: tuple[tuple[str, int, str], ...]) -> EvalHost:
    """A host holding a campaign whose arms differ by their chunk size and extractor instructions, named by neither."""
    profile = toyhost_profile()
    host = toyhost_host(profile=profile)
    runs = [_batch(label, instructions, chunk_tokens=chunk) for label, chunk, instructions in arms]
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
            created_by="test:fixture",
        )
    )
    return host


def test_two_arms_spelled_alike_each_print_their_own_digest() -> None:
    """Two arms spelled from the same levers at levels a name cuts alike are told apart only by the digest their
    spelling holds apart from their levels (``ArmSpelling.digest``): each prints its own, never the other's."""
    host = _levered_host(_TOLD_BY_DIGEST)
    bundle = assemble_campaign_bundle(host, _PROMPTS_CAMPAIGN, TOYHOST_SCOPE)
    printed = printed_arm_labels(bundle)

    alike = [arm for arm in bundle.arms if bundle.arm_spellings[arm.variant_key].digest]
    assert len(alike) == 2, [arm.label for arm in bundle.arms]
    first, second = (bundle.arm_spellings[arm.variant_key] for arm in alike)
    assert first.levers == second.levers and first.levels == second.levels, "spelled alike, level for level"
    assert first.digest != second.digest
    for arm in alike:
        spelling = bundle.arm_spellings[arm.variant_key]
        told = f" (arm {arm.variant_key[:12]})"
        assert (spelling.digest, spelling.suffix()) == (arm.variant_key[:12], told)
        assert arm.label.endswith(told) and printed[arm.variant_key].endswith(told), (arm.label, printed)
        assert not any(level.endswith(told) for level in spelling.levels), "held apart from its levels"


#: Two prompts that keep a ``, `` once a name cuts them, at one chunk size: arms named by that one lever.
_ONE_LEVER_COMMA_ARMS: tuple[tuple[str, int, str], ...] = (
    ("brief", 256, "You are a support agent, answer in two lines and quote the vendor's own words for each total"),
    ("full", 256, "You are a support agent, answer in full and quote the vendor's own words for each total today"),
)


def test_a_derived_arm_named_by_one_cut_level_holding_a_comma_reads_by_its_lever_s_name() -> None:
    """An arm named by one level was read back off its ``axis=level`` name split on ``, ``: a cut prompt that kept a
    comma left the split unread, and the frozen label carried the lever id and ``=`` on every page. It is named by its
    level as data now, holds no spelling (one lever names it), and every surface prints it by its lever's name, the
    text reader reading the name whole before it splits it."""
    host = _levered_host(_ONE_LEVER_COMMA_ARMS)
    bundle = assemble_campaign_bundle(host, _PROMPTS_CAMPAIGN, TOYHOST_SCOPE)
    levers = {lever for aspect in bundle.aspects for entry in aspect.variant_index for lever in entry.named_levers}

    derived = [arm for arm in bundle.arms if arm.role == "candidate"]
    assert len(derived) == 2
    for arm in derived:
        assert "=" not in arm.label and not any(lever in arm.label for lever in levers), arm.label
        assert ", " in arm.label, "the label is the cut prompt, its comma kept"
        assert arm.variant_key not in bundle.arm_spellings, "one lever names it: no spelling"
    named = reader_arm_labels(bundle, input_arm_tags(bundle))
    page = serialize_report(campaign_report(host, _PROMPTS_CAMPAIGN, TOYHOST_SCOPE), "markdown")
    line = _analysed(host, _PROMPTS_CAMPAIGN)
    assert line.report is not None
    listed = " ".join(_lines(line))
    for arm in derived:
        name = named[arm.variant_key]
        assert _bare(name) == "Extractor instructions", name
        assert name in page and name in listed, (name, listed)
    for printed in _lines(line):
        assert "=" not in printed and not any(lever in printed for lever in levers), printed


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
    # The code page's tags, as the service tags its arms: the campaign's declared arms first.
    code_tags = campaign_arm_tags(
        bundle.aspects,
        bundle.readings.control,
        arms=[arm.variant_key for arm in bundle.arms],
        label_of=bundle.label_of,
    )
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
        assert render(
            f"{{arm:{held.id}}}", {arm.id: arm.name for arm in doc.values()}, forms=PageForms.of(built)
        ) == written_arm(held.name, held.tag)
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
    written = render(
        " vs ".join(f"{{arm:{arm.id}}}" for arm in built.arms),
        {arm.id: arm.name for arm in arms},
        forms=PageForms.of(built),
    )

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
    assert line.report.headline == (
        "The evidence cannot tell Extractor instructions or read twice apart from plain on any reading the question "
        "asks."
    )
    assert "Adopting Extractor instructions overrides the bar it misses on extract_invoice_fields." in (
        line.report.caveats
    ), "the arm no one labelled reads one name in the headline and its caveats"


def test_a_cut_level_two_settings_share_keeps_its_text_rather_than_take_one_lever_s_name() -> None:
    """Two settings whose long values cut to one text both spell the name: the first matched lever named it, which
    could print the wrong lever. Which one the name means cannot be told, so it keeps its level's text."""
    shown = _PROMPTS[0][:59] + ELISION
    cut = _FROZEN_LABELS[0]
    text = frozenset({"system_prompt", "tool_prompt"})
    levels = {"system_prompt": shown, "tool_prompt": shown}
    assert reader_arm_name(cut, levels, text=text) == cut
    assert reader_arm_name(cut, {"system_prompt": shown}, text=text) == "System prompt", "one alone is named"


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
