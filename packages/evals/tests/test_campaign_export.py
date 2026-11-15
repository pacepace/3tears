"""A campaign's closure exported (``ops.campaign_export``), and the ``campaign_export`` action that pages it.

Over the toy host's corpus campaign, its store held in hand so a test writes what the closure must reach or skip:

- **every document reachable from the campaign appears exactly once**, in ``(doc_type, id)`` order, verbatim, and the
  manifest counts every type the engine stores;
- **a sibling campaign in the same scope contributes nothing but the insights on the campaign's subject**;
- **what reasoned on the campaign is named** (an analysis, a report document never by its headline, an insight), each
  name cut at a word boundary;
- **a recorded judge identity is read by the package's reader**, over Discodon's table of identities, verdict for
  verdict and refusal for refusal; a configured judge reaches its config and the config's dim, a deleted one its
  tombstone, a built-in one is listed;
- **a run's judge configs are read as the config set it committed to**: the bare config id a launch records, never a
  result's ``<config id>@<sha12>``, so a campaign launched with a configured judge exports (production's refusal of
  2026-11-08);
- **every type the engine stores is reached by the closure or declared unreachable**, read from the module's source;
- **the profile's digest is the package's**, the one the courier's committed case records;
- **pages joined are the single call's export**, through the operation and through the action.
"""

from __future__ import annotations

import ast
import dataclasses
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

import threetears.evals.ops.campaign_closure as campaign_closure_module
from packages.evals.tests.fixtures.writer_cases.export_held_out_cases import held_out_path
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import (
    CALLER,
    TOYHOST_SCOPE,
    generate,
    judged_as_launched,
    ops_fixture,
    settled,
)
from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis import host_declarations_digest
from threetears.evals.harness import ReasonedOn, WriterCaseFile
from threetears.evals.kernel import EvalInsight, NotFoundError, UnreadableRecordError, ValidationFailedError
from threetears.evals.ops import CampaignExport, OpsHost, UnresolvedLink, campaign_export
from threetears.evals.schema import (
    BUILTIN_JUDGE_CONFIG_PREFIX,
    DOC_TYPE_FACTS,
    CatalogRubricDim,
    JudgeConfig,
    JudgeConfigTombstone,
    RubricDim,
    judge_config_of,
)
from threetears.evals.storage import InMemoryDocumentStore

SCOPE = TOYHOST_SCOPE

#: Any factory name: the closure reads it as text and checks only its form.
BUILT_BY = "packages.evals.tests.fixtures.toyhost.profile:toyhost_profile"


class Toy:
    """The toy host's operations host over a store this test holds."""

    def __init__(self, *, generation: bool = False, store: InMemoryDocumentStore | None = None) -> None:
        self.store = store if store is not None else InMemoryDocumentStore()
        self.fixture = ops_fixture(generation=generation, store=self.store)
        self.host: OpsHost = self.fixture.host
        self.campaign_id = self.fixture.campaign.id

    def export(self, campaign_id: str | None = None, **given: Any) -> CampaignExport:
        return campaign_export(
            self.host.eval_host, SCOPE, campaign_id or self.campaign_id, **{"built_by": BUILT_BY, **given}
        )

    def documents(self, doc_type: str, **field_eq: Any) -> list[dict[str, Any]]:
        return self.store.by_doc_type(doc_type, SCOPE, **field_eq)

    def campaign(self) -> dict[str, Any]:
        (campaign,) = self.store.get_many("eval_campaign", [self.campaign_id], SCOPE)
        return campaign


def _keys(documents: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(document["doc_type"], document["id"]) for document in documents]


# --- the closure -----------------------------------------------------------------------------------------------------


def test_every_document_reachable_from_the_campaign_appears_exactly_once() -> None:
    toy = Toy()
    exported = toy.export()
    observed = _keys(exported.documents)

    # The closure derived by hand from the fixture's own links: the campaign, its runs, their results.
    campaign = toy.campaign()
    run_ids = [run_id for aspect in campaign["aspects"] for run_id in aspect["run_ids"]]
    expected = {("eval_campaign", toy.campaign_id)} | {("eval_run", run_id) for run_id in run_ids}
    for run_id in run_ids:
        expected |= {("eval_result", result["id"]) for result in toy.documents("eval_result", eval_run_id=run_id)}
    assert len(expected) > 40 and Counter(doc_type for doc_type, _ in expected)["eval_run"] == 2
    assert set(observed) == expected
    assert len(observed) == len(set(observed)), "a document appears twice"
    assert observed == sorted(observed), "documents are in (doc_type, id) order"
    # Verbatim: the exported run is the stored run, whole.
    (stored_run,) = toy.store.get_many("eval_run", [run_ids[0]], SCOPE)
    assert stored_run in exported.documents
    # The toy's template is in the scope and no run names it.
    assert not any(doc_type == "eval_template" for doc_type, _ in observed)

    manifest = exported.manifest
    assert set(manifest.doc_type_counts) == set(DOC_TYPE_FACTS), "every type the engine stores is counted"
    assert manifest.doc_type_counts == {
        doc_type: Counter(doc_type for doc_type, _ in observed).get(doc_type, 0) for doc_type in DOC_TYPE_FACTS
    }
    assert manifest.not_reachable == ["eval_action_job", "rubric_dim_tombstone"]
    assert all(
        any(rule.startswith(f"{doc_type}: ") for rule in manifest.closure_rules) for doc_type in manifest.not_reachable
    )
    # The toy stores no test case: each case a run names is unresolved, by id.
    case_ids = {case_id for run in toy.documents("eval_run") for case_id in run["test_case_ids"]}
    assert case_ids and set(manifest.unresolved) == {
        UnresolvedLink(doc_type="eval_test_case", id=case_id) for case_id in case_ids
    }
    assert manifest.builtin_judge_config_ids == [] and manifest.reasoned_on == []
    assert (manifest.campaign_id, manifest.scope_id, manifest.subject_id) == (
        toy.campaign_id,
        SCOPE,
        campaign["subject_id"],
    )
    assert exported.page is None
    assert toy.export() == exported, "two exports of one closure are equal: nothing in one records when it was made"


def _sibling(toy: Toy, *, tag: str, subject_id: str) -> tuple[str, list[str], list[str]]:
    """A second campaign in the toy's scope over copies of its runs, results and traces, on ``subject_id``."""
    campaign = toy.campaign()
    copies: dict[str, str] = {}
    trace_ids: list[str] = []
    for run in toy.documents("eval_run"):
        copies[run["id"]] = f"{tag}-{run['id']}"
        toy.store.upsert({**run, "id": copies[run["id"]]})
        for result in toy.documents("eval_result", eval_run_id=run["id"]):
            result_id = f"{tag}-{result['id']}"
            toy.store.upsert({**result, "id": result_id, "eval_run_id": copies[run["id"]]})
            trace_ids.append(f"{result_id}:trace")
            toy.store.upsert(
                {"doc_type": "eval_trace", "id": trace_ids[-1], "scope_id": SCOPE, "eval_run_id": copies[run["id"]]}
            )
    aspects = [
        {**aspect, "run_ids": [copies[run_id] for run_id in aspect["run_ids"]]} for aspect in campaign["aspects"]
    ]
    sibling_id = f"{tag}-campaign"
    toy.store.upsert({**campaign, "id": sibling_id, "subject_id": subject_id, "aspects": aspects})
    return sibling_id, list(copies.values()), trace_ids


def _insight(subject_id: str, statement: str, source_campaign_id: str) -> dict[str, Any]:
    return EvalInsight(
        scope_id=SCOPE,
        subject_id=subject_id,
        subject_kind="callable",
        statement=statement,
        confidence="medium",
        invalidation_trigger="never: a test fixture",
        source_campaign_id=source_campaign_id,
    ).to_dict()


def test_a_sibling_campaign_in_the_scope_contributes_only_the_insights_on_the_campaigns_subject() -> None:
    toy = Toy()
    subject = toy.campaign()["subject_id"]
    other, other_runs, other_traces = _sibling(toy, tag="other", subject_id="arm-other")
    same, same_runs, same_traces = _sibling(toy, tag="same", subject_id=subject)
    on_other = _insight("arm-other", "the other subject answers differently", other)
    on_same_by_sibling = _insight(subject, "the exact arm holds under the sibling's runs too", same)
    on_same_by_toy = _insight(subject, "the exact arm passes every case", toy.campaign_id)
    for insight in (on_other, on_same_by_sibling, on_same_by_toy):
        toy.store.upsert(insight)

    exported = toy.export()
    observed = set(_keys(exported.documents))

    assert ("eval_campaign", other) not in observed and ("eval_campaign", same) not in observed
    for run_id in [*other_runs, *same_runs]:
        results = toy.documents("eval_result", eval_run_id=run_id)
        assert results, "the sibling run holds results: the population the exclusion is tested on"
        assert ("eval_run", run_id) not in observed
        assert not {("eval_result", result["id"]) for result in results} & observed
    assert not {("eval_trace", trace_id) for trace_id in [*other_traces, *same_traces]} & observed
    assert (
        exported.manifest.doc_type_counts["eval_run"] == 2 and exported.manifest.doc_type_counts["eval_campaign"] == 1
    )

    # Insights follow the SUBJECT: both on the campaign's, whichever campaign minted them; none on the other.
    assert {key for key in observed if key[0] == "eval_insight"} == {
        ("eval_insight", on_same_by_sibling["id"]),
        ("eval_insight", on_same_by_toy["id"]),
    }
    assert sorted(
        (named.id, named.name) for named in exported.manifest.reasoned_on if named.doc_type == "eval_insight"
    ) == sorted((insight["id"], insight["statement"]) for insight in (on_same_by_sibling, on_same_by_toy))
    assert any("SUBJECT" in rule for rule in exported.manifest.closure_rules), "the manifest states the insight rule"


async def test_the_manifest_names_what_reasoned_on_the_campaign() -> None:
    """A report the package's writer stored (its model scripted) is reached whole: the analysis, the snapshot it froze,
    the document, its sessions, and the records of the generation."""
    toy = off_loop(Toy, generation=True)
    (job,) = (await generate(toy.host, toy.campaign_id, SCOPE)).jobs
    await settled(toy.host, job.job_id)

    exported = off_loop(toy.export)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for document in exported.documents:
        by_type.setdefault(document["doc_type"], []).append(document)
    (analysis,) = by_type["eval_analysis"]
    (report_doc,) = by_type["eval_report_doc"]
    (campaign,) = by_type["eval_campaign"]
    assert campaign["name"], "the campaign is named, so the report's name carries more than its id"
    (analysed,) = [entry for entry in exported.manifest.reasoned_on if entry.doc_type == "eval_analysis"]
    (reported,) = [entry for entry in exported.manifest.reasoned_on if entry.doc_type == "eval_report_doc"]
    minted = {document["id"] for document in by_type.get("eval_insight", [])}
    assert {entry.id for entry in exported.manifest.reasoned_on if entry.doc_type == "eval_insight"} == minted
    assert (analysed.doc_type, analysed.id) == ("eval_analysis", analysis["id"])
    assert analysed.name is not None and analysed.name.startswith(f"analysis {analysis['created_at']}")
    assert len(analysed.name) <= 80
    assert reported == ReasonedOn(
        doc_type="eval_report_doc", id=report_doc["id"], name=f"report {report_doc['id']}: {campaign['name']}"
    )
    for doc_type in ("eval_report_snapshot", "eval_report_session", "eval_analysis_attempt"):
        assert exported.manifest.doc_type_counts[doc_type] >= 1, doc_type
    assert analysis["report_doc_id"] == report_doc["id"]
    assert report_doc["snapshot_id"] == by_type["eval_report_snapshot"][0]["id"]


#: A campaign name that, after ``report <36-character id>: ``, runs past the 80-character cut, which lands on the word
#: after its comma.
LONG_NAME = "Character v2 apparatus 11-07 dev, over every persona the dev scope holds and a few it lacks"


async def test_a_report_document_is_named_by_its_campaign_never_by_its_headline_and_cut_at_a_word() -> None:
    """A stored headline holds the writer's figure keys (``{cases|words}``), which no page prints and no note repeats;
    the name is the campaign's with the document's id, cut at a word boundary, a trailing comma stripped."""
    toy = off_loop(Toy, generation=True)
    off_loop(toy.store.merge_fields, toy.campaign_id, SCOPE, {"name": LONG_NAME})
    (job,) = (await generate(toy.host, toy.campaign_id, SCOPE)).jobs
    await settled(toy.host, job.job_id)
    exported = off_loop(toy.export)
    (report_doc,) = [document for document in exported.documents if document["doc_type"] == "eval_report_doc"]
    (named,) = [entry for entry in exported.manifest.reasoned_on if entry.doc_type == "eval_report_doc"]

    full = f"report {report_doc['id']}: {LONG_NAME}"
    whole_words = full[:79].rsplit(" ", 1)[0]
    assert len(full) > 80 and whole_words.endswith(","), "the fixture reaches the cut, just after a comma"
    assert named.name == whole_words.rstrip(",") + "…" and len(named.name) <= 80
    assert "{" not in (named.name or "")
    assert any("never by its headline" in rule for rule in exported.manifest.closure_rules)


def test_a_long_insight_is_named_by_its_statement_cut_at_a_word_and_a_blank_one_by_nothing() -> None:
    toy = Toy()
    subject = toy.campaign()["subject_id"]
    long = _insight(subject, "insight " * 20, toy.campaign_id)
    blank = {**_insight(subject, "placeholder", toy.campaign_id), "statement": "   "}
    toy.store.upsert(long)
    toy.store.upsert(blank)
    named = {entry.id: entry.name for entry in toy.export().manifest.reasoned_on}
    assert named[long["id"]] == ("insight " * 9).strip() + "…", "nine whole words fit before the 80th character"
    assert named[blank["id"]] is None, "a name is a statement or nothing, never an empty string"


def test_a_campaign_not_in_the_scope_is_refused_naming_it() -> None:
    toy = Toy()
    with pytest.raises(NotFoundError) as refused:
        toy.export("no-such-campaign")
    assert refused.value.status_code == 404 and refused.value.message == "campaign 'no-such-campaign' not found"
    assert refused.value.remedy
    with pytest.raises(NotFoundError):
        campaign_export(toy.host.eval_host, "another-scope", toy.campaign_id, built_by=None)


# --- judge identities ------------------------------------------------------------------------------------------------

#: Discodon's table of recorded judge identities, each with the verdict its reader gave and the package's gives: a
#: configured judge's config, a built-in prompt (well- and ill-formed: the reader checks the prefix), a dim no judge
#: asked, and the refusals: a bare config id, an empty string and three malformed ``@`` shapes.
JUDGE_IDENTITIES: tuple[tuple[str | None, str], ...] = (
    ("01a15a15-f896-70a5-946f-b987ea7c2276@7f5d3e70b47a", "config"),
    (f"{BUILTIN_JUDGE_CONFIG_PREFIX}576896bef534", "builtin"),
    (f"{BUILTIN_JUDGE_CONFIG_PREFIX}not-twelve-hex", "builtin"),
    (None, "none"),
    ("bare-config-id", "refused"),
    ("", "refused"),
    ("@7f5d3e70b47a", "refused"),
    ("01a15a15-f896-70a5-946f-b987ea7c2276@", "refused"),
    ("01a15a15-f896-70a5-946f-b987ea7c2276@7F5D3E70B47A", "refused"),
)


def _first_result(toy: Toy) -> dict[str, Any]:
    run_id = toy.campaign()["aspects"][0]["run_ids"][0]
    result = min(toy.documents("eval_result", eval_run_id=run_id), key=lambda document: document["id"])
    assert result["judge_config_ids"] == {}, "the toy's results record no judge, so what is read is what is merged"
    return result


@pytest.mark.parametrize(("recorded", "verdict"), JUDGE_IDENTITIES, ids=lambda value: repr(value))
def test_a_recorded_judge_identity_is_read_as_the_packages_reader_reads_it(recorded: str | None, verdict: str) -> None:
    """The table's verdict is the package reader's, and the export's: a configured judge's config is followed (here
    unresolved, since none is stored), a built-in one listed, an unasked dim skipped, and what the reader refuses
    refuses the export, naming the value and the document that records it."""
    try:
        read: str | None | type[ValueError] = judge_config_of(recorded)
    except ValueError:
        read = ValueError
    assert {
        "config": isinstance(read, str),
        "builtin": read is None,
        "none": read is None,
        "refused": read is ValueError,
    }[verdict], (recorded, read)

    toy = Toy()
    result = _first_result(toy)
    assert toy.store.merge_fields(result["id"], SCOPE, {"judge_config_ids": {"toy.dim": recorded}})
    if verdict == "refused":
        with pytest.raises(UnreadableRecordError) as refused:
            toy.export()
        assert repr(recorded) in refused.value.message and result["id"] in refused.value.message
        assert "an eval_result records <config id>@<sha12> or builtin:<sha12>" in refused.value.message
        return
    manifest = toy.export().manifest
    assert manifest.builtin_judge_config_ids == ([recorded] if verdict == "builtin" else [])
    judges = [link for link in manifest.unresolved if link.doc_type == "judge_config"]
    assert judges == ([UnresolvedLink(doc_type="judge_config", id=str(read))] if verdict == "config" else [])


#: What a run records per dim (``EvalRun.judge_config_ids``), each with the export's verdict: the bare config id a
#: launch commits to, a built-in prompt's identity, a dim no judge asked, and the refusals: a result's identity, which
#: names the prompt sent rather than the config committed to, and an empty string, which names no config.
RUN_JUDGE_IDENTITIES: tuple[tuple[str | None, str], ...] = (
    ("01a15a15-f896-70a5-946f-b987ea7c2276", "config"),
    (f"{BUILTIN_JUDGE_CONFIG_PREFIX}576896bef534", "builtin"),
    (None, "none"),
    ("01a15a15-f896-70a5-946f-b987ea7c2276@7f5d3e70b47a", "refused"),
    ("", "refused"),
)


@pytest.mark.parametrize(("recorded", "verdict"), RUN_JUDGE_IDENTITIES, ids=lambda value: repr(value))
def test_a_run_s_recorded_judge_config_is_read_as_the_config_set_it_committed_to(
    recorded: str | None, verdict: str
) -> None:
    """A run records the bare config id it committed to (``EvalRun.judge_config_ids``), which the export follows (here
    unresolved, since none is stored); a built-in identity is listed and an unasked dim skipped; a result's form, or
    no id at all, refuses, naming the run and the form a run records."""
    toy = Toy()
    run_id = toy.campaign()["aspects"][0]["run_ids"][0]
    assert toy.store.merge_fields(run_id, SCOPE, {"judge_config_ids": {"toy.dim": recorded}})
    if verdict == "refused":
        with pytest.raises(UnreadableRecordError) as refused:
            toy.export()
        message = refused.value.message
        assert repr(recorded) in message and f"the eval_run {run_id!r}" in message
        assert "an eval_run records the bare config id it committed to" in message and "builtin" not in message
        return
    manifest = toy.export().manifest
    assert manifest.builtin_judge_config_ids == ([recorded] if verdict == "builtin" else [])
    judges = [link for link in manifest.unresolved if link.doc_type == "judge_config"]
    assert judges == ([UnresolvedLink(doc_type="judge_config", id=str(recorded))] if verdict == "config" else [])


def test_a_campaign_launched_with_a_configured_judge_exports_reaching_its_config_and_dim() -> None:
    """Production's shape (2026-11-08): each run records the bare config id it committed to, each of its results the
    configured judge's identity, ``<config id>@<sha12>``, both as the launch's own judge build writes them. Every such
    campaign's export was refused for the run's bare id; now the run's is read as the config it names, so the export
    reaches the config and its dim and lists the built-in judge beside it."""
    toy = Toy()
    judges = judged_as_launched(toy.fixture, toy.store)
    config_id = judges.config.id
    assert judges.run == {"toy.dim": config_id}, "a run records the bare config id"
    assert judge_config_of(judges.result["toy.dim"]) == config_id != judges.result["toy.dim"]

    exported = toy.export()
    observed = set(_keys(exported.documents))
    assert ("judge_config", config_id) in observed and ("rubric_dim", judges.dim.id) in observed
    assert exported.manifest.builtin_judge_config_ids == [judges.result["toy.plain"]]
    assert not [link for link in exported.manifest.unresolved if link.doc_type in {"judge_config", "rubric_dim"}]
    runs = [document for document in exported.documents if document["doc_type"] == "eval_run"]
    assert len(runs) == 2 and all(run["judge_config_ids"] == judges.run for run in runs), "runs export as stored"


def test_the_judge_identity_table_reaches_every_verdict() -> None:
    """The table is only evidence if it reaches a configured verdict, a built-in one, an unasked dim and a refusal."""
    verdicts = Counter(verdict for _, verdict in JUDGE_IDENTITIES)
    assert verdicts["config"] >= 1 and verdicts["builtin"] >= 2 and verdicts["none"] >= 1 and verdicts["refused"] >= 4


def test_judged_results_reach_their_judge_configs_and_dims_and_name_the_builtin_judges() -> None:
    """A configured judge reaches its stored config and the config's dim by name, a deleted one its tombstone and an
    unresolved dim name, a built-in one is listed and reaches nothing."""
    toy = Toy()
    storage = toy.host.eval_host.storage
    config = JudgeConfig(scope_id=SCOPE, name="toy", rubric_dim_id="toy.dim", prompt_template="Judge {artifact}.")
    dim = CatalogRubricDim(
        scope_id=SCOPE, key="toy_dim", dim=RubricDim(name="toy.dim", description="A toy dim.", scale="pass_fail")
    )
    stray = CatalogRubricDim(
        scope_id=SCOPE, key="stray_dim", dim=RubricDim(name="stray.dim", description="No config.", scale="pass_fail")
    )
    tombstone = JudgeConfigTombstone(
        scope_id=SCOPE, rubric_dim_id="toy.gone", name="gone", deleted_config_id="cfg-gone"
    )
    storage.save_judge_config(config)
    storage.save_rubric_dim(dim)
    storage.save_rubric_dim(stray)
    storage.save_judge_config_tombstone(tombstone)
    result = _first_result(toy)
    builtin = f"{BUILTIN_JUDGE_CONFIG_PREFIX}576896bef534"
    judged = {"toy.dim": f"{config.id}@{'0' * 12}", "toy.gone": f"cfg-gone@{'1' * 12}", "toy.builtin": builtin}
    assert toy.store.merge_fields(result["id"], SCOPE, {"judge_config_ids": judged})

    exported = toy.export()
    observed = set(_keys(exported.documents))
    assert exported.manifest.builtin_judge_config_ids == [builtin]
    assert ("judge_config", config.id) in observed and ("rubric_dim", dim.id) in observed
    assert ("rubric_dim", stray.id) not in observed, "a dim no config names is not in the closure"
    assert ("judge_config_tombstone", tombstone.id) in observed
    assert [link for link in exported.manifest.unresolved if link.doc_type != "eval_test_case"] == [
        UnresolvedLink(doc_type="rubric_dim", id="toy.gone", field="dim.name")
    ]
    (stored,) = [document for document in exported.documents if document["id"] == result["id"]]
    assert stored["judge_config_ids"] == judged, "the result is exported as stored"


#: The name the module's closure is bound to wherever it reads the store; the scan below reads its calls.
_CLOSURE = "closure"


def _types_the_closure_reads() -> set[str]:
    """The document types the module passes, as a literal first argument, to a method of its closure."""
    tree = ast.parse(Path(campaign_closure_module.__file__).read_text(encoding="utf-8"))
    return {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == _CLOSURE
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }


def test_every_type_the_engine_stores_is_reached_by_the_closure_or_declared_unreachable() -> None:
    """A type the engine registers later that the closure neither reads nor declares unreachable names itself here,
    rather than counting 0 in every manifest."""
    reached = _types_the_closure_reads()
    unreachable = set(Toy().export().manifest.not_reachable)
    assert {"eval_campaign", "eval_run", "eval_trace", "rubric_dim", "eval_launch"} <= reached and len(reached) > 20
    assert not reached & unreachable, f"declared unreachable yet read: {sorted(reached & unreachable)}"
    registered = set(DOC_TYPE_FACTS)
    assert not registered - reached - unreachable, (
        f"neither reached nor declared: {sorted(registered - reached - unreachable)}"
    )
    assert not (reached | unreachable) - registered, f"not registered: {sorted((reached | unreachable) - registered)}"


# --- the profile -----------------------------------------------------------------------------------------------------


def test_the_profile_declaration_carries_the_packages_digest_and_the_factory_named() -> None:
    """The oracle is the courier's committed case: the digest its bundle recorded is the one its export declares."""
    from packages.evals.tests.fixtures.courierhost import COURIER_SCOPE, courier_profile
    from packages.evals.tests.hot_path_support import campaign_id_of, courierhost_from

    capture = Path(__file__).resolve().parent / "fixtures" / "hot_path_oracle" / "inputs" / "courierhost.json"
    documents = json.loads(capture.read_text())
    courier = campaign_export(
        courierhost_from(documents), COURIER_SCOPE, campaign_id_of(documents), built_by="courier.evals:profile"
    ).profile
    bundle = WriterCaseFile.read(held_out_path("courierhost")).snapshot.bundle
    assert bundle is not None
    (recorded,) = {aspect["host_declarations_digest"] for aspect in bundle["aspects"]}
    assert courier.host_declarations_digest == recorded == host_declarations_digest(courier_profile())
    assert courier.built_by == "courier.evals:profile" and courier.host_id == "courier"
    assert {"built_by", "host_declarations_digest"} <= set(json.loads(courier.model_dump_json()))

    toy = Toy().export(built_by=None).profile
    assert toy.built_by is None and toy.host_declarations_digest != courier.host_declarations_digest
    assert courier.world is not None and toy.world is not None
    assert len(courier.measures) == len(courier_profile().measures.names) > 0
    declared = {sweepable.name: sweepable.reader_name for sweepable in courier_profile().sweepables.declarations}
    assert {sweepable.name: sweepable.reader_name for sweepable in courier.sweepables} == declared, (
        "each lever's reader name is recorded as the host declared it"
    )
    assert any(name is not None for name in declared.values())


@pytest.mark.parametrize("built_by", ["courier 0.92", "module:", ":callable", " :x"])
def test_a_factory_not_named_module_callable_is_refused(built_by: str) -> None:
    """The mint reads any other text as no factory, so an export naming one wrongly is refused before it is written."""
    with pytest.raises(ValidationFailedError, match="module:callable"):
        Toy().export(built_by=built_by)


# --- pages -----------------------------------------------------------------------------------------------------------

#: Bytes merged into each of several results' kind payload (a field the model declares, as a real result's heavy fields
#: are), so the toy's closure spans several pages.
PADDING = 300_000


def _padded(toy: Toy, count: int = 8) -> None:
    for result in sorted(toy.documents("eval_result"), key=lambda document: document["id"])[:count]:
        assert toy.store.merge_fields(result["id"], SCOPE, {"kind_payload": {"padding": "x" * PADDING}})


def _pages(toy: Toy) -> list[CampaignExport]:
    pages = [toy.export(paged=True)]
    while (page := pages[-1].page) is not None and page.next_after_doc_type is not None:
        pages.append(
            toy.export(
                paged=True, after_doc_type=page.next_after_doc_type, after_document_id=page.next_after_document_id
            )
        )
    return pages


def test_pages_joined_are_the_single_calls_export() -> None:
    toy = Toy()
    _padded(toy)
    whole = toy.export()
    pages = _pages(toy)

    assert len(pages) >= 3, "the padding spans several pages"
    joined = [document for page in pages for document in page.documents]
    assert pages[0].model_copy(update={"documents": joined, "page": None}) == whole
    assert all((page.profile, page.manifest) == (whole.profile, whole.manifest) for page in pages)
    before = 0
    for page in pages:
        assert page.page is not None
        assert (page.page.documents_before, page.page.documents_after) == (
            before,
            len(whole.documents) - before - len(page.documents),
        )
        sizes = [len(json.dumps(document, separators=(",", ":"))) for document in page.documents]
        assert len(sizes) == 1 or sum(sizes) <= page.page.byte_ceiling
        before += len(page.documents)
    assert pages[-1].page is not None and pages[-1].page.next_after_doc_type is None
    with pytest.raises(ValueError, match="one page of a paged export"):
        pages[0].files()


def test_a_closure_inside_one_page_is_exported_whole_when_paged() -> None:
    toy = Toy()
    assert toy.export(paged=True) == toy.export()


def test_a_page_after_a_document_deleted_since_starts_at_the_next_one() -> None:
    """The cursor is a place in the export's order, not a document that must still be there."""
    toy = Toy()
    _padded(toy)
    first = toy.export(paged=True)
    assert first.page is not None and first.page.next_after_document_id is not None
    after = (first.page.next_after_doc_type, first.page.next_after_document_id)
    assert toy.store.delete(after[1], SCOPE)
    second = toy.export(paged=True, after_doc_type=after[0], after_document_id=after[1])
    whole = _keys(toy.export().documents)
    assert _keys(second.documents)[0] == next(key for key in whole if key > after)


@pytest.mark.parametrize(
    ("given", "said"),
    [
        ({"after_doc_type": "eval_result", "after_document_id": "r"}, "this export is whole"),
        ({"paged": True, "after_doc_type": "eval_result"}, "its type and its id together"),
        ({"paged": True, "after_document_id": "r"}, "its type and its id together"),
        ({"paged": True, "after_doc_type": "eval_resultz", "after_document_id": "r"}, "no document type the engine"),
    ],
)
def test_a_cursor_the_export_would_misread_is_refused(given: dict[str, Any], said: str) -> None:
    with pytest.raises(ValidationFailedError, match=said):
        Toy().export(**given)


# --- the action ------------------------------------------------------------------------------------------------------


async def test_the_action_pages_the_closure_and_its_pages_join_to_the_export() -> None:
    toy = off_loop(Toy)
    off_loop(_padded, toy)
    (evals, _admin) = eval_catalogue().mount_all(standard_tools("evals"))
    call: dict[str, Any] = {"action": "campaign_export", "campaign_id": toy.campaign_id, "built_by": BUILT_BY}
    pages: list[CampaignExport] = []
    while True:
        outcome = await evals.call(call, host=toy.host, caller=CALLER)
        assert not outcome.is_error, outcome.text
        pages.append(CampaignExport.model_validate(outcome.structured))
        page = pages[-1].page
        assert page is not None, "a closure past one page is answered a page at a time"
        assert (
            "Pages are not a snapshot: when a page's manifest closure_keys_digest is not the first page's "
            f"({pages[0].manifest.closure_keys_digest[:12]}…), the campaign moved between calls; read it again from "
            "the first page."
        ) in outcome.text
        if page.next_after_doc_type is None:
            assert "More:" not in outcome.text
            break
        assert (
            f"More: action='campaign_export', campaign_id='{toy.campaign_id}', "
            f"export_after_doc_type='{page.next_after_doc_type}', "
            f"export_after_document_id='{page.next_after_document_id}', built_by='{BUILT_BY}'."
        ) in outcome.text
        call |= {
            "export_after_doc_type": page.next_after_doc_type,
            "export_after_document_id": page.next_after_document_id,
        }

    whole = off_loop(toy.export)
    assert len(pages) >= 3
    joined = [document for page in pages for document in page.documents]
    assert pages[0].model_copy(update={"documents": joined, "page": None}) == whole


async def test_the_mint_hint_names_every_flag_the_mint_requires() -> None:
    toy = off_loop(Toy)
    (evals, _admin) = eval_catalogue().mount_all(standard_tools("evals"))

    outcome = await evals.call(
        {"action": "campaign_export", "campaign_id": toy.campaign_id, "built_by": BUILT_BY},
        host=toy.host,
        caller=CALLER,
    )

    assert not outcome.is_error, outcome.text
    for required in ("--documents", "--profile", "--host", "--campaign", "--name <case>", "--out <file>.json.gz"):
        assert required in outcome.text, required
    assert "Pages are not a snapshot" not in outcome.text, "a closure inside one page is answered whole, unpaged"


async def test_the_action_reads_the_callers_scope_and_refuses_a_campaign_it_does_not_hold() -> None:
    toy = off_loop(Toy)
    (evals, _admin) = eval_catalogue().mount_all(standard_tools("evals"))
    outcome = await evals.call({"action": "campaign_export", "campaign_id": "elsewhere"}, host=toy.host, caller=CALLER)
    assert outcome.is_error and "campaign 'elsewhere' not found" in outcome.text
    whole = await evals.call(
        {"action": "campaign_export", "campaign_id": toy.campaign_id}, host=toy.host, caller=CALLER
    )
    exported = CampaignExport.model_validate(whole.structured)
    assert exported == off_loop(toy.export, built_by=None) and exported.profile.built_by is None
    assert f"{len(exported.documents)} documents of campaign {toy.campaign_id}" in whole.text


def test_a_page_read_after_the_closure_moved_carries_another_marker_where_the_counts_do_not_move() -> None:
    """A result replaced by another between two page calls leaves every count as it was, so only the keys' digest
    tells the reader its pages no longer join to one export."""
    toy = Toy()
    _padded(toy)
    first = toy.export(paged=True)
    assert first.page is not None and first.page.next_after_doc_type is not None
    replaced = max(toy.documents("eval_result"), key=lambda document: document["id"])
    assert toy.store.delete(replaced["id"], SCOPE)
    toy.store.upsert({**replaced, "id": f"{replaced['id']}-again"})

    second = toy.export(
        paged=True,
        after_doc_type=first.page.next_after_doc_type,
        after_document_id=first.page.next_after_document_id,
    )
    assert second.manifest.doc_type_counts == first.manifest.doc_type_counts, "the counts cannot see the swap"
    assert second.manifest.closure_documents == first.manifest.closure_documents
    assert second.manifest.closure_keys_digest != first.manifest.closure_keys_digest
    # Unmoved, a later page carries the first page's marker.
    again = toy.export(paged=True)
    third = toy.export(
        paged=True,
        after_doc_type=again.page.next_after_doc_type if again.page else None,
        after_document_id=again.page.next_after_document_id if again.page else None,
    )
    assert third.manifest.closure_keys_digest == again.manifest.closure_keys_digest
    assert again.manifest.closure_documents == len(toy.export().documents)


async def test_a_campaign_stored_under_another_scope_is_not_the_callers_to_export() -> None:
    toy = off_loop(Toy)
    elsewhere = "another-scope"
    campaign = off_loop(toy.campaign)
    off_loop(toy.store.upsert, {**campaign, "id": "campaign-elsewhere", "scope_id": elsewhere})
    (evals, _admin) = eval_catalogue().mount_all(standard_tools("evals"))
    outcome = await evals.call(
        {"action": "campaign_export", "campaign_id": "campaign-elsewhere"}, host=toy.host, caller=CALLER
    )
    assert outcome.is_error and "campaign 'campaign-elsewhere' not found" in outcome.text
    # Read in its own scope, it is there: the refusal is the scope's, not the id's.
    exported = off_loop(campaign_export, toy.host.eval_host, elsewhere, "campaign-elsewhere", built_by=None)
    assert exported.manifest.scope_id == elsewhere and exported.documents[0]["id"] == "campaign-elsewhere"


# --- the action's ceiling --------------------------------------------------------------------------------------------


def _closure_bytes(exported: CampaignExport) -> int:
    """What the closure's documents weigh, each as canonical JSON: the measure the ceiling counts in."""
    return sum(len(json.dumps(document, sort_keys=True, separators=(",", ":"))) for document in exported.documents)


async def _action_pages(host: OpsHost, campaign_id: str) -> list[Any]:
    """Every page the action answers for ``campaign_id``, following each page's next call; stops at a refusal."""
    (evals, _admin) = eval_catalogue().mount_all(standard_tools("evals"))
    call: dict[str, Any] = {"action": "campaign_export", "campaign_id": campaign_id}
    outcomes = []
    while True:
        outcome = await evals.call(call, host=host, caller=CALLER)
        outcomes.append(outcome)
        if outcome.is_error:
            return outcomes
        page = CampaignExport.model_validate(outcome.structured).page
        if page is None or page.next_after_doc_type is None:
            return outcomes
        call |= {
            "export_after_doc_type": page.next_after_doc_type,
            "export_after_document_id": page.next_after_document_id,
        }


async def test_the_action_refuses_a_closure_past_its_hosts_ceiling_naming_the_size_and_the_way_out() -> None:
    toy = off_loop(Toy)
    off_loop(_padded, toy)
    whole = off_loop(toy.export)
    assert _closure_bytes(whole) > 1_000_000, "the padded closure is past the ceiling set below"
    lowered = dataclasses.replace(toy.host, campaign_export_max_bytes=1_000_000)

    (refused,) = await _action_pages(lowered, toy.campaign_id)
    assert refused.is_error
    assert (
        f"the closure of campaign '{toy.campaign_id}' passed the export ceiling of 1,000,000 bytes, at " in refused.text
    )
    assert "python -m threetears.evals.harness export" in refused.text
    assert "threetears.evals.ops.campaign_export(host, scope_id, campaign_id" in refused.text


#: The most whole documents the module states one read under a ceiling returns.
CHUNK = 16

#: The types the closure scans whole, outside the ceiling's count.
SCANNED = {"rubric_dim", "eval_launch", "eval_judge_profile"}


class RecordingStore(InMemoryDocumentStore):
    """The memory store, recording what each read asked for and what it returned."""

    def __init__(self) -> None:
        super().__init__()
        self.reads: list[tuple[str, str, tuple[str, ...], list[dict[str, Any]]]] = []

    def get_many(
        self,
        doc_type: str,
        doc_ids: Sequence[str],
        scope_id: str,
        *,
        exclude: Sequence[str] = (),
        keep: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        found = super().get_many(doc_type, doc_ids, scope_id, exclude=exclude, keep=keep)
        self.reads.append(("get_many", doc_type, tuple(exclude), found))
        return found

    def by_doc_type(
        self,
        doc_type: str,
        scope_id: str,
        *,
        order_by: str | None = None,
        descending: bool = True,
        limit: int | None = None,
        exclude: Sequence[str] = (),
        field_in: Mapping[str, Sequence[Any]] | None = None,
        **field_eq: Any,
    ) -> list[dict[str, Any]]:
        found = super().by_doc_type(
            doc_type,
            scope_id,
            order_by=order_by,
            descending=descending,
            limit=limit,
            exclude=exclude,
            field_in=field_in,
            **field_eq,
        )
        self.reads.append(("by_doc_type", doc_type, tuple(exclude), found))
        return found

    def whole_reads(self) -> list[list[dict[str, Any]]]:
        """What each read of whole documents returned: every read but the reads of ids and the whole-type scans."""
        return [found for _, doc_type, exclude, found in self.reads if not exclude and doc_type not in SCANNED]


def test_a_refused_export_read_no_more_than_one_chunk_of_whole_documents_at_a_time() -> None:
    """The bound, not just the counter: a closure several times the ceiling is refused, and no read before the refusal
    returned more than a chunk of whole documents, so what the call held is the ceiling and one chunk. A read of ids
    returns ids alone. Mutation checked: a where that reads its documents whole returned a run's 36 results in one read
    and went red here."""
    store = RecordingStore()
    toy = Toy(store=store)
    _padded(toy, count=60)
    whole = toy.export()
    largest = max(len(json.dumps(document, sort_keys=True, separators=(",", ":"))) for document in whole.documents)
    ceiling = 1_000_000
    assert _closure_bytes(whole) > 5 * ceiling, "the closure is several times the ceiling"
    store.reads.clear()

    with pytest.raises(ValidationFailedError) as refused:
        toy.export(max_bytes=ceiling)
    details = refused.value.details
    assert details["max_bytes"] == ceiling and ceiling < details["kept_bytes"] <= ceiling + largest
    assert f"at {details['kept_bytes']:,} bytes after {details['kept_documents']:,} documents" in refused.value.message
    whole_reads = store.whole_reads()
    assert whole_reads and max(len(found) for found in whole_reads) <= CHUNK
    assert any(len(found) == CHUNK for found in whole_reads), "a run's results were read a full chunk at a time"
    id_reads = [found for kind, _, exclude, found in store.reads if exclude]
    assert id_reads and all(set(document) <= {"id"} for found in id_reads for document in found), (
        "a read of ids carries nothing else of a document"
    )
    held = sum(
        len(json.dumps(document, sort_keys=True, separators=(",", ":"))) for found in whole_reads for document in found
    )
    assert held < _closure_bytes(whole) / 2, "the refused call read under half the closure's documents"


def test_a_ceilinged_export_reads_what_an_unbounded_one_reads_over_every_field_read() -> None:
    """Every read by a field runs under a ceiling (the runs pinned to a case set, a cassette corpus and a sweep, a
    report document and a snapshot stored, a judge whose config is gone): the export is the unbounded one, each read of
    ids carries ids alone, and every other read returns at most a chunk."""
    store = RecordingStore()
    toy = Toy(store=store)
    campaign = toy.campaign()
    pins = {
        "case_set": {"name": "toy-set", "version": 1},
        "cassette_corpus_id": "corpus-1",
        "launch_group_id": "group-1",
    }
    for run_id in campaign["aspects"][0]["run_ids"]:
        assert store.merge_fields(run_id, SCOPE, pins)
    stored = [
        {"doc_type": "eval_report_doc", "id": "report-1", "campaign_id": toy.campaign_id},
        {"doc_type": "eval_report_snapshot", "id": "snapshot-1", "campaign_id": toy.campaign_id},
        {"doc_type": "eval_report_session", "id": "session-1", "report_doc_id": "report-1"},
        {"doc_type": "eval_analyst_gap", "id": "gap-1", "snapshot_id": "snapshot-1"},
        {"doc_type": "judge_config_tombstone", "id": "tomb-1", "deleted_config_id": "cfg-gone", "rubric_dim_id": "d"},
        {"doc_type": "case_set", "id": "set-1", "name": "toy-set", "version": 1},
        {"doc_type": "eval_cassette", "id": "cassette-1", "corpus_id": "corpus-1"},
        {"doc_type": "eval_sweep", "id": "sweep-1", "launch_group_id": "group-1"},
    ]
    for document in stored:
        store.upsert({**document, "scope_id": SCOPE})
    assert store.merge_fields(_first_result(toy)["id"], SCOPE, {"judge_config_ids": {"d": f"cfg-gone@{'0' * 12}"}})
    unbounded = toy.export()
    assert {(document["doc_type"], document["id"]) for document in stored} <= set(_keys(unbounded.documents))
    store.reads.clear()

    assert toy.export(max_bytes=10 * _closure_bytes(unbounded)) == unbounded
    read_by_field = {doc_type for kind, doc_type, exclude, _ in store.reads if kind == "by_doc_type" and exclude}
    assert read_by_field >= {document["doc_type"] for document in stored} - {"eval_report_doc"} | {"eval_result"}
    assert not [
        doc_type
        for kind, doc_type, exclude, _ in store.reads
        if kind == "by_doc_type" and not exclude and doc_type not in SCANNED
    ]
    assert all(set(document) <= {"id"} for _, _, exclude, found in store.reads if exclude for document in found)
    assert max(len(found) for found in store.whole_reads()) <= CHUNK


async def test_under_the_ceiling_the_action_pages_exactly_as_before() -> None:
    toy = off_loop(Toy)
    off_loop(_padded, toy)
    whole = off_loop(toy.export, built_by=None)
    at_the_ceiling = dataclasses.replace(toy.host, campaign_export_max_bytes=_closure_bytes(whole))
    for host in (toy.host, at_the_ceiling):
        outcomes = await _action_pages(host, toy.campaign_id)
        assert not any(outcome.is_error for outcome in outcomes), outcomes[-1].text
        pages = [CampaignExport.model_validate(outcome.structured) for outcome in outcomes]
        assert len(pages) >= 3
        joined = [document for page in pages for document in page.documents]
        assert pages[0].model_copy(update={"documents": joined, "page": None}) == whole


def test_the_function_exports_a_closure_past_the_actions_ceiling() -> None:
    """The function holds no ceiling unless one is passed, whatever its host's action holds."""
    toy = Toy()
    _padded(toy)
    host = dataclasses.replace(toy.host, campaign_export_max_bytes=1_000)
    exported = campaign_export(host.eval_host, SCOPE, toy.campaign_id, built_by=None)
    assert _closure_bytes(exported) > host.campaign_export_max_bytes and exported.page is None


async def test_the_host_raises_and_lowers_the_ceiling() -> None:
    toy = off_loop(Toy)
    size = _closure_bytes(off_loop(toy.export))
    assert toy.host.campaign_export_max_bytes == 16_000_000, "the default a host that sets none holds"
    for ceiling, refused in ((size - 1, True), (size, False), (size * 10, False)):
        outcomes = await _action_pages(
            dataclasses.replace(toy.host, campaign_export_max_bytes=ceiling), toy.campaign_id
        )
        assert outcomes[-1].is_error is refused, (ceiling, outcomes[-1].text)
    for ceiling in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="campaign_export_max_bytes must be a positive integer"):
            dataclasses.replace(toy.host, campaign_export_max_bytes=ceiling)


def test_a_reading_host_takes_the_ceiling() -> None:
    """A host serving a reading host (``as_host``, then ``OpsHost.reading``) sets the action's ceiling there."""
    toy = Toy()
    _padded(toy)
    reading = OpsHost.reading(toy.host.eval_host, campaign_export_max_bytes=1_000_000)
    assert reading.campaign_export_max_bytes == 1_000_000
    assert OpsHost.reading(toy.host.eval_host).campaign_export_max_bytes == 16_000_000
    with pytest.raises(ValueError, match="positive integer"):
        OpsHost.reading(toy.host.eval_host, campaign_export_max_bytes=0)


async def test_the_action_on_a_reading_host_holds_the_ceiling_it_was_given() -> None:
    toy = off_loop(Toy)
    off_loop(_padded, toy)
    reading = OpsHost.reading(toy.host.eval_host, campaign_export_max_bytes=1_000_000)
    (refused,) = await _action_pages(reading, toy.campaign_id)
    assert refused.is_error and "export ceiling of 1,000,000 bytes" in refused.text
