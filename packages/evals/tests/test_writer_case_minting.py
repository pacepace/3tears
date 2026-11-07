"""A writer case minted from a host's export (``writer_case_from_export``, ``python -m threetears.evals.harness mint``).

The courier's capture (``hot_path_oracle/inputs/courierhost.json``) is the export: every document of its planner
bake-off, as stored. Minted through the door with the held-out export's metadata and freeze time, it is the case that
export mints (``export_held_out_cases.host_case``), byte for byte, and the committed ``held_out/courierhost.json.gz``
under the house rule that file is held to (``test_held_out_cases.py``: the same snapshot id, every figure the same
to twelve places, without the build's ``package_version``). The command line builds the host's three files from the
capture (the documents, the courier's profile declaration naming its factory and its declarations' digest, the
manifest of what reasoned on the campaign) and writes that case. The notes withheld are the shipped skill's.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.fixtures.writer_cases.export_held_out_cases import HOST_CAMPAIGNS, held_out_path, host_case
from packages.evals.tests.fixtures.writer_cases.export_writer_cases import FROZEN_AT, FROZEN_BY, PACKAGE
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.harness import CaseMint, ReasonedOn, WithheldNote, WriterCaseFile, writer_case_from_export
from threetears.evals.harness.__main__ import main
from threetears.evals.kernel import EvalInsight
from threetears.evals.kernel.errors import RefusedInput
from threetears.evals.kernel.host import HostProfile
from threetears.evals.quick.cli import EXIT_OK, EXIT_REFUSED
from threetears.evals.schema.hashing import identity_json

#: The courier's capture: the export.
CAPTURE = PACKAGE / "tests" / "fixtures" / "hot_path_oracle" / "inputs" / "courierhost.json"

#: The courier's profile factory, as a host names its own.
FACTORY = "packages.evals.tests.fixtures.courierhost:courier_profile"

#: The held-out courier case's metadata, as the held-out export names it.
NAME, GROUP, STRATUM = "courierhost", "courier-planner", "host"

#: An insight whose statement a shipped note holds, word for word: it leads to that note.
RETOLD = {"doc_type": "eval_insight", "id": "insight-1", "name": "Neither candidate is   better on anything."}

#: An analysis no note names.
UNTOLD = {"doc_type": "eval_analysis", "id": "analysis-1", "name": "analysis 2026-10-30T14:09:11+00:00 planner-pro"}


def _declared_digest() -> str:
    """The courier's declarations' digest, as the committed case's bundle records it."""
    bundle = WriterCaseFile.read(held_out_path(NAME)).snapshot.bundle
    assert bundle is not None
    (digest,) = {aspect["host_declarations_digest"] for aspect in bundle["aspects"]}
    return str(digest)


def _documents() -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = json.loads(CAPTURE.read_text())
    return documents


def _campaign(documents: list[dict[str, Any]]) -> dict[str, Any]:
    (campaign,) = [document for document in documents if document["doc_type"] == "eval_campaign"]
    return campaign


def _profile() -> HostProfile:
    from packages.evals.tests.fixtures.courierhost import courier_profile

    return courier_profile()


async def _minted(documents: list[dict[str, Any]], **given: Any) -> WriterCaseFile:
    metadata: dict[str, Any] = {"name": NAME, "group": GROUP, "stratum": STRATUM, "created_by": FROZEN_BY}
    return await writer_case_from_export(
        documents, _profile(), campaign_id=_campaign(_documents())["id"], **{**metadata, **given}
    )


def _figures(case: WriterCaseFile) -> str:
    """The case as ``test_held_out_cases.py`` compares it: floats to twelve figures, no ids, no build version."""

    def without_build(value: object) -> object:
        if isinstance(value, dict):
            return {key: without_build(item) for key, item in value.items() if key != "package_version"}
        if isinstance(value, list):
            return [without_build(item) for item in value]
        return value

    document = json.loads(case.to_json())
    for derived in ("id", "report_input_digest"):
        del document["snapshot"][derived]
    return identity_json(without_build(document))


def _files(
    tmp_path: Path,
    *,
    reasoned_on: list[dict[str, str]] | None = None,
    campaign_id: str | None = None,
    built_by: str = FACTORY,
    digest: str | None = None,
) -> list[str]:
    """The host's three files, built from the capture, and the arguments naming them."""
    documents = _documents()
    (tmp_path / "documents.json").write_text(json.dumps(documents))
    # The declaration carries the host's record of its declarations; the mint reads its factory and its digest.
    declaration = {
        "host_id": "courier",
        "sweepables": {"declarations": [{"name": "candidate_model", "role": "lever"}]},
        "built_by": built_by,
        "host_declarations_digest": digest or _declared_digest(),
    }
    (tmp_path / "profile.json").write_text(json.dumps(declaration))
    manifest = {
        "campaign_id": campaign_id or _campaign(documents)["id"],
        "reasoned_on": reasoned_on or [],
        "unreachable": ["eval_action_job"],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return [
        "mint",
        "--documents",
        str(tmp_path / "documents.json"),
        "--profile",
        str(tmp_path / "profile.json"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--campaign",
        _campaign(documents)["id"],
    ]


def _case_args(out: Path) -> list[str]:
    return ["--name", NAME, "--group", GROUP, "--stratum", STRATUM, "--created-by", FROZEN_BY, "--out", str(out)]


async def test_the_door_mints_the_case_the_held_out_export_mints(tmp_path: Path) -> None:
    (courier,) = [campaign for campaign in HOST_CAMPAIGNS if campaign.name == NAME]
    exported = (await host_case(courier)).write(tmp_path / "exported.json.gz")
    minted = (await _minted(_documents(), as_of=FROZEN_AT)).write(tmp_path / "minted.json.gz")
    assert minted.read_bytes() == exported.read_bytes()
    committed = WriterCaseFile.read(held_out_path(NAME))
    case = WriterCaseFile.read(minted)
    assert case.snapshot.id == committed.snapshot.id
    assert _figures(case) == _figures(committed)


def test_the_command_line_mints_it_from_the_hosts_three_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cases" / f"{NAME}.json.gz"
    assert main([*_files(tmp_path), *_case_args(out), "--as-of", FROZEN_AT]) == EXIT_OK
    case = WriterCaseFile.read(out)
    committed = WriterCaseFile.read(held_out_path(NAME))
    assert (case.format, case.name, case.group, case.stratum, case.withhold) == (
        "writer-case/1",
        NAME,
        GROUP,
        STRATUM,
        [],
    )
    assert case.snapshot.id == committed.snapshot.id
    assert _figures(case) == _figures(committed)
    assert case.minted is None
    assert capsys.readouterr().out == f"{out}: case {NAME} over {case.snapshot.id}, withholding nothing\n"
    # The factory named on the command line, with no declaration, writes the same bytes.
    again = tmp_path / f"again{out.name}"
    args = [*_files(tmp_path), *_case_args(again), "--as-of", FROZEN_AT]
    at = args.index("--profile")
    args[at : at + 2] = ["--host", FACTORY]
    assert main(args) == EXIT_OK
    assert again.read_bytes() == out.read_bytes()


def test_a_profile_whose_declarations_are_not_the_ones_declared_is_refused_naming_both(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / f"{NAME}.json.gz"
    assert main([*_files(tmp_path, digest="0" * 64), *_case_args(out)]) == EXIT_REFUSED
    err = capsys.readouterr().err
    assert f"host 'courier''s profile declares {_declared_digest()}, and the export's declaration is {'0' * 64}" in err
    assert not out.exists()


def test_a_declaration_naming_no_factory_needs_the_factory_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / f"{NAME}.json.gz"
    assert main([*_files(tmp_path, built_by="courier 0.92"), *_case_args(out)]) == EXIT_REFUSED
    assert "no profile factory is named" in capsys.readouterr().err
    assert not out.exists()


async def test_minting_the_same_evidence_again_writes_the_same_bytes_at_the_evidences_own_instant(
    tmp_path: Path,
) -> None:
    first = (await _minted(_documents())).write(tmp_path / "first.json.gz")
    second = (await _minted(_documents())).write(tmp_path / "second.json.gz")
    assert first.read_bytes() == second.read_bytes()
    case = WriterCaseFile.read(first)
    # The campaign was filed after its runs (none of them records a completion), so its creation is the instant.
    documents = _documents()
    stamps = [_campaign(documents)["created_at"]] + [
        document.get("completed_at") or document["created_at"]
        for document in documents
        if document["doc_type"] == "eval_run"
    ]
    assert case.snapshot.bundle_assembled_at == case.snapshot.created_at == max(stamps)
    assert case.snapshot.id == WriterCaseFile.read(held_out_path(NAME)).snapshot.id


@pytest.mark.parametrize("doc_type", ["eval_campaign", "eval_run", "eval_template", "eval_result", "eval_trace"])
async def test_an_export_without_a_document_type_the_freeze_reads_is_refused_by_its_name(doc_type: str) -> None:
    documents = [document for document in _documents() if document["doc_type"] != doc_type]
    with pytest.raises(RefusedInput, match=rf"the export holds no {doc_type}\b"):
        await _minted(documents)


async def test_an_export_missing_one_of_the_campaigns_runs_is_refused_naming_it() -> None:
    documents = _documents()
    dropped = _campaign(documents)["aspects"][0]["run_ids"][0]
    kept = [document for document in documents if (document["doc_type"], document["id"]) != ("eval_run", dropped)]
    with pytest.raises(RefusedInput, match=rf"the export holds no eval_run {dropped}, which campaign"):
        await _minted(kept)


async def test_the_insight_ledger_is_read_as_it_stood_at_the_instant_frozen() -> None:
    documents = _documents()
    campaign = _campaign(documents)

    def insight(observed_at: str) -> dict[str, Any]:
        return EvalInsight(
            scope_id=campaign["scope_id"],
            subject_id=campaign["subject_id"],
            subject_kind=campaign["subject_kind"],
            statement="The pro planner is never late.",
            confidence="high",
            invalidation_trigger="its analysis archived",
            observed_at=observed_at,
        ).to_dict()

    bare = await _minted(documents, as_of=FROZEN_AT)
    later = await _minted([*documents, insight("2026-11-04T00:00:00+00:00")], as_of=FROZEN_AT)
    earlier = await _minted([*documents, insight("2026-11-02T00:00:00+00:00")], as_of=FROZEN_AT)
    assert later.snapshot.id == bare.snapshot.id
    assert earlier.snapshot.id != bare.snapshot.id


def test_what_reasoned_on_the_campaign_withholds_the_notes_retelling_it_and_the_case_says_which(
    tmp_path: Path,
) -> None:
    out = tmp_path / f"{NAME}.json.gz"
    args = [*_files(tmp_path, reasoned_on=[RETOLD, UNTOLD]), *_case_args(out), "--withhold", "r1-model-selection.md"]
    assert main(args) == EXIT_OK
    case = WriterCaseFile.read(out)
    assert case.withhold == ["kairo-round2.md", "r1-model-selection.md"]
    assert case.minted == CaseMint(
        withheld=[
            WithheldNote(note="kairo-round2.md", reasoned_on=[ReasonedOn.model_validate(RETOLD)]),
            WithheldNote(note="r1-model-selection.md"),
        ],
        unmatched=[ReasonedOn.model_validate(UNTOLD)],
    )
    assert case.snapshot.id == WriterCaseFile.read(held_out_path(NAME)).snapshot.id


async def test_a_note_naming_the_campaign_is_withheld() -> None:
    campaign_id = _campaign(_documents())["id"]
    shipped = load_skill()
    note = "r2-config-sweep.md"
    skill = shipped.with_file(f"exemplars/{note}", f"{shipped.files[f'exemplars/{note}']}\nCampaign {campaign_id}.")
    case = await _minted(_documents(), skill=skill)
    assert case.withhold == [note]
    assert case.minted == CaseMint(
        withheld=[WithheldNote(note=note, reasoned_on=[ReasonedOn(doc_type="eval_campaign", id=campaign_id)])]
    )


async def test_a_note_the_skill_does_not_hold_is_refused() -> None:
    with pytest.raises(RefusedInput, match="the skill holds no note courier.md"):
        await _minted(_documents(), withhold=["courier.md"])


def test_a_manifest_of_another_campaign_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / f"{NAME}.json.gz"
    assert main([*_files(tmp_path, campaign_id="another"), *_case_args(out)]) == EXIT_REFUSED
    assert "is campaign 'another''s, not" in capsys.readouterr().err
    assert not out.exists()


def test_a_factory_returning_neither_a_profile_nor_a_host_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [*_files(tmp_path), *_case_args(tmp_path / f"{NAME}.json.gz")]
    at = args.index("--profile")
    args[at : at + 2] = ["--host", "builtins:dict"]
    assert main(args) == EXIT_REFUSED
    assert "builtins:dict returned a dict, not a HostProfile or an EvalHost" in capsys.readouterr().err


def test_a_malformed_mint_command_line_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["mint", "--documents", "documents.json"]) == EXIT_REFUSED
    assert "the following arguments are required" in capsys.readouterr().err
