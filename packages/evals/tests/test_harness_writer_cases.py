"""Portable writer cases: a campaign frozen to a file, read back, and minted as a writer case in another store's scope.

A writer eval ships its cases beside itself, so a case must leave the store it was frozen in. A
:class:`~threetears.evals.harness.writer_cases.WriterCaseFile` is one frozen report snapshot and what minting it names
(its group, its stratum, the notes it withholds), as one gzipped JSON file:

- **A quick campaign travels.** A comparison run on scripted runners in a memory store is frozen through its own
  reading host, written, read back whole, restored into the writer host's scope (another store, another scope) and
  minted there by its snapshot's id: the stored case carries the stratum, and the side its mint recorded.
- **A bare report input reads through the same loader**: one directory holds both sources, and one reader reads them.
- **A stratum is metadata**: minting the same case under another stratum relabels it in place; the engine's mint
  action carries the stratum as the operation does.
- **A skill loads from a directory** (``AnalystSkill.at``), and ``with_file`` makes a variant under its own digest.
- **A relative path is the eval file's**: the skill directory and the case directory an eval file names are read beside
  it, as its ``Eval``'s cases and its recordings are, whatever the working directory.

Mutations that turn this file red: the file's bytes carrying a timestamp; a reader that drops or re-derives a field;
the snapshot restored under the source scope, or minted under another id; the stratum not stored on the case, or
dropped from the receipt or the action; a reader branching on the snapshot's source, so one of the two fails; two
cases of one name read from one directory; ``at`` reading another tree than the one named; ``with_file`` keeping the
old digest or adding a file it should refuse; ``at`` or the case reader resolving a relative path in the working
directory while an eval file loads.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.analyst_support import fixture_snapshots
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import CALLER
from packages.evals.tests.writer_case_support import WRITER_TEMPLATE, writer_host
from threetears.evals import Eval, Result
from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.ops import OpsHost
from threetears.evals.ops.report_writer.writer_bank import WriterSplit
from threetears.evals.ops.report_writer.writer_kind import writer_case_of
from threetears.evals.harness.writer_cases import WriterCaseFile, read_writer_cases, writer_case_freeze
from threetears.evals.kernel.eval_file import loading_eval_file

#: The quick eval's own scope, never the writer host's.
SOURCE_SCOPE = "writer-cases-source"

#: Every group lands held-out at this share: the side is the split's, and the test reads it off the stored case.
HELD_OUT = WriterSplit(held_out_fraction=0.99)

#: Eight tickets; the last four carry a second label's word, which only ``careful`` reads past.
CASES = [
    {"text": "refund please", "label": "billing"},
    {"text": "charged twice", "label": "billing"},
    {"text": "app crashes", "label": "bug"},
    {"text": "button broken", "label": "bug"},
    {"text": "refund page crashes", "label": "billing"},
    {"text": "charged then crashed", "label": "billing"},
    {"text": "refund form broken", "label": "billing"},
    {"text": "charged though the button is broken", "label": "billing"},
]

#: A misfit: a reference input reshaped, frozen bare with no campaign behind it.
MISFIT = "r1 without its gates"


async def careful(case: Mapping[str, Any]) -> str:
    """Billing whenever money is mentioned, else a bug."""
    text = case["text"]
    return "billing" if "refund" in text or "charged" in text else "bug"


async def hasty(case: Mapping[str, Any]) -> str:
    """A bug whenever anything crashed, else billing."""
    text = case["text"]
    return "bug" if "crash" in text or "broken" in text else "billing"


async def _quick_campaign() -> Result:
    """A two-arm comparison on scripted runners, in memory."""
    ev = Eval(CASES, {"hasty": hasty, "careful": careful}, ["label"], k=1, name="triage")
    return await ev.run(store=":memory:", scope=SOURCE_SCOPE)


async def _quick_case(result: Result) -> WriterCaseFile:
    """The campaign frozen as a portable case, through the quick eval's own host."""
    return await writer_case_freeze(
        OpsHost.reading(result.host),
        result.scope_id,
        campaign_id=result.campaign_id,
        name="1-triage",
        stratum="support-triage",
        created_by=CALLER.identity,
    )


def _misfit_case() -> WriterCaseFile:
    snapshot = fixture_snapshots("misfits-elsewhere")[MISFIT]
    assert snapshot.source == "report_input" and snapshot.campaign_id is None
    return WriterCaseFile(
        name=MISFIT, group="r1", stratum="misfit", withhold=["r1-model-selection.md"], snapshot=snapshot
    )


class TestAQuickCampaignTravels:
    async def test_frozen_written_read_restored_and_minted_with_its_stratum_and_side(self, tmp_path: Path) -> None:
        result = await _quick_campaign()
        case = await _quick_case(result)
        assert (case.snapshot.source, case.snapshot.campaign_id) == ("campaign", result.campaign_id)
        assert case.snapshot.scope_id == SOURCE_SCOPE

        path = case.write(tmp_path / "cases" / "1-triage.json.gz")
        again = case.write(tmp_path / "again" / "1-triage.json.gz")
        assert path.read_bytes() == again.read_bytes(), "the same case writes the same bytes"
        read = WriterCaseFile.read(path)
        assert read == case

        fixture = off_loop(writer_host)
        assert CALLER.scope_id != SOURCE_SCOPE
        minted = await read.mint(
            fixture.host, WRITER_TEMPLATE.id, CALLER.scope_id, created_by=CALLER.identity, split=HELD_OUT
        )
        assert minted.snapshot_id == case.snapshot.id
        assert (minted.group, minted.stratum, minted.side, minted.reused) == (
            "1-triage",
            "support-triage",
            "held_out",
            False,
        )
        assert minted.case_set.name == HELD_OUT.held_out

        storage = fixture.host.eval_host.storage
        restored = off_loop(storage.load_report_snapshot, case.snapshot.id, CALLER.scope_id)
        assert restored is not None and restored.scope_id == CALLER.scope_id
        assert restored.report_input == case.snapshot.report_input
        stored = off_loop(storage.load_test_case, minted.test_case_id, CALLER.scope_id)
        assert stored is not None and stored.stratum == "support-triage"
        writer_case = writer_case_of(stored)
        assert writer_case is not None and writer_case.origin is not None
        assert (writer_case.origin.side, writer_case.origin.group) == (minted.side, "1-triage")
        assert writer_case.snapshot_id == case.snapshot.id

    async def test_a_case_with_no_group_is_its_own_group_by_name(self) -> None:
        case = (await _quick_case(await _quick_campaign())).model_copy(update={"stratum": None})
        fixture = off_loop(writer_host)
        minted = await case.mint(fixture.host, WRITER_TEMPLATE.id, CALLER.scope_id, created_by=CALLER.identity)
        assert (minted.group, minted.stratum) == ("1-triage", None)


class TestOneLoaderReadsEverySource:
    async def test_a_directory_of_a_campaign_case_and_a_bare_input_reads_and_mints_alike(self, tmp_path: Path) -> None:
        campaign_case = await _quick_case(await _quick_campaign())
        campaign_case.write(tmp_path / "1-triage.json.gz")
        misfit = _misfit_case()
        misfit.write(tmp_path / "r1-without-gates.json.gz")
        (tmp_path / "README.md").write_text("not a case", encoding="utf-8")

        cases = read_writer_cases(tmp_path)
        assert [case.name for case in cases] == ["1-triage", MISFIT]
        assert [case.snapshot.source for case in cases] == ["campaign", "report_input"]
        assert cases == (campaign_case, misfit)

        fixture = off_loop(writer_host)
        minted = [
            await case.mint(
                fixture.host, WRITER_TEMPLATE.id, CALLER.scope_id, created_by=CALLER.identity, split=HELD_OUT
            )
            for case in cases
        ]
        assert [(receipt.stratum, receipt.side) for receipt in minted] == [
            ("support-triage", "held_out"),
            ("misfit", "held_out"),
        ]
        assert minted[1].withhold == ["r1-model-selection.md"] and minted[1].group == "r1"
        assert minted[1].source_key.startswith("report-input:")

    def test_two_cases_of_one_name_are_refused(self, tmp_path: Path) -> None:
        _misfit_case().write(tmp_path / "a.json.gz")
        _misfit_case().write(tmp_path / "b.json.gz")
        with pytest.raises(ValueError, match=f"a.json.gz and b.json.gz both hold a case named '{MISFIT}'"):
            read_writer_cases(tmp_path)

    def test_an_empty_directory_and_a_file_that_is_not_a_case_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="holds no writer case file"):
            read_writer_cases(tmp_path)
        (tmp_path / "broken.json.gz").write_bytes(b"not gzip")
        with pytest.raises(ValueError, match="broken.json.gz does not read as a writer case"):
            read_writer_cases(tmp_path)
        with pytest.raises(ValueError, match="no writer case directory"):
            read_writer_cases(tmp_path / "missing")

    def test_a_file_no_directory_read_would_find_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="ends in .json.gz"):
            _misfit_case().write(tmp_path / "case.json")


class TestAStratumIsMetadata:
    async def test_minting_again_under_another_stratum_relabels_the_case_in_place(self) -> None:
        fixture = off_loop(writer_host)
        case = _misfit_case()
        first = await case.mint(fixture.host, WRITER_TEMPLATE.id, CALLER.scope_id, created_by=CALLER.identity)
        same = await case.mint(fixture.host, WRITER_TEMPLATE.id, CALLER.scope_id, created_by=CALLER.identity)
        assert (same.test_case_id, same.reused, same.stratum) == (first.test_case_id, True, "misfit")

        relabelled = await case.model_copy(update={"stratum": "reshaped"}).mint(
            fixture.host, WRITER_TEMPLATE.id, CALLER.scope_id, created_by=CALLER.identity
        )
        assert (relabelled.test_case_id, relabelled.reused, relabelled.stratum) == (
            first.test_case_id,
            True,
            "reshaped",
        )
        assert relabelled.side == first.side and relabelled.case_set == first.case_set
        stored = off_loop(fixture.host.eval_host.storage.load_test_case, first.test_case_id, CALLER.scope_id)
        assert stored is not None and stored.stratum == "reshaped"

        unnamed = await case.model_copy(update={"stratum": None}).mint(
            fixture.host, WRITER_TEMPLATE.id, CALLER.scope_id, created_by=CALLER.identity
        )
        assert unnamed.stratum == "reshaped", "a mint naming no stratum keeps the one the case has"

    async def test_the_mint_action_carries_the_stratum(self) -> None:
        fixture = off_loop(writer_host)
        snapshot = fixture_snapshots(CALLER.scope_id)[MISFIT]
        await _misfit_case().mint(fixture.host, WRITER_TEMPLATE.id, CALLER.scope_id, created_by=CALLER.identity)
        tool = {tool.name: tool for tool in eval_catalogue().mount_all(standard_tools())}["evals"]
        called = await tool.call(
            {
                "action": "writer_case_mint",
                "template_id": WRITER_TEMPLATE.id,
                "snapshot_id": snapshot.id,
                "group": "r1",
                "withhold": ["r1-model-selection.md"],
                "stratum": "by-the-action",
            },
            host=fixture.host,
            caller=CALLER,
        )
        assert not called.is_error, called.text
        assert (called.structured["stratum"], called.structured["reused"]) == ("by-the-action", True)


class TestASkillLoadsFromADirectory:
    def test_a_copy_of_the_shipped_skill_loads_and_a_variant_has_its_own_digest(self, tmp_path: Path) -> None:
        shipped = load_skill()
        for name, text in shipped.files.items():
            (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / name).write_text(text, encoding="utf-8")
        (tmp_path / ".DS_Store").write_bytes(b"\x00")

        copy = AnalystSkill.at(tmp_path)
        assert copy.digest == shipped.digest and dict(copy.files) == dict(shipped.files)
        assert AnalystSkill.at(str(tmp_path)).digest == shipped.digest

        lead = "Lead with the decision, then the evidence."
        variant = copy.with_file("SKILL.md", shipped.files["SKILL.md"] + "\n" + lead + "\n")
        assert variant.digest != copy.digest
        assert {name for name in shipped.files if variant.files[name] != shipped.files[name]} == {"SKILL.md"}
        assert lead in variant.analyst().text and lead not in copy.analyst().text
        assert copy.digest == shipped.digest, "with_file leaves the skill it was called on as it was"
        assert variant.with_file("SKILL.md", shipped.files["SKILL.md"]).digest == shipped.digest

    def test_a_missing_directory_an_incomplete_skill_and_an_unknown_file_are_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no analyst skill directory"):
            AnalystSkill.at(tmp_path / "missing")
        (tmp_path / "SKILL.md").write_text("# only the role", encoding="utf-8")
        with pytest.raises(ValueError, match="the analyst skill is missing"):
            AnalystSkill.at(tmp_path)
        with pytest.raises(ValueError, match="holds no file 'skill.md' to replace"):
            load_skill().with_file("skill.md", "a typo of SKILL.md")


class TestRelativePathsReadBesideTheEvalFile:
    def test_the_skill_and_the_cases_an_eval_file_names_are_read_beside_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shipped, example = load_skill(), tmp_path / "example"
        for name, text in shipped.files.items():
            (example / "skills" / "control" / name).parent.mkdir(parents=True, exist_ok=True)
            (example / "skills" / "control" / name).write_text(text, encoding="utf-8")
        misfit = _misfit_case()
        misfit.write(example / "cases" / "r1.json.gz")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)

        with loading_eval_file(example / "writer.py"):
            assert AnalystSkill.at("skills/control").digest == shipped.digest
            assert read_writer_cases("cases") == (misfit,)
            assert WriterCaseFile.read("cases/r1.json.gz") == misfit
        # Outside a load, a relative path is the working directory's, which holds neither.
        with pytest.raises(ValueError, match="no analyst skill directory"):
            AnalystSkill.at("skills/control")
        with pytest.raises(ValueError, match="no writer case directory"):
            read_writer_cases("cases")
