"""The reset's P2 acceptance script, dry: the whole path on scripted models, offline, its criterion decided by code.

``tests/fixtures/writer_cases/writer_acceptance.py --dry`` runs one writer eval: two arms (the shipped skill and its
hollow variant) over the held-out writer cases, then the eval's own pairwise pass and promotion read, with every
writer turn replayed from a scripted recording and a judge that sends nothing. Every socket connect is refused here, so a run that reached a
provider fails. The criterion's two answers are each checked on a read of the right shape.

Mutations that turn this file red: the hollow variant changing ``editor.md`` or a third file, or leaving a hollowed
file longer than one line; any variant changing ``editor.md``, a file it does not name, or nothing that tells its arm
from the control's; the minimal variant keeping a module, a note or the index's prose, or losing a note a case may
withhold; the no-editor arm given a revision; ``--variant`` ignored, or a variant's readout not naming it; the pass
asking one order; the read made without the pass; the criterion passing on
``undecided``, on a primary not separated, on a void primary, on bounces separated either way or on an arm stopped
short or with a case it stored no page on; the readout's first table not counting the cases with two pages; a
report budget under the first turn's ceiling launched, or one under two of it launched with no warning; ``--cases`` ignored; a live run over fewer than 10 held-out cases, or one refused for a key after making its
directory; the doc's commands drifting from the script's.
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from packages.evals.tests.analyst_support import KAIRO, fixture_snapshots
from packages.evals.tests.fixtures.writer_cases import writer_acceptance
from packages.evals.tests.fixtures.writer_cases.writer_acceptance import (
    HOLLOW,
    HOLLOWED,
    INDEX,
    MINIMAL,
    MINIMAL_KEPT,
    NO_EDITOR,
    ONE_PARAGRAPH,
    ONE_PARAGRAPH_SKILL,
    SHIPPED,
    STUB,
    VARIANTS,
    Acceptance,
    acceptance,
    files_differing,
    hollow_variant,
    minimal_variant,
    pass_criterion,
    refuse_unlike_editors,
)
from packages.evals.tests.writer_case_support import SEED
from threetears.evals.analysis.analyst.skill import load_skill
from threetears.evals.harness import WriterCaseFile
from threetears.evals.ops.report_writer.loop import MOST_REVISIONS
from threetears.evals.ops.report_writer.preference import preference_rubric
from threetears.evals.ops.report_writer.writer_contract import BOUNCES_MEASURE
from threetears.evals.ops.report_writer.writer_promotion import WriterPromotion

#: The variables a live run reads its keys from: a dry run reads neither.
KEYS: Final = ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")

#: The default cases: every fixture prototype 5 wrote over.
DEFAULT_CASES: Final = 7

#: The page the doc's acceptance commands sit on.
DOC: Final = Path(__file__).resolve().parents[1] / "docs" / "judges-and-calibration.md"


def _offline(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """No key in the environment, and every socket connect refused and counted."""
    for name in KEYS:
        monkeypatch.delenv(name, raising=False)
    attempts: list[object] = []

    def connect(_self: socket.socket, address: object) -> None:
        attempts.append(address)
        raise AssertionError(f"a dry acceptance run connected to {address}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    return attempts


@pytest.fixture(scope="module")
def dry() -> Iterator[tuple[Acceptance, list[object]]]:
    """One dry run over the default cases, offline; what it found and every connect it tried."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        attempts = _offline(monkeypatch)
        found = asyncio.run(acceptance(dry=True, cases_dir=None, cap_usd=10.0, out=None))
        yield found, attempts


def test_the_hollow_variant_differs_in_exactly_the_two_files_and_keeps_the_editor() -> None:
    shipped, hollow = load_skill(), hollow_variant()
    assert files_differing(shipped, hollow) == sorted(HOLLOWED) == ["principles.md", "statistics.md"]
    for name in HOLLOWED:
        assert hollow.files[name] == STUB
        assert len(hollow.files[name].splitlines()) == 1
        assert len(shipped.files[name].splitlines()) > 1
    assert hollow.editor().digest == shipped.editor().digest
    assert hollow.files["editor.md"] == shipped.files["editor.md"]
    assert hollow.digest != shipped.digest
    refuse_unlike_editors(shipped, hollow)


def test_a_variant_whose_editor_or_a_third_file_differs_is_refused() -> None:
    shipped, hollow = load_skill(), hollow_variant()
    with pytest.raises(SystemExit, match=r"editor\.md differs"):
        refuse_unlike_editors(shipped, hollow.with_file("editor.md", shipped.files["editor.md"] + "\nOne more rule.\n"))
    with pytest.raises(SystemExit, match=r"differs in \['principles\.md', 'statistics\.md', 'writing\.md'\]"):
        refuse_unlike_editors(shipped, hollow.with_file("writing.md", STUB))


def test_every_variant_keeps_the_editor_changes_only_its_files_and_differs_from_the_control() -> None:
    shipped = load_skill()
    assert list(VARIANTS) == [HOLLOW, MINIMAL, NO_EDITOR, ONE_PARAGRAPH]
    for name, variant in VARIANTS.items():
        candidate = variant.build(shipped)
        assert candidate.files["editor.md"] == shipped.files["editor.md"], name
        assert candidate.editor().digest == shipped.editor().digest, name
        assert files_differing(shipped, candidate) == variant.changes(shipped), name
        refuse_unlike_editors(shipped, candidate, variant)
        # What the arm runs differs from the control's: its skill, or (no-editor) its revisions alone.
        assert (candidate.digest, variant.revisions) != (shipped.digest, MOST_REVISIONS), name
        if name == NO_EDITOR:
            assert candidate.digest == shipped.digest and variant.revisions == 0
        else:
            assert candidate.digest != shipped.digest and variant.revisions == MOST_REVISIONS, name
    assert VARIANTS[ONE_PARAGRAPH].build(shipped).files["SKILL.md"] == ONE_PARAGRAPH_SKILL
    with pytest.raises(SystemExit, match=r"the minimal variant differs in"):
        refuse_unlike_editors(shipped, minimal_variant(shipped).with_file("SKILL.md", STUB), VARIANTS[MINIMAL])


def test_the_minimal_variant_keeps_the_role_file_and_the_editor_and_a_bare_row_per_note() -> None:
    shipped, minimal = load_skill(), minimal_variant()
    notes = sorted(name for name in shipped.files if name.startswith("exemplars/") and name != INDEX)
    assert len(notes) == 5
    for name in shipped.files:
        if name in MINIMAL_KEPT:
            assert minimal.files[name] == shipped.files[name], name
        elif name != INDEX:
            assert minimal.files[name] == STUB, name
    index = minimal.files[INDEX]
    assert index.startswith(STUB)
    assert [line for line in index.splitlines() if line.startswith("| `")] == [
        f"| `{name.removeprefix('exemplars/')}` |" for name in notes
    ]
    # A case still withholds a note by name, and the analyst's text is the role file and stubs.
    text = minimal.analyst(withhold=["kairo-round2.md"])
    assert "kairo-round2.md" not in text.text
    assert len(text.text) < len(shipped.files["SKILL.md"]) + 1000


@pytest.mark.parametrize("variant", [MINIMAL, NO_EDITOR, ONE_PARAGRAPH])
def test_each_variant_runs_dry_offline_and_its_readout_names_it(variant: str) -> None:
    chosen = VARIANTS[variant]
    with pytest.MonkeyPatch.context() as monkeypatch:
        attempts = _offline(monkeypatch)
        found = asyncio.run(acceptance(dry=True, cases_dir=None, cap_usd=10.0, out=None, variant=variant))
    assert attempts == []
    assert list(found.result.arms) == [SHIPPED, variant]
    assert all(summary.status == "completed" for summary in found.result.arms.values())
    assert found.report.candidate == variant
    assert found.read.changed == (["report_writer.revisions"] if variant == NO_EDITOR else ["report_writer.skill"])
    text = "\n".join(found.readout())
    assert chosen.question in text
    files = ", ".join(chosen.changes(found.shipped)) or "none"
    assert f"| {variant} | candidate | {found.candidate.digest[:12]} | {files} | " in text
    assert ("0 revisions and the control 2" in text) is (variant == NO_EDITOR)
    assert found.summary()["variant"] == variant
    assert found.summary()["skills"] == {SHIPPED: found.shipped.digest, variant: found.candidate.digest}
    campaign = found.result.host.storage.load_campaign(found.result.campaign_id, found.result.scope_id)
    assert campaign is not None and campaign.question == chosen.question
    passed = pass_criterion(
        _shaped(found.read, decision="keep", outcome="regressed", reason="separated", high=-0.1),
        found.result.arms,
        found.report,
        variant=chosen,
    )
    assert passed.verdict() == f"PASS: the gate reads {chosen.noun} as writing worse reports, on the primary reading."


def test_main_takes_a_variant_by_name_and_refuses_an_unknown_one(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    attempts = _offline(monkeypatch)
    assert writer_acceptance.main(["--dry", "--variant", ONE_PARAGRAPH]) == 0
    printed = capsys.readouterr().out
    assert attempts == []
    assert VARIANTS[ONE_PARAGRAPH].question in printed
    assert f"| {ONE_PARAGRAPH} | candidate | " in printed
    with pytest.raises(SystemExit):
        writer_acceptance.main(["--dry", "--variant", "nonesuch"])


def test_the_dry_run_makes_no_network_call_over_two_arms_and_every_held_out_case(
    dry: tuple[Acceptance, list[object]],
) -> None:
    found, attempts = dry
    assert attempts == []
    result = found.result
    assert list(result.arms) == [SHIPPED, HOLLOW]
    assert result.control == SHIPPED
    assert {arm: summary.n_cases for arm, summary in result.arms.items()} == {
        SHIPPED: DEFAULT_CASES,
        HOLLOW: DEFAULT_CASES,
    }
    assert all(summary.status == "completed" for summary in result.arms.values())
    assert found.read.cases == DEFAULT_CASES
    assert [ref.name for ref in found.read.case_sets] == ["writer-held-out"]
    assert found.with_documents == DEFAULT_CASES
    assert found.read.changed == ["report_writer.skill"]


def test_the_pass_asks_every_pair_in_both_orders(dry: tuple[Acceptance, list[object]]) -> None:
    found, _ = dry
    report = found.report
    assert report.campaign_id == found.result.campaign_id
    assert (report.candidate, report.control) == (HOLLOW, SHIPPED)
    assert len(report.pairs) == DEFAULT_CASES
    for pair in report.pairs:
        assert pair.missing is None
        assert [answer.candidate for answer in pair.answers] == ["B", "A"]
        assert all(answer.answer == "scored" for answer in pair.answers)
    assert report.calls_made == 2 * DEFAULT_CASES
    assert report.reliability.both_orders == DEFAULT_CASES
    # The eval's own pass, under its part of the cap: a fifth of it.
    read = found.result.pairwise[HOLLOW]
    assert found.result.pairwise_unread == {} and read.primary == report.whole.verdict
    assert read.read_on == read.job_id == found.read.pairwise_job
    assert report.cap_usd == pytest.approx(10.0 * 0.2)


def test_the_rule_is_declared_before_the_first_result_and_recorded_and_the_readout_says_the_floor(
    dry: tuple[Acceptance, list[object]],
) -> None:
    """The default rule reads every case, declared as the eval starts, so before its first result; the record and
    acceptance.json carry it, and the readout says what it did and the floor its seven cases set."""
    found, _ = dry
    summary = found.summary()
    stopping = summary["stopping"]
    assert stopping == summary["pairwise"]["stopping"]
    assert stopping["rule"]["stop_when"] == "never" and stopping["rule"]["max_cases"] == DEFAULT_CASES
    assert stopping["late"] is False and stopping["stopped_at"] is None
    assert len(stopping["looks"]) == DEFAULT_CASES
    floor = "At 7 cases this run can separate only a preference of 89% or more, held on every case."
    assert summary["resolution"] == floor
    assert floor in found.result.disclosures
    assert stopping["line"] in summary["readout"] and floor in summary["readout"]


def test_a_rule_to_stop_on_separation_is_recorded_as_declared(monkeypatch: pytest.MonkeyPatch) -> None:
    _offline(monkeypatch)
    found = asyncio.run(acceptance(dry=True, cases_dir=None, cap_usd=10.0, out=None, stop_when="separated"))
    stopping = found.summary()["stopping"]
    assert stopping["rule"]["stop_when"] == "separated" and stopping["rule"]["max_cases"] == DEFAULT_CASES
    # The planned cases are the held-out set as declared, every one the arms ran.
    planned = sorted({pair.case_id for pair in found.report.pairs})
    assert stopping["rule"]["case_ids"] == planned and stopping["absent"] == stopping["unplanned"] == []
    assert stopping["late"] is False
    reached = sum(1 for pair in found.report.pairs if pair.reached)
    assert found.report.calls_made == 2 * reached
    assert (stopping["stopped_at"] is None) == (reached == DEFAULT_CASES)


@pytest.mark.parametrize("same", [True, False], ids=["same cases", "other cases"])
def test_a_prior_run_on_the_same_cases_carries_its_projection_into_the_floors_line(
    dry: tuple[Acceptance, list[object]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same: bool
) -> None:
    """``--prior`` reads an earlier acceptance.json; its projection is carried only when it ran the same case files."""
    earlier = dry[0].summary()
    assert len(earlier["cases"]) == DEFAULT_CASES
    cases = earlier["cases"] if same else earlier["cases"][1:]
    record = tmp_path / "acceptance.json"
    record.write_text(
        json.dumps({**earlier, "cases": cases, "promotion": {**earlier["promotion"], "cases_to_settle": 43}}),
        encoding="utf-8",
    )
    _offline(monkeypatch)
    prior = writer_acceptance.read_prior(record)
    found = asyncio.run(acceptance(dry=True, cases_dir=None, cap_usd=10.0, out=None, prior=prior))
    floor = "At 7 cases this run can separate only a preference of 89% or more, held on every case"
    resolution = found.summary()["resolution"]
    if same:
        assert resolution == f"{floor}; the last run on these cases projected 43 to settle."
    else:
        assert resolution == f"{floor}. The prior run given ran other cases, so its projection is not carried."
    assert resolution in found.result.disclosures


def test_the_read_takes_the_pass_as_its_primary_and_the_criterion_is_decided_on_it(
    dry: tuple[Acceptance, list[object]],
) -> None:
    found, _ = dry
    read = found.read
    assert read.pairwise_job is not None and found.summary()["pass_job"] == read.pairwise_job
    assert read.primary is not None
    assert read.primary.name == preference_rubric().name
    assert read.primary == found.report.whole.verdict
    assert read.decision in ("promote", "keep", "undecided")
    expected = (
        read.decision == "keep"
        and read.primary.outcome == "regressed"
        and read.primary.reason == "separated"
        and read.primary.interval is not None
        and read.primary.interval[1] < 0
        and read.void is None
    )
    assert found.criterion.passed is expected
    holds = {requirement.asks: requirement.holds for requirement in found.criterion.requirements}
    assert holds["both arms stored a page on every case"] is True
    assert holds["the read answers keep"] is (read.decision == "keep")


def test_the_readout_prints_the_read_and_the_criterion_with_its_verdict(
    dry: tuple[Acceptance, list[object]],
) -> None:
    found, _ = dry
    text = "\n".join(found.readout())
    assert writer_acceptance.QUESTION in text
    # The first table counts the pages the judge had to compare, before any other.
    first = text[text.index("| ") :]
    assert first.startswith(
        f"| Pages | Cases |\n|---|---|\n| Cases with two pages to compare | {DEFAULT_CASES} of {DEFAULT_CASES} |"
    )
    assert "| No page on either side, so a tie | 0 |" in text
    assert found.read.readout in text
    assert found.read.why in text
    assert "| Pass criterion | Reads | Holds |" in text
    for requirement in found.criterion.requirements:
        assert f"| {requirement.asks} | {requirement.reads} | {'yes' if requirement.holds else 'no'} |" in text
    assert found.criterion.verdict() in text
    assert re.search(r"^(PASS|FAIL): ", text, re.M)
    assert "principles.md, statistics.md" in text
    assert found.summary()["criterion"]["passed"] is found.criterion.passed
    # The campaign is filed asking the acceptance's own question.
    campaign = found.result.host.storage.load_campaign(found.result.campaign_id, found.result.scope_id)
    assert campaign is not None and campaign.question == writer_acceptance.QUESTION


def _shaped(read: WriterPromotion, *, decision: str, outcome: str, reason: str, high: float) -> WriterPromotion:
    """The dry read with its decision and primary set as a live run might read them, the order check standing."""
    assert read.primary is not None and read.reliability is not None
    primary = read.primary.model_copy(update={"outcome": outcome, "reason": reason, "interval": (-0.4, high)})
    reliability = read.reliability.model_copy(
        update={"both_orders": 10, "order_consistent": 9, "order_consistency": 0.9}
    )
    return read.model_copy(
        update={
            "decision": decision,
            "primary": primary,
            "void": None,
            "reliability": reliability,
            "cases_to_settle": None,
        }
    )


def test_the_criterion_passes_only_on_keep_with_a_separated_regressed_primary(
    dry: tuple[Acceptance, list[object]],
) -> None:
    read, runs, report = dry[0].read, dry[0].result.arms, dry[0].report
    passed = pass_criterion(
        _shaped(read, decision="keep", outcome="regressed", reason="separated", high=-0.1), runs, report
    )
    assert passed.passed
    assert passed.verdict() == "PASS: the gate reads the hollow skill as writing worse reports, on the primary reading."
    # A keep reached through a gated reading, beside a primary that did not separate, fails.
    gated = pass_criterion(
        _shaped(read, decision="keep", outcome="not_separated", reason="no_margin", high=0.2), runs, report
    )
    assert not gated.passed
    assert "the primary reading reads regressed" in gated.verdict()
    # A regression on too few cases keeps the control, and is no acceptance.
    few = pass_criterion(
        _shaped(read, decision="keep", outcome="regressed", reason="too_few_cases", high=-0.1), runs, report
    )
    assert not few.passed
    undecided = _shaped(read, decision="undecided", outcome="not_separated", reason="no_margin", high=0.2)
    settled = pass_criterion(undecided.model_copy(update={"cases_to_settle": 23}), runs, report)
    assert not settled.passed
    assert "About 23 held-out cases like these would settle the primary" in settled.verdict()
    void = _shaped(read, decision="keep", outcome="regressed", reason="separated", high=-0.1)
    assert not pass_criterion(void.model_copy(update={"void": "order consistency below 0.7"}), runs, report).passed
    # An arm stopped before its last case reads no acceptance, whatever the read says.
    stopped = {**runs, HOLLOW: runs[HOLLOW].model_copy(update={"status": "budget_stopped", "n_results": 3})}
    cut = pass_criterion(
        _shaped(read, decision="keep", outcome="regressed", reason="separated", high=-0.1), stopped, report
    )
    assert not cut.passed
    assert "both arms stored a page on every case" in cut.verdict()
    # Every cell completed, and the candidate stored no page on two cases: the criterion counts pages, not cells.
    blank = report.model_copy(
        update={
            "pairs": [
                pair.model_copy(update={"missing": "candidate" if index == 0 else "both"}) if index < 2 else pair
                for index, pair in enumerate(report.pairs)
            ]
        }
    )
    paged = pass_criterion(
        _shaped(read, decision="keep", outcome="regressed", reason="separated", high=-0.1), runs, blank
    )
    first = paged.requirements[0]
    assert (first.asks, first.holds) == ("both arms stored a page on every case", False)
    assert (
        first.reads
        == f"{SHIPPED} {DEFAULT_CASES - 1} of {DEFAULT_CASES}; {HOLLOW} {DEFAULT_CASES - 2} of {DEFAULT_CASES}"
    )


@pytest.mark.parametrize("outcome", ["regressed", "improved"])
def test_bounces_that_separate_either_way_void_the_acceptance(
    dry: tuple[Acceptance, list[object]], outcome: str
) -> None:
    read = _shaped(dry[0].read, decision="keep", outcome="regressed", reason="separated", high=-0.1)
    verdicts = [
        verdict.model_copy(update={"outcome": outcome}) if verdict.name == BOUNCES_MEASURE else verdict
        for verdict in read.verdicts
    ]
    assert any(verdict.name == BOUNCES_MEASURE for verdict in verdicts)
    found = pass_criterion(read.model_copy(update={"verdicts": verdicts}), dry[0].result.arms, dry[0].report)
    assert not found.passed
    assert "bounces do not separate" in found.verdict()


def test_main_reads_a_case_directory_and_prints_the_readout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    attempts = _offline(monkeypatch)
    snapshots = fixture_snapshots("elsewhere")
    names = [KAIRO, "r1", "r1 without its gates", "r2", "r2 without its gates"]
    for name in names:
        group, withhold = SEED[name]
        case = WriterCaseFile(
            name=name.replace(" ", "-"), group=group, withhold=list(withhold), snapshot=snapshots[name]
        )
        case.write(tmp_path / f"{case.name}.json.gz")
    assert writer_acceptance.main(["--dry", "--cases", str(tmp_path)]) == 0
    printed = capsys.readouterr().out
    assert attempts == []
    assert f"| {HOLLOW} | candidate | " in printed
    assert f"| {len(names)} | completed |" in printed
    # r1 and r2 have no prototype 5 document, so their dry pages are missing on both sides and their pairs tie.
    assert f"Dry: 3 of {len(names)} cases have a scripted document; the other 2 write no page" in printed
    assert f"| Cases with two pages to compare | 3 of {len(names)} |" in printed
    assert "| No page on either side, so a tie | 2 |" in printed
    # Every cell completed, and the criterion still reads the two cases with no page as unmet.
    assert (
        f"| both arms stored a page on every case | {SHIPPED} 3 of {len(names)}; {HOLLOW} 3 of {len(names)} | no |"
        in printed
    )
    assert re.search(r"^(PASS|FAIL): ", printed, re.M)


def test_a_report_budget_that_cannot_admit_one_turn_is_refused_and_one_under_two_is_warned(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The incident's writer. Its first turn's ceiling moves with the skill's size, so each figure is read back from the
    # line that states it and held to the line's own arithmetic.
    attempts = _offline(monkeypatch)
    with pytest.raises(ValueError) as refused:
        writer_acceptance.main(["--dry", "--writer", "claude-sonnet-5-5", "--budget-usd", "0.25"])
    said = re.fullmatch(
        r"the writer eval was not launched, and nothing was sent or spent: the analyst's first turn of arm 'shipped' on "
        r"claude-sonnet-5-5, priced at its 32,000-token output cap and its text before any snapshot is read, can cost "
        r"up to \$([0-9.]+), above the report's \$0\.25 budget, so it cannot admit even one turn of the 2 a report "
        r"takes; a budget of \$([0-9.]+) or more admits it: raise budget_usd",
        str(refused.value),
    )
    assert said is not None, str(refused.value)
    ceiling, admits = float(said[1]), float(said[2])
    assert 0.25 < ceiling <= admits < ceiling + 0.011
    assert attempts == [] and capsys.readouterr().out == ""
    # The 2026-11-06 run's $0.75 fits the first turn, so the run goes through, warned once with the records.
    assert (
        writer_acceptance.main(["--dry", "--writer", "claude-sonnet-5-5", "--budget-usd", "0.75", "--cap-usd", "30"])
        == 0
    )
    printed = capsys.readouterr().out
    (warning,) = [line for line in printed.splitlines() if "recorded Sonnet 5.5 acceptance runs" in line]
    twice = re.search(r"under 2 of a first turn's ceiling on claude-sonnet-5-5 \(\$([0-9.]+)\)", warning)
    assert twice is not None and float(twice[1]) > 0.75
    assert "at $0.75 16 of 50 reports stored a page, at $1.00 17 to 22 of 22, and at $1.50 99 of 100" in warning
    # At two of the ceiling and above, no warning.
    assert (
        writer_acceptance.main(["--dry", "--writer", "claude-sonnet-5-5", "--budget-usd", "1.5", "--cap-usd", "30"])
        == 0
    )
    assert "recorded Sonnet 5.5" not in capsys.readouterr().out
    assert attempts == []


def test_a_live_run_names_its_cap_and_its_directory() -> None:
    with pytest.raises(SystemExit):
        writer_acceptance.main([])
    with pytest.raises(SystemExit):
        writer_acceptance.main(["--cap-usd", "10"])


def test_a_live_run_is_refused_before_its_directory_is_made_on_too_few_cases_or_a_missing_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in KEYS:
        monkeypatch.delenv(name, raising=False)
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="a live run names --cases"):
        writer_acceptance.main(["--cap-usd", "12", "--out", str(out)])
    cases = writer_acceptance.default_cases(tmp_path / "cases")
    with pytest.raises(
        SystemExit, match=r"puts 7 case\(s\) on the held-out side at 0.99, and a live run reads at least 10"
    ):
        writer_acceptance.main(["--cap-usd", "12", "--out", str(out), "--cases", str(cases)])
    # With enough cases, a missing key is refused before the output directory is made, so the retry is not blocked.
    monkeypatch.setattr(writer_acceptance, "MIN_LIVE_CASES", 7)
    with pytest.raises(SystemExit, match="a live run needs ANTHROPIC_API_KEY, OPENROUTER_API_KEY"):
        writer_acceptance.main(["--cap-usd", "12", "--out", str(out), "--cases", str(cases)])
    assert not out.exists()


def test_the_docs_commands_are_the_scripts_own() -> None:
    page = DOC.read_text(encoding="utf-8")
    section = page[page.index("### The acceptance check: a hollowed skill must read worse") :]
    (block,) = re.findall(r"```bash\n(.*?)```", section.split("\n## ", 1)[0], re.S)
    script = writer_acceptance.__doc__
    assert script is not None
    for line in block.strip().splitlines():
        assert f"    {line}\n" in script, line
