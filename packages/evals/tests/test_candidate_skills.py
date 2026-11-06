"""Candidate skills: whole skills under ``tests/fixtures/writer_cases/candidates/``, each run by the acceptance script's
``--candidate-skill`` door against the shipped skill.

Mutations that turn this file red: a candidate directory missing a file of the shipped skill, or holding one it does
not; the door accepting such a directory, both ``--variant`` and ``--candidate-skill``, a directory named for the
control, or the shipped skill byte for byte; ``headline-answers`` changing a file its README does not name, or its
README's diff or digests drifting from the files; a candidate's arm not named for its directory, or the record not
naming it by its skill digest; the criterion asking a candidate skill to read worse; bounces compared across two
editors; the shipped skill as its own candidate reading anything but not separated.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import re
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest

from packages.evals.tests.fixtures.writer_cases import writer_acceptance
from packages.evals.tests.fixtures.writer_cases.writer_acceptance import (
    SHIPPED,
    VARIANTS,
    Acceptance,
    acceptance,
    candidate_files,
    candidate_variant,
    files_differing,
    pass_criterion,
    read_candidate_skill,
    refuse_unlike_editors,
)
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.ops.report_writer.writer_contract import BOUNCES_MEASURE
from threetears.evals.ops.report_writer.writer_promotion import WriterPromotion

#: Where the candidate skills sit.
CANDIDATES: Final = Path(__file__).resolve().parent / "fixtures" / "writer_cases" / "candidates"

#: The first candidate: the editor's rubric carrying the owner's 2026-11-06 reading.
HEADLINE_ANSWERS: Final = CANDIDATES / "headline-answers"

#: The variables a live run reads its keys from: a dry run reads neither.
KEYS: Final = ("ANTHROPIC_API_KEY", "OPENROUTER_API_KEY")


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


def _shipped_copy(directory: Path) -> Path:
    """The shipped skill's files written into ``directory``, byte for byte."""
    for name, text in load_skill().files.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text(text, encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def like_against_like(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Acceptance, list[object]]]:
    """One dry run, offline, of the shipped skill as its own candidate: its files with only the README, which no
    session reads, changed, so the arms differ in their skill digest alone and every session's text is the control's.
    """
    copy = _shipped_copy(tmp_path_factory.mktemp("candidates") / "shipped-again")
    (copy / "README.md").write_text("The shipped skill, run as its own candidate.\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as monkeypatch:
        attempts = _offline(monkeypatch)
        yield asyncio.run(acceptance(dry=True, cases_dir=None, cap_usd=10.0, out=None, candidate_skill=copy)), attempts


def _readme_section(heading: str) -> str:
    readme = (HEADLINE_ANSWERS / "README.md").read_text(encoding="utf-8")
    return readme.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]


def test_every_candidate_holds_the_shipped_skills_file_set() -> None:
    shipped = load_skill()
    found = sorted(path for path in CANDIDATES.iterdir() if path.is_dir())
    assert HEADLINE_ANSWERS in found
    for directory in found:
        assert candidate_files(directory) == set(shipped.files), directory.name
        assert set(read_candidate_skill(directory, shipped).files) == set(shipped.files), directory.name


def test_headline_answers_differs_only_in_the_files_its_readme_names() -> None:
    shipped, candidate = load_skill(), AnalystSkill.at(HEADLINE_ANSWERS)
    named = re.findall(r"^\| `([^`]+)` \|", _readme_section("Files changed"), re.M)
    assert sorted(named) == files_differing(shipped, candidate) == ["README.md", "SKILL.md", "editor.md"]
    # Both texts a session reads changed; the editor is no longer the control's instrument, and the door allows it.
    assert candidate.analyst().digest != shipped.analyst().digest
    assert candidate.editor().digest != shipped.editor().digest
    refuse_unlike_editors(shipped, candidate, candidate_variant(HEADLINE_ANSWERS, shipped))


def test_the_readmes_diff_and_digests_are_the_files_own() -> None:
    shipped, candidate = load_skill(), AnalystSkill.at(HEADLINE_ANSWERS)
    (pasted,) = re.findall(r"```diff\n(.*?)```", _readme_section("The diff"), re.S)
    diff = "".join(
        "".join(
            difflib.unified_diff(
                shipped.files[name].splitlines(keepends=True),
                candidate.files[name].splitlines(keepends=True),
                f"a/{name}",
                f"b/{name}",
            )
        )
        for name in ("SKILL.md", "editor.md")
    )
    assert [line.rstrip() for line in pasted.splitlines()] == [line.rstrip() for line in diff.splitlines()]
    digests = _readme_section("Digests")
    assert f"| shipped | `{shipped.digest}` | `{shipped.analyst().digest}` | `{shipped.editor().digest}` |" in digests
    assert f"| headline-answers | printed by the run | `{candidate.analyst().digest}` | " in digests
    assert f"`{candidate.editor().digest}` |" in digests


def test_the_door_refuses_by_name_a_directory_missing_a_file_or_holding_an_extra_one(tmp_path: Path) -> None:
    short = _shipped_copy(tmp_path / "short")
    (short / "exemplars" / "r3-classifier.md").unlink()
    with pytest.raises(SystemExit, match=r"candidate skill short is missing exemplars/r3-classifier\.md, which the"):
        writer_acceptance.main(["--dry", "--candidate-skill", str(short)])
    extra = _shipped_copy(tmp_path / "extra")
    (extra / "voice.md").write_text("# Voice\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=r"candidate skill extra holds voice\.md, which the shipped skill does not"):
        writer_acceptance.main(["--dry", "--candidate-skill", str(extra)])
    with pytest.raises(SystemExit, match="no candidate skill directory"):
        candidate_variant(tmp_path / "nonesuch")


def test_the_door_refuses_the_shipped_skill_itself_a_control_name_and_two_candidates(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="is the shipped skill byte for byte"):
        candidate_variant(_shipped_copy(tmp_path / "same"))
    named = _shipped_copy(tmp_path / SHIPPED)
    (named / "README.md").write_text("Another README.\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not named 'shipped'"):
        candidate_variant(named)
    with pytest.raises(SystemExit):
        writer_acceptance.main(["--dry", "--variant", "hollow", "--candidate-skill", str(HEADLINE_ANSWERS)])
    with pytest.raises(SystemExit, match="not both"):
        asyncio.run(
            acceptance(dry=True, cases_dir=None, cap_usd=10.0, out=None, variant="hollow", candidate_skill=named)
        )


def test_the_shipped_skill_as_its_own_candidate_reads_not_separated(
    like_against_like: tuple[Acceptance, list[object]],
) -> None:
    found, attempts = like_against_like
    assert attempts == []
    assert list(found.result.arms) == [SHIPPED, "shipped-again"]
    assert found.candidate.analyst().digest == found.shipped.analyst().digest
    assert found.candidate.editor().digest == found.shipped.editor().digest
    assert found.read.primary is not None and found.read.primary.outcome == "not_separated"
    assert found.read.decision != "promote" and not found.criterion.passed
    text = "\n".join(found.readout())
    assert "Does the gate read the shipped-again skill as writing better reports than the shipped skill?" in text
    assert "| the read answers promote | " in text
    # One editor, so bounces are compared, and no sentence says the editors differ.
    assert "| bounces do not separate | not separated | yes |" in text
    assert "editor.md differs" not in text


def test_headline_answers_runs_dry_and_the_record_names_it_by_its_digest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    attempts = _offline(monkeypatch)
    out = tmp_path / "out"
    assert writer_acceptance.main(["--dry", "--candidate-skill", str(HEADLINE_ANSWERS), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert attempts == []
    digest = AnalystSkill.at(HEADLINE_ANSWERS).digest
    record = json.loads((out / "acceptance.json").read_text(encoding="utf-8"))
    assert record["variant"] == "headline-answers"
    assert record["candidate_skill"] == {
        "directory": "headline-answers",
        "path": str(HEADLINE_ANSWERS),
        "digest": digest,
    }
    assert record["skills"] == {SHIPPED: load_skill().digest, "headline-answers": digest}
    assert record["readout"] == printed.rstrip("\n")
    assert f"| headline-answers | candidate | {digest[:12]} | README.md, SKILL.md, editor.md | " in printed
    assert (
        "The headline-answers arm's editor.md differs, so each arm's reports are reviewed by its own editor." in printed
    )
    assert "| bounces do not separate | not compared: the arms' editors differ | yes |" in printed
    # Each arm's recordings are keyed by its own skill digest.
    assert f"recordings/headline-answers-{digest[:12]}-" in printed
    assert f"recordings/{SHIPPED}-{load_skill().digest[:12]}-" in printed
    assert re.search(r"^(PASS|FAIL): ", printed, re.M)


def _shaped(read: WriterPromotion, *, decision: str, outcome: str, low: float, high: float) -> WriterPromotion:
    """A read with its decision and primary set as a live run might read them, the order check standing."""
    assert read.primary is not None and read.reliability is not None
    primary = read.primary.model_copy(update={"outcome": outcome, "reason": "separated", "interval": (low, high)})
    reliability = read.reliability.model_copy(
        update={"both_orders": 10, "order_consistent": 9, "order_consistency": 0.9}
    )
    return read.model_copy(update={"decision": decision, "primary": primary, "void": None, "reliability": reliability})


def test_a_candidate_skill_passes_only_when_the_gate_reads_it_better(
    like_against_like: tuple[Acceptance, list[object]],
) -> None:
    found, _ = like_against_like
    variant, runs = found.variant, found.result.arms
    better = pass_criterion(
        _shaped(found.read, decision="promote", outcome="improved", low=0.1, high=0.4), runs, variant=variant
    )
    assert better.passed
    assert (
        better.verdict()
        == "PASS: the gate reads the shipped-again skill as writing better reports, on the primary reading."
    )
    worse = pass_criterion(
        _shaped(found.read, decision="keep", outcome="regressed", low=-0.4, high=-0.1), runs, variant=variant
    )
    assert not worse.passed
    assert "the read answers promote; the primary reading reads improved; it separated, its interval above 0" in (
        worse.verdict()
    )
    # A variant made to write worse still asks for worse.
    assert pass_criterion(
        _shaped(found.read, decision="keep", outcome="regressed", low=-0.4, high=-0.1), runs, variant=VARIANTS["hollow"]
    ).passed
    # Bounces that separate fail a candidate read by one editor, and are not compared across two.
    read = _shaped(found.read, decision="promote", outcome="improved", low=0.1, high=0.4)
    verdicts = [
        verdict.model_copy(update={"outcome": "regressed"}) if verdict.name == BOUNCES_MEASURE else verdict
        for verdict in read.verdicts
    ]
    bounced = read.model_copy(update={"verdicts": verdicts})
    assert not pass_criterion(bounced, runs, variant=variant).passed
    assert pass_criterion(bounced, runs, variant=variant, one_editor=False).passed
