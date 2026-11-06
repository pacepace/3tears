"""Candidate skills: whole skills under ``tests/fixtures/writer_cases/candidates/``, each run by the acceptance script's
``--candidate-skill`` door against the shipped skill.

Mutations that turn this file red: a candidate directory missing a file of the shipped skill, or holding one it does
not; the door accepting such a directory, both ``--variant`` and ``--candidate-skill``, a directory named for the
control, or the shipped skill byte for byte; ``headline-answers`` changing a file its README does not name, or its
README's diff or digests drifting from the files; a candidate's arm not named for its directory, or the record not
naming it by its skill digest; the criterion asking a candidate skill to read worse; with two editors, bounces left
uncompared, an editor's own reading (must-fix findings open, report finished) deciding, the read's keep standing in for
the readings it folds in, or the finished pairs not asked to point the primary's way; a live run's refusal coming after
its directory is made; the shipped skill as its own candidate reading anything but not separated.
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
    NOT_COMPARED,
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
from threetears.evals.ops.report_writer.writer_contract import BOUNCES_MEASURE, MUST_FIX_OPEN_MEASURE
from threetears.evals.ops.report_writer.writer_promotion import WriterPromotion

#: Where the candidate skills sit.
CANDIDATES: Final = Path(__file__).resolve().parent / "fixtures" / "writer_cases" / "candidates"

#: The first candidate: the editor's rubric carrying the owner's 2026-11-06 reading.
HEADLINE_ANSWERS: Final = CANDIDATES / "headline-answers"

#: Every file ``headline-answers`` changes: its README names each; every other file is the shipped skill's.
HEADLINE_ANSWERS_CHANGES: Final = [
    "README.md",
    "SKILL.md",
    "editor.md",
    "exemplars/kairo-round2.md",
    "exemplars/r1-model-selection.md",
    "exemplars/r2-config-sweep.md",
    "exemplars/r3-classifier.md",
    "exemplars/r4-tradeoff.md",
    "principles.md",
    "stories.md",
    "writing.md",
]

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
    assert sorted(named) == files_differing(shipped, candidate) == HEADLINE_ANSWERS_CHANGES
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
        for name in HEADLINE_ANSWERS_CHANGES
        if name != "README.md"
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
    assert "PASS means the fixed judge prefers" not in text and "| Analyst rounds |" not in text


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
    assert f"| headline-answers | candidate | {digest[:12]} | {', '.join(HEADLINE_ANSWERS_CHANGES)} | " in printed
    assert (
        "The headline-answers arm's editor.md differs from the control's, so with two editors a keep may be an "
        "instrument difference rather than a verdict on the writing." in printed
    )
    # Two editors: bounces, counted before any editor sees a page, are compared; each editor's own readings are not,
    # and the read's decision, which folds them in, gives way to the readings it would otherwise decide on.
    assert "| bounces do not separate | not separated | yes |" in printed
    assert f"| Must-fix findings open | not separated | {NOT_COMPARED} |" in printed
    assert f"| Report finished | not separated | {NOT_COMPARED} |" in printed
    assert "| the read answers promote |" not in printed
    assert "| no other gated reading regressed, and every guardrail held | none regressed | yes |" in printed
    assert "| the preference over pairs both finished points the same way | " in printed
    # Beside the verdict, once: what a PASS means here, the pairs both finished, and bounces beside analyst rounds.
    after = printed[printed.index("\nFAIL: ") :]
    assert after.count("PASS means the fixed judge prefers") == 1
    assert (
        "PASS means the fixed judge prefers the candidate's pages; the shipped promotion rule, which counts each "
        "editor's own readings, is not applied here." in after
    )
    assert "7 of 7 pairs had both pages stored finished." in after
    assert "| Arm | Bounces | Analyst rounds | Reports |" in after
    assert f"| {SHIPPED} | 0 | 7 | 7 |" in after and "| headline-answers | 0 | 7 | 7 |" in after
    assert "Bounces read not separated for headline-answers, beside each arm's analyst rounds" in after
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
    # Bounces are counted before any editor sees a page, so they separate a candidate with one editor or two.
    read = _shaped(found.read, decision="promote", outcome="improved", low=0.1, high=0.4)
    bounced = _with(read, BOUNCES_MEASURE, "regressed")
    assert not pass_criterion(bounced, runs, variant=variant).passed
    assert not pass_criterion(_finished(bounced, 0.2), runs, variant=variant, one_editor=False).passed
    with pytest.raises(ValueError, match="expected worse"):
        pass_criterion(read, runs, variant=VARIANTS["hollow"], one_editor=False)


def _with(read: WriterPromotion, name: str, outcome: str) -> WriterPromotion:
    """The read with one gated reading's outcome set."""
    assert any(verdict.name == name for verdict in read.verdicts), name
    verdicts = [
        verdict.model_copy(update={"outcome": outcome}) if verdict.name == name else verdict
        for verdict in read.verdicts
    ]
    return read.model_copy(update={"verdicts": verdicts})


def _finished(read: WriterPromotion, delta: float) -> WriterPromotion:
    """The read with its preference over the pairs both stored finished set to ``delta``."""
    assert read.finished_only is not None
    return read.model_copy(update={"finished_only": read.finished_only.model_copy(update={"delta": delta})})


def test_with_two_editors_the_editors_own_readings_decide_nothing_and_the_finished_pairs_must_agree(
    like_against_like: tuple[Acceptance, list[object]],
) -> None:
    found, _ = like_against_like
    variant, runs = found.variant, found.result.arms
    # A stricter editor leaves more must-fix findings open, so the read keeps the control on an instrument difference.
    kept = _with(
        _shaped(found.read, decision="keep", outcome="improved", low=0.1, high=0.4), MUST_FIX_OPEN_MEASURE, "regressed"
    )
    assert not pass_criterion(_finished(kept, 0.2), runs, variant=variant).passed
    two = pass_criterion(_finished(kept, 0.2), runs, variant=variant, one_editor=False)
    assert two.passed, two.verdict()
    rows = {requirement.asks: requirement for requirement in two.requirements}
    assert "the read answers promote" not in rows
    assert rows["Must-fix findings open"].holds is None and rows["Must-fix findings open"].reads == "regressed"
    assert rows["Report finished"].holds is None
    # The finished pairs, free of draft banners, must point the primary's way.
    against = pass_criterion(_finished(kept, -0.1), runs, variant=variant, one_editor=False)
    assert not against.passed
    assert "the preference over pairs both finished points the same way" in against.verdict()
    # Any other gated reading regressed still fails it, by name.
    judged = next(verdict.name for verdict in kept.verdicts if verdict.reading == "judged")
    other = pass_criterion(_finished(_with(kept, judged, "regressed"), 0.2), runs, variant=variant, one_editor=False)
    assert not other.passed
    assert "no other gated reading regressed, and every guardrail held" in other.verdict()
    assert (
        "regressed: "
        in {requirement.asks: requirement for requirement in other.requirements}[
            "no other gated reading regressed, and every guardrail held"
        ].reads
    )


def test_a_live_run_is_refused_by_name_before_its_directory_is_made(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _offline(monkeypatch)
    short = _shipped_copy(tmp_path / "short")
    (short / "writing.md").unlink()
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match=r"candidate skill short is missing writing\.md"):
        writer_acceptance.main(["--candidate-skill", str(short), "--cap-usd", "12", "--out", str(out)])
    assert not out.exists()
