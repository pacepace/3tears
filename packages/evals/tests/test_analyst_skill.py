"""The analyst skill as package data: whole, shipped in the wheel, digested, and its exemplars chosen by structure.

Contracts held here (``threetears.evals.analysis.analyst.skill``):

- **every file loads**: the shipped skill reads as an :class:`AnalystSkill` holding every file under ``skill/``, byte
  for byte; the analyst's text holds every module and the index, the editor's holds ``editor.md`` and nothing of the
  analyst's;
- **the wheel ships it**: a wheel built the way the release builds holds every file under ``skill/``;
- **one home**: ``docs/eval-report-kit/analyst/`` keeps only its pointer, and no file under ``docs/`` repeats a skill
  file;
- **the digest** is the same for the same files in any order and moves when a file's text or path does;
- **exemplars by structure**: each reference campaign reads as the structure its note's index row names and is given
  that note; a blind run's withheld note and its index row are left out; renaming a measure, changing the eval's kind,
  its design's shape or its name selects the same notes;
- **refusals**: a skill missing a module, a note the index does not name or names without a file, a structural fact
  code does not read, and a withheld note that does not exist are each refused by name.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import pytest

from packages.evals.tests.analyst_support import fixture_inputs
from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill
from threetears.evals.analysis.report_input import ReportInput

#: The package's root, and the workspace's: the release builds from the workspace root, whose ignore rules hatchling
#: reads, so the wheel test builds there too.
PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]

#: The skill as the source tree holds it.
SKILL = PACKAGE / "src" / "threetears" / "evals" / "analysis" / "analyst" / "skill"

#: Where the skill's files sit inside the wheel.
WHEEL_PREFIX = "threetears/evals/analysis/analyst/skill/"

#: The documentation directory the skill used to be read from, which now only points at the package.
DOCS_POINTER = REPO / "docs" / "eval-report-kit" / "analyst"

#: Each reference campaign's own note, and the structure its input reads as: what the note's index row must name.
REFERENCE_NOTES: dict[str, tuple[str, frozenset[str]]] = {
    "r1": (
        "r1-model-selection.md",
        frozenset(
            {"judged", "criteria", "slice_criteria", "tasks", "cross_cutting_task", "routable_tasks", "traffic_shares"}
        ),
    ),
    "r2": ("r2-config-sweep.md", frozenset({"criteria", "tasks", "traffic_shares", "factors", "monthly_volume"})),
    "r3": ("r3-classifier.md", frozenset({"criteria", "slice_criteria", "factors", "target_matrix", "monthly_volume"})),
    "r4": (
        "r4-tradeoff.md",
        frozenset(
            {
                "control",
                "judged",
                "criteria",
                "standing_criteria",
                "strata",
                "traffic_shares",
                "human_preference",
                "monthly_volume",
            }
        ),
    ),
    "r5": (
        "kairo-round2.md",
        frozenset({"control", "margin", "judged", "judge_uncalibrated", "late_criteria", "strata", "prior_insights"}),
    ),
}


def _source_files() -> dict[str, str]:
    return {
        path.relative_to(SKILL).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(SKILL.rglob("*"))
        if path.is_file()
    }


def _index_row(text: str, note: str) -> bool:
    return any(line.startswith(f"| `{note}` |") for line in text.splitlines())


# Every file loads


def test_the_shipped_skill_holds_every_file_under_skill_byte_for_byte() -> None:
    """``load_skill`` reads every file the source tree holds, and nothing else."""
    assert dict(load_skill().files) == _source_files()


def test_the_analyst_text_holds_every_module_and_the_index_and_the_editor_text_holds_only_its_own() -> None:
    """The analyst reads the role file and its five modules in the order it names them, then the index and notes."""
    skill = load_skill()
    analyst = skill.analyst(fixture_inputs()["r5"])
    modules = ("SKILL.md", "principles.md", "stories.md", "charts.md", "statistics.md", "writing.md")
    assert analyst.files[: len(modules) + 1] == (*modules, "exemplars/INDEX.md")
    assert analyst.files[len(modules) + 1 :] == tuple(f"exemplars/{note}" for note in analyst.exemplars)
    for path in analyst.files[: len(modules)]:
        assert f'<file path="{path}">\n{skill.files[path].strip()}\n</file>' in analyst.text
    assert "editor.md" not in analyst.files
    assert skill.files["editor.md"].strip() not in analyst.text

    editor = skill.editor()
    assert editor.files == ("editor.md",)
    assert editor.text == f'<file path="editor.md">\n{skill.files["editor.md"].strip()}\n</file>'
    assert editor.exemplars == () and editor.structure == frozenset()


def test_the_editor_text_carries_the_rubric_and_the_answer_shape() -> None:
    """The editor's text names all seven rubric items, the pass rule, both severities and the fenced answer."""
    text = load_skill().editor().text
    for item in (
        "First screen",
        "Coherent story",
        "Honest emphasis",
        "Sound elision",
        "Confounds handled",
        "Actionable next steps",
        "Evidence fits claims",
    ):
        assert f"**{item}.**" in text
    assert "no item scores 0 and the total is at least 11 of 14" in text
    assert "`must_fix`" in text and "`should_fix`" in text
    assert '"items"' in text and '"findings"' in text
    assert "critic" not in text.lower()


# The wheel ships it


def test_every_skill_file_reaches_a_built_wheel(tmp_path: Path) -> None:
    """A wheel built from the workspace root, as the release builds, holds every file under ``skill/``."""
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv is not available, so nothing can be built")
    built = subprocess.run(  # noqa: S603 - fixed argv, repo-local paths
        [uv, "build", "--package", "3tears-evals", "--wheel", "--out-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        cwd=REPO,
        check=False,
    )
    assert built.returncode == 0, f"the wheel would not build: {built.stderr}"
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {[w.name for w in wheels]}"
    with zipfile.ZipFile(wheels[0]) as wheel:
        shipped = {
            name.removeprefix(WHEEL_PREFIX): wheel.read(name).decode("utf-8")
            for name in wheel.namelist()
            if name.startswith(WHEEL_PREFIX)
        }
    assert shipped == _source_files(), "the wheel's skill/ is not the source tree's"


# One home


def test_the_docs_keep_only_a_pointer_to_the_package() -> None:
    """``docs/eval-report-kit/analyst/`` holds its README alone, and the README names the package's skill directory."""
    held = sorted(path.relative_to(DOCS_POINTER).as_posix() for path in DOCS_POINTER.rglob("*") if path.is_file())
    assert held == ["README.md"]
    assert "packages/evals/src/threetears/evals/analysis/analyst/skill/" in (DOCS_POINTER / "README.md").read_text(
        encoding="utf-8"
    )


def test_no_file_under_docs_repeats_a_skill_file() -> None:
    """No copy of a skill file sits anywhere under ``docs/``, where it would drift from the one the package ships."""
    texts = {text.strip() for text in _source_files().values()}
    copies = [
        path.relative_to(REPO).as_posix()
        for path in (REPO / "docs").rglob("*.md")
        if path.read_text(encoding="utf-8").strip() in texts
    ]
    assert not copies, f"docs repeat skill files: {copies}"


# The digest


def test_the_digest_is_the_same_for_the_same_files_in_any_order() -> None:
    """Two reads, and the same files given in reverse, digest alike; the digest is a full sha256."""
    skill = load_skill()
    reversed_files = dict(reversed(list(skill.files.items())))
    assert AnalystSkill(files=_source_files()).digest == skill.digest
    assert AnalystSkill(files=reversed_files).digest == skill.digest
    assert len(skill.digest) == 64 and int(skill.digest, 16) >= 0


def test_the_digest_moves_when_a_file_changes_or_moves() -> None:
    """An edit to one module, and a note's text under another path the index names, each give a new digest."""
    skill = load_skill()
    edited = dict(skill.files)
    edited["writing.md"] += "\nOne more line.\n"
    assert AnalystSkill(files=edited).digest != skill.digest

    note = "exemplars/r3-classifier.md"
    moved = {path: text for path, text in skill.files.items() if path != note}
    moved["exemplars/r3-classifier-moved.md"] = skill.files[note]
    moved["exemplars/INDEX.md"] = skill.files["exemplars/INDEX.md"].replace(
        "`r3-classifier.md`", "`r3-classifier-moved.md`"
    )
    assert AnalystSkill(files=moved).digest != skill.digest


def test_a_skill_text_digest_follows_its_text() -> None:
    """Two campaigns given different notes start from different texts, and so from different digests."""
    skill = load_skill()
    first, again = skill.analyst(fixture_inputs()["r5"]), skill.analyst(fixture_inputs()["r5"])
    other = skill.analyst(fixture_inputs()["r3"])
    assert first.digest == again.digest
    assert first.exemplars != other.exemplars and first.digest != other.digest


# Exemplars by structure


@pytest.mark.parametrize("name", sorted(REFERENCE_NOTES))
def test_each_reference_campaign_reads_as_its_notes_structure_and_is_given_that_note(name: str) -> None:
    """The structure code reads off each reference input is the one its note's index row names, so it is chosen."""
    note, structure = REFERENCE_NOTES[name]
    chosen = load_skill().analyst(fixture_inputs()[name])
    assert chosen.structure == structure
    assert note in chosen.exemplars
    assert len(chosen.exemplars) == 3
    index = load_skill().files["exemplars/INDEX.md"]
    row = next(line for line in index.splitlines() if line.startswith(f"| `{note}` |"))
    assert {fact.strip(" `") for fact in row.strip().strip("|").split("|")[-1].split(",")} == structure


@pytest.mark.parametrize("name", sorted(REFERENCE_NOTES))
def test_a_withheld_note_and_its_index_row_are_left_out(name: str) -> None:
    """A blind run withholds its campaign's own note: neither the note nor its index row reaches the analyst."""
    note, _structure = REFERENCE_NOTES[name]
    skill = load_skill()
    blind = skill.analyst(fixture_inputs()[name], withhold=[note])
    assert note not in blind.exemplars and len(blind.exemplars) == 3
    assert f"exemplars/{note}" not in blind.files
    assert not _index_row(blind.text, note)
    assert skill.files[f"exemplars/{note}"].strip() not in blind.text
    for given in blind.exemplars:
        assert _index_row(blind.text, given)


def test_the_index_rows_of_notes_not_given_are_left_out_and_its_prose_kept() -> None:
    """The analyst reads the index's guidance whole, and a row only for each note it is given."""
    skill = load_skill()
    chosen = skill.analyst(fixture_inputs()["r3"])
    for note in (
        "kairo-round2.md",
        "r1-model-selection.md",
        "r2-config-sweep.md",
        "r3-classifier.md",
        "r4-tradeoff.md",
    ):
        assert _index_row(chosen.text, note) == (note in chosen.exemplars)
    assert "**Don't copy an exemplar's structure.**" in chosen.text


def _renamed(node: Any, old: str, new: str) -> Any:
    if isinstance(node, dict):
        return {(new if key == old else key): _renamed(value, old, new) for key, value in node.items()}
    if isinstance(node, list):
        return [_renamed(item, old, new) for item in node]
    return new if node == old else node


@pytest.mark.parametrize("name", sorted(REFERENCE_NOTES))
def test_renaming_a_measure_or_changing_the_kind_shape_or_name_selects_the_same(name: str) -> None:
    """Selection reads structure alone: the same input under another kind, shape, name and measure key (one a criterion
    reads, where one does) is given the same notes for the same reason."""
    held = fixture_inputs()[name]
    document: dict[str, Any] = json.loads(held.model_dump_json(by_alias=True))
    gated = [criterion["measure"] for criterion in document["criteria"] if criterion["measure"] in document["measures"]]
    measure = next(iter(gated), next(iter(document["measures"])))
    document = _renamed(document, measure, "renamed_measure")
    document["meta"] |= {"kind": "something_else", "design_shape": "unheard_of", "campaign": "another name"}
    moved = ReportInput.model_validate(document)
    assert "renamed_measure" in moved.measures and moved.meta.kind == "something_else"

    skill = load_skill()
    before, after = skill.analyst(held), skill.analyst(moved)
    assert (after.exemplars, after.structure) == (before.exemplars, before.structure)


def test_a_campaign_sharing_no_structure_is_still_given_notes() -> None:
    """An eval that fits no exemplar is normal: it is given the first notes in index order, never none."""
    held = fixture_inputs()["r3"]
    document: dict[str, Any] = json.loads(held.model_dump_json(by_alias=True))
    document |= {"criteria": [], "factors": [], "configs": [], "target_matrix": None, "cells": {}, "confusion": {}}
    document["meta"]["volume_per_month"] = None
    bare = load_skill().analyst(ReportInput.model_validate(document))
    assert bare.structure == frozenset()
    assert bare.exemplars == ("kairo-round2.md", "r1-model-selection.md", "r2-config-sweep.md")


# Refusals


def test_a_skill_missing_a_module_is_refused_by_name() -> None:
    files = dict(load_skill().files)
    del files["statistics.md"]
    with pytest.raises(ValueError, match=r"missing statistics\.md"):
        AnalystSkill(files=files)


def test_a_note_without_an_index_row_is_refused() -> None:
    files = dict(load_skill().files)
    files["exemplars/r6-new.md"] = "# Reasoning note: a new campaign\n"
    with pytest.raises(ValueError, match=r"no row for r6-new\.md"):
        AnalystSkill(files=files)


def test_an_index_row_without_its_note_is_refused() -> None:
    files = dict(load_skill().files)
    del files["exemplars/r4-tradeoff.md"]
    with pytest.raises(ValueError, match=r"names notes that are not in the skill: r4-tradeoff\.md"):
        AnalystSkill(files=files)


def test_a_structural_fact_code_does_not_read_is_refused() -> None:
    files = dict(load_skill().files)
    files["exemplars/INDEX.md"] = files["exemplars/INDEX.md"].replace("`target_matrix`", "`classifier`")
    with pytest.raises(ValueError, match=r"r3-classifier\.md structure code does not read: classifier"):
        AnalystSkill(files=files)


def test_withholding_a_note_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ValueError, match=r"no exemplar is named r9\.md"):
        load_skill().analyst(fixture_inputs()["r1"], withhold=["r9.md"])
