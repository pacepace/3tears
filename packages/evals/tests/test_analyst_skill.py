"""The analyst skill as package data: whole, shipped in the wheel, digested, and every exemplar given to the analyst.

Contracts held here (``threetears.evals.analysis.analyst.skill``):

- **every file loads**: the shipped skill reads as an :class:`AnalystSkill` holding every Markdown file under
  ``skill/``, byte for byte, and nothing else there; the analyst's text holds every module and the index, the editor's holds ``editor.md`` and nothing of the
  analyst's;
- **the wheel ships it**: a wheel built the way the release builds holds every file under ``skill/``;
- **one home**: ``docs/eval-report-kit/analyst/`` keeps only its pointer, and no file under ``docs/`` repeats a
  paragraph of a skill file, whole or embedded;
- **the digest** is the same for the same files in any order and moves when a file's text or path does;
- **every note**: the analyst is given the whole index and every note, in index order, and nothing about a campaign
  enters the choice; a blind run's withheld note and its index row are left out; the index carries no maintainer's
  instruction, which ``README.md`` beside the skill holds, and no system text holds that README;
- **refusals**: a skill missing a module, a note the index does not name or names without a file, a withheld note
  that does not exist, and a single name passed where a collection is expected are each refused by name.
"""

from __future__ import annotations

import inspect
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from threetears.evals.analysis.analyst.skill import AnalystSkill, load_skill

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

#: Every note the skill ships, in index order.
NOTES = ("kairo-round2.md", "r1-model-selection.md", "r2-config-sweep.md", "r3-classifier.md", "r4-tradeoff.md")

#: The skill's role file and the modules it names, in the order the analyst reads them.
MODULES = ("SKILL.md", "principles.md", "stories.md", "charts.md", "statistics.md", "writing.md")


def _source_files() -> dict[str, str]:
    return {
        path.relative_to(SKILL).as_posix(): path.read_text(encoding="utf-8")
        for path in sorted(SKILL.rglob("*.md"))
        if path.is_file()
    }


def _index_row(text: str, note: str) -> bool:
    return any(line.startswith(f"| `{note}` |") for line in text.splitlines())


# Every file loads


def test_the_shipped_skill_holds_every_markdown_file_under_skill_byte_for_byte() -> None:
    """``load_skill`` reads every Markdown file the source tree holds, and nothing else."""
    assert dict(load_skill().files) == _source_files()


def test_a_file_that_is_not_markdown_is_never_read(tmp_path: Path) -> None:
    """A stray ``.DS_Store`` (bytes no UTF-8 reader takes) at the skill's root or among its notes is not part of it.

    The package is copied with both strays added and the skill loaded from the copy in a fresh interpreter, so what
    is tested is the loader reading the package, through its front door."""
    evals = PACKAGE / "src" / "threetears" / "evals"
    copy = tmp_path / "threetears" / "evals"
    shutil.copytree(evals, copy, ignore=shutil.ignore_patterns("__pycache__"))
    for directory in (copy / "analysis" / "analyst" / "skill", copy / "analysis" / "analyst" / "skill" / "exemplars"):
        (directory / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1\xff\xfe")
    probe = (
        "import threetears.evals as e, sys; "
        "from threetears.evals.analysis.analyst.skill import load_skill; "
        "print(e.__file__); print(load_skill().digest)"
    )
    loaded = subprocess.run(  # noqa: S603 - fixed argv, a copy of this package
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        check=False,
    )
    assert loaded.returncode == 0, f"the skill would not load beside a stray file: {loaded.stderr}"
    where, digest = loaded.stdout.split()
    assert Path(where).is_relative_to(tmp_path), f"the probe loaded {where}, not the copy"
    assert digest == load_skill().digest


def test_the_analyst_text_holds_every_module_and_the_index_and_the_editor_text_holds_only_its_own() -> None:
    """The analyst reads the role file and its five modules in the order it names them, then the index and notes."""
    skill = load_skill()
    analyst = skill.analyst()
    modules = MODULES
    assert analyst.files[: len(modules) + 1] == (*modules, "exemplars/INDEX.md")
    assert analyst.files[len(modules) + 1 :] == tuple(f"exemplars/{note}" for note in analyst.exemplars)
    for path in analyst.files[: len(modules)]:
        assert f'<file path="{path}">\n{skill.files[path].strip()}\n</file>' in analyst.text
    assert "editor.md" not in analyst.files
    assert skill.files["editor.md"].strip() not in analyst.text

    editor = skill.editor()
    assert editor.files == ("editor.md",)
    assert editor.text == f'<file path="editor.md">\n{skill.files["editor.md"].strip()}\n</file>'
    assert editor.exemplars == ()


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
    """No paragraph of a skill file sits anywhere under ``docs/``, whole or embedded in a longer page, where it would
    drift from the one the package ships."""
    paragraphs = {
        paragraph.strip(): path
        for path, text in _source_files().items()
        for paragraph in text.split("\n\n")
        if len(paragraph.strip()) >= 120
    }
    copies = sorted(
        f"{page.relative_to(REPO).as_posix()} repeats {path}"
        for page in (REPO / "docs").rglob("*.md")
        for text in [page.read_text(encoding="utf-8")]
        for paragraph, path in paragraphs.items()
        if paragraph in text
    )
    assert not copies, "docs repeat skill text; point at the package instead:\n" + "\n".join(sorted(set(copies)))


# The Kairo note's correction (W4 skill round 2)


def test_the_kairo_note_no_longer_teaches_that_the_caveats_leave_measured_cost_untouched() -> None:
    """The note once said the call "rests on measured cost, which none of the caveats touch", the error its own run's
    editor raised: the snapshot attaches the reasoning-ratio confound to 1017's cost contrast too. The note now carries
    the confound on that cost, with its likely direction, and Watch names it beside both of 1017's readings."""
    note = load_skill().files["exemplars/kairo-round2.md"]

    assert "none of the caveats touch" not in note
    assert "27–28%" not in note, "a percent read off unpaired means beside a claim about paired gaps"
    assert "1017's cost contrast" in note and "would likely cost more" in note
    assert "beside both of 1017's readings" in note


# Cost belongs to the provider routing (owner, 2026-10-25, restated 2026-10-29)


def test_the_analyst_and_the_editor_carry_that_a_cost_is_the_provider_routings() -> None:
    """The analyst's principles say a cost comparison names who served each model and never ranks models on cost alone;
    the editor's rubric makes ranking models by cost with no provider named a must_fix, beside the fact that says who
    served each."""
    skill = load_skill()
    principles = skill.files["principles.md"]
    assert "**A cost belongs to the provider routing, not the model.**" in principles
    assert "names the provider each was served by" in principles and "`served_by`" in principles
    assert "never ranks models on cost alone" in principles

    editor = skill.editor().text
    must_fix = editor[editor.index("These are always must_fix:") : editor.index("- **`should_fix`**")]
    assert "a claim that ranks models by cost with no provider named" in must_fix
    assert "`cost_served_by`" in must_fix


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
    """The same notes give the same text and digest; withholding one gives another text, and another digest."""
    skill = load_skill()
    first, again = skill.analyst(), skill.analyst()
    blind = skill.analyst(withhold=["r3-classifier.md"])
    assert first.digest == again.digest
    assert first.digest != blind.digest


# Every note


def test_the_analyst_is_given_the_whole_index_and_every_note_in_index_order() -> None:
    """With nothing withheld, every note the index lists reaches the analyst, each with its row, as the prototypes read
    them."""
    skill = load_skill()
    given = skill.analyst()
    assert given.exemplars == NOTES
    assert given.files == (*MODULES, "exemplars/INDEX.md", *(f"exemplars/{note}" for note in NOTES))
    assert f'<file path="exemplars/INDEX.md">\n{skill.files["exemplars/INDEX.md"].strip()}\n</file>' in given.text
    for note in NOTES:
        assert _index_row(given.text, note)
        assert f'<file path="exemplars/{note}">\n{skill.files[f"exemplars/{note}"].strip()}\n</file>' in given.text


def test_nothing_about_a_campaign_chooses_its_notes() -> None:
    """The analyst's text takes no campaign at all, only the notes to withhold: whatever a campaign's kind, design,
    names or measures, it is given the same notes, so no condition on what kind of eval it is can choose them."""
    skill = load_skill()
    assert list(inspect.signature(skill.analyst).parameters) == ["withhold"]
    assert skill.analyst() == skill.analyst()


@pytest.mark.parametrize("note", NOTES)
def test_a_withheld_note_and_its_index_row_are_left_out(note: str) -> None:
    """A blind run withholds its campaign's own note: neither the note nor its index row reaches the analyst, and every
    other note does."""
    skill = load_skill()
    blind = skill.analyst(withhold=[note])
    assert blind.exemplars == tuple(other for other in NOTES if other != note)
    assert f"exemplars/{note}" not in blind.files
    assert not _index_row(blind.text, note)
    assert skill.files[f"exemplars/{note}"].strip() not in blind.text
    for given in blind.exemplars:
        assert _index_row(blind.text, given)
    assert "**Don't copy an exemplar's structure.**" in blind.text


def test_the_index_carries_no_maintainers_instruction_and_no_system_text_holds_the_readme() -> None:
    """How to add a note, and what the Structure column is for, are the maintainers'; they sit in ``README.md`` beside
    the skill, which neither role reads."""
    skill = load_skill()
    assert "Add a note" not in skill.files["exemplars/INDEX.md"]
    assert "Add one whenever a report teaches something new" in skill.files["README.md"]
    for text in (skill.analyst().text, skill.editor().text):
        assert "README.md" not in text and skill.files["README.md"].strip() not in text


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


def test_withholding_a_note_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ValueError, match=r"no exemplar is named r9\.md"):
        load_skill().analyst(withhold=["r9.md"])


def test_withholding_one_name_given_as_a_bare_string_is_refused_by_its_type() -> None:
    """A single file name where a collection belongs is refused saying so, never split into its characters."""
    with pytest.raises(TypeError, match=r"not a str: pass \['kairo-round2\.md'\]"):
        load_skill().analyst(withhold="kairo-round2.md")
