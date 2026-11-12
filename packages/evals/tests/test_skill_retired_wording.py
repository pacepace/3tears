"""Every analyst skill this package ships teaches the ruled wording for an undecided difference, and nothing else.

The owner retired "not separated" from everything the package prints: a sentence says "the evidence cannot tell A and B
apart", a table cell "cannot tell apart". The renderer prints those words, and a report the writer drafts sits beside
them, so a skill that teaches "separated" puts the retired words back on the page. Held here on every Markdown file of
every skill the package ships: the shipped skill, each candidate skill under ``fixtures/writer_cases/candidates/`` and
each example's skills under ``examples/*/skills/``.

A skill names the retired words only to forbid them, as one quoted list (:data:`AVOIDED`): where it defines the ruled
phrase (``statistics.md``), where it lists terms to define or avoid (``writing.md``) and where the editor raises a
draft that uses them (``editor.md``). Any other form of "separated" fails, naming its file and line.

One directory is left out, by name: rung 5's ``skills/control/``, a frozen copy of the skill shipped on 2026-11-03.
Every rung-5 writer recording is keyed by its digest (and the lead-first arm's, which is built from it), so a word
changed there leaves all 14 reports unreplayable until they are recorded again at a model's price.

Mutations that turn this file red: a skill file writing "not separated", "nothing separates" or "separated from
zero" anywhere outside the quoted list; the list dropped from the definition, the terms rule or the editor's rules; a
skill directory added under candidates or an example without being scanned.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Final

PACKAGE: Final = Path(__file__).resolve().parents[1]

#: The shipped skill's directory, as the source tree holds it.
SHIPPED: Final = PACKAGE / "src" / "threetears" / "evals" / "analysis" / "analyst" / "skill"

#: The candidate skills the acceptance script reads by ``--candidate-skill``.
CANDIDATES: Final = PACKAGE / "tests" / "fixtures" / "writer_cases" / "candidates"

#: Rung 5's frozen control: its recordings are keyed by its digest, so its words stay as recorded.
FROZEN: Final = PACKAGE / "examples" / "5-writer" / "skills" / "control"

#: The statistical verb in any form: "separated", "separates", "separating". "Separate charts" and "separately" are
#: other words.
RETIRED: Final = re.compile(r"\bseparat(?:ed|es|ing|ion)\b", re.IGNORECASE)

#: The one form a skill names the retired words in: quoted, as the words never to write.
AVOIDED: Final = '"not separated", "nothing separated" or "separated from zero"'

#: The files that name the retired words to forbid them, in each skill that holds them.
NAMING_FILES: Final = ("editor.md", "statistics.md", "writing.md")


def _skill_files() -> Iterator[Path]:
    """Every Markdown file of every skill the package ships, the frozen control's aside."""
    yield from SHIPPED.rglob("*.md")
    yield from CANDIDATES.rglob("*.md")
    for skills in sorted((PACKAGE / "examples").glob("*/skills")):
        yield from (path for path in skills.rglob("*.md") if FROZEN not in path.parents)


def test_no_skill_file_writes_separated_outside_the_quoted_list_of_retired_words() -> None:
    files = sorted(_skill_files())
    hits = [
        f"{path.relative_to(PACKAGE)}:{number}: {match.group(0)!r}"
        for path in files
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        for match in RETIRED.finditer(line.replace(AVOIDED, ""))
    ]
    assert hits == [], "a skill teaches the retired wording; write 'the evidence cannot tell A and B apart':\n" + (
        "\n".join(hits)
    )


def test_every_skill_with_the_definition_forbids_the_retired_words_where_it_defines_avoids_and_edits() -> None:
    skills = [SHIPPED, *sorted(path for path in CANDIDATES.iterdir() if path.is_dir())]
    for skill in skills:
        for name in NAMING_FILES:
            text = (skill / name).read_text(encoding="utf-8")
            assert AVOIDED in text, f"{skill.relative_to(PACKAGE)}/{name} does not name the retired words to avoid"
            assert "cannot tell" in text and "apart" in text, (
                f"{skill.relative_to(PACKAGE)}/{name} lacks the ruled phrase"
            )


def test_the_scan_reads_every_skill_directory_and_leaves_out_only_the_frozen_control() -> None:
    files = set(_skill_files())
    assert SHIPPED / "statistics.md" in files
    assert CANDIDATES / "headline-answers" / "statistics.md" in files
    assert PACKAGE / "examples" / "5-writer" / "skills" / "lead-first.md" in files
    assert not any(FROZEN in path.parents for path in files)
    assert (FROZEN / "statistics.md").is_file(), "the frozen control moved; point FROZEN at it"


def test_the_pattern_reads_the_retired_verb_and_none_of_its_neighbours() -> None:
    for retired in ("not separated", "nothing separates", "Separated from zero", "separating the arms"):
        assert RETIRED.search(retired), retired
    for kept in ("separate charts", "show it separately", "cannot tell apart"):
        assert not RETIRED.search(kept), kept
