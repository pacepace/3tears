"""The analyst skill, as the package ships it: the text an analyst's and an editor's sessions start from.

The skill is package data under ``skill/`` beside this module, so it ships in the wheel and every install reads the
same files: ``SKILL.md`` (the role, the workflow, the report document and the tools), the five modules it names
(``principles.md``, ``stories.md``, ``charts.md``, ``statistics.md``, ``writing.md``), the reasoning notes under
``exemplars/`` with their ``INDEX.md``, and ``editor.md``, the report editor's own text. :func:`load_skill` reads them
into an :class:`AnalystSkill`, which checks they are whole and assembles each role's system text as a
:class:`SkillText`.

**An analyst is given every note**, with the whole index, but for those a run withholds: a blind run withholds the
note reasoned on its own campaign (or, for a variant, on the campaign it was built from), and the note's index row with
it, since a row tells the note's story. Nothing about a campaign chooses which notes it reads, so no condition on what
kind of eval it is, its design or its names can reach the skill text. Each index row's Structure cell records the
campaign its note was reasoned on, for a reader of the notes; code does not read it.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from functools import cache, cached_property
from importlib.resources.abc import Traversable
from importlib.resources import files
from types import MappingProxyType

from threetears.evals.schema.hashing import canonical_digest

#: The skill's role file, then the modules it tells the analyst to read with it, in the order it names them.
_MODULES: tuple[str, ...] = (
    "SKILL.md",
    "principles.md",
    "stories.md",
    "charts.md",
    "statistics.md",
    "writing.md",
)

#: The exemplars' index: its preamble, one table row per note, and what follows the table.
_INDEX = "exemplars/INDEX.md"

#: The report editor's system text.
_EDITOR = "editor.md"

#: Where the reasoning notes sit, relative to the skill's root.
_NOTES = "exemplars/"

#: A note's row in the index: a table row whose first cell names the note's file in backticks.
_ROW = re.compile(r"^\|\s*`(?P<note>[^`]+\.md)`\s*\|")


@dataclass(frozen=True)
class SkillText:
    """One role's system text, assembled from the skill's files, and what it was assembled from.

    :ivar text: The system text, each file in a ``<file path="...">`` element, in reading order.
    :ivar files: The skill-relative path of each file the text holds, in the order it holds them.
    :ivar exemplars: The notes the text holds, by file name, in index order; empty for the editor's.
    :ivar digest: The sha256 of :attr:`text`'s canonical JSON: which text a session started from.
    """

    text: str
    files: tuple[str, ...]
    exemplars: tuple[str, ...] = ()
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", canonical_digest(self.text))


@dataclass(frozen=True)
class AnalystSkill:
    """The analyst skill's files, checked whole, and each role's system text built from them.

    Constructing one refuses a skill that is not whole: a module, the index or the editor's text missing; a note the
    index names that is not there, or a note there that the index does not name.

    :ivar files: Each file's text, by its path relative to the skill's root (``exemplars/INDEX.md``).
    """

    files: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))
        missing = [path for path in (*_MODULES, _INDEX, _EDITOR) if path not in self.files]
        if missing:
            raise ValueError(
                f"the analyst skill is missing {', '.join(missing)}: every module, the index and editor.md"
            )
        listed = set(self._notes)
        present = {path.removeprefix(_NOTES) for path in self.files if path.startswith(_NOTES) and path != _INDEX}
        if absent := sorted(listed - present):
            raise ValueError(f"exemplars/INDEX.md names notes that are not in the skill: {', '.join(absent)}")
        if unlisted := sorted(present - listed):
            raise ValueError(
                f"exemplars/INDEX.md has no row for {', '.join(unlisted)}: give each note a row and its Structure"
            )

    @cached_property
    def _notes(self) -> tuple[str, ...]:
        """Each note the index has a row for, in index order."""
        return tuple(row["note"] for line in self.files.get(_INDEX, "").splitlines() if (row := _ROW.match(line)))

    @cached_property
    def digest(self) -> str:
        """The sha256 of every file's path and text, as canonical JSON: which skill a report was written with.

        :return: A 64-character lowercase hex digest, the same for the same files in any order.
        :rtype: str
        """
        return canonical_digest(dict(self.files))

    def analyst(self, *, withhold: Collection[str] = ()) -> SkillText:
        """The analyst's system text: the role file and its modules, then the index and every note not withheld.

        :param withhold: Notes, by file name, the analyst must not be given, as a blind run withholds the note
            reasoned on its own campaign. The index drops their rows too, since a row tells a note's story.
        :ptype withhold: Collection[str]
        :return: The text, and the notes it holds.
        :rtype: SkillText
        :raises TypeError: When ``withhold`` is a single string rather than a collection of names.
        :raises ValueError: When ``withhold`` names a note the index has no row for.
        """
        if isinstance(withhold, str):
            raise TypeError(
                f"withhold takes a collection of note file names, not a str: pass [{withhold!r}], not {withhold!r}"
            )
        if unknown := sorted(set(withhold) - set(self._notes)):
            raise ValueError(f"no exemplar is named {', '.join(unknown)}; the notes are {', '.join(self._notes)}")
        given = tuple(note for note in self._notes if note not in withhold)
        paths = (*_MODULES, _INDEX, *(_NOTES + note for note in given))
        texts = [*(self.files[path] for path in _MODULES), self._index_of(given)]
        texts += [self.files[_NOTES + note] for note in given]
        return SkillText(text=_assemble(paths, texts), files=paths, exemplars=given)

    def editor(self) -> SkillText:
        """The report editor's system text: ``editor.md``, the rubric and the answer it returns.

        :return: The text, from ``editor.md`` alone.
        :rtype: SkillText
        """
        return SkillText(text=_assemble((_EDITOR,), (self.files[_EDITOR],)), files=(_EDITOR,))

    def _index_of(self, given: Collection[str]) -> str:
        """The index as an analyst reads it: every line but the rows of notes it is not given."""
        lines = self.files[_INDEX].splitlines()
        kept = [line for line in lines if (row := _ROW.match(line)) is None or row["note"] in given]
        return "\n".join(kept)


def _assemble(paths: Collection[str], texts: Collection[str]) -> str:
    """Files as one system text: each in a ``<file>`` element naming its path, so a file's name in another reads."""
    return "\n\n".join(
        f'<file path="{path}">\n{text.strip()}\n</file>' for path, text in zip(paths, texts, strict=True)
    )


def _read_tree(root: Traversable, prefix: str = "") -> dict[str, str]:
    """Every Markdown file under ``root``, by its path relative to it: the skill is Markdown, and nothing else a
    checkout may hold there (a ``.DS_Store``, an editor's swap file) is read as part of it."""
    read: dict[str, str] = {}
    for entry in root.iterdir():
        if entry.is_dir():
            read.update(_read_tree(entry, f"{prefix}{entry.name}/"))
        elif entry.is_file() and entry.name.endswith(".md"):
            read[f"{prefix}{entry.name}"] = entry.read_text(encoding="utf-8")
    return read


@cache
def load_skill() -> AnalystSkill:
    """The analyst skill this build ships, read once from the package's ``skill/`` data.

    :return: The skill, checked whole.
    :rtype: AnalystSkill
    :raises ValueError: When the shipped skill is not whole (see :class:`AnalystSkill`).
    """
    return AnalystSkill(files=_read_tree(files("threetears.evals.analysis.analyst").joinpath("skill")))


#: The names with a production caller: the report writer's sessions take their system text as a ``SkillText``.
#: ``AnalystSkill`` and ``load_skill`` join it with their first production caller, the report writer's entry point
#: (``tests/test_analyst_skill_exports.py`` holds the list).
__all__: list[str] = ["SkillText"]
