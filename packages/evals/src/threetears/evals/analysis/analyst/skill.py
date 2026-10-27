"""The analyst skill, as the package ships it: the text an analyst's and an editor's sessions start from.

The skill is package data under ``skill/`` beside this module, so it ships in the wheel and every install reads the
same files: ``SKILL.md`` (the role, the workflow, the report document and the tools), the five modules it names
(``principles.md``, ``stories.md``, ``charts.md``, ``statistics.md``, ``writing.md``), the reasoning notes under
``exemplars/`` with their ``INDEX.md``, and ``editor.md``, the report editor's own text. :func:`load_skill` reads them
into an :class:`AnalystSkill`, which checks they are whole and assembles each role's system text as a
:class:`SkillText`.

**Which exemplars an analyst is given** is read off the structure of the campaign's report input, never off what kind
of eval it is, its design's shape, its name or its measures' names. :data:`_STRUCTURE` names each structural fact code
reads, each a condition on data present in the input:

=========================  =====================================================================================
``control``                an arm is the control
``margin``                 an aspect declares a margin
``judged``                 the input holds a judged dimension
``judge_uncalibrated``     it holds judged dimensions, and none was calibrated on human ratings
``late_criteria``          something was set after the results were visible
``criteria``               the campaign declared criteria
``standing_criteria``      a criterion is standing policy rather than this campaign's
``slice_criteria``         a criterion applies within each task or cell, or within one task
``tasks``                  the input names tasks
``cross_cutting_task``     a task occurs inside every other task
``routable_tasks``         a task's traffic can be routed to its own arm
``strata``                 the input slices cases into strata
``traffic_shares``         a task or stratum states its share of traffic
``factors``                the arms are levels of a grid of settings
``target_matrix``          the cases are laid out on a target matrix of cells
``human_preference``       the input holds blind human ratings
``monthly_volume``         the campaign states its volume per month
``prior_insights``         the campaign carries insights from earlier campaigns
=========================  =====================================================================================

Each note's row in ``exemplars/INDEX.md`` names the structure of the campaign it was reasoned on, in those names. A
campaign is given the :data:`_EXEMPLARS` notes whose structure it shares most (shared facts over the facts either
holds), ties in INDEX order. The ranking is one computation over sets, the same for every campaign: no fact is weighed
above another, and no note is reached for by name. A campaign that shares nothing with any note is still given notes,
since an eval that fits no exemplar is normal and the notes teach the thinking, not a structure. Which notes reach an
analyst changes only what it reads first; it never selects a part of the report.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cache, cached_property
from importlib.resources.abc import Traversable
from importlib.resources import files
from types import MappingProxyType

from threetears.evals.analysis.report_input import ReportInput
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

#: How many notes an analyst is given: enough to see the thinking from more than one campaign, few enough that none
#: reads as the template.
_EXEMPLARS = 3

#: A note's row in the index: a table row whose first cell names the note's file in backticks.
_ROW = re.compile(r"^\|\s*`(?P<note>[^`]+\.md)`\s*\|")

#: A name in backticks, as the index's Structure cell lists each fact.
_TICKED = re.compile(r"`([^`]+)`")


def _judge_uncalibrated(report_input: ReportInput) -> bool:
    judged = report_input.judged.values()
    return bool(report_input.judged) and not any(dimension.calibration_ratings for dimension in judged)


def _slice_criteria(report_input: ReportInput) -> bool:
    return any(c.scope != "overall" or c.task is not None for c in report_input.criteria)


def _traffic_shares(report_input: ReportInput) -> bool:
    return any(t.share for t in report_input.tasks) or any(s.share for s in report_input.strata)


#: Each structural fact an exemplar is chosen by, as a condition on the data a report input holds. None reads the
#: eval's kind, its design's shape, a name or a measure key.
_STRUCTURE: Mapping[str, Callable[[ReportInput], bool]] = MappingProxyType(
    {
        "control": lambda r: any(arm.role == "control" for arm in r.arms),
        "margin": lambda r: any(aspect.margin is not None for aspect in r.aspects),
        "judged": lambda r: bool(r.judged),
        "judge_uncalibrated": _judge_uncalibrated,
        "late_criteria": lambda r: bool(r.caveats.late_criteria),
        "criteria": lambda r: bool(r.criteria),
        "standing_criteria": lambda r: any(c.standing for c in r.criteria),
        "slice_criteria": _slice_criteria,
        "tasks": lambda r: bool(r.tasks),
        "cross_cutting_task": lambda r: any(t.cross_cutting for t in r.tasks),
        "routable_tasks": lambda r: any(t.routable for t in r.tasks),
        "strata": lambda r: bool(r.strata),
        "traffic_shares": _traffic_shares,
        "factors": lambda r: bool(r.factors),
        "target_matrix": lambda r: r.target_matrix is not None,
        "human_preference": lambda r: r.human_pref is not None,
        "monthly_volume": lambda r: r.meta.volume_per_month is not None,
        "prior_insights": lambda r: bool(r.prior_insights),
    }
)


@dataclass(frozen=True)
class SkillText:
    """One role's system text, assembled from the skill's files, and what it was assembled from.

    :ivar text: The system text, each file in a ``<file path="...">`` element, in reading order.
    :ivar files: The skill-relative path of each file the text holds, in the order it holds them.
    :ivar exemplars: The notes the text holds, by file name; empty for the editor's.
    :ivar structure: The structural facts the notes were chosen by, as :data:`_STRUCTURE` names them; empty for the
        editor's.
    :ivar digest: The sha256 of :attr:`text`'s canonical JSON: which text a session started from.
    """

    text: str
    files: tuple[str, ...]
    exemplars: tuple[str, ...] = ()
    structure: frozenset[str] = frozenset()
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "digest", canonical_digest(self.text))


@dataclass(frozen=True)
class AnalystSkill:
    """The analyst skill's files, checked whole, and each role's system text built from them.

    Constructing one refuses a skill that is not whole: a module, the index or the editor's text missing; a note the
    index names that is not there, or a note there that the index does not name; a structural fact the index names
    that code does not read.

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
        for note, facts in self._notes.items():
            if unknown := sorted(facts - _STRUCTURE.keys()):
                raise ValueError(
                    f"exemplars/INDEX.md gives {note} structure code does not read: {', '.join(unknown)}; "
                    f"the facts are {', '.join(_STRUCTURE)}"
                )

    @cached_property
    def _notes(self) -> Mapping[str, frozenset[str]]:
        """Each note the index has a row for, in index order, with the structural facts its Structure cell names."""
        notes: dict[str, frozenset[str]] = {}
        for line in self.files.get(_INDEX, "").splitlines():
            if row := _ROW.match(line):
                cells = line.strip().strip("|").split("|")
                notes[row["note"]] = frozenset(_TICKED.findall(cells[-1]))
        return MappingProxyType(notes)

    @cached_property
    def digest(self) -> str:
        """The sha256 of every file's path and text, as canonical JSON: which skill a report was written with.

        :return: A 64-character lowercase hex digest, the same for the same files in any order.
        :rtype: str
        """
        return canonical_digest(dict(self.files))

    def analyst(self, report_input: ReportInput, *, withhold: Collection[str] = ()) -> SkillText:
        """The analyst's system text for one campaign: the skill's modules, then the notes its structure selects.

        :param report_input: The campaign's frozen report input, whose structure selects the notes.
        :ptype report_input: ReportInput
        :param withhold: Notes, by file name, the analyst must not be given, as a blind run withholds the note
            reasoned on the same campaign. The index drops their rows too, since a row tells a note's story.
        :ptype withhold: Collection[str]
        :return: The text, the notes it holds and the structure that chose them.
        :rtype: SkillText
        :raises ValueError: When ``withhold`` names a note the index has no row for.
        """
        if unknown := sorted(set(withhold) - self._notes.keys()):
            raise ValueError(f"no exemplar is named {', '.join(unknown)}; the notes are {', '.join(self._notes)}")
        structure = frozenset(name for name, holds in _STRUCTURE.items() if holds(report_input))
        candidates = [note for note in self._notes if note not in withhold]

        def shared(note: str) -> Fraction:
            union = structure | self._notes[note]
            return Fraction(len(structure & self._notes[note]), len(union)) if union else Fraction(0)

        ranked = sorted(candidates, key=lambda note: -shared(note))
        chosen = tuple(note for note in candidates if note in ranked[:_EXEMPLARS])
        paths = (*_MODULES, _INDEX, *(_NOTES + note for note in chosen))
        texts = [*(self.files[path] for path in _MODULES), self._index_of(chosen)]
        texts += [self.files[_NOTES + note] for note in chosen]
        return SkillText(text=_assemble(paths, texts), files=paths, exemplars=chosen, structure=structure)

    def editor(self) -> SkillText:
        """The report editor's system text: ``editor.md``, the rubric and the answer it returns.

        :return: The text, from ``editor.md`` alone.
        :rtype: SkillText
        """
        return SkillText(text=_assemble((_EDITOR,), (self.files[_EDITOR],)), files=(_EDITOR,))

    def _index_of(self, chosen: Collection[str]) -> str:
        """The index as an analyst reads it: every line but the rows of notes it is not given."""
        lines = self.files[_INDEX].splitlines()
        kept = [line for line in lines if (row := _ROW.match(line)) is None or row["note"] in chosen]
        return "\n".join(kept)


def _assemble(paths: Collection[str], texts: Collection[str]) -> str:
    """Files as one system text: each in a ``<file>`` element naming its path, so a file's name in another reads."""
    return "\n\n".join(
        f'<file path="{path}">\n{text.strip()}\n</file>' for path, text in zip(paths, texts, strict=True)
    )


def _read_tree(root: Traversable, prefix: str = "") -> dict[str, str]:
    """Every file under ``root``, by its path relative to it."""
    read: dict[str, str] = {}
    for entry in root.iterdir():
        if entry.is_dir():
            read.update(_read_tree(entry, f"{prefix}{entry.name}/"))
        elif entry.is_file():
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


__all__ = ["AnalystSkill", "SkillText", "load_skill"]
