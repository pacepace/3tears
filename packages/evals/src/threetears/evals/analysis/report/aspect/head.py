"""The first screen's lines above the scoreboard: the question, the arms line and the floors a candidate breaks."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle
from threetears.evals.analysis.report.aspect.common import _control_key, _date, _floor_verdicts, _namer
from threetears.evals.analysis.report.campaign.warnings import floor_line
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.model import DisclosureBlock, ReportBlock, ReportSource
from threetears.evals.analysis.report.unit_once import interval_beside
from threetears.evals.analysis.style import ArmName, humanize
from threetears.evals.kernel.declaration import RateDrop


def _question(bundle: AnalysisContextBundle) -> str | None:
    """The question the campaign asks the aspect, in its words, whenever it was asked; None where none is."""
    declared = bundle.declared
    if declared is None or not bundle.question_declared:
        return None
    return declared.question


#: A first word that is a plain word, which a title may capitalize; an identifier (``queue_v5``, ``v4-flash``) is not.
_PLAIN_WORD = re.compile(r"[a-z]+(?:['-][a-z]+)*(?=[\s,;:?!.]|$)")


def _capitalized(title: str) -> str:
    """A title starting with a capital where its first word is a plain word; an identifier is left as written."""
    return title[:1].upper() + title[1:] if _PLAIN_WORD.match(title) else title


def _heading_blocks(
    bundle: AnalysisContextBundle, names: dict[str, ArmName], source: ReportSource, *, aspect: str | None
) -> list[ReportBlock]:
    """The first screen's lines between the title, which is the campaign's name, and the verdict.

    With a question declared, the question quoted on its own line; then the subhead carrying what the question need
    not: the arms line, each arm with its version, and the aspect where its name is not the title's (``aspect``).
    When its criteria were set is a caveat under the verdict
    (:func:`~threetears.evals.analysis.report.campaign.provenance.caveat_blocks`), said once.
    """
    arms = DisclosureBlock(section="header", source="arms", text=_arms_line(bundle, names, source, aspect=aspect))
    asked = _question_line(bundle)
    basis = [] if asked is None else [DisclosureBlock(section="header", source="basis", text=asked)]
    return [*basis, arms]


def _floor_warning_blocks(
    bundle: AnalysisContextBundle, arm: Callable[[str], str], aspect: str, forms: PageForms
) -> list[ReportBlock]:
    """One line per candidate breaking a floor, under the verdict, in the words a campaign's page warns in
    (:func:`~threetears.evals.analysis.report.campaign.warnings.floor_line`): what it means for the candidate, never the
    rate or the interval, which the Floors section reads clause by clause. A floor held or unclear says nothing here."""
    blocks: list[ReportBlock] = []
    for key, read in _floor_verdicts(bundle, arm).items():
        broken = [verdict.floor for verdict in read if verdict.outcome == "broken"]
        if broken:
            margins = [floor.clause.max_points if isinstance(floor.clause, RateDrop) else None for floor in broken]
            text = floor_line(arm(key), margins, aspect, forms=forms)
            blocks.append(DisclosureBlock(section="header", source="floor", text=text))
    return blocks


def _question_line(bundle: AnalysisContextBundle) -> str | None:
    """Under the title: the question declared, quoted whole and alone on its line — the contract the
    verdict reads, never paraphrased, its first plain word capitalized as a sentence starts; None where none was
    declared. The arms' versions are the subhead's (:func:`_heading_blocks`), so a question need not carry them.
    """
    question = _question(bundle)
    if question is None:
        return None
    return f'"{_capitalized(" ".join(question.split()))}"'


def _arms_line(
    bundle: AnalysisContextBundle, names: dict[str, ArmName], source: ReportSource, *, aspect: str | None = None
) -> str:
    """The arms line: each arm named in full, once, the control first and marked, then the aspect where the title
    does not name it (``aspect``), then the date, then the design.

    An arm named by its model is named by the version that answered it where the responses named one model other
    than the one asked for (``arm_served_models``, the served-models check's own reading): what ran, not the alias,
    with the label every table calls it by in brackets, ``gpt-4o-2024-08-06 (gpt-4o)``.
    """
    control = _control_key(bundle)
    keys = [board.variant_key for board in bundle.scoreboard]
    ordered = ([control] if control is not None else []) + [key for key in keys if key != control]
    served = {
        reading.variant_key: humanize("model", reading.served_models[0])
        for reading in bundle.arm_served_models
        if reading.state == "one" and len(reading.served_models) == 1
    }
    versioned = {
        key: replace(name, model=served[key], given=True)
        if name.model is not None and key in served and served[key] != humanize("model", name.model)
        else name
        for key, name in names.items()
    }
    arms = " vs ".join(versioned[key].first() if key in versioned else _namer(names)(key) for key in ordered)
    return " · ".join(
        part for part in (arms or "no arm ran", aspect, _date(source.generated_at), _design_words(bundle)) if part
    )


def _design_words(bundle: AnalysisContextBundle) -> str:
    """The design at the end of the arms line: ``30 cases × 3 repeats``; where the arms differ, each range its own segment
    (``4 to 6 cases per arm · 2 to 3 repeats``), so no segment carries more figures than a sentence may. Empty when
    no arm ran; the cases alone where no repeat count was recorded."""
    cases = sorted({board.cases for board in bundle.scoreboard})
    repeats = sorted(
        {
            value
            for cell in bundle.cell_measures
            for value in (cell.repeats_per_case_min, cell.repeats_per_case_max)
            if value is not None
        }
    )
    if not cases:
        return ""
    case_words = (
        f"{cases[0]} case{'' if cases[0] == 1 else 's'}"
        if len(cases) == 1
        else f"{interval_beside(str(cases[0]), str(cases[-1]))} cases per arm"
    )
    if not repeats:
        return case_words
    repeat_words = (
        f"{repeats[0]} repeat{'' if repeats[0] == 1 else 's'}"
        if len(repeats) == 1
        else f"{interval_beside(str(repeats[0]), str(repeats[-1]))} repeats"
    )
    if len(cases) == 1 and len(repeats) == 1:
        return f"{case_words} × {repeat_words}"
    return f"{case_words} · {repeat_words}"


__all__: list[str] = []
