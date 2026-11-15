"""Reader names: the one name a reader is shown for each id a key names (a check, a measure, a stratum, an arm)."""

from __future__ import annotations

from typing import Literal

from threetears.evals.analysis.report.keys.vocabulary import _criterion, _printed_stratum, _stratum, _task
from threetears.evals.analysis.report.per_input import kept_for
from threetears.evals.analysis.report.words import (
    arm_tags,
    aspect_stratum_key,
    given_name,
    reader_arm_name,
    text_levers,
    told_apart,
)
from threetears.evals.analysis.style import judged_name
from threetears.evals.analysis.report_input import Arm, ReportInput
from threetears.evals.analysis.report_margins import aspect_of
from threetears.evals.analysis.style import check_names, measure_name
from threetears.evals.kernel.metrics import goal_check_of


# --- reader names: the one name a reader is shown for each id -----------------------------------------------------

#: The kinds of id a reader is given a name for (GENERATOR §6.3: reader names, not identifiers).
NamedKind = Literal["aspect", "measure", "judged", "stratum", "task", "criterion"]


def input_check_name(key: str, report_input: ReportInput) -> str | None:
    """What a report document's page calls the goal check ``key`` reads on a report input, through the one rule its
    tables and its evidence record both name a check by (:func:`~threetears.evals.analysis.style.check_names`):
    the name the input's label declares, else what it tests (``Field accuracy check``), as the code page names it,
    over every check the input reads. The input's label for an undeclared check is its expression, which the writer
    reads.

    :param key: Any measure key.
    :ptype key: str
    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: The check's name; None for a key that is no goal check's.
    :rtype: str | None
    """
    check = goal_check_of(key)
    if check is None:
        return None
    read = {
        expression: measure.label
        for name, measure in report_input.measures.items()
        if (expression := goal_check_of(name)) is not None
    }
    declared = {expression: label for expression, label in read.items() if label and label != expression}
    return check_names([*read, check], declared)[check]


def _measure_reader_name(key: str, report_input: ReportInput) -> str:
    """A measure's or judged dimension's reader name, whichever the key names."""
    if key in report_input.measures:
        return reader_name("measure", key, report_input)
    if key in report_input.judged:
        return reader_name("judged", key, report_input)
    return measure_name(key)


def reader_name(kind: NamedKind, identifier: str, report_input: ReportInput, *, aspect: str | None = None) -> str:
    """The one name a reader is shown for an id: the name the input declares, else the package's reader name for it.

    An aspect, stratum or task by its declared name, else its id with its underscores and dots as spaces
    (:func:`~threetears.evals.analysis.report.words.given_name`); a stratum among its own aspect's strata, two that
    would print alike each by its id (:func:`~threetears.evals.analysis.report.words.stratum_names`); a measure by its
    label (the catalog's reader name), else as :func:`~threetears.evals.analysis.style.measure_name` names it; a judged
    dimension by its declared label, else its name (:func:`~threetears.evals.analysis.report.words.judged_name`); a
    criterion by its label, else its measure's reader name. The ``.name`` and ``.label`` figure keys print the same
    name.

    :param kind: What the id names.
    :ptype kind: NamedKind
    :param identifier: The id, as the input keys it.
    :ptype identifier: str
    :param report_input: The report input.
    :ptype report_input: ReportInput
    :param aspect: For a stratum, the aspect whose stratum of that id is named: a stratum id held on more than one
        aspect is a stratum on each, and each aspect may name its own differently. None names the first the input
        holds.
    :ptype aspect: str | None
    :return: The reader name.
    :rtype: str
    :raises KeyError: The input holds no such id of that kind (a stratum: on that aspect).
    """
    if kind == "aspect":
        named = aspect_of(report_input, identifier)
        return given_name(named.id, named.name)
    if kind == "measure":
        label = report_input.measures[identifier].label
        # A goal check the input declares no display name for carries its bare expression as its label: it is named
        # through the written page's one door for a check, as every table header and chart names it.
        if (check := input_check_name(identifier, report_input)) is not None:
            return check
        return measure_name(identifier) if label is None else label
    if kind == "judged":
        return judged_name(identifier, report_input.judged[identifier].label)
    if kind == "stratum":
        return _printed_stratum(report_input, _stratum(report_input, identifier, aspect))
    if kind == "task":
        task = _task(report_input, identifier)
        return given_name(task.id, task.name)
    criterion = _criterion(report_input, identifier)
    return criterion.label or _measure_reader_name(criterion.measure, report_input)


def reader_names(report_input: ReportInput) -> dict[NamedKind, dict[str, str]]:
    """Every id's reader name (:func:`reader_name`), by kind, then id, in the input's order.

    A stratum id held on more than one aspect that the aspects print differently (each declares its own name, or two
    of one aspect's ids would print alike there) is a name for each: keyed by the stem its figure keys take on that
    aspect (``aspects.<aspect>.strata.<id>``), so no aspect's name of it is dropped. One every aspect prints alike is
    keyed by its id.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Aspects, measures, judged dimensions, strata, tasks and criteria, each kind present only where the input
        holds one.
    :rtype: dict[NamedKind, dict[str, str]]
    """
    ids: dict[NamedKind, list[str]] = {
        "aspect": [aspect.id for aspect in report_input.aspects],
        "measure": list(report_input.measures),
        "judged": list(report_input.judged),
        "stratum": [stratum.id for stratum in report_input.strata],
        "task": [task.id for task in report_input.tasks],
        "criterion": [criterion.id for criterion in report_input.criteria],
    }
    return {
        kind: (
            _strata_names(report_input)
            if kind == "stratum"
            else {identifier: reader_name(kind, identifier, report_input) for identifier in dict.fromkeys(held)}
        )
        for kind, held in ids.items()
        if held
    }


def _strata_names(report_input: ReportInput) -> dict[str, str]:
    """Each stratum's reader name, by id, or by its aspect's stem where its aspects print one id differently."""
    printed: dict[str, dict[str | None, str]] = {}
    for stratum in report_input.strata:
        printed.setdefault(stratum.id, {}).setdefault(stratum.aspect, _printed_stratum(report_input, stratum))
    named: dict[str, str] = {}
    for identifier, by_aspect in printed.items():
        if len(set(by_aspect.values())) == 1:
            named[identifier] = next(iter(by_aspect.values()))
            continue
        for aspect, name in by_aspect.items():
            named[identifier if aspect is None else aspect_stratum_key(aspect, identifier)] = name
    return named


def page_arm_names(report_input: ReportInput) -> dict[str, str]:
    """The name a page prints for each arm, read at render time: its declared short name, else its alias read by
    :func:`~threetears.evals.analysis.report.words.reader_arm_name` (a cut level by its lever, an empty text level
    ``no system prompt``), told apart by its tag where two arms then read alike
    (:func:`~threetears.evals.analysis.report.words.told_apart`), else its id. The input itself is never renamed: a
    snapshot's id is its content's digest. The page, its blocks, the analyst's cited figures and the store gate all
    read arms by these names.

    The names are pure in the input, which is frozen, so they are read once per input
    (:func:`~threetears.evals.analysis.report.per_input.kept_for`); each caller gets its own copy.

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: Arm id -> its name on the page.
    :rtype: dict[str, str]
    """
    return dict(kept_for(report_input).get(_page_arm_names, lambda: _page_arm_names(report_input)))


def _page_arm_names(report_input: ReportInput) -> dict[str, str]:
    """The names :func:`page_arm_names` reads, worked out of the input."""
    arms = report_input.arms
    control = next((arm.id for arm in arms if arm.role == "control"), None)
    tags = arm_tags([arm.id for arm in arms], control)
    text = text_levers(arm.settings for arm in arms)
    printed = {arm.id: printed_arm_name(arm) for arm in arms}
    names = {
        arm.id: printed[arm.id]
        if arm.short_name
        else reader_arm_name(printed[arm.id], arm.settings, text=text, declared=report_input.lever_names)
        for arm in arms
    }
    return told_apart(names, tags, {key for key in names if names[key] != printed[key]})


def printed_arm_name(arm: Arm) -> str:
    """The name a page prints for an arm: its declared short name, else its alias, else its id.

    Every page passes these names to :func:`render`, so ``{arm:<id>}`` prints this beside the arm's tag.

    :param arm: The arm.
    :ptype arm: Arm
    :return: Its name on the page.
    :rtype: str
    """
    return arm.short_name or arm.name or arm.id


__all__ = [
    "NamedKind",
    "input_check_name",
    "page_arm_names",
    "printed_arm_name",
    "reader_name",
    "reader_names",
]
