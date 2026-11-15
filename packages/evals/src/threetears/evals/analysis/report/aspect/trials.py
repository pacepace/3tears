"""The **Cases** and **Trials** sections: one row per case, each arm's trials as marks, then the trials' census and
each failed or lost trial's record.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import TypedDict

from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, TrialRow
from threetears.evals.analysis.report.aspect.common import (
    _case_namer,
    _case_numbers,
    _control_key,
    _printed_strata,
    _stratum_word,
)
from threetears.evals.analysis.report.forms import PageForms
from threetears.evals.analysis.report.model import (
    CellValue,
    DisclosureBlock,
    PrintedCheck,
    PrintedJudgement,
    ReportBlock,
    TableBlock,
    TableCell,
    TableColumn,
    TrialBlock,
    table_row,
)
from threetears.evals.analysis.report.verdict import (
    CauseWords,
    arm_exclusions,
    arm_run_statuses,
    cause_short,
    fail_words,
    loss_words,
)
from threetears.evals.analysis.report.words import judged_name, trial_dimension
from threetears.evals.analysis.style import check_held, code
from threetears.evals.kernel.metrics import MATCH_MEASURE, goal_check_of

#: A trial's state as a mark in the cases table, always beside words for a lost one.
_MARKS = {"passed": "✓", "failed": "✗", "lost": "–"}


#: The order the trials section lists what went wrong in: a failure (a record a reader reads) before a loss (the
#: test setup's), and a passed trial last, wherever trials are listed by state.
_STATE_ORDER = {"failed": 0, "lost": 1, "passed": 2}


#: How many failed or lost records the page shows open, in the trials table's order. Twelve reads every failure of
#: a small campaign without a click, and a campaign that failed hundreds still scrolls; the rest stay closed.
_OPEN_TRIALS = 12


def _case_blocks(bundle: AnalysisContextBundle, arm: Callable[[str], str], forms: PageForms) -> list[ReportBlock]:
    """One row per case: each arm's trials as marks with their causes, regressions first, each linked to its trials."""
    if not bundle.trials:
        return []
    control = _control_key(bundle)
    arms = [board.variant_key for board in bundle.scoreboard]
    numbers = _case_numbers(bundle)
    case_name = _case_namer(bundle)
    by_case: dict[str, list[TrialRow]] = {}
    for trial in bundle.trials:
        by_case.setdefault(trial.case_id, []).append(trial)
    strata = any(trial.stratum for trial in bundle.trials)
    printed = _printed_strata(bundle) if strata else {}

    def counts(case: str, key: str) -> tuple[int, int]:
        """How many of ``key``'s usable trials on ``case`` passed, of how many."""
        usable = [t for t in by_case[case] if t.arm == key and t.state != "lost"]
        return sum(t.state == "passed" for t in usable), len(usable)

    def group(case: str) -> int:
        """0 a regression, 1 an arm with no usable trial, 2 the rest."""
        # A case-level flag: a candidate's share of its usable trials passed below the control's. A tie or a gain
        # never flags.
        ours, of = counts(case, control) if control is not None else (0, 0)
        for key in arms:
            theirs, n = counts(case, key)
            if key != control and n and of and theirs * of < ours * n:
                return 0
        if any(counts(case, key)[1] == 0 for key in arms):
            return 1
        return 2

    rows: list[dict[str, CellValue]] = []
    for case in sorted(by_case, key=lambda case: (group(case), numbers[case])):
        trials = by_case[case]
        row: dict[str, CellValue] = {
            "case": case_name(case),
            "flag": {0: "regression", 1: "no usable trial", 2: None}[group(case)],
        }
        if strata:
            row["stratum"] = _stratum_word(trials[0].stratum, printed) if trials[0].stratum else None
        for index, key in enumerate(arms):
            row[f"arm_{index}"] = _marks([trial for trial in trials if trial.arm == key])
        # The ids are navigation, not information: in their own order, so a row reads T1 T2 T3 on every replay; which
        # failed is the marks' to say.
        row["trials"] = " ".join(trial.trial_id for trial in sorted(trials, key=_trial_order))
        rows.append(row)
    blocks: list[ReportBlock] = [
        TableBlock(
            section="cases",
            name="cases",
            title="Cases",
            columns=[
                TableColumn(key="case", header="Case", term="case"),
                TableColumn(key="flag", header="Flag", term="flag"),
                *([TableColumn(key="stratum", header="Stratum", term="stratum")] if strata else []),
                *(
                    TableColumn(
                        key=f"arm_{index}", header=f"▸ {arm(key)}" if key == control else arm(key), term="marks"
                    )
                    for index, key in enumerate(arms)
                ),
                TableColumn(key="trials", header="Trials", term="trial"),
            ],
            rows=[table_row(row) for row in rows],
            order="regressions first, then cases an arm has no usable trial of, then the rest, each in case order",
            total_rows=len(rows),
        )
    ]
    if bundle.trials_omitted:
        blocks.append(
            DisclosureBlock(
                section="cases",
                source="runs",
                text=(
                    f"{forms.count(bundle.trials_omitted).text} more trials count on the scoreboard and are not listed "
                    "here, the passed "
                    "ones first."
                ),
            )
        )
    return blocks


def _marks(trials: Sequence[TrialRow]) -> str:
    """One arm's trials of one case as marks, repeat by repeat, its losses and failures named by cause."""
    # Repeat by repeat, two trials of one repeat in their ids' order, so a replay of one recording marks alike.
    ordered = sorted(trials, key=lambda trial: (trial.repeat, _trial_order(trial)))
    marks = " ".join(_MARKS[trial.state] for trial in ordered) or "not run"
    named = Counter(
        f"{'lost' if trial.state == 'lost' else 'failed'}: {cause_short(trial.cause or 'unknown')}"
        for trial in ordered
        if trial.state != "passed"
    )
    if named:
        marks += " (" + ", ".join(f"{words} ×{count}" for words, count in sorted(named.items())) + ")"
    return marks


#: A check's name that says it is one (``Ticket status check``, ``Goal check 2``).
_SAYS_CHECK = re.compile(r"\bcheck\b", re.IGNORECASE)


def _trial_why(trial: TrialRow, check_names: Mapping[str, str], check_failures: Mapping[str, str]) -> str | None:
    """Which check or judged dimension failed a trial, or why it was lost, on one line; None for a passed trial. A
    goal check by its name in ``check_names`` (``AnalysisContextBundle.named_checks``), never its expression; any
    other check by the value that failed it, as its pass rule words it (``check_failures``, by the check's measure
    key, which a trial's failed check holds: ``Never topped up is true``)."""
    if trial.state == "passed":
        return None
    if trial.state == "lost":
        return f"lost: {cause_short(trial.cause or 'unknown')}"
    if trial.cause == "check" and trial.failed_checks:
        # A check is named by its name, never by its expression and never a "pass rate" on one trial.
        # A goal check by its name, a check the catalog describes by the value that failed it, and anything else a
        # trial records as failed (a grader's ``total is False`` with no descriptor) as code, as written.
        named = [check_names.get(check.expression) or check_failures.get(check.name) for check in trial.failed_checks]
        said = ", ".join(name or code(check.expression) for name, check in zip(named, trial.failed_checks, strict=True))
        # Where every name already says it is a check (``Ticket status check``), the cell never says so twice.
        return said if all(name and _SAYS_CHECK.search(name) for name in named) else f"check: {said}"
    if trial.cause == "judged":
        under = [entry.dim for entry in trial.judge if entry.scale == "pass_fail" and entry.score == 0]
        return "judge: " + ", ".join(judged_name(dim) for dim in under) if under else "failed by the judge"
    return cause_short(trial.cause or "unknown")


def _wrong_first(trials: Sequence[TrialRow]) -> list[TrialRow]:
    """``trials`` with the failed ones first, then the lost, then the passed, each group as the bundle numbers it."""
    return sorted(trials, key=lambda trial: _STATE_ORDER[trial.state])


#: A heading in what a case asked: a ``## <name>`` line, which opens a part.
_ASKED_HEADING = re.compile(r"^## (?P<name>[^\n]+)$", re.MULTILINE)


class _AskedAndWorld(TypedDict):
    first_input: str
    world: str
    world_parts: list[str]


def _asked_and_world(asked: str) -> _AskedAndWorld:
    """What a case asked, split from the world state its host wrote into it: each ``## <name>`` part whose body opens
    on JSON (``{`` or ``[``), up to the next heading or the end, is the state's; everything else stays what the case
    asked. The state is a fold of its own, counted by its parts, so no page prints a host's JSON open."""
    starts = [match.start() for match in _ASKED_HEADING.finditer(asked)]
    if not starts:
        return {"first_input": asked, "world": "", "world_parts": []}
    kept, world, names = [asked[: starts[0]]], [], []
    for start, end in zip(starts, [*starts[1:], len(asked)], strict=True):
        part = asked[start:end]
        heading, _, body = part.partition("\n")
        if body.lstrip().startswith(("{", "[")):
            world.append(part.strip())
            names.append(heading.removeprefix("## ").strip())
        else:
            kept.append(part)
    if not world:
        return {"first_input": asked, "world": "", "world_parts": []}
    return {"first_input": "".join(kept).strip(), "world": "\n\n".join(world), "world_parts": names}


#: A trial id's number: ``T12`` is 12, so ids sort as numbers, never as text (``T11`` after ``T7``).
_TRIAL_NUMBER = re.compile(r"\d+")


def _trial_order(trial: TrialRow) -> tuple[int, str]:
    """Where a trial sorts among its case's: by its id's number, then by its id."""
    found = _TRIAL_NUMBER.search(trial.trial_id)
    return (int(found.group(0)) if found else 0, trial.trial_id)


def _census_line(bundle: AnalysisContextBundle, forms: PageForms) -> str:
    """The trials' census in one line, from the scoreboard, which counts every trial whether listed or not.

    ``180 trials ran. Of them 179 passed and 1 failed.`` The lost are counted only when there are any. Two
    sentences, the second opening on a word: the page's sentence rule would read a second sentence opening on a
    digit as the first one's tail, and four counts in one sentence is a table.
    """
    boards = bundle.scoreboard
    ran = sum(board.ran for board in boards)
    passed = sum(board.passed for board in boards)
    failed = sum(sum(board.failed.values()) for board in boards)
    lost = sum(sum(board.lost.values()) for board in boards)
    opened = f"{forms.count(ran).text} trial{'' if ran == 1 else 's'} ran."
    if passed == ran:
        return f"{opened} {'It' if ran == 1 else 'Every one'} passed."
    counts = [
        f"{forms.count(passed).text} passed",
        f"{forms.count(failed).text} failed",
        *([f"{forms.count(lost).text} were lost"] if lost else []),
    ]
    return f"{opened} Of them {', '.join(counts[:-1])} and {counts[-1]}."


def _trial_blocks(
    bundle: AnalysisContextBundle, arm: Callable[[str], str], forms: PageForms, *, records: bool
) -> list[ReportBlock]:
    """The census in one line, then only what went wrong, then the passed trials folded away in one place.

    A campaign that ran hundreds of trials and failed one is read for that one: the table lists the failed trials
    first and the lost ones after, each by trial id, and with ``records`` each has a record (what it asked, what the
    judge saw and said, and why) in the same order, the first :data:`_OPEN_TRIALS` shown open. Without them each
    row names its stored result instead, read with ``result_get``, and carries the trial's id, so a link to the trial
    lands on its row: the records are most of a large campaign's page. A passed trial has nothing to read, so it is
    one row of a closed table, there so its stored result stays reachable with ``result_get``. When every trial
    passed the line stands alone.
    """
    if not bundle.trials:
        return []
    blocks: list[ReportBlock] = [DisclosureBlock(section="trials", source="runs", text=_census_line(bundle, forms))]
    arms = len(bundle.scoreboard)
    exclusions = arm_exclusions(bundle)
    statuses = arm_run_statuses(bundle)
    case_name = _case_namer(bundle)
    checks = bundle.named_checks()
    # Each check that is no goal check, by the measure key a trial's failed check carries, worded at its failing value
    # as the pass rule words its passing one.
    failures = {
        name: check_held(name, descriptor, failed=True)
        for name, descriptor in bundle.measure_catalog.items()
        if goal_check_of(name) is None and (name == MATCH_MEASURE or descriptor.is_check)
    }
    wrong = [trial for trial in _wrong_first(bundle.trials) if trial.state != "passed"]
    rows: list[dict[str, TableCell]] = []
    held: list[ReportBlock] = []
    for position, trial in enumerate(wrong):
        why = _trial_why(trial, checks, failures)
        row = table_row(
            {
                "trial": trial.trial_id,
                "case": case_name(trial.case_id),
                "arm": arm(trial.arm),
                "outcome": trial.state,
                "why": why,
            }
        )
        if not records:
            # No record carries the trial's id, so its row does: a link to the trial lands here, beside its result.
            rows.append(
                {
                    **row,
                    "trial": TableCell(value=trial.trial_id, anchor=trial.trial_id),
                    "result": TableCell(value=code(trial.result_id)),
                }
            )
            continue
        rows.append(row)
        words: CauseWords
        if trial.state == "lost":
            words = loss_words(
                trial.cause or "unknown",
                exclusions=exclusions.get(trial.arm, ()),
                arms=arms,
                run_statuses=statuses.get(trial.arm, ()),
            )
        else:
            words = fail_words(trial.cause or "unknown")
        held.append(
            TrialBlock(
                section="trials",
                anchor=trial.trial_id,
                trial_id=trial.trial_id,
                case_label=case_name(trial.case_id),
                arm=arm(trial.arm),
                repeat=trial.repeat,
                state=trial.state,
                cause_words=f"{words.words}. {words.remedy[:1].upper()}{words.remedy[1:]}.",
                why=why,
                **_asked_and_world(trial.first_input),
                excerpt=trial.excerpt,
                # The trial's own record keeps the check's measure key; the page prints the check's name beside the
                # expression: a goal check's as the aspect names it, any other's as the catalog does.
                failed_checks=[
                    PrintedCheck(
                        name=checks[check.expression]
                        if check.expression in checks
                        else bundle.catalog_name(check.name),
                        expression=check.expression,
                    )
                    for check in trial.failed_checks
                ],
                # And each judged dimension by its id; the page prints a reserved axis by its word.
                judge=[
                    PrintedJudgement(
                        name=trial_dimension(entry.dim),
                        criterion=entry.criterion,
                        score=entry.score,
                        scale=entry.scale,
                        reason=entry.reason,
                    )
                    for entry in trial.judge
                ],
                error_words=trial.error_words,
                result_id=trial.result_id,
                open=position < _OPEN_TRIALS,
            )
        )
    if rows:
        columns = [
            TableColumn(key="trial", header="Trial", term="trial"),
            TableColumn(key="case", header="Case", term="case"),
            TableColumn(key="arm", header="Arm"),
            TableColumn(key="outcome", header="Outcome", term="outcome"),
            TableColumn(key="why", header="Why", term="why"),
        ]
        if not records:
            columns.append(TableColumn(key="result", header="Stored result"))
        blocks.append(
            TableBlock(
                section="trials",
                name="trials",
                title="Trials",
                columns=columns,
                rows=rows,
                order="the failed first, then the lost, each by trial id",
                total_rows=len(rows),
            )
        )
    blocks += held
    passed = [
        {
            "trial": trial.trial_id,
            "case": case_name(trial.case_id),
            "arm": arm(trial.arm),
            "result": code(trial.result_id),
        }
        for trial in bundle.trials
        if trial.state == "passed"
    ]
    if passed:
        blocks.append(
            TableBlock(
                section="trials",
                name="passed_trials",
                title="Passed trials",
                columns=[
                    TableColumn(key="trial", header="Trial", term="trial"),
                    TableColumn(key="case", header="Case", term="case"),
                    TableColumn(key="arm", header="Arm"),
                    TableColumn(key="result", header="Stored result"),
                ],
                rows=[table_row(row) for row in passed],
                order="as the bundle numbers them",
                total_rows=len(passed),
                details=True,
            )
        )
    return blocks


__all__: list[str] = []
