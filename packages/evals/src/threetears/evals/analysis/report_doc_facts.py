"""The editor's facts: what code computes about a report document on its snapshot, for the report editor to weigh.

A report document (:class:`~threetears.evals.kernel.report_doc.ReportDoc`) is held to two kinds of finding (ruling 1).
The mechanical ones are the invariants (``analysis/report_doc_checks.py``): code can prove them wrong, and a verdict
blocks storage. The interpretive ones are here. Code can compute that something holds, but not whether the document
handles it well, so each is handed to the editor as a fact: one sentence stating a computed truth, with the figure keys
it rests on. **A fact never refuses**, never blocks storage, and never says what the editor should conclude. The
editor reads each beside the page and decides whether the document answers it.

**The facts** (:func:`editor_facts`), in this order, each from the code the report plan's checks and the analyst's
tools already hold:

- ``gate_failed``: an arm the document recommends misses a declared gate or guardrail figure
  (:func:`~threetears.evals.analysis.analyst.analyses.gate_reads`), one fact per figure missed.
- ``warning_unanswered``: code raises a warning on an arm the document recommends
  (:func:`~threetears.evals.analysis.report_plan_checks.raised_warnings`: a floor read broken or unclear, an aspect
  where a pair reading shows another arm ahead, a bar missed), and no figure key in the document names it. A key
  names a floor's warning where its path holds the floor's key and the arm, a pair's where it holds both arms, a bar's
  where it holds the bar's measure and the arm.
- ``call_vs_dominance``: the call (``recommends``) differs from code's call as ``options_dominance`` reads it
  (:func:`~threetears.evals.analysis.analyst.analyses.dominance`): one option dominates and the document recommends
  anything else, or nothing; or code's rule allows no decide mode (a tradeoff, or no call) and the document recommends
  an arm.
- ``weakest_not_in_watch``: the snapshot has a weakest margin
  (:func:`~threetears.evals.analysis.report_margins.weakest_margin`) and no Watch entry of the decision block cites a
  key naming it: a ``weakest.`` key, or one whose path holds the margin's arm and its line (the criterion or its
  measure, the aspect, or the floor). Whether Watch should say more about it is the editor's.
- ``adoption``: a recommended arm is a candidate (the snapshot has a control and it is not that arm), and the
  evidence for switching to it is short, as the report plan's adoption check read it: the snapshot asks no question of
  any aspect; or a guardrail the pipeline read on it is breached
  (:func:`~threetears.evals.analysis.analyst.analyses.breached_guardrails`); or it is shown better on no asked aspect
  (:func:`~threetears.evals.analysis.report_decision.shown_better`) and not shown no worse
  (:func:`~threetears.evals.analysis.report_decision.floors_not_held`).
- ``settled_loss_unasked``: a recommended candidate's reading on an aspect the campaign does not ask about is a loss
  settled past the aspect's margin: its interval lies wholly at or below minus the margin. It is the other side of
  #980's open caveat (``unsettled:<aspect>``, an interval that allows a loss past the margin and a change short of
  it): code's decision never reads an unasked aspect, so no caveat carries a settled loss there.
- ``undecided_implied``: an option a gate has no figure for somewhere is undecided
  (:func:`~threetears.evals.analysis.report_margins.undecided_arms`), and its figure over every case misses the line,
  so if the places split the cases between them at least one place misses too. ``options_dominance`` hands the same
  inference to the analyst, labelled inferred; here it reaches the editor whatever the document says of that option.
- ``count_in_words``: prose writes a count as words with no figure key: one count word (``zero`` to ``twenty``, as
  the ``words`` filter spells counts) or key, then ``of``, then another, where at least one of the two is a bare word
  (``four of five``, ``two of the three``). That is a count a reader reads as a number. A lone count word (``one
  arm``) reads as an article or a name, and is not one; nor is ``one`` before ``of``, which picks an item (``the only
  one of {n_candidates|words} configurations``).

**What a fact is.** An :class:`EditorFact` is ``{kind, path, fact, figures}``. ``path`` is the document field the fact
is about (``recommends[0]``, ``decision[2]``, ``claims[1].text``), or empty for the document as a whole. ``fact`` is the
sentence, every figure in it written as the page writes its key; ``figures`` are those keys' paths, in the order the
sentence cites them, so the editor can read each with its tools. The sentences are short, since the editor's user
message carries them all.

**Every condition is on data.** Nothing here reads ``story``, a slot's word other than Watch's, a claim's weight or the
eval's kind or shape. Each fact reads the snapshot's figures and what the document names: its ``recommends``, its
Watch entries and its prose. So the facts are byte-identical when only ``story`` changes.

**An arm the snapshot does not hold.** The document's validity
(:func:`~threetears.evals.kernel.report_doc.report_doc_validity`) refuses a recommended arm the snapshot does not
hold, before the invariants run. The facts never refuse, so they read such an arm as one code says nothing about: no
gate, warning, adoption or settled-loss fact names it, and the call fact writes its id as the document does.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.analyses import (
    Dominance,
    GateRead,
    breached_guardrails,
    dominance,
    gate_reads,
)
from threetears.evals.analysis.report.figure_keys import WEAKEST_PREFIX, Figure, format_figure, reader_name, render
from threetears.evals.analysis.report_decision import control_arm, floors_not_held, shown_better
from threetears.evals.analysis.report_input import Aspect, Reading, ReportInput
from threetears.evals.analysis.report_margins import Margin, weakest_margin
from threetears.evals.analysis.report_plan_checks import raised_warnings
from threetears.evals.kernel.report_doc import REPORT_DOC_PROSE_FIELDS, ReportDoc
from threetears.evals.kernel.report_plan import WarningRef

#: The names with a caller outside this module: none yet. :func:`editor_facts` and :class:`EditorFact` join it with
#: their first production caller, the writing loop's editor turn, never for a test alone.
__all__: list[str] = []

#: What a fact is about, one word per fact this module computes, in the order :func:`editor_facts` lists them.
EditorFactKind = Literal[
    "gate_failed",
    "warning_unanswered",
    "call_vs_dominance",
    "weakest_not_in_watch",
    "adoption",
    "settled_loss_unasked",
    "undecided_implied",
    "count_in_words",
]


class EditorFact(BaseModel):
    """One computed truth about a report document on its snapshot, for the editor to weigh; never a verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: EditorFactKind = Field(description="What the fact is about.")
    path: str = Field(
        description="The document field it is about (`recommends[0]`, `decision[2]`, `claims[1].text`); empty for "
        "the document as a whole."
    )
    fact: str = Field(
        min_length=1, description="The fact, in one sentence, its figures written as the page writes them."
    )
    figures: list[str] = Field(description="The figure keys the sentence cites, by path, in its order; may be empty.")


def editor_facts(doc: ReportDoc, report_input: ReportInput) -> list[EditorFact]:
    """Every interpretive fact code computes about a report document on its snapshot (this module's facts).

    :param doc: The document, as the model accepts it; its validity is not required.
    :ptype doc: ReportDoc
    :param report_input: The report input of the snapshot it was written from.
    :ptype report_input: ReportInput
    :return: Each fact, kinds in this module's order, then in document order; empty when none holds.
    :rtype: list[EditorFact]
    """
    arms = {arm.id for arm in report_input.arms}
    recommended = [(f"recommends[{index}]", arm) for index, arm in enumerate(doc.recommends) if arm in arms]
    keys = _cited_keys(doc)
    options = dominance(report_input)
    facts: list[EditorFact] = []
    facts += [fact for path, arm in recommended for fact in _gate_facts(report_input, path, arm)]
    facts += [fact for path, arm in recommended for fact in _warning_facts(report_input, path, arm, keys)]
    facts += _call_facts(doc, report_input, options)
    facts += _weakest_facts(doc, report_input)
    facts += [fact for path, arm in recommended for fact in _adoption_facts(report_input, path, arm)]
    facts += [fact for path, arm in recommended for fact in _settled_loss_facts(report_input, path, arm)]
    facts += _undecided_facts(doc, report_input, options)
    facts += _count_facts(doc)
    return facts


# --- writing a fact ------------------------------------------------------------------------------------------------

#: A figure key as prose holds it: one pair of single braces, its path before any filter.
_KEY = re.compile(r"\{([^{}|]*)(?:\|[^{}]*)?\}")


def _fact(kind: EditorFactKind, path: str, template: str, report_input: ReportInput) -> EditorFact:
    """A fact from a sentence that names its figures by key: the keys written out, and listed by path.

    Every key in ``template`` is built here from what the input holds, so it resolves.
    """
    return EditorFact(
        kind=kind,
        path=path,
        fact=render(template, report_input, {}),
        figures=[match.group(1) for match in _KEY.finditer(template)],
    )


def _arm(arm: str) -> str:
    """An arm as a fact's template names it: its key, which the page writes as its name and tag."""
    return "{arm:" + arm + "}"


def _bounded(name: str, path: str) -> bool:
    """Whether a key's path holds a name as whole dotted parts: at its start or after a dot, and before a dot or its
    end. Ids carry dots, so a name is matched whole, never split."""
    return re.search(rf"(?:^|\.){re.escape(name)}(?:\.|$)", path) is not None


# --- the keys a document cites -------------------------------------------------------------------------------------


def _prose(doc: ReportDoc) -> Iterator[tuple[str, str]]:
    """Every prose field the document holds (:data:`REPORT_DOC_PROSE_FIELDS`), by its field path, in document order."""
    for field in REPORT_DOC_PROSE_FIELDS:
        yield from _walk(doc, field.split("."), "")


def _walk(node: object, parts: Sequence[str], at: str) -> Iterator[tuple[str, str]]:
    if not parts:
        if isinstance(node, str):
            yield at, node
        return
    head, rest = parts[0], parts[1:]
    name = head.removesuffix("[]")
    value = getattr(node, name, None)
    here = f"{at}.{name}" if at else name
    if head.endswith("[]"):
        for index, item in enumerate(value if isinstance(value, list) else []):
            yield from _walk(item, rest, f"{here}[{index}]")
    elif value is not None:
        yield from _walk(value, rest, here)


def _cited_keys(doc: ReportDoc) -> list[str]:
    """The path of every figure key the document's prose cites, as written; whether each resolves is invariant 1's."""
    return [match.group(1) for _, text in _prose(doc) for match in _KEY.finditer(text)]


# --- a recommended arm and the gates -------------------------------------------------------------------------------


def _place(report_input: ReportInput, read: GateRead) -> str:
    """Where a gate read sits, for a sentence: nothing overall, a task by its reader name, a cell as its key names it."""
    if read.place is None:
        return ""
    if any(task.id == read.place for task in report_input.tasks):
        return f" in {reader_name('task', read.place, report_input)}"
    return f" in cell {read.place}"


def _gate_facts(report_input: ReportInput, path: str, arm: str) -> Iterator[EditorFact]:
    for read in gate_reads(report_input, arm, "fail"):
        threshold, figure = read.figures[0], read.figures[1]
        kind = next(criterion.kind for criterion in report_input.criteria if criterion.id == read.criterion)
        name = reader_name("criterion", read.criterion, report_input)
        yield _fact(
            "gate_failed",
            path,
            f"Recommended {_arm(arm)} misses the declared {kind} {name}{_place(report_input, read)}: "
            f"{{{figure.key}}} against {{{threshold.key}}}.",
            report_input,
        )


# --- a recommended arm and the warnings code raises on it -----------------------------------------------------------


def _names_warning(key: str, warning: WarningRef, arm: str) -> bool:
    """Whether a cited key's path names a warning's figure: a floor's by its key, a pair's by the other arm, a bar's
    by its measure, each beside the arm."""
    return _bounded(arm, key) and _bounded(warning.name, key)


_WARNING_SAID = {
    "floor_broken": "floor {name} on {aspect} reads broken",
    "floor_unclear": "floor {name} on {aspect} reads unclear",
    "behind_on": "a pair reading on {aspect} shows {name} ahead of it",
    "bar_missed": "it misses the bar on {name} on {aspect}",
}


def _warning_facts(report_input: ReportInput, path: str, arm: str, keys: Sequence[str]) -> Iterator[EditorFact]:
    for warning in raised_warnings(report_input, arm):
        if any(_names_warning(key, warning, arm) for key in keys):
            continue
        aspect = _aspect_name(report_input, warning.aspect)
        name = _arm(warning.name) if warning.kind == "behind_on" else warning.name
        said = _WARNING_SAID[warning.kind].format(name=name, aspect=aspect)
        yield _fact(
            "warning_unanswered",
            path,
            f"Code raises {warning.kind} on recommended {_arm(arm)}: {said}. No figure key in the document names it.",
            report_input,
        )


def _aspect_name(report_input: ReportInput, aspect: str) -> str:
    """An aspect's reader name; a read may name an aspect the input holds no record for, which reads by its id."""
    if any(held.id == aspect for held in report_input.aspects):
        return reader_name("aspect", aspect, report_input)
    return aspect


# --- the call and code's -------------------------------------------------------------------------------------------

#: How code's modes read in a sentence.
_MODE_WORDS = {"decide": "a decision", "tradeoff": "a tradeoff", "no_call": "no call"}


def _arms_said(arms: Sequence[str], held: set[str]) -> str:
    """Arms as a sentence names them: each held one by its key, any other by its id as the document wrote it."""
    if not arms:
        return "no arm"
    return ", ".join(_arm(arm) if arm in held else f"`{arm}`" for arm in arms)


def _call_facts(doc: ReportDoc, report_input: ReportInput, options: Dominance) -> Iterator[EditorFact]:
    call = options.code_call
    dominant = [arm.removeprefix("arm:") for arm in call.dominant]
    held = {arm.id for arm in report_input.arms}
    recommends = _arms_said(doc.recommends, held)
    if len(dominant) == 1:
        if list(doc.recommends) == dominant:
            return
        code = f"{_arm(dominant[0])} as the one option dominating"
    elif "decide" not in call.modes and doc.recommends:
        modes = " or ".join(_MODE_WORDS[mode] for mode in call.modes)
        code = f"code's call as {modes}, with no option dominating"
    else:
        return
    yield _fact(
        "call_vs_dominance",
        "recommends",
        f"options_dominance reads {code}; the document recommends {recommends}.",
        report_input,
    )


# --- the weakest margin and Watch ----------------------------------------------------------------------------------


def _line_names(margin: Margin) -> list[str]:
    """What a key's path may name a margin's line by: the criterion and its measure, the aspect, or the floor's key."""
    return [name for name in (margin.source, margin.measure, margin.aspect) if name]


def _names_weakest(key: str, margin: Margin) -> bool:
    if key.startswith(WEAKEST_PREFIX):
        return True
    return _bounded(margin.arm, key) and any(_bounded(name, key) for name in _line_names(margin))


#: How the weakest margin reads in a sentence, its keys written out.
_WEAKEST_SAID = "{weakest.arm}, {weakest.value} against {weakest.criterion} at {weakest.threshold}"


def _weakest_facts(doc: ReportDoc, report_input: ReportInput) -> Iterator[EditorFact]:
    margin = weakest_margin(report_input)
    if margin is None:
        return
    watch = [(f"decision[{index}]", entry.text) for index, entry in enumerate(doc.decision) if entry.slot == "watch"]
    cited = [match.group(1) for _, text in watch for match in _KEY.finditer(text)]
    if any(_names_weakest(key, margin) for key in cited):
        return
    where = "no Watch entry cites a key naming it" if watch else "the decision block has no Watch entry"
    yield _fact(
        "weakest_not_in_watch",
        watch[0][0] if watch else "decision",
        f"The weakest margin is {_WEAKEST_SAID}, a gap of {{weakest.margin}}; {where}.",
        report_input,
    )


# --- switching to a candidate --------------------------------------------------------------------------------------


def _adoption_facts(report_input: ReportInput, path: str, arm: str) -> Iterator[EditorFact]:
    control = control_arm(report_input)
    if control is None or arm == control:
        return
    asked = [aspect for aspect in report_input.aspects if aspect.asked]
    named = f"Recommended {_arm(arm)} is a candidate against {_arm(control)}"
    if not asked:
        said = "the snapshot asks no question of any aspect, so no reading is confirmatory"
    elif breached := breached_guardrails(report_input, arm):
        said = f"it breached the guardrail on {', '.join(breached)}"
    elif not shown_better(report_input, arm) and (open_floors := floors_not_held(report_input, arm, asked)):
        said = (
            "no confirmatory reading shows it better and no contrast on an asked aspect shows it improved, and it is "
            f"not shown no worse: {'; '.join(open_floors)}"
        )
    else:
        return
    yield _fact("adoption", path, f"{named}; {said}.", report_input)


# --- a settled loss on an aspect nobody asked about ----------------------------------------------------------------


def _settled_loss(aspect: Aspect, reading: Reading) -> bool:
    """Whether a reading is a loss settled past its margin: its interval wholly at or below minus the margin."""
    margin = reading.margin if reading.margin is not None else aspect.margin
    if margin is None or reading.within_interval is None:
        return False
    return float(reading.within_interval[1]) <= -float(margin)


def _settled_loss_facts(report_input: ReportInput, path: str, arm: str) -> Iterator[EditorFact]:
    if arm == control_arm(report_input):
        return
    for aspect in report_input.aspects:
        if aspect.asked:
            continue
        for reading in aspect.readings:
            if reading.arm != arm or not _settled_loss(aspect, reading):
                continue
            prefix = f"aspects.{aspect.id}"
            margin = f"{prefix}.reading.{arm}.margin" if reading.margin is not None else f"{prefix}.margin"
            yield _fact(
                "settled_loss_unasked",
                path,
                f"Recommended {_arm(arm)}'s reading on {reader_name('aspect', aspect.id, report_input)}, which the "
                f"campaign does not ask about, is a loss settled past the margin: its interval, "
                f"{{{prefix}.reading.{arm}.interval}} points, lies wholly at or below minus the margin of "
                f"{{{margin}}} points.",
                report_input,
            )


# --- options a gate has no figure for ------------------------------------------------------------------------------


def _undecided_facts(doc: ReportDoc, report_input: ReportInput, options: Dominance) -> Iterator[EditorFact]:
    recommended = {arm: f"recommends[{index}]" for index, arm in enumerate(doc.recommends)}
    for row in options.undecided:
        arm = row.arm.removeprefix("arm:")
        for implied in row.implied:
            threshold, figure = implied.figures[0], implied.figures[1]
            name = reader_name("criterion", implied.criterion, report_input)
            yield _fact(
                "undecided_implied",
                recommended.get(arm, ""),
                f"{_arm(arm)} is undecided on {name}, which has no figure for it at some place it reads; its figure "
                f"over every case, {{{figure.key}}}, misses {{{threshold.key}}}, so if the places split the cases at "
                "least one misses too (inferred, not stated).",
                report_input,
            )


# --- counts written as words ---------------------------------------------------------------------------------------

#: The count words, as the ``words`` filter spells a count from zero to twenty: the one vocabulary, read off it.
_COUNT_WORDS = tuple(
    format_figure(Figure(path="count", kind="number", value=float(count), scale="count"), ("words",))
    for count in range(21)
)

#: What stands in for a figure key once prose is read for counts: no word character, so it bounds the words beside it.
_KEY_MARK = "\x00"

#: The count word that, before ``of``, picks one item rather than counting (``the only one of``, ``one of them``).
_PICKS_ONE = _COUNT_WORDS[1]

#: A count written ``<count> of [the] <count>``, either side a count word or a key, longest words first; ``one`` is no
#: count before ``of``.
_RATIO = re.compile(
    r"(?<![\w-])(?P<first>{first}|{mark}) of (?:the )?(?P<second>{words}|{mark})(?![\w-])".format(
        first="|".join(sorted(set(_COUNT_WORDS) - {_PICKS_ONE}, key=len, reverse=True)),
        words="|".join(sorted(_COUNT_WORDS, key=len, reverse=True)),
        mark=_KEY_MARK,
    ),
    re.IGNORECASE,
)


def _count_facts(doc: ReportDoc) -> Iterator[EditorFact]:
    for path, text in _prose(doc):
        masked = _KEY.sub(_KEY_MARK, text)
        for match in _RATIO.finditer(masked):
            if match.group("first") == _KEY_MARK and match.group("second") == _KEY_MARK:
                continue
            written = match.group(0).replace(_KEY_MARK, "{…}")
            yield EditorFact(
                kind="count_in_words",
                path=path,
                fact=f'The text writes the count "{written}" in words, with no figure key for it.',
                figures=[],
            )
