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
  measure, the aspect, or the floor). The sentence says whether the margin's arm is one the document recommends;
  whether Watch should say more about it is the editor's.
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
  so if the places split the cases between them at least one place misses too; and no figure key in the document
  names that read (a key whose path holds the option's arm beside the criterion or the measure it reads).
  ``options_dominance`` hands the same inference to the analyst, labelled inferred; a document that cites the read
  answers it, as a cited key answers a warning.
- ``count_in_words``: prose writes a count as words with no figure key: one count word (``zero`` to ``twenty``, as
  the ``words`` filter spells counts) or key, then ``of``, then another, where at least one of the two is a bare word
  (``four of five``, ``two of the three``). That is a count a reader reads as a number. A lone count word (``one
  arm``) reads as an article or a name, and is not one; nor is ``one`` before ``of``, which picks an item (``the only
  one of {n_candidates|words} configurations``).
- ``caveat_not_carried``: code attaches a caveat to a contrast a claim's evidence reads (a confound the contrast
  names, as ``caveats_list`` ids it: ``confound:<dimension>``, or ``apparatus:<dimension>`` where the apparatus
  confound is that dimension), and neither the claim nor the confidence carries it. A claim's evidence reads a
  contrast when one of its blocks reads the contrast's measure for its arm (the query names the measure, or reads the
  ``contrasts`` source with no measure named; it names the arm or every arm; it names the contrast's aspect or none),
  or a figure key in the claim's text or a caption names the contrast's arm and measure. A block reading every reading
  of an aspect does not name a contrast's measure, so it reads none; the editor's rubric (a finding called immune while
  a caveat on its evidence could move it is a must_fix) covers what this fact does not. A method-reach caveat landed
  on a Method term is carried too (it may land in Method alone). One fact per claim, contrast and caveat. Whether the
  caveat could move the claim, and which way, is the editor's.
- ``high_over_open``: the confidence's form is ``high_if_immaterial``, so the page's confidence line reads "High",
  conditional, above caveats the confidence leaves open: each one listed with its ``material_if`` and none of them one
  code proves immaterial (an open caveat carries no ``checked``). One fact for the confidence, naming how many and
  which. Code writes the form's words and checks only that the form matches its caveats (invariant 4); whether a high
  stands over these caveats, each clause saying what would make it matter and why the call holds until then, is the
  editor's, and the editor's rubric makes a high that does not stand a must_fix.

**What a fact is.** An :class:`EditorFact` is ``{kind, path, fact, figures}``. ``path`` is the document field the fact
is about (``recommends[0]``, ``decision[2]``, ``claims[1].text``), or empty for the document as a whole. ``fact`` is the
sentence, every figure in it written as the page writes its key; ``figures`` are those keys' paths, in the order the
sentence cites them, so the editor can read each with its tools. A sentence is built from parts, text and the keys
this module builds, and each key is written out on its own: a name the snapshot holds is text, never read for a key,
so a brace in a label, a name or an id is written as it is and never raises. The sentences are short, since the editor's user
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
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from threetears.evals.analysis.analyst.analyses import (
    Dominance,
    GateRead,
    breached_guardrails,
    dominance,
    gate_reads,
)
from threetears.evals.analysis.analyst.cited import cite
from threetears.evals.analysis.analyst.facts import confound_caveat_id, contrast_key
from threetears.evals.analysis.errors import UnresolvableReference
from threetears.evals.analysis.report.figure_keys import WEAKEST_PREFIX, Figure, format_figure, reader_name
from threetears.evals.analysis.report_decision import control_arm, floors_not_held, report_caveats, shown_better
from threetears.evals.analysis.report_input import Aspect, Contrast, Criterion, Reading, ReportInput
from threetears.evals.analysis.report_margins import Margin, weakest_margin
from threetears.evals.analysis.report_plan_checks import raised_warnings
from threetears.evals.kernel.report_doc import Claim, ReportDoc, report_doc_prose
from threetears.evals.kernel.report_plan import ChartQuery, WarningRef

#: The names with a caller outside this module: the writing loop's editor turn, whose user message carries the facts.
#: ``EditorFactKind`` joins them with its first production caller outside this module.
__all__ = ["EditorFact", "editor_facts"]

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
    "caveat_not_carried",
    "high_over_open",
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
    facts += _undecided_facts(doc, report_input, options, keys)
    facts += _count_facts(doc)
    facts += _caveat_facts(doc, report_input)
    facts += _high_facts(doc, report_input)
    return facts


# --- writing a fact ------------------------------------------------------------------------------------------------

#: A figure key as prose holds it: one pair of single braces, its path before any filter.
_KEY = re.compile(r"\{([^{}|]*)(?:\|[^{}]*)?\}")


@dataclass(frozen=True)
class _Key:
    """A figure key a fact's sentence cites, built here from what the input holds: its path, with no filter."""

    path: str


#: One part of a fact's sentence: text written as it is, or a key written as the page writes it.
_Part = str | _Key


def _fact(kind: EditorFactKind, path: str, parts: Sequence[_Part], report_input: ReportInput) -> EditorFact:
    """A fact from its sentence's parts: each text as it is, each key written out on its own and listed by path.

    The text is never read for keys, so a name the input holds that carries a brace (a label, a task's or aspect's
    name, a floor's key) is written as it is, and ``figures`` holds exactly the keys built here, each of which names
    something the input holds.
    """
    return EditorFact(
        kind=kind,
        path=path,
        fact="".join(_written(part, report_input) if isinstance(part, _Key) else part for part in parts),
        figures=[part.path for part in parts if isinstance(part, _Key)],
    )


def _written(key: _Key, report_input: ReportInput) -> str:
    """A key as the page writes it, with no filter (:func:`~threetears.evals.analysis.analyst.cited.cite`'s
    ``reads``): resolved by its path alone, never parsed out of braces, so an id that carries a brace reads too."""
    return cite(key.path, report_input).reads


def _arm(arm: str) -> _Key:
    """An arm as a fact cites it: its key, which the page writes as its name and tag."""
    return _Key(f"arm:{arm}")


def _bounded(name: str, path: str) -> bool:
    """Whether a key's path holds a name as whole dotted parts: at its start or after a dot, and before a dot or its
    end. Ids carry dots, so a name is matched whole, never split."""
    return re.search(rf"(?:^|\.){re.escape(name)}(?:\.|$)", path) is not None


# --- the keys a document cites -------------------------------------------------------------------------------------


def _cited_keys(doc: ReportDoc) -> list[str]:
    """The path of every figure key the document's prose cites, as written; whether each resolves is invariant 1's."""
    return [match.group(1) for _, text in report_doc_prose(doc) for match in _KEY.finditer(text)]


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
        parts: list[_Part] = [
            "Recommended ",
            _arm(arm),
            f" misses the declared {kind} {name}{_place(report_input, read)}: ",
            _Key(figure.key),
            " against ",
            _Key(threshold.key),
            ".",
        ]
        yield _fact("gate_failed", path, parts, report_input)


# --- a recommended arm and the warnings code raises on it -----------------------------------------------------------


def _names_warning(key: str, warning: WarningRef, arm: str) -> bool:
    """Whether a cited key's path names a warning's figure: a floor's by its key, a pair's by the other arm, a bar's
    by its measure, each beside the arm."""
    return _bounded(arm, key) and _bounded(warning.name, key)


def _warning_said(report_input: ReportInput, warning: WarningRef) -> list[_Part]:
    """What a warning says, in a sentence's parts: its floor's key, the arm ahead or the bar's measure as written."""
    aspect = _aspect_name(report_input, warning.aspect)
    if warning.kind == "behind_on":
        return [f"a pair reading on {aspect} shows ", _arm(warning.name), " ahead of it"]
    if warning.kind == "bar_missed":
        return [f"it misses the bar on {warning.name} on {aspect}"]
    return [f"floor {warning.name} on {aspect} reads {warning.kind.removeprefix('floor_')}"]


def _warning_facts(report_input: ReportInput, path: str, arm: str, keys: Sequence[str]) -> Iterator[EditorFact]:
    for warning in raised_warnings(report_input, arm):
        if any(_names_warning(key, warning, arm) for key in keys):
            continue
        parts: list[_Part] = [
            f"Code raises {warning.kind} on recommended ",
            _arm(arm),
            ": ",
            *_warning_said(report_input, warning),
            ". No figure key in the document names it.",
        ]
        yield _fact("warning_unanswered", path, parts, report_input)


def _aspect_name(report_input: ReportInput, aspect: str) -> str:
    """An aspect's reader name; a read may name an aspect the input holds no record for, which reads by its id."""
    if any(held.id == aspect for held in report_input.aspects):
        return reader_name("aspect", aspect, report_input)
    return aspect


# --- the call and code's -------------------------------------------------------------------------------------------

#: How code's modes read in a sentence.
_MODE_WORDS = {"decide": "a decision", "tradeoff": "a tradeoff", "no_call": "no call"}


def _arms_said(arms: Sequence[str], held: set[str]) -> list[_Part]:
    """Arms as a sentence names them: each held one by its key, any other by its id as the document wrote it."""
    if not arms:
        return ["no arm"]
    parts: list[_Part] = []
    for index, arm in enumerate(arms):
        parts += [", "] if index else []
        parts += [_arm(arm) if arm in held else f"`{arm}`"]
    return parts


def _call_facts(doc: ReportDoc, report_input: ReportInput, options: Dominance) -> Iterator[EditorFact]:
    call = options.code_call
    dominant = [arm.removeprefix("arm:") for arm in call.dominant]
    held = {arm.id for arm in report_input.arms}
    code: list[_Part]
    if len(dominant) == 1:
        if list(doc.recommends) == dominant:
            return
        code = [_arm(dominant[0]), " as the one option dominating"]
    elif "decide" not in call.modes and doc.recommends:
        modes = " or ".join(_MODE_WORDS[mode] for mode in call.modes)
        code = [f"code's call as {modes}, with no option dominating"]
    else:
        return
    parts: list[_Part] = [
        "options_dominance reads ",
        *code,
        "; the document recommends ",
        *_arms_said(doc.recommends, held),
        ".",
    ]
    yield _fact("call_vs_dominance", "recommends", parts, report_input)


# --- the weakest margin and Watch ----------------------------------------------------------------------------------


def _line_names(margin: Margin) -> list[str]:
    """What a key's path may name a margin's line by: the criterion and its measure, the aspect, or the floor's key."""
    return [name for name in (margin.source, margin.measure, margin.aspect) if name]


def _names_weakest(key: str, margin: Margin) -> bool:
    if key.startswith(WEAKEST_PREFIX):
        return True
    return _bounded(margin.arm, key) and any(_bounded(name, key) for name in _line_names(margin))


#: How the weakest margin reads in a sentence, by its keys.
_WEAKEST_SAID: tuple[_Part, ...] = (
    _Key("weakest.arm"),
    ", ",
    _Key("weakest.value"),
    " against ",
    _Key("weakest.criterion"),
    " at ",
    _Key("weakest.threshold"),
    ", a gap of ",
    _Key("weakest.margin"),
)


def _weakest_facts(doc: ReportDoc, report_input: ReportInput) -> Iterator[EditorFact]:
    margin = weakest_margin(report_input)
    if margin is None:
        return
    watch = [(f"decision[{index}]", entry.text) for index, entry in enumerate(doc.decision) if entry.slot == "watch"]
    cited = [match.group(1) for _, text in watch for match in _KEY.finditer(text)]
    if any(_names_weakest(key, margin) for key in cited):
        return
    where = "no Watch entry cites a key naming it" if watch else "the decision block has no Watch entry"
    whose = "an arm the document recommends" if margin.arm in doc.recommends else "no arm the document recommends"
    parts = ["The weakest margin is ", *_WEAKEST_SAID, f", on {whose}; {where}."]
    yield _fact("weakest_not_in_watch", watch[0][0] if watch else "decision", parts, report_input)


# --- switching to a candidate --------------------------------------------------------------------------------------


def _adoption_facts(report_input: ReportInput, path: str, arm: str) -> Iterator[EditorFact]:
    control = control_arm(report_input)
    if control is None or arm == control:
        return
    asked = [aspect for aspect in report_input.aspects if aspect.asked]
    named: list[_Part] = ["Recommended ", _arm(arm), " is a candidate against ", _arm(control)]
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
    yield _fact("adoption", path, [*named, f"; {said}."], report_input)


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
            parts: list[_Part] = [
                "Recommended ",
                _arm(arm),
                f"'s reading on {reader_name('aspect', aspect.id, report_input)}, which the campaign does not ask "
                "about, is a loss settled past the margin: its interval, ",
                _Key(f"{prefix}.reading.{arm}.interval"),
                " points, lies wholly at or below minus the margin of ",
                _Key(margin),
                " points.",
            ]
            yield _fact("settled_loss_unasked", path, parts, report_input)


# --- options a gate has no figure for ------------------------------------------------------------------------------


def _names_undecided(key: str, arm: str, criterion: Criterion) -> bool:
    """Whether a cited key's path names an undecided read: the arm, beside the criterion or the measure it reads."""
    return _bounded(arm, key) and (_bounded(criterion.id, key) or _bounded(criterion.measure, key))


def _undecided_facts(
    doc: ReportDoc, report_input: ReportInput, options: Dominance, keys: Sequence[str]
) -> Iterator[EditorFact]:
    recommended = {arm: f"recommends[{index}]" for index, arm in enumerate(doc.recommends)}
    criteria = {criterion.id: criterion for criterion in report_input.criteria}
    for row in options.undecided:
        arm = row.arm.removeprefix("arm:")
        for implied in row.implied:
            if any(_names_undecided(key, arm, criteria[implied.criterion]) for key in keys):
                continue
            threshold, figure = implied.figures[0], implied.figures[1]
            name = reader_name("criterion", implied.criterion, report_input)
            parts: list[_Part] = [
                _arm(arm),
                f" is undecided on {name}, which has no figure for it at some place it reads; its figure over every "
                "case, ",
                _Key(figure.key),
                ", misses ",
                _Key(threshold.key),
                ", so if the places split the cases at least one misses too (inferred, not stated). No figure key in "
                "the document names it.",
            ]
            yield _fact("undecided_implied", recommended.get(arm, ""), parts, report_input)


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
    for path, text in report_doc_prose(doc):
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


# --- a caveat code attaches to a contrast a claim reads ------------------------------------------------------------


def _block_reads(query: ChartQuery, contrast: Contrast) -> bool:
    """Whether a block's query reads a contrast's measure for its arm, on its aspect or none."""
    measure = contrast.measure in query.measures or (query.source == "contrasts" and not query.measures)
    arm = not query.arms or contrast.arm in query.arms
    aspect = not query.aspect or query.aspect == contrast.aspect
    return measure and arm and aspect


def _claim_reads(claim: Claim, contrast: Contrast) -> bool:
    """Whether a claim's evidence reads a contrast: a block reading it, or a key in its text or a caption naming its
    arm and measure."""
    if any(_block_reads(evidence.query, contrast) for evidence in claim.evidence):
        return True
    texts = [claim.text, *(evidence.caption for evidence in claim.evidence)]
    keys = [match.group(1) for text in texts for match in _KEY.finditer(text)]
    return any(_bounded(contrast.arm, key) and _bounded(contrast.measure, key) for key in keys)


def _measure_said(report_input: ReportInput, measure: str) -> str:
    """A contrast's measure, for a sentence: its reader name where the input holds the measure or judged dimension,
    else its key."""
    if measure in report_input.measures:
        return reader_name("measure", measure, report_input)
    if measure in report_input.judged:
        return reader_name("judged", measure, report_input)
    return f"`{measure}`"


def _resolves(path: str, report_input: ReportInput) -> bool:
    """Whether a key the fact would cite reads on the input; one that does not is left out of the sentence."""
    try:
        cite(path, report_input)
    except UnresolvableReference:
        # NOSILENT: an expected condition, not a fault: a contrast's measure the key grammar cannot name (or a figure
        # the input leaves unstated) is written in words, never as a key, so the fact still never refuses.
        return False
    return True


def _contrast_said(report_input: ReportInput, contrast: Contrast) -> list[_Part]:
    """A contrast as a sentence cites it: its arm and measure, then its change and interval by key where both read."""
    said: list[_Part] = [_arm(contrast.arm), f"'s contrast on {_measure_said(report_input, contrast.measure)}"]
    prefix = contrast_key(report_input, contrast)
    delta, interval = _Key(f"{prefix}.delta"), _Key(f"{prefix}.interval")
    if all(_resolves(key.path, report_input) for key in (delta, interval)):
        said += [", ", delta, " (", interval, ")"]
    return said


def _caveat_facts(doc: ReportDoc, report_input: ReportInput) -> Iterator[EditorFact]:
    assigned = report_caveats(report_input)
    method_reach = {caveat.id for caveat in assigned if caveat.reach == "method"}
    in_method = {caveat for item in doc.method for caveat in item.caveats if caveat in method_reach}
    carried_anywhere = {caveat.id for caveat in doc.confidence.caveats} | in_method
    ids = {caveat.id for caveat in assigned}
    for index, claim in enumerate(doc.claims):
        carried = carried_anywhere | set(claim.caveats)
        for contrast in report_input.contrasts:
            if not contrast.confounds or not _claim_reads(claim, contrast):
                continue
            for dimension in dict.fromkeys(contrast.confounds):
                caveat = confound_caveat_id(assigned, dimension)
                if caveat not in ids or caveat in carried:
                    continue
                parts: list[_Part] = [
                    f"The evidence of claim {claim.id} reads ",
                    *_contrast_said(report_input, contrast),
                    f", and code attaches caveat `{caveat}` to that contrast; neither the claim nor the confidence "
                    "carries it.",
                ]
                yield _fact("caveat_not_carried", f"claims[{index}].caveats", parts, report_input)


# --- a high confidence over caveats it leaves open -----------------------------------------------------------------


def _high_facts(doc: ReportDoc, report_input: ReportInput) -> Iterator[EditorFact]:
    if doc.confidence.form != "high_if_immaterial":
        return
    left = [caveat.id for caveat in doc.confidence.caveats if caveat.material_if is not None]
    if not left:
        return  # invariant 4's fault, not a fact: the form rests on no open caveat
    named = ", ".join(f"`{caveat}`" for caveat in left)
    count = "1 open caveat" if len(left) == 1 else f"{len(left)} open caveats"
    parts: list[_Part] = [
        f"The confidence reads high, if the caveats aren't material, over {count}, none of which code proves "
        f"immaterial: {named}."
    ]
    yield _fact("high_over_open", "confidence", parts, report_input)
