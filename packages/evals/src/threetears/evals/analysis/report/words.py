"""The words a report is read in — one vocabulary for every surface a page is read on.

A confidence tier, an evidence tier, a verdict and an arm are each spelled once, here, and the
:class:`~threetears.evals.analysis.report.model.Report` every door renders takes them from this module. Two spellings
of one arm on two surfaces would read as two arms.

**The first two screens speak a reader's words, not the engine's.** A trial is one case run once on one arm, and it
is passed, failed (the candidate's failure) or lost (the test setup's). The engine's own nouns for the same things —
a cell, an observation, an exclusion, the rig, the harness, the apparatus, a reserved judge axis's stored id — never
reach those screens; :data:`ENGINE_WORDS` maps each to what a reader is told instead, and
:data:`FIRST_SCREEN_BANNED` is what a test holds the rendered screens to. No arm is ever called a winner.
"""

from __future__ import annotations

import re
import typing
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

from threetears.evals.analysis.arms import (
    ELISION,
    LABEL_LEVEL_CHARS,
    applicable_levers,
    arm_names,
    distinguishing_axes,
    elide_level,
    naming_levels,
)
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, ComparisonVerdict
from threetears.evals.kernel.analysis_measures import BarDecision
from threetears.evals.analysis.stats import UNIFORM_MOVE_NEEDS_RANGE
from threetears.evals.analysis.style import (
    arm_display_name,
    arm_name,
    cannot_tell_apart,
    lever_reader_name,
    humanize,
    measure_name,
)
from threetears.evals.kernel.host.kinds import EMPTY_VALUE_DISPLAY, NO_VALUE_DISPLAY
from threetears.evals.kernel.campaign import (
    ConfidenceTier,
)
from threetears.evals.kernel.declaration import CampaignArm, resolve_control_arm
from threetears.evals.kernel.metrics import ACCURACY_MEASURE, MATCH_MEASURE, goal_check_of
from threetears.evals.kernel.surface import DecisionSurface, GuardrailDecision
from threetears.evals.schema.models import OUTCOME_DIM_ID, TRANSCRIPT_DIM_ID

if TYPE_CHECKING:
    from threetears.evals.analysis.campaign_bundle import ArmSpelling, CampaignBundle


def _literal_values(annotation: Any) -> frozenset[str]:
    """Every value a ``Literal`` annotation admits, through an optional ``| None``."""
    if typing.get_origin(annotation) is Literal:
        return frozenset(typing.get_args(annotation))
    return frozenset(value for arg in typing.get_args(annotation) for value in _literal_values(arg))


def worded(words: dict[str, str], annotation: Any, what: str) -> dict[str, str]:
    """Refuse at import a word table that does not cover exactly the values its field admits.

    A value added to the model and not here would otherwise reach the page as its raw token — the very vocabulary
    a reader's layout exists to keep from them.

    :param words: Value → its words.
    :ptype words: dict[str, str]
    :param annotation: The ``Literal`` the values come from.
    :ptype annotation: Any
    :param what: What the values are, for the refusal.
    :ptype what: str
    :return: ``words``, unchanged.
    :rtype: dict[str, str]
    :raises RuntimeError: The table and the ``Literal`` disagree.
    """
    if set(words) != _literal_values(annotation):
        raise RuntimeError(
            f"the page's words for {what} {sorted(words)} do not match the model's {sorted(_literal_values(annotation))}"
        )
    return words


#: The engine's nouns, and what a reader of a report's first two screens is told in their place.
ENGINE_WORDS: dict[str, str] = {
    "cell": "arm",
    "observation": "trial",
    "excluded": "lost to the test setup",
    "rig": "test setup",
    "harness": "test setup",
    "apparatus": "test setup",
    "__outcome__": "outcome",
    "__transcript__": "transcript",
}

#: Words that never appear on a report's first two screens: the engine's nouns, and "winner", which no arm is.
FIRST_SCREEN_BANNED: tuple[str, ...] = (*ENGINE_WORDS, "winner")

#: A confidence tier as a reader says it.
CONFIDENCE_WORDS = worded(
    {"very_high": "very high", "high": "high", "medium": "medium", "low": "low"}, ConfidenceTier, "a confidence tier"
)

#: What a contrast's test said against the control, as a reader says it: each arm by its label and the direction by
#: name, never "the control", which a reader took for a place ("improved on the control" read as "in the control").
#: Filled by :func:`comparison_words`.
COMPARISON_VERDICT_WORDS = worded(
    {
        "improved": "{arm} better than {control}{better_end}",
        "regressed": "{arm} worse than {control}{better_end}",
        "equivalent": "{arm} and {control} equivalent, within the measure's margin",
        "not_separated": cannot_tell_apart(["{arm}"], "{control}"),
        "untested": "untested",
    },
    ComparisonVerdict,
    "a comparison verdict",
)

#: A measure's better end, in brackets, where it is not the higher one a reader assumes.
LOWER_IS_BETTER = " (lower is better)"

#: Each reason a bundle stores in the engine's words, as a page says it. A stored reason is part of the evidence a
#: bundle's fingerprint and evidence digest read, so its text never changes once stored: rewording it in the engine
#: would move every stored campaign carrying it and make each host regenerate its reports for a wording change. The
#: stored text stays byte for byte as written, and a page rewords it here, at render time (:func:`reason_words`).
_STORED_REASON_WORDS: dict[str, str] = {
    UNIFORM_MOVE_NEEDS_RANGE: (
        "declare value_range on this measure for a uniform move to be tested: every case moved by the same amount, "
        "and with no declared range no test of a mean can tell that from no move (the exact sign-flip test asks "
        "whether the move is symmetric about zero, not whether the mean moved, and an unbounded value can hide a rare "
        "large move the other way), so the evidence cannot tell the two apart, and nothing here says they are alike"
    ),
}


#: A campaign reading's stored reason for a superiority row whose interval reaches zero
#: (``its interval reaches zero, so v2 and v1 are not separated``), stored with an analysis's readings (its frame):
#: the arms it names, ``{arm} and {control}``, are kept and said in the glossary's sentence (:func:`cannot_tell_apart`).
_STORED_UNSEPARATED = re.compile(r"its interval reaches zero, so (?P<arms>.+?) are not separated")

#: A stored reason naming the level a corrected family read a row at (``clears zero at 95%, not at the 97.5% its gains
#: are read at, corrected across v2's aspects``): a page says what the correction does, never the adjusted level,
#: which is the Method's (owner ruling, level in headings).
_STORED_CORRECTED_LEVEL = re.compile(
    r"clears zero at (?P<nominal>\S+%), not at the \S+% its (?P<side>gain|drop)s are read at, corrected across "
)


#: A family's stored disclosure naming its per-interval level beside its joint one (``Each interval is at 99.55%, so
#: the family's intervals hold together at 95%.``).
_STORED_PER_INTERVAL = re.compile(
    r"Each interval is at \S+%, so the family's intervals hold together at (?P<joint>\S+%)\."
)


def disclosure_words(disclosure: str) -> str:
    """A family's disclosure, as the bundle stores it, as a page says it: what the family's intervals guarantee
    together, never each interval's own adjusted level, which is the Method's and the glossary's (owner ruling, level
    in headings). Any other disclosure reads unchanged.

    :param disclosure: The disclosure as stored.
    :ptype disclosure: str
    :return: The disclosure in the page's words.
    :rtype: str
    """
    return _STORED_PER_INTERVAL.sub(r"The family's intervals hold together at \g<joint>.", disclosure)


def reason_words(reason: str | None) -> str | None:
    """A reason a bundle stores, as a page says it: each stored text :data:`_STORED_REASON_WORDS` rewords in the
    reader's words wherever it appears, a reading's stored "are not separated" sentence (:data:`_STORED_UNSEPARATED`)
    says the evidence cannot tell its arms apart, and any other reason reads unchanged.

    :param reason: The reason as the bundle stores it, or None where it stores none.
    :ptype reason: str | None
    :return: The reason in the page's words, or None.
    :rtype: str | None
    """
    if reason is None:
        return None
    for stored, said in _STORED_REASON_WORDS.items():
        reason = reason.replace(stored, said)
    reason = _STORED_CORRECTED_LEVEL.sub(r"clears zero at \g<nominal> on its own, not once corrected across ", reason)
    return _STORED_UNSEPARATED.sub(r"its interval reaches zero, so the evidence cannot tell \g<arms> apart", reason)


def comparison_words(verdict: ComparisonVerdict, arm: str, control: str, *, higher_is_better: bool = True) -> str:
    """A contrast's verdict in words, each arm by its label: ``v2 better than v1``, ``v2 worse than v1 (lower is
    better)``, ``the evidence cannot tell v2 and v1 apart``.

    :param verdict: What the contrast's test came to.
    :ptype verdict: ComparisonVerdict
    :param arm: The arm read against the control, as the page names it.
    :ptype arm: str
    :param control: The control, as the page names it.
    :ptype control: str
    :param higher_is_better: The reading's direction; a lower-is-better reading's better end is said in brackets.
    :ptype higher_is_better: bool
    :return: The words.
    :rtype: str
    """
    better_end = "" if higher_is_better else LOWER_IS_BETTER
    return COMPARISON_VERDICT_WORDS[verdict].format(arm=arm, control=control, better_end=better_end)


#: What a guardrail came to for an arm against the control, as a reader says it, each arm by its label. Filled by
#: :func:`guardrail_words`.
GUARDRAIL_DECISION_WORDS = worded(
    {
        "held": "held: {arm} shown no worse than {control} by more than the margin",
        "breached": "breached: {arm} shown worse than {control} by more than the margin",
        "undecided": "undecided: {arm} may be worse than {control} by more than the margin",
    },
    GuardrailDecision,
    "a guardrail decision",
)


def guardrail_words(decision: GuardrailDecision, arm: str, control: str) -> str:
    """A guardrail's decision in words, each arm by its label: ``held: v2 shown no worse than v1 by more than the
    margin``."""
    return GUARDRAIL_DECISION_WORDS[decision].format(arm=arm, control=control)


#: What a bar came to on one arm, as a reader says it.
BAR_DECISION_WORDS = worded(
    {
        "cleared": "cleared: the interval lies wholly on the good side of the bar, less its margin",
        "missed": "missed: the interval lies wholly on the bad side of the bar, less its margin",
        "undecided": "undecided: the interval straddles the bar, so not shown cleared",
        "no_interval": "no interval: fewer than two trials, so not read",
        "no_data": "no data: no trial of the measure",
    },
    BarDecision,
    "a bar decision",
)


def reading_name(name: str) -> str:
    """A judged dimension or measure key as a reader reads it: a reserved judge axis by its plain name."""
    return ENGINE_WORDS.get(name, name)


def reading_heading(
    surface: DecisionSurface, name: str, reading: str = "measure", *, declared: str | None = None
) -> str:
    """A reading as the page heads it, in a column, a row or a chart's title.

    A goal check by its name on its aspect, declared else what it tests (``Refund amount check``),
    never its expression, which the page's glossary defines the name by
    (:func:`~threetears.evals.analysis.style.measure_name`); how
    each trial ended (a ``trial`` reading) as :data:`PASS_RATE`; any other reading by the
    surface's heading, a reserved judge axis by its plain name (:func:`reading_name`).

    :param surface: The decision surface the reading is on.
    :ptype surface: DecisionSurface
    :param name: The measure's key, or a judged dimension's name.
    :ptype name: str
    :param reading: ``judged`` for a judged dimension, ``trial`` for the overall pass rate; anything else reads a
        measure.
    :ptype reading: str
    :param declared: A goal check's declared display name (``AnalysisContextBundle.check_name``), printed in place of
        its expression; None for none declared.
    :ptype declared: str | None
    :return: The heading.
    :rtype: str
    """
    if reading == "trial":
        return PASS_RATE
    if reading != "judged" and goal_check_of(name) is not None:
        return measure_name(name, declared=declared)
    if reading == "judged":
        return judged_heading(name)
    return reading_name(surface.measure_heading(name, reading))


#: The overall pass rate as every surface names it: a name, capitalised as every other reading's is.
PASS_RATE = "Pass rate"


def judged_heading(name: str) -> str:
    """A judged dimension as every surface names it: its name as :func:`~threetears.evals.analysis.style.measure_name`
    spells it, marked judged (``reply stays on topic (judged)``, ``outcome (judged)``), never its stored id.

    :param name: The dimension, as the rubric names it.
    :ptype name: str
    :return: The name.
    :rtype: str
    """
    return f"{judged_name(name)} (judged)"


#: How a trial's drill-down names a reserved judge axis, where a reader sees it: never by its stored id.
_RESERVED_AXIS_WORDS: dict[str, str] = {TRANSCRIPT_DIM_ID: "transcript", OUTCOME_DIM_ID: "outcome"}


def trial_dimension(dimension: str) -> str:
    """A judged dimension as a trial's drill-down and its examples name it: a reserved axis by its word (``outcome``),
    any other by its id as the rubric names it. A bundle's trials hold every dimension by its id
    (:attr:`~threetears.evals.analysis.bundle.schema.TrialJudgement.dim`); this names it when a page is rendered.

    :param dimension: The dimension's id.
    :ptype dimension: str
    :return: What a trial prints for it.
    :rtype: str
    """
    return _RESERVED_AXIS_WORDS.get(dimension, dimension)


def judged_name(dimension: str, declared: str | None = None) -> str:
    """A judged dimension's name, the one rule every surface names it by: the label its input declares for it
    (``Judged.label``) wherever one is declared, on a chart, a table, a figure key and in prose alike; else its name as
    :func:`~threetears.evals.analysis.style.measure_name` spells it (``reply stays on topic``). Unmarked: a heading adds
    the judged mark (:func:`judged_heading`), and a label its capital (:func:`label_case`).

    :param dimension: The dimension, as the rubric names it.
    :ptype dimension: str
    :param declared: The label its input declares, or None for none.
    :ptype declared: str | None
    :return: The name.
    :rtype: str
    """
    return declared or measure_name(dimension)


def _judged_label(heading: str) -> str:
    """A judged dimension's heading as a table prints it in a row or a header: the heading :func:`judged_heading`
    writes (a score's scale or a ruler's words after it where the table adds them), starting with a capital, as every
    measure's label does where it is a label rather than words in a sentence
    (:func:`~threetears.evals.analysis.report.parts.charting.measure_words`), so a judged row reads in the register of
    the measure rows around it (``Reply polite (judged)`` beside ``Lost-call rate``) and a table names a dimension as
    the page's charts do. The printed text only: a table's cell keeps the heading as its value, which a program reads.

    :param heading: The judged dimension's heading.
    :ptype heading: str
    :return: The label.
    :rtype: str
    """
    return label_case(heading)


def label_case(text: str) -> str:
    """Words as a label prints them, where they are a label rather than words in a sentence (a header, a chart's title,
    a series, a table's row): the first letter a capital, the rest as written. The one owner of a label's capital:
    :func:`_judged_label` and :func:`~threetears.evals.analysis.report.parts.charting.measure_words` both take it.

    :param text: The words.
    :ptype text: str
    :return: The label.
    :rtype: str
    """
    return text[:1].upper() + text[1:]


#: A judged dimension's heading as :func:`judged_heading` writes it, a score's scale or a ruler's words after its mark
#: where a table adds them (``outcome (judged, 1–5)``).
_JUDGED_HEADING = re.compile(r".+ \(judged(?:, [^()]+)?\)")


#: The reading a table by stratum opens each arm with: how many cases and trials each column rests on. A program reads
#: the row by it (``reading == "cases"``); a page prints it as the label it is (:func:`table_label`).
CASES_READING = "cases"


def table_label(text: str) -> str:
    """A table cell's words as a page prints them (the one rule every serializer's table applies, at render time): a
    judged dimension's heading (:func:`_judged_label`), and the cases reading a table by stratum opens each arm with
    (:data:`CASES_READING`), starting with a capital, so each reads as the measure rows around it do; any other words
    as written. The cell keeps its words as its value, which a program reads and looks a row up by.

    :param text: The cell's words.
    :ptype text: str
    :return: The words printed.
    :rtype: str
    """
    if text == CASES_READING:
        return label_case(text)
    return _judged_label(text) if _JUDGED_HEADING.fullmatch(text) else text


def given_name(identifier: str, declared: str | None = None) -> str:
    """What a host may name (an aspect, a stratum, a task) as every surface prints it, the one rule for each: its
    declared display name, else its id with its underscores and dots as spaces (``edge case``). The printed name only:
    a program reads each by its id.

    :param identifier: The id.
    :ptype identifier: str
    :param declared: Its declared display name, or None for none.
    :ptype declared: str | None
    :return: The name.
    :rtype: str
    """
    return declared or humanize("dimension", identifier)


def stratum_name(stratum: str, declared: str | None = None) -> str:
    """A stratum as every surface prints it (:func:`given_name`). Where other strata sit beside it, two that would
    print alike each print as their id (:func:`stratum_names`).

    :param stratum: The stratum's id.
    :ptype stratum: str
    :param declared: Its declared display name, or None for none.
    :ptype declared: str | None
    :return: The name.
    :rtype: str
    """
    return given_name(stratum, declared)


def aspect_stratum_key(aspect: str, stratum: str) -> str:
    """The key a stratum held on more than one aspect, named differently there, is named by on its aspect: the stem its
    figure keys take there (``aspects.<aspect>.strata.<id>``). A stratum every aspect names alike is named by its id.

    :param aspect: The aspect's id.
    :ptype aspect: str
    :param stratum: The stratum's id.
    :ptype stratum: str
    :return: The key.
    :rtype: str
    """
    return f"aspects.{aspect}.strata.{stratum}"


def stratum_names(strata: Iterable[str], declared: Mapping[str, str | None] | None = None) -> dict[str, str]:
    """Each stratum's printed name (:func:`stratum_name`), by id, in the order given, where no two strata print alike:
    two ids whose names would read the same (``off_lane`` and ``off.lane`` both ``off lane``) each print as the id
    itself, so a reader never sees one stratum where the cases declare two.

    :param strata: The strata's ids.
    :ptype strata: Iterable[str]
    :param declared: Each stratum's declared display name, by id, where it declares one.
    :ptype declared: Mapping[str, str | None] | None
    :return: Each id's printed name.
    :rtype: dict[str, str]
    """
    named = {stratum: stratum_name(stratum, (declared or {}).get(stratum)) for stratum in dict.fromkeys(strata)}
    counts: dict[str, int] = {}
    for name in named.values():
        counts[name] = counts.get(name, 0) + 1
    return {stratum: name if counts[name] == 1 else stratum for stratum, name in named.items()}


#: The verb a counted reading's figure takes in a sentence, by its name as the page heads it: only the pass rate's
#: trials "passed" (the glossary's "passed"); a classifier's accuracy counts the trials it was right on.
_READING_VERBS: dict[str, str] = {PASS_RATE: "passed", "Accuracy": "was right"}

#: The verb any other counted reading takes: a pass/fail check's or score's trials it was met on. Neutral, and no
#: glossary term: "held" is a guardrail's decision (held or breached), so a lead's figure never borrows it.
_MET = "was met"


def reading_verb(heading: str) -> str:
    """The verb a counted reading's figure takes in a sentence (:data:`_READING_VERBS`), never "passed" for a reading
    that is not the pass rate.

    :param heading: The reading, as the page heads it.
    :ptype heading: str
    :return: The verb.
    :rtype: str
    """
    return _READING_VERBS.get(heading, _MET)


def one_figure_each(measures: Iterable[str], *, read: Iterable[str] = ()) -> list[str]:
    """``measures`` with each figure listed once: a classifier's per-trial label match is its accuracy, so where the
    accuracy is read the match is not listed under a second name.

    :param measures: Measure keys, in order.
    :ptype measures: Iterable[str]
    :param read: Every measure the page reads elsewhere, whose accuracy names the match where ``measures`` lacks it.
    :ptype read: Iterable[str]
    :return: The keys, in order, the match left out where the accuracy is read.
    :rtype: list[str]
    """
    held = list(measures)
    if ACCURACY_MEASURE not in held and ACCURACY_MEASURE not in set(read):
        return held
    return [key for key in held if key != MATCH_MEASURE]


def positions(numbers: list[int]) -> str:
    """Finding positions as a reader counts them: from one, each named ``F<n>``."""
    return ", ".join(f"F{position + 1}" for position in numbers)


#: The control's letter tag, which no candidate wears.
_CONTROL_TAG = "P"


def arm_tags(variant_keys: Sequence[str], control: str | None) -> dict[str, str]:
    """Each arm's letter tag (GENERATOR.md §6.1): ``P`` for the control, then ``A``, ``B``, ``C``… for the others in
    the order given (``AA`` after ``Z``), skipping ``P``, which is the control's alone, so every chart of one page tags
    an arm alike and colours it by its tag.

    :param variant_keys: The arms, in the page's order.
    :ptype variant_keys: Sequence[str]
    :param control: The control's variant key; None where there is none.
    :ptype control: str | None
    :return: Variant key -> tag, for every arm given.
    :rtype: dict[str, str]
    """
    tags: dict[str, str] = {}
    place = 0
    for key in variant_keys:
        if key == control:
            tags[key] = _CONTROL_TAG
            continue
        tag = _CONTROL_TAG
        while tag == _CONTROL_TAG:
            place += 1
            tag, number = "", place
            while number:
                number, letter = divmod(number - 1, 26)
                tag = chr(ord("A") + letter) + tag
        tags[key] = tag
    return tags


def campaign_arm_tags(
    bundles: Iterable[AnalysisContextBundle],
    control: str | None,
    *,
    arms: Iterable[str] = (),
    also: Iterable[str] = (),
    label_of: Callable[[str], str] | None = None,
) -> dict[str, str]:
    """Each arm's letter tag across a campaign's pages (:func:`arm_tags`), in the one order every page tags arms in:
    the campaign's own arms as it declares them (``arms``), then the other variants its aspects index, aspect by
    aspect, then any its scoreboards list besides, then ``also``'s. It is the order a campaign's report input lists
    its arms in (``report_input_build``), so the code page, every aspect's page and the report document's page tag
    and colour one arm alike (the review's P40); an arm one aspect lacks shifts no other arm's tag.

    :param bundles: Each aspect's bundle, in the campaign's order.
    :ptype bundles: Iterable[AnalysisContextBundle]
    :param control: The campaign's control's variant key; None where there is none.
    :ptype control: str | None
    :param arms: The campaign's arms' variant keys, in its declared order; none for a page with no campaign.
    :ptype arms: Iterable[str]
    :param also: Arms to tag after the others, where they are not among them.
    :ptype also: Iterable[str]
    :param label_of: Each variant key's arm, where several keys name one arm (the campaign bundle's ``label_of``), so
        keys of one arm share its tag; None where each key is its own arm.
    :ptype label_of: Callable[[str], str] | None
    :return: Variant key -> tag.
    :rtype: dict[str, str]
    """
    held = list(bundles)
    listed = dict.fromkeys(
        [
            *arms,
            *(entry.variant_key for bundle in held for entry in bundle.variant_index),
            *(board.variant_key for bundle in held for board in bundle.scoreboard),
            *also,
        ]
    )
    named = label_of or (lambda key: key)
    first: dict[str, str] = {}
    for key in listed:
        first.setdefault(named(key), key)
    tagged = arm_tags(list(first.values()), first.get(named(control)) if control is not None else None)
    return {key: tagged[first[named(key)]] for key in listed}


def declared_arm_names(arms: Iterable[CampaignArm]) -> dict[str, str]:
    """The short name each arm declares (``CampaignArm.short_name``), by every variant key that names its cells: its
    own, and those it resolves to under this build's identity predicate. A declared name is what a page prints for
    the arm in its tight places, ahead of any short name a page would compute (GENERATOR.md §6.1); an arm declaring
    none is absent.

    :param arms: A campaign's arms.
    :ptype arms: Iterable[CampaignArm]
    :return: Variant key -> declared short name.
    :rtype: dict[str, str]
    """
    return {
        key: arm.short_name
        for arm in arms
        if arm.short_name is not None
        for key in (arm.variant_key, *resolve_control_arm(arm).variant_keys)
    }


def printed_arm_labels(bundle: CampaignBundle) -> dict[str, str]:
    """The name every page prints for each arm of a campaign, by every variant key a label names: the one display-name
    rule (:func:`~threetears.evals.analysis.style.arm_display_name`) every surface reads, the doc page's ``Arm.name``,
    the campaign page's folds and readings, and the plan page alike.

    "Computed" is read from data: an arm no arm of the campaign's declaration covers is labelled by code. One code
    spelled from several levers is printed from the spelling the bundle carries for it
    (:class:`~threetears.evals.analysis.campaign_bundle.ArmSpelling`: each lever by id and its level), each lever by the
    name every page calls it (``lever_names``), never by reading its frozen label's text back; the bundle refuses a
    spelling that does not spell the label. Any other arm, a person's or one labelled by a single level, keeps its label
    as written. The labels themselves, which name the arms as ids and are frozen and fingerprinted with the evidence,
    are unchanged: a name is applied here, at render time only.

    :param bundle: The campaign's evidence.
    :ptype bundle: CampaignBundle
    :return: Variant key -> the name a page prints.
    :rtype: dict[str, str]
    """
    return {
        key: arm_display_name(label, levers, bundle.lever_names)
        for key, (label, levers) in _named_labels(bundle).items()
    }


def _named_labels(bundle: CampaignBundle) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Each arm's label by every variant key a label names, a derived arm code spelled from several levers printed
    from its spelling with each lever named as every page names it (:attr:`CampaignBundle.arm_spellings`), and the
    levers code spelled it from (none for a label a person gave): what :func:`printed_arm_labels` prints by the display
    rule, and :func:`reader_arms` names an arm by."""
    given = {key for arm in bundle.declaration.arms for key in bundle.arm_keys(arm)}
    entries = {entry.variant_key: entry for aspect in bundle.aspects for entry in aspect.variant_index}
    spelled = arm_names(list(entries.values()))
    spellings = {
        key: spelling
        for arm in bundle.arms
        if (spelling := bundle.arm_spellings.get(arm.variant_key)) is not None
        for key in bundle.arm_keys(arm)
    }
    keys = [*(key for arm in bundle.arms for key in bundle.arm_keys(arm)), *entries]
    labels: dict[str, tuple[str, tuple[str, ...]]] = {}
    for key in dict.fromkeys(keys):
        if (spelling := spellings.get(key)) is not None:
            labels[key] = (spelling.label(bundle.lever_names), tuple(spelling.levers))
            continue
        levers = arm_name(spelled[key]).levers if key in spelled and key not in given else ()
        labels[key] = (bundle.label_of(key), levers)
    return labels


# --- an arm's reader name: what a page prints for a computed name that holds a cut level or a placeholder ---------
#
# A computed arm name is spelled from its levels (:func:`~threetears.evals.analysis.arms.arm_names`), and that spelling
# is frozen with the evidence: a derived arm's label, the bundle's arms and the census order read it, so it never
# changes. What a reader is shown is decided here, at render time only (owner ruling, 2026-11-07): a level too long
# to print whole is never shown cut, and the engine's placeholder for an empty text level never shows at all.

#: The engine's placeholders for an empty level (:mod:`~threetears.evals.kernel.host.kinds`).
_PLACEHOLDERS = frozenset({NO_VALUE_DISPLAY, EMPTY_VALUE_DISPLAY})

#: What tells two arms that would read alike apart in a computed name: a digest (``(arm 0123456789ab)``, or the bare
#: ``(0123456789ab)`` a page adds) or a taken label's number (``(2)``). A lettered name needs neither.
_TELL_APART = re.compile(r"^(?P<name>.*?)(?P<apart> \((?:arm )?[0-9a-f]{12,64}\)| \(\d+\))?$")

#: How an empty text level reads.
EMPTY_LEVEL = "no {lever}"


def _whole(display: str) -> str:
    return " ".join(display.split())


def _too_long(display: str) -> bool:
    """Whether a level cannot be printed whole in a name: past the cut, or cut already by its host."""
    shown = _whole(display)
    return len(shown) > LABEL_LEVEL_CHARS or shown.endswith(ELISION)


def text_levers(levels: Iterable[Mapping[str, object]]) -> frozenset[str]:
    """The levers that hold text among some arms' levels: one some arm carries at a level that reads as words (a
    prompt, instructions), never a number or a single token. Only such a lever's empty level means the arm carries
    nothing on it; an empty ``temperature`` is the provider's default, which ``no temperature`` would misstate.

    :param levels: Each arm's levels, lever to display.
    :ptype levels: Iterable[Mapping[str, object]]
    :return: The text levers.
    :rtype: frozenset[str]
    """
    return frozenset(
        axis
        for held in levels
        for axis, display in held.items()
        if isinstance(display, str) and _whole(display) not in _PLACEHOLDERS and " " in _whole(display)
    )


def _read_part(
    part: str, levels: Mapping[str, object], text: Collection[str], declared: Mapping[str, str] | None
) -> tuple[str, bool] | None:
    """One part of a computed name as a reader reads it, and whether it was a cut: a cut level as its lever alone, an
    empty text level as ``no <lever>``, each lever by its stored reader name (``declared``); None for a part no level of
    ``levels`` spells."""
    for axis, display in levels.items():
        if not isinstance(display, str):
            continue
        shown = _whole(display)
        reader = lever_reader_name(axis, declared)
        spelled = {elide_level(shown), f"{reader} {elide_level(shown)}"}
        if part not in spelled and part[:1].lower() + part[1:] not in spelled:
            continue
        if _too_long(shown):
            return reader, True
        if shown in _PLACEHOLDERS and axis in text:
            return EMPTY_LEVEL.format(lever=reader), False
    return None


def _spelled_reader_name(
    name: str,
    spelling: ArmSpelling,
    levels: Mapping[str, object],
    *,
    text: Collection[str],
    declared: Mapping[str, str] | None = None,
) -> str:
    """What a page prints for a derived arm code spelled from several levers, read from its spelling lever by lever,
    never from its printed name's text: :func:`reader_arm_name`'s rule, each part matched by position. A level too long
    to print whole is printed as its lever alone, and then the suffix that told the arm from another spelled alike goes
    (:func:`told_apart` tells such arms apart by their tags); an empty level of a text lever reads ``no <lever>``.
    Where no level reads otherwise, ``name`` is printed as it is.

    :param name: The name the page would print (:func:`printed_arm_labels`).
    :ptype name: str
    :param spelling: How code spelled the arm (:attr:`~threetears.evals.analysis.campaign_bundle.CampaignBundle.
        arm_spellings`).
    :ptype spelling: ArmSpelling
    :param levels: The arm's levels, whole, by lever id.
    :ptype levels: Mapping[str, object]
    :param text: The campaign's text levers.
    :ptype text: Collection[str]
    :param declared: Each lever's stored reader name, by id (:func:`~threetears.evals.analysis.style.lever_reader_name`).
    :ptype declared: Mapping[str, str] | None
    :return: The name a reader is shown.
    :rtype: str
    """
    parts: list[str] = []
    cut = moved = False
    for lever, level in zip(spelling.levers, spelling.levels, strict=True):
        reader = lever_reader_name(lever, declared)
        display = levels.get(lever)
        shown = _whole(display) if isinstance(display, str) else None
        if shown is not None and _too_long(shown):
            parts.append(reader)
            cut = moved = True
        elif shown is not None and shown in _PLACEHOLDERS and lever in text:
            parts.append(EMPTY_LEVEL.format(lever=reader))
            moved = True
        else:
            parts.append(f"{reader} {level}")
    if not moved:
        return name
    words = ", ".join(parts)
    # A computed name opens upper case, as every page prints it (``arm_display_name``).
    words = words[:1].upper() + words[1:]
    return words if cut else words + spelling.suffix


def reader_arm_name(
    name: str, levels: Mapping[str, object], *, text: Collection[str], declared: Mapping[str, str] | None = None
) -> str:
    """What a page prints for an arm whose name code computed from its levels, at render time only.

    A level too long to print whole (past :data:`~threetears.evals.analysis.arms.LABEL_LEVEL_CHARS`, or cut by its
    host) is printed as its lever alone, ``Extractor instructions``, never as a cut of itself; the digest or number a
    frozen name carried to tell it from another arm cut alike goes, and :func:`told_apart` tells such arms apart by
    their tags. An empty level of a text lever (:func:`text_levers`) reads ``no <lever>``: ``no system prompt``, never
    ``(none)``. Any other name, and a name a person gave, is printed as it is.

    :param name: The name as the page would print it (a frozen label through
        :func:`~threetears.evals.analysis.style.arm_display_name`).
    :ptype name: str
    :param levels: The levels the name may be spelled from, lever to display.
    :ptype levels: Mapping[str, object]
    :param text: The campaign's text levers.
    :ptype text: Collection[str]
    :param declared: Each lever's stored reader name, by id (:func:`~threetears.evals.analysis.style.lever_reader_name`).
    :ptype declared: Mapping[str, str] | None
    :return: The name a reader is shown.
    :rtype: str
    """
    held = _TELL_APART.fullmatch(name)
    base, apart = (held["name"], held["apart"] or "") if held is not None else (name, "")
    # The name read whole first: a single level may itself hold the separator (a prompt's text).
    parts = [base] if _read_part(base, levels, text, declared) is not None else base.split(", ")
    read = [_read_part(part, levels, text, declared) for part in parts]
    if all(one is None for one in read):
        return name
    cut = any(one is not None and one[1] for one in read)
    words = ", ".join(one[0] if one is not None else part for part, one in zip(parts, read, strict=True))
    # A name opening on a lever's reader name opens upper case, as every computed name does (``arm_display_name``).
    if name[:1].isupper() or (read[0] is not None and read[0][1]):
        words = words[:1].upper() + words[1:]
    return words if cut else words + apart


#: How a name several arms would print alike is told apart: by the arm's tag, as a figure key writes it.
TOLD_APART = "{name} ({tag})"


def told_apart(names: Mapping[str, str], tags: Mapping[str, str], read: Collection[str]) -> dict[str, str]:
    """Names with every one of ``read`` that another arm also prints told apart by its tag (:data:`TOLD_APART`,
    ``Extractor instructions (B)``), the tag the page prints that arm by; keys of one arm share a tag, and are one arm.

    :param names: Each arm's name, by key (a variant key, or a report input's arm id).
    :ptype names: Mapping[str, str]
    :param tags: Each arm's tag on the page, by the same key.
    :ptype tags: Mapping[str, str]
    :param read: The keys whose names :func:`reader_arm_name` rewrote; only they are told apart, the rest as they are.
    :ptype read: Collection[str]
    :return: The names.
    :rtype: dict[str, str]
    """
    worn: dict[str, set[str | None]] = {}
    for key, name in names.items():
        worn.setdefault(name, set()).add(tags.get(key))
    return {
        key: TOLD_APART.format(name=name, tag=tags[key])
        if key in read and key in tags and len(worn[name]) > 1
        else name
        for key, name in names.items()
    }


def reader_arm_labels(bundle: CampaignBundle, tags: Mapping[str, str]) -> dict[str, str]:
    """The name a code-written surface prints for each arm of a campaign: :func:`printed_arm_labels`, each computed name
    read by :func:`reader_arm_name`, and told apart by its tag among ``tags`` where two arms then read alike
    (:func:`told_apart`). Render time only: the labels the bundle froze, its fingerprint and the order it lays arms out
    in are read under the frozen spelling, and a report input's arm names stay :func:`printed_arm_labels`' (a
    snapshot's id is its content's digest).

    :param bundle: The campaign's evidence.
    :ptype bundle: CampaignBundle
    :param tags: Each arm's tag on the surface, by variant key: the code page's (:func:`campaign_arm_tags`), or the
        document page's (:func:`input_arm_tags`) for a surface that prints the document's words beside code's.
    :ptype tags: Mapping[str, str]
    :return: Variant key -> the name a reader is shown.
    :rtype: dict[str, str]
    """
    printed = printed_arm_labels(bundle)
    given = {key for arm in bundle.declaration.arms for key in bundle.arm_keys(arm)}
    entries = {entry.variant_key: entry for aspect in bundle.aspects for entry in aspect.variant_index}
    distinguishing = distinguishing_axes(list(entries.values()))
    text = text_levers({axis: level.display for axis, level in applicable_levers(entry)} for entry in entries.values())
    spellings = {
        key: spelling
        for arm in bundle.arms
        if (spelling := bundle.arm_spellings.get(arm.variant_key)) is not None
        for key in bundle.arm_keys(arm)
    }
    names = dict(printed)
    for key, name in printed.items():
        if key not in given and key in entries:
            levels = {level.axis_id: level.display for level in naming_levels(entries[key], distinguishing)}
            names[key] = (
                _spelled_reader_name(name, spellings[key], levels, text=text, declared=bundle.lever_names)
                if key in spellings
                else reader_arm_name(name, levels, text=text, declared=bundle.lever_names)
            )
    return told_apart(names, tags, {key for key in names if names[key] != printed[key]})


def input_arm_tags(bundle: CampaignBundle) -> dict[str, str]:
    """Each arm's tag as the campaign's report document page tags it, by every variant key that names it: over the
    arms in its report input's order (the campaign's arms, then the other variants its aspects index, one per label;
    :func:`~threetears.evals.analysis.report_input_build.report_input`), the control ``P``. It is
    :func:`campaign_arm_tags` over the campaign's declared arms, the call the code page tags by, so every page of a
    campaign tags an arm alike. Read off the bundle, so a surface that prints the document's words beside code's (the
    campaign listing) tags an arm as the document does.

    :param bundle: The campaign's evidence.
    :ptype bundle: CampaignBundle
    :return: Variant key -> tag.
    :rtype: dict[str, str]
    """
    control = bundle.control()
    return campaign_arm_tags(
        bundle.aspects,
        control.variant_key if control is not None else None,
        arms=[arm.variant_key for arm in bundle.arms],
        label_of=bundle.label_of,
    )


def reader_arms(bundle: CampaignBundle, tags: Mapping[str, str]) -> list[CampaignArm]:
    """The bundle's arms, each a person did not name carrying the name its page prints
    (:func:`reader_arm_labels`) where that differs from its frozen label, else its label with each lever named as
    every page names it: for a title or a listing's few words, never stored. The bundle's own arms, and everything they
    fingerprint, are not touched.

    :param bundle: The campaign's evidence.
    :ptype bundle: CampaignBundle
    :param tags: Each arm's letter tag on the page, by variant key.
    :ptype tags: Mapping[str, str]
    :return: The arms, in the bundle's order.
    :rtype: list[CampaignArm]
    """
    printed, reader, named = printed_arm_labels(bundle), reader_arm_labels(bundle, tags), _named_labels(bundle)
    arms = []
    for arm in bundle.arms:
        key = arm.variant_key
        if reader.get(key, printed.get(key)) != printed.get(key):
            arms.append(arm.model_copy(update={"label": reader[key]}))
        elif (label := named[key][0] if key in named else arm.label) != arm.label:
            # A label code spelled from levers, each lever by the name every page calls it.
            arms.append(arm.model_copy(update={"label": label}))
        else:
            arms.append(arm)
    return arms


def counted(n: int, of: int, what: str) -> str:
    """``n of m <what>``: a count, never a rate."""
    return f"{n} of {of} {what}"


__all__ = [
    "BAR_DECISION_WORDS",
    "CASES_READING",
    "COMPARISON_VERDICT_WORDS",
    "CONFIDENCE_WORDS",
    "ENGINE_WORDS",
    "FIRST_SCREEN_BANNED",
    "GUARDRAIL_DECISION_WORDS",
    "LOWER_IS_BETTER",
    "arm_tags",
    "campaign_arm_tags",
    "declared_arm_names",
    "printed_arm_labels",
    "TOLD_APART",
    "input_arm_tags",
    "reader_arm_labels",
    "reader_arms",
    "told_apart",
    "reader_arm_name",
    "text_levers",
    "comparison_words",
    "counted",
    "guardrail_words",
    "positions",
    "PASS_RATE",
    "aspect_stratum_key",
    "given_name",
    "judged_heading",
    "judged_name",
    "trial_dimension",
    "label_case",
    "table_label",
    "stratum_name",
    "stratum_names",
    "one_figure_each",
    "reading_heading",
    "reading_verb",
    "reading_name",
    "reason_words",
    "worded",
]
