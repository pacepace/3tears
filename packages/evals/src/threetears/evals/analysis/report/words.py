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

import typing
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal

from threetears.evals.analysis.arms import arm_names
from threetears.evals.analysis.bundle.schema import AnalysisContextBundle, ComparisonVerdict
from threetears.evals.kernel.analysis_measures import BarDecision
from threetears.evals.analysis.style import arm_display_name, arm_name, cannot_tell_apart, measure_name
from threetears.evals.kernel.campaign import (
    ConfidenceTier,
)
from threetears.evals.kernel.declaration import CampaignArm, resolve_control_arm
from threetears.evals.kernel.metrics import goal_check_of
from threetears.evals.kernel.surface import DecisionSurface, GuardrailDecision

if TYPE_CHECKING:
    from threetears.evals.analysis.campaign_bundle import CampaignBundle


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

    A goal check by its declared display name, else its expression, verbatim as code
    (:func:`~threetears.evals.analysis.style.measure_name`); how
    each trial ended (a ``trial`` reading) as a floor names it, "the pass rate"; any other reading by the
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
        return "the pass rate"
    if reading != "judged" and goal_check_of(name) is not None:
        return measure_name(name, declared=declared)
    if reading == "judged":
        # A reserved judge axis by its plain name before the mark, never its stored id (``__outcome__ (judged)``).
        return f"{reading_name(name)} (judged)"
    return reading_name(surface.measure_heading(name, reading))


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

    "Computed" is read from data: an arm no arm of the campaign's declaration covers is labelled by code, from the
    levers its variant's ``axis=level`` name spells (:func:`~threetears.evals.analysis.style.arm_name`'s ``levers``,
    as a code-only page reads them), so it prints by the rule; an arm a person named (any the declaration covers) keeps
    its label as written. The labels themselves, which name the arms as ids, are unchanged.

    :param bundle: The campaign's evidence.
    :ptype bundle: CampaignBundle
    :return: Variant key -> the name a page prints.
    :rtype: dict[str, str]
    """
    given = {key for arm in bundle.declaration.arms for key in bundle.arm_keys(arm)}
    entries = {entry.variant_key: entry for aspect in bundle.aspects for entry in aspect.variant_index}
    spelled = arm_names(list(entries.values()))
    keys = [*(key for arm in bundle.arms for key in bundle.arm_keys(arm)), *entries]
    printed: dict[str, str] = {}
    for key in dict.fromkeys(keys):
        label = bundle.label_of(key)
        levers = arm_name(spelled[key]).levers if key in spelled and key not in given else ()
        printed[key] = arm_display_name(label, levers)
    return printed


def counted(n: int, of: int, what: str) -> str:
    """``n of m <what>``: a count, never a rate."""
    return f"{n} of {of} {what}"


__all__ = [
    "BAR_DECISION_WORDS",
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
    "comparison_words",
    "counted",
    "guardrail_words",
    "positions",
    "reading_heading",
    "reading_name",
    "worded",
]
