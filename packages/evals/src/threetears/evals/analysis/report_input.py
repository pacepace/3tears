"""The report input: one campaign's results, normalized for the report writer (``report-input/1``).

The input sits between the pipeline and the writer: an adapter builds it from the campaign bundle, the analyst reads
it through its snapshot, and code reads it to classify the eval (candidate structure, baseline, score type, strata,
criteria, cross-cutting aspects, spread). :func:`report_input_digest` is what a frozen input is held to. Its
documentation copy is ``docs/eval-report-kit/schema/report-input.schema.json``, which states
the same fields, required sets and enums as these models.

**Statistics are copied through, never recomputed.** Every reading, interval, contrast, judge tier and floor status
is the pipeline's own figure. The models check shape only: nothing here derives a number from another.

**Beyond the control-and-candidates shape.** A campaign of one control and its candidates fills the core fields. The
optional ones carry what other comparison structures need, so their signals are read from data rather than guessed:
``tasks`` (named slices of traffic, some routable, some cutting across every other), ``factors`` and ``configs`` (a
factorial grid, the level of each factor every arm ran), ``target_matrix`` with ``cells`` (two label axes and each
arm's accuracy in every cell), ``confusion`` (where a classifier's errors go), ``human_pref`` (blind human ratings
of the control against one arm), ``criteria`` (declared gates, floors, bars and guardrails, each a threshold on one
measure) and the named, weighted ``strata`` of more than one stratification.

**More than one aspect.** The by-arm maps (``scoreboard``, each measure's and judged dimension's ``per_arm``) read the
lead aspect: the first the campaign asks about, else the first. Every other aspect's figures sit beside them, in its
own ``Aspect.scoreboard`` and each measure's ``by_aspect``; a contrast, contrast family, stratum and example names
its aspect, and so does a criterion declared on one aspect, which reads that aspect's figures. A choice's pair
readings sit on their aspect (``Aspect.pairs``).

**Counted off the bundle, for an analyst.** ``failed_trials`` (each slice's trials its pass rule failed, for any
cause, as the scoreboard's ``failed`` counts them) and ``case_overlap`` (which cases each two aspects share, by stored
case id) are tallies of the bundle's trials, not statistics: an analyst's snapshot counts them
(:func:`~threetears.evals.analysis.analyst.snapshot.analyst_input`). An uncounted input and a bare one leave them
``None``, which reads as not known, never as none. The examples are no sample of every failure where
``primary_judged`` names a dimension: they are trials scoring a nought on it, and a trial a check failed is never
among them.

**The rig, and the calls nobody answered.** ``apparatus`` copies each run's apparatus off the bundle's run summary
(``RunSummary.apparatus``): the judge each dimension was asked of and what it was asked for, the temperature, the
simulated user, whether latency was under test, how the arms shared time, how many cells ran at once, each result's
execution mode and each rubric dimension's scale. A run that recorded nothing past a serial run pinning and declaring
nothing carries no entry, and an input none of whose runs recorded one carries no key. A scoreboard row's
``unanswered`` copies the arm's unanswered calls by usage role (``ArmScoreboard.unanswered``), beside its cost and
never inside it; a row no role counted them for carries none, which reads as not counted, never as none. Both are
left out of the document when unset, so an input built before they existed digests as it did. So is a row's
``pricing`` (:class:`Pricing`): the price sources, serving providers and priced share of the calls its cost counts,
which the options table prints beside the cost.

**How a trial passes.** Each aspect built from a bundle states its pass rule (``Aspect.pass_rule``: the criteria in
force, the rubric threshold, the judged dimensions that count and the checks that fail a trial, with the trials that
passed with nothing to fail them), and says how it is graded (``graded_by`` and a one-sentence ``description``). The
pass rule is left out of the document when unset, so an input built before it existed digests as it did.

**What a contrast counted.** A contrast tests one mean per case, its repeats averaged first, and pairs the arms over
the cases both ran where they share at least two. It states the cases it read (``n_cases``, and ``control_n_cases``
where the control's side read a different number), whether it paired them (``paired``) and how closely the arms
tracked each other across them (``pairing_r``), each left out of the document when unset.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.evals.analysis.checks import Check, Corrected, Looks
from threetears.evals.schema.reading_states import (
    ArmIdentity,
    CaveatClass,
    CaveatReach,
    Coordinate,
    Maker,
    ReadingState,
    Reason,
    StateOf,
    check_state,
)
from threetears.evals.analysis.cells import NextExperiment
from threetears.evals.kernel.evidence_tiers import CriterionState
from threetears.evals.kernel.host.bars import PassCriterion
from threetears.evals.schema.hashing import bytes_digest

__all__ = [
    "ANALYST_ASPECT_FIELDS",
    "ANALYST_CONTRAST_FIELDS",
    "ReportInput",
    "report_input_digest",
    "report_input_document_digest",
]


class _Part(BaseModel):
    """A part of the report input: immutable, and refusing a key it does not declare, so a misspelled field is an
    error rather than a figure silently dropped."""

    model_config = ConfigDict(frozen=True, extra="forbid")


#: A figure as the pipeline gave it: an integer stays an integer, so a copied figure reads as it was written.
Number = int | float

#: One lever's value in an arm's settings: a display string, a number, or unset.
SettingValue = str | Number | None


class Meta(_Part):
    """What the campaign was, as the masthead and the method section name it."""

    campaign: str = Field(description="The campaign's name.")
    question: str | None = Field(default=None, description="The question the campaign asked, in its own words.")
    behavior: str | None = Field(default=None, description="The system or behavior under test.")
    kind: str = Field(description="The pipeline kind, e.g. change (control + candidates).")
    kind_means: str | None = Field(default=None, description="What the kind recommends, in one sentence.")
    design_shape: str | None = Field(default=None, description="e.g. one_factor_at_a_time, factorial.")
    run_date: str | None = Field(default=None, description="The run date, ISO 8601.")
    provenance: str | None = Field(default=None, description="Where these figures came from.")
    synthetic: bool | None = Field(default=None, description="Whether the data is made up for an example.")
    volume_per_month: Number | None = Field(
        default=None,
        description="Replies or calls per month, when the host declares it; enables monthly cost.",
    )


class Arm(_Part):
    """One arm: what it ran and the part it plays."""

    id: str = Field(description="The arm's id, as every by-arm map keys it.")
    name: str | None = Field(default=None, description="The campaign's alias for the arm.")
    short_name: str | None = Field(
        default=None,
        description="The short name the campaign declares for the arm, printed in tight places; none computes one.",
    )
    role: Literal["control", "challenger"] = Field(description="The control, or a candidate measured against it.")
    served_model: str | None = Field(default=None, description="The model that served the arm's replies.")
    variant_key: str | None = Field(default=None, description="The pipeline's variant key.")
    settings: dict[str, SettingValue] = Field(
        description=(
            "Model plus every lever that differs from the control (the control lists every lever); a lever an arm "
            "leaves out has the control's value."
        )
    )
    reasoning_ratio: Number | None = Field(default=None, description="The share of output spent reasoning.")
    description: str | None = Field(default=None, description="What kind of candidate the arm is, in a few words.")
    # The four below are left out of the document when unset, so an input built before they existed digests as it did.
    arm_id: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The id that names the arm apart from its label; names one cell where the arm is split.",
    )
    cell: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The one cell the arm id names, where the arm is split over several.",
    )
    spelling: dict[str, SettingValue] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The levers code spelled the arm's computed name from, each with its level, in order.",
    )
    identity: ArmIdentity | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Whether the arm is one thing, a state per coordinate.",
    )


class Precision(_Part):
    """What limits a reading's interval at its nominal level."""

    half_width: Number | None = Field(default=None, description="The interval's half-width, in points.")
    own_half_width: dict[str, Number] = Field(default_factory=dict, description="Each arm's own half-width, by arm.")
    limited_by: str | None = Field(default=None, description="The arm whose noise limits the interval.")
    between_case_share: Number | None = Field(
        default=None, description="The share of the variance that lies between cases."
    )
    means: str | None = Field(default=None, description="What the decomposition means, in one sentence.")


class Reading(_Part):
    """One arm's reading on an aspect: its change from the control, with the interval it was read on."""

    arm: str = Field(description="The arm's id.")
    reading: str = Field(description="The pipeline's verdict, e.g. within_margin, regressed, not_separated.")
    change: Number = Field(description="The arm minus the control, in points.")
    within_interval: tuple[Number, Number] | None = Field(
        default=None,
        description="The interval on the change at the nominal level, low then high, in points: what within_margin is "
        "read on. None where the aspect declares no margin.",
    )
    gain_interval: tuple[Number, Number] | None = Field(
        default=None,
        description="The interval on the change at gain_confidence, low then high, in points: what a gain is read on.",
    )
    gain_confidence: Number | None = Field(default=None, description="The gain interval's confidence level.")
    margin: Number | None = Field(default=None, description="The margin the reading was read against, in points.")
    control_passed: str | None = Field(default=None, description="The control's passes, as 'k of n'.")
    arm_passed: str | None = Field(default=None, description="The arm's passes, as 'k of n'.")
    cases: list[int] = Field(default_factory=list, description="The paired cases the interval rests on.")
    confirmatory: bool | None = Field(default=None, description="Whether the reading answers the asked question.")
    why: str | None = Field(default=None, description="Why, in one plain sentence.")
    precision: Precision | None = Field(default=None, description="What limits the interval.")
    # The two below are left out of the document when unset, so an input built before they existed digests as it did.
    state: StateOf | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The reading's state where it is not counted, with its reason; none reads counted.",
    )
    flip_margin: Number | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The margin at which the reading would read the other way, in points.",
    )


class Depth(_Part):
    """How deeply one arm ran an aspect."""

    cases: int = Field(description="Distinct cases.")
    trials: int = Field(description="Trials over those cases.")
    repeats_per_case: list[Number] = Field(description="Fewest, mean and most repeats a case got.")


class PassK(_Part):
    """Where pass^k is compared, and at what depth."""

    where: str = Field(description="The aspect read.")
    k: int = Field(description="The k compared.")
    common_depth: int | None = Field(
        description="The fewest runs any arm gave a shared case; none where the arms share no case."
    )
    shares_no_case: bool = Field(description="Whether the arms share no case at that depth.")
    unmeasured_for: list[str] = Field(default_factory=list, description="Arms pass^k is not measured for.")
    said: list[str] = Field(default_factory=list, description="What the pipeline says about it.")
    # The two below are left out of the document when unset, so an input built before they existed digests as it did.
    per_arm: dict[str, Stat] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Each arm's pass^k at k, 0 to 1, by arm: mean, and the cases scored k times or more (n_cases); "
        "an arm no case reached k for is not available, too_few.",
    )
    at_common_depth: dict[str, Stat] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Each arm's pass^k at the common depth, 0 to 1, by arm, over the cases every arm ran (n_cases); "
        "none where the arms share no case.",
    )
    cells: list[PassKCell] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Each pass^k cell that pools with nothing or is kept out of its arm's figure, each arm's in the "
        "frontier's order, the headline first; none where every arm's figure pools all its runs.",
    )


class PassKCell(_Part):
    """One cell of an arm's pass^k runs that pools with nothing, or is kept out of the arm's figure: its runs, its own
    pass^k at k, and why, in the shared states."""

    arm: str = Field(description="The arm's id.")
    runs: list[str] = Field(description="The runs whose attempts pooled in the cell, by id.")
    headline: bool = Field(description="Whether the cell is the one the arm's figure reads: then it rests on it alone.")
    pass_k: Stat = Field(
        description="The cell's own pass^k at k, 0 to 1, over its cases scored k times (n_cases); not available, "
        "too_few, where none was."
    )
    state: StateOf = Field(
        description="Why the cell pools with no other: withheld, split or identity_unrecorded on the coordinate its "
        "runs differ on or left unrecorded."
    )


class Unanswered(_Part):
    """One role's calls of one arm's trials that went unanswered, beside the arm's cost and never inside it."""

    retried: int | None = Field(
        default=None,
        description="Calls a forced tool choice was refused on, answered unforced in the same call: attempts and "
        "latency, never a failure; none where nothing counted them.",
    )
    lost: int | None = Field(
        default=None,
        description="Calls sent and never answered, so their turn ended in error: failures the arm's trials already "
        "count; none where nothing counted them.",
    )
    reasons: dict[str, int] | None = Field(default=None, description="The calls each reason accounts for, by word.")
    calls: int | None = Field(
        default=None, description="The answered calls beside them, which the cost counts; none where uncounted."
    )
    share: Number | None = Field(
        default=None, description="Retried and lost as a share of every call sent, 0 to 1; none where uncounted."
    )
    uncounted: int = Field(
        default=0, description="Trials whose calls in the role counted none, absent from the sums: 'at least'."
    )


class Pricing(_Part):
    """How one arm's cost was priced: what priced its calls, who served them, and how many of them were priced.

    As its producer states it for the calls the arm's cost counts, so a reader sees where each cost comes from beside
    it: a cost is the provider routing's, never the model's alone. A partly priced arm (``priced_calls`` under
    ``calls``) has a cost over part of its calls, which no page states as if it were whole. Copied from the bundle's
    census (``ArmScoreboard.pricing``), which reads it off the arm's production usage rows. ``providers`` names who
    served the model's calls alone; a tool the model called (a web-search API) is in ``tool_providers``, its spend in
    the cost and its calls in ``calls``.

    It is part of a report input's content address (``report_input_digest``, a snapshot's id), so two inputs priced
    differently are two snapshots; it is no part of any fingerprint (owner ruling, 2026-10-28).
    """

    sources: list[str] = Field(
        default_factory=list, description="Each price source its priced calls name (RoleUsage.price_source), sorted."
    )
    providers: list[str] = Field(
        default_factory=list,
        description="Each provider its model calls name as serving them (RoleUsage.served_provider, else "
        "RoleUsage.provider, on every role but external), sorted; empty where none does: not recorded.",
    )
    # Left out of the document while False, so an input whose arms' providers were all or none recorded digests as it
    # did.
    providers_in_part: bool = Field(
        default=False,
        exclude_if=lambda value: not value,
        description="Some of its model calls named no provider beside calls that named one: providers names who "
        "served part of them, and the rest are not recorded.",
    )
    # Left out of the document while empty, so an input whose rows named no tool provider digests as it did.
    tool_providers: list[str] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
        description="Each provider its tool calls name (RoleUsage.provider on an external row, such as a web-search "
        "API), sorted: tool spend its cost counts, never a provider of the model.",
    )
    calls: int | None = Field(
        default=None, description="The calls its cost counts; none where a row did not count its."
    )
    priced_calls: int | None = Field(
        default=None, description="Those of them a price was observed for; none where a row did not count its calls."
    )


class ScoreboardRow(_Part):
    """How one arm's trials came out."""

    ran: int | None = Field(default=None, description="Trials run.")
    lost: dict[str, int] = Field(default_factory=dict, description="Trials lost, by cause.")
    usable: int | None = Field(default=None, description="Trials that produced a usable measurement.")
    passed: int | None = Field(default=None, description="Trials that passed.")
    failed: dict[str, int] = Field(default_factory=dict, description="Trials that failed, by cause.")
    cases: int | None = Field(default=None, description="Distinct cases.")
    reply_cost_usd: Number | None = Field(default=None, description="Cost per reply, dollars.")
    test_cost_usd: Number | None = Field(default=None, description="What the arm's trials cost, dollars.")
    # Left out of the document when unset, so an input whose usage counted no unanswered call digests as it did.
    unanswered: dict[str, Unanswered] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The arm's unanswered calls by usage role, for each role that counted them; none where no role "
        "did (not counted, never none).",
    )
    # Left out of the document when unset, so an input built before arms carried their pricing digests as it did.
    pricing: Pricing | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="How the arm's cost was priced: its price sources, serving providers and priced calls; none where "
        "the input does not say.",
    )
    # Left out of the document when unset, so an input built before the field existed digests as it did.
    cost: Stat | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The arm's one cost reading, the fallback from cost per reply decided once.",
    )


class PairRead(_Part):
    """Two arms of a choice on one aspect: the difference between them, and which one is shown ahead."""

    arms: tuple[str, str] = Field(description="The two arms' ids, first then second, in the campaign's arm order.")
    change: Number = Field(description="The first arm minus the second, in points.")
    interval: tuple[Number, Number] | None = Field(default=None, description="The interval on the change, in points.")
    confidence: Number | None = Field(default=None, description="The interval's confidence level.")
    p_holm: Number | None = Field(default=None, description="The Holm-adjusted p.")
    ahead: str | None = Field(default=None, description="The arm shown ahead; none when neither is.")
    confirmatory: bool | None = Field(default=None, description="Whether the pair answers the asked question.")
    passed: tuple[str, str] | None = Field(default=None, description="Each arm's passes, as 'k of n'.")
    cases: tuple[int, int] | None = Field(default=None, description="Each arm's usable cases.")
    why: str | None = Field(default=None, description="Why, in one plain sentence.")
    precision: Precision | None = Field(default=None, description="What limits the interval.")
    # Left out of the document when unset, so an input built before the field existed digests as it did.
    state: StateOf | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The pair's state where it is not counted, with its reason; none reads counted.",
    )


class PassRule(_Part):
    """How code decides whether one trial on an aspect passed, read off the bundle it was decided in.

    A trial passes when its candidate did not fail, every check in ``checks`` passed (each that is no goal check named
    in ``named_checks`` with the value it must hold), and every judged dimension in ``dimensions`` scored at least
    ``threshold`` (a pass/fail dimension's bar is its pass). A boundary dimension is a guardrail, decided apart, so it
    never fails a trial. A trial with nothing to fail passes, and ``no_criterion`` counts those.
    """

    criteria: list[PassCriterion] = Field(
        description="What a trial is held to: goal_checks (every goal check), rubric (every counted judged dimension at "
        "the threshold), both, or neither."
    )
    threshold: int = Field(description="The 1 to 5 level a counted judged dimension must reach.")
    label: str = Field(description="The pass^k label every surface prints for this rule.")
    dimensions: list[str] = Field(
        default_factory=list, description="The judged dimensions that count, by key: the capability ones."
    )
    boundary: list[str] = Field(
        default_factory=list, description="The judged dimensions that do not count, by key: guardrails, decided apart."
    )
    checks: list[str] = Field(
        default_factory=list,
        description="The checks that fail a trial on their own, by measure key: goal checks, a label match and "
        "pass/fail scorers.",
    )
    # Left out of the document when empty, so a rule held to goal checks alone digests as it did.
    named_checks: list[str] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
        description="Each check in `checks` that is no goal check, as the rule names it: the measure's reader name and "
        "the value it must hold for the trial to pass (`Never topped up is false`); the other value fails the trial. "
        "Goal checks are named on the page's glossary, so the rule sums them up as every goal check.",
    )
    no_criterion: int = Field(description="Trials that passed with nothing to fail them, across every arm.")


class Aspect(_Part):
    """One aspect the campaign read: its question and each candidate's reading against the control."""

    id: str = Field(description="The aspect's id.")
    name: str | None = Field(default=None, description="What a reader calls the aspect.")
    margin: Number | None = Field(default=None, description="The non-inferiority margin, in points; none declared.")
    asked: bool | None = Field(default=None, description="Whether the campaign's question asks about it.")
    exploratory_measures: list[str] = Field(
        default_factory=list, description="Measures read on the aspect but not asked about."
    )
    readings: list[Reading] = Field(description="Each candidate's reading against the control.")
    depth: dict[str, Depth] | None = Field(default=None, description="How deeply each arm ran it, by arm.")
    pass_k: list[PassK] | None = Field(default=None, description="Where pass^k is compared.")
    pairs: list[PairRead] = Field(default_factory=list, description="A choice's pair readings on the aspect.")
    scoreboard: dict[str, ScoreboardRow] | None = Field(
        default=None,
        description="How each arm's trials came out on this aspect, by arm; none on the lead aspect (the first asked, "
        "else the first), whose scoreboard is the input's own.",
    )
    description: str | None = Field(default=None, description="How the aspect is graded, in a sentence or two.")
    graded_by: Literal["judged", "checked"] | None = Field(
        default=None,
        description="Whether a judge grades the aspect (with any checks beside it) or checks in code alone do.",
    )
    # Left out of the document when unset, so an input built before the field existed digests as it did.
    pass_rule: PassRule | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="How a trial on the aspect passes: its criteria, threshold, counted dimensions and checks; none "
        "where the input does not say.",
    )
    # The two below are left out of the document when unset, so an input built before they existed digests as it did.
    margin_informative: bool | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Whether any margin a reader could plausibly hold changes a reading on the aspect.",
    )
    looks: Looks | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="How often the aspect's test was looked at, against how often it planned to be.",
    )


#: The aspect fields only an analyst's input states: how a trial on the aspect passes, read off the bundle's census.
#: An uncounted input leaves each unset.
ANALYST_ASPECT_FIELDS: frozenset[str] = frozenset({"description", "graded_by", "pass_rule"})


class FloorRead(_Part):
    """One floor on one candidate: held, broken or unclear, as the pipeline read it."""

    floor: str = Field(description="The floor's key.")
    aspect: str = Field(description="The aspect it is read on.")
    arm: str = Field(description="The candidate's id.")
    status: str = Field(description="broken, held or unclear.")
    reading: str | None = Field(default=None, description="The rate that decided the status.")
    speaks_for: str | None = Field(default=None, description="The rate the floor speaks for.")
    interval: tuple[Number, Number] | None = Field(default=None, description="The interval the status was read on.")
    worst_drop: Number | None = Field(default=None, description="The largest drop the interval allows, in points.")
    margin: Number | None = Field(default=None, description="The floor's margin, in points.")
    short_by: Number | None = Field(default=None, description="How far an unclear floor is from held, in points.")
    reason: str | None = Field(default=None, description="Why, in one plain sentence.")


class BarRead(_Part):
    """One arm against one absolute bar: met, missed or unclear, as the pipeline read it."""

    aspect: str = Field(description="The aspect it is read on.")
    arm: str = Field(description="The arm's id.")
    measure: str = Field(description="What the bar is read on.")
    bar: Number = Field(description="The threshold, in the measure's units.")
    direction: Literal["higher_is_better", "lower_is_better"] = Field(description="Which way clearing it runs.")
    status: str = Field(description="met, missed or unclear.")
    value: Number | None = Field(default=None, description="The arm's value on the measure.")
    interval: tuple[Number, Number] | None = Field(default=None, description="The interval the status was read on.")
    why: str | None = Field(default=None, description="Why, in one plain sentence.")


class Criterion(_Part):
    """A declared threshold: a gate, floor, bar or guardrail on one measure.

    A criterion states the rule; one gate function applies it, so the page, the options and any explore control
    cannot disagree. ``floors`` and ``bars`` carry the pipeline's own statuses where it read them.
    """

    id: str = Field(description="The criterion's id.")
    kind: Literal["gate", "floor", "bar", "guardrail"] = Field(
        description="gate: an option failing it is out; floor and bar: they warn; guardrail: a standing must-not-worsen."
    )
    measure: str = Field(description="The measure, judged dimension or aspect it reads, by key.")
    statistic: Literal["mean", "p50", "p95"] = Field(default="mean", description="Which statistic it reads.")
    direction: Literal["higher_is_better", "lower_is_better"] = Field(
        description="higher_is_better: the value must be at least the threshold; lower_is_better: at most."
    )
    threshold: Number = Field(description="The threshold, in the measure's units.")
    scope: Literal["overall", "each_task", "each_cell"] = Field(
        default="overall",
        description="overall: the arm's overall value; each_task: every task's value; each_cell: every cell's.",
    )
    task: str | None = Field(default=None, description="The one task the criterion applies within.")
    standing: bool = Field(default=False, description="Whether it is standing policy rather than this campaign's.")
    label: str | None = Field(default=None, description="What a reader calls it.")
    # Left out of the document when unset, so an input frozen before the field existed digests as it did.
    aspect: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The aspect whose figures it reads, by id: the lead aspect's are the by-arm maps, any other's its "
        "by_aspect figures; none: the lead aspect's (the first asked, else the first).",
    )


class Stat(_Part):
    """One arm's figure on one measure: a mean with its spread, a latency's percentiles, or a distribution."""

    mean: Number | None = Field(default=None, description="The mean, in the measure's units.")
    ci: tuple[Number, Number] | None = Field(default=None, description="The interval on the mean, low then high.")
    half_width: Number | None = Field(default=None, description="The interval's half-width, where only that is given.")
    sem: Number | None = Field(default=None, description="The standard error of the mean.")
    n: int | None = Field(default=None, description="Observations.")
    n_cases: int | None = Field(default=None, description="Distinct cases behind them.")
    p50: Number | None = Field(default=None, description="The median.")
    p95: Number | None = Field(default=None, description="The 95th percentile.")
    change: Number | None = Field(default=None, description="The arm minus the control, where only that is given.")
    shares: list[Number] | None = Field(default=None, description="The share in each of the measure's bins, percent.")
    # The three below are left out of the document when unset, so an input built before they existed digests as it did.
    spread: Number | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The spread of the arm's per-case values, in the measure's units.",
    )
    no_variation: Number | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The one value every case took, where the cases did not vary.",
    )
    state: StateOf | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The interval's state where it is not counted, with its reason; none reads counted.",
    )


class Level(_Part):
    """One level of a factor, axis or bin."""

    id: str = Field(description="The level's id.")
    name: str | None = Field(default=None, description="What a reader calls it.")


#: The unit of a measure that is a share on 0 to 1 (a host's fraction): a rate, read in percent.
SHARE_UNIT = "share"


class Measure(_Part):
    """One measure: what it means and each arm's figure on it."""

    label: str | None = Field(default=None, description="What a reader calls it.")
    unit: str | None = Field(
        default=None,
        description="Its unit, e.g. usd, pct, ms, usd_per_1k; share for a fraction on 0 to 1, which reads as a rate.",
    )
    higher_is_better: bool | None = Field(default=None, description="Which way is better.")
    merit_axis: str | None = Field(default=None, description="The merit axis it counts toward.")
    means: str | None = Field(default=None, description="Its definition.")
    per_arm: dict[str, Stat] = Field(default_factory=dict, description="Each arm's figure, by arm.")
    grade: Literal["pass_fail", "ordinal", "class_label"] | None = Field(
        default=None,
        description="For a rate or score over graded cases, what one case's grade is: a pass or fail, a point on a "
        "scale, or a class label right or wrong.",
    )
    level: str | None = Field(default=None, description="The confidence level of the intervals per_arm carries.")
    bins: list[Level] | None = Field(default=None, description="The bins a distribution's shares fall in, in order.")
    minimum: Number | None = Field(
        default=None, description="The least the measure can be, where the test set fixes it (a gold call count)."
    )
    by_aspect: dict[str, dict[str, Stat]] = Field(
        default_factory=dict,
        description="Each other aspect's figures, by aspect then arm; per_arm is the lead aspect's (the first asked, "
        "else the first).",
    )


class Judged(_Part):
    """One judged dimension: its judge and each arm's mean."""

    label: str | None = Field(default=None, description="What a reader calls it.")
    scale: Literal["ordinal", "pass_fail"] | None = Field(default=None, description="The rubric's scale.")
    tier: str | None = Field(default=None, description="The judge's evidence tier.")
    judge_model: str | None = Field(default=None, description="The judge's model.")
    calibration: str | None = Field(default=None, description="The judge's calibration state.")
    calibration_ratings: int | None = Field(default=None, description="Human ratings the judge was calibrated on.")
    agreement: Number | None = Field(default=None, description="The judge's agreement with those ratings.")
    level: str | None = Field(default=None, description="The confidence level of the intervals per_arm carries.")
    per_arm: dict[str, Stat] = Field(default_factory=dict, description="Each arm's mean, by arm.")
    # Left out of the document while empty, so an input built before the field existed digests as it did.
    checks: list[Check] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
        description="The judge's checks: what it was checked against, on what sample, with what result.",
    )
    by_aspect: dict[str, dict[str, Stat]] = Field(
        default_factory=dict,
        description="Each other aspect's figures, by aspect then arm; per_arm is the lead aspect's (the first asked, "
        "else the first).",
    )


class Contrast(_Part):
    """One multiple-comparison-corrected contrast of an arm against the control on one measure."""

    arm: str = Field(description="The arm's id.")
    control: str | None = Field(default=None, description="The control's id.")
    kind: str | None = Field(default=None, description="measure or judged.")
    measure: str = Field(description="The measure or judged dimension, by key.")
    aspect: str | None = Field(default=None, description="The aspect it is read on.")
    verdict: str = Field(description="improved, regressed, equivalent, not_separated or untested.")
    delta: Number | None = Field(
        default=None, description="The arm minus the control, in the measure's units; none when untested."
    )
    interval: tuple[Number, Number] | None = Field(
        default=None, description="The corrected interval on the delta; none when untested."
    )
    level: str | None = Field(default=None, description="The interval's level: its family's.")
    p_adjusted: Number | None = Field(default=None, description="The adjusted p.")
    materiality: str | None = Field(default=None, description="Whether the delta is material.")
    confounds: list[str] = Field(default_factory=list, description="Confounds on the contrast, by dimension.")
    reason: str | None = Field(default=None, description="Why it is untested or not separated, in one sentence.")
    # The four below are left out of the document when unset, so an input built before they existed digests as it did.
    n_cases: int | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The cases the test read on the arm's side, each one mean over its repeats: when paired, the "
        "cases both arms ran.",
    )
    control_n_cases: int | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The cases the test read on the control's side, where they differ from n_cases (an unpaired test).",
    )
    paired: bool | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Whether the test paired the arms case by case; none where it was not tested.",
    )
    pairing_r: Number | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The arms' correlation across the paired cases: how much pairing narrowed the interval.",
    )
    # Every field below is left out of the document when unset, so an input built before it existed digests as it did.
    # The typed state is `state`: `reason` above is the pipeline's sentence, which a stored input carries.
    state: StateOf | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The contrast's state where it is not counted, with its reason; none reads counted.",
    )
    baseline: Number | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The control's mean over the cases the test paired: what the delta is a change from.",
    )
    baseline_n_cases: int | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The cases that baseline is the mean of.",
    )
    arm_mean: Number | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The arm's mean over the cases the test paired: what the delta is a change to.",
    )
    family: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The family the contrast was corrected in, by id.",
    )
    delta_note: Literal["paired_cases", "adjusted"] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Why the delta is not the difference of the two means printed beside it, where it is not.",
    )
    margin: Number | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The margin the delta was read against, in the measure's units: its equivalence margin, else the "
        "measure's declared materiality threshold, else the margin its runs declared; none for none.",
    )
    corrected: Corrected | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The contrast as what its judge was checked against would read it: a check's result, never a second verdict.",
    )
    aa: Check | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The A/A check covering the contrast: identical arms read on the same reading.",
    )


#: The contrast fields only an analyst's input states: what its test counted. An uncounted input leaves each unset.
ANALYST_CONTRAST_FIELDS: frozenset[str] = frozenset({"n_cases", "control_n_cases", "paired", "pairing_r"})


class ContrastFamily(_Part):
    """One family of contrasts corrected together."""

    aspect: str | None = Field(default=None, description="The aspect whose contrasts it corrects.")
    merit_axes: list[str] = Field(
        default_factory=list, description="The declared merit priority; empty when no rule ranks the axes."
    )
    correction: str | None = Field(default=None, description="The correction method, e.g. holm, none.")
    alpha: Number | None = Field(default=None, description="The family's alpha.")
    family_size: int | None = Field(default=None, description="Contrasts in the family.")
    disclosure: str | None = Field(default=None, description="What the correction means, in plain words.")
    level: str | None = Field(default=None, description="Each interval's level.")


class AcrossFigure(_Part):
    """One arm's figure across every aspect, each aspect weighed alike, with its interval at the aggregate's level; or
    withheld, naming the aspects that lack it, since a mean over the aspects that have it would weigh the rest at
    nought."""

    value: Number | None = Field(description="The equal-weight mean of the aspects' figures; none where withheld.")
    interval: tuple[Number, Number] | None = Field(description="Its interval, low then high; none where withheld.")
    missing: list[str] = Field(
        default_factory=list,
        description="The aspects lacking the figure or its standard error, by id; set exactly where it is withheld.",
    )


class ArmAcross(_Part):
    """One arm's figures across every aspect. Descriptive: none is tested, and none decides a reading."""

    pass_rate: AcrossFigure = Field(description="Its pass rate, percent: each aspect's usable trials passed.")
    change: AcrossFigure | None = Field(
        default=None, description="A change's candidate: its change against the control, in points; none otherwise."
    )
    reply_cost_usd: AcrossFigure = Field(description="What a reply costs it, US dollars: each aspect's cost per reply.")


class AcrossAspects(_Part):
    """Every arm's figures across a campaign's aspects, each aspect weighed alike, its intervals taking the aspects as
    independent: the campaign aggregate, copied."""

    weights: dict[str, Number] = Field(description="Each aspect's weight, by id; they sum to 1.")
    level: str = Field(description="The level every interval here is read at.")
    per_arm: dict[str, ArmAcross] = Field(description="Each arm's figures, by arm, in the scorecard's order.")


class Stratum(_Part):
    """One stratum: a slice of the cases and each arm's figures in it."""

    id: str = Field(description="The stratum's id.")
    name: str | None = Field(default=None, description="What a reader calls it.")
    aspect: str | None = Field(default=None, description="The aspect whose cases it slices.")
    dimension: str | None = Field(
        default=None, description="The stratification it belongs to, when the input holds more than one."
    )
    n_cases: int | None = Field(default=None, description="Cases in the stratum.")
    share: Number | None = Field(default=None, description="Its share of traffic, 0 to 1.")
    cost_multiplier: Number | None = Field(default=None, description="Its cost per reply relative to the average.")
    per_arm: dict[str, dict[str, Stat]] = Field(
        default_factory=dict, description="Each arm's figures in the stratum: arm, then measure or dimension."
    )


class Example(_Part):
    """One trial shown as an example, usually a failure."""

    trial: str | None = Field(default=None, description="The trial's id.")
    case: str | None = Field(default=None, description="The case's id.")
    stratum: str | None = Field(default=None, description="The stratum or task it falls in.")
    aspect: str | None = Field(default=None, description="The aspect it ran on.")
    arm: str | None = Field(default=None, description="The arm that ran it.")
    repeat: int | None = Field(default=None, description="Which repeat of the case.")
    reason: str | None = Field(default=None, description="Why it failed: the judge's reason or a note.")
    excerpt: str | None = Field(default=None, description="The end of the reply, or what the arm produced.")
    outcome: str | None = Field(default=None, description="The failure class.")
    input: str | None = Field(default=None, description="The input the case gave.")
    expected: str | None = Field(default=None, description="What the case expected.")
    # Left out of the document when unset, so an input built without every dimension's reason digests as it did.
    reasons: dict[str, str] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Each judged dimension's reason for the trial, by dimension, the primary's among them; none where "
        "the input does not carry them.",
    )


class FailedTrials(_Part):
    """How many trials of one slice of the run failed, for any cause, as the scoreboard's ``failed`` counts them."""

    aspect: str = Field(description="The aspect the slice ran on.")
    arm: str = Field(description="The arm that ran it.")
    stratum: str | None = Field(default=None, description="The stratum or task it falls in; none for none.")
    failed: int = Field(ge=0, description="The trials of the slice its pass rule failed, for any cause.")


class CaseOverlap(_Part):
    """Whether two aspects ran the same cases: the distinct cases each ran, by their stored ids, shared or not."""

    aspects: tuple[str, str] = Field(description="The two aspects' ids, in the input's aspect order.")
    shared: int = Field(ge=0, description="The cases both aspects ran.")
    only_first: int = Field(ge=0, description="The cases only the first ran.")
    only_second: int = Field(ge=0, description="The cases only the second ran.")


class LateCriterion(_Part):
    """A criterion set after the results were visible."""

    criterion: str = Field(description="What was set: aspect, control, margin, floor, gate.")
    name: str | None = Field(default=None, description="Which one, where several could be.")
    set_at: str = Field(description="When it was set.")
    # The two below are left out of the document when unset, so an input built before they existed digests as it did.
    aspect: str | None = Field(
        default=None, exclude_if=lambda value: value is None, description="A floor's aspect, by its id."
    )
    reading: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The reading a floor names; none for a floor on every pass rate.",
    )


class ApparatusConfound(_Part):
    """An apparatus setting that may have varied between arms."""

    dimension: str = Field(description="The setting.")
    kind: str | None = Field(default=None, description="The confound's kind.")
    status: str | None = Field(default=None, description="Whether it varied.")
    level_values: dict[str, SettingValue] = Field(default_factory=dict, description="Its value per level.")
    threshold: Number | None = Field(default=None, description="The threshold it was read against.")


class ShortRun(_Part):
    """A run that produced fewer usable measurements than it was asked for."""

    run: str = Field(description="The run's id.")
    arm: str | None = Field(default=None, description="The arm it ran.")
    note: str = Field(description="What it lost, and what that does to its rates.")


class Caveats(_Part):
    """Every disclosure the flag and the method section may need."""

    late_criteria: list[LateCriterion] = Field(default_factory=list, description="Criteria set after the results.")
    judge_tiers: dict[str, str] = Field(default_factory=dict, description="Each judged dimension's evidence tier.")
    confounds: dict[str, str] = Field(default_factory=dict, description="Each confound and what it means.")
    apparatus_confounds: list[ApparatusConfound] = Field(default_factory=list, description="Apparatus confounds.")
    latency: str | None = Field(default=None, description="Why latency is not compared, when it is not.")
    short_runs: list[ShortRun] = Field(default_factory=list, description="Runs short of usable measurements.")
    unreported: list[str] = Field(default_factory=list, description="Observations recorded but not reported.")
    # Left out of the document while empty, so an input over runs that all finished reads, and is keyed, as one did
    # before the field existed.
    unfinished: list[str] = Field(
        default_factory=list,
        exclude_if=lambda said: not said,
        description=(
            "What stopped each run that did not finish, a sentence each in code's words, then what the readings "
            "cover: what a page opens with in place of a verdict. Empty where every run completed."
        ),
    )


class RequestSettings(_Part):
    """What a role's calls were asked for: the output cap, the reasoning asked, and whether every parameter was
    enforced."""

    max_tokens: int = Field(description="The output cap.")
    reasoning_max_tokens: int | None = Field(default=None, description="The reasoning budget, in tokens.")
    reasoning_effort: str | None = Field(default=None, description="The reasoning effort word.")
    strict_output: bool = Field(default=False, description="Whether the request enforced every parameter it carried.")


class Disclosure(_Part):
    """What the launching host disclosed about one run's launch, as it stated it: values by its own names, and its
    statements of how the run was measured. Shown, never compared: it splits no rig and names no confound."""

    values: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Each disclosed value by the host's name for it."
    )
    statements: list[str] = Field(default_factory=list, description="What the host states, in its own words.")


class RunRig(_Part):
    """The rig one run of one arm was measured with, as the run recorded it at launch."""

    run: str = Field(description="The run's id.")
    arm: str | None = Field(
        default=None, description="The arm it ran, by id; none for a run the design placed in no arm."
    )
    aspect: str = Field(description="The aspect it ran, by id.")
    judges: dict[str, str] | None = Field(
        default=None, description="The model each judged dimension was asked of; none where not recorded."
    )
    judge_asked: RequestSettings | None = Field(
        default=None, description="What every judge call was asked for; none where not recorded."
    )
    judge_temperature: Number | None = Field(
        default=None, description="The temperature judge calls were asked at; none where not recorded."
    )
    simulator: str | None = Field(default=None, description="The simulated user's model; none where none was pinned.")
    measure_latency: bool | None = Field(
        default=None, description="Whether latency was declared under test; none where nothing was declared."
    )
    arm_schedule: Literal["interleaved", "sequential", "concurrent"] | None = Field(
        default=None, description="How the arm shared time with the other arms; none where not recorded."
    )
    cell_concurrency: int = Field(description="How many of the run's cells executed at once; 1 is serial.")
    execution_modes: dict[str, int] = Field(
        default_factory=dict, description="The run's results by the execution mode each recorded."
    )
    rubric_scales: dict[str, Literal["ordinal", "pass_fail"]] = Field(
        default_factory=dict, description="The scale each rubric dimension was answered on."
    )
    # Left out of the document when unset, so an input whose runs' hosts disclosed nothing digests as it did.
    disclosure: Disclosure | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="What the launching host disclosed about the launch; none where it disclosed nothing.",
    )


class PriorInsight(_Part):
    """An insight an earlier campaign recorded."""

    statement: str = Field(description="The insight.")
    confidence: str | None = Field(default=None, description="How sure the earlier campaign was.")


class Task(_Part):
    """A named slice of the work, with each arm's figures in it.

    A routable task's traffic can go to its own arm; a cross-cutting one occurs inside every other task, so an arm
    serving any traffic must pass it.
    """

    id: str = Field(description="The task's id.")
    name: str | None = Field(default=None, description="What a reader calls it.")
    description: str | None = Field(default=None, description="What it covers, in a sentence.")
    n_cases: int | None = Field(default=None, description="Cases in it.")
    share: Number | None = Field(default=None, description="Its share of traffic, 0 to 1; 0 for a test category.")
    routable: bool | None = Field(default=None, description="Whether its traffic can be routed to its own arm.")
    cross_cutting: bool = Field(default=False, description="Whether it occurs inside every other task.")
    per_arm: dict[str, dict[str, Stat]] = Field(
        default_factory=dict, description="Each arm's figures in the task: arm, then measure or dimension."
    )


class Factor(_Part):
    """One factor of a factorial grid, or one axis of a target matrix, with its levels in order."""

    id: str = Field(description="The factor's id, as configs and settings key it.")
    name: str | None = Field(default=None, description="What a reader calls it.")
    levels: list[Level] = Field(min_length=1, description="Its levels, in order.")
    # Left out of the document while unset, so an input built before the field existed digests as it did. Unset is
    # its own state: a reader takes None as "not stated", never as "unordered".
    ordered: bool | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "Whether the levels are points on one scale, so a line may be drawn across them; False: they are kinds, "
            "no line. None = not stated."
        ),
    )


class Config(_Part):
    """The level of each factor one arm ran."""

    arm: str = Field(description="The arm's id.")
    levels: dict[str, str] = Field(description="Each factor's level id, by factor id.")


class TargetMatrix(_Part):
    """Two label axes whose product is the cells a classifier is graded in."""

    rows: Factor = Field(description="The row axis.")
    columns: Factor = Field(description="The column axis.")


class Cell(_Part):
    """One cell of the target matrix and each arm's figures in it."""

    row: str = Field(description="The row axis's level id.")
    column: str = Field(description="The column axis's level id.")
    n_cases: int | None = Field(default=None, description="Cases in the cell.")
    per_arm: dict[str, dict[str, Stat]] = Field(
        default_factory=dict, description="Each arm's figures in the cell: arm, then measure."
    )


class ConfusionTable(_Part):
    """Where one label axis's predictions went: rows are the expected label, columns the predicted one."""

    labels: list[str] = Field(min_length=2, description="The labels, in row and column order.")
    unit: Literal["row_percent", "count"] = Field(description="row_percent: each row sums to 100; count: observations.")
    per_arm: dict[str, list[list[Number]]] = Field(description="Each arm's matrix, by arm.")


class PreferenceRow(_Part):
    """Blind human preference in one slice."""

    id: str = Field(description="The slice's id.")
    name: str | None = Field(default=None, description="What a reader calls it.")
    prefers_control: Number = Field(description="The share of ratings preferring the control, percent.")
    half_width: Number | None = Field(default=None, description="The interval's half-width, in points.")


class HumanPreference(_Part):
    """Blind side-by-side ratings of the control against one arm."""

    control: str = Field(description="The control's id.")
    arm: str = Field(description="The arm rated against it.")
    n: int = Field(description="Rated items.")
    rows: list[PreferenceRow] = Field(min_length=1, description="The preference in each slice.")
    # The five below are left out of the document when unset, so an input built before they existed digests as it did.
    read_id: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The blind read the ratings come from.",
    )
    person: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Who rated, where the read records it.",
    )
    test: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The test the preference was read by.",
    )
    p: Number | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Its p.",
    )
    verdict: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The contrast verdict it comes to.",
    )


#: The states that change the answer: what a class A caveat is raised by.
_CHANGES_THE_ANSWER = frozenset({"set_aside", "withheld"})


class CaveatEntry(_Part):
    """One row of the one caveat list: a qualification of the answer, typed so code ranks it without reading a word.

    Every first-screen qualification is one of these. ``against`` and ``criterion`` say what a caveat about an
    instrument's check was checked against and on which criterion, which is what lets a check against people silence a
    check against a model, and what ranks a judge checked against a model only.
    """

    id: str = Field(description="The caveat's stable id, as a report document names it.")
    kind: str = Field(description="The id's kind, which decides its class.")
    cls: CaveatClass | None = Field(
        default=None, alias="class", description="What it does to the answer, A to E; none for a method-reach kind."
    )
    reach: CaveatReach = Field(description="report, scoped or method.")
    state: ReadingState | None = Field(default=None, description="The non-counted state that raised it, where one did.")
    reason: Reason | None = Field(default=None, description="That state's reason, or the reason a check is not met.")
    coordinate: Coordinate | None = Field(
        default=None, description="The coordinate a split or an unrecorded identity names."
    )
    arms: list[str] = Field(default_factory=list, description="The arm ids it touches; empty for all.")
    readings: list[str] = Field(
        default_factory=list,
        description="The readings it is scoped to, each '<aspect>/<measure>'; empty for the whole report.",
    )
    asked: bool = Field(
        default=False,
        description="Whether it touches an asked aspect: one of its readings is on one, or it is scoped to the whole "
        "report and the report asks about one.",
    )
    deciding: bool = Field(
        default=False,
        description="Whether the answer turns on a reading it names: a class E caveat reaches the first screen only "
        "where it does.",
    )
    immaterial: bool | None = Field(
        default=None, description="True once code proved it changes no answer; none where code cannot say."
    )
    subsumed_by: str | None = Field(default=None, description="The caveat that silences it, by id.")
    against: Literal["people", "model"] | None = Field(
        default=None, description="For a caveat about an instrument's check: what the instrument was checked against."
    )
    criterion: str | None = Field(
        default=None,
        description="For such a caveat: the criterion the check decides; for a gate cleared on a point estimate or a "
        "borderline figure: the gate, by id.",
    )
    same_family: bool = Field(
        default=False, description="For a check against a model: whether the model is of a candidate's family."
    )
    check: CriterionState | None = Field(
        default=None,
        description="For a caveat about an instrument's check: the check's result. Only a result that decides (met or "
        "not met) speaks for the instrument, so only such a check against people silences one against a model.",
    )
    runs: list[str] = Field(default_factory=list, description="The runs it is about, by id; empty for none named.")
    cases: int | None = Field(
        default=None, description="The cases the readings it names rest on, where one count says it."
    )

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    @model_validator(mode="after")
    def _belongs_together(self) -> Self:
        if self.state is not None:
            check_state(self.state, self.reason, self.coordinate)
        if self.cls == "A" and self.state not in _CHANGES_THE_ANSWER:
            raise ValueError("a class A caveat is a reading set aside or withheld, and carries that state")
        return self


class GateRun(_Part):
    """One run a gate names."""

    run: str = Field(description="The run's id.")
    status: str = Field(description="Its stored status.")
    ended_because: str | None = Field(
        default=None, description="Why it ended, in its record's own words; never one of the shared reasons."
    )


class Gate(_Part):
    """What stops a campaign from being read at all; a campaign that can be read has none."""

    rule: Literal["setup", "unfinished", "too_small"] = Field(description="Which rule stopped the read.")
    runs: list[GateRun] = Field(default_factory=list, description="The runs the rule names.")
    fewest_cases: int | None = Field(
        default=None, ge=0, description="For too few cases: the fewest usable cases an arm read holds."
    )
    waiting: int = Field(default=0, ge=0, description="The open report-wide caveats waiting below the gate.")


class CallRead(_Part):
    """The call on a campaign, as of the freeze: the arm decided, who decided, and what has moved since."""

    arm: str = Field(description="The decided arm's id.")
    made_by: Maker | None = Field(default=None, description="A person or a model; none where the call recorded none.")
    model: str | None = Field(default=None, description="The model that made it, by id.")
    on_behalf_of: str | None = Field(default=None, description="Who a model made it for.")
    runs_joined_since: int | None = Field(default=None, description="Runs that joined the evidence after the call.")
    identity_changed: bool | None = Field(
        default=None, description="Whether the evidence's fingerprint differs from the one the call was read against."
    )


#: The one act that would close a caveat, as a closed kind; ``no_action`` where nothing the package knows of closes it,
#: with the reason the step states (:data:`NoActionReason`).
Remedy = Literal[
    "resume_or_rerun",
    "rerun_on_one_cell",
    "remeasure_or_attest",
    "rerun_serially",
    "rerun_balanced",
    "check_against_people",
    "replace_instrument",
    "sharpen_criterion",
    "repair_rig",
    "add_cases",
    "confirm_on_fresh_cases",
    "record_and_rerun",
    "run_live",
    "draw_at_random",
    "finish_read",
    "declare_looks",
    "declare_criteria",
    "name_control",
    "no_action",
]

#: Why a step that is more evidence states no planned size, and so makes no call: ``no_interval``, the figure it would
#: settle states no interval or standard error; ``no_spread``, every case read alike, so at today's estimate no added
#: case is expected to differ and no size can be promised; ``on_the_line``, today's estimate sits exactly on its line,
#: so no finite size settles it there; ``no_count``, no count says how many cases the figure rests on;
#: ``nothing_to_settle``, the caveat names no figure with a line, a margin or a floor a size could be read against.
UnsizedReason = Literal["no_interval", "no_spread", "on_the_line", "no_count", "nothing_to_settle"]

#: Why no catalogue action takes a step: ``unclassed_kind``, an analyst's own kind, outside the package's classes;
#: ``method_only``, a kind that only ever lands in the method section and qualifies no answer;
#: ``no_interval_to_gain``, a figure the engine states no interval for at any size (a percentile, or a mean read from
#: enough cases that still states none), so no added case gives it one; ``lift_not_shown``, a step no sequence of the
#: catalogue's actions has been driven, followed as written, to lift, so it names its remedy and makes no call: a
#: re-run whose fault (an answering model split by the provider, a rig, recording, a replay, a confound no launch
#: setting holds alike) is not a parameter a launch sets; the serial re-run of cases run at once, which a launch can
#: ask for (``measure_latency``) but which has not been driven; and more cases on a contrast where every arm the
#: caveat names but the control would run in a call of its own, which adds no pair (``unpaired_arms``).
NoActionReason = Literal["unclassed_kind", "method_only", "no_interval_to_gain", "lift_not_shown"]

#: A value one argument of a catalogue action's call takes.
StepArgument = str | bool | int | list[str]


class RemedyStep(_Part):
    """What would change one first-screen reason, or what stops a campaign being read: the one act that closes it, as
    a value a person reads and an agent can take.

    Where an action of the engine's catalogue takes the step, ``action`` names it, ``calls`` holds each call of it the
    step makes, every argument named and typed as the action's own parameters (a step that adds cases passes its
    planned size as the count of new cases), and ``needs`` names the parameters the step cannot fill from what the
    caveat holds, which the taker supplies to every call. A launch repeats the set-up of the arms it runs: what the
    reading holds of it is filled, and ``as_launched`` names the rest, each set as the arms' runs were launched, except
    the dimensions ``alike`` names, held at one value on every call; ``call_arms`` says which arms each call runs.
    Where one action's calls do not finish the act, ``then`` holds the calls that do, in order, each of the same
    shape: a launch's runs joined to the campaign it reads, the runs a re-run replaces taken out of it, and for a
    confirmation the campaign created first; ``after`` names each argument that only an earlier call's result holds.
    A step ships a sequence only where following it, through the real actions on a fixture campaign, has been shown
    to do what it says; ``limits`` states, typed, what following it cannot guarantee, and a re-run no sequence has
    been shown to lift makes no call and says so (``no_action`` ``lift_not_shown``).
    Where the step is more evidence, ``planned_cases`` is its size, read from the figure it would settle and stated
    before it runs, never "until it separates"; where no size can be read, ``unsized`` says why and the step makes no
    call, so it is never taken open-ended. Where no action takes it, ``no_action`` says why. A step that spends names
    the action that prices it first; code never takes a step.
    """

    remedy: Remedy = Field(description="What would close it.")
    closes: str | None = Field(
        default=None,
        description="The caveat it answers, by id; none for the gate's. A step whose first call is campaign_create "
        "closes nothing on this report: the caveat stays raised here, and the confirmation is read in the campaign it "
        "creates (limits: this_report_stays_flagged).",
    )
    action: str | None = Field(
        default=None, description="The catalogue action that takes it; none where a host or a person acts."
    )
    calls: list[dict[str, StepArgument]] = Field(
        default_factory=list,
        description="Each call of the action the step makes, its arguments by the action's own parameter names; empty "
        "where no action takes it.",
    )
    needs: list[str] = Field(
        default_factory=list,
        description="The action's parameters no call fills, by its own names, which the taker supplies to every call.",
    )
    as_launched: list[str] = Field(
        default_factory=list,
        description="For a launch: the parameters that set an arm up which the reading holds no value for, by the "
        "action's own names; the taker sets each as the runs of the arms it launches were launched, never to its "
        "default, or the new runs are not more of those arms. The one exception is a dimension alike names, which "
        "the taker sets to one value on every call.",
    )
    alike: list[str] = Field(
        default_factory=list,
        description="For a re-run that balances a confound: each apparatus dimension the confound is in, by its "
        "apparatus_settings key, which the taker sets to one value on every call of the launch, never as each arm "
        "was launched: the arms then differ only in what is compared.",
    )
    call_arms: list[list[str]] = Field(
        default_factory=list,
        description="For a launch: the arms each call runs, by id, in the calls' order; empty where the reading does "
        "not hold the arms' runs.",
    )
    after: dict[str, Literal["launched_runs", "launched_control_run", "created_campaign"]] = Field(
        default_factory=dict,
        description="For a call that follows another of the step: each parameter only an earlier call's result holds, "
        "and which: launched_runs, the run ids job_poll names on the step's launch job once its runs start (run_ids); "
        "launched_control_run, the one of them the control arm's model ran, named as soon as job_poll names it and "
        "before any result is scored, or the campaign reads its control as named late; created_campaign, the id the "
        "step's campaign_create returned.",
    )
    then: list[RemedyStep] = Field(
        default_factory=list,
        description="The calls that finish the act after this step's own, in order, each of one action and of this "
        "shape: the runs a launch makes joined to the campaign (campaign_runs_add), the runs a re-run replaces taken "
        "out (campaign_runs_remove), the control named in a new campaign (campaign_update). Empty on a call that "
        "follows another.",
    )
    arms: list[str] = Field(default_factory=list, description="The arms it is about, by id; empty for all.")
    runs: list[str] = Field(default_factory=list, description="The runs it is about, by id; empty for none named.")
    planned_cases: int | None = Field(
        default=None,
        ge=1,
        description="For more evidence: the cases to add per arm named (on fresh cases, the cases to run), read from "
        "the figure it would settle, planned before it runs, so that once the runs it launches are in the campaign "
        "that reads it, a design of that size clears the line the caveat is lifted by, on the interval it is lifted "
        "by, at least four times in five if the truth is today's estimate. It counts cases the launch keeps and did "
        "not reuse (limits: axis_may_not_generate), and holds only for arms launched in one call with the control "
        "(limits: unpaired_arms). For a step whose first call is campaign_create, the line is the one the created "
        "campaign reads, never this report's.",
    )
    unsized: UnsizedReason | None = Field(
        default=None, description="For more evidence no size can be read for: why. Such a step makes no call."
    )
    size_bias: Literal["picked_best_of_n"] | None = Field(
        default=None,
        description="Why the planned size is known to be too small, where it is: picked_best_of_n, read from the pick's "
        "own estimate, which reads high because the pick was the best of several, so fewer than four designs in five "
        "of that size clear.",
    )
    declared_look: bool = Field(
        default=False,
        description="Whether taking it reads the same readings again with the cases it adds: one more look at them, "
        "stated before it is taken. Nothing counts looks yet: the reading after it is read at its full level as if it "
        "were the first, so adding cases and reading again even once errs more often than that level says.",
    )
    results_needed: int | None = Field(
        default=None, ge=1, description="For a check: how many more results it needs before it can decide."
    )
    recording: list[NextExperiment] = Field(
        default_factory=list, description="What recording an unrecorded coordinate would pool, cell by cell."
    )
    spends: bool = Field(default=False, description="Whether it costs money to take.")
    priced_by: str | None = Field(
        default=None, description="For a step that spends: the catalogue action that prices it before it is taken."
    )
    no_action: NoActionReason | None = Field(
        default=None,
        description="Why no catalogue action takes it: for no_action, why no act closes it; for an act no one action "
        "takes, or no sequence has been shown to finish, why not.",
    )
    limits: list[
        Literal["unpaired_arms", "axis_may_not_generate", "this_report_stays_flagged", "control_may_read_late"]
    ] = Field(
        default_factory=list,
        description="What following the step as written cannot guarantee. unpaired_arms: an arm that differs from the "
        "control beyond its model would run in a launch call of its own, whose new cases the control does not run; a "
        "contrast pairs only the cases both sides ran, so such runs add no pair to it (unpaired_arms names each). A "
        "step whose caveat reads a contrast on this report launches no such arm, and makes no call where no arm it "
        "names is left beside the control (no_action lift_not_shown); a confirmation launches it, and the planned "
        "size does not hold for it. A gate or a figure reads each arm's own cases, so its step names none. "
        "axis_may_not_generate: n_variations writes unseen cases only on a template "
        "axis a model writes (llm); on an axis of stated values (enum, sample) a launch reuses stored cases with the "
        "same values and keeps fewer where they run out, and a kind that generates none refuses it. The reading holds "
        "no template's axes, so the cases are new only where the launch reports reused 0 and kept the planned count; "
        "for a confirmation, unseen cases on such an axis are authored first. this_report_stays_flagged: the caveat "
        "is read off this report's own results (a criterion set after them, a pick among them), which nothing added "
        "anywhere lifts; the confirmation is read in the campaign the step creates, and that reading is the one to "
        "trust. control_may_read_late: the control is named from a launched run, which exists only once the launch "
        "starts; named after the first result is scored, the created campaign reads its control as named late.",
    )
    unpaired_arms: list[str] = Field(
        default_factory=list,
        description="For limits unpaired_arms: the arms whose new cases no call of the control's runs, by id: "
        "launched in a call of their own on a confirmation, not launched on a step that reads this report's contrast.",
    )

    @model_validator(mode="after")
    def _priced_and_planned(self) -> Self:
        if self.spends and self.priced_by is None:
            raise ValueError("a step that spends names the action that prices it")
        if self.planned_cases is not None and self.unsized is not None:
            raise ValueError("a step states its planned size, or why it has none, and not both")
        if self.size_bias is not None and self.planned_cases is None:
            raise ValueError("a size bias qualifies a planned size")
        if self.declared_look and self.planned_cases is None:
            raise ValueError("a step that is one more look states its planned size")
        if self.unsized is not None and self.action is not None:
            raise ValueError("a step with no planned size makes no call: it is never taken open-ended")
        if self.remedy == "no_action" and self.no_action is None:
            raise ValueError("a step that takes no action says why none does")
        if self.no_action is not None and self.action is not None:
            raise ValueError("a step an action takes has no reason no action takes it")
        if (self.action is None) != (not self.calls):
            raise ValueError("a step an action takes makes at least one call of it, and a step none takes makes none")
        if self.action is None and (self.then or self.after or self.as_launched):
            raise ValueError("a step no action takes makes no call, and none follows it")
        named = [*self.needs, *self.as_launched, *self.after]
        if len(set(named)) != len(named) or any(set(call) & set(named) for call in self.calls):
            raise ValueError("each parameter is filled by a call, left to the taker, or read off an earlier call: once")
        if any(step.then for step in self.then):
            raise ValueError("a call that follows another holds no calls of its own to follow it")
        if bool(self.unpaired_arms) != ("unpaired_arms" in self.limits):
            raise ValueError("a step names the arms it cannot pair exactly where it states that it cannot")
        if self.call_arms and len(self.call_arms) != len(self.calls):
            raise ValueError("a step that says which arms its calls run says it for every call")
        if self.alike and "apparatus_settings" not in self.as_launched:
            raise ValueError("a dimension held alike is one the taker sets in apparatus_settings")
        return self


class FirstScreen(_Part):
    """What a report's first screen holds beside the answer, in four fixed slots, as typed values and no text.

    The gate (can this be read at all), the trust slot's reasons (at most three caveats, ranked, with the count past
    them), and the call. The answer slot holds no field: it is the verdicts and warnings the input already carries.
    Computed by one function over the caveat list
    (:func:`~threetears.evals.analysis.first_screen.first_screen`), and left out of the input's digest, since it is a
    function of the rest of the input and of a call that may be made after the freeze.
    """

    gate: Gate | None = Field(default=None, description="What stopped the read; none when the campaign can be read.")
    reasons: list[str] = Field(default_factory=list, max_length=3, description="The ranked caveats shown, by id.")
    remainder: int = Field(default=0, ge=0, description="The eligible caveats past the three shown.")
    open_report_caveats: int = Field(default=0, ge=0, description="Every open report-wide caveat, shown or not.")
    called_arms: list[str] = Field(
        default_factory=list, description="The arms code's own read of the options calls: what ranks a caveat on them."
    )
    call: CallRead | None = Field(default=None, description="The call, where one was made.")
    steps: list[RemedyStep] = Field(
        default_factory=list,
        description="What would change each reason shown, in their order; under a gate, what would lift it.",
    )


class ReportInput(_Part):
    """One campaign's results as the report planner reads them (``report-input/1``)."""

    schema_: Literal["report-input/1"] = Field(alias="schema", description="The document's schema.")
    meta: Meta
    arms: list[Arm] = Field(min_length=1)
    aspects: list[Aspect]
    floors: list[FloorRead] = Field(default_factory=list, description="Each floor's status on each candidate.")
    bars: list[BarRead] = Field(default_factory=list, description="Each arm against each absolute bar.")
    guardrails: list[dict[str, object]] = Field(
        default_factory=list, description="Each guardrail check, as the pipeline read it."
    )
    criteria: list[Criterion] = Field(default_factory=list, description="The declared thresholds.")
    scoreboard: dict[str, ScoreboardRow] = Field(
        description="How each arm's trials came out on the lead aspect, by arm."
    )
    measures: dict[str, Measure] = Field(description="Each measure, by key.")
    judged: dict[str, Judged] = Field(description="Each judged dimension, by key.")
    primary_judged: str | None = None
    contrasts: list[Contrast]
    contrast_families: list[ContrastFamily] = Field(default_factory=list)
    strata: list[Stratum] = Field(default_factory=list)
    examples: list[Example] = Field(default_factory=list)
    caveats: Caveats
    prior_insights: list[PriorInsight] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list, description="The named tasks, in order.")
    factors: list[Factor] = Field(default_factory=list, description="A factorial grid's factors, in order.")
    # Left out of the document when empty, so an input naming no lever digests as it did.
    lever_names: dict[str, str] = Field(
        default_factory=dict,
        exclude_if=lambda names: not names,
        description=(
            "What a reader calls each lever an arm's settings key, by its id: the name the campaign stores for it, "
            "else the one its host declares. The one source every name of a lever reads (an arm's name computed from "
            "its levers, a setting, a factor, a lever-effect row); a lever absent here is printed as its id in words."
        ),
    )
    configs: list[Config] = Field(default_factory=list, description="Each arm's level of each factor.")
    target_matrix: TargetMatrix | None = Field(default=None, description="A classifier's target matrix.")
    cells: dict[str, Cell] = Field(default_factory=dict, description="Each target-matrix cell, by id.")
    confusion: dict[str, ConfusionTable] = Field(
        default_factory=dict, description="Each label axis's confusion, by axis id."
    )
    human_pref: HumanPreference | None = Field(default=None, description="Blind human ratings.")
    # Left out of the document when unset, as an uncounted input and a bare one leave them: their digest is as it was.
    failed_trials: list[FailedTrials] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The trials the run failed per aspect, arm and stratum, for any cause; none where the input does "
        "not count them.",
    )
    case_overlap: list[CaseOverlap] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="For each two aspects, the cases both ran and the cases only one ran; none where the input does "
        "not hold the aspects' case ids.",
    )
    # Left out of the document when unset, as an input over runs that recorded no apparatus leaves it, so its digest
    # is as it was.
    apparatus: list[RunRig] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The rig each run was measured with, for each run that recorded one past a serial run pinning and "
        "declaring nothing; none where no run did, or the bundle carried none.",
    )
    # Left out of the document when unset, so an input built before the field existed digests as it did.
    across_aspects: AcrossAspects | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Each arm's figures across every aspect, each aspect weighed alike; none where the bundle read "
        "none.",
    )
    # The two below are left out of the document when unset, so an input built before they existed digests as it did.
    caveat_list: list[CaveatEntry] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Every qualification of the answer, each once, typed; none on an input built before the list. "
        "Derived, so never part of the input's digest.",
    )
    first_screen: FirstScreen | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The first screen's slots, as shown when the input was frozen; never part of the input's digest.",
    )


#: The keys of the document no digest reads: each a function of the rest of the input, of the bundle the input was
#: built from (whose own fingerprint a snapshot holds) and of a call made later. Identity is over inputs only, so a
#: change to how caveats are raised or ranked re-keys no snapshot and moves no recording.
_OUTSIDE_THE_DIGEST = frozenset({"caveat_list", "first_screen"})


def report_input_digest(report_input: ReportInput) -> str:
    """The digest of a report input: sha256 of its canonical JSON (keys sorted, by alias).

    :param report_input: The report input.
    :ptype report_input: ReportInput
    :return: A 64-character lowercase hex digest.
    :rtype: str
    """
    return report_input_document_digest(report_input.model_dump(mode="json", by_alias=True))


def report_input_document_digest(document: Mapping[str, Any]) -> str:
    """The digest of a report input already dumped to JSON (by alias), as :func:`report_input_digest` takes it.

    What lets a frozen report input be checked against the digest stored beside it without being validated by the
    model the running build holds, which may have moved since the freeze.

    :param document: ``ReportInput.model_dump(mode="json", by_alias=True)``, or a document read back from one.
    :ptype document: Mapping[str, Any]
    :return: A 64-character lowercase hex digest.
    :rtype: str
    """
    document = {key: value for key, value in document.items() if key not in _OUTSIDE_THE_DIGEST}
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return bytes_digest(canonical.encode("utf-8"))
