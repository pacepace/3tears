"""The report input: one campaign's results, normalized for the report planner (``report-input/1``).

The input sits between the pipeline and the planner (``docs/eval-report-kit/GENERATOR.md``): an adapter builds it
from the campaign bundle, the planner reads it to write a report plan, and code reads it to classify the eval
(candidate structure, baseline, score type, strata, criteria, cross-cutting aspects, spread) before the planner
chooses anything. Its documentation copy is ``docs/eval-report-kit/schema/report-input.schema.json``, which states
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

**Counted off the bundle, for an analyst.** ``failed_trials`` (each slice's failed trials, the pool its examples are
kept from) and ``case_overlap`` (which cases each two aspects share, by stored case id) are tallies of the bundle's
trials, not statistics: an analyst's snapshot counts them
(:func:`~threetears.evals.analysis.analyst.snapshot.analyst_input`). A planner's input and a bare one leave them
``None``, which reads as not known, never as none.

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
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ReportInput"]


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


class Depth(_Part):
    """How deeply one arm ran an aspect."""

    cases: int = Field(description="Distinct cases.")
    trials: int = Field(description="Trials over those cases.")
    repeats_per_case: list[Number] = Field(description="Fewest, mean and most repeats a case got.")


class PassK(_Part):
    """Where pass^k is compared, and at what depth."""

    where: str = Field(description="The aspect read.")
    k: int = Field(description="The k compared.")
    common_depth: int = Field(description="The fewest runs any arm gave a shared case.")
    shares_no_case: bool = Field(description="Whether the arms share no case at that depth.")
    unmeasured_for: list[str] = Field(default_factory=list, description="Arms pass^k is not measured for.")
    said: list[str] = Field(default_factory=list, description="What the pipeline says about it.")


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
    census (``ArmScoreboard.pricing``), which reads it off the arm's production usage rows.

    It is part of a report input's content address (``report_input_digest``, a snapshot's id), so two inputs priced
    differently are two snapshots; it is no part of any fingerprint (owner ruling, 2026-10-28).
    """

    sources: list[str] = Field(
        default_factory=list, description="Each price source its priced calls name (RoleUsage.price_source), sorted."
    )
    providers: list[str] = Field(
        default_factory=list,
        description="Each provider its calls name as serving them (RoleUsage.provider), sorted; empty where none does.",
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
        default=None, description="Whether a judge grades the aspect or a check in code does."
    )


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


class Level(_Part):
    """One level of a factor, axis or bin."""

    id: str = Field(description="The level's id.")
    name: str | None = Field(default=None, description="What a reader calls it.")


class Measure(_Part):
    """One measure: what it means and each arm's figure on it."""

    label: str | None = Field(default=None, description="What a reader calls it.")
    unit: str | None = Field(default=None, description="Its unit, e.g. usd, pct, ms, usd_per_1k.")
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


class FailedTrials(_Part):
    """How many trials of one slice of the run failed: the pool its examples are kept from."""

    aspect: str = Field(description="The aspect the slice ran on.")
    arm: str = Field(description="The arm that ran it.")
    stratum: str | None = Field(default=None, description="The stratum or task it falls in; none for none.")
    failed: int = Field(ge=0, description="The trials of the slice that failed, as the examples count a failure.")


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
    configs: list[Config] = Field(default_factory=list, description="Each arm's level of each factor.")
    target_matrix: TargetMatrix | None = Field(default=None, description="A classifier's target matrix.")
    cells: dict[str, Cell] = Field(default_factory=dict, description="Each target-matrix cell, by id.")
    confusion: dict[str, ConfusionTable] = Field(
        default_factory=dict, description="Each label axis's confusion, by axis id."
    )
    human_pref: HumanPreference | None = Field(default=None, description="Blind human ratings.")
    # Left out of the document when unset, as a planner's input and a bare one leave them, so their digest is as it was.
    failed_trials: list[FailedTrials] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="The failed trials the run holds per aspect, arm and stratum, which the examples are kept from; "
        "none where the input does not count them.",
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
