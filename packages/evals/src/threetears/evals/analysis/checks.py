"""Every instrument is checked: the check record, what decides one, and which check speaks for a judge.

A check names an instrument, what it was checked against (people, a model, itself, or identical arms), on what sample,
with one result (a :data:`~threetears.evals.kernel.evidence_tiers.CriterionState`, with a reason from
:mod:`~threetears.evals.schema.reading_states` where it is not ``met``). It is one of three kinds, each its own shape
holding only its own fields: a judge's (:class:`JudgeCheck`), a reviewer's (:class:`ReviewerCheck`) and the rig's
(:class:`RigCheck`); :data:`Check` is the union of the three, told apart by a required ``kind``. It replaces the evidence tier as what a reader
and a decision are told about an instrument: the tier is still stored, decided from the people check and the itself
check exactly as before (:func:`tier_checks` reads those two off it), and nothing here prints or reads a tier value.

- :func:`tier_checks` is a judge's check against people and against itself, from the two criteria its stored tier was
  decided by; :func:`judge_checks` lists them for the judges that scored a dimension.
- :func:`reviewer_check` is a model that answered a blind read, under one served model and effort, checked against
  people: by its kappa, and by its lean toward an arm over the pair items both read (the model's difference between
  the arms less the person's). Kappa is blind to a lean the model and a judge share, so the lean is the check that can
  see it. ``met`` is an equivalence reading: the lean shown inside ``±δ`` (:data:`REVIEWER_LEAN_BAR` of the scale's
  width, recorded on the check) by the bounded equivalence test at 5%.
- :func:`blind_read_checks` is every check one closed blind read yields: the judge against people, the judge against
  each model that answered (against itself, where the model is the judge's own), and each such model against people.
- :func:`speaking_check` is which of a judge's checks speaks for it on a criterion: people's once decided, else a
  model's, else none.

Person and model ratings are never pooled into one figure, and a model's rating never enters a check against people.
A corrected reading is a check's result carried onto one contrast; a re-score is the judge checked against itself;
looks are how often a contrast's test was read against how often it planned to be. One class per shape, held by the
bundle and the report input alike, so a fact has one spelling at both ends; no shape carries a sentence saying why a
figure is absent. ``tests/test_checks.py`` and ``tests/test_simulated_checks.py`` hold what this module computes.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Final, Literal, NamedTuple, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from threetears.evals.analysis.agreement import (
    JudgeKey,
    instrument_agreement,
    judge_agreement,
    judge_instrument,
    judge_key,
    judge_key_of,
    judge_sort_key,
    person_scores_by_result,
)
from threetears.evals.analysis.stats import (
    INTERVAL_LEVEL,
    paired_equivalence,
    t_critical_two_sided,
)
from threetears.evals.kernel.evidence_tiers import (
    CriterionState,
    JudgeEvidenceTier,
    TierCriterion,
    agreement_statistic,
    calibration_criterion,
    separation_criterion,
)
from threetears.evals.schema.models import SCALES, RubricScale
from threetears.evals.schema.reading_states import (
    Coordinate,
    NotAvailableReason,
    Reason,
    SampleKind,
    SetAsideReason,
    StateOf,
)

if TYPE_CHECKING:
    from threetears.evals.analysis.blind_read import BlindClose
    from threetears.evals.kernel.result_reading import ResultView
    from threetears.evals.schema.blind_read import BlindRead
    from threetears.evals.schema.models import CalibrationRating, RaterKind

__all__ = [
    "Check",
    "Corrected",
    "JudgeCheck",
    "Looks",
    "Rescore",
    "ReviewerCheck",
    "RigCheck",
    "blind_read_checks",
    "judge_checks",
    "speaking_check",
]

#: The bar a reviewer's lean is held inside, as a share of its criterion's scale width: an instrument's acceptance bar,
#: as CALIBRATION_MIN_AGREEMENT is its kappa's, and recorded on every check that reads a lean (``lean_bar``, in score
#: units). It is the package's, never a host's margin: a reading's margin compares arms in the measure's units, and a
#: lean is in score units. ``met`` reads it by the bounded equivalence test at 5%: on pass/fail an unbiased reviewer is
#: shown inside it in about 40% of reads at 105 pair items and 98% at 260, never below 59 (``0.95 ** n`` must fall
#: under 5%) (``tests/test_simulated_checks.py``).
REVIEWER_LEAN_BAR: Final = 0.10

#: The fewest pair items a reviewer's lean demotes it from. A floor in the safe direction: a reviewer shown to lean only
#: sets an improvement aside.
REVIEWER_LEAN_MIN_PAIR_ITEMS: Final = 10

#: The halvings the search for the lean bound an undecided check reached takes: to within 1/1024 of the scale's width.
_BOUND_STEPS: Final = 10

#: The share of a count the search for the pair items a check needs may stop within: a plan needs no exact count.
_NEEDED_RESOLUTION: Final = 0.05

#: How far past the pair items in hand that search reads, as a multiple of them (and at least
#: :data:`_NEEDED_SEARCH_FLOOR`): past it a check says "more than" that many, which already tells a host the double
#: reads it would collect are out of proportion to the read it has.
_NEEDED_SEARCH_FACTOR: Final = 10
_NEEDED_SEARCH_FLOOR: Final = 1_000

#: The results that decide a check: a criterion read at its floor, either way.
_DECIDED: Final = frozenset({"met", "not_met"})


class _Shape(BaseModel):
    """Immutable, and refusing a key it does not declare."""

    model_config = ConfigDict(frozen=True, extra="forbid")


def _result_and_reason(
    result: CriterionState, reason: Reason | None, coordinate: Coordinate | None, state: StateOf | None
) -> None:
    """Refuse a reason a check's result does not carry, and a coordinate apart from the reason that names one: the one
    rule every shape carrying a check's result is held to.

    :raises ValueError: Naming which.
    """
    if result in ("met", "undecided") and reason is not None:
        raise ValueError(f"a {result} check carries no reason, and {reason!r} was given")
    if result == "not_met" and reason not in get_args(SetAsideReason):
        raise ValueError(f"a not_met check carries the reason it sets a reading aside for, not {reason!r}")
    if result == "insufficient":
        if reason is None and state is None:
            raise ValueError("an insufficient check says why: a reason, or the state of a check not made")
        if reason is not None and reason not in get_args(NotAvailableReason):
            raise ValueError(f"an insufficient check carries why there is too little, not {reason!r}")
    if (reason == "identity_unrecorded") != (coordinate is not None):
        raise ValueError("an unrecorded identity names its coordinate, and only it names one")


class _CheckBase(_Shape):
    """What every kind of check holds: the instrument, the criterion, the sample, and one result with its reason.

    ``not_met`` carries the reason it sets a reading aside for (``instrument_below_bar``, ``instrument_leans``,
    ``people_disagree``, ``identical_arms_separated``). ``insufficient`` carries why there is too little to decide
    (``too_few``, ``not_recorded``, ``no_variation``, ``identity_unrecorded`` with its coordinate, ...), or, on a check
    of the rig, ``state`` where it could not be made at all. ``met`` and ``undecided`` carry none.
    """

    instrument: str = Field(
        description="Which one: a judge by its served model, config, temperature and sent settings "
        "(`judge_instrument`), a reviewer by the model id it was asked as, the rig by its launch group."
    )
    criterion: str | None = Field(default=None, description="The criterion the check decides, where it names one.")
    sample: SampleKind | None = Field(
        default=None,
        description="Whether the items it was read on were drawn at random, chosen, or no sample stands behind it.",
    )
    sample_results: int | None = Field(default=None, description="Results read.")
    sample_cases: int | None = Field(default=None, description="Distinct cases behind them.")
    result: CriterionState = Field(description="met, not_met, undecided or insufficient.")
    reason: Reason | None = Field(default=None, description="Why the result is not met, where a shared reason says it.")
    coordinate: Coordinate | None = Field(
        default=None, description="The coordinate an unrecorded identity names, where that is the reason."
    )

    @model_validator(mode="after")
    def _result_and_reason_belong_together(self) -> Self:
        """Refuse a reason its result does not carry, and a coordinate apart from the reason that names one.

        :raises ValueError: Naming which.
        """
        _result_and_reason(self.result, self.reason, self.coordinate, getattr(self, "state", None))
        return self


class JudgeCheck(_CheckBase):
    """A judge checked against people, against a model that answered a blind read, or against itself."""

    kind: Literal["judge"] = Field(description="A judge's check.")
    against: Literal["people", "model", "itself"] = Field(description="What it was checked against.")
    against_id: str | None = Field(default=None, description="The model it was checked against, by the id asked as.")
    against_served_model: str | None = Field(
        default=None, description="The model that served that model's answers, as the provider named it."
    )
    against_effort: str | None = Field(default=None, description="The effort that model was asked for, where it was.")
    same_family_as: list[str] = Field(
        default_factory=list,
        description="The arms whose candidate model is of the family of the model checked against.",
    )
    same_family_as_judge: bool | None = Field(
        default=None,
        description="Whether the model checked against is of the judge's own family; None where either family is not "
        "recorded, or no model was checked against.",
    )
    skipped: int | None = Field(default=None, description="Items a reader skipped.")
    read_id: str | None = Field(default=None, description="The blind read the sample is, by id.")
    kappa: float | None = Field(default=None, description="Agreement beyond chance.")
    kappa_interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    reads: int | None = Field(
        default=None, description="Pairs read behind the figures; on a check with no figure, the pairs left out."
    )


class ReviewerCheck(_CheckBase):
    """A model that answered a blind read, under one served model and effort, checked against people: by its kappa,
    and by its lean toward an arm over the pair items both read."""

    kind: Literal["reviewer"] = Field(description="A reviewer's check.")
    against: Literal["people"] = Field(default="people", description="A reviewer is checked against people.")
    served_model: str | None = Field(
        default=None,
        description="The model that served the answers, as the provider named it; None where it was not recorded, "
        "and then no figure is read.",
    )
    effort: str | None = Field(default=None, description="The effort the model was asked for, where it was.")
    skipped: int | None = Field(default=None, description="Items a reader skipped.")
    read_id: str | None = Field(default=None, description="The blind read the sample is, by id.")
    kappa: float | None = Field(default=None, description="Agreement with people beyond chance.")
    kappa_interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    reads: int | None = Field(
        default=None,
        description="The model's reads of the criterion; on a check with no figure, the reads left out.",
    )
    pair_items: int | None = Field(
        default=None, description="The pair items its lean was read over, both arms read by both."
    )
    left_out: int | None = Field(
        default=None,
        description="The pair items left out of its lean because a run of theirs has an unrecorded coordinate, which "
        "matches nothing.",
    )
    lean: float | None = Field(
        default=None, description="The lean toward the candidate arm: the model's difference less people's."
    )
    lean_interval: tuple[float, float] | None = Field(
        default=None, description="The t interval on it, at the house level: what shows a lean."
    )
    lean_bar: float | None = Field(
        default=None,
        description="The lean the equivalence test holds it inside, in score units: REVIEWER_LEAN_BAR of the scale's "
        "width. Set wherever a lean was read.",
    )
    lean_bound: float | None = Field(
        default=None,
        description="On an undecided check, the smallest lean toward either arm the bounded equivalence test rules "
        "out at 5%: a lean up to it is not ruled out. None where no bound within the scale's width is reached.",
    )
    pair_items_needed: int | None = Field(
        default=None,
        description="On an undecided check over two pair items or more: about how many pair items like these (the "
        "observed leans repeated to the count) the bounded test would show inside the bar at.",
    )
    pair_items_needed_above: int | None = Field(
        default=None,
        description="On an undecided check where no count up to its search's cap shows the lean inside the bar (as "
        "where the observed lean is at the bar or past it): that cap, ten times the pair items in hand and at least "
        "1,000. More than this many would be needed.",
    )

    @model_validator(mode="after")
    def _projections_are_an_undecided_check_s(self) -> Self:
        """Refuse a bound or a count of items needed on a check that is not undecided.

        :raises ValueError: Naming which.
        """
        projected = (self.lean_bound, self.pair_items_needed, self.pair_items_needed_above)
        if self.result != "undecided" and any(value is not None for value in projected):
            raise ValueError("only an undecided check carries the bound it reached and the pair items it needs")
        if self.pair_items_needed is not None and self.pair_items_needed_above is not None:
            raise ValueError("a check needs a count of pair items, or more than its search read: not both")
        if (
            self.result == "undecided"
            and (self.pair_items or 0) >= 2
            and self.pair_items_needed is None
            and self.pair_items_needed_above is None
        ):
            raise ValueError("an undecided check over two pair items or more says how many it needs")
        return self


class RigCheck(_CheckBase):
    """The rig checked against identical arms: the A/A check on one reading (``Contrast.aa``)."""

    kind: Literal["rig"] = Field(description="The rig's check.")
    against: Literal["identical_arms"] = Field(
        default="identical_arms", description="The rig is checked against identical arms."
    )
    state: StateOf | None = Field(
        default=None,
        description="Where the check could not be made at all (both sides replayed a recording): not available, why.",
    )
    delta: float | None = Field(default=None, description="The difference between identical arms.")
    interval: tuple[float, float] | None = Field(default=None, description="The interval on that difference.")
    p_adjusted: float | None = Field(default=None, description="Its adjusted p.")
    reading: str | None = Field(default=None, description="The reading a check of the rig covers.")
    covers_arms: list[str] = Field(default_factory=list, description="The arms a check of the rig covers.")


#: One check of one instrument: what it was checked against, on what sample, with one result. Its ``kind`` names which
#: shape it is, and is required: each kind holds its own fields and refuses another's.
Check = Annotated[JudgeCheck | ReviewerCheck | RigCheck, Field(discriminator="kind")]


# --- a judge's checks off its stored tier ---------------------------------------------------------------------------


def _criterion_reason(criterion: TierCriterion) -> Reason | None:
    """The shared reason a tier criterion's state carries as a check's result."""
    if criterion.state == "not_met":
        return "instrument_below_bar"
    if criterion.state != "insufficient":
        return None
    if criterion.n == 0:
        return "not_recorded"
    return "no_variation" if criterion.agreement is None else "too_few"


def _criterion_check(
    criterion: TierCriterion,
    *,
    instrument: str,
    dim: str,
    against: Literal["people", "model", "itself"],
    sample: SampleKind | None = None,
    sample_cases: int | None = None,
    read_id: str | None = None,
) -> JudgeCheck:
    """One agreement criterion of a judge, as the check it is."""
    return JudgeCheck(
        kind="judge",
        instrument=instrument,
        criterion=dim,
        against=against,
        sample="none" if criterion.n == 0 else sample,
        sample_results=criterion.results,
        sample_cases=sample_cases,
        read_id=read_id,
        result=criterion.state,
        reason=_criterion_reason(criterion),
        kappa=criterion.agreement,
        kappa_interval=criterion.interval,
        reads=criterion.n,
    )


def tier_checks(tier: JudgeEvidenceTier) -> list[JudgeCheck]:
    """A judge's check against people and its check against itself, from the two criteria its stored tier was decided
    by (``calibration`` and ``separation``). The tier is those two results as
    :func:`~threetears.evals.kernel.evidence_tiers.tier_of` reads them, and nothing else.

    :param tier: The judge's stored tier on one dimension.
    :ptype tier: JudgeEvidenceTier
    :return: The people check, then the itself check.
    :rtype: list[JudgeCheck]
    """
    instrument = judge_instrument(judge_key_of(tier))
    return [
        _criterion_check(tier.calibration, instrument=instrument, dim=tier.rubric_dim, against="people"),
        _criterion_check(tier.separation, instrument=instrument, dim=tier.rubric_dim, against="itself"),
    ]


def judge_checks(tiers: Iterable[JudgeEvidenceTier], judges: Iterable[JudgeKey]) -> list[Check]:
    """The checks of each judge in ``judges``, read off its stored tier (:func:`tier_checks`), ordered by judge.

    A judge with no tier was measured by nothing: its two checks are ``insufficient``, ``not_recorded``.

    :param tiers: The judges' stored tiers (``judge_evidence_tiers``).
    :ptype tiers: Iterable[JudgeEvidenceTier]
    :param judges: The judges whose checks to list (:func:`~threetears.evals.analysis.agreement.judge_key`).
    :ptype judges: Iterable[JudgeKey]
    :return: Two checks per judge.
    :rtype: list[Check]
    """
    by_judge = {judge_key_of(tier): tier for tier in tiers}
    checks: list[Check] = []
    for key in sorted({JudgeKey(*judge) for judge in judges}, key=judge_sort_key):
        tier = by_judge.get(key)
        if tier is not None:
            checks += tier_checks(tier)
            continue
        instrument = judge_instrument(key)
        checks += [
            _criterion_check(
                calibration_criterion(0, 0, None), instrument=instrument, dim=key.rubric_dim, against="people"
            ),
            _criterion_check(
                separation_criterion(0, 0, None), instrument=instrument, dim=key.rubric_dim, against="itself"
            ),
        ]
    return checks


# --- a reviewer checked against people ------------------------------------------------------------------------------


def _lean_interval(leans: Sequence[float]) -> tuple[float | None, tuple[float, float] | None]:
    """The mean lean over pair items and its two-sided interval at the house level, on t with ``n - 1`` degrees.

    :param leans: Per pair item, the model's difference between the arms less the person's.
    :ptype leans: Sequence[float]
    :return: ``(mean, interval)``: the mean None with no item, the interval None below two.
    :rtype: tuple[float | None, tuple[float, float] | None]
    """
    n = len(leans)
    if n == 0:
        return None, None
    mean = math.fsum(leans) / n
    if n < 2:
        return mean, None
    spread = math.sqrt(math.fsum((lean - mean) ** 2 for lean in leans) / (n - 1))
    half = t_critical_two_sided(INTERVAL_LEVEL, n - 1) * spread / math.sqrt(n)
    return mean, (mean - half, mean + half)


def _shown_inside(leans: Sequence[float], bar: float, width: float) -> bool:
    """Whether the bounded equivalence test at 5% shows the mean lean inside ``±bar``; leans lie within twice the width."""
    equivalent, _ = paired_equivalence(leans, bar, value_range=(-width, width))
    return equivalent is True


def _lean_bound(leans: Sequence[float], width: float) -> float | None:
    """The smallest bar the bounded equivalence test shows the leans inside, found by halving to within
    ``width / 2 ** _BOUND_STEPS``: the test reaches it, so a lean beyond it is ruled out and one up to it is not.

    The test's p falls as the bar widens, so the halving keeps a bar the test reaches above one it does not.

    :return: The bound; None where even the scale's whole width is not reached.
    """
    if not _shown_inside(leans, width, width):
        return None
    low, high = 0.0, width
    for _ in range(_BOUND_STEPS):
        middle = (low + high) / 2
        if _shown_inside(leans, middle, width):
            high = middle
        else:
            low = middle
    return high


def _pair_items_needed(leans: Sequence[float], bar: float, width: float) -> tuple[int | None, int | None]:
    """About how many pair items like these would show the lean inside ``±bar``: the observed leans repeated in order to
    the count (their lean and their spread, taken as new items' would be), and the bounded test read again at each
    count, doubling and then halving as the house's projections do
    (:func:`~threetears.evals.analysis.stats.bounded_cases_to_clear`). A size to plan, never a promise: a fresh sample
    lands either side of it.

    :return: ``(needed, above)``: the count, the observed count where the lean already shows (the kappa is what is
        undecided), with ``above`` None; or None and the search's cap (:data:`_NEEDED_SEARCH_FACTOR` times the items in
        hand, at least :data:`_NEEDED_SEARCH_FLOOR`) where no count up to it shows, as where the observed lean is at
        the bar or past it.
    """
    observed = len(leans)
    cap = max(_NEEDED_SEARCH_FLOOR, _NEEDED_SEARCH_FACTOR * observed)
    if abs(math.fsum(leans) / observed) >= bar:
        # No count of items like these shows a lean at the bar or past it inside the bar.
        return None, cap

    def shows(count: int) -> bool:
        return _shown_inside([leans[i % len(leans)] for i in range(count)], bar, width)

    if shows(observed):
        return observed, None
    low = high = observed
    while True:
        high = min(2 * high, cap)
        if shows(high):
            break
        if high >= cap:
            return None, cap
        low = high
    while high - low > max(1, int(high * _NEEDED_RESOLUTION)):
        middle = (low + high) // 2
        if shows(middle):
            high = middle
        else:
            low = middle
    return high, None


def _reviewer_result(
    kappa: TierCriterion,
    leans: Sequence[float],
    interval: tuple[float, float] | None,
    bar: float,
    width: float,
    left_out: Sequence[Coordinate],
) -> tuple[CriterionState, Reason | None, Coordinate | None]:
    """The reviewer's own check, its rows in the order they bind: below the bar, then a lean, then nothing to read a
    lean over, then the equivalence reading, else undecided.

    The equivalence reading is the bounded test (:func:`~threetears.evals.analysis.stats.paired_equivalence`) at 5% on
    the leans, which lie within twice the scale's width: a t interval inside ``±δ`` claimed it about 8% of the time at
    the bound over 40 pass/fail items (a few rare leans give an interval far too narrow), and the bounded test holds 5%
    at every n.
    """
    if kappa.state == "not_met":
        return "not_met", "instrument_below_bar", None
    if interval is not None and len(leans) >= REVIEWER_LEAN_MIN_PAIR_ITEMS and (interval[0] > 0 or interval[1] < 0):
        return "not_met", "instrument_leans", None
    if not leans and left_out:
        # Every pair item was left out for an unrecorded coordinate: the first the shared order of coordinates lists.
        return "insufficient", "identity_unrecorded", next(c for c in get_args(Coordinate) if c in left_out)
    if not leans:
        return "insufficient", "not_recorded", None
    if kappa.state == "met" and _shown_inside(leans, bar, width):
        return "met", None, None
    return "undecided", None, None


def reviewer_check(
    reviewer: str,
    dim: str,
    scale: RubricScale,
    *,
    kappa_pairs: Sequence[tuple[int, int, str, str]],
    leans: Sequence[float],
    left_out: Sequence[Coordinate] = (),
    served_model: str | None = None,
    effort: str | None = None,
    reads: int | None = None,
    read_id: str | None = None,
    sample: SampleKind | None = None,
) -> ReviewerCheck:
    """A model that answered a blind read, under one served model and effort, checked against people on one criterion:
    its kappa, and its lean.

    ``met`` is an equivalence reading: the lean shown inside ``±δ`` by the bounded equivalence test at 5% (valid at
    every n on coarse scores, as a t interval is not), and the kappa criterion ``met`` by its rule. ``δ`` is
    :data:`REVIEWER_LEAN_BAR` of the scale's width, recorded on the check as ``lean_bar``. ``not_met`` is the kappa
    shown below its bar (``instrument_below_bar``: the judge reads as not checked against this model on the
    criterion), or the lean's t interval excluding zero over :data:`REVIEWER_LEAN_MIN_PAIR_ITEMS` items or more
    (``instrument_leans``: an improvement read by a judge checked only against this model is set aside for the arm it
    leans toward). With no pair item it is ``insufficient``: ``identity_unrecorded`` on a coordinate where every one
    was left out for an unrecorded one, else ``not_recorded``. Anything else is ``undecided``, carrying the bound the equivalence test
    reached (a lean of either sign up to it is not ruled out, and "cannot tell apart" is never "no lean") and about how
    many pair items like these would decide it.

    :param reviewer: The model's id, as it was asked.
    :ptype reviewer: str
    :param dim: The criterion.
    :ptype dim: str
    :param scale: Its scale.
    :ptype scale: RubricScale
    :param kappa_pairs: ``(model's score, person's score, person, output)`` for every output both read.
    :ptype kappa_pairs: Sequence[tuple[int, int, str, str]]
    :param leans: Per pair item both read on both arms, the model's candidate-less-control difference less the
        people's (averaged over the people who read it).
    :ptype leans: Sequence[float]
    :param left_out: Per pair item left out of ``leans``, the coordinate a run of its left unrecorded.
    :ptype left_out: Sequence[Coordinate]
    :param served_model: The model that served the answers read here.
    :ptype served_model: str | None
    :param effort: The effort it was asked for, where it was.
    :ptype effort: str | None
    :param reads: How many reads of the criterion the model made over the population.
    :ptype reads: int | None
    :param read_id: The blind read, by id.
    :ptype read_id: str | None
    :param sample: How the people's items were sampled.
    :ptype sample: SampleKind | None
    :return: The check.
    :rtype: ReviewerCheck
    """
    low, high = SCALES[scale].scores
    width = float(high - low)
    bar = REVIEWER_LEAN_BAR * width
    agreement = instrument_agreement(scale, kappa_pairs)
    kappa = calibration_criterion(agreement.n, agreement.results, agreement.agreement, agreement.interval)
    lean, interval = _lean_interval(leans)
    result, reason, coordinate = _reviewer_result(kappa, leans, interval, bar, width, left_out)
    undecided = result == "undecided" and len(leans) >= 2
    needed, above = _pair_items_needed(leans, bar, width) if undecided else (None, None)
    return ReviewerCheck(
        kind="reviewer",
        instrument=reviewer,
        served_model=served_model,
        effort=effort,
        criterion=dim,
        sample=sample,
        sample_results=agreement.results,
        pair_items=len(leans),
        left_out=len(left_out),
        read_id=read_id,
        result=result,
        reason=reason,
        coordinate=coordinate,
        kappa=agreement.agreement,
        kappa_interval=agreement.interval,
        lean=lean,
        lean_interval=interval,
        lean_bar=bar,
        lean_bound=_lean_bound(leans, width) if undecided else None,
        pair_items_needed=needed,
        pair_items_needed_above=above,
        reads=reads,
    )


# --- which check speaks ---------------------------------------------------------------------------------------------


def speaking_check(checks: Iterable[Check], instrument: str, criterion: str) -> JudgeCheck | None:
    """Which of a judge's checks speaks for it on a criterion.

    The check against people, once it is decided (``met`` or ``not_met``, which its floor of distinct results gates).
    Else a decided check against a model, unless that model's own check against people, under the same served model
    and effort, shows its kappa below the bar (``instrument_below_bar``); a model shown to lean still speaks, and its
    lean sets an improvement aside where the decision is read. Else none: the judge is not checked on the criterion.

    :param checks: Every check in hand: the judge's, and the reviewers' own.
    :ptype checks: Iterable[Check]
    :param instrument: The judge (:func:`~threetears.evals.analysis.agreement.judge_instrument`).
    :ptype instrument: str
    :param criterion: The criterion.
    :ptype criterion: str
    :return: The check that speaks, or None.
    :rtype: JudgeCheck | None
    """
    held = list(checks)
    mine = [c for c in held if isinstance(c, JudgeCheck) and c.instrument == instrument and c.criterion == criterion]
    people = next((c for c in mine if c.against == "people" and c.result in _DECIDED), None)
    if people is not None:
        return people
    own = {(c.instrument, c.served_model, c.effort, c.criterion): c for c in held if isinstance(c, ReviewerCheck)}
    for check in sorted(
        (c for c in mine if c.against == "model" and c.result in _DECIDED),
        key=lambda c: (c.against_id or "", c.against_served_model or "", c.against_effort or ""),
    ):
        reviewer = own.get((check.against_id or "", check.against_served_model, check.against_effort, criterion))
        if reviewer is not None and reviewer.reason == "instrument_below_bar":
            continue
        return check
    return None


# --- every check one blind read yields ------------------------------------------------------------------------------


class _Reviewer(NamedTuple):
    """One reviewer configuration: the model id asked as, the model that served it (None where not recorded), and the
    effort asked for. Answers under two configurations are two instruments, and never pool."""

    model: str
    served_model: str | None
    effort: str | None


def _reviewer_of(read: BlindRead) -> dict[tuple[str, str, str], _Reviewer]:
    """The configuration behind each model rating of the read, by ``(rater, result, criterion)``: off the answer the
    rating came from, the last answer winning as it does where the ratings are read (one rating per result, criterion
    and rater)."""
    items = {item.item: item for item in read.items}
    of: dict[tuple[str, str, str], _Reviewer] = {}
    for answer in read.answers:
        item = items.get(answer.item)
        if answer.rater_kind != "reference" or answer.skipped or item is None:
            continue
        outputs = {output.slot: output.result_id for output in item.outputs}
        configuration = _Reviewer(answer.rater, answer.reviewer_served_model, answer.reviewer_effort)
        for verdict in answer.verdicts:
            if verdict.slot in outputs:
                of[(answer.rater, outputs[verdict.slot], verdict.dim)] = configuration
    return of


def _judge_against(
    ratings: Sequence[CalibrationRating],
    results: Sequence[ResultView],
    *,
    against: Literal["people", "model"],
    read_id: str,
    sample_of: Callable[[str], SampleKind | None],
    rated_by: RaterKind = "person",
) -> list[tuple[JudgeKey, TierCriterion, JudgeCheck]]:
    """Each judge's agreement with ``ratings``, per criterion, as the check it is beside its judge and criterion."""
    if not ratings:
        return []
    agreement = judge_agreement(ratings, results, rated_by=rated_by)
    scores = person_scores_by_result(ratings, results, rated_by=rated_by)
    by_id = {result.id: result for result in results}
    read: list[tuple[JudgeKey, TierCriterion, JudgeCheck]] = []
    for entry in sorted(agreement.dimensions, key=lambda entry: judge_sort_key(judge_key_of(entry))):
        key = judge_key_of(entry)
        statistic = agreement_statistic(entry.scale, entry.kappa, entry.weighted_kappa)
        criterion = calibration_criterion(entry.n, entry.results, statistic, entry.agreement_interval)
        cases = {
            by_id[result_id].test_case_id
            for (result_id, dim) in scores
            if dim == entry.rubric_dim and result_id in by_id and judge_key(by_id[result_id], dim) == key
        }
        check = _criterion_check(
            criterion,
            instrument=judge_instrument(key),
            dim=entry.rubric_dim,
            against=against,
            sample=sample_of(entry.rubric_dim),
            sample_cases=len(cases),
            read_id=read_id,
        )
        read.append((key, criterion, check))
    return read


def blind_read_checks(
    closed: BlindClose,
    results: Iterable[ResultView],
    *,
    other_ratings: Sequence[CalibrationRating] = (),
    judge_models: Collection[str] = (),
    family_of: Callable[[str], str | None] | None = None,
    candidate_models: Mapping[str, str | None] | None = None,
    unrecorded_runs: Mapping[str, Coordinate] | None = None,
) -> list[Check]:
    """Every check a closed blind read yields, people's and each model's apart and never pooled.

    A model that answered is one reviewer per configuration it answered under: the id asked as, the model that served
    it, and the effort asked for. Answers under two configurations are two reviewers' (their checks are apart), and
    answers whose served model was not recorded are left out of every figure: their checks are ``insufficient``,
    ``identity_unrecorded`` on the served model, counting what was left out.

    - **The judge against people**, per judge and criterion, over every person's rating that pairs (the read's own
      and ``other_ratings``): ``random`` where every one came from a rater whose answers are the top of the order.
    - **The judge against each reviewer**, per judge and criterion. A reviewer that is the judge's own (its id, or the
      model that served it, is one the judge was asked as or served by) checks the judge against itself, held to the
      bar a judge's agreement with its repeats is. Where which model served the judge's scores was not recorded, it
      cannot be shown to be another model: the check is ``insufficient``, ``identity_unrecorded`` on the served
      model, with no figure. The family the model shares with an arm's candidate or with the judge is a property of
      the check, never a caveat of its own.
    - **Each reviewer against people**, per criterion a person answered (:func:`reviewer_check`). Pair items either of
      whose runs is in ``unrecorded_runs`` are left out of its lean, and counted.

    :param closed: The closed read: its report, its answers as ratings, which are random and which a model's.
    :ptype closed: BlindClose
    :param results: The results the read was drawn over.
    :ptype results: Iterable[ResultView]
    :param other_ratings: People's ratings from outside the read, which count toward the people check as selected.
    :ptype other_ratings: Sequence[CalibrationRating]
    :param judge_models: The ids the judge was asked as.
    :ptype judge_models: Collection[str]
    :param family_of: Which family a model id is of (``EvalHost.family_of``), None where not recorded; None here
        records no family.
    :ptype family_of: Callable[[str], str | None] | None
    :param candidate_models: Each arm's candidate model, by the arm's key on the read; None where it has none.
    :ptype candidate_models: Mapping[str, str | None] | None
    :param unrecorded_runs: Runs with an unrecorded coordinate, whose pair items no lean reads, each with the coordinate
        it leaves unrecorded.
    :ptype unrecorded_runs: Mapping[str, Coordinate] | None
    :return: The judge's checks against people, then against each reviewer in turn, then each reviewer's own.
    :rtype: list[Check]
    """
    read = closed.read
    results = list(results)
    people = [
        *(rating for rating in closed.ratings if rating.rater_kind == "person"),
        *(rating for rating in other_ratings if rating.rater_kind == "person"),
    ]

    def people_sample(dim: str) -> SampleKind:
        on_dim = [rating for rating in people if rating.rubric_dim == dim]
        return "random" if all(rating.id in closed.random for rating in on_dim) else "selected"

    checks: list[Check] = [
        check
        for _, _, check in _judge_against(people, results, against="people", read_id=read.id, sample_of=people_sample)
    ]
    configuration_of = _reviewer_of(read)
    by_reviewer: dict[_Reviewer, list[CalibrationRating]] = {}
    for rating in closed.ratings:
        if rating.id in closed.reviewed:
            reviewer = configuration_of[(rating.rater, rating.result_id, rating.rubric_dim)]
            by_reviewer.setdefault(reviewer, []).append(rating)
    reviewers = sorted(by_reviewer, key=lambda r: (r.model, r.served_model or "", r.effort or ""))
    samples = {(s.rater, s.rater_kind): s.sample for s in (read.report.samples if read.report else [])}
    family = family_of or (lambda model: None)
    arms = dict(candidate_models or {})
    for reviewer in reviewers:
        mine = family(reviewer.model)
        reviewer_sample = samples.get((reviewer.model, "reference"))

        def sampled(dim: str, sample: SampleKind | None = reviewer_sample) -> SampleKind | None:
            return sample

        who = {
            "against_id": reviewer.model,
            "against_served_model": reviewer.served_model,
            "against_effort": reviewer.effort,
        }
        for key, criterion, check in _judge_against(
            by_reviewer[reviewer], results, against="model", read_id=read.id, sample_of=sampled, rated_by="agent"
        ):
            names = set(judge_models) | ({key.judge_model} if key.judge_model is not None else set())
            if {reviewer.model, reviewer.served_model} & names:
                repeat = separation_criterion(criterion.n, criterion.results, criterion.agreement, criterion.interval)
                itself = _criterion_check(
                    repeat,
                    instrument=check.instrument,
                    dim=key.rubric_dim,
                    against="itself",
                    sample=check.sample,
                    sample_cases=check.sample_cases,
                    read_id=read.id,
                )
                checks.append(itself.model_copy(update=who))
                continue
            if key.judge_model is None or reviewer.served_model is None:
                checks.append(
                    JudgeCheck(
                        kind="judge",
                        instrument=check.instrument,
                        criterion=key.rubric_dim,
                        against="model",
                        against_id=reviewer.model,
                        against_served_model=reviewer.served_model,
                        against_effort=reviewer.effort,
                        read_id=read.id,
                        result="insufficient",
                        reason="identity_unrecorded",
                        coordinate="served_model",
                        reads=criterion.n,
                    )
                )
                continue
            judge_family = family(key.judge_model)
            checks.append(
                check.model_copy(
                    update={
                        **who,
                        "same_family_as": sorted(
                            arm
                            for arm, model in arms.items()
                            if model is not None and mine is not None and family(model) == mine
                        ),
                        "same_family_as_judge": None if mine is None or judge_family is None else mine == judge_family,
                    }
                )
            )
    return checks + _reviewers_against_people(
        closed, people, by_reviewer, reviewers, configuration_of, unrecorded_runs=dict(unrecorded_runs or {})
    )


def _reviewers_against_people(
    closed: BlindClose,
    people: Sequence[CalibrationRating],
    by_reviewer: Mapping[_Reviewer, Sequence[CalibrationRating]],
    reviewers: Sequence[_Reviewer],
    configuration_of: Mapping[tuple[str, str, str], _Reviewer],
    *,
    unrecorded_runs: Mapping[str, Coordinate],
) -> list[Check]:
    """Each reviewer, checked against people on each criterion a person answered (:func:`reviewer_check`).

    Kappa pairs every output both read, by its label key, singles and other ratings included. The lean reads the pair
    items of the read both read on both arms: the reviewer's candidate-less-control difference less the people's mean.
    A reviewer whose served model was not recorded reads neither: its check counts the reads it leaves out.
    """
    read = closed.read
    by_people: dict[tuple[str, str, str], list[tuple[str, int]]] = {}
    for rating in people:
        if rating.label_key is not None:
            by_people.setdefault((rating.rubric_dim, *rating.label_key), []).append((rating.rater, rating.score))
    candidate = {arm.key for arm in read.arms if arm.role == "candidate"}
    order = {item.item: position for position, item in enumerate(read.items)}
    items = {item.item: item for item in read.items}
    person_differences: dict[tuple[str, str], list[int]] = {}
    model_differences: dict[tuple[_Reviewer, str, str], int] = {}
    for answer in read.answers:
        item = items.get(answer.item)
        if answer.skipped or item is None or len(item.outputs) != 2:
            continue
        arm_of = {output.slot: output.arm for output in item.outputs}
        given: dict[str, dict[bool, int]] = {}
        for verdict in answer.verdicts:
            given.setdefault(verdict.dim, {})[arm_of[verdict.slot] in candidate] = verdict.score
        for name, both in given.items():
            if len(both) != 2:
                continue
            difference = both[True] - both[False]
            if answer.rater_kind == "person":
                person_differences.setdefault((answer.item, name), []).append(difference)
                continue
            # The configuration the rating of this item and criterion is read under: the last answer's, as the ratings.
            result_id = item.outputs[0].result_id
            if configuration_of.get((answer.rater, result_id, name)) == (
                reviewer := _Reviewer(answer.rater, answer.reviewer_served_model, answer.reviewer_effort)
            ):
                model_differences[(reviewer, answer.item, name)] = difference
    random_people = all(rating.id in closed.random for rating in people)
    checks: list[Check] = []
    for reviewer in reviewers:
        own = by_reviewer[reviewer]
        for dim in read.criteria:
            if not any(rating.rubric_dim == dim.name for rating in people):
                continue
            reads = sum(1 for rating in own if rating.rubric_dim == dim.name)
            if reviewer.served_model is None:
                checks.append(
                    ReviewerCheck(
                        kind="reviewer",
                        instrument=reviewer.model,
                        effort=reviewer.effort,
                        criterion=dim.name,
                        read_id=read.id,
                        result="insufficient",
                        reason="identity_unrecorded",
                        coordinate="served_model",
                        reads=reads,
                    )
                )
                continue
            kappa_pairs = [
                (rating.score, score, person, "|".join(rating.label_key))
                for rating in own
                if rating.rubric_dim == dim.name and rating.label_key is not None
                for person, score in by_people.get((rating.rubric_dim, *rating.label_key), ())
            ]
            leans: list[float] = []
            left_out: list[Coordinate] = []
            for item_id, criterion in sorted(person_differences, key=lambda key: order[key[0]]):
                mine = model_differences.get((reviewer, item_id, criterion))
                if criterion != dim.name or mine is None:
                    continue
                unrecorded = [unrecorded_runs[o.run_id] for o in items[item_id].outputs if o.run_id in unrecorded_runs]
                if unrecorded:
                    left_out.append(unrecorded[0])
                    continue
                theirs = person_differences[(item_id, criterion)]
                leans.append(mine - math.fsum(theirs) / len(theirs))
            checks.append(
                reviewer_check(
                    reviewer.model,
                    dim.name,
                    dim.scale,
                    kappa_pairs=kappa_pairs,
                    leans=leans,
                    left_out=left_out,
                    served_model=reviewer.served_model,
                    effort=reviewer.effort,
                    reads=reads,
                    read_id=read.id,
                    sample="random" if random_people else "selected",
                )
            )
    return checks


class Corrected(_Shape):
    """A check's result on one judged contrast: the contrast as what the judge was checked against would read it."""

    against: Literal["people", "model"] = Field(description="Whose reading corrects the judge's.")
    against_id: str | None = Field(default=None, description="The model, by id.")
    result: CriterionState | None = Field(default=None, description="Whether the corrected reading confirms.")
    state: StateOf | None = Field(default=None, description="Its state where it is not counted.")
    estimate: float | None = Field(default=None, description="The corrected difference.")
    interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    level: str | None = Field(default=None, description="The interval's level.")
    lean: float | None = Field(default=None, description="The judge's lean the correction removes.")
    lean_interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    labelled_cases: int | None = Field(default=None, description="Cases both the judge and the other reader read.")
    identical_cases: int | None = Field(default=None, description="Of those, the cases they read alike.")
    read_id: str | None = Field(default=None, description="The blind read the correction rests on.")


class Rescore(_Shape):
    """A judge checked against itself on one arm: how much a second scoring of the same reply moves.

    The per-arm figures of a re-score. The judge's check against itself (a :class:`JudgeCheck` against ``itself``) is
    the per-judge agreement criterion its repeats decide; this shape is what the repeats show on one arm beside it.
    """

    state: StateOf | None = Field(default=None, description="Its state where it is not counted.")
    variance: float | None = Field(default=None, description="The variance between scorings of one reply.")
    variance_interval: tuple[float, float] | None = Field(
        default=None, exclude_if=lambda value: value is None, description="The interval on it."
    )
    repeated_results: int | None = Field(default=None, description="Results scored more than once.")
    repeated_cases: int | None = Field(default=None, description="Distinct cases behind them.")
    fail_given_pass: float | None = Field(default=None, description="How often a passed reply fails on a re-score.")
    fail_given_pass_interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    ceiling_low: float | None = Field(default=None, description="The low end of the ceiling the noise puts on pass^k.")
    ceiling_high: float | None = Field(default=None, description="Its high end.")
    k: int | None = Field(default=None, description="The k the ceiling is read at: the most scores one case holds.")
    ceiling_state: StateOf | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description="Where the ceiling is read for the dimension and could not be: not available, why.",
    )


class Looks(_Shape):
    """How often a reading's test was looked at, against how often it planned to be."""

    planned: int | None = Field(default=None, description="The looks declared; none where none were.")
    taken: int = Field(ge=1, description="The looks taken.")
