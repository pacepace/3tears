"""Every instrument is checked: the check record, what decides one, and which check speaks for a judge.

A check names an instrument, what it was checked against (people, a model, itself, or identical arms), on what sample,
with one result (a :data:`~threetears.evals.kernel.evidence_tiers.CriterionState`, with a reason from
:mod:`~threetears.evals.schema.reading_states` where it is not ``met``). It replaces the evidence tier as what a reader
and a decision are told about an instrument: the tier is still stored, decided from the people check and the itself
check exactly as before (:func:`tier_checks` reads those two off it), and nothing here prints or reads a tier value.

- :func:`tier_checks` is a judge's check against people and against itself, from the two criteria its stored tier was
  decided by; :func:`judge_checks` lists them for the judges that scored a dimension.
- :func:`reviewer_check` is a model that answered a blind read, checked against people: by its kappa, and by its lean
  toward an arm over the pair items both read (the model's difference between the arms less the person's). Kappa is
  blind to a lean the model and a judge share, so the lean is the check that can see it. ``met`` is an equivalence
  reading: the lean's interval inside ``±δ`` over at least :data:`REVIEWER_MET_MIN_PAIR_ITEMS` items.
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
from typing import TYPE_CHECKING, Any, Final, Literal, Self, get_args

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
from threetears.evals.analysis.stats import INTERVAL_LEVEL, t_critical_two_sided
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
    from threetears.evals.schema.models import CalibrationRating, RaterKind

__all__ = [
    "REVIEWER_LEAN_MIN_PAIR_ITEMS",
    "REVIEWER_MET_MIN_PAIR_ITEMS",
    "Check",
    "CheckKind",
    "Corrected",
    "Looks",
    "Rescore",
    "blind_read_checks",
    "judge_checks",
    "lean_interval",
    "reviewer_check",
    "speaking_check",
    "tier_checks",
]

#: What a check checks: a judge, a model that answered a blind read (a reviewer), or the rig.
CheckKind = Literal["judge", "reviewer", "rig"]

#: The fewest pair items a reviewer's lean is read as ``met`` over. An equivalence reading needs this many before its
#: interval can sit inside ``±δ`` at all often: at 40 it rarely does, and inside ``±0.10`` with 80% chance takes about
#: 105 to 260 on pass/fail. ``tests/test_simulated_checks.py`` holds it.
REVIEWER_MET_MIN_PAIR_ITEMS: Final = 40

#: The fewest pair items a reviewer's lean demotes it from. A floor in the safe direction: a reviewer shown to lean only
#: sets an improvement aside.
REVIEWER_LEAN_MIN_PAIR_ITEMS: Final = 10

#: The share of a scale's width a lean is held inside where its reading declares no margin.
_LEAN_SHARE_OF_WIDTH: Final = 0.10

#: The results that decide a check: a criterion read at its floor, either way.
_DECIDED: Final = frozenset({"met", "not_met"})


class _Shape(BaseModel):
    """Immutable, and refusing a key it does not declare."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Check(_Shape):
    """One check of one instrument: what it was checked against, on what sample, with one result.

    ``not_met`` carries the reason it sets a reading aside for (``instrument_below_bar``, ``instrument_leans``,
    ``people_disagree``, ``identical_arms_separated``). ``insufficient`` carries why there is too little to decide
    (``too_few``, ``not_recorded``, ``no_variation``, ``identity_unrecorded`` with its coordinate, ...), or ``state``
    where the check could not be made at all. ``met`` and ``undecided`` carry none.
    """

    kind: CheckKind = Field(
        description="What was checked: a judge, a model that answered a blind read, or the rig. A check that names "
        "none is the rig's when it is against identical arms, else a judge's."
    )
    instrument: str = Field(
        description="Which one: a judge by its served model, config, temperature and sent settings "
        "(`judge_instrument`), a reviewer by its model id, the rig by its launch group."
    )
    criterion: str | None = Field(default=None, description="The criterion the check decides, where it names one.")
    against: Literal["people", "model", "itself", "identical_arms"] = Field(description="What it was checked against.")
    against_id: str | None = Field(default=None, description="The model it was checked against, by id.")
    same_family_as: list[str] = Field(
        default_factory=list,
        description="The arms whose candidate model is of the family of the model checked against.",
    )
    same_family_as_judge: bool | None = Field(
        default=None,
        description="Whether the model checked against is of the judge's own family; None where either family is not "
        "recorded, or no model was checked against.",
    )
    sample: SampleKind | None = Field(
        default=None,
        description="Whether the items it was read on were drawn at random, chosen, or no sample stands behind it.",
    )
    sample_results: int | None = Field(default=None, description="Results read.")
    sample_cases: int | None = Field(default=None, description="Distinct cases behind them.")
    pair_items: int | None = Field(
        default=None, description="A reviewer's check: the pair items its lean was read over, both arms read by both."
    )
    left_out: int | None = Field(
        default=None,
        description="A reviewer's check: the pair items left out of its lean because a run of theirs has an unrecorded "
        "coordinate, which matches nothing.",
    )
    skipped: int | None = Field(default=None, description="Items a reader skipped.")
    read_id: str | None = Field(default=None, description="The blind read the sample is, by id.")
    result: CriterionState = Field(description="met, not_met, undecided or insufficient.")
    reason: Reason | None = Field(default=None, description="Why the result is not met, where a shared reason says it.")
    coordinate: Coordinate | None = Field(
        default=None, description="The coordinate an unrecorded identity names, where that is the reason."
    )
    state: StateOf | None = Field(
        default=None,
        description="Where the check could not be made at all (both sides replayed a recording): not available, why.",
    )
    kappa: float | None = Field(default=None, description="Agreement beyond chance.")
    kappa_interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    lean: float | None = Field(
        default=None, description="The instrument's lean toward the candidate arm: its difference less the other's."
    )
    lean_interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    half_width: float | None = Field(default=None, description="The half-width the check was read at.")
    reads: int | None = Field(default=None, description="Reads behind the figures.")
    delta: float | None = Field(default=None, description="The difference between identical arms.")
    interval: tuple[float, float] | None = Field(default=None, description="The interval on that difference.")
    p_adjusted: float | None = Field(default=None, description="Its adjusted p.")
    reading: str | None = Field(default=None, description="The reading a check of the rig covers.")
    covers_arms: list[str] = Field(default_factory=list, description="The arms a check of the rig covers.")

    @model_validator(mode="before")
    @classmethod
    def _kind_from_what_it_was_checked_against(cls, data: Any) -> Any:
        """A check naming no kind is the rig's against identical arms, else a judge's."""
        if isinstance(data, dict) and "kind" not in data:
            return {**data, "kind": "rig" if data.get("against") == "identical_arms" else "judge"}
        return data

    @model_validator(mode="after")
    def _result_and_reason_belong_together(self) -> Self:
        """Refuse a reason its result does not carry, a kind its ``against`` does not fit, and a coordinate apart from
        the reason that names one.

        :raises ValueError: Naming which.
        """
        if (self.kind == "rig") != (self.against == "identical_arms"):
            raise ValueError("the rig, and only the rig, is checked against identical arms")
        if self.kind == "reviewer" and self.against != "people":
            raise ValueError("a model that answered a blind read is checked against people")
        if self.result in ("met", "undecided") and self.reason is not None:
            raise ValueError(f"a {self.result} check carries no reason, and {self.reason!r} was given")
        if self.result == "not_met" and self.reason not in get_args(SetAsideReason):
            raise ValueError(f"a not_met check carries the reason it sets a reading aside for, not {self.reason!r}")
        if self.result == "insufficient":
            if self.reason is None and self.state is None:
                raise ValueError("an insufficient check says why: a reason, or the state of a check not made")
            if self.reason is not None and self.reason not in get_args(NotAvailableReason):
                raise ValueError(f"an insufficient check carries why there is too little, not {self.reason!r}")
        if (self.reason == "identity_unrecorded") != (self.coordinate is not None):
            raise ValueError("an unrecorded identity names its coordinate, and only it names one")
        return self


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
    against_id: str | None = None,
) -> Check:
    """One agreement criterion of a judge, as the check it is."""
    return Check(
        kind="judge",
        instrument=instrument,
        criterion=dim,
        against=against,
        against_id=against_id,
        sample="none" if criterion.n == 0 else sample,
        sample_results=criterion.results,
        sample_cases=sample_cases,
        read_id=read_id,
        result=criterion.state,
        reason=_criterion_reason(criterion),
        kappa=criterion.agreement,
        kappa_interval=criterion.interval,
    )


def tier_checks(tier: JudgeEvidenceTier) -> list[Check]:
    """A judge's check against people and its check against itself, from the two criteria its stored tier was decided
    by (``calibration`` and ``separation``). The tier is those two results as
    :func:`~threetears.evals.kernel.evidence_tiers.tier_of` reads them, and nothing else.

    :param tier: The judge's stored tier on one dimension.
    :ptype tier: JudgeEvidenceTier
    :return: The people check, then the itself check.
    :rtype: list[Check]
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


def lean_interval(leans: Sequence[float]) -> tuple[float | None, tuple[float, float] | None]:
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


def _reviewer_result(
    kappa: TierCriterion, pair_items: int, interval: tuple[float, float] | None, delta: float
) -> tuple[CriterionState, Reason | None]:
    """The reviewer's own check, its rows in the order they bind: below the bar, then a lean, then nothing to read a
    lean over, then the equivalence reading, else undecided."""
    if kappa.state == "not_met":
        return "not_met", "instrument_below_bar"
    if interval is not None and pair_items >= REVIEWER_LEAN_MIN_PAIR_ITEMS and (interval[0] > 0 or interval[1] < 0):
        return "not_met", "instrument_leans"
    if pair_items == 0:
        return "insufficient", "not_recorded"
    if (
        pair_items >= REVIEWER_MET_MIN_PAIR_ITEMS
        and interval is not None
        and -delta < interval[0]
        and interval[1] < delta
        and kappa.state == "met"
    ):
        return "met", None
    return "undecided", None


def reviewer_check(
    reviewer: str,
    dim: str,
    scale: RubricScale,
    *,
    kappa_pairs: Sequence[tuple[int, int, str, str]],
    leans: Sequence[float],
    left_out: int = 0,
    margin: float | None = None,
    reads: int | None = None,
    read_id: str | None = None,
    sample: SampleKind | None = None,
) -> Check:
    """A model that answered a blind read, checked against people on one criterion: its kappa, and its lean.

    ``met`` is an equivalence reading: at least :data:`REVIEWER_MET_MIN_PAIR_ITEMS` pair items, the lean's interval
    inside ``±δ``, and the kappa criterion ``met`` by its rule. ``not_met`` is the kappa shown below its bar
    (``instrument_below_bar``: the judge reads as not checked against this model on the criterion), or the lean's
    interval excluding zero over :data:`REVIEWER_LEAN_MIN_PAIR_ITEMS` items or more (``instrument_leans``: an
    improvement read by a judge checked only against this model is set aside for the arm it leans toward). With no
    pair item it is ``insufficient``, ``not_recorded``. Anything else is ``undecided``, carrying the half-width: a lean
    of either sign is not ruled out below it, and "cannot tell apart" is never "no lean".

    :param reviewer: The model's id.
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
    :param left_out: Pair items left out of ``leans`` because a run of theirs has an unrecorded coordinate.
    :ptype left_out: int
    :param margin: The reading's declared materiality margin, which the lean is held inside; None holds it inside a
        tenth of the scale's width.
    :ptype margin: float | None
    :param reads: How many reads of the criterion the model made over the population.
    :ptype reads: int | None
    :param read_id: The blind read, by id.
    :ptype read_id: str | None
    :param sample: How the people's items were sampled.
    :ptype sample: SampleKind | None
    :return: The check.
    :rtype: Check
    """
    low, high = SCALES[scale].scores
    delta = margin if margin is not None and margin > 0 else _LEAN_SHARE_OF_WIDTH * (high - low)
    agreement = instrument_agreement(scale, kappa_pairs)
    kappa = calibration_criterion(agreement.n, agreement.results, agreement.agreement, agreement.interval)
    lean, interval = lean_interval(leans)
    result, reason = _reviewer_result(kappa, len(leans), interval, delta)
    return Check(
        kind="reviewer",
        instrument=reviewer,
        criterion=dim,
        against="people",
        sample=sample,
        sample_results=agreement.results,
        pair_items=len(leans),
        left_out=left_out,
        read_id=read_id,
        result=result,
        reason=reason,
        kappa=agreement.agreement,
        kappa_interval=agreement.interval,
        lean=lean,
        lean_interval=interval,
        half_width=None if interval is None else (interval[1] - interval[0]) / 2,
        reads=reads,
    )


# --- which check speaks ---------------------------------------------------------------------------------------------


def speaking_check(checks: Iterable[Check], instrument: str, criterion: str) -> Check | None:
    """Which of a judge's checks speaks for it on a criterion.

    The check against people, once it is decided (``met`` or ``not_met``, which its floor of distinct results gates).
    Else a decided check against a model, unless that model's own check against people shows its kappa below the bar
    (``instrument_below_bar``); a model shown to lean still speaks, and its lean sets an improvement aside where the
    decision is read. Else none: the judge is not checked on the criterion.

    :param checks: Every check in hand: the judge's, and the reviewers' own.
    :ptype checks: Iterable[Check]
    :param instrument: The judge (:func:`~threetears.evals.analysis.agreement.judge_instrument`).
    :ptype instrument: str
    :param criterion: The criterion.
    :ptype criterion: str
    :return: The check that speaks, or None.
    :rtype: Check | None
    """
    held = list(checks)
    mine = [c for c in held if c.kind == "judge" and c.instrument == instrument and c.criterion == criterion]
    people = next((c for c in mine if c.against == "people" and c.result in _DECIDED), None)
    if people is not None:
        return people
    own = {(c.instrument, c.criterion): c for c in held if c.kind == "reviewer"}
    for check in sorted(
        (c for c in mine if c.against == "model" and c.result in _DECIDED), key=lambda c: c.against_id or ""
    ):
        reviewer = own.get((check.against_id or "", criterion))
        if reviewer is not None and reviewer.reason == "instrument_below_bar":
            continue
        return check
    return None


# --- every check one blind read yields ------------------------------------------------------------------------------


def _judge_against(
    ratings: Sequence[CalibrationRating],
    results: Sequence[ResultView],
    *,
    against: Literal["people", "model"],
    read_id: str,
    sample_of: Callable[[str], SampleKind | None],
    rated_by: RaterKind = "person",
) -> list[tuple[JudgeKey, TierCriterion, Check]]:
    """Each judge's agreement with ``ratings``, per criterion, as the check it is beside its judge and criterion."""
    if not ratings:
        return []
    agreement = judge_agreement(ratings, results, rated_by=rated_by)
    scores = person_scores_by_result(ratings, results, rated_by=rated_by)
    by_id = {result.id: result for result in results}
    read: list[tuple[JudgeKey, TierCriterion, Check]] = []
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
    unrecorded_runs: Collection[str] = (),
    margins: Mapping[str, float] | None = None,
) -> list[Check]:
    """Every check a closed blind read yields, people's and each model's apart and never pooled.

    - **The judge against people**, per judge and criterion, over every person's rating that pairs (the read's own
      and ``other_ratings``): ``random`` where every one came from a rater whose answers are the top of the order.
    - **The judge against each model that answered**, per judge and criterion. A model that is the judge's own (its
      id, or a model that served one of its answers, is one the judge was asked as or served by) checks the judge
      against itself, held to the bar a judge's agreement with its repeats is. Where which model served the judge's
      scores or the model's answers was not recorded, it cannot be shown to be another model: the check is
      ``insufficient``, ``identity_unrecorded`` on the served model, with no figure. The family the model shares with
      an arm's candidate or with the judge is a property of the check, never a caveat of its own.
    - **Each model against people**, per criterion a person answered (:func:`reviewer_check`). Pair items either of
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
    :param unrecorded_runs: Runs with an unrecorded coordinate, whose pair items no lean reads.
    :ptype unrecorded_runs: Collection[str]
    :param margins: A criterion's declared materiality margin, by criterion, which a model's lean is held inside.
    :ptype margins: Mapping[str, float] | None
    :return: The judge's checks against people, then against each model in turn, then each model's own.
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

    checks = [
        check
        for _, _, check in _judge_against(people, results, against="people", read_id=read.id, sample_of=people_sample)
    ]
    reviewers = sorted(set(closed.reviewed.values()))
    served: dict[str, set[str]] = {}
    unrecorded: set[str] = set()
    for answer in read.answers:
        if answer.rater_kind != "reference" or answer.skipped:
            continue
        if answer.reviewer_served_model is None:
            unrecorded.add(answer.rater)
        else:
            served.setdefault(answer.rater, set()).add(answer.reviewer_served_model)
    samples = {(s.rater, s.rater_kind): s.sample for s in (read.report.samples if read.report else [])}
    family = family_of or (lambda model: None)
    arms = dict(candidate_models or {})
    for reviewer in reviewers:
        own = [rating for rating in closed.ratings if closed.reviewed.get(rating.id) == reviewer]
        mine = family(reviewer)
        reviewer_sample = samples.get((reviewer, "reference"))
        for key, criterion, check in _judge_against(
            own,
            results,
            against="model",
            read_id=read.id,
            sample_of=lambda dim, sampled=reviewer_sample: sampled,
            rated_by="agent",
        ):
            names = set(judge_models) | ({key.judge_model} if key.judge_model is not None else set())
            if ({reviewer} | served.get(reviewer, set())) & names:
                repeat = separation_criterion(criterion.n, criterion.results, criterion.agreement, criterion.interval)
                checks.append(
                    _criterion_check(
                        repeat,
                        instrument=check.instrument,
                        dim=key.rubric_dim,
                        against="itself",
                        against_id=reviewer,
                        sample=check.sample,
                        sample_cases=check.sample_cases,
                        read_id=read.id,
                    )
                )
                continue
            if key.judge_model is None or reviewer in unrecorded:
                checks.append(
                    Check(
                        kind="judge",
                        instrument=check.instrument,
                        criterion=key.rubric_dim,
                        against="model",
                        against_id=reviewer,
                        read_id=read.id,
                        result="insufficient",
                        reason="identity_unrecorded",
                        coordinate="served_model",
                    )
                )
                continue
            judge_family = family(key.judge_model)
            checks.append(
                check.model_copy(
                    update={
                        "against_id": reviewer,
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
        closed, people, reviewers, unrecorded_runs=frozenset(unrecorded_runs), margins=margins or {}
    )


def _reviewers_against_people(
    closed: BlindClose,
    people: Sequence[CalibrationRating],
    reviewers: Sequence[str],
    *,
    unrecorded_runs: frozenset[str],
    margins: Mapping[str, float],
) -> list[Check]:
    """Each model that answered, checked against people on each criterion a person answered (:func:`reviewer_check`).

    Kappa pairs every output both read, by its label key, singles and other ratings included. The lean reads the pair
    items of the read both read on both arms: the model's candidate-less-control difference less the people's mean.
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
    model_differences: dict[tuple[str, str, str], int] = {}
    for answer in read.answers:
        item = items.get(answer.item)
        if answer.skipped or item is None or len(item.outputs) != 2:
            continue
        arm_of = {output.slot: output.arm for output in item.outputs}
        given: dict[str, dict[bool, int]] = {}
        for verdict in answer.verdicts:
            given.setdefault(verdict.dim, {})[arm_of[verdict.slot] in candidate] = verdict.score
        for dim, both in given.items():
            if len(both) != 2:
                continue
            difference = both[True] - both[False]
            if answer.rater_kind == "person":
                person_differences.setdefault((answer.item, dim), []).append(difference)
            else:
                model_differences[(answer.rater, answer.item, dim)] = difference
    random_people = all(rating.id in closed.random for rating in people)
    checks: list[Check] = []
    for reviewer in reviewers:
        own = [rating for rating in closed.ratings if closed.reviewed.get(rating.id) == reviewer]
        for dim in read.criteria:
            if not any(rating.rubric_dim == dim.name for rating in people):
                continue
            kappa_pairs = [
                (rating.score, score, person, "|".join(rating.label_key))
                for rating in own
                if rating.rubric_dim == dim.name and rating.label_key is not None
                for person, score in by_people.get((rating.rubric_dim, *rating.label_key), ())
            ]
            leans: list[float] = []
            left_out = 0
            for item_id, criterion in sorted(person_differences, key=lambda key: order[key[0]]):
                mine = model_differences.get((reviewer, item_id, criterion))
                if criterion != dim.name or mine is None:
                    continue
                if any(output.run_id in unrecorded_runs for output in items[item_id].outputs):
                    left_out += 1
                    continue
                theirs = person_differences[(item_id, criterion)]
                leans.append(mine - math.fsum(theirs) / len(theirs))
            checks.append(
                reviewer_check(
                    reviewer,
                    dim.name,
                    dim.scale,
                    kappa_pairs=kappa_pairs,
                    leans=leans,
                    left_out=left_out,
                    margin=margins.get(dim.name),
                    reads=sum(1 for rating in own if rating.rubric_dim == dim.name),
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
    """A judge checked against itself on one arm: how much a second scoring of the same reply moves."""

    state: StateOf | None = Field(default=None, description="Its state where it is not counted.")
    variance: float | None = Field(default=None, description="The variance between scorings of one reply.")
    repeated_results: int | None = Field(default=None, description="Results scored more than once.")
    repeated_cases: int | None = Field(default=None, description="Distinct cases behind them.")
    fail_given_pass: float | None = Field(default=None, description="How often a passed reply fails on a re-score.")
    fail_given_pass_interval: tuple[float, float] | None = Field(default=None, description="The interval on it.")
    ceiling_low: float | None = Field(default=None, description="The low end of the ceiling the noise puts on pass^k.")
    ceiling_high: float | None = Field(default=None, description="Its high end.")


class Looks(_Shape):
    """How often a reading's test was looked at, against how often it planned to be."""

    planned: int | None = Field(default=None, description="The looks declared; none where none were.")
    taken: int = Field(ge=1, description="The looks taken.")
