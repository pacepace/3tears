"""The compact per-result reading: what the engine's readers read of a stored result, and nothing else.

A stored :class:`~threetears.evals.schema.models.EvalResult` carries its judge's prose, its kind's payload and its
world's events beside the figures every reader folds. A reader that folds a whole scope holds every result it read,
so a result's bytes are what bound how many results one read can answer. :class:`ResultReading` is a result as the
readers read it: exactly the attributes they access, under the same names, copied from a hydrated ``EvalResult``
(:meth:`ResultReading.of`) and never recomputed, so a fold over readings gives the fold over results to the last bit.

**What it carries** is measured, not guessed: an access recorder runs every reader family over production-shaped
campaigns and pins each attribute path it reads (``packages/evals/tests/test_result_reading_fields.py``). The
judge's prose is left out: a result's ``judge_reasoning`` and each score's ``reasoning`` are read by no rule, and the
judge's "cannot tell" notes are carried as the dimensions they name (their text is read by no rule either). A record
list is carried as slim items of the fields read (:class:`ScoreReading`, :class:`UsageReading`, ...), but for the
background deliveries and the settings a request was sent with, which readers copy (``model_copy``) or compare whole
(``==``, ``model_dump``) and so are carried as the stored models themselves.

**What reads it** is :class:`ResultView`, the structural protocol an ``EvalResult`` and a ``ResultReading`` both
satisfy. The rules a reader folds with take a view, so the same rule reads either; a rule reading outside the view
fails loudly (an ``AttributeError`` at run time, a type error under mypy), never quietly.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self, cast

from threetears.evals.schema.models import SCALES, LabelKey

if TYPE_CHECKING:
    from threetears.evals.schema.models import (
        AsyncDelivery,
        CellTermination,
        ClientRequestSettings,
        ConversationStopCause,
        EvalResult,
        GoalStateOutcome,
        JudgeCutShort,
        JudgeRepeat,
        JudgeRescore,
        JudgeTemperature,
        LatencyMetrics,
        PreconditionOutcome,
        RepeatedScore,
        RoleUsage,
        RubricAxis,
        RubricScale,
        RubricScore,
        SecondJudge,
        SecondJudgeScore,
        SecondJudging,
    )

__all__ = [
    "GoalReading",
    "GoalView",
    "GroupsRead",
    "LatencyReading",
    "LatencyView",
    "PreconditionReading",
    "PreconditionView",
    "ReadPath",
    "ReadingsRead",
    "RepeatReading",
    "RepeatView",
    "RepeatedScoreReading",
    "RepeatedScoreView",
    "RescoreReading",
    "RescoreView",
    "ResultReading",
    "ResultView",
    "ScoreReading",
    "ScoreView",
    "SecondJudgeReading",
    "SecondJudgeView",
    "SecondJudgingReading",
    "SecondJudgingView",
    "SecondScoreReading",
    "SecondScoreView",
    "SharedValues",
    "UsageReading",
    "UsageView",
]


class ScoreView(Protocol):
    """A judge's score as the readers read it: a :class:`~threetears.evals.schema.models.RubricScore` or a
    :class:`ScoreReading`."""

    @property
    def dim(self) -> str: ...
    @property
    def scale(self) -> RubricScale: ...
    @property
    def axis(self) -> RubricAxis: ...
    @property
    def score(self) -> int: ...
    @property
    def served_model(self) -> str | None: ...
    @property
    def judge_temperature(self) -> JudgeTemperature | None: ...
    @property
    def judge_sent_settings(self) -> ClientRequestSettings | None: ...
    @property
    def output_fingerprint(self) -> str | None: ...
    @property
    def criterion_fingerprint(self) -> str | None: ...
    @property
    def label_key(self) -> LabelKey | None: ...
    @property
    def normalized(self) -> float: ...
    def clears(self, rubric_threshold: int) -> bool: ...


class GoalView(Protocol):
    """A goal-state check's outcome as the readers read it."""

    @property
    def expression(self) -> str: ...
    @property
    def passed(self) -> bool: ...
    @property
    def detail(self) -> str: ...


class PreconditionView(Protocol):
    """A precondition's outcome as the readers read it."""

    @property
    def expression(self) -> str: ...
    @property
    def presumes(self) -> str: ...
    @property
    def detail(self) -> str: ...


class UsageView(Protocol):
    """One usage row as the readers read it: a :class:`~threetears.evals.schema.models.RoleUsage` or a
    :class:`UsageReading`."""

    @property
    def role(self) -> str: ...
    @property
    def model(self) -> str | None: ...
    @property
    def served_model(self) -> str | None: ...
    @property
    def prompt_tokens(self) -> int | None: ...
    @property
    def completion_tokens(self) -> int | None: ...
    @property
    def reasoning_tokens(self) -> int | None: ...
    @property
    def cost_usd(self) -> float | None: ...
    @property
    def price_source(self) -> str | None: ...
    @property
    def call_count(self) -> int | None: ...
    @property
    def provider(self) -> str | None: ...
    @property
    def provider_unit(self) -> str | None: ...
    @property
    def provider_units(self) -> int | None: ...
    @property
    def actor_id(self) -> str | None: ...
    @property
    def purpose(self) -> str | None: ...
    @property
    def sent_settings(self) -> Sequence[ClientRequestSettings] | None: ...
    @property
    def unanswered_retried(self) -> int | None: ...
    @property
    def unanswered_lost(self) -> int | None: ...
    @property
    def unanswered_reasons(self) -> Mapping[str, int] | None: ...


class LatencyView(Protocol):
    """A result's latency as the readers read it."""

    @property
    def total_ms(self) -> float | None: ...
    @property
    def llm_ms(self) -> float | None: ...
    @property
    def tool_ms(self) -> float | None: ...
    @property
    def async_wait_ms(self) -> float | None: ...
    @property
    def judge_ms(self) -> float | None: ...


class RescoreView(Protocol):
    """A re-judge's record as the readers read it: its spend."""

    @property
    def cost_usd(self) -> float | None: ...


class RepeatedScoreView(Protocol):
    """One score of a repeat of the judge, beside the first answer it repeats."""

    @property
    def dim(self) -> str: ...
    @property
    def scale(self) -> RubricScale: ...
    @property
    def first_score(self) -> int: ...
    @property
    def first_served_model(self) -> str | None: ...
    @property
    def first_judge_config_id(self) -> str | None: ...
    @property
    def first_judge_temperature(self) -> JudgeTemperature | None: ...
    @property
    def first_judge_sent_settings(self) -> ClientRequestSettings | None: ...
    @property
    def repeat(self) -> ScoreView | None: ...
    @property
    def cannot_tell(self) -> str | None: ...


class RepeatView(Protocol):
    """A repeat of the judge's scores."""

    @property
    def judge_config_ids(self) -> Mapping[str, str]: ...
    @property
    def scores(self) -> Sequence[RepeatedScoreView]: ...


class SecondScoreView(Protocol):
    """One score of a second judge's pass, beside the first judge's answer."""

    @property
    def dim(self) -> str: ...
    @property
    def scale(self) -> RubricScale: ...
    @property
    def first_score(self) -> int: ...
    @property
    def first_served_model(self) -> str | None: ...
    @property
    def first_judge_config_id(self) -> str | None: ...
    @property
    def first_judge_temperature(self) -> JudgeTemperature | None: ...
    @property
    def first_judge_sent_settings(self) -> ClientRequestSettings | None: ...
    @property
    def second(self) -> ScoreView | None: ...
    @property
    def cannot_tell(self) -> str | None: ...


class SecondJudgeView(Protocol):
    """The judge a second pass was asked of."""

    @property
    def model(self) -> str: ...
    @property
    def config_ids(self) -> Mapping[str, str] | None: ...
    @property
    def temperature(self) -> float | None: ...


class SecondJudgingView(Protocol):
    """A second judge's pass over a result."""

    @property
    def pass_id(self) -> str: ...
    @property
    def judge(self) -> SecondJudgeView: ...
    @property
    def judge_config_ids(self) -> Mapping[str, str]: ...
    @property
    def scores(self) -> Sequence[SecondScoreView]: ...


class ResultView(Protocol):
    """A stored result as the readers read it: an :class:`~threetears.evals.schema.models.EvalResult` or its
    :class:`ResultReading`. Each attribute means what the ``EvalResult`` field of its name means; see there."""

    @property
    def id(self) -> str: ...
    @property
    def doc_type(self) -> str: ...
    @property
    def schema_version(self) -> int: ...
    @property
    def schema_minor(self) -> int: ...
    @property
    def scope_id(self) -> str: ...
    @property
    def eval_run_id(self) -> str: ...
    @property
    def test_case_id(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def k_iteration(self) -> int: ...
    @property
    def lease_epoch(self) -> int: ...
    @property
    def superseded_spend_usd(self) -> float | None: ...
    @property
    def metered_calls(self) -> int | None: ...
    @property
    def metered_calls_refused(self) -> int | None: ...
    @property
    def candidate_kind(self) -> str | None: ...
    @property
    def candidate_instance_id(self) -> str | None: ...
    @property
    def subject_id(self) -> str | None: ...
    @property
    def has_trace(self) -> bool: ...
    @property
    def precondition_outcomes(self) -> Sequence[PreconditionView]: ...
    @property
    def goal_state_outcomes(self) -> Sequence[GoalView]: ...
    @property
    def rubric_scores(self) -> Sequence[ScoreView]: ...
    @property
    def judge_model(self) -> str | None: ...
    @property
    def transcript_score(self) -> ScoreView | None: ...
    @property
    def outcome_score(self) -> ScoreView | None: ...
    @property
    def judge_config_ids(self) -> Mapping[str, str]: ...
    @property
    def judge_rescores(self) -> Sequence[RescoreView]: ...
    @property
    def judge_repeats(self) -> Sequence[RepeatView]: ...
    @property
    def judge_seconds(self) -> Sequence[SecondJudgingView]: ...
    @property
    def cost_usd(self) -> float | None: ...
    @property
    def cost_roles(self) -> Sequence[str]: ...
    @property
    def usage(self) -> Sequence[UsageView]: ...
    @property
    def latency(self) -> LatencyView | None: ...
    @property
    def async_deliveries(self) -> Sequence[AsyncDelivery] | None: ...
    @property
    def covariates(self) -> Mapping[str, str | float]: ...
    @property
    def phase_timings(self) -> Mapping[str, float]: ...
    @property
    def host_measures(self) -> Mapping[str, bool | float | str]: ...
    @property
    def variant_key(self) -> str: ...
    @property
    def identity_version(self) -> int: ...
    @property
    def termination(self) -> CellTermination: ...
    @property
    def stop_cause(self) -> ConversationStopCause | None: ...
    @property
    def turns_delivered(self) -> int | None: ...
    @property
    def runner_error(self) -> str | None: ...
    @property
    def judge_error(self) -> str | None: ...
    @property
    def judge_cannot_tell(self) -> Collection[str]: ...
    @property
    def judge_cannot_tell_boundary(self) -> Sequence[str]: ...
    @property
    def judge_cut_short(self) -> Mapping[str, JudgeCutShort]: ...
    @property
    def candidate_error(self) -> str | None: ...
    @property
    def infra_error(self) -> str | None: ...
    @property
    def scored_at(self) -> str: ...
    def judge_score(self, dim: str) -> ScoreView | None: ...
    def judge_scores(self) -> Sequence[ScoreView]: ...
    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self: ...


class SharedValues:
    """The values one read's readings hold once: each repeated text, and each distinct request-settings model.

    A scope's results repeat most of their text (a run's id, a model's name, a dimension, a criterion's fingerprint)
    and every score carries a copy of the settings its judge was sent with; hydrated, each is its own object. A read
    that builds its readings through one ``SharedValues`` holds each equal value once, so a reading's bytes are its
    own. Settings are matched by equality, as the readers compare them (``==``), so one held in place of another is
    one every reader reads as the same. Its size is the read's vocabulary, never its results'; it lives as long as the
    read does.

    **A mapping it holds is shared by every reading that holds an equal one** (:meth:`names`, an empty
    :meth:`mapping`), so a reading's mappings are read and never changed in place: a reader that needs a changed
    reading copies it with the change (:meth:`ResultReading.model_copy`), as it would a stored result.
    """

    __slots__ = ("_texts", "_tuples", "_names", "_empty", "_settings")

    def __init__(self) -> None:
        self._texts: dict[str, str] = {}
        self._tuples: dict[tuple[str, ...], tuple[str, ...]] = {}
        self._names: dict[tuple[tuple[str, str], ...], dict[str, str]] = {}
        self._empty: dict[str, Any] = {}
        self._settings: dict[tuple[object, ...], list[ClientRequestSettings]] = {}

    def text[T: str | None](self, value: T) -> T:
        """``value``, or the equal text already held."""
        if value is None:
            return value
        return cast(T, self._texts.setdefault(value, value))

    def texts(self, values: Iterable[str]) -> tuple[str, ...]:
        """``values`` as a tuple, the equal tuple already held if there is one."""
        held = tuple(self.text(value) for value in values)
        return self._tuples.setdefault(held, held)

    def names(self, mapping: Mapping[str, str]) -> Mapping[str, str]:
        """A mapping of text to text (a result's judge configs by dimension), the equal one already held if there is
        one."""
        key = tuple(mapping.items())
        held = self._names.get(key)
        if held is None:
            held = self._names[key] = {self.text(name): self.text(value) for name, value in key}
        return held

    def mapping[V](self, mapping: Mapping[str, V]) -> Mapping[str, V]:
        """``mapping`` with its keys held once, in its order: the one empty mapping when it is empty."""
        if not mapping:
            return cast(dict[str, V], self._empty)
        return {self.text(name): value for name, value in mapping.items()}

    def settings[S: ClientRequestSettings | None](self, value: S) -> S:
        """``value``, or an equal settings model already held."""
        if value is None:
            return value
        held = self._settings.setdefault((type(value), *vars(value).values()), [])
        for candidate in held:
            if candidate == value:
                return cast(S, candidate)
        held.append(value)
        return value


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreReading:
    """A judge's score without its reasoning (:class:`ScoreView`)."""

    dim: str
    scale: RubricScale
    axis: RubricAxis
    score: int
    served_model: str | None
    judge_temperature: JudgeTemperature | None
    judge_sent_settings: ClientRequestSettings | None
    output_fingerprint: str | None
    criterion_fingerprint: str | None

    @classmethod
    def of(cls, score: RubricScore, shared: SharedValues) -> ScoreReading:
        """The reading of a stored score, its repeated values held once in ``shared``."""
        return cls(
            dim=shared.text(score.dim),
            scale=shared.text(score.scale),
            axis=shared.text(score.axis),
            score=score.score,
            served_model=shared.text(score.served_model),
            judge_temperature=score.judge_temperature,
            judge_sent_settings=shared.settings(score.judge_sent_settings),
            output_fingerprint=shared.text(score.output_fingerprint),
            criterion_fingerprint=shared.text(score.criterion_fingerprint),
        )

    @property
    def label_key(self) -> LabelKey | None:
        """What this score was given on, as :attr:`RubricScore.label_key` reads it; the stored score was validated to
        carry both halves or neither."""
        if self.output_fingerprint is None or self.criterion_fingerprint is None:
            return None
        return LabelKey(self.output_fingerprint, self.criterion_fingerprint)

    @property
    def normalized(self) -> float:
        """The score on 0–1, as :attr:`RubricScore.normalized` reads it."""
        return SCALES[self.scale].normalized(self.score)

    def clears(self, rubric_threshold: int) -> bool:
        """Whether this score meets the bar, as :meth:`RubricScore.clears` decides it."""
        fixed = SCALES[self.scale].fixed_bar
        return self.score >= (rubric_threshold if fixed is None else fixed)


@dataclass(frozen=True, slots=True, kw_only=True)
class GoalReading:
    """A goal-state check's outcome (:class:`GoalView`)."""

    expression: str
    passed: bool
    detail: str

    @classmethod
    def of(cls, outcome: GoalStateOutcome, shared: SharedValues) -> GoalReading:
        """The reading of a stored outcome."""
        return cls(
            expression=shared.text(outcome.expression), passed=outcome.passed, detail=shared.text(outcome.detail)
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class PreconditionReading:
    """A precondition's outcome (:class:`PreconditionView`)."""

    expression: str
    presumes: str
    detail: str

    @classmethod
    def of(cls, outcome: PreconditionOutcome, shared: SharedValues) -> PreconditionReading:
        """The reading of a stored outcome."""
        return cls(
            expression=shared.text(outcome.expression),
            presumes=shared.text(outcome.presumes),
            detail=shared.text(outcome.detail),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageReading:
    """One usage row (:class:`UsageView`), without the reasons a call went unpriced. The count of each reason a call
    went unanswered is carried: each arm's census sums them (``analysis/bundle/trials.py``)."""

    role: str
    model: str | None
    served_model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    price_source: str | None
    call_count: int | None
    provider: str | None
    provider_unit: str | None
    provider_units: int | None
    actor_id: str | None
    purpose: str | None
    sent_settings: tuple[ClientRequestSettings, ...] | None
    unanswered_retried: int | None
    unanswered_lost: int | None
    unanswered_reasons: Mapping[str, int] | None

    @classmethod
    def of(cls, row: RoleUsage, shared: SharedValues) -> UsageReading:
        """The reading of a stored row."""
        sent = row.sent_settings
        return cls(
            role=shared.text(row.role),
            model=shared.text(row.model),
            served_model=shared.text(row.served_model),
            prompt_tokens=row.prompt_tokens,
            completion_tokens=row.completion_tokens,
            reasoning_tokens=row.reasoning_tokens,
            cost_usd=row.cost_usd,
            price_source=shared.text(row.price_source),
            call_count=row.call_count,
            provider=shared.text(row.provider),
            provider_unit=shared.text(row.provider_unit),
            provider_units=row.provider_units,
            actor_id=shared.text(row.actor_id),
            purpose=shared.text(row.purpose),
            sent_settings=None if sent is None else tuple(shared.settings(settings) for settings in sent),
            unanswered_retried=row.unanswered_retried,
            unanswered_lost=row.unanswered_lost,
            unanswered_reasons=None if (reasons := row.unanswered_reasons) is None else shared.mapping(reasons),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class LatencyReading:
    """A result's latency (:class:`LatencyView`)."""

    total_ms: float | None
    llm_ms: float | None
    tool_ms: float | None
    async_wait_ms: float | None
    judge_ms: float | None

    @classmethod
    def of(cls, latency: LatencyMetrics) -> LatencyReading:
        """The reading of a stored latency."""
        return cls(
            total_ms=latency.total_ms,
            llm_ms=latency.llm_ms,
            tool_ms=latency.tool_ms,
            async_wait_ms=latency.async_wait_ms,
            judge_ms=latency.judge_ms,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RescoreReading:
    """A re-judge's record: its spend (:class:`RescoreView`)."""

    cost_usd: float | None

    @classmethod
    def of(cls, rescore: JudgeRescore) -> RescoreReading:
        """The reading of a stored re-judge."""
        return cls(cost_usd=rescore.cost_usd)


@dataclass(frozen=True, slots=True, kw_only=True)
class RepeatedScoreReading:
    """One score of a repeat of the judge (:class:`RepeatedScoreView`)."""

    dim: str
    scale: RubricScale
    first_score: int
    first_served_model: str | None
    first_judge_config_id: str | None
    first_judge_temperature: JudgeTemperature | None
    first_judge_sent_settings: ClientRequestSettings | None
    repeat: ScoreReading | None
    #: The repeat's "cannot tell" note, read only as whether there is one.
    cannot_tell: str | None

    @classmethod
    def of(cls, entry: RepeatedScore, shared: SharedValues) -> RepeatedScoreReading:
        """The reading of a stored entry."""
        return cls(
            dim=shared.text(entry.dim),
            scale=shared.text(entry.scale),
            first_score=entry.first_score,
            first_served_model=shared.text(entry.first_served_model),
            first_judge_config_id=shared.text(entry.first_judge_config_id),
            first_judge_temperature=entry.first_judge_temperature,
            first_judge_sent_settings=shared.settings(entry.first_judge_sent_settings),
            repeat=None if entry.repeat is None else ScoreReading.of(entry.repeat, shared),
            cannot_tell=entry.cannot_tell,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class RepeatReading:
    """A repeat of the judge's scores (:class:`RepeatView`)."""

    judge_config_ids: Mapping[str, str]
    scores: tuple[RepeatedScoreReading, ...]

    @classmethod
    def of(cls, repeat: JudgeRepeat, shared: SharedValues) -> RepeatReading:
        """The reading of a stored repeat."""
        return cls(
            judge_config_ids=shared.names(repeat.judge_config_ids),
            scores=tuple(RepeatedScoreReading.of(entry, shared) for entry in repeat.scores),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SecondScoreReading:
    """One score of a second judge's pass (:class:`SecondScoreView`)."""

    dim: str
    scale: RubricScale
    first_score: int
    first_served_model: str | None
    first_judge_config_id: str | None
    first_judge_temperature: JudgeTemperature | None
    first_judge_sent_settings: ClientRequestSettings | None
    second: ScoreReading | None
    #: The second judge's "cannot tell" note, read only as whether there is one.
    cannot_tell: str | None

    @classmethod
    def of(cls, entry: SecondJudgeScore, shared: SharedValues) -> SecondScoreReading:
        """The reading of a stored entry."""
        return cls(
            dim=shared.text(entry.dim),
            scale=shared.text(entry.scale),
            first_score=entry.first_score,
            first_served_model=shared.text(entry.first_served_model),
            first_judge_config_id=shared.text(entry.first_judge_config_id),
            first_judge_temperature=entry.first_judge_temperature,
            first_judge_sent_settings=shared.settings(entry.first_judge_sent_settings),
            second=None if entry.second is None else ScoreReading.of(entry.second, shared),
            cannot_tell=entry.cannot_tell,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SecondJudgeReading:
    """The judge a second pass was asked of (:class:`SecondJudgeView`): the prompt per dimension it named is carried,
    since the bundle's judge-change reading names a judge by its prompts (``analysis/bundle/judges.py``)."""

    model: str
    config_ids: Mapping[str, str] | None
    temperature: float | None

    @classmethod
    def of(cls, judge: SecondJudge, shared: SharedValues) -> SecondJudgeReading:
        """The reading of a stored judge."""
        configs = judge.config_ids
        return cls(
            model=shared.text(judge.model),
            config_ids=None if configs is None else shared.names(configs),
            temperature=judge.temperature,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SecondJudgingReading:
    """A second judge's pass (:class:`SecondJudgingView`)."""

    pass_id: str
    judge: SecondJudgeReading
    judge_config_ids: Mapping[str, str]
    scores: tuple[SecondScoreReading, ...]

    @classmethod
    def of(cls, judging: SecondJudging, shared: SharedValues) -> SecondJudgingReading:
        """The reading of a stored pass."""
        return cls(
            pass_id=shared.text(judging.pass_id),
            judge=SecondJudgeReading.of(judging.judge, shared),
            judge_config_ids=shared.names(judging.judge_config_ids),
            scores=tuple(SecondScoreReading.of(entry, shared) for entry in judging.scores),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class ResultReading:
    """A stored result as the readers read it (:class:`ResultView`): built from a hydrated result by :meth:`of`.

    Its record lists are tuples (an empty one costs nothing); a copy (:meth:`model_copy`) holds what it is given.
    """

    id: str
    doc_type: str
    schema_version: int
    schema_minor: int
    scope_id: str
    eval_run_id: str
    test_case_id: str
    model: str
    k_iteration: int
    lease_epoch: int
    superseded_spend_usd: float | None
    metered_calls: int | None
    metered_calls_refused: int | None
    candidate_kind: str | None
    candidate_instance_id: str | None
    subject_id: str | None
    has_trace: bool
    precondition_outcomes: Sequence[PreconditionReading]
    goal_state_outcomes: Sequence[GoalReading]
    rubric_scores: Sequence[ScoreReading]
    judge_model: str | None
    transcript_score: ScoreReading | None
    outcome_score: ScoreReading | None
    judge_config_ids: Mapping[str, str]
    judge_rescores: Sequence[RescoreReading]
    judge_repeats: Sequence[RepeatReading]
    judge_seconds: Sequence[SecondJudgingReading]
    cost_usd: float | None
    cost_roles: Sequence[str]
    usage: Sequence[UsageReading]
    latency: LatencyReading | None
    async_deliveries: Sequence[AsyncDelivery] | None
    covariates: Mapping[str, str | float]
    phase_timings: Mapping[str, float]
    host_measures: Mapping[str, bool | float | str]
    variant_key: str
    identity_version: int
    termination: CellTermination
    stop_cause: ConversationStopCause | None
    turns_delivered: int | None
    runner_error: str | None
    judge_error: str | None
    #: The dimensions the judge could not tell, in the order the result holds them; the judge's note on each is prose,
    #: which no rule reads.
    judge_cannot_tell: tuple[str, ...]
    judge_cannot_tell_boundary: Sequence[str]
    judge_cut_short: Mapping[str, JudgeCutShort]
    candidate_error: str | None
    infra_error: str | None
    scored_at: str

    @classmethod
    def of(cls, result: EvalResult, shared: SharedValues | None = None) -> ResultReading:
        """The reading of a hydrated result: each attribute copied from it, never recomputed.

        :param result: A result the store returned, hydrated and validated.
        :ptype result: EvalResult
        :param shared: Where a read's repeated values are held once (:class:`SharedValues`); ``None`` shares none
            beyond this result's own.
        :ptype shared: SharedValues | None
        :return: Its reading.
        :rtype: ResultReading
        """
        held = SharedValues() if shared is None else shared
        text = held.text
        transcript, outcome, latency = result.transcript_score, result.outcome_score, result.latency
        deliveries = result.async_deliveries
        return cls(
            id=result.id,
            doc_type=text(result.doc_type),
            schema_version=result.schema_version,
            schema_minor=result.schema_minor,
            scope_id=text(result.scope_id),
            eval_run_id=text(result.eval_run_id),
            test_case_id=text(result.test_case_id),
            model=text(result.model),
            k_iteration=result.k_iteration,
            lease_epoch=result.lease_epoch,
            superseded_spend_usd=result.superseded_spend_usd,
            metered_calls=result.metered_calls,
            metered_calls_refused=result.metered_calls_refused,
            candidate_kind=text(result.candidate_kind),
            candidate_instance_id=text(result.candidate_instance_id),
            subject_id=text(result.subject_id),
            has_trace=result.has_trace,
            precondition_outcomes=tuple(PreconditionReading.of(item, held) for item in result.precondition_outcomes),
            goal_state_outcomes=tuple(GoalReading.of(item, held) for item in result.goal_state_outcomes),
            rubric_scores=tuple(ScoreReading.of(score, held) for score in result.rubric_scores),
            judge_model=text(result.judge_model),
            transcript_score=None if transcript is None else ScoreReading.of(transcript, held),
            outcome_score=None if outcome is None else ScoreReading.of(outcome, held),
            judge_config_ids=held.names(result.judge_config_ids),
            judge_rescores=tuple(RescoreReading.of(rescore) for rescore in result.judge_rescores),
            judge_repeats=tuple(RepeatReading.of(repeat, held) for repeat in result.judge_repeats),
            judge_seconds=tuple(SecondJudgingReading.of(judging, held) for judging in result.judge_seconds),
            cost_usd=result.cost_usd,
            cost_roles=held.texts(result.cost_roles),
            usage=tuple(UsageReading.of(row, held) for row in result.usage),
            latency=None if latency is None else LatencyReading.of(latency),
            async_deliveries=None if deliveries is None else tuple(deliveries),
            covariates=held.mapping(result.covariates),
            phase_timings=held.mapping(result.phase_timings),
            host_measures=held.mapping(result.host_measures),
            variant_key=text(result.variant_key),
            identity_version=result.identity_version,
            termination=text(result.termination),
            stop_cause=result.stop_cause,
            turns_delivered=result.turns_delivered,
            runner_error=result.runner_error,
            judge_error=result.judge_error,
            judge_cannot_tell=held.texts(result.judge_cannot_tell),
            judge_cannot_tell_boundary=held.texts(result.judge_cannot_tell_boundary),
            judge_cut_short=held.mapping(result.judge_cut_short),
            candidate_error=result.candidate_error,
            infra_error=result.infra_error,
            scored_at=result.scored_at,
        )

    def judge_scores(self) -> list[ScoreReading]:
        """Every score the judge gave: the template's dimensions, then the two reserved axes when scored, as
        :meth:`EvalResult.judge_scores` lists them."""
        return [
            score for score in (*self.rubric_scores, self.transcript_score, self.outcome_score) if score is not None
        ]

    def judge_score(self, dim: str) -> ScoreReading | None:
        """The judge's score for one dimension, wherever the reading carries it, as :meth:`EvalResult.judge_score`
        finds it."""
        return next((score for score in self.judge_scores() if score.dim == dim), None)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """A copy with ``update``'s attributes replaced, as a pydantic model's ``model_copy`` makes one: shallow unless
        ``deep``, the replaced values taken as given.

        :param update: Attribute values to replace; a name the reading does not carry is refused (``TypeError``).
        :ptype update: Mapping[str, Any] | None
        :param deep: Copy every held value too.
        :ptype deep: bool
        :return: The copy.
        :rtype: Self
        """
        source = copy.deepcopy(self) if deep else self
        return replace(source, **dict(update or {}))


#: Which path served a read of readings or groups: the store's own narrowed and grouped reads (``store``), or the
#: engine's reads by key and by id, folded in Python (``engine``).
type ReadPath = Literal["store", "engine"]


class ReadingsRead:
    """The readings a read hands out, in its order, beside the path that served it and the store calls it has made.

    Walked once: each page of results is read from the store when the one before it has been walked, so
    :attr:`store_calls` counts the calls made so far, and the whole read's once the walk ends.

    :ivar path: The path that served the read.
    :ivar store_calls: The store calls made so far.
    """

    __slots__ = ("path", "store_calls", "_pages", "_walked")

    def __init__(self, path: ReadPath, *, store_calls: int, pages: Iterator[tuple[int, list[ResultReading]]]) -> None:
        """Hold a read.

        :param path: The path that serves it.
        :ptype path: ReadPath
        :param store_calls: The store calls made before the first page (its keys).
        :ptype store_calls: int
        :param pages: Each page's readings, beside the store calls reading it took, read lazily.
        :ptype pages: Iterator[tuple[int, list[ResultReading]]]
        """
        self.path = path
        self.store_calls = store_calls
        self._pages = pages
        self._walked = False

    def __iter__(self) -> Iterator[ResultReading]:
        """Each reading, in the read's order.

        :raises RuntimeError: The read was walked before: its pages are read once, and a second walk would hand out
            nothing as though nothing were stored.
        """
        if self._walked:
            raise RuntimeError("a read of readings is walked once; read again for a second walk")
        self._walked = True
        for calls, page in self._pages:
            self.store_calls += calls
            yield from page


@dataclass(frozen=True, slots=True)
class GroupsRead:
    """The groups a grouped read answered, beside the path that served it and the store calls it made.

    :ivar path: The path that served the read.
    :ivar store_calls: The store calls it made.
    :ivar groups: One dict per group: each ``by`` path and its value, then each measure by name, as
        :meth:`~threetears.evals.schema.store_port.ResultReadings.grouped` answers. In no promised order.
    """

    path: ReadPath
    store_calls: int
    groups: list[dict[str, Any]]
