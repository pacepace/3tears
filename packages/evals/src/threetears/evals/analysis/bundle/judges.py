"""The judge-change reading: each judge the judged member runs recorded, and every drift reading spanning two of them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from threetears.evals.analysis.judge_drift import judge_drift
from threetears.evals.schema.models import EvalResult
from threetears.evals.analysis.bundle.schema import (
    JudgeChange,
    JudgeDriftLink,
    JudgeIdentityLevel,
)
from threetears.evals.analysis.bundle.observations import variant_key_of_run

if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.schema.models import EvalRun, SecondJudge


#: What a judge change across the member runs means for a judged comparison, quoted when there is one.
_JUDGE_CHANGE_SENTENCE = (
    "The judged member runs were scored by {n} different judges (model, prompts or temperature), so a judged "
    "difference between arms judged differently may be the judge's rather than the subject's."
)
_JUDGE_CHANGE_DRIFT = (
    " A drift reading re-scored one side's evidence under the other side's judge ({links}); read its movement before "
    "attributing a judged difference to the subject."
)
_JUDGE_CHANGE_NO_DRIFT = (
    " Nothing measured how far the judge change alone moves the scores: re-score one side's stored evidence under "
    "the other side's judge (judge_drift_check) to read it."
)


def _judge_identity(run: EvalRun) -> tuple[str, tuple[tuple[str, str], ...], float | None] | None:
    """A judged run's judge as it recorded it at launch — pin, configs, temperature — or None for an unjudged run."""
    if run.judge_model is None:
        return None
    return run.judge_model, tuple(sorted((run.judge_config_ids or {}).items())), run.judge_temperature


def _names_level(
    judge: SecondJudge,
    level: tuple[str, tuple[tuple[str, str], ...], float | None],
    own: tuple[str, tuple[tuple[str, str], ...], float | None],
) -> bool:
    """Whether a second judge is the judge ``level`` names, asked of a run judged as ``own``.

    The model must be the level's pin; the prompts are the run's own when the second judge named none; and its
    temperature, when it named none, is what each prompt asks for — the run's own sampling.
    """
    configs = own[1] if judge.config_ids is None else tuple(sorted(judge.config_ids.items()))
    temperature = own[2] if judge.temperature is None else judge.temperature
    return judge.model == level[0] and configs == level[1] and temperature == level[2]


def _judge_change(runs: Sequence[EvalRun], results_by_run: Mapping[str, Sequence[EvalResult]]) -> JudgeChange:
    """Each judge the judged member runs recorded, and every drift reading among them that spans two of them."""
    by_level: dict[tuple[str, tuple[tuple[str, str], ...], float | None], list[EvalRun]] = {}
    for run in runs:
        if (identity := _judge_identity(run)) is not None:
            by_level.setdefault(identity, []).append(run)
    ordered = sorted(by_level, key=lambda level: (level[0], level[1], "" if level[2] is None else str(level[2])))
    levels = [
        JudgeIdentityLevel(
            judge_model=level[0],
            judge_config_ids=dict(level[1]),
            judge_temperature=level[2],
            run_ids=sorted(run.id for run in by_level[level]),
            variant_keys=sorted(
                {key for run in by_level[level] if (key := variant_key_of_run(results_by_run.get(run.id, [])))}
            ),
        )
        for level in ordered
    ]
    if len(levels) < 2:
        return JudgeChange(levels=levels)
    links = []
    for source_index, source in enumerate(ordered):
        for target_index, target in enumerate(ordered):
            if source_index == target_index:
                continue
            spanning = [
                (run, judging.pass_id)
                for run in by_level[source]
                for result in results_by_run.get(run.id, [])
                for judging in result.judge_seconds
                if _names_level(judging.judge, target, source)
            ]
            if not spanning:
                continue
            pass_ids = sorted({pass_id for _, pass_id in spanning})
            run_ids = sorted({run.id for run, _ in spanning})
            read = [
                result.model_copy(update={"judge_seconds": [j for j in result.judge_seconds if j.pass_id in pass_ids]})
                for run_id in run_ids
                for result in results_by_run.get(run_id, [])
            ]
            links.append(
                JudgeDriftLink(
                    from_level=source_index,
                    to_level=target_index,
                    run_ids=run_ids,
                    pass_ids=pass_ids,
                    drift=judge_drift(read),
                )
            )
    sentence = _JUDGE_CHANGE_SENTENCE.format(n=len(levels))
    if links:
        named = ", ".join(f"level {link.from_level} under level {link.to_level}'s judge" for link in links)
        sentence += _JUDGE_CHANGE_DRIFT.format(links=named)
    else:
        sentence += _JUDGE_CHANGE_NO_DRIFT
    return JudgeChange(levels=levels, drift_links=links, sentence=sentence)


__all__: list[str] = []
