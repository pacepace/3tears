"""``compare``: run two or more candidates over one case list, with one as the control, and report which separated.

The second rung of adopting the engine. :func:`~threetears.evals.quick.run_eval` measures one candidate;
a newcomer's next question is whether a changed prompt or a different model does better, and that is a
campaign with a control. Everything here is the engine's own path, composed:

- **Each arm is one** :func:`~threetears.evals.quick.run_eval` **run**, labelled by its arm name, all into
  one host and one scope. The arm name is the run's candidate model, so it is what the variant key is
  built from: two arms with different names are two variants, over one content-addressed case set.
- **The campaign declares its design** — one axis, the candidate-model lever, at a level per arm; a
  controlled stimulus, since every arm saw the same cases; a commissioned apparatus, since the runs were
  launched for it; and the repeats each arm ran — through
  :func:`~threetears.evals.analysis.create_campaign`, which gates the declaration against the host's
  vocabulary as it gates any other.
- **The control is designated from the control arm's run** through
  :func:`~threetears.evals.analysis.set_campaign_control`, which reads the variant key that run's
  observations carry and writes it to the declaration's ``control``. No key is computed here.
- **The report is the campaign's report** (:func:`~threetears.evals.analysis.campaign_report`): with no
  analysis generated, the code-only report, whose contrasts table tests every other arm against the
  control on every reading, Holm-corrected, and says which separated.

**Two factors or more.** Named ``factors=("model", "prompt")``, the arms are keyed by their level of each
factor (``("model-a", "v2")``) and each factor is a lever of its own: ``model`` is the run's candidate model
and every other factor is a lever :func:`~threetears.evals.quick.run_eval` states on the run (``levers=``),
which the engine names ``callable.<factor>`` and keys into the variant beside the model. The campaign
declares one axis per factor, so the report names every arm by both coordinates. The engine tests
contrasts against ONE control, so a factor's effect at another level of the other factor — the prompt's
effect on the second model — is read against a second control over the same runs:
:meth:`Comparison.against` files them as a second campaign, with no run repeated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from threetears.evals.analysis import (
    Report,
    TableBlock,
    campaign_report,
    create_campaign,
    get_campaign,
    report_markdown,
    set_campaign_control,
)
from threetears.evals.contracts import DEFAULT_LAUNCH_K_RUNS, CassetteMode
from threetears.evals.contracts.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.ops.summary import EvalSummary
from threetears.evals.quick.levers import refuse_unusable_lever_names
from threetears.evals.quick.one_call import CALLABLE_KIND, Candidate, ExpectedLabel, Scorer, callable_host, run_eval
from threetears.evals.quick.tools import Tool, ToolUsingCandidate
from threetears.evals.quick.world import CaseSeed, World, WorldCandidate

#: Who a :func:`compare` campaign and its control are recorded as created by, unless the caller says.
COMPARE_CREATED_BY = "compare"

#: An arm's key: its name, which is its model, when :func:`compare` is given no ``factors``; with them, its
#: level of each factor, in the order ``factors`` names them.
ArmKey = str | tuple[str, ...]

#: The one factor of a :func:`compare` named no ``factors``: the candidate model, which each arm's name is.
_MODEL_ONLY = (CANDIDATE_MODEL_LEVER,)


@dataclass(frozen=True)
class Comparison:
    """What :func:`compare` ran and what its campaign's report says.

    Attributes:
        campaign_id: The campaign holding every arm's run.
        scope_id: The scope the campaign and its runs are stored in.
        name: The campaign's name, which titles its report.
        control: The arm every other arm is tested against.
        arms: Each arm's run summary, by arm key, in the order the arms were given.
        report: The campaign's report, as :func:`~threetears.evals.analysis.campaign_report` read it.
        host: The host the runs and the campaign are stored in, for reading them further.
        factors: The factors each arm key names a level of, in key order: ``("model",)`` when the arms
            are keyed by name.
    """

    campaign_id: str
    scope_id: str
    name: str
    control: ArmKey
    arms: dict[ArmKey, EvalSummary]
    report: Report
    host: EvalHost
    factors: tuple[str, ...] = _MODEL_ONLY

    def render(self) -> str:
        """The report as Markdown: the arms, the contrasts against the control and their verdicts, the charts."""
        return report_markdown(self.report)

    def contrasts(self, reading: str | None = None) -> list[dict[str, Any]]:
        """The rows of the report's "Contrasts against the control" table, each arm tested against the control.

        Args:
            reading: Only the rows on this reading (``"accuracy"``); ``None`` keeps every row.

        Returns:
            One row per arm and reading, keyed ``question``, ``reading``, ``contrast`` (the arm, as the report
            names it), ``control``, ``delta`` (arm minus control), ``p_adjusted`` (Holm, over the campaign's
            family) and ``verdict``; empty when the report tested nothing.
        """
        rows = [
            row
            for block in self.report.blocks
            if isinstance(block, TableBlock) and block.name == "comparisons"
            for row in block.rows
        ]
        return [dict(row) for row in rows if reading is None or row["reading"] == reading]

    def against(self, control: ArmKey, *, name: str | None = None, created_by: str = COMPARE_CREATED_BY) -> Comparison:
        """The same runs read against another control: a second campaign over them, with no run repeated.

        The engine tests every arm against one control, so in a factorial the effect of one factor at the
        control's level of the other is in :attr:`report`, and its effect at another level is read here —
        the prompt's effect on the second model is the contrast against the first prompt on that model.
        Each campaign Holm-corrects its own contrasts, so the two reports are two families, not one.

        Args:
            control: The arm to test every other against, by its key in :attr:`arms`.
            name: The new campaign's name; ``None`` names it after this one and the new control.
            created_by: Who the campaign and its control are recorded as created by.

        Returns:
            A comparison of the same arms, over the same runs, against ``control``.

        Raises:
            ValueError: ``control`` names no arm.
            ValidationFailedError: The host refuses the campaign's declaration.
        """
        _refuse_an_unknown_control(self.arms, control)
        campaign = get_campaign(self.host.storage, self.campaign_id, self.scope_id)
        design = campaign.declared_design
        repetitions = design.intended_repetitions if design is not None else None
        return _declare(
            self.host,
            self.arms,
            self.factors,
            control=control,
            scope_id=self.scope_id,
            name=name or f"{self.name}, against {_label(control, self.factors)}",
            behavior=campaign.behavior,
            repetitions=repetitions,
            created_by=created_by,
        )


def _label(arm: ArmKey, factors: tuple[str, ...]) -> str:
    """How a title names an arm: its name, or each factor at its level."""
    if isinstance(arm, str):
        return arm
    return ", ".join(f"{factor}={level}" for factor, level in zip(factors, arm, strict=True))


def _coordinates(arm: ArmKey, factors: tuple[str, ...]) -> dict[str, str]:
    """An arm's level of each factor, by factor."""
    return {CANDIDATE_MODEL_LEVER: arm} if isinstance(arm, str) else dict(zip(factors, arm, strict=True))


def _refuse_an_unknown_control(arms: Mapping[ArmKey, Any], control: ArmKey) -> None:
    if control not in arms:
        raise ValueError(f"control {control!r} names no arm; the arms are {', '.join(map(repr, arms))}")


def _factors(factors: Sequence[str] | None) -> tuple[str, ...]:
    """The factors as a tuple, refusing a set no arm key could be read by."""
    if factors is None:
        return _MODEL_ONLY
    if isinstance(factors, str):
        raise ValueError("factors is a sequence of factor names, not one string")
    given = tuple(factors)
    if CANDIDATE_MODEL_LEVER not in given:
        raise ValueError(
            f"factors {given!r} leave out {CANDIDATE_MODEL_LEVER!r}: every run is at some candidate model, "
            f"so name it as a factor, at one level if the arms share it"
        )
    refuse_unusable_lever_names([factor for factor in given if factor != CANDIDATE_MODEL_LEVER])
    if given.count(CANDIDATE_MODEL_LEVER) > 1:
        raise ValueError(f"factors name {CANDIDATE_MODEL_LEVER!r} more than once; each name is one factor")
    return given


def _refuse_unusable_arms(
    candidates: Mapping[ArmKey, Candidate | ToolUsingCandidate | WorldCandidate],
    control: ArmKey,
    factors: tuple[str, ...],
) -> None:
    if isinstance(candidates, str) or not isinstance(candidates, Mapping):
        raise ValueError("compare needs its candidates as a mapping of arm name to candidate")
    if len(candidates) < 2:
        raise ValueError(f"compare needs at least two candidates to compare, and was given {len(candidates)}")
    if factors == _MODEL_ONLY and all(isinstance(arm, str) for arm in candidates):
        if blank := [repr(arm) for arm in candidates if not str(arm).strip()]:
            raise ValueError(f"an arm's name labels its run and its variant, and {', '.join(blank)} is blank")
    elif misshapen := [
        repr(arm)
        for arm in candidates
        if not isinstance(arm, tuple)
        or len(arm) != len(factors)
        or not all(isinstance(level, str) and level.strip() for level in arm)
    ]:
        raise ValueError(
            f"with factors {factors!r} each arm is keyed by a non-blank level of each, as a tuple in that order, "
            f"and {', '.join(misshapen)} is not"
        )
    _refuse_an_unknown_control(candidates, control)


def _declare(
    host: EvalHost,
    arms: Mapping[ArmKey, EvalSummary],
    factors: tuple[str, ...],
    *,
    control: ArmKey,
    scope_id: str,
    name: str,
    behavior: str,
    repetitions: int | None,
    created_by: str,
) -> Comparison:
    """File the arms' runs as one campaign, one axis per factor, designate ``control``, and read its report."""
    ordered = [control, *(arm for arm in arms if arm != control)]
    prefix = host.profile.kind_contract(CALLABLE_KIND).lever_prefix
    axes = []
    for factor in factors:
        levels = list(dict.fromkeys(_coordinates(arm, factors)[factor] for arm in ordered))
        axes.append(
            {
                "axis_id": factor if factor == CANDIDATE_MODEL_LEVER else f"{prefix}.{factor}",
                "values": [{"content": level, "display": level} for level in levels],
                "rationale": (
                    f"does any arm do better than {control}"
                    if factors == _MODEL_ONLY
                    else f"does moving {factor} change what the arms score, against {_label(control, factors)}"
                ),
            }
        )
    design: dict[str, Any] = {"axes": axes, "controls": {"stimulus": "controlled", "apparatus": "commissioned"}}
    if repetitions is not None:
        design["intended_repetitions"] = repetitions
    campaign = create_campaign(
        host.storage,
        {
            "name": name,
            "subject_id": name,
            "behavior": behavior,
            "run_ids": [summary.run_id for summary in arms.values()],
            "declared_design": design,
        },
        scope_id=scope_id,
        created_by=created_by,
        profile=host.profile,
    )
    set_campaign_control(
        host.storage, campaign.id, scope_id, arms[control].run_id, set_by=created_by, profile=host.profile
    )
    return Comparison(
        campaign_id=campaign.id,
        scope_id=scope_id,
        name=name,
        control=control,
        arms=dict(arms),
        report=campaign_report(host, campaign.id, scope_id),
        host=host,
        factors=factors,
    )


async def compare(
    cases: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Candidate | ToolUsingCandidate | WorldCandidate]
    | Mapping[tuple[str, ...], Candidate | ToolUsingCandidate | WorldCandidate],
    scorers: Sequence[Scorer] = (),
    *,
    control: ArmKey,
    scope_id: str,
    expected: ExpectedLabel | None = None,
    host: EvalHost | None = None,
    k: int = DEFAULT_LAUNCH_K_RUNS,
    name: str | None = None,
    created_by: str = COMPARE_CREATED_BY,
    factors: Sequence[str] | None = None,
    tools: Mapping[str, Tool] | None = None,
    cassette_mode: CassetteMode = "off",
    cassette_corpus_id: str | None = None,
    world: World | None = None,
    seed: CaseSeed | None = None,
    goal_checks: Sequence[str] = (),
) -> Comparison:
    """Run each candidate over every case ``k`` times as one arm, test every arm against ``control``, and report.

    Args:
        cases: The cases, each a JSON object, the same for every arm.
        candidates: The arms. Keyed by name when no ``factors`` are given: each name labels its run and is
            the level its arm is declared at, so it is what the report calls the arm (``model=<name>``).
            With ``factors``, keyed by a tuple of the arm's level of each, in order
            (``("model-a", "v2")``): the ``model`` level is the run's candidate model, and every other
            level is stated on the run as that factor's lever (``callable.<factor>=<level>``).
        scorers: The grades, as :func:`~threetears.evals.quick.run_eval` takes them.
        control: The arm every other arm is tested against, by its key.
        scope_id: The scope every run and the campaign are stored in.
        expected: Declares every candidate a classifier, as :func:`~threetears.evals.quick.run_eval` takes it.
        host: Where to run and store: ``None`` builds one :func:`~threetears.evals.quick.callable_host` over
            the scorers and the factors other than ``model`` for every arm. A host of the caller's own is held
            to what ``run_eval`` holds it to, and must declare those factors as levers.
        k: Repeats per case, per arm.
        name: The campaign's name, which titles its report; ``None`` names it by its arms, control first, or
            by its factors.
        created_by: Who the campaign and its control are recorded as created by.
        factors: The factors the arms are keyed by, ``model`` among them (``("model", "prompt")``); each is one
            axis of the campaign's declared design. ``None`` keys the arms by name, on the model axis alone.
        tools: The tools every arm's candidate calls, as :func:`~threetears.evals.quick.run_eval` takes them.
        cassette_mode: Every arm's cassette mode, as :func:`~threetears.evals.quick.run_eval` takes it. Replay
            is what makes the arms comparable when the tools' answers vary: every arm is served the one
            capture ``cassette_corpus_id`` names, so no difference between them is a difference in what
            their tools said.
        cassette_corpus_id: The capture every arm replays, made over the same cases in ``host`` and ``scope_id``.
        world: The world every arm's candidate acts on, as :func:`~threetears.evals.quick.run_eval` takes it;
            with no ``host``, the one built declares it.
        seed: Each case's starting state, the same for every arm, as :func:`~threetears.evals.quick.run_eval`
            takes it.
        goal_checks: The checks every arm's end state is graded by, as :func:`~threetears.evals.quick.run_eval`
            takes them.

    Returns:
        The comparison: every arm's summary, the campaign's id and its report.

    Raises:
        ValueError: Fewer than two candidates, a blank arm name, an arm key that is not a level of each
            factor, factors without ``model`` or with an unusable or repeated name, a ``control`` that names
            no arm, or anything :func:`~threetears.evals.quick.run_eval` refuses.
        ValidationFailedError: The launch refused, or the host refuses the campaign's declaration.
    """
    named = _factors(factors)
    arms_given: Mapping[ArmKey, Candidate | ToolUsingCandidate | WorldCandidate] = candidates  # type: ignore[assignment]
    _refuse_unusable_arms(arms_given, control, named)
    levers = tuple(factor for factor in named if factor != CANDIDATE_MODEL_LEVER)
    if host is None:
        host = callable_host(scorers, levers=levers, world=world)
    arms: dict[ArmKey, EvalSummary] = {}
    for arm, candidate in arms_given.items():
        coordinates = _coordinates(arm, named)
        arms[arm] = await run_eval(
            cases,
            candidate,
            scorers,
            scope_id=scope_id,
            expected=expected,
            host=host,
            k=k,
            model=coordinates[CANDIDATE_MODEL_LEVER],
            levers={lever: coordinates[lever] for lever in levers} or None,
            tools=tools,
            cassette_mode=cassette_mode,
            cassette_corpus_id=cassette_corpus_id,
            world=world,
            seed=seed,
            goal_checks=goal_checks,
        )
    if name is None:
        if named == _MODEL_ONLY:
            name = " vs ".join(_label(arm, named) for arm in [control, *(arm for arm in arms if arm != control)])
        else:
            name = " × ".join(named)
    return _declare(
        host,
        arms,
        named,
        control=control,
        scope_id=scope_id,
        name=name,
        behavior="classify" if expected is not None else "score",
        repetitions=k,
        created_by=created_by,
    )


__all__ = ["COMPARE_CREATED_BY", "ArmKey", "Comparison", "compare"]
