"""``compare``: run two or more candidates over one case list, with one as the control, and report which separated.

The second rung of adopting the engine. :func:`~threetears.evals.quick.run_eval` measures one candidate;
a newcomer's next question is whether a changed prompt or a different model does better, and that is a
campaign with a control. Everything here is the engine's own path, composed:

- **Each arm is one run**, as :func:`~threetears.evals.quick.run_eval` makes it, labelled by its arm name,
  all into one host and one scope. The arm name is the run's level of the host's arm lever
  (:data:`~threetears.evals.quick.one_call.ARM_LEVER`, ``candidate``), every arm at one shared candidate model,
  so the report calls the arm ``candidate=<name>`` and it is what the variant key is built from: two arms with
  different names are two variants, over one content-addressed case set. Arms that ARE models are keyed with
  ``factors=("model",)``, and their names are then the runs' candidate models (``model=<name>``); a host of the
  caller's own that declares no arm lever names arms that way too.
- **Every arm is started in one launch.** The arms' runs are one launch group, every one prepared (every
  refusal made) before any starts and all started together, so they are measured side by side rather than
  one after another, and a refusal on the last arm leaves none run.
- **The campaign declares its design** — one axis, the arm lever (or the candidate-model lever), at a level per arm; a
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
from dataclasses import dataclass, field
from typing import Any

from threetears.evals.analysis import (
    DisclosureBlock,
    Report,
    TableBlock,
    assemble_context_bundle,
    build_code_only_report,
    create_campaign,
    get_campaign,
    report_markdown,
    set_campaign_control,
    variant_key_of_run,
)
from threetears.evals.contracts import DEFAULT_LAUNCH_K_RUNS, CassetteMode, utc_now_iso
from threetears.evals.contracts.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.ops.summary import CaseResult, EvalSummary, self_judging_text
from threetears.evals.quick.judged import Judge
from threetears.evals.quick.levers import refuse_unusable_lever_names
from threetears.evals.quick.one_call import (
    ARM_LEVER,
    CALLABLE_KIND,
    SHARED_ARM_MODEL,
    JUDGED_CALLABLE_KIND,
    Candidate,
    ExpectedLabel,
    Scorer,
    CallableArm,
    run_arms,
    callable_host,
)
from threetears.evals.run import list_results
from threetears.evals.quick.tools import Tool, ToolUsingCandidate
from threetears.evals.quick.world import CaseSeed, World, WorldCandidate

#: Who a :func:`compare` campaign and its control are recorded as created by, unless the caller says.
COMPARE_CREATED_BY = "compare"

#: An arm's key: its name, which is its model, when :func:`compare` is given no ``factors``; with them, its
#: level of each factor, in the order ``factors`` names them.
ArmKey = str | tuple[str, ...]

#: The one factor of a :func:`compare` named no ``factors``: the candidate model, which each arm's name is.
_MODEL_ONLY = (CANDIDATE_MODEL_LEVER,)

#: The one factor of a :func:`compare` named no ``factors`` on a host declaring the arm lever: the arm's name,
#: stated on its run as :data:`~threetears.evals.quick.one_call.ARM_LEVER`, every arm at one shared model.
_NAMED_ARMS = (ARM_LEVER,)

#: The factors that are levers of the engine or the host, never of the callable kind: no kind prefix names them.
_UNPREFIXED = frozenset({CANDIDATE_MODEL_LEVER, ARM_LEVER})


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
        factors: The factors each arm key names a level of, in key order: ``("candidate",)`` when the arms
            are keyed by name on the arm lever, ``("model",)`` when they are keyed by name as models.
        contrast_arms: The arm each row of the report's contrasts table tests, by its key in :attr:`arms`, row for
            row; ``None`` for a row no arm's variant matches. Empty, or of another length than the table, and
            :meth:`contrasts` names no arm on any row rather than guess one.
        kind: The kind every arm ran — the callable kind, or the judged one — whose levers the campaign's axes name.
    """

    campaign_id: str
    scope_id: str
    name: str
    control: ArmKey
    arms: dict[ArmKey, EvalSummary]
    report: Report
    host: EvalHost
    factors: tuple[str, ...] = _MODEL_ONLY
    contrast_arms: tuple[ArmKey | None, ...] = field(default=(), repr=False, compare=False)
    kind: str = field(default=CALLABLE_KIND, repr=False, compare=False)

    def results(self, arm: ArmKey) -> list[CaseResult]:
        """Every result of one arm: each case's answer on each repeat, its grades, and why it failed or was excluded.

        Args:
            arm: The arm, by its key in :attr:`arms` (``"baseline"``, or a tuple of levels with ``factors``).

        Returns:
            The arm's results, as :meth:`~threetears.evals.ops.summary.EvalSummary.results` reads them.

        Raises:
            ValueError: ``arm`` names no arm.
        """
        return self._arm(arm).results()

    def misses(self, arm: ArmKey) -> list[CaseResult]:
        """The results one arm's candidate missed, each saying why.

        Args:
            arm: The arm, by its key in :attr:`arms`.

        Returns:
            The arm's misses, as :meth:`~threetears.evals.ops.summary.EvalSummary.misses` reads them.

        Raises:
            ValueError: ``arm`` names no arm.
        """
        return self._arm(arm).misses()

    def _arm(self, arm: ArmKey) -> EvalSummary:
        _refuse_an_unknown_control(self.arms, arm, said="arm")
        return self.arms[arm]

    def render(self) -> str:
        """The report as Markdown: the arms, the contrasts against the control and their verdicts, the charts."""
        return report_markdown(self.report)

    def contrasts(self, reading: str | None = None) -> list[dict[str, Any]]:
        """The rows of the report's "Contrasts against the control" table, each arm tested against the control.

        Args:
            reading: Only the rows on this reading (``"accuracy"``); ``None`` keeps every row.

        Returns:
            One row per arm and reading, keyed ``arm`` (the arm tested, by its key in :attr:`arms`: the name
            you gave it, or its tuple of levels), ``question``, ``reading``, ``contrast`` (the arm, as the report
            names it), ``control`` (the control, as the report names it), ``control_mean`` and ``arm_mean`` (over the cases the test read),
            ``cases`` (how many, paired or not, and any one side ran that the test left out), ``delta`` (arm
            minus control), ``interval`` (on the delta, simultaneous over the family), ``hedges_g`` (the
            standardized effect), ``p_adjusted`` (Holm, over the campaign's family) and ``verdict``; empty when
            the report tested nothing.
        """
        rows = [
            row
            for block in self.report.blocks
            if isinstance(block, TableBlock) and block.name == "comparisons"
            for row in block.rows
        ]
        arms = self.contrast_arms if len(self.contrast_arms) == len(rows) else (None,) * len(rows)
        return [
            {"arm": arm, **row}
            for row, arm in zip(rows, arms, strict=True)
            if reading is None or row["reading"] == reading
        ]

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
            kind=self.kind,
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
    return {factors[0]: arm} if isinstance(arm, str) else dict(zip(factors, arm, strict=True))


def _refuse_an_unknown_control(arms: Mapping[ArmKey, Any], control: ArmKey, *, said: str = "control") -> None:
    if control not in arms:
        raise ValueError(f"{said} {control!r} names no arm; the arms are {', '.join(map(repr, arms))}")


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
    if len(factors) == 1 and all(isinstance(arm, str) for arm in candidates):
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
    kind: str,
    control: ArmKey,
    scope_id: str,
    name: str,
    behavior: str,
    repetitions: int | None,
    created_by: str,
) -> Comparison:
    """File the arms' runs as one campaign, one axis per factor, designate ``control``, and read its report.

    The report is the campaign's code-only report, as :func:`~threetears.evals.analysis.campaign_report` reads
    a campaign no analysis has been generated for — which a campaign filed a moment ago is — laid out here from
    the evidence bundle it is assembled from, so each contrast the report tests is matched to the arm whose run
    carries its variant key rather than to the words the report names it by.
    """
    ordered = [control, *(arm for arm in arms if arm != control)]
    prefix = host.profile.kind_contract(kind).lever_prefix
    axes = []
    for factor in factors:
        levels = list(dict.fromkeys(_coordinates(arm, factors)[factor] for arm in ordered))
        axes.append(
            {
                "axis_id": factor if factor in _UNPREFIXED else f"{prefix}.{factor}",
                "values": [{"content": level, "display": level} for level in levels],
                "rationale": (
                    f"does any arm do better than {control}"
                    if len(factors) == 1
                    else f"does moving {factor} change what the arms score, against {_label(control, factors)}"
                ),
            }
        )
    design: dict[str, Any] = {"axes": axes, "held_fixed": {"stimulus": "controlled", "apparatus": "commissioned"}}
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
    campaign = set_campaign_control(
        host.storage, campaign.id, scope_id, arms[control].run_id, set_by=created_by, profile=host.profile
    )
    bundle = assemble_context_bundle(campaign, storage=host.storage, profile=host.profile)
    report = build_code_only_report(
        bundle, measures=host.profile.measures, assembled_at=utc_now_iso(), campaign_name=campaign.name
    )
    report = _with_self_judging_disclosed(report, arms, factors)
    arm_of_variant = {
        variant_key_of_run(list_results(host.storage, summary.run_id, scope_id)): arm for arm, summary in arms.items()
    }
    return Comparison(
        campaign_id=campaign.id,
        scope_id=scope_id,
        name=name,
        control=control,
        arms=dict(arms),
        report=report,
        host=host,
        factors=factors,
        contrast_arms=tuple(
            arm_of_variant.get(tested.contrast.variant_key)
            for family in bundle.multiple_comparisons.families
            for tested in family.comparisons
        ),
        kind=kind,
    )


def _with_self_judging_disclosed(
    report: Report, arms: Mapping[ArmKey, EvalSummary], factors: tuple[str, ...]
) -> Report:
    """The report with a disclosure, above its contrasts, for each arm whose answers the judge's own model produced.

    The engine's judge-alternate rule catches a judge that is one of the launch's candidate models by its
    label; a quick arm is labelled by its name, and its real model is the one its candidate's
    :class:`~threetears.evals.quick.answer.Answer` names, which only the summary reads. A model tends to rate its
    own output higher, so a contrast on a judged dimension may favour that arm for a reason that is not quality.
    """
    disclosures = [
        DisclosureBlock(
            section="surface",
            source="apparatus",
            text=self_judging_text(summary.judge_shares_candidate_model, f"arm {_label(arm, factors)}")
            + "; a contrast on a judged dimension may favour that arm for that reason alone.",
        )
        for arm, summary in arms.items()
        if summary.judge_shares_candidate_model
    ]
    if not disclosures:
        return report
    blocks = list(report.blocks)
    at = next(
        (i for i, block in enumerate(blocks) if isinstance(block, TableBlock) and block.name == "comparisons"),
        len(blocks),
    )
    return report.model_copy(update={"blocks": [*blocks[:at], *disclosures, *blocks[at:]]})


async def compare(
    cases: Sequence[Mapping[str, Any]],
    candidates: Mapping[str, Candidate | ToolUsingCandidate | WorldCandidate]
    | Mapping[tuple[str, ...], Candidate | ToolUsingCandidate | WorldCandidate],
    scorers: Sequence[Scorer] = (),
    *,
    control: ArmKey,
    scope_id: str,
    expected: ExpectedLabel | None = None,
    judge: Judge | None = None,
    intent: str | None = None,
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
    max_cost_usd: float | None = None,
) -> Comparison:
    """Run each candidate over every case ``k`` times as one arm, test every arm against ``control``, and report.

    Args:
        cases: The cases, each a JSON object, the same for every arm, each named as
            :func:`~threetears.evals.quick.run_eval` names it.
        candidates: The arms. Keyed by name when no ``factors`` are given: each name labels its run and is
            the level its arm is declared at, so it is what :meth:`Comparison.contrasts` calls the arm (its
            ``arm`` key) and what :meth:`Comparison.results` takes. The report names it ``candidate=<name>``:
            the name is stated on the run as the arm lever, every arm at the one candidate model
            :data:`~threetears.evals.quick.one_call.SHARED_ARM_MODEL`. On a ``host`` of the caller's own that
            declares no arm lever (``callable_host(arms=True)`` does), the name is the run's candidate model
            and the report calls it ``model=<name>``, as it does with ``factors=("model",)``.
            With ``factors``, keyed by a tuple of the arm's level of each, in order
            (``("model-a", "v2")``): the ``model`` level is the run's candidate model, and every other
            level is stated on the run as that factor's lever (``callable.<factor>=<level>``).
        scorers: The grades, as :func:`~threetears.evals.quick.run_eval` takes them.
        control: The arm every other arm is tested against, by its key.
        scope_id: The scope every run and the campaign are stored in.
        expected: Declares every candidate a classifier, as :func:`~threetears.evals.quick.run_eval` takes it.
        judge: A model grading every arm's answers on one rubric, as :func:`~threetears.evals.quick.run_eval`
            takes it. ONE judge for every arm — its model, rubric and judge configs — so no difference between
            the arms is a difference in how they were judged, and the report contrasts every arm against the
            control on each judged dimension beside the measures.
        intent: What every case asks, as :func:`~threetears.evals.quick.run_eval` takes it: the intent of the one
            template every arm shares, which a judge reads beside each answer. ``None`` takes the first line of
            the candidates' docstring when every arm's has one and they share it, and a generic sentence
            otherwise. With no judge, nothing that grades an arm reads it: it describes the template, as a
            listing of templates shows.
        host: Where to run and store: ``None`` builds one :func:`~threetears.evals.quick.callable_host` over
            the scorers and the factors other than ``model`` for every arm. A host of the caller's own is held
            to what ``run_eval`` holds it to, and must declare those factors as levers.
        k: Repeats per case, per arm.
        name: The campaign's name, which titles its report; ``None`` names it by its arms, control first, or
            by its factors.
        created_by: Who the campaign and its control are recorded as created by.
        factors: The factors the arms are keyed by, ``model`` among them (``("model", "prompt")``); each is one
            axis of the campaign's declared design. ``None`` keys the arms by name, on the arm lever alone;
            ``("model",)`` keys them by name as models, each name the run's candidate model.
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
        max_cost_usd: The most the whole comparison may spend, in US dollars: each arm's run is capped at an
            equal share, as :func:`~threetears.evals.quick.run_eval` caps one. An arm that reaches its share stops
            ``budget_stopped``, and its contrasts read only the cases it finished, the left-out ones disclosed.
            ``None`` (the default) runs every arm uncapped, which each arm's summary states.

    Returns:
        The comparison: every arm's summary, the campaign's id and its report.

    Raises:
        ValueError: Fewer than two candidates, a blank arm name, an arm key that is not a level of each
            factor, factors without ``model`` or with an unusable or repeated name, a ``control`` that names
            no arm, a ``max_cost_usd`` that is not a positive number, or anything
            :func:`~threetears.evals.quick.run_eval` refuses.
        ValidationFailedError: The launch refused, or the host refuses the campaign's declaration.
    """
    named = _factors(factors)
    if factors is None and (host is None or host.profile.host_sweepables.get(ARM_LEVER) is not None):
        # The arms' names are not models: each is stated as the arm lever's level, every arm at one model.
        named = _NAMED_ARMS
    arms_given: Mapping[ArmKey, Candidate | ToolUsingCandidate | WorldCandidate] = candidates  # type: ignore[assignment]
    _refuse_unusable_arms(arms_given, control, named)
    if max_cost_usd is not None and (
        isinstance(max_cost_usd, bool) or not isinstance(max_cost_usd, int | float) or not max_cost_usd > 0
    ):
        raise ValueError(f"max_cost_usd= is a spend ceiling in US dollars: a positive number, not {max_cost_usd!r}")
    levers = tuple(factor for factor in named if factor not in _UNPREFIXED)
    if host is None:
        host = callable_host(scorers, levers=levers, world=world, arms=named == _NAMED_ARMS)
    coordinates = {arm: _coordinates(arm, named) for arm in arms_given}
    # Every arm in ONE launch, started together, so the arms are measured side by side rather than one after
    # another: what differs between their runs is their settings, not when they ran.
    summaries = await run_arms(
        cases,
        [
            CallableArm(
                candidate,
                model=coordinates[arm].get(CANDIDATE_MODEL_LEVER, SHARED_ARM_MODEL),
                levers={lever: coordinates[arm][lever] for lever in levers} or None,
                arm=coordinates[arm].get(ARM_LEVER),
            )
            for arm, candidate in arms_given.items()
        ],
        scorers,
        scope_id=scope_id,
        expected=expected,
        judge=judge,
        intent=intent,
        host=host,
        k=k,
        tools=tools,
        cassette_mode=cassette_mode,
        cassette_corpus_id=cassette_corpus_id,
        world=world,
        seed=seed,
        goal_checks=goal_checks,
        max_cost_usd=None if max_cost_usd is None else max_cost_usd / len(arms_given),
    )
    arms: dict[ArmKey, EvalSummary] = dict(zip(arms_given, summaries, strict=True))
    if name is None:
        if len(named) == 1:
            name = " vs ".join(_label(arm, named) for arm in [control, *(arm for arm in arms if arm != control)])
        else:
            name = " × ".join(named)
    return _declare(
        host,
        arms,
        named,
        kind=CALLABLE_KIND if judge is None else JUDGED_CALLABLE_KIND,
        control=control,
        scope_id=scope_id,
        name=name,
        behavior="classify" if expected is not None else "score",
        repetitions=k,
        created_by=created_by,
    )


__all__ = ["COMPARE_CREATED_BY", "ArmKey", "Comparison", "compare"]
