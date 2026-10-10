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
from dataclasses import dataclass, field, replace
from typing import Any

from threetears.evals.analysis import (
    AnalysisContextBundle,
    DisclosureBlock,
    Report,
    ReportSection,
    TableBlock,
    assemble_context_bundle,
    build_code_only_report,
    create_campaign,
    get_campaign,
    report_markdown,
    set_campaign_control,
    variant_key_of_run,
)
from threetears.evals.contracts import (
    ACCURACY_MEASURE,
    DEFAULT_LAUNCH_K_RUNS,
    ArmGuardrails,
    CassetteMode,
    DocumentStore,
    GuardrailCheck,
    GuardrailMargin,
    GuardrailReadings,
    utc_now_iso,
)
from threetears.evals.contracts.host import CANDIDATE_MODEL_LEVER, EvalHost
from threetears.evals.contracts.metrics import METRIC_DESCRIPTORS
from threetears.evals.ops.summary import CaseResult, EvalSummary, self_judging_text
from threetears.evals.quick.guardrails import Guardrail
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
    refuse_a_store_beside_a_host,
    refuse_unusable_guardrails,
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
        contrast_measures: The key of the measure each row of the contrasts table reads, row for row — the table
            heads it in words. Empty, or of another length than the table, and :meth:`contrasts` names no key.
        kind: The kind every arm ran — the callable kind, or the judged one — whose levers the campaign's axes name.
        guardrail_readings: Every guardrail, decided for each arm against the control, as the campaign's evidence
            bundle decided it; empty when no reading is a guardrail. :meth:`guardrails` reads it row by row and
            :meth:`guardrail_standing` arm by arm.
        arm_variants: Each arm's variant key, by its key in :attr:`arms`: what the bundle names the arm by.
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
    contrast_measures: tuple[str | None, ...] = field(default=(), repr=False, compare=False)
    kind: str = field(default=CALLABLE_KIND, repr=False, compare=False)
    guardrail_readings: GuardrailReadings = field(default_factory=GuardrailReadings, repr=False, compare=False)
    arm_variants: dict[ArmKey, str] = field(default_factory=dict, repr=False, compare=False)

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
            reading: Only the rows on this reading, by its key (``"accuracy"``) or as the report heads it
                (``"Accuracy"``); ``None`` keeps every row.

        Returns:
            One row per arm and reading, keyed ``arm`` (the arm tested, by its key in :attr:`arms`: the name
            you gave it, or its tuple of levels), ``measure_id`` (the key of the measure read, to cite or filter
            on), ``question`` (in the words it was asked), ``reading`` (the measure as the report heads it),
            ``contrast`` (the arm, as the report names it), ``control`` (the control, as the report names it),
            ``control_mean`` and ``arm_mean`` (over the cases the test read), ``cases`` (how many, paired or not,
            and any one side ran that the test left out), ``delta`` (arm minus control), ``interval`` (on the
            delta, simultaneous over the family), ``hedges_g`` (the standardized effect), ``p_adjusted`` (Holm,
            over the campaign's family) and ``verdict``; empty when
            the report tested nothing.
        """
        rows = [
            row
            for block in self.report.blocks
            if isinstance(block, TableBlock) and block.name == "comparisons"
            for row in block.rows
        ]
        arms = self.contrast_arms if len(self.contrast_arms) == len(rows) else (None,) * len(rows)
        keys = self.contrast_measures if len(self.contrast_measures) == len(rows) else (None,) * len(rows)
        return [
            {"arm": arm, "measure_id": key, **row}
            for row, arm, key in zip(rows, arms, keys, strict=True)
            if reading is None or reading in (key, row["reading"])
        ]

    def guardrails(self, arm: ArmKey | None = None) -> list[dict[str, Any]]:
        """The rows of the report's "Guardrails against the control" table: each guardrail, for each arm.

        A guardrail (``compare(guardrails=...)``) is decided on its own for each arm against the control, never in
        the contrasts: ``held`` when the arm is shown no worse than the control by more than the margin,
        ``breached`` when it is shown worse by more, and ``undecided`` otherwise, which is never safe.

        Args:
            arm: Only this arm's rows, by its key in :attr:`arms`; ``None`` keeps every arm's.

        Returns:
            One row per guardrail and arm, keyed ``arm`` (the arm checked, by its key in :attr:`arms`),
            ``measure_id`` (the guardrail's key: a scorer's name or a judged dimension's), ``outcome`` (``held``,
            ``breached`` or ``undecided``), ``contrast`` (the arm, as the report names it), and the table's other
            columns: ``guardrail`` (as the report heads it), ``control_mean`` and ``arm_mean``, ``cases``,
            ``delta`` (arm minus control), ``interval`` (on the delta), ``margin`` and ``decision`` (the outcome in
            words, with why when undecided); empty when no reading is a guardrail or no control resolved.

        Raises:
            ValueError: ``arm`` names no arm.
        """
        if arm is not None:
            _refuse_an_unknown_control(self.arms, arm, said="arm")
        rows = [
            row
            for block in self.report.blocks
            if isinstance(block, TableBlock) and block.name == "guardrails"
            for row in block.rows
        ]
        checks = self.guardrail_readings.checks
        if len(checks) != len(rows):
            return []
        arm_of_variant = {variant: key for key, variant in self.arm_variants.items()}
        return [
            {
                "arm": arm_of_variant.get(check.contrast.variant_key),
                "measure_id": check.name,
                "outcome": check.decision,
                "contrast": row["arm"],
                **{key: value for key, value in row.items() if key != "arm"},
            }
            for check, row in zip(checks, rows, strict=True)
            if arm is None or arm_of_variant.get(check.contrast.variant_key) == arm
        ]

    def guardrail_standing(self, arm: ArmKey) -> ArmGuardrails:
        """Where one arm stands on every guardrail: breached anywhere, else undecided anywhere, else held.

        An arm with any guardrail ``breached`` is not to be adopted, whatever its contrasts show it gained; one
        ``undecided`` is not known to be safe, which says nothing either way about whether it is.

        Args:
            arm: The arm, by its key in :attr:`arms`.

        Returns:
            The guardrails the arm breached, those it is undecided on, and those it held, each by key; all empty
            for the control, which no guardrail is checked against itself, and when no reading is a guardrail.

        Raises:
            ValueError: ``arm`` names no arm.
        """
        _refuse_an_unknown_control(self.arms, arm, said="arm")
        variant = self.arm_variants.get(arm)
        if variant is None:
            return ArmGuardrails(breached=[], undecided=[], held=[])
        return self.guardrail_readings.of_arm(variant)

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
            guardrail_margins=design.guardrail_margins if design is not None else (),
            measure_latency=design.measure_latency if design is not None else False,
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
    guardrail_margins: Sequence[GuardrailMargin] = (),
    measure_latency: bool = False,
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
    if measure_latency:
        design["measure_latency"] = True
    if guardrail_margins:
        design["guardrail_margins"] = [entry.model_dump() for entry in guardrail_margins]
    filed: dict[str, Any] = {
        "name": name,
        "subject_id": name,
        "behavior": behavior,
        "run_ids": [summary.run_id for summary in arms.values()],
        "declared_design": design,
    }
    # Every arm ran the one template compare authored, so the campaign names it: a judged guardrail's margin is
    # declared on a dimension of its rubric, which the declaration gate reads from it.
    if arms[control].template_id is not None:
        filed["template_id"] = arms[control].template_id
    campaign = create_campaign(
        host.storage,
        filed,
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
    arm_variants = {
        arm: variant
        for arm, summary in arms.items()
        if (variant := variant_key_of_run(list_results(host.storage, summary.run_id, scope_id))) is not None
    }
    arm_of_variant = {variant: arm for arm, variant in arm_variants.items()}
    report = _with_self_judging_disclosed(report, arms, factors)
    report = _with_no_margin_disclosed(report, bundle)
    report = _with_guardrail_standing_disclosed(report, bundle, arm_variants, factors)
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
        contrast_measures=tuple(
            tested.name for family in bundle.multiple_comparisons.families for tested in family.comparisons
        ),
        kind=kind,
        guardrail_readings=bundle.guardrails,
        arm_variants=arm_variants,
    )


def _guardrail_heading(check: GuardrailCheck, bundle: AnalysisContextBundle) -> str:
    """A guardrail as a reader reads it: a measure by its reader name, a judged dimension by its own."""
    descriptor = bundle.measure_catalog.get(check.name) if check.reading == "measure" else None
    return (descriptor.reader_name if descriptor is not None else None) or check.name


def _with_guardrail_standing_disclosed(
    report: Report, bundle: AnalysisContextBundle, arm_variants: Mapping[ArmKey, str], factors: tuple[str, ...]
) -> Report:
    """The report with a line, above its contrasts, for each arm a guardrail leaves not shown safe.

    The contrasts table reads capability, and an arm can improve there while it breaches a guardrail, which the
    table never shows: a reader taking the arm that improved most is told here, where they would pick it, that a
    breached arm is not adopted whatever it gained, and that an undecided guardrail is not known to be safe. A
    report with no contrast (every scorer a guardrail) carries the lines under its guardrails table instead.
    """
    blocks = list(report.blocks)
    contrasts = next(
        (
            i
            for i, block in enumerate(blocks)
            if isinstance(block, TableBlock) and block.name == "comparisons" and block.rows
        ),
        None,
    )
    disclosures = []
    for arm, variant in arm_variants.items():
        said: dict[str, list[str]] = {"breached": [], "undecided": []}
        for check in bundle.guardrails.checks:
            if check.contrast.variant_key == variant and check.decision in said:
                heading = _guardrail_heading(check, bundle)
                if heading not in said[check.decision]:
                    said[check.decision].append(heading)
        said["undecided"] = [name for name in said["undecided"] if name not in said["breached"]]
        breached, undecided = said["breached"], said["undecided"]
        if breached:
            disclosures.append(
                f"Arm {_label(arm, factors)} breached the guardrail{'s' if len(breached) > 1 else ''} "
                f"{', '.join(breached)}: it is shown worse than the control by more than the margin, so it is not "
                + ("adopted, whatever the contrasts below show it gained." if contrasts is not None else "adopted.")
            )
        if undecided:
            disclosures.append(
                f"Arm {_label(arm, factors)} is undecided on the guardrail{'s' if len(undecided) > 1 else ''} "
                f"{', '.join(undecided)}, so it is not known to be safe; the guardrails table says why."
            )
    if not disclosures:
        return report
    section: ReportSection
    if contrasts is not None:
        at, section = contrasts, "surface"
    else:
        at = 1 + max((i for i, block in enumerate(blocks) if block.section == "guardrails"), default=len(blocks) - 1)
        section = "guardrails"
    lines = [DisclosureBlock(section=section, source="guardrails", text=text) for text in disclosures]
    return report.model_copy(update={"blocks": [*blocks[:at], *lines, *blocks[at:]]})


def _with_no_margin_disclosed(report: Report, bundle: AnalysisContextBundle) -> Report:
    """The report with one line, above its contrasts, naming the measures tested with no margin to be equivalent on.

    ``equivalent`` is the only verdict that says two arms are alike, and it needs a margin declared on the measure;
    with none, ``not separated`` is the most a contrast can say, and a reader looking for "good enough" should be
    told why it never appears rather than left to infer it.
    """
    catalog = bundle.measure_catalog
    names = dict.fromkeys(
        tested.name
        for family in bundle.multiple_comparisons.families
        for tested in family.comparisons
        if tested.name in catalog and catalog[tested.name].materiality_threshold is None
    )
    if not names:
        return report
    # The scorers among them with no range: a margin on one is refused unless its range comes with it.
    unranged = [name for name in names if name not in METRIC_DESCRIPTORS]
    headings = ", ".join(catalog[name].reader_name or name for name in names)
    disclosure = DisclosureBlock(
        section="surface",
        source="comparisons",
        text=f"No margin is declared on {headings}, so no contrast on {'it' if len(names) == 1 else 'them'} can read "
        "equivalent, and not separated never means the arms are alike. "
        + (
            "Accuracy takes no margin: grade with a scorer too, and declare one on it with compare(margins=...)."
            if ACCURACY_MEASURE in names
            else "Declare a scorer's margin with compare(margins=...)."
        )
        + (
            " A scorer that returns a number, not a bool, takes its range beside its margin, with ranges=."
            if any(catalog[name].family == "mechanical" and catalog[name].value_range is None for name in unranged)
            else ""
        ),
    )
    blocks = list(report.blocks)
    at = next(
        (i for i, block in enumerate(blocks) if isinstance(block, TableBlock) and block.name == "comparisons"),
        len(blocks),
    )
    return report.model_copy(update={"blocks": [*blocks[:at], disclosure, *blocks[at:]]})


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
    store: DocumentStore | None = None,
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
    margins: Mapping[str, float] | None = None,
    ranges: Mapping[str, tuple[float, float]] | None = None,
    guardrails: Mapping[str, Guardrail] | None = None,
    measure_latency: bool = False,
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
        store: Where the host ``compare`` builds stores every run and the campaign, as
            :func:`~threetears.evals.quick.run_eval` takes it: ``store=SqliteDocumentStore("evals.sqlite")`` keeps
            them in a file, and ``margins=``, ``ranges=`` and everything else this call builds work as without it.
            Never with ``host``.
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
        margins: A margin declared on a scorer's measure, by the scorer's name (``{"correct": 0.05}``): the
            most the arms may differ on it and still be alike. A contrast on it then reads ``equivalent`` when an
            equivalence test shows the difference inside it, the only verdict that says two arms are alike — the
            way "the cheaper model is good enough" is shown. ``None`` declares none and no margin is ever assumed,
            so no contrast can read ``equivalent``, which the report says in one line. A classifier's accuracy is a
            core measure and takes none: grade it with a scorer too, and declare the margin on that. With a
            ``host`` of your own, declare them on it instead (``callable_host(margins=...)``). A margin
            on a scorer that does not return a ``bool`` needs its range in ``ranges``: with no range no
            equivalence test holds its error rate, so it is refused rather than never tested.
        ranges: The lowest and highest score a scorer returning a number can give, by the scorer's name
            (``{"rating": (1, 5)}``). Its intervals stay inside it, a margin on it can be tested, and a score
            outside it excludes the cell, naming the scorer. A scorer annotated ``-> bool`` is a pass/fail on 0
            to 1 already. With a ``host`` of your own, declare them on it instead (``callable_host(ranges=...)``).
        guardrails: The readings no arm may get worse on, by name — a scorer's, or a judge's rubric dimension's —
            each a :class:`~threetears.evals.quick.Guardrail` with its margin and direction
            (``{"no_leak": Guardrail(margin=0.02, direction="higher_is_better")}``). A guardrail joins no contrast
            and no composite: each is decided for each arm against the control, ``held``, ``breached`` or
            ``undecided``, in the report's guardrails table and :meth:`Comparison.guardrails`, and an arm that
            breached one is not adopted whatever it gained. A judged dimension named here is scored on the
            boundary axis, and its margin is declared on the campaign. ``None`` declares none, and nothing is a
            guardrail but a dimension the judge's rubric already puts on the boundary axis. With a ``host`` of your
            own, declare a measure ``guardrail`` on it instead.
        measure_latency: Declare latency under test, as :func:`~threetears.evals.quick.run_eval` takes it: every
            arm runs its cases one at a time and the arms run one after another. ``False`` (the default) runs
            each arm's cases several at once and the arms side by side.

    Returns:
        The comparison: every arm's summary, the campaign's id and its report.

    Raises:
        ValueError: Fewer than two candidates, a blank arm name, an arm key that is not a level of each
            factor, factors without ``model`` or with an unusable or repeated name, a ``control`` that names
            no arm, a ``max_cost_usd`` that is not a positive number, a margin that names no scorer, is not a
            positive number, is on a scorer with no range or comes with a ``host``, a range that is unusable or
            comes with a ``host``, a guardrail that is not a ``Guardrail``, names neither a scorer nor a rubric
            dimension, sits on a scorer given a margin too or comes with a ``host``, a ``store`` with a ``host``, or anything
            :func:`~threetears.evals.quick.run_eval` refuses.
        ValidationFailedError: The launch refused, or the host refuses the campaign's declaration.
    """
    refuse_a_store_beside_a_host(store, host)
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
    if margins and host is not None:
        raise ValueError(
            "margins= declares margins on the host compare builds; a host of your own declares them on its measures "
            "(callable_host(margins=...), or MetricDescriptor.materiality_threshold), so pass one or the other"
        )
    if ranges and host is not None:
        raise ValueError(
            "ranges= declares ranges on the host compare builds; a host of your own declares them on its measures "
            "(callable_host(ranges=...), or MetricDescriptor.value_range), so pass one or the other"
        )
    guardrails = dict(guardrails or {})
    if guardrails and host is not None:
        raise ValueError(
            "guardrails= declares guardrails on the host compare builds; a host of your own declares a measure a "
            "guardrail on it (MetricDescriptor(guardrail=True), its materiality_threshold the margin), so pass one or "
            "the other"
        )
    refuse_unusable_guardrails(scorers, guardrails, margins=margins, ranges=ranges, judge=judge)
    scorer_names = {getattr(scorer, "__name__", None) for scorer in scorers}
    judged_guardrails = {
        judge.dim_name(name): guardrail
        for name, guardrail in guardrails.items()
        if judge is not None and name not in scorer_names
    }
    if judge is not None and judged_guardrails:
        # The dimensions named guardrails are scored on the boundary axis, which the judge stamps on every score.
        judge = replace(
            judge,
            rubric=[
                dim.model_copy(update={"axis": "boundary"}) if dim.name in judged_guardrails else dim
                for dim in judge.dims
            ],
        )
    levers = tuple(factor for factor in named if factor not in _UNPREFIXED)
    if host is None:
        host = callable_host(
            scorers,
            levers=levers,
            world=world,
            arms=named == _NAMED_ARMS,
            margins=margins,
            ranges=ranges,
            store=store,
            guardrails={name: guardrail for name, guardrail in guardrails.items() if name in scorer_names},
        )
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
        measure_latency=measure_latency,
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
        guardrail_margins=[
            GuardrailMargin(dimension=dimension, margin=guardrail.margin)
            for dimension, guardrail in judged_guardrails.items()
        ],
        measure_latency=measure_latency,
    )


__all__ = ["COMPARE_CREATED_BY", "ArmKey", "Comparison", "compare"]
