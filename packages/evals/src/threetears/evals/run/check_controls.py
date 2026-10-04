"""Every goal check proves, where it is written, that it can tell its outcomes apart.

A goal check that gives the same verdict whatever the candidate did measures nothing, and nothing
downstream can tell: it scores a pass rate like any other check. The two ways that has happened are
a check that reads the wrong thing (a call's parameter rather than the world the call produced, so a
candidate that did the forbidden thing by another route passed) and a world read that graded the
seed rather than the end state. Both were found by running and auditing; this makes them refusals
at authoring instead.

**Two controls per check.** Each check is evaluated against:

* **the do-nothing control** — the template's own seed as the end state, with an empty call ledger:
  the candidate did nothing. Derived, so it needs no data and cannot be authored wrong.
* **its named control** — an end state the template's author states in
  :class:`~threetears.evals.contracts.models.GoalCheckControls`: for an ``act`` check, one where the
  behaviour happened; for a ``hold`` check, one where the forbidden thing happened.

An ``act`` check must fail the first and pass the second; a ``hold`` check must pass the first and
fail the second. Equal verdicts are refused — the check does not depend on what the candidate did —
and so are inverted ones, which say the check grades the opposite of what its intent declares.

**Graded by the run's own rule.** Both controls go through
:func:`~threetears.evals.run.runner.grade_goal_checks`, the function a finished cell's checks go
through, so a check is proven under the evaluation it will be scored by.

**Controls are authoring data.** Nothing that runs a cell reads them: the candidate is seeded from
``world_seed``, the simulated user from the ``conversation`` block, and the judge from the intent
and the evidence the kind renders. A control reaching the candidate would be a hint about the answer.

**What the do-nothing control does not model.** It is the seed as written, not as a host's carriers
read it back after a cell: a field the host's read adds with a default (a flag reading ``false``)
is absent here. A check that distinguishes "absent" from "false" can therefore pass this gate and
grade differently in a run; the host's own suite is where that agreement is proven for its
templates. A seed value a run resolves from the subject (a reference to the subject's own state) resolves to empty, as
it does for a run with no subject.

**Which writes it binds** — see :func:`refuse_non_discriminating_checks`. A template written past
authoring (a host's seed) with no controls is not refused where it is read or run; its checks are
shown as unproven wherever a template is rendered, and the first write that authors its checks must
prove them.
"""

from __future__ import annotations

import copy
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from threetears.evals.contracts.dsl import undefined_action
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.host.profile import HostProfile
from threetears.evals.contracts.host.world_schema import schema_violations
from threetears.evals.contracts.models import ControlEndState, EvalTemplate, GoalCheckIntent, GoalStateOutcome
from threetears.evals.contracts.world_state import WorldState, init_world
from threetears.evals.run.runner import GoalCheckUnevaluable, grade_goal_checks

#: The fields whose write puts the proof in question: the checks themselves, the controls, and the
#: seed — which IS the do-nothing control, and the base every named control is laid over.
_PROOF_FIELDS: frozenset[str] = frozenset({"goal_state_checks", "goal_check_controls", "world_seed"})

#: The fields whose write AUTHORS a check's proof. A write touching only the seed of a template that
#: has no controls re-authors no check, so it does not demand controls it never had.
_AUTHORING_FIELDS: frozenset[str] = frozenset({"goal_state_checks", "goal_check_controls"})


@dataclass(frozen=True)
class CheckDiscrimination:
    """One goal check's verdicts on its two controls, and whether they are the ones its intent requires.

    Attributes:
        check: The goal check.
        intent: What the check declares it grades.
        control: The name of its named control end state.
        did_nothing: Its outcome when the candidate did nothing.
        controlled: Its outcome on the named control.
    """

    check: str
    intent: GoalCheckIntent
    control: str
    did_nothing: GoalStateOutcome
    controlled: GoalStateOutcome

    @property
    def discriminates(self) -> bool:
        """Whether both verdicts are the ones the intent requires.

        Returns:
            True when an ``act`` check fails doing nothing and passes its control, or a ``hold``
            check passes doing nothing and fails its control.
        """
        holds_when_idle = self.intent == "hold"
        return self.did_nothing.passed is holds_when_idle and self.controlled.passed is not holds_when_idle

    def refusal(self) -> str | None:
        """The sentence a refusal names this check with, or None when it discriminates.

        Returns:
            None, or one sentence naming the check, its intent and both verdicts.
        """
        if self.discriminates:
            return None
        verdicts = (
            f"goal check {self.check!r} ({self.intent}) {_verdict(self.did_nothing)} when the candidate did nothing "
            f"and {_verdict(self.controlled)} on control {self.control!r}"
        )
        if self.did_nothing.passed is self.controlled.passed:
            return f"{verdicts} — the same verdict on both, so it does not depend on what the candidate did"
        required = (
            "an act check must fail when the candidate did nothing and pass on its control"
            if self.intent == "act"
            else "a hold check must pass when the candidate did nothing and fail on its control"
        )
        return f"{verdicts} — {required}; the check grades the opposite of its intent, or the intent is wrong"


def _verdict(outcome: GoalStateOutcome) -> str:
    """``passes`` or ``fails``, with what the check read.

    Args:
        outcome: One evaluation of a check.

    Returns:
        The verdict and its detail, as a refusal names it.
    """
    return f"{'passes' if outcome.passed else 'fails'} ({outcome.detail})"


def do_nothing_end_state(template: EvalTemplate) -> WorldState:
    """The end state of a candidate that did nothing: the template's seed, and no calls.

    Args:
        template: The template whose seed it is.

    Returns:
        A fresh world. Built anew on each call, so evaluating one control cannot leak into another.
    """
    return init_world(template.world_seed)


def control_end_state(template: EvalTemplate, end_state: ControlEndState) -> WorldState:
    """A named control end state: the seed with the stated keys replaced, and the stated calls recorded.

    The calls are recorded through :meth:`~threetears.evals.contracts.world_state.WorldState.record_call`,
    the method a run records the candidate's calls with, so both ledgers a check can read — the
    cross-tool one and each namespace's own — hold them exactly as a run's would.

    Args:
        template: The template whose seed the control is laid over.
        end_state: The control.

    Returns:
        A fresh world.
    """
    world = do_nothing_end_state(template)
    for namespace, keys in end_state.world.items():
        world.namespace(namespace).update(copy.deepcopy(keys))
    for call in end_state.calls:
        world.record_call(call.tool, call.action, copy.deepcopy(call.params))
    return world


def check_discriminations(template: EvalTemplate, *, profile: HostProfile) -> list[CheckDiscrimination]:
    """Evaluate each controlled goal check against the do-nothing control and its named control.

    Args:
        template: A template carrying ``goal_check_controls``.
        profile: The host, whose world a check's paths are read through.

    Returns:
        One entry per controlled check, in the controls' order; empty for a template without controls.

    Raises:
        GoalCheckUnevaluable: A check raised while being evaluated against a control.
    """
    controls = template.goal_check_controls
    if controls is None:
        return []
    results: list[CheckDiscrimination] = []
    for entry in controls.checks:
        end_state = controls.end_states[entry.control]
        (idle,) = grade_goal_checks(
            [entry.check],
            world_state=do_nothing_end_state(template),
            variation=end_state.variation,
            world=profile.world,
        )
        (acted,) = grade_goal_checks(
            [entry.check],
            world_state=control_end_state(template, end_state),
            variation=end_state.variation,
            world=profile.world,
        )
        results.append(
            CheckDiscrimination(
                check=entry.check, intent=entry.intent, control=entry.control, did_nothing=idle, controlled=acted
            )
        )
    return results


def refuse_non_discriminating_checks(
    template: EvalTemplate, *, profile: HostProfile, authored: Collection[str] | None = None
) -> None:
    """Refuse a template whose goal checks are not each proven to tell their outcomes apart.

    **Which writes it binds.** A create (``authored`` None) is checked whole: every goal check it
    declares must carry a control, and every control must discriminate. An update is checked when it
    writes ``goal_state_checks``, ``goal_check_controls`` or ``world_seed``, and only then — the
    rule every authoring guard here follows, so a stored template stays editable in its other fields.
    An update that writes checks or controls must prove every check the merged template declares,
    including one a host seeded without controls; one that writes only the seed re-proves the
    controls a template has (the seed is the do-nothing control) and demands none from a template
    that never had them, since it authors no check.

    **Not where a template is read, listed or launched.** A template a host seeded without controls
    keeps loading and running. Refusing it there would take the catalogue down with it; what it
    gets instead is a visible "unproven" wherever a template is rendered.

    Args:
        template: The template being written, merged over the stored one for an update.
        profile: The host whose world and tools a control end state is held to.
        authored: The field names this write authors, or None for a create.

    Raises:
        ValidationFailedError: A declared check has no control; a control names a check the template
            does not declare; a control end state names world state or a call this host does not
            define, or a value its dimension's schema refuses; a check cannot be evaluated against a
            control; or a check gives the same verdict on both controls, or the verdicts its intent
            forbids. Every defect is named at once.
    """
    if authored is not None and not (_PROOF_FIELDS & set(authored)):
        return
    controls = template.goal_check_controls
    if controls is None:
        if not template.goal_state_checks:
            return
        if authored is not None and not (_AUTHORING_FIELDS & set(authored)):
            return
        raise ValidationFailedError(
            f"template {template.name!r} declares goal checks without goal_check_controls: each check must state "
            "its intent ('act' or 'hold') and name a control end state that proves it discriminates — "
            "an act check must fail when the candidate did nothing and pass on its control, a hold check must pass "
            "when the candidate did nothing and fail on its control. Unproven: "
            + "; ".join(repr(check) for check in template.goal_state_checks)
        )

    controlled = {entry.check for entry in controls.checks}
    declared = set(template.goal_state_checks)
    defects = [
        f"goal check {check!r} has no control" for check in template.goal_state_checks if check not in controlled
    ]
    defects += [
        f"goal_check_controls names {entry.check!r}, which is not one of this template's goal checks"
        for entry in controls.checks
        if entry.check not in declared
    ]
    for name, end_state in controls.end_states.items():
        defects += [f"control {name!r}: {defect}" for defect in _end_state_defects(end_state, profile)]
    if defects:
        raise ValidationFailedError(
            f"template {template.name!r} has goal_check_controls that do not hold: " + "; ".join(defects)
        )

    try:
        discriminations = check_discriminations(template, profile=profile)
    except GoalCheckUnevaluable as unevaluable:
        raise ValidationFailedError(
            f"template {template.name!r}: {unevaluable} against a control end state — "
            "a check that cannot be evaluated is not one a run can grade"
        ) from unevaluable
    refusals = [refusal for discrimination in discriminations if (refusal := discrimination.refusal()) is not None]
    if refusals:
        raise ValidationFailedError(
            f"template {template.name!r} has goal checks that do not discriminate: "
            + "; ".join(refusals)
            + " — correct the check so it reads what the behaviour changes, correct its intent, or correct the control"
        )


def _end_state_defects(end_state: ControlEndState, profile: HostProfile) -> list[str]:
    """What a control end state states that this host's world or tools could not hold.

    A control is evidence only if it is a state a run could leave: a key no dimension declares, a
    value its dimension's schema refuses, or a call to an action the host does not define would let
    a check pass or fail its control for a reason no run reproduces. Asked of the host's profile,
    as the world gate asks of a goal check. A key names its dimension through the host's addressing
    (:meth:`~threetears.evals.contracts.host.world.WorldRegistry.address`), as a seed key does. A host
    that declares no world, or cannot list a tool's actions, leaves that part unchecked rather than
    refused; a tool the host does not have is refused, since its reader answers that with an empty
    action set (:func:`~threetears.evals.contracts.dsl.undefined_action`, the rule a goal check's own
    call references are held to).

    Args:
        end_state: The control.
        profile: The host whose world and tools the control is held to.

    Returns:
        One sentence per defect.
    """
    world = profile.world
    defects: list[str] = []
    # A host with no world has no vocabulary to hold a control's keys to, so they go unchecked.
    if world is not None:
        for namespace, keys in end_state.world.items():
            for key, value in keys.items():
                # Named through the host's addressing, the one a seed key and a goal check's path are
                # read by — so a control states exactly the dimension the check will find it under.
                name = world.address(namespace, key)
                dimension = world.get(name)
                if dimension is None:
                    defects.append(f"{namespace}.{key} addresses no dimension this host's world declares")
                elif dimension.carrier != namespace:
                    defects.append(
                        f"{namespace}.{key} puts {name!r} under {namespace!r}, but {dimension.carrier!r} supplies "
                        "it — a check reads it there, so this value would never be seen"
                    )
                else:
                    defects.extend(schema_violations(dimension.schema, value, at=name))
    for call in end_state.calls:
        if (undefined := undefined_action(call.tool, call.action, profile.tool_actions)) is not None:
            defects.append(undefined)
            continue
        schema = profile.action_parameters(call.tool, call.action) if profile.action_parameters is not None else None
        properties = schema.get("properties") if isinstance(schema, Mapping) else None
        if isinstance(properties, Mapping):
            defects.extend(
                f"{call.tool}.{call.action} has no parameter {param!r} (its parameters: {', '.join(sorted(properties))})"
                for param in call.params
                if param not in properties
            )
    return defects


__all__ = [
    "CheckDiscrimination",
    "check_discriminations",
    "control_end_state",
    "do_nothing_end_state",
    "refuse_non_discriminating_checks",
]
