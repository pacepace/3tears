"""Every goal check proves, where it is written, that it can tell its outcomes apart.

A goal check that gives the same verdict whatever the candidate did measures nothing, and nothing
downstream can tell: it scores a pass rate like any other check. The two ways that has happened are
a check that reads the wrong thing (a call's parameter rather than the world the call produced, so a
candidate that did the forbidden thing by another route passed) and a world read that graded the
seed rather than the end state. Both were found by running and auditing; this makes them refusals
at authoring instead.

**Two controls per check.** Each check is evaluated against:

* **the do-nothing control** — what a cell whose candidate did nothing would leave: the template's own seed as the
  end state, named through the host's world registry, an empty call ledger, and whatever the world does on its own. A
  triggered dimension's seed arms it rather than setting it, so an armed ``event`` or ``human`` dimension is *known
  absent* — present, holding ``None`` rather than its seeded value — since its condition is the candidate's act or a
  person's, and neither happened. Known absent and not :data:`~threetears.evals.contracts.dsl.Missing`: the gate knows
  the value never arrived, so a hold check such as ``not state.payment_hold == "held"`` passes here, as it does in a
  run whose host reads an unfired dimension back as ``None``. A ``turn`` dimension is the exception, because its
  condition is the passage of turns, which happens in a cell whatever its candidate does. The session does not advance
  a clock itself — :meth:`~threetears.evals.contracts.world_session.WorldSession.at_turn` only applies ambient
  perturbation — so a turn trigger fires when the kind fires it as its turns pass
  (:meth:`~threetears.evals.contracts.world_session.WorldSession.fire`), or when the world's own clock fires it and
  the kind records it (``observe``). Whether a given kind fires its turn triggers is the kind's code, which this gate
  cannot see, so it assumes the worst case for a do-nothing candidate: every clock-driven dimension FIRES in the
  do-nothing control — the world's own clock may fire one the seed never armed, and the engine cannot rule that out —
  and one the seed arms fires as the seed's armed event (``fired_armed``), with its seeded value arrived in the end
  state. That over-refuses rather than over-admits: a check on a clock dimension that a particular kind never fires is
  still refused. A check that passes on a clock firing alone passes for a candidate that did nothing, and is refused.
  Derived, so it needs no data and cannot be authored wrong.
* **its named control** — an end state the template's author states in
  :class:`~threetears.evals.contracts.models.GoalCheckControls`, laid over the do-nothing control: for an
  ``act`` check, one where the behaviour happened; for a ``hold`` check, one where the forbidden thing
  happened. What the world does on its own happens there too, so its firings are the do-nothing
  control's plus the ones the control states.

An ``act`` check must fail the first and pass the second; a ``hold`` check must pass the first and
fail the second. Equal verdicts are refused — the check does not depend on what the candidate did —
and so are inverted ones, which say the check grades the opposite of what its intent declares.

**Graded by the run's own rule.** Both controls go through
:func:`~threetears.evals.run.runner.grade_goal_checks`, the function a finished cell's checks go
through, so a check is proven under the evaluation it will be scored by.

**Controls are authoring data.** Nothing that runs a cell reads them: the candidate is seeded from
``world_seed``, the simulated user from the ``conversation`` block, and the judge from the intent
and the evidence the kind renders. A control reaching the candidate would be a hint about the answer.

**What the do-nothing control does not model.** It is the seed as written, not as a host's carriers read it back after
a cell: a field the host's read adds with a default (a flag reading ``false``) is absent here — a check reading it is
*not established* on this control, and refused naming the path — and so is the value a clock-driven dimension takes
when the world's own clock fires it unarmed — the firing is modelled, the value it brings is the host's. An armed
event or human dimension that never fired is ``None`` here, whatever a host's read gives back for one — a host whose
read answers ``"released"`` for an unfired hold grades ``state.payment_hold == None`` differently in a run. A check
that distinguishes "absent" from "false" can therefore pass this gate and grade differently in a run; the host's own
suite is where that agreement is proven for its templates. A seed value a run resolves from the subject (a reference
to the subject's own state) resolves to empty, as it does for a run with no subject.

**Which writes it binds** — see :func:`refuse_non_discriminating_checks`. A template written past
authoring (saved straight to the store) with no controls is not refused where it is read or run; its checks are
shown as unproven wherever a template is rendered, and the first write that authors its checks must
prove them.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from threetears.evals.contracts.call_ledger import CallLedger
from threetears.evals.contracts.dsl import NOT_ESTABLISHED, undefined_action, undefined_fired_dimension
from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.host.profile import HostProfile
from threetears.evals.contracts.host.world import Triggered, WorldRegistry
from threetears.evals.contracts.host.world_schema import schema_violations
from threetears.evals.contracts.models import ControlEndState, EvalTemplate, GoalCheckIntent, GoalStateOutcome
from threetears.evals.contracts.world_events import Firings
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
        unestablished = [
            name
            for name, outcome in (
                ("the do-nothing control", self.did_nothing),
                (f"control {self.control!r}", self.controlled),
            )
            if outcome.detail.startswith(NOT_ESTABLISHED)
        ]
        if unestablished:
            # Not "the same verdict on both": a check not established on a control was not decided by it at
            # all, so the diagnosis is the path the control holds nothing at, never the check's dependence.
            return (
                f"{verdicts} — {' and '.join(unestablished)} hold(s) nothing at a path the check reads, so the check "
                "cannot be proven against it: the do-nothing control is the template's seed (an armed event or human "
                "dimension known absent, None) with no value a host's read would add by default, and a named control "
                "is that seed with what it states laid over it. Seed or state the path the check reads, or read a "
                'triggered dimension\'s arrival through fired("<dimension>") or fired_armed("<dimension>")'
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


class ControlEnd(NamedTuple):
    """What a control states the candidate left behind: the world, and the calls it made."""

    #: The world, keyed by declared dimension name — the shape a goal check reads ``state.<dimension>`` from.
    end_state: dict[str, Any]
    #: The calls, recorded through :meth:`~threetears.evals.contracts.call_ledger.CallLedger.record`,
    #: the method a kind records its candidate's calls with.
    ledger: CallLedger
    #: What fired — read by ``fired()`` and ``fired_armed()``. The do-nothing control's are the world's
    #: clock-driven dimensions (:func:`do_nothing_end_state`).
    fired: Firings


def _named(world: WorldRegistry | None, namespaces: Mapping[str, Any]) -> dict[str, Any]:
    """A seed-shaped world, keyed by dimension name through the host's registry.

    Args:
        world: The host's world registry, or None for a host that declares no world.
        namespaces: Carrier → (key → value).

    Returns:
        Dimension name → value. Empty for a worldless host stating no world.

    Raises:
        ValueError: The world states something the host's registry does not declare — including
            any world state at all on a host that declares no world.
    """
    if world is not None:
        return world.named(namespaces)
    if namespaces:
        raise ValueError("this host declares no world, so world state stated for it names nothing a check can read")
    return {}


def do_nothing_end_state(template: EvalTemplate, *, world: WorldRegistry | None) -> ControlEnd:
    """The end state of a candidate that did nothing: the template's seed, no calls, and what the world does alone.

    The seed with each of its ``event`` and ``human`` triggered dimensions *known absent* (``None``): seeding
    one arms it, and with nothing done its condition — the candidate's act, or a person's — never happened,
    so its seeded value is not in the world. Known absent rather than left out, because left out it reads
    as :data:`~threetears.evals.contracts.dsl.Missing` — unknown — and a hold check over it
    (``not state.payment_hold == "held"``) could then never pass here, though the gate knows the hold never
    arrived. A ``turn`` dimension is different: its condition is turns passing, which happens in every cell
    whatever the candidate does, and whether a kind fires it then is the kind's code, not something this
    gate can see — so it is taken as firing. Every declared ``turn`` dimension fires, since the world's own
    clock may fire one the seed never armed; one the seed arms fires as the seed's armed event, and its
    seeded value is in the end state.

    Args:
        template: The template whose seed it is.
        world: The host's world registry, which names each seeded value's dimension.

    Returns:
        A fresh end state. Built anew on each call, so evaluating one control cannot leak into another.

    Raises:
        ValueError: The seed states something the registry does not declare.
    """
    seeded = _named(world, template.world_seed.namespaces)
    clock = _clock_driven(world)
    # A triggered dimension's seed ARMS it rather than setting it: a candidate that did nothing never met an event's
    # or a person's condition, so the value never arrived — known absent, ``None``. Naming its seeded value here would
    # grade a check on such a dimension's end state as already satisfied by the seed — exactly the "graded the seed,
    # not the end state" defect this gate exists to refuse. A clock-driven one is the opposite case: its condition is
    # turns passing, so leaving it out would let a check on it pass this gate and then pass, in every cell, for a
    # candidate that did nothing — the same defect from the far side.
    idle = {name: value if not _is_triggered(world, name) or name in clock else None for name, value in seeded.items()}
    return ControlEnd(
        end_state=idle,
        ledger=CallLedger(),
        fired=Firings(dimensions=clock, armed=clock & frozenset(seeded)),
    )


def _is_triggered(world: WorldRegistry | None, name: str) -> bool:
    """Whether ``name`` is a dimension that arrives on a condition rather than at t=0.

    Args:
        world: The host's world registry, or None for a host that declares no world.
        name: A declared dimension name.

    Returns:
        Whether its declaration is triggered.
    """
    declared = world.get(name) if world is not None else None
    return declared is not None and isinstance(declared.when, Triggered)


def _clock_driven(world: WorldRegistry | None) -> frozenset[str]:
    """Every dimension whose trigger is the passage of turns — what fires in a cell whatever its candidate does.

    Args:
        world: The host's world registry, or None for a host that declares no world.

    Returns:
        The names of every declared ``turn``-triggered dimension.
    """
    if world is None:
        return frozenset()
    return frozenset(
        declared.name
        for declared in world.declarations
        if isinstance(declared.when, Triggered) and declared.when.kind == "turn"
    )


def control_end_state(template: EvalTemplate, end_state: ControlEndState, *, world: WorldRegistry | None) -> ControlEnd:
    """A named control end state: the seed with the stated dimensions replaced, the stated calls, and what fired.

    Laid over the do-nothing control throughout, since what the world does alone happens in that cell
    too: its end state, and its firings plus the ones the control states (``fired``, ``fired_armed``).

    Args:
        template: The template whose seed the control is laid over.
        end_state: The control.
        world: The host's world registry, which names each stated value's dimension.

    Returns:
        A fresh end state.

    Raises:
        ValueError: The seed or the control states something the registry does not declare.
    """
    idle = do_nothing_end_state(template, world=world)
    named = {**idle.end_state, **_named(world, end_state.world)}
    ledger = CallLedger()
    for call in end_state.calls:
        ledger.record(call.tool, call.action, call.params)
    armed = idle.fired.armed | frozenset(end_state.fired_armed)
    fired = Firings(dimensions=idle.fired.dimensions | frozenset(end_state.fired) | armed, armed=armed)
    return ControlEnd(end_state=named, ledger=ledger, fired=fired)


def check_discriminations(template: EvalTemplate, *, profile: HostProfile) -> list[CheckDiscrimination]:
    """Evaluate each controlled goal check against the do-nothing control and its named control.

    Args:
        template: A template carrying ``goal_check_controls``.
        profile: The host, whose world a check's paths are read through.

    Returns:
        One entry per controlled check, in the controls' order; empty for a template without controls.

    Raises:
        GoalCheckUnevaluable: A check raised while being evaluated against a control.
        ValueError: The seed or a control states world state the host's registry does not declare.
    """
    controls = template.goal_check_controls
    if controls is None:
        return []
    results: list[CheckDiscrimination] = []
    for entry in controls.checks:
        end_state = controls.end_states[entry.control]
        nothing = do_nothing_end_state(template, world=profile.world)
        (idle,) = grade_goal_checks(
            [entry.check],
            ledger=nothing.ledger,
            end_state=nothing.end_state,
            fired=nothing.fired,
            variation=end_state.variation,
            world=profile.world,
        )
        stated = control_end_state(template, end_state, world=profile.world)
        (acted,) = grade_goal_checks(
            [entry.check],
            ledger=stated.ledger,
            end_state=stated.end_state,
            fired=stated.fired,
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
    including one written past authoring without controls; one that writes only the seed re-proves the
    controls a template has (the seed is the do-nothing control) and demands none from a template
    that never had them, since it authors no check.

    **Not where a template is read, listed or launched.** A template written past authoring without controls
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
    try:
        armed = _armed_by_seed(template, profile.world)
    except ValueError as unnamed:
        raise ValidationFailedError(
            f"template {template.name!r}: its world_seed states world state a control cannot be built over: {unnamed}"
        ) from unnamed
    for name, end_state in controls.end_states.items():
        defects += [f"control {name!r}: {defect}" for defect in _end_state_defects(end_state, profile, armed=armed)]
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
    except ValueError as unnamed:
        raise ValidationFailedError(
            f"template {template.name!r}: its world_seed states world state a control cannot be built over: {unnamed}"
        ) from unnamed
    refusals = [refusal for discrimination in discriminations if (refusal := discrimination.refusal()) is not None]
    if refusals:
        raise ValidationFailedError(
            f"template {template.name!r} has goal checks that do not discriminate: "
            + "; ".join(refusals)
            + " — correct the check so it reads what the behaviour changes, correct its intent, or correct the control"
        )


def _armed_by_seed(template: EvalTemplate, world: WorldRegistry | None) -> frozenset[str]:
    """The triggered dimensions the template's seed arms.

    Args:
        template: The template.
        world: The host's world registry, or None for a host that declares no world.

    Returns:
        The names.

    Raises:
        ValueError: The seed states something the registry does not declare.
    """
    seeded = _named(world, template.world_seed.namespaces)
    return frozenset(name for name in seeded if _is_triggered(world, name))


def _end_state_defects(end_state: ControlEndState, profile: HostProfile, *, armed: frozenset[str]) -> list[str]:
    """What a control end state states that this host's world or tools could not hold.

    A control is evidence only if it is a state a run could leave: a key no dimension declares, a
    value its dimension's schema refuses, a fired dimension that is not a triggered one the host
    declares (:func:`~threetears.evals.contracts.dsl.undefined_fired_dimension`), a seed-armed firing
    under a template whose seed arms no event at all, or a call to an
    action the host does not define would let a check pass or fail its control for a reason no run
    reproduces. Asked of the host's profile,
    as the world gate asks of a goal check. A key names its dimension through the host's addressing
    (:meth:`~threetears.evals.contracts.host.world.WorldRegistry.address`), as a seed key does. A host
    that declares no world refuses any world state a control states, since a goal check can only
    read a declared dimension and there is none. A host that cannot list a tool's actions leaves
    that part unchecked rather than refused; a tool the host does not have is refused, since its reader answers that with an empty
    action set (:func:`~threetears.evals.contracts.dsl.undefined_action`, the rule a goal check's own
    call references are held to).

    Args:
        end_state: The control.
        profile: The host whose world and tools the control is held to.
        armed: The triggered dimensions the template's seed arms.

    Returns:
        One sentence per defect.
    """
    world = profile.world
    defects: list[str] = []
    # A host with no world has no dimension a check could read a control's keys as.
    stated = [f"{namespace}.{key}" for namespace, keys in end_state.world.items() for key in keys]
    if world is None and stated:
        defects.append(
            "it states world state ("
            + ", ".join(stated)
            + "), and this host declares no world — a goal check reads only declared dimensions, so none of it "
            "could ever be read"
        )
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
    # Held to the rule a goal check's own fired() references are held to, so a control cannot prove a
    # check over a dimension that can never fire.
    stated = [*end_state.fired, *(name for name in end_state.fired_armed if name not in end_state.fired)]
    defects.extend(undefined for name in stated if (undefined := undefined_fired_dimension(name, world)) is not None)
    # The session's rule, not a second one (WorldSession.observe): a firing is armed when its EVENT is one
    # the seed armed, on the dimension the seed armed it on or on another that event also moves. So a seed-armed
    # firing on a dimension the seed did not arm is a state a run can leave — what no run can leave is one
    # under a seed that armed no event at all.
    if not armed:
        defects.extend(
            f"fired_armed names {name!r}, and this template's seed arms no event — a firing of the seed's armed "
            "event is a state no run could leave when the seed armed none"
            for name in end_state.fired_armed
            if undefined_fired_dimension(name, world) is None
        )
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
    "ControlEnd",
    "check_discriminations",
    "control_end_state",
    "do_nothing_end_state",
    "refuse_non_discriminating_checks",
]
