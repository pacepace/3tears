"""Lever and configuration resolution: each run's effective value for every lever the host declares.

:func:`_effective_config` resolves a run's levers through the host profile, keeping which of them the launch
engaged (:func:`_resolve_config`), and :func:`_lever_value` reads one lever's level off a run. The rest of the
bundle's derivations read levers only through here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal


from threetears.evals.analysis.reporting import (
    ScoreRecord,
    lever_level,
)
from threetears.evals.kernel.host.profile import CANDIDATE_MODEL_LEVER, HostProfile

# At runtime for its field set, which tells a result-level measure from a row-level one.
from threetears.evals.schema.models import EvalResult


if TYPE_CHECKING:  # runtime models — TYPE_CHECKING-only to keep the runtime import graph minimal.
    from threetears.evals.schema.models import EvalRun


#: The one name the candidate model answers to, everywhere. The coverage map, the divergence
#: lens and the effective configuration all reach for this axis, and two of them holding two
#: names for it would let the bundle report a lever twice — or, worse, report coverage for one
#: name while the config names the other, which is a bundle disagreeing with itself in one
#: payload.
#:
#: It is the one lever :func:`_lever_value` reads off the OBSERVATION rather than off the run,
#: because it is a declared coordinate of one (``ScoreRecord.model``): every record knows which
#: model produced it, including a record whose candidate role left no usage row for the run-level
#: resolution to recover a model from. **The name's shape has nothing to do with it** — that rule, dotted through
#: the effective config and plain off the record, is what bound every plain-named host lever to
#: ``'—'``, which is why it was retired.
#:
#: Declared in :mod:`threetears.evals.kernel.host.sweepables`, beside the core declaration that carries it,
#: so the literal has one owner; :mod:`threetears.evals.kernel.host.profile` re-exports it, and that is
#: where a host recovery rule claiming the name is refused. Read here.
_CANDIDATE_MODEL_LEVER = CANDIDATE_MODEL_LEVER


def _observed_model_levers(profile: HostProfile) -> dict[str, str]:
    """Levers whose *inherited* value is recoverable from what the run actually did.

    Maps a lever name to the :class:`RoleUsage` role whose ``model`` records it. A run that
    never exercised the role simply does not carry the lever — "the role never ran" and "the
    role ran at an unknown value" are different answers.

    The candidate entry is the engine's and is here for the same reason a host's is, arrived at
    from the other side. The candidate model was a lever only where a campaign held more than
    one, so a single-arm campaign's configuration named an INNER-AGENT model and stayed silent
    about the model that actually produced its numbers. A generator handed those two side by
    side — one in a slot labelled "the config", the other in a slot naming the candidate model — read the
    pair as one value contradicting itself and reported a config-provenance defect that did not
    exist. They never disagreed; they are different roles, and nothing in the bundle said so.

    Everything else comes from the host's profile, because the lever names are its tool
    vocabulary: hardcoding one here would attribute a second consumer's inner agent to the first
    consumer's tool, silently, in the module a paid generator reads.

    Built per call from the profile handed in rather than at import, because two hosts can
    assemble bundles in one process.

    Returns:
        The engine's rule merged with the host's. A host declaration colliding with the engine's
        reserved name cannot reach here — :class:`~threetears.evals.kernel.host.profile.HostProfile`
        refuses it at registration.
    """
    return {**profile.observed_model_levers, _CANDIDATE_MODEL_LEVER: "candidate"}


#: The level a run sits at for a lever it did not override and that nothing recovers — "ran
#: at the subject's own setting". A real cohort every un-overriding run shares, NOT a stand-in
#: for missing data: a lever whose value IS recoverable never lands here, and one that applied
#: but resolved ambiguously is left out of the lever entirely rather than pooled into it.
_INHERITED_DEFAULT_LEVEL = "—"


@dataclass(frozen=True)
class EffectiveLever:
    """One lever's value for one run, with how that value was established.

    Three states, never collapsed to two. ``overridden`` means the value was NAMED rather than
    recovered, and it has three sources: a launch that named it, an open family that RESTATES a
    fixed lever — the restatement IS the naming, which is the rule the one-vocabulary rule adds
    and the one most likely to be re-litigated — and a fixed declaration's own reader, which is
    why a campaign that swept nothing still shows every applicable lever stamped this way. A value recovered from what the run observably
    did is ``inherited``; a lever that applied but whose value no record pins is ``unknown`` with
    ``value=None``.

    The distinction is load-bearing for cohort assignment. Reading absence as a
    level of its own puts a run that INHERITED a value and a run that explicitly
    SET the same value into different cohorts, and reports a lever as moved when
    nothing moved — the control arm of any sweep inherits, so it is exactly the
    arm that gets mislabelled.
    """

    value: str | None
    provenance: Literal["overridden", "inherited", "unknown"]


def _observed_models(results: list[EvalResult], role: str) -> set[str]:
    """Every distinct model the given usage role spent tokens on across ``results``."""
    return {usage.model for result in results for usage in result.usage if usage.role == role and usage.model}


def _effective_config(run: EvalRun, results: list[EvalResult], *, profile: HostProfile) -> dict[str, EffectiveLever]:
    """Resolve a run's levers to the values it actually ran at, with provenance.

    **Read through the host's registry, never off one host's carriers.** The registry is the
    single lever vocabulary, so this asks
    :meth:`~threetears.evals.kernel.host.sweepables.SweepableRegistry.resolve_levers` what this run's
    levers are called and what it carried under them. While it read one host's overlay fields by
    hand, a host whose levers live anywhere else got ``coverage == []`` and an empty
    ``RunSummary.config``, and a real model reading that bundle correctly refused to attach
    outcomes to levels.

    Three sources, in this order, and the order is the provenance:

    1. **An open family's members** — the levers the launch NAMED. A kind overlay
       ``house_rules={'flanking': 'on'}`` resolves the member ``gm.house_rules.flanking``, which is
       the lever's declared name rather than a carrier path this function flattened for itself.
       A member named as ``null`` is the level :data:`~threetears.evals.analysis.reporting.NULL_LEVEL`,
       ``overridden`` — unless the lever has a recovery rule, which reads a null as "not stated"
       and resolves it in the second pass instead.
    2. **Recovery from observation** — :func:`_observed_model_levers` maps a lever to the usage
       role whose ``model`` is the value that ran. Two or more distinct models under one role
       make the value genuinely ambiguous, which is ``unknown`` rather than a guess at the first.
       A lever with a recovery rule is resolved by it alone: where the role never ran, the lever
       does not apply to this run.
    3. **The declaration's own reader** — a host lever carried on the run record with no overlay
       carrier and no inheritance tier, which is what the toy host's ``chunk_tokens`` is.

    **Recovered means ``inherited``, and that is a statement about how the value was
    established here, not about whether the run chose it.** The candidate model is the
    case that makes the distinction visible: a launch names its candidate model on
    ``EvalRun.candidate_model``, yet what this function reports is what the ``candidate`` role
    observably spent tokens on — the model that produced the numbers, which a provider's
    routing can make something other than the one declared. ``overridden`` means the value was
    NAMED rather than recovered, and it has three sources — a launch that named it, an open
    family that RESTATES a fixed lever, and a fixed declaration's own reader (source 3 above),
    which is why a campaign that swept nothing still shows every applicable lever stamped this
    way. This sentence said ``overridden`` was "reserved for the overlays above, the launch's
    departure from the subject's own configuration"; that was the third site of one correction
    already applied to :class:`EffectiveLever` and to ``RunIndexEntry.config_provenance``, and
    :func:`_resolve_config` twelve lines below has stated the rule correctly the whole time — a
    reader who believed this one would have concluded a lever stamped ``overridden`` was a
    departure when it may simply be what the declaration reads.

    Args:
        run: The run to read overlays from.
        results: That run's results — the observation side of the resolution.
        profile: The host whose vocabulary this reads.

    Returns:
        Each applicable lever mapped to its :class:`EffectiveLever`.
    """
    return _resolve_config(run, results, profile=profile)[0]


def _resolve_config(
    run: EvalRun, results: list[EvalResult], *, profile: HostProfile
) -> tuple[dict[str, EffectiveLever], frozenset[str]]:
    """:func:`_effective_config`, keeping the ENGAGED set its first two passes already know.

    The coverage map needs both: the resolved levers, and which of them the launch named or
    observation recovered — that pair is what :func:`_reportable_levers` decides a row on, and
    provenance cannot supply the second half, since ``overridden`` is written both by a family
    member and by a fixed declaration's own reader.

    Returned rather than re-derived at the call site. ``SweepableRegistry._resolve`` is one pass
    precisely so two surfaces cannot get different answers out of a reader that is not perfectly
    pure, and a caller re-asking reopens that window one layer up — on top of paying for every
    declaration's reader a second time, per run, for a lens that already had the answer.

    Args:
        run: The run to read.
        results: That run's results.
        profile: The host whose vocabulary this reads.

    Returns:
        ``(each applicable lever's EffectiveLever, the levers this run ENGAGED)``.
    """
    resolution = profile.sweepables.resolve_levers(run, results)
    recovery = _observed_model_levers(profile=profile)
    flat: dict[str, EffectiveLever] = {}
    for lever in sorted(resolution.overlaid):
        value = resolution.values.get(lever)
        if value is None and lever in recovery:
            # A recovery rule gives ``null`` its own meaning — "not stated, read it off what ran" —
            # so the second pass resolves it, to ``inherited`` or ``unknown``, never to a level.
            continue
        # Anywhere else a NAMED null is a level the operator set (``NULL_LEVEL``). Skipping it, as
        # this once did, dropped the lever from the config, its provenance and the coverage map
        # together, so a sweep between a value and ``null`` read ``unswept`` (#574).
        flat[lever] = EffectiveLever(lever_level(value), "overridden")
    for lever, role in recovery.items():
        if lever in flat:
            continue
        observed = _observed_models(results, role)
        if not observed:
            # The role never ran, so the lever does not apply to this run at all.
            continue
        one = next(iter(observed)) if len(observed) == 1 else None
        flat[lever] = EffectiveLever(one, "inherited" if one else "unknown")
    for lever, value in resolution.values.items():
        # A lever the host declared a recovery rule for is resolved by that rule ALONE: reaching
        # its declaration's reader here would report the run-level projection (the candidate
        # model lever reads the run's whole model LIST) as one observation's level, and a lever
        # whose role never ran genuinely does not apply to the run rather than sitting at
        # whatever the record happens to hold. ``None`` here is a declaration's reader finding
        # nothing on this run — the lever does not apply — unlike a launch-named null, which the
        # first pass has already placed.
        if lever in flat or lever in recovery or value is None:
            continue
        flat[lever] = EffectiveLever(lever_level(value), "overridden")
    return flat, frozenset(resolution.overlaid | (set(recovery) & set(flat)))


def _effective_values(run: EvalRun, results: list[EvalResult], *, profile: HostProfile) -> dict[str, str]:
    """Each lever whose value IS established, as plain strings.

    Levers resolving to ``unknown`` are absent rather than present-and-empty: a
    caller comparing cohorts must be unable to accidentally treat "we could not
    establish this" as a level.
    """
    return {
        lever: eff.value
        for lever, eff in _effective_config(run, results, profile=profile).items()
        if eff.value is not None
    }


def _lever_value(record: ScoreRecord, lever: str, effective_by_run: dict[str, dict[str, EffectiveLever]]) -> str | None:
    """Read one lever's value for a score record, resolved the way the run lenses resolve it.

    Every lever but one is answered from the record's RUN effective configuration, never from
    ``record.factors``. ``factors`` is the launch-overlay flattening, so reading a lever off
    it puts a run that inherited a value in a different cohort from one that named the same
    value — the coverage surface reproducing, per record, the defect the run-derived lenses
    were converted to remove.

    The exception is :data:`_CANDIDATE_MODEL_LEVER`, and it is the ONLY one: the engine reserves
    that name for a declared coordinate of the observation itself, and
    :class:`~threetears.evals.kernel.host.profile.HostProfile` refuses a host recovery rule that claims it.
    Every record knows which model produced it, including one whose candidate role left no usage
    row for the run-level resolution to recover a model from, so reading it off the record is the
    true answer.

    **The name's SHAPE decides nothing.** This once switched on the dot — dotted through
    the effective config, plain off the record — which bound every host lever with a plain name
    (``chunk_tokens``) to ``'—'`` and reported a genuinely swept axis as ``unswept``, while
    ``RunSummary.config`` showed its two levels plainly. A lever's name is the host's, and a host
    that spells one without a dot is not thereby making a claim about the observation.

    This and :func:`_lever_levels` agree about every lever but that one, because a coverage entry
    and a divergence read over disagreeing cohorts cannot both be right about the same lever. On
    the candidate model they read different sources, and that function's docstring says why: it bins
    on the run's DECLARED model so a run whose candidate left no usage row keeps its level, where
    this reads the model one observation ran on. A run carries one model, so the two name the same
    level for every run assembly admits.

    Args:
        record: The score record to read.
        lever: A declared lever name.
        effective_by_run: Each run's resolved levers, keyed by run id.

    Returns:
        The level this record sits at, or ``None`` when the lever applied to its run but
        resolved ambiguously — no level can be claimed there, so the caller drops the record
        rather than pooling it into a cohort it cannot support.
    """
    if lever == _CANDIDATE_MODEL_LEVER:
        return _INHERITED_DEFAULT_LEVEL if not record.model else str(record.model)
    effective = effective_by_run.get(record.run_id, {}).get(lever)
    return _INHERITED_DEFAULT_LEVEL if effective is None else effective.value


__all__: list[str] = []
