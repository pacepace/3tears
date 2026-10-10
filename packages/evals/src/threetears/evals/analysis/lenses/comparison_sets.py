"""Comparison sets: which of a set of runs may honestly be compared with each other.

:func:`compute_comparison_sets` groups runs by context identity and case set, and badges every group whose
members differ in a way a reader must see (context, case set, roles, tool config, measurement windows,
cassette mode). :func:`difference_was_declared_at_launch` is the rule beneath a lever-origin disclosure: a
difference between arms is a swept axis only when each arm's launch declared it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.hashing import canonical_digest
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.identity import resolve_context_identity
from threetears.observe import get_logger
from threetears.evals.analysis.reporting import _subject_id_of, measurement_window, measurement_window_disclosure
from threetears.evals.analysis.cassette_mode import cassette_mode_disclosure

if TYPE_CHECKING:
    from threetears.evals.schema.models import (
        EvalResult,
        EvalRun,
    )


log = get_logger(__name__)


class CaseSetIdentity(EvalBaseModel):
    """One distinct case set inside a group, and which runs executed it.

    A ``template_id`` does not establish case-set identity while templates are
    mutable in place: a template can gain or lose cases between runs, so two runs
    of "the same suite" may have scored different denominators. The identity is
    already computed — it is ``ContextComponents.case_basis``, the digest of
    ``(template_id, sorted test_case_ids)`` stamped on every run at launch — but
    nothing rendered it, so the fact was recorded and unreadable.

    ``n_cases`` is the size of the set this fingerprint stands for, which is the
    number the group's ``shared_test_case_ids`` intersection cannot give you: an
    intersection of 3 is equally consistent with runs of 3 cases and runs of 31.
    """

    #: Full ``case_basis`` digest — the run's own stamped value where present,
    #: re-derived from its stored fields otherwise (a run its host assembled without
    #: the launch), so every run is identified on the same predicate. ``None`` when the run's basis could
    #: not be resolved at all: the entry then stands for that ONE run (``run_ids`` has
    #: one member), because an unresolved basis says the set is unknown and merging
    #: unknowns would assert a sameness nothing established. It is a state, not a
    #: digest, so a renderer branches on ``None`` rather than displaying anything.
    fingerprint: str | None
    n_cases: int
    run_ids: list[str]


#: The one origin value that means *this launch said so*. The other tiers — a value the
#: subject carried, or one inherited from a system default — supplied it from below the launch,
#: so a difference across them is one nobody asked this comparison for.
#:
#: The vocabulary of origins is a host's; what the engine owns is the RULE beneath it, and the
#: surface that renders a particular lever's sentence lives with that lever's vocabulary, in the host.
DECLARED_INPUT_ORIGIN = "chosen"


def difference_was_declared_at_launch(origins: Iterable[str | None]) -> bool:
    """Whether every one of these runs got its value from its own launch declaration.

    **The generic half of a lever-origin disclosure, and the part that stayed.** The rule is
    not about any one lever: a difference between arms is a *swept axis* when each arm's launch
    declared the value, and a *confound* when any arm inherited it from a tier the comparison
    never named. Only the vocabulary (``chosen`` / a subject-carried tier / an inherited
    default) is a host's; the rule is the same one a campaign's declared design will state
    formally once it declares one, at which point this reads the declaration instead of
    the origins. A host calls it to write, for example, the sentence about which model its
    arms delegated background work to.

    Conservative by construction: one arm that inherited its value is enough to make the
    whole difference undeclared, because the comparison cannot claim to be sweeping an
    axis one of its arms was never pointed at.

    Args:
        origins: One origin per compared run, as the run recorded it. ``None`` means the
            run recorded no origin and therefore declared nothing that can be read.

    Returns:
        ``True`` when every origin is :data:`DECLARED_INPUT_ORIGIN`. ``False`` for an
        empty set — nothing declared anything — so a caller can never read silence as a
        declaration.
    """
    listed = list(origins)
    return bool(listed) and all(origin == DECLARED_INPUT_ORIGIN for origin in listed)


class ComparisonSet(EvalBaseModel):
    """A group of runs that may be compared with each other, and on what basis.

    ``badges`` names every respect in which the group is *less* than fully
    comparable. An empty list means the group shares template, subject, context
    key, and test-case set — the only case where run-vs-run subtraction needs no
    caveat.

    ``shared_test_case_ids`` is the INTERSECTION across the group, and on its own
    it misleads in exactly the direction that matters: a sweep whose runs
    report 2 shared cases can have runs that each executed 6, and the intersection
    was small *because* the sets differed. ``case_sets`` names each distinct set
    and who ran it, so the reader sees how the group is split rather than only
    what survives the overlap.
    """

    subject_id: str
    subject_label: str = ""
    template_id: str | None = None
    run_ids: list[str]
    shared_test_case_ids: list[str]
    #: Distinct case sets within the group, most-used first. More than one entry is
    #: the ``case_set_differs`` badge's substance — the badge says *that* they
    #: differ, these say *how*.
    case_sets: list[CaseSetIdentity] = []
    badges: list[str]
    #: The ``measurement_windows_disjoint`` badge's substance: the sentence naming
    #: when each run was measured. The badge says the group did not share a clock;
    #: only this says whether that means four minutes or four hours, which is the
    #: whole of the judgement being handed to the reader. ``None`` whenever the
    #: badge is absent — the two are derived from one predicate and cannot disagree.
    measurement_window_disclosure: str | None = None
    #: The ``cassette_mode_differs`` badge's substance: the sentence naming what each run
    #: recorded and what the span costs. The badge says the group's arms did not all
    #: record one mode; only this says whether that means a recording difference between
    #: two live arms or a replayed arm that never measured the third party at all.
    #: ``None`` whenever the badge is absent — one predicate, so the two cannot disagree.
    cassette_mode_disclosure: str | None = None
    #: The runs that share this group's (subject, template) and that the caller's scope
    #: left out. Empty when unscoped, which is every group's ordinary state.
    #:
    #: A scope makes the badges honest about the set under analysis, and in doing so it
    #: hides how much else the group holds — so a reader meeting a badge-clean scoped
    #: group cannot tell whether the group really is three replicates or three of seven runs
    #: spanning several days. Naming the excluded runs is what keeps the narrowing visible:
    #: the argument is that a set quietly reduced reads identically to a set that was always
    #: that size.
    out_of_scope_run_ids: list[str] = []


# Badge vocabulary. ``context_differs`` extends the existing ``comparison_basis``
# vocabulary used by run comparison; ``case_set_differs`` records the drift that
# makes a longitudinal series dishonest when a suite silently gained or lost
# cases between runs.
BADGE_CONTEXT_DIFFERS = "context_differs"
BADGE_CASE_SET_DIFFERS = "case_set_differs"
# The pinned roles: models held fixed so the candidate is the only thing varying.
# When they differ between runs, a quality delta is confounded by a judge or
# simulator change and the comparison measures two things at once.
BADGE_ROLES_DIFFER = "roles_differ"
# At least one run's measurement context could not be fully reconstructed — a role pin
# it never recorded. Distinct from every other badge here, which reports a difference
# that WAS observed: this one reports that the comparison cannot be decided. It has to
# exist separately because an unrecorded pin makes the other badges come out CLEAN —
# two runs that both stored a blank compare equal on the blank — so the group would
# otherwise render as the one case needing no caveat at all.
BADGE_CONTEXT_INCOMPLETE = "context_incomplete"
# At least one run's case basis could not be resolved. The sibling of
# ``context_incomplete``, one axis over, and separate from ``case_set_differs`` for
# the reason that badge cannot carry: an unresolved basis means the sets are
# UNKNOWN, not known-different. Asserting a difference we did not observe is the
# same dishonesty as asserting a match. See ``CaseSetIdentity.fingerprint`` for
# why such runs never merge into one entry.
BADGE_CASE_SET_UNRESOLVED = "case_set_unresolved"
# The candidate's own resolved tool config (search depth, token budgets, call caps,
# the inner-agent model). Unlike the three above it is not a caveat on the
# comparison — sweeping it is usually the POINT — but it has to be said out loud,
# because it is invisible in the one place an operator looks for it: tool config
# composes the VARIANT key, so runs that differ across it still share a context key
# and, badged only on the conditions, render as one configuration — runs differing
# in three tool caps at once can read as one.
BADGE_TOOL_CONFIG_DIFFERS = "tool_config_differs"
# At least one pair of the group's runs was measured over spans that do not
# overlap, so anything that moved on the box, the providers or the account between
# those spans varies with the runs. It is the one badge here whose evidence is not
# in the run documents at all — it is derived from the results' ``scored_at``, so a
# group whose results were not supplied cannot be asked and is silent rather than
# clean. Descriptive only: how much a gap costs depends on what else changed in it,
# which this surface cannot see and the operator can.
BADGE_MEASUREMENT_WINDOWS_DISJOINT = "measurement_windows_disjoint"
# The group's runs did not all record the same cassette mode. It sits with
# ``roles_differ`` and ``tool_config_differs`` as a per-component badge over a condition
# the context key already hashes but cannot name: ``cassette`` is a context component
# (``identity.py``), so a capture-versus-replay pair fires ``context_differs`` — "something
# about the conditions moved" — and the operator is left to bisect which. That is the exact
# opacity these badges exist to remove, and it went unnamed on the one condition where the
# arms may not have measured the same thing at all. Like the disjoint-window badge it is
# gated on its DISCLOSURE rather than on a predicate of its own, so the flag can never
# appear with no sentence naming the modes behind it: the flag cannot tell an
# ``off``/``capture`` recording difference from a live-versus-replayed substitution, and
# that distinction is the whole of what a reader needs.
BADGE_CASSETTE_MODE_DIFFERS = "cassette_mode_differs"


class ComparisonSetsResult(EvalBaseModel):
    """Comparability groups plus the runs the caller's own scope left out.

    There is no blank-subject exclusion count, and its absence is the point: every run carries a
    non-blank subject key, so "a scope whose runs cannot be grouped" is not a state. A counter
    that could only ever read zero is worse than none — a reader takes zero as evidence.

    ``out_of_scope_run_ids`` names every supplied run the scope left out, so a reader meeting a
    small clean group can tell whether it is the whole story. Empty when unscoped.
    """

    comparison_sets: list[ComparisonSet] = []
    out_of_scope_run_ids: list[str] = []


def compute_comparison_sets(
    runs: list[EvalRun],
    *,
    results: Sequence[EvalResult] = (),
    full_windows: bool = False,
    scope_run_ids: Iterable[str] | None = None,
    profile: HostProfile,
) -> ComparisonSetsResult:
    """Group runs into sets that may honestly be compared, badging every caveat.

    Runs are grouped by (subject, template): those are the two coordinates whose
    difference makes a comparison meaningless rather than merely noisy. Within a
    group, differences that weaken — but do not invalidate — the comparison are
    reported as badges, never by silently dropping runs.

    **A badge speaks for the runs it was computed over, and a reader is usually
    asking about a smaller set than the group holds.** A campaign can hold
    three runs on one template while the group holds seven, spanning several days,
    and badge ``cassette_mode_differs`` because some non-members recorded with a
    cassette while every member recorded ``off`` — a caveat earned by runs
    the analysed set does not contain. ``scope_run_ids`` is how a caller
    asks about the set actually under analysis: the grouping key is unchanged and
    the badge logic is unchanged, and only the population they run over narrows.

    Two badges are not "a difference was observed". ``tool_config_differs``
    reports one that usually IS the point of the runs — it is here because it
    lives in the *variant* key rather than the context key, so a swept set is
    otherwise badge-silent and reads as one configuration. ``context_incomplete``
    reports the opposite: a comparison that cannot be decided, because a run
    never recorded a role pin. That one cannot be inferred from the others —
    an unrecorded pin makes every value-equality here come out clean.

    ``measurement_windows_disjoint`` is the one badge whose evidence is not in the
    run documents. When two runs occupied non-overlapping spans of wall-clock
    time, whatever moved on the box or at the providers between them varies with
    the runs — a real confound that every field on an :class:`EvalRun` compares
    equal on. The spans come from the results' ``scored_at``, which is why
    ``results`` exists as a parameter at all.

    Runs with no captured subject identity are excluded, for the reason given in
    :func:`project_score_records`.

    Args:
        runs: Runs to group. Order is irrelevant; output is sorted for stability.
        results: The results those runs produced, in any order and from any run —
            they are indexed by ``eval_run_id`` here. **A caller that omits them
            asks a narrower question**: the measurement-window badge cannot be
            decided without them, and an undecidable badge is silent, so a group
            reads as sharing a clock when nothing checked. Every production caller
            supplies the scope's results; the default exists for callers whose
            question really is only about the run documents.
        full_windows: List every measurement span rather than summarising above
            :data:`MAX_INLINE_MEASUREMENT_WINDOWS`. Off by default because this
            surface groups a whole scope — the group that most needs the
            disclosure is the one with the most runs, and so the one the listing
            form renders unreadable.
        scope_run_ids: Narrow every group to these runs before badging — the run
            ids of the set actually under analysis, typically a campaign's members
            resolved by the caller. ``None`` groups and badges every supplied run,
            which is the unscoped behaviour and is unchanged. **Run ids rather than
            a campaign id deliberately**: this module knows nothing about
            campaigns and must not learn — new eval machinery lands host-agnostic,
            and resolving a grouping concept
            to its members is the caller's one lookup. A scope naming runs that are
            not here is logged rather than refused: unlike an unknown run *status*,
            which names nothing that can exist, a member absent from the scope
            is a real and reachable state.
        profile: The host whose vocabulary this reads.

    Returns:
        A :class:`ComparisonSetsResult` — one :class:`ComparisonSet` per
        (subject, template) group, sorted by subject then template, plus the
        count of runs excluded for having no captured subject identity, plus
        ``out_of_scope_run_ids`` naming what a scope left out — at the result
        level every supplied run the scope excluded, and per group the comparable
        runs it excluded from THAT group. Both are empty when unscoped. Under a
        scope, a group whose every member is out of scope is not emitted at all,
        because a group with no in-scope run has nothing to say about the set the
        caller asked about — its members are still named at the result level, so
        the narrowing stays visible rather than becoming a group that vanished.
    """
    results_by_run: dict[str, list[EvalResult]] = {}
    for result in results:
        results_by_run.setdefault(result.eval_run_id, []).append(result)

    # Narrowed BEFORE grouping, so every derived quantity below — the shared case
    # intersection, the case-set fingerprints, every badge and both disclosures —
    # is computed over one population. Filtering afterwards would leave a badge
    # earned by a run the returned group no longer lists, which is the defect one
    # level down from the one this parameter exists to fix.
    scoped: frozenset[str] | None = frozenset(scope_run_ids) if scope_run_ids is not None else None
    # What the scope left out, captured BEFORE the narrowing and keyed the same way the
    # groups are, so a scoped group can say what a reader would otherwise have to re-read
    # the scope to discover. Grouped here rather than derived by the caller because
    # only this function knows the grouping rule, and a second derivation of it is how a
    # group and its own exclusion list come to disagree about which runs are comparable.
    out_of_scope_run_ids: list[str] = []
    out_of_scope_by_key: dict[tuple[str, str | None], list[str]] = {}
    if scoped is not None:
        present = {run.id for run in runs}
        out_of_scope = [run for run in runs if run.id not in scoped]
        out_of_scope_run_ids = sorted(run.id for run in out_of_scope)
        for run in out_of_scope:
            out_of_scope_by_key.setdefault((_subject_id_of(run), run.template_id), []).append(run.id)
        if missing := sorted(scoped - present):
            # Logged rather than raised, and never silent: a scope that resolves to
            # nothing returns no groups, which reads exactly like "these runs are
            # not comparable with anything".
            log.info(
                "comparison_sets scope named %d run(s) not among the supplied runs: %s",
                len(missing),
                ", ".join(missing),
            )
        runs = [run for run in runs if run.id in scoped]

    # No blank-subject exclusion, and nothing to count: every run carries a non-blank subject key,
    # because ``SubjectSnapshot`` refuses a blank one. A counter that could only ever read zero is
    # a disclosure that tells a reader the opposite of the truth.
    groups: dict[tuple[str, str | None], list[EvalRun]] = {}
    for run in runs:
        groups.setdefault((_subject_id_of(run), run.template_id), []).append(run)

    sets: list[ComparisonSet] = []
    for (subject_id, template_id), grouped in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        member_case_sets = [set(run.test_case_ids) for run in grouped]
        shared = set.intersection(*member_case_sets) if member_case_sets else set()

        # Through the resolver, not off the raw field. `resolve_context_identity` is
        # the single place the stamped-or-derived decision is made, and reading
        # `run.context_key` directly opted this surface out of it: a run stamped under
        # an older predicate kept comparing on a key that predicate no longer produces,
        # and a run whose roles were never recorded reported a key as if it were whole.
        identities = {run.id: resolve_context_identity(run, profile) for run in grouped}

        # The group's distinct case sets, keyed by the run's own `case_basis` digest.
        # Computed BEFORE the badges because the case-set badge is derived from it:
        # one notion of "the sets differ", so the badge and the fingerprints cannot
        # come apart. Deciding the badge separately — over `set(test_case_ids)` — was
        # a second derivation that really did disagree, since a set comparison cannot
        # see a duplicate id that the sorted-list digest does.
        by_fingerprint: dict[str, list[EvalRun]] = {}
        unresolved: list[EvalRun] = []
        for run in grouped:
            basis = identities[run.id].context_components.case_basis
            # An unresolved basis is held apart, one entry PER RUN. Keying every such
            # run on a shared ``""`` merged them into one bucket, which broke two things
            # at once: the case-set badge could not fire between them (they compared as
            # a single set), and ``n_cases`` below — read from the first member as the
            # set's representative, sound only while a bucket really IS one set —
            # reported that one member's denominator as the whole bucket's. Keeping them
            # out of the digest-keyed buckets makes the merge structurally impossible
            # rather than merely unlikely, which matters because the state is latent
            # today: nothing stamps an identity without ``case_basis`` at the current
            # IDENTITY_VERSION, so the reachable version would arrive unnoticed on the
            # next bump.
            if basis:
                by_fingerprint.setdefault(basis, []).append(run)
            else:
                unresolved.append(run)

        badges: list[str] = []
        # Only RESOLVED fingerprints can evidence a difference. Counting the
        # per-run unresolved entries here would report "the sets differ" on the
        # strength of not knowing what they were.
        if len(by_fingerprint) > 1:
            badges.append(BADGE_CASE_SET_DIFFERS)
        if unresolved:
            badges.append(BADGE_CASE_SET_UNRESOLVED)
        if len({identities[run.id].context_key for run in grouped}) > 1:
            badges.append(BADGE_CONTEXT_DIFFERS)
        # Equality on the pins is only meaningful once both are recorded. Two runs that
        # each inherited a judge before the pin was resolved both carry None, and every
        # comparison here — the tuple below, the key above — comes out EQUAL, so the group
        # ships with no badge at all, which this surface defines as needing no caveat.
        # The absence has to be its own badge; nothing about the values can express it.
        if any(identities[run.id].partial for run in grouped):
            badges.append(BADGE_CONTEXT_INCOMPLETE)
        # Per-dim attribution joins the pins here for the same reason it joined the roles
        # component: two runs can agree on every pin and still have had their rubric scored
        # by different models, which is a difference in the apparatus wearing the appearance
        # of a replicate. Hashed rather than compared as a mapping so the tuple stays
        # hashable, and sorted so dim resolution order cannot fake a difference.
        #
        # RECORDED attribution only, via the same accessor ``derive_context_identity``
        # gates on, and compared SEPARATELY from the pins rather than folded into one
        # tuple per run. Both halves matter. Reading the map whenever it was merely
        # present let a DERIVED reconstruction speak with the authority of a record —
        # asserting or suppressing a comparability claim that ``identity.py`` refuses to
        # hash in the same breath. Folding it into the per-run tuple then reproduced the
        # absence-as-difference bug one level down: a run with no recorded attribution
        # contributes a different tuple element from one that has it, so merely MIXING
        # them fired the badge on the strength of what one run could not say. This is the
        # shape the case-basis arm above already uses — only resolved values evidence a
        # difference, and the undecidable case is BADGE_CONTEXT_INCOMPLETE's to carry.
        # The pinned CONFIG set is the third arm, on the argument above carried one step
        # further: two runs can agree on every pin AND on every effective judge model and
        # still have been scored by different judge PROMPTS, which is what a judge A/B is
        # made of. Without this arm such a pair fires only ``context_differs`` — "something
        # about the conditions moved" — which is precisely the opaque key mismatch the
        # per-component badges exist to save the operator from bisecting by hand.
        #
        # THE PINS COME FROM THE REGISTRY, through the declared readers, so this badge and the two
        # surfaces that print the delta cannot disagree about one pair of runs. It was a hand-built
        # ``{(run.judge_model, run.simulator_model)}``, which was wrong twice over: it spoke only
        # for the ENGINE's two roles, so a host that grades with code and declares its own role got
        # no arm at all however registry-driven everything upstream had become; and a tuple of raw
        # fields reproduced the absence-as-difference bug the attribution arm below had already
        # been fixed for — one arm recording a pin and another recording nothing is not an observed
        # difference, and ``BADGE_CONTEXT_INCOMPLETE`` is what carries it. ``comparability`` applies
        # that rule per input rather than per tuple.
        #
        # ``omits_apparatus`` takes every arm's VALUE rather than the declaration alone: a host that
        # left a seat unfilled and whose runs recorded one is contradicting itself, and the
        # runs win.
        #
        # **The two digest arms below stay, and are not redundant with it.** Both read a RUN-LEVEL
        # declaration that no ``Sweepable`` reads: ``judge_config_ids`` is declared here at launch,
        # while the core declaration of the same name reads the set the results were actually scored
        # WITH — the reader's own docstring says those are different questions. A judge A/B moves
        # the launch pin, and a run whose results were never scored would stop badging if this arm
        # were folded into the observed one. ``effective_judges`` is the same shape: the declaration
        # isolates DIVERGENCE from the pin, which is deliberately narrower than the whole map.
        sweepables = profile.sweepables
        pins_by_run = {run.id: sweepables.read_role_pins(run, results_by_run.get(run.id, ())) for run in grouped}
        pin_values = {name: [pins_by_run[run.id].get(name) for run in grouped] for name in sweepables.role_pins}
        pins_differ = any(
            sweepables.comparability(
                name,
                [profile.apparatus_level(run, name, value) for run, value in zip(grouped, values, strict=True)],
            )
            == "differs"
            for name, values in pin_values.items()
            if not profile.omits_apparatus(name, zip(grouped, values, strict=True))
        )
        recorded_attributions = {
            canonical_digest(dict(sorted(judges.items())))
            for run in grouped
            if (judges := run.hashable_effective_judges)
        }
        # ``is not None``, never truthiness: an empty set is the recording "no scored dim
        # carried a config", and two runs that both recorded it agree. Reading it as absent
        # would drop them out of the comparison and let a genuinely-differing third run in
        # the group pass unbadged.
        recorded_config_sets = {
            canonical_digest(dict(sorted(run.judge_config_ids.items())))
            for run in grouped
            if run.judge_config_ids is not None
        }
        if pins_differ or len(recorded_attributions) > 1 or len(recorded_config_sets) > 1:
            badges.append(BADGE_ROLES_DIFFER)
        # Digest the RESOLVED configs, not the overrides: an override restating the
        # subject's own value changes nothing the candidate faced, and comparing
        # overrides would badge that as a difference. This mirrors what the variant
        # key hashes, so the badge and the key can never disagree about what moved.
        if (
            len(
                {
                    canonical_digest(run.subject_snapshot.component_hashes().get("resolved_tool_configs"))
                    for run in grouped
                }
            )
            > 1
        ):
            badges.append(BADGE_TOOL_CONFIG_DIFFERS)
        # Only RESOLVED windows are collected, on the rule the case-basis and
        # attribution arms already follow: a run that cannot say when it was
        # measured is not thereby evidence that it was measured somewhere else.
        # The badge is gated on the DISCLOSURE rather than on the predicate
        # directly, so a group can never carry the flag with no sentence naming
        # the spans behind it — the flag alone cannot tell four minutes from four
        # hours, and that judgement is the reader's.
        windows = [
            w for run in grouped if (w := measurement_window(run.id, results_by_run.get(run.id, ()))) is not None
        ]
        window_disclosure = measurement_window_disclosure(windows, full=full_windows)
        if window_disclosure:
            badges.append(BADGE_MEASUREMENT_WINDOWS_DISJOINT)
        # Gated on the disclosure, never on a predicate of its own — the discipline the
        # window arm above follows, and for a sharper reason here: the flag alone cannot
        # tell an `off`/`capture` recording difference from a live-versus-replayed
        # substitution, and only the second makes the group's QUALITY numbers a mixture.
        # A badge with no sentence behind it would be the less useful half of the pair.
        cassette_disclosure = cassette_mode_disclosure({run.id: run.cassette_mode for run in grouped})
        if cassette_disclosure:
            badges.append(BADGE_CASSETTE_MODE_DIFFERS)

        case_sets = [
            CaseSetIdentity(
                fingerprint=fingerprint,
                n_cases=len(set(members[0].test_case_ids)),
                run_ids=sorted(run.id for run in members),
            )
            # Most-used set first, so the group's dominant denominator leads and the
            # odd run out is visibly the exception rather than one row among equals.
            for fingerprint, members in sorted(by_fingerprint.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ]
        # The unresolved runs last, one entry each: a set of one is the least-used there
        # is, and nothing established that any two of them ran the same cases.
        case_sets.extend(
            CaseSetIdentity(fingerprint=None, n_cases=len(set(run.test_case_ids)), run_ids=[run.id])
            for run in sorted(unresolved, key=lambda run: run.id)
        )

        sets.append(
            ComparisonSet(
                subject_id=subject_id,
                subject_label=grouped[0].subject_snapshot.subject_label,
                template_id=template_id,
                run_ids=sorted(run.id for run in grouped),
                shared_test_case_ids=sorted(shared),
                case_sets=case_sets,
                badges=badges,
                measurement_window_disclosure=window_disclosure,
                cassette_mode_disclosure=cassette_disclosure,
                out_of_scope_run_ids=sorted(out_of_scope_by_key.get((subject_id, template_id), [])),
            )
        )
    return ComparisonSetsResult(
        comparison_sets=sets,
        out_of_scope_run_ids=out_of_scope_run_ids,
    )


__all__ = [
    "BADGE_CASE_SET_DIFFERS",
    "BADGE_CASE_SET_UNRESOLVED",
    "BADGE_CASSETTE_MODE_DIFFERS",
    "BADGE_CONTEXT_DIFFERS",
    "BADGE_CONTEXT_INCOMPLETE",
    "BADGE_MEASUREMENT_WINDOWS_DISJOINT",
    "BADGE_ROLES_DIFFER",
    "BADGE_TOOL_CONFIG_DIFFERS",
    "CaseSetIdentity",
    "ComparisonSet",
    "ComparisonSetsResult",
    "compute_comparison_sets",
    "DECLARED_INPUT_ORIGIN",
    "difference_was_declared_at_launch",
]
