"""Measurement-condition covariates and per-phase timings for eval results.

An eval observation is only poolable if the conditions it was made under are recorded
beside it. The campaign that motivated this found stop mode *dominating* synthesis latency
and concurrent runs contaminating latency pools — without those two recorded per
observation, a pooled latency number is unexplainable variance rather than a measurement.

Two rules this module exists to hold:

**A covariate nothing measured has no key.** Every function here returns a sparse mapping.
An absent key says "not measured"; a present key is an observation. The alternative — a
zero or an ``"unknown"`` sentinel — reads downstream as fact, and a marginal computed over
fabricated zeros is worse than one computed over a smaller honest n. This mirrors
:class:`~threetears.evals.contracts.models.RoleUsage`, which carries the same rule as ``None`` fields,
and the run-level async-delivery rollups, which omit their elapsed keys when no delivery
measured one.

**Phase timings are folded, not interpreted.** A kind whose candidate hands work to a background
tool folds that tool's own phase names into its telemetry through :func:`fold_phase_timings`, which
sums them per phase and prefixes them with the tool. Nothing here knows what any tool's phases are,
so a second background tool needs no change in the eval.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from threetears.evals.contracts.metrics import engine_owned_names
from threetears.evals.contracts.models import RoleUsage
from threetears.evals.contracts.provider import sum_optional_tokens
from threetears.observe import get_logger

log = get_logger(__name__)

#: Covariate key: how many tool calls the candidate emitted that the LLM client dropped
#: before dispatch (a ``dropped_tool_calls`` list on each turn record the kind dumps into the trace).
#:
#: One of the two covariates whose ZERO is load-bearing — see
#: :data:`REFUSED_TOOL_ATTACHES_KEY` for its sibling — and that is why the key exists.
#: A template's ``tools_allowed`` bound is enforced by simply not registering the tool, so a
#: candidate reaching for a bounded-away tool leaves no dispatch, no ``tool_result``, and no
#: error — the reach was visible only as a WARNING in the process log, which no eval read
#: surface carries. Silence therefore read identically to a candidate that never reached,
#: and a candidate could score as better-behaved under a bound than it was. ``0`` here means
#: the cell's turns were watched and nothing was dropped; an ABSENT key still means nobody
#: looked (no candidate turn ran), which is the module's sparse rule intact.
DROPPED_TOOL_CALLS_KEY = "dropped_tool_calls"

#: Covariate key: how many times the candidate asked to attach a tool outside its run's
#: ``tools_allowed`` and the harness refused. Counted off a ``refused_tool_attaches`` list on
#: the turn record, which the host's turn loop stamps on every record it builds.
#: The name is spelled here rather than cross-referenced because this module is the eval
#: engine and the field is the host's. Nothing in the engine can pin the two together: a host
#: whose turn record carries the field owns a test that a real dumped record reaches this
#: counter, so a rename on its side reddens rather than silently unmeasuring the covariate.
#:
#: The sibling of :data:`DROPPED_TOOL_CALLS_KEY`, and the same defect one seam over. A
#: bounded-away tool the candidate CALLS leaves a dropped call; a bounded-away tool the
#: candidate asks to ATTACH leaves a ``_plog.warning`` and a system perception telling it no,
#: and no eval read surface counted either half. So an operator could not distinguish a
#: candidate that never wanted another tool from one that asked twice and was refused twice
#: — and the refusal is not neutral under measurement: the candidate spends a turn asking,
#: is told no, and re-plans, and that turn is scored as ordinary conduct.
#:
#: Its zero is load-bearing for the same reason its sibling's is: ``0`` says the cell's turns
#: were watched and nothing was refused. An ABSENT key still means nobody looked.
REFUSED_TOOL_ATTACHES_KEY = "refused_tool_attaches"

#: Covariate key: how many of the candidate's LLM rounds the provider cut off at the output
#: cap — rounds whose ``stop_reason`` was ``max_tokens`` (the output reached the cap, or the
#: provider reported ``finish_reason=length``; providers do not always say so). Counted off
#: a ``truncated_rounds`` integer on the turn record, which the host's turn loop accumulates
#: across the turn's rounds by the same route as ``dropped_tool_calls`` and stamps on every
#: record it builds. Pinned to the field by the same test class as its siblings.
#:
#: The third of the load-bearing zeros, and a different seam from the first two: the harness
#: did not intervene, the PROVIDER did. A candidate that spends the whole cap reasoning
#: delivers no text and no tool call, the turn loop exits on the non-``tool_use`` stop exactly
#: as it would on a model that finished. Outside this covariate the cut's only trace is a
#: ``LLM response truncated (finish_reason=…)`` WARNING from the candidate's client, which no
#: stored record carries — so the judge read "the candidate produced no response" as a failure of
#: decision-making when the truth was that the output cap cut the model off mid-reasoning
#: (observed: ``completion=2048, reasoning=2048, reasoning_ratio=1``, no actions,
#: no text). Every quality measure on such a cell is confounded by the cap, and this is the
#: covariate that says so. ``0`` means the cell's turns were watched and none was cut; an
#: ABSENT key still means nobody looked.
TRUNCATED_ROUNDS_KEY = "truncated_rounds"

#: Covariate key: how many of the candidate's turns the HOST's turn budget ended before they
#: finished. A host that bounds a turn in production cancels a turn that outlives it, and a
#: cancelled turn delivers nothing: whatever its rounds had decided is discarded with it. Reported by the candidate
#: kind on :attr:`~threetears.evals.contracts.candidate_kind.CandidateTelemetry.turns_ended_by_budget`
#: rather than counted off the trace, because an ended turn leaves no turn record to count.
#:
#: A fourth load-bearing zero, and the third seam: not the harness intervening, nor the provider,
#: but the host's own operational bound — which the eval must run under or it scores, as a pass, a
#: turn production would have killed. ``0`` means the cell's turns ran under a budget and none
#: outlived it; an ABSENT key means no budget bounded them (a kind or a host with none), which is not
#: the same fact.
TURN_BUDGET_ENDED_KEY = "turns_ended_by_budget"


#: Covariate key: the share of the candidate's generated tokens that were reasoning — candidate
#: ``reasoning_tokens`` over candidate ``completion_tokens``, summed across the candidate's usage
#: rows. Absent unless both halves were measured and the candidate generated something; see
#: :func:`_reasoning_ratio_of`.
#:
#: Named because the analysis reads it as an OBSERVED MECHANISM as well as a covariate: a reasoning
#: effort sent as a word maps to a different effective budget per vendor, so two arms pinned to one
#: effort word can reason very differently, and the bundle discloses a contrast whose arms' shares
#: diverge (``threetears.evals.analysis.bundle``).
REASONING_RATIO_KEY = "reasoning_ratio"


def _candidate_turn_records(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The ``TurnRecord`` dumps a cell's trace carries, in order.

    One definition of "a candidate turn was observed", shared by every counter below, so the
    absent-versus-zero distinction cannot come out differently for two covariates reading the
    same trace.

    Every candidate turn — the simulator-driven ones and the drained async-delivery ones
    alike — is appended by the conversing kind's one turn-recording path, which dumps the ``TurnRecord`` onto its entry (all of it but the harness-injected
    perceptions, which are withheld there and are neither tool calls nor attach requests). So
    the trace already IS the complete per-cell record, with exactly one producer.

    Args:
        trace: The cell's judge-visible trace entries. Non-candidate entries (simulator
            utterances) carry no ``turn_record`` and contribute nothing.

    Returns:
        The dumped turn records, empty when no candidate turn ran.
    """
    return [entry["turn_record"] for entry in trace if isinstance(entry.get("turn_record"), dict)]


def count_dropped_tool_calls(trace: list[dict[str, Any]]) -> int | None:
    """Count the dropped tool calls across a cell's candidate turns.

    Args:
        trace: The cell's judge-visible trace entries.

    Returns:
        The total number of dropped calls, counting repeats — two reaches in one turn is a
        different observation from one. ``0`` when candidate turns ran and dropped nothing,
        and ``None`` when **no candidate turn ran at all**, which is not the same fact.

        The ``None`` arm is the one this function used to get wrong, and the key's own
        docstring already promised it: a cell whose first ``process_input`` raised breaks out
        of the turn loop before any record reaches the trace, yet still builds a full
        ``EvalResult`` — so it recorded ``dropped_tool_calls=0``, "watched and clean", for a
        cell nobody watched. An analyst pooling the covariate read crashed cells as
        well-behaved ones.
    """
    records = _candidate_turn_records(trace)
    if not records:
        return None
    return sum(len(record.get("dropped_tool_calls") or []) for record in records)


def count_refused_tool_attaches(trace: list[dict[str, Any]]) -> int | None:
    """Count the self-attach requests the harness refused across a cell's candidate turns.

    Reads a ``refused_tool_attaches`` list off each turn record, by the same route and with
    the same absent-versus-zero rule as :func:`count_dropped_tool_calls` —
    see :data:`REFUSED_TOOL_ATTACHES_KEY` for why the refusal has to be counted at all.

    **A turn record that carries no such field contributes nothing and does not make the
    cell measured.** A turn loop that does not stamp it is honestly "not measured", and the
    sparse rule then omits the key rather than publishing a zero the harness never observed.

    Args:
        trace: The cell's judge-visible trace entries.

    Returns:
        The total number of refusals, counting repeats — a candidate that asked twice and
        was refused twice behaved differently from one refused once. ``0`` when candidate
        turns reported the field and none was refused; ``None`` when no candidate turn
        reported it at all.
    """
    reporting = [record for record in _candidate_turn_records(trace) if "refused_tool_attaches" in record]
    if not reporting:
        return None
    return sum(len(record.get("refused_tool_attaches") or []) for record in reporting)


def count_truncated_rounds(trace: list[dict[str, Any]]) -> int | None:
    """Count the candidate rounds the provider cut off at the output cap across a cell's turns.

    Reads a ``truncated_rounds`` integer off each turn record, by the same route and with the
    same absent-versus-zero rule as :func:`count_refused_tool_attaches` — see
    :data:`TRUNCATED_ROUNDS_KEY` for why the cut has to be counted at all.

    **A turn record that carries no such field contributes nothing and does not make the
    cell measured**, for the reason its sibling gives: a turn loop that does not stamp it was not
    watched for this, and publishing a zero for it would claim it was.

    Args:
        trace: The cell's judge-visible trace entries.

    Returns:
        The total number of cut rounds over the turns that reported the field. ``0`` when
        candidate turns reported it and none was cut; ``None`` when no candidate turn
        reported it at all.
    """
    reporting = [record for record in _candidate_turn_records(trace) if "truncated_rounds" in record]
    if not reporting:
        return None
    return sum(int(record.get("truncated_rounds") or 0) for record in reporting)


def count_delivered_turns(trace: list[dict[str, Any]], *, reported: int | None) -> int | None:
    """How many turns a cell's candidate delivered — what the kind reported, else the turn records it stamped.

    The kind's own count wins: it is the one producer that knows what a turn is for it — one call for a
    classifier, one artifact for a generator. A conversing kind stamps a ``TurnRecord`` on every candidate turn
    it records, and a turn that raised never reaches the trace (see :func:`count_dropped_tool_calls`), so the
    records ARE the delivered turns. With neither, nothing counted, and that is ``None`` — never a zero a
    reader would take for "refused before any turn".

    Args:
        trace: The cell's judge-visible trace entries.
        reported: The kind's own count (``CandidateTelemetry.turns_delivered``), or ``None``.

    Returns:
        The count, or ``None`` when nothing counted.
    """
    if reported is not None:
        return reported
    records = _candidate_turn_records(trace)
    return len(records) if records else None


#: Every key :func:`derive_covariates` writes: the covariates a result can carry, each a core measure. A test reads
#: the writer's own assignments and fails when this set and they disagree.
COVARIATE_KEYS: frozenset[str] = frozenset(
    {
        "execution_mode",
        DROPPED_TOOL_CALLS_KEY,
        REFUSED_TOOL_ATTACHES_KEY,
        TRUNCATED_ROUNDS_KEY,
        TURN_BUDGET_ENDED_KEY,
        "context_tokens_in",
        REASONING_RATIO_KEY,
    }
)


def undeclarable_covariates(names: Iterable[str]) -> list[str]:
    """The covariate keys that name a measure only the engine measures and that no covariate writer lands, sorted.

    The rule ``host_measures`` is held to (:func:`~threetears.evals.contracts.metrics.undeclarable_host_measures`),
    carried to the covariates map: its legitimate core-named keys are :data:`COVARIATE_KEYS`, and any other
    engine-owned key would pool into the engine's own observations of that name.

    Args:
        names: The keys a result's covariates carry.

    Returns:
        The engine-owned keys among them that are not covariates.
    """
    return engine_owned_names(names, written_by_the_engine=COVARIATE_KEYS)


def derive_covariates(
    *,
    usage: list[RoleUsage],
    concurrent_eval_jobs: int | None,
    dropped_tool_calls: int | None = None,
    refused_tool_attaches: int | None = None,
    truncated_rounds: int | None = None,
    turns_ended_by_budget: int | None = None,
) -> dict[str, str | float]:
    """Derive one result's measurement-condition covariates (R4).

    Args:
        usage: The per-role rows captured for this result. ``context_tokens_in`` and
            ``reasoning_ratio`` are read off the candidate rows — plural, because rows are
            keyed by (role, model).
        concurrent_eval_jobs: How many eval jobs were executing when this cell ran,
            *including* this one, or ``None`` when nothing sampled it. Jobs, not runs —
            a distinction with no live instance since template generation was removed with
            the v6 eval surface, so every job this counts today IS a run. The wider word is
            kept deliberately: the probe samples the job manager, and a future non-run job
            would contend on the same provider account exactly as the generation one would
            have. None yields no ``execution_mode`` key — "nobody
            looked" is not "nothing else was running".
        dropped_tool_calls: How many tool calls the candidate emitted that the client
            dropped before dispatch, or ``None`` when no candidate turn was observed —
            see :data:`DROPPED_TOOL_CALLS_KEY`. Unlike a resolved-model origin on the
            run renders, defaulting this one asserts nothing: a caller that omits it
            produces no key, which reads as "not measured" and is exactly what a cell
            that never reached a candidate turn should say.
        refused_tool_attaches: How many self-attach requests the harness refused for
            being outside the run's ``tools_allowed``, or ``None`` when no candidate turn
            reported the field — see :data:`REFUSED_TOOL_ATTACHES_KEY`. Defaults the same
            way and for the same reason as ``dropped_tool_calls``: omitting it asserts
            nothing.
        truncated_rounds: How many of the candidate's rounds the provider cut off at the
            output cap, or ``None`` when no candidate turn reported the field — see
            :data:`TRUNCATED_ROUNDS_KEY`. Defaults the same way as its two siblings:
            omitting it asserts nothing.
        turns_ended_by_budget: How many of the candidate's turns the host's turn budget ended,
            or ``None`` when no budget bounded them — see :data:`TURN_BUDGET_ENDED_KEY`.
            Defaults the same way as its siblings: omitting it asserts nothing.

    Returns:
        A sparse ``{covariate: value}`` mapping — only what was actually measured.
    """
    out: dict[str, str | float] = {}

    if concurrent_eval_jobs is not None:
        out["execution_mode"] = "concurrent" if concurrent_eval_jobs > 1 else "serial"

    if dropped_tool_calls is not None:
        out[DROPPED_TOOL_CALLS_KEY] = dropped_tool_calls

    if refused_tool_attaches is not None:
        out[REFUSED_TOOL_ATTACHES_KEY] = refused_tool_attaches

    if truncated_rounds is not None:
        out[TRUNCATED_ROUNDS_KEY] = truncated_rounds

    if turns_ended_by_budget is not None:
        out[TURN_BUDGET_ENDED_KEY] = turns_ended_by_budget

    candidate_rows = [row for row in usage if row.role == "candidate"]

    context_tokens_in = _sum_measured(row.prompt_tokens for row in candidate_rows)
    if context_tokens_in is not None:
        out["context_tokens_in"] = context_tokens_in

    reasoning_ratio = _reasoning_ratio_of(candidate_rows)
    if reasoning_ratio is not None:
        out[REASONING_RATIO_KEY] = reasoning_ratio

    # The one writer of a result's covariates, so the rule host_measures is held to is held here, at the write: a
    # key outside COVARIATE_KEYS — a core-named one would pool into the engine's own observations of that name —
    # is refused, and the set read by the walk and the bar gate cannot drift from what is written.
    if stray := sorted(set(out) - COVARIATE_KEYS):
        raise ValueError(f"derive_covariates wrote {stray!r}, which are not covariates (COVARIATE_KEYS)")
    return out


def _sum_measured(values: Iterable[int | None]) -> int | None:
    """Sum the reported values across rows, or ``None`` when none of them reported.

    A thin iterable adapter over :func:`~threetears.evals.contracts.provider.sum_optional_tokens`, which
    is the eval package's single definition of "sum what was observed, keep unmeasured
    unmeasured" for token counts — reimplementing it here would let the two drift apart on
    exactly the case they exist to get right.
    """
    return sum_optional_tokens(*values)


def _reasoning_ratio_of(candidate_rows: list[RoleUsage]) -> float | None:
    """Reasoning tokens as a fraction of the candidate's generated tokens.

    ``None`` unless BOTH halves were measured and the candidate actually generated
    something: a provider that reported no reasoning split leaves the ratio unknown (never
    0.0, which would claim the model did no thinking), and a zero denominator has no ratio
    at all. A genuine ``0`` reasoning against real output is a measurement and returns 0.0.
    """
    completion = _sum_measured(row.completion_tokens for row in candidate_rows)
    reasoning = _sum_measured(row.reasoning_tokens for row in candidate_rows)
    if completion is None or reasoning is None or completion <= 0:
        return None
    return round(reasoning / completion, 6)


def fold_phase_timings(
    accumulator: dict[str, float],
    *,
    source_tool: str,
    timings: Any,
) -> None:
    """Fold one delivery's carried phase timings into a result's accumulator.

    Keys land as ``<source_tool>_<phase>_ms`` so two tools reporting a ``synthesis`` phase
    stay distinct, and repeat deliveries SUM: a cell that ran that tool twice spent both
    synthesis windows, and the covariate is wall-clock attributable to that phase within
    the cell.

    ``timings`` arrives as whatever the background tool reported, so a malformed entry is
    dropped with a warning rather than wedging the cell — but it is dropped LOUDLY, because a
    bad value silently becoming a plausible duration is exactly the conflation this capture
    exists to prevent. A negative duration is not a measurement.

    Public, because the caller is a kind, not the runner: the key form is the engine's, so any
    kind whose candidate hands work to a detached tool reports that work's phases through this
    one fold.
    """
    if timings is None:
        return
    if not isinstance(timings, dict):
        log.warning("Delivery from %s carried non-mapping phase timings (%r); ignoring", source_tool, timings)
        return
    for phase, value in timings.items():
        if value is None:
            # The tool measured no such phase on this run — the trace field was None.
            continue
        try:
            ms = float(value)
        except TypeError, ValueError:
            log.warning("Phase timing %s.%s is not a number (%r); recording as unmeasured", source_tool, phase, value)
            continue
        if ms < 0:
            log.warning("Phase timing %s.%s is negative (%r); recording as unmeasured", source_tool, phase, value)
            continue
        key = f"{source_tool}_{phase}_ms"
        accumulator[key] = round(accumulator.get(key, 0.0) + ms, 3)


__all__ = [
    "COVARIATE_KEYS",
    "undeclarable_covariates",
    "DROPPED_TOOL_CALLS_KEY",
    "REASONING_RATIO_KEY",
    "REFUSED_TOOL_ATTACHES_KEY",
    "TRUNCATED_ROUNDS_KEY",
    "TURN_BUDGET_ENDED_KEY",
    "count_delivered_turns",
    "count_dropped_tool_calls",
    "count_refused_tool_attaches",
    "count_truncated_rounds",
    "derive_covariates",
    "fold_phase_timings",
]
