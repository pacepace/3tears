"""Re-measure one stored run: read its results' host measures again, under the host's current definitions.

A run records the definition digest of every measure its host declared when it was launched
(:attr:`~threetears.evals.schema.models.EvalRun.measure_definitions`), and runs read under two definitions of one
measure never pool. A host that changes what a measure counts therefore cuts its earlier runs off from its later ones,
and so does a run stored before runs recorded definitions at all. This is the one door back in: the host's reader
(:attr:`~threetears.evals.kernel.host.profile.HostProfile.measure_rereader`) reads each stored result again from what
its cell stored, the new values replace the old on the result beside a record of what they replaced
(:class:`~threetears.evals.schema.models.MeasureReread`), and the run records the definitions its values now hold and
that it was re-measured (:class:`~threetears.evals.schema.models.MeasureRemeasure`). From then on its context identity
is derived from those definitions, so it joins the cell of the runs launched under them.

**Nothing is interpolated.** A stored value of a measure whose definition moved is either read again by the host's
reader or the whole run is refused, before anything is written: no value read under one definition is ever kept under
the digest of another, and none is guessed. A measure whose definition moved and of which no result holds a value
needs no reading: there is nothing to read again, and the run takes the current digest.

**A run its measure shaped is refused.** A measure that declares
:attr:`~threetears.evals.kernel.metrics.MetricDescriptor.shapes_run` decided what the run did (a stop rule, a floor the
world topped up to), so under another definition the run would have done something else, and reading its artifacts
again cannot say what. Such a run stays in the cell it was measured in; measuring under the current definition takes
a new run.

It reports first, like :func:`~threetears.evals.run.regrade.regrade`: with ``apply=False`` (the default) it reads and
checks everything and writes nothing. Applied, it writes in three steps: the run is marked as being re-measured (a
:class:`~threetears.evals.schema.models.MeasureRemeasure` with no ``completed_at``, which makes the run a cell of its
own), then each result, conditional on what was read, then the run's definitions and the mark's completion. A pass
cut short therefore leaves the run pooling with nothing, never in the old cell with some values read under the new
definition, and a second pass finishes it: a result already read again under the current definition is not read again.

**The other door is a host's statement.** :func:`attest_measure_definitions` stamps a run that recorded no
definitions with the ones its host says it ran under, touching no result: for a host that knows which revision of
its reading code was live when the run was made. The run then pools with the runs of those definitions, and can
later be re-measured into the current ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import Field

from threetears.evals.kernel.errors import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.kernel.host.measures import MeasureRereadRequest
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.remedies import FIND_RUN
from threetears.evals.run.run_document import update_eval_run
from threetears.evals.run.runner import refuse_misfit_host_measures
from threetears.evals.schema.base import EvalBaseModel
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import (
    NON_TERMINAL_RUN_STATUSES,
    MeasureAttestation,
    MeasureRemeasure,
    MeasureReread,
    utc_now_iso,
)

if TYPE_CHECKING:
    from threetears.evals.kernel.host.eval_host import EvalHost
    from threetears.evals.schema.models import EvalResult, EvalRun

log = get_logger(__name__)

#: How many results a refusal names before it counts the rest.
_NAMED_IN_A_REFUSAL = 5


class Remeasure(EvalBaseModel):
    """What re-measuring one stored run found, and whether it was written."""

    run_id: str
    applied: bool = Field(description="Whether the new values and the run's record were stored.")
    already_current: bool = Field(
        default=False,
        description="The run already records the host's current definitions: nothing was read and nothing written.",
    )
    was: dict[str, str] | None = Field(
        description="The run's measure definitions as read, measure key -> digest. None = it recorded none."
    )
    now: dict[str, str] = Field(description="The host's current measure definitions, measure key -> digest.")
    moved: list[str] = Field(
        default_factory=list,
        description="The measures whose definition the run did not record, or recorded otherwise, sorted.",
    )
    reread: list[str] = Field(
        default_factory=list, description="The measures a stored value of which was read again, sorted."
    )
    results_reread: int = Field(default=0, description="The results one or more of whose values were read again.")
    values_changed: int = Field(default=0, description="The stored values the current definition reads otherwise.")


@dataclass(frozen=True, slots=True)
class _Reread:
    """One result's new values, with the etag its rewrite presents."""

    result: EvalResult
    etag: str | None
    values: dict[str, bool | float | str]


@dataclass(frozen=True, slots=True)
class _Plan:
    """Everything a re-measure reads and checks before it writes."""

    run: EvalRun
    now: dict[str, str]
    moved: list[str]
    rereads: list[_Reread]


async def remeasure_run(host: EvalHost, run_id: str, scope_id: str, *, apply: bool = False) -> Remeasure:
    """Read one stored run's host measures again under the host's current definitions, and report what moved.

    :param host: The host: the run's store, its measure registry (``profile.measures``, whose definitions are the
        current ones) and its reader of a stored result (``profile.measure_rereader``).
    :ptype host: EvalHost
    :param run_id: The stored run to re-measure.
    :ptype run_id: str
    :param scope_id: The scope it lives in.
    :ptype scope_id: str
    :param apply: Store the new values and mark the run re-measured; ``False`` (the default) reads, checks and
        reports, and writes nothing.
    :ptype apply: bool
    :return: What moved and what was read again, and whether it was written.
    :rtype: Remeasure
    :raises NotFoundError: No such run in the scope.
    :raises ValidationFailedError: The run is still pending or running; a measure whose definition moved shaped the
        run (``MetricDescriptor.shapes_run``); a result holds a value of a moved measure and the host wires no
        reader; or the reader left such a value unanswered. Raised before anything is written.
    :raises ValueError: The host's reader answered a measure it was not asked, or a value whose type contradicts its
        measure's declared data type: the host's defect, raised before anything is written.
    :raises ConflictError: Something else wrote a result, or the run, since it was read. Results written before the
        conflict stay written, the run is marked as being re-measured (it pools with nothing), and a second pass
        finishes it.
    """
    plan = await run_blocking(host.blocking_executor, _plan, host, run_id, scope_id)
    run = plan.run
    report = Remeasure(
        run_id=run.id,
        applied=apply,
        already_current=run.measure_definitions == plan.now and not _unfinished(run),
        was=run.measure_definitions,
        now=plan.now,
        moved=plan.moved,
        reread=sorted({name for reread in plan.rereads for name in reread.values}),
        results_reread=len(plan.rereads),
        values_changed=sum(
            1
            for reread in plan.rereads
            for name, value in reread.values.items()
            if name in reread.result.host_measures and reread.result.host_measures[name] != value
        ),
    )
    if report.already_current:
        return report.model_copy(update={"applied": False})
    if apply:
        await run_blocking(host.blocking_executor, _write, host, plan, report, scope_id)
    log.info(
        "eval.remeasure run=%s applied=%s moved=%s results_reread=%d values_changed=%d",
        run.id,
        apply,
        ",".join(plan.moved) or "-",
        report.results_reread,
        report.values_changed,
    )
    return report


def _plan(host: EvalHost, run_id: str, scope_id: str) -> _Plan:
    """The store reads and every check of :func:`remeasure_run`, and each result's new values.

    :raises NotFoundError: No such run.
    :raises ValidationFailedError: See :func:`remeasure_run`.
    :raises ValueError: The host's reader answered what it was not asked, or a mistyped value.
    """
    storage, measures, reader = host.storage, host.profile.measures, host.profile.measure_rereader
    run = storage.load_eval_run(run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id, remedy=FIND_RUN)
    if run.status in NON_TERMINAL_RUN_STATUSES:
        raise ValidationFailedError(
            f"run '{run.id}' is {run.status}: its cells are still being written",
            remedy=f"re-measure once it has finished (job_poll run:{run.id})",
        )
    now, was = measures.definition_digests(), run.measure_definitions
    if was == now and not _unfinished(run):
        return _Plan(run=run, now=now, moved=[], rereads=[])
    moved = sorted(name for name in {*now, *(was or {})} if was is None or was.get(name) != now.get(name))
    # A measure the host no longer declares has no current definition to read under: its stored values stay as
    # the cell wrote them, undescribed, and the run's record drops its digest.
    declared = [name for name in moved if name in now]
    if shaped := [name for name in declared if _shapes_run(host, name)]:
        raise ValidationFailedError(
            f"run '{run.id}' cannot be re-measured: {', '.join(shaped)} shaped what the run did, and its definition "
            "is not the one the run was measured under"
            + (" (the run recorded no definitions)" if was is None else "")
            + ", so the run would have gone otherwise under the current one and its stored cells cannot say how",
            remedy="run it again under the current definition; the stored run stays in the cell it was measured in",
        )
    rereads: list[_Reread] = []
    unanswered: list[str] = []
    for listed in storage.query_eval_results_by_run(run.id, scope_id):
        result, etag = storage.load_eval_result_with_etag(listed.id, scope_id)
        if result is None:
            raise ValidationFailedError(
                f"run '{run.id}' cannot be re-measured: its result '{listed.id}' was listed and did not load, so "
                "whether it holds a value read under another definition cannot be told",
                remedy="re-measure it again once the result reads; nothing was written",
            )
        asked = tuple(name for name in declared if not _read_under(result, name, now[name]))
        held = [name for name in asked if name in result.host_measures]
        if reader is None:
            if held:
                raise ValidationFailedError(
                    f"run '{run.id}' cannot be re-measured: result '{result.id}' holds {', '.join(held)}, read under "
                    "another definition than the host's current one, and this host wires no reader of a stored "
                    "result (HostProfile.measure_rereader)",
                    remedy="wire a measure_rereader on the host's profile, or run it again under the current definition",
                )
            continue
        if not asked:
            continue
        trace = storage.load_eval_trace(result.id, scope_id) if result.has_trace else None
        values = dict(reader(MeasureRereadRequest(run=run, result=result, trace=trace, measures=asked)))
        if unasked := sorted(set(values) - set(asked)):
            raise ValueError(
                f"the host's measure_rereader answered {', '.join(unasked)} for result '{result.id}', which the "
                f"re-measure did not ask (it asked {', '.join(asked)})"
            )
        refuse_misfit_host_measures(values, measures)
        unanswered.extend(f"{result.id} ({name})" for name in held if name not in values)
        if values:
            rereads.append(_Reread(result=result, etag=etag, values=values))
    if unanswered:
        named = ", ".join(unanswered[:_NAMED_IN_A_REFUSAL])
        more = len(unanswered) - _NAMED_IN_A_REFUSAL
        raise ValidationFailedError(
            f"run '{run.id}' cannot be re-measured: the host's reader left {len(unanswered)} stored value(s) "
            f"unanswered ({named}{f', and {more} more' if more > 0 else ''}). Each was read under another definition "
            "and nothing guesses it under the current one",
            remedy="have the reader answer every measure a result holds, or run it again under the current definition",
        )
    return _Plan(run=run, now=now, moved=moved, rereads=rereads)


def _shapes_run(host: EvalHost, name: str) -> bool:
    """Whether the host's current descriptor of ``name`` says the measure shapes a run."""
    descriptor = host.profile.measures.get(name)
    return descriptor is not None and descriptor.shapes_run


def _read_under(result: EvalResult, name: str, digest: str) -> bool:
    """Whether the result's latest re-read of ``name`` was under ``digest``: a pass cut short already read it."""
    for reread in reversed(result.measure_rereads):
        if name in reread.definitions:
            return reread.definitions[name] == digest
    return False


def _unfinished(run: EvalRun) -> bool:
    """Whether a re-measure of the run began and did not finish."""
    return bool(run.measure_remeasures) and run.measure_remeasures[-1].completed_at is None


def _write(host: EvalHost, plan: _Plan, report: Remeasure, scope_id: str) -> None:
    """Mark the run as being re-measured, store each result's new values, then complete the run's record.

    :raises ConflictError: A result or the run was written since it was read.
    """
    storage, run = host.storage, plan.run
    record = MeasureRemeasure(
        was=run.measure_definitions, now=plan.now, reread=report.reread, results_reread=report.results_reread
    )

    def _begin(current: EvalRun) -> dict[str, Any] | None:
        if current.measure_definitions != run.measure_definitions:
            # Something else re-measured or attested it since the read.
            return None
        data = current.to_dict()
        # A pass cut short left its own unfinished mark: this pass takes its place.
        kept = current.measure_remeasures[:-1] if _unfinished(current) else current.measure_remeasures
        data["measure_remeasures"] = [*(entry.to_dict() for entry in kept), record.to_dict()]
        return data

    if (outcome := update_eval_run(storage, run.id, scope_id, _begin)) != "saved":
        raise ConflictError(
            f"run '{run.id}' changed while it was being re-measured ({outcome}); nothing was written",
            remedy="re-measure it again",
        )
    for reread in plan.rereads:
        result = reread.result
        updated = result.model_copy(
            update={
                "host_measures": {**result.host_measures, **reread.values},
                "measure_rereads": [
                    *result.measure_rereads,
                    MeasureReread(
                        definitions={name: plan.now[name] for name in sorted(reread.values)},
                        was={
                            name: result.host_measures[name]
                            for name in sorted(reread.values)
                            if name in result.host_measures
                        },
                    ),
                ],
            }
        )
        try:
            storage.replace_eval_result(updated, if_match=reread.etag)
        except ConflictError as raced:
            raise ConflictError(
                f"result '{result.id}' changed while run '{run.id}' was being re-measured; the run is marked as "
                "being re-measured and pools with nothing until a re-measure finishes it",
                remedy="re-measure it again: a result already read again is not read twice",
            ) from raced

    def _complete(current: EvalRun) -> dict[str, Any] | None:
        if not _unfinished(current) or current.measure_remeasures[-1].remeasured_at != record.remeasured_at:
            return None
        data = current.to_dict()
        data["measure_definitions"] = dict(plan.now)
        finished = record.model_copy(update={"completed_at": utc_now_iso()})
        data["measure_remeasures"] = [
            *(entry.to_dict() for entry in current.measure_remeasures[:-1]),
            finished.to_dict(),
        ]
        return data

    if (outcome := update_eval_run(storage, run.id, scope_id, _complete)) != "saved":
        raise ConflictError(
            f"run '{run.id}' was re-read but its record was not completed ({outcome}); it is marked as being "
            "re-measured and pools with nothing",
            remedy="re-measure it again: its results are already read again and are not read twice",
        )


class Attested(EvalBaseModel):
    """What a host's attestation of stored runs' measure definitions wrote."""

    run_ids: list[str] = Field(description="The runs stamped by this call, in the order given.")
    already_attested: list[str] = Field(
        default_factory=list,
        description="The runs an earlier attestation had stamped with these same definitions: left as they were.",
    )
    definitions: dict[str, str] = Field(description="The definitions stated for each, measure key -> digest.")
    attested_by: str


async def attest_measure_definitions(
    host: EvalHost,
    run_ids: Sequence[str],
    scope_id: str,
    definitions: Mapping[str, str],
    *,
    attested_by: str,
    note: str | None = None,
) -> Attested:
    """Stamp stored runs that recorded no measure definitions with the ones their host says they ran under.

    The host's statement, never the engine's guess: for a host that knows which revision of its reading code was
    live when each run was made. It computes the digests from the descriptors as they were
    (``MeasureRegistry(descriptors_then).definition_digests()``, or ``profile.measures.definition_digests()`` for
    runs made under today's). No result is read or written. Each run's ``measure_definitions`` become the stated
    map and the run records who stated it and when (``EvalRun.measure_attestation``), so it is never taken for a
    launch's record; a stamped run pools only with runs whose definitions match, and a page names it as attested.

    Every run is checked before any is written, and each write is one read and one conditional write of the run
    alone, so hundreds of runs are stamped in one call and a run with no stored result or trace is stamped like any
    other. Stating again what an earlier attestation stated is no error: the run is left as it was and named in
    ``already_attested``, so a backfill job may be run twice. Stating OTHER definitions over an attestation is
    refused: a statement is never replaced, and a host that stated the wrong one corrects the run's document itself.

    :param host: The host whose store holds the runs.
    :ptype host: EvalHost
    :param run_ids: The stored runs to stamp.
    :ptype run_ids: Sequence[str]
    :param scope_id: The scope they live in.
    :ptype scope_id: str
    :param definitions: The definitions they ran under, measure key -> definition digest. Not empty.
    :ptype definitions: Mapping[str, str]
    :param attested_by: Who states it, as the host names them: a person or a job.
    :ptype attested_by: str
    :param note: Why the host knows, in its own words.
    :ptype note: str | None
    :return: What was written.
    :rtype: Attested
    :raises NotFoundError: A run is not in the scope.
    :raises ValidationFailedError: ``definitions`` or ``attested_by`` is empty; a run is still pending or running; or
        a run already records definitions (its launch's, a re-measure's, or an earlier attestation of other
        definitions), which a statement never replaces. Raised before anything is written.
    :raises ConflictError: A run changed between the check and its write; runs written before it stay stamped.
    """
    stated = dict(sorted(definitions.items()))
    if not stated or not attested_by.strip():
        raise ValidationFailedError(
            "an attestation states at least one measure's definition and who states it",
            remedy="pass the definitions the runs ran under (measure key -> digest) and attested_by",
        )

    def write() -> list[str]:
        storage = host.storage
        already: list[str] = []
        for run_id in run_ids:
            run = storage.load_eval_run(run_id, scope_id)
            if run is None:
                raise NotFoundError("run", run_id, remedy=FIND_RUN)
            if run.status in NON_TERMINAL_RUN_STATUSES:
                raise ValidationFailedError(
                    f"run '{run.id}' is {run.status}: its cells are still being written",
                    remedy=f"attest once it has finished (job_poll run:{run.id})",
                )
            if _attested_as(run, stated):
                already.append(run.id)
                continue
            if run.measure_definitions is not None or run.measure_remeasures:
                how = (
                    "a re-measure wrote them"
                    if run.measure_remeasures
                    else "its host attested other definitions"
                    if run.measure_attestation is not None
                    else "its launch recorded them"
                )
                raise ValidationFailedError(
                    f"run '{run.id}' already records measure definitions ({how}); a statement never replaces a record",
                    remedy="leave it as recorded; to read it under the current definitions, re-measure it",
                )
        record = MeasureAttestation(attested_by=attested_by, definitions=stated, note=note)

        def _stamp(current: EvalRun) -> dict[str, Any] | None:
            if current.measure_definitions is not None or current.measure_remeasures:
                # Stamped by something else since the check: read below.
                return None
            data = current.to_dict()
            data["measure_definitions"] = dict(stated)
            data["measure_attestation"] = record.to_dict()
            return data

        for run_id in run_ids:
            if run_id in already:
                continue
            if (outcome := update_eval_run(storage, run_id, scope_id, _stamp)) != "saved":
                raced = storage.load_eval_run(run_id, scope_id)
                if raced is not None and _attested_as(raced, stated):
                    already.append(run_id)
                    continue
                raise ConflictError(
                    f"run '{run_id}' changed while its measure definitions were being attested ({outcome}); it was "
                    "not stamped, and the runs named before it were",
                    remedy="attest the remaining runs again",
                )
        return already

    already = await run_blocking(host.blocking_executor, write)
    stamped = [run_id for run_id in run_ids if run_id not in already]
    log.info("eval.attest_measure_definitions runs=%d already=%d by=%s", len(stamped), len(already), attested_by)
    return Attested(run_ids=stamped, already_attested=already, definitions=stated, attested_by=attested_by)


def _attested_as(run: EvalRun, stated: Mapping[str, str]) -> bool:
    """Whether an attestation already stamped the run with exactly ``stated``, and nothing re-measured it since."""
    return (
        run.measure_attestation is not None and not run.measure_remeasures and run.measure_definitions == dict(stated)
    )


__all__ = ["Attested", "Remeasure", "attest_measure_definitions", "remeasure_run"]
