"""Taking a run up again: a run whose holding ended before its matrix did, continued as the same run.

A run lost to a reboot, an OOM kill, a deploy or a provider outage longer than its lease, a run an operator cancelled
to ship a fix, and a run its budget or its account stopped each end with cells of their matrix never measured. A
new run over the cases left pools with the first in pass^k (a case set is no part of a cell), but it is another run:
it measures every repeat of each case it covers, so a case the first reached part-way is measured past ``k``, and
the first stays short on its own record. :func:`resume_run` continues the run itself:

- **Only what is missing runs.** A cell with no stored row runs, and so does a cell whose row was cut short by a
  cancel (termination ``cancelled``: the attempt a shutdown or a cancel abandoned), refused by the provider
  account paying for the run (``EvalResult.account_refused``: the call a funded account would have served), or
  answered that its model is not served (``EvalResult.model_unserved``: the call a served model would have
  answered). Every
  other row is what the run would have stored uninterrupted, so its cell is done (:func:`_planned_cells`). The cells
  run in the run's own order, its done ones left out, and a re-run cell is written over its row.
- **The same measurement, or none.** The run is launched again from what it recorded, through its kind's launcher,
  and the run that launch assembles is compared with the stored one: every context component, the variant key, the
  simulator and judge pins, the apparatus, what the run froze of its template (its kind spec, rubric scales, world
  seed, tool bound and goal checks), and the digest of everything its template says that a run measures
  (``EvalRun.template_digest``: every field but its name, description, archive flag and record bookkeeping). Any
  difference refuses the take-up, naming every one (:func:`_resume_drift`). A template edited after the run began in
  none of that — its description, say — measures alike: the edit is disclosed on the holding that began after it,
  not refused. A run stored before it froze that digest cannot tell the two apart, so its take-up is refused once
  its template was edited after it began. Nor are the host's labels — a build, a version — the measurement: they
  are recorded on the holding that began under them (``EvalRun.lifetimes``). Nor is what the host disclosed about the
  launch (``EvalRun.run_disclosure``: its machine, its load, its build), which is printed as given and never compared:
  the run keeps the launch's. Nor is the engine's build (``EvalRun.package_version``, ``package_rev``): a run taken
  up across an engine upgrade goes ahead, and keeps the launch's build, the only one it records, since a holding
  records none (``RunLifetime``). So a resumed run can span engine builds, and the framing its judge is sent may
  change between holdings with no judge-prompt digest recording it. Nor is the version of the goal-check proof rules
  (``EvalRun.goal_check_proof_rules``): the launch re-proves the run's own goal checks under the rules in force, and
  only a check proved otherwise than the run's proof refuses. A re-proof that matches moves the run's rules forward
  to the ones it was made under, so a ``proven`` its launch recorded reads as proven under today's rules, and each
  holding records the rules its checks were proved under (``RunLifetime.goal_check_proof_rules``): the launch's on
  the first, the take-up's on each later one.
- **One holding at a time.** The run is reopened under its lease's next epoch, conditionally on the record as read,
  so two take-ups of one run cannot both win, and a holding the run has passed from writes nothing more into it.
- **Its spend counts.** What its stored rows spent seeds its cost cap, and the metered calls they landed seed its
  metered-call ceiling; a run its budget stopped is taken up only under a cap above that spend, and none is taken up
  once its stored cells have landed the metered calls its ceiling allows.

**A launch, or a sweep, as a whole.** A launch's unfinished runs are taken up together, in one group under the
launch's own id, so an interleaved launch's arms take their remaining cells in turn in the order they always had
(:func:`resume_launch`); a sweep's are taken up by the sweep's own job, as its arm schedule says, beside the arms it
never launched (:func:`~threetears.evals.ops.sweeps.sweep_resume`). Every arm's run is planned and replayed in the call
(:func:`take_up_arms`), so one arm that would measure otherwise refuses the whole take-up, naming every difference of
every arm, before any arm is started.

A take-up is always asked for, never made at boot. A host offers one: :func:`resumable_jobs` lists the runs, sweeps
and launches a take-up would accept, which a host reads at boot, after settling what its last process left running
(:func:`~threetears.evals.run.lifecycle.sweep_abandoned_runs`), and offers to an operator.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, NamedTuple, Protocol, cast

from pydantic import Field

from threetears.evals.kernel.numbers import round_half_even
from threetears.evals.kernel.campaign import EvalSweep
from threetears.evals.kernel.errors import ConflictError, NotFoundError, ValidationFailedError
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.identity import LeverCoordinateError, resolve_context_identity, resolve_variant_identity
from threetears.evals.kernel.job_leases import (
    RUN_JOB,
    LeasedJobKind,
    read_to_take_up,
    reopened_job_fields,
    taken_up_meanwhile,
    write_reopened,
)
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.remedies import FIND_JOB, FIND_RUN
from threetears.evals.kernel.run_pages import RUN_RESULTS_PAGE, iter_run_results
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.run.budget import EvalRunCostCap
from threetears.evals.run.job_work import start_job_work
from threetears.evals.run.launch import LaunchGroup, LaunchHost, admit_group, admit_resumed_run
from threetears.evals.run.launch_job import LAUNCH_JOB, EvalLaunch, resumed_launch_work
from threetears.evals.run.metering import MeteredCallLedger
from threetears.evals.run.runner import StoredCell, StoredCells
from threetears.evals.schema.base import EvalBaseModel, EvalDocumentModel
from threetears.evals.schema.job_lease import JobLease, JobLoss
from threetears.evals.schema.logs import get_logger
from threetears.evals.schema.models import (
    TERMINAL_RUN_STATUSES,
    EvalResult,
    EvalRun,
    EvalRunStatus,
    EvalTemplate,
    EvalTestCase,
    RunLifetime,
    ended_lifetimes,
    run_ending_reason,
    utc_now_iso,
)

log = get_logger(__name__)

#: The fields of a run a take-up does not hold equal field by field, each for its reason. Every other field is pinned
#: (:data:`_PINNED_FIELDS`), so a field a later build adds to the run is pinned unless it is left out here on purpose.
_UNPINNED_FIELDS: frozenset[str] = frozenset(
    {
        # The record's own: which document it is, in which scope, written when.
        "id",
        "doc_type",
        "schema_version",
        "schema_minor",
        "scope_id",
        "created_at",
        # Held by :func:`_resume_drift` in its own way: the context identity read at the current identity version, the
        # variant levers, the case set as a set, and the template's measured content.
        "context_key",
        "context_components",
        "identity_version",
        "variant_levers",
        "test_case_ids",
        "template_digest",
        # Held by its outcome, never its number: the take-up's launch re-proves the run's own goal checks under the
        # proof rules in force (``goal_check_proofs``, pinned), so a rules change that proves them as the run's own
        # proofs did is taken up. The run then reads as proved under the newer rules, and each holding records the
        # rules it proved the checks under (``RunLifetime.goal_check_proof_rules``).
        "goal_check_proof_rules",
        # Replayed as the run recorded them by the take-up's launch (``admit_resumed_run``), so equal by construction:
        # its template, its candidate, its repeats, its margins, its latency declaration and its arm order. The case
        # set's name is a label on the case ids, which are held.
        "template_id",
        "candidate_model",
        "k_runs",
        "declared_margins",
        "measure_latency",
        "arm_schedule",
        "case_set",
        # Who was measured: its id is replayed, its component levels reach the variant levers and the context's
        # subject state, and its labels identify and determine nothing.
        "subject_snapshot",
        # The host's own payload, which the engine never reads.
        "host_payload",
        # What the host disclosed about the launch (its machine, its load, its build): printed as given, never
        # compared. The run keeps the launch's; a holding has no field for its own (``RunLifetime``).
        "run_disclosure",
        # The engine's build: a resume across an engine upgrade is taken up, and the run keeps the launch's build.
        "package_version",
        "package_rev",
        # Whether the cost cap was checked on estimated rates: about admission under the cap, not measurement. A
        # take-up keeps it set when either holding was admitted on an estimate.
        "cap_checked_on_estimates",
        # How the pinned roles, apparatus and cell deadline were arrived at, not what they are: those are pinned.
        "apparatus_provenance",
        "model_role_provenance",
        "effective_judges_source",
        "judge_config_provenance",
        "cell_timeout_s_origin",
        # What generation asked for and kept: a take-up generates nothing.
        "variation_counts",
        # How a reader reads the host's measures, never what a cell measured.
        "declared_measures",
        # Derived from the seed the run froze and its subject's carriers, which the seed, the subject state and the
        # seeded world component hold.
        "world_placements",
        # How many cells ran at once: each result stamps the execution mode it ran under.
        "cell_concurrency",
        # Wired by the kind's launcher as it would wire them now (`admit_resumed_run`): a deadline the launch named is
        # replayed and one it inherited is inherited again, so a take-up runs under the host's current bounds rather
        # than being refused when they moved. The kind is the template's, which `kind_spec` and the template digest
        # hold.
        "candidate_kind",
        "cell_timeout_s",
        "turn_budget_s",
        # The ceilings a holding runs under, which a take-up may raise and each holding records.
        "max_cost_usd",
        "max_cost_usd_origin",
        "max_metered_calls",
        "max_metered_calls_origin",
        "metered_calls_refused",
        # When it ran beside its sibling arms: scheduling, not measurement.
        "launch_group_id",
        # How the run stands and ended, which a take-up reopens, and its curation.
        "status",
        "completeness",
        "progress",
        "cancellation_reason",
        "budget_stop_reason",
        "budget_stops",
        "failed_because",
        "exhausted_because",
        "error_details",
        "completed_at",
        "archived",
        "labels",
        "lease",
        "cancel_requested_at",
        "cancel_request_reason",
        "lost",
        "faults",
        "lifetimes",
    }
)

#: The fields of a run's judge and simulator, of its apparatus and of what it froze of its template, a take-up holds
#: equal: every field but :data:`_UNPINNED_FIELDS`, in the run's own order. Most also reach the context key; each is
#: named on its own so a refusal says which moved.
_PINNED_FIELDS: tuple[str, ...] = tuple(name for name in EvalRun.model_fields if name not in _UNPINNED_FIELDS)


def _run_again(row: EvalResult) -> bool:
    """Whether a stored row's cell runs again when its run is taken up: an attempt the run abandoned, or one whose call
    the account or the model's provider turned away, which a funded account or a served model would have answered."""
    return row.termination == "cancelled" or row.account_refused or row.model_unserved


@dataclass(frozen=True)
class _StoredRows:
    """What a take-up keeps of a run's stored rows, read a page at a time: each row as a :class:`StoredCell`, every one
    and then the done and the run-again apart, each in the page order (:mod:`~threetears.evals.kernel.run_pages`)."""

    every: tuple[StoredCell, ...]
    done: tuple[StoredCell, ...]
    again: tuple[StoredCell, ...]


def _stored_rows(storage: EvalStorage, run: EvalRun) -> _StoredRows:
    """Read a run's stored rows a page at a time (:data:`~threetears.evals.kernel.run_pages.RUN_RESULTS_PAGE`), keeping
    each as a :class:`StoredCell`, so a take-up of a large run never holds its rows' reasoning, payloads or events.

    A row whose cell was cut short by a cancel (``cancelled``) is an attempt a shutdown or a cancel abandoned, one
    the provider account paying for the run refused (``account_refused``) is a call a funded account would have
    served, and one whose model the provider did not serve (``model_unserved``) is a call a served model would have
    answered: an uninterrupted run on a funded account and a served model stores none of them, so each cell runs
    again, written over its row.
    Every other row — a timeout, a rig fault, a call the provider itself declined, a failed seed included — is what
    the uninterrupted run would also have stored, so its cell is done. So is a row whose judging the cost cap set
    aside (``judging_set_aside``): its candidate's answer is measured and stored, and running the cell again would pay
    for it twice and measure another answer, so a re-judge (``regrade`` with ``apply``, through
    :func:`~threetears.evals.run.lifecycle.rejudge_result`) takes its judging up instead. A cell
    with no row is neither, and runs.

    :param storage: The run's store.
    :ptype storage: EvalStorage
    :param run: The run.
    :ptype run: EvalRun
    :return: Its rows, done and run again.
    :rtype: _StoredRows
    """
    every: list[StoredCell] = []
    done: list[StoredCell] = []
    again: list[StoredCell] = []
    for row in iter_run_results(storage, run, page_results=RUN_RESULTS_PAGE):
        cell = StoredCell.from_result(row)
        every.append(cell)
        (again if _run_again(row) else done).append(cell)
    return _StoredRows(every=tuple(every), done=tuple(done), again=tuple(again))


def _planned_cells(rows: _StoredRows) -> StoredCells:
    """What a run taken up again holds of its matrix: the rows that count as done, and the rows it runs again by cell
    (:func:`_stored_rows` says which is which).

    :param rows: Every row stored for the run.
    :ptype rows: _StoredRows
    :return: The done rows, and the rows run again by cell.
    :rtype: StoredCells
    """
    return StoredCells(
        done=rows.done,
        replaced={(cell.summary.test_case_id, cell.summary.k_iteration): cell for cell in rows.again},
    )


def _counted_spend(rows: Sequence[StoredCell]) -> list[float | None]:
    """What a run's stored rows spent, as its cost cap counts it: each row's attempt, and the attempts it was written
    over (``EvalResult.superseded_spend_usd``), so a cell cut short and run again counts every attempt once. ``None``
    is spend nobody priced.

    The out-of-run spend ledger (``OutOfRunSpend``) is not counted: it records calls made outside every run, such as
    case generation, which no run's cap counted while the run was uninterrupted, and none of a run's cell spend.
    Counting it here would hold a resumed run to a stricter cap than the same run uninterrupted.

    :param rows: Every row stored for the run.
    :ptype rows: Sequence[StoredCell]
    :return: One amount per attempt the rows record.
    :rtype: list[float | None]
    """
    spend: list[float | None] = []
    for row in rows:
        # Beside its cost, what its dropped sends may have billed, as the uninterrupted run's cap counted it.
        spend.append(None if row.summary.cost_usd is None else row.summary.cost_usd + row.summary.maybe_billed_usd)
        if row.superseded_spend_usd != 0.0:
            spend.append(row.superseded_spend_usd)
    return spend


def next_epoch(lease: JobLease | None, lost: JobLoss | None, held: Iterable[int] = ()) -> int:
    """The lease epoch a take-up of a job holds: one above every epoch its record has held, so a holding the job has
    passed from never holds it again.

    The record's lease names the epoch it is held at until a shutdown releases the lease as it writes the job lost;
    the loss then keeps that epoch (``JobLoss.epoch``). A run's and a sweep's lifetimes keep each earlier holding's.

    :param lease: The record's lease, or ``None`` once released.
    :ptype lease: JobLease | None
    :param lost: The record's loss, or ``None``.
    :ptype lost: JobLoss | None
    :param held: The epochs of the record's earlier holdings, where it keeps them.
    :ptype held: Iterable[int]
    :return: The next epoch: 1 for a job never taken up again.
    :rtype: int
    """
    return 1 + max([lease.epoch if lease is not None else 0, lost.epoch if lost is not None else 0, *held])


def _shown(value: Any) -> str:
    """A value as a refusal names it: a long digest cut to its head."""
    text = repr(value)
    return text if len(text) <= 40 else f"{text[:37]}..."


def _resume_drift(stored: EvalRun, fresh: EvalRun, template: EvalTemplate, profile: HostProfile) -> list[str]:
    """Every way the run a launch of ``stored`` assembles now would measure otherwise than ``stored`` did.

    Each context component by name, then the context key when no component accounts for its move, each variant
    lever, each of :data:`_PINNED_FIELDS`, the frozen case set (as a set, in any order), and the template's measured
    content: the digest the run froze at launch (``EvalRun.template_digest``) against the one the launch froze now,
    so an edit to the intent, the rubric's descriptions and scoring guides, the conversation or any other field a run
    measures is a difference, and one to the template's name or description is not. A run stored before it froze the
    digest is refused once its template was edited after the run was created, since what the edit changed cannot be
    known. The subject's labels and the run's own are none of these: they identify, and determine nothing.

    The context identity is compared as it reads at the current identity version on both sides
    (:func:`~threetears.evals.kernel.identity.resolve_context_identity`), the comparison pooling makes, so a run
    stamped before a version bump is re-derived rather than refused for the bump alone. A run that recorded no lever
    map, and whose variant this host's registry can no longer place, is a difference named as such.

    :param stored: The stored run.
    :ptype stored: EvalRun
    :param fresh: What a launch of it assembles now, stamped.
    :ptype fresh: EvalRun
    :param template: The template that launch read.
    :ptype template: EvalTemplate
    :param profile: The host, which re-derives the context identity of a run stamped under an older identity version,
        and whose lever reader derives the variant of a run that recorded no lever map.
    :ptype profile: HostProfile
    :return: One phrase per difference; empty when the two measure alike.
    :rtype: list[str]
    """
    differences: list[str] = []
    was_identity, now_identity = resolve_context_identity(stored, profile), resolve_context_identity(fresh, profile)
    was, now = was_identity.context_components, now_identity.context_components
    if was is not None and now is not None:
        for name in type(was).model_fields:
            before, after = getattr(was, name), getattr(now, name)
            if before == after:
                continue
            if isinstance(before, Mapping) and isinstance(after, Mapping):
                moved = sorted(key for key in {*before, *after} if before.get(key) != after.get(key))
                differences.append(f"context component {name} ({', '.join(moved)})")
            else:
                differences.append(f"context component {name} ({_shown(before)} -> {_shown(after)})")
    if was_identity.context_key != now_identity.context_key and not differences:
        differences.append(
            f"context key ({_shown(was_identity.context_key)} -> {_shown(now_identity.context_key)}, at identity "
            f"version {now_identity.identity_version})"
        )
    try:
        levers_was = resolve_variant_identity(run=stored, profile=profile).levers
    except LeverCoordinateError as unplaced:
        # A run that recorded no lever map has its variant derived through today's registry, which may no longer
        # place it: what it varied cannot be compared, so the take-up cannot show it measures alike.
        differences.append(
            f"variant levers (the run recorded none, and this host's registry cannot place it: {unplaced})"
        )
    else:
        levers_now = resolve_variant_identity(run=fresh, profile=profile).levers
        for name in sorted({*levers_was, *levers_now}):
            before_lever, after_lever = levers_was.get(name), levers_now.get(name)
            if (before_lever.content_hash if before_lever else None) != (
                after_lever.content_hash if after_lever else None
            ):
                differences.append(f"variant lever {name}")
    for name in _PINNED_FIELDS:
        before, after = getattr(stored, name), getattr(fresh, name)
        if before != after:
            differences.append(f"{name} ({_shown(before)} -> {_shown(after)}{_proof_rules_moved(name, stored, fresh)})")
    # The case set reaches no key, so it is held here: a take-up runs the cases the run froze, never others. Compared
    # over the sorted ids, as `case_set_fingerprint` compares them, so the same cases in another order are no change;
    # the template that fingerprint also digests is named by the context component that holds it.
    if sorted(stored.test_case_ids) != sorted(fresh.test_case_ids):
        differences.append(f"test_case_ids ({_case_set_drift(stored.test_case_ids, fresh.test_case_ids)})")
    if stored.template_digest is not None and stored.template_digest != fresh.template_digest:
        differences.append(
            f"template {template.id!r}: its measured content changed since the run started, and such an edit "
            "changes what the run measures, so the cells left to run would not measure what its stored ones did "
            "(its intent, rubric, conversation, preconditions, goal checks, world seed, tool bounds, kind spec, "
            "variation axes, required tools, universal flag or another field a run measures; its name and "
            "description are not)"
        )
    elif stored.template_digest is None and _edited_since(template, stored):
        differences.append(
            f"template {template.id!r}: edited at {template.updated_at}, after the run was created at "
            f"{stored.created_at}, and the run was stored before runs froze what their template measures, so whether "
            "the edit changed the measurement cannot be told"
        )
    return differences


def _proof_rules_moved(name: str, stored: EvalRun, fresh: EvalRun) -> str:
    """What a refusal adds to a goal-check proof difference when the proof rules moved too: the two rule versions,
    so an operator reads that the current rules prove the run's own checks otherwise; empty for any other field, or
    when the rules did not move."""
    if name != "goal_check_proofs" or stored.goal_check_proof_rules == fresh.goal_check_proof_rules:
        return ""
    return (
        f", re-proved under proof rules {fresh.goal_check_proof_rules}; the run's were derived under "
        f"{stored.goal_check_proof_rules}"
    )


def _proof_rules_after(stored: EvalRun, fresh: EvalRun) -> int | None:
    """The proof rules a run taken up reads as proved under: the newer of the run's own and those its take-up re-proved
    its checks under, the two having proved them alike; the run's own when either recorded none.

    :param stored: The stored run.
    :ptype stored: EvalRun
    :param fresh: What its take-up's launch assembled.
    :ptype fresh: EvalRun
    :return: The rules version, or None when the run recorded no proofs.
    :rtype: int | None
    """
    if stored.goal_check_proof_rules is None or fresh.goal_check_proof_rules is None:
        return stored.goal_check_proof_rules
    return max(stored.goal_check_proof_rules, fresh.goal_check_proof_rules)


def _case_set_drift(was: Sequence[str], now: Sequence[str]) -> str:
    """How a launch's case ids differ from the ones a run froze: the cases left out, added, and listed a different
    number of times, each sorted.

    :param was: The run's frozen case ids.
    :ptype was: Sequence[str]
    :param now: The ones a launch of it assembled.
    :ptype now: Sequence[str]
    :return: The parts that differ, joined with ``; ``.
    :rtype: str
    """
    before, after = Counter(was), Counter(now)
    left_out = sorted(case for case in before if case not in after)
    added = sorted(case for case in after if case not in before)
    repeated = sorted(case for case in before if case in after and before[case] != after[case])
    parts = [
        f"left out: {', '.join(left_out)}" if left_out else "",
        f"added: {', '.join(added)}" if added else "",
        "duplicated: " + ", ".join(f"{case} listed {after[case]} times, was {before[case]}" for case in repeated)
        if repeated
        else "",
    ]
    return "; ".join(part for part in parts if part)


def _edited_since(template: EvalTemplate, run: EvalRun) -> bool:
    """Whether ``template`` was edited after ``run`` was created."""
    return template.updated_at > run.created_at


@dataclass(frozen=True)
class RunResumption:
    """A stored run taken up again, as its launch replays it: carried on the replayed arm's request
    (:attr:`~threetears.evals.run.launch.LaunchRequest.resuming`).

    :ivar run: The run as stored, settled.
    :ivar cases: Its frozen cases, in its own order: the launcher freezes these.
    :ivar stored: What it holds of its matrix (:func:`_planned_cells`).
    :ivar rows: Every row stored for it, done or run again, as a :class:`~threetears.evals.run.runner.StoredCell`:
        what its cost cap and metered-call ceiling count (:func:`_counted_spend`).
    :ivar holder: The process taking it up.
    :ivar epoch: The lease epoch it is taken up under.
    :ivar cell_order_seed: What its cells were ordered by when that was its launch's group, shared with its sibling
        arms; ``None`` when they were ordered by its own id.
    :ivar etag: The token its record was read with: its reopen is written conditionally on it, so two take-ups of
        one run cannot both win.
    :ivar max_cost_usd: The cost cap it is taken up under: a raised one, or the one it recorded; ``None`` inherits
        the host's.
    :ivar notices: What checking its take-up told the caller beside admitting it (a served-model check that could
        not be made), set by :func:`take_up_arms`; empty until checked, and when there was nothing to tell.
    """

    run: EvalRun
    cases: tuple[EvalTestCase, ...]
    stored: StoredCells
    rows: tuple[StoredCell, ...]
    holder: str | None
    epoch: int
    cell_order_seed: str | None
    etag: str | None
    max_cost_usd: float | None
    notices: tuple[str, ...] = ()

    def take_up(
        self, fresh: EvalRun, template: EvalTemplate, profile: HostProfile
    ) -> tuple[EvalRun, list[EvalTestCase]]:
        """Compare what its launch assembled with the stored run, and reopen the stored run to go on in its place.

        The reopened run reads ``running``. What it carried of its ending — its loss, its reasons, its budget stops,
        its completeness and when it ended — moves to the holding that ended (its first holding's entry is written
        now, for a run never taken up before), and a new holding opens: this process, at :attr:`epoch`, under the
        labels the host captured now and the cost cap the launch resolved, which the run records as its own. A
        template edited after the run was created in nothing it measures (its digest unmoved) is disclosed on the new
        holding. A re-proof of the run's goal checks under newer proof rules that matched the run's own proofs moves
        the run's ``goal_check_proof_rules`` forward, and the new holding records the rules it re-proved under (the
        first holding's entry, written now, the launch's). The host's disclosure and the engine's build are never
        compared, so a moved one refuses nothing,
        and the run keeps the ones its launch stated: a holding records neither. The run reads admitted on estimated
        rates (``cap_checked_on_estimates``) when either its launch or this take-up was, and records none when the take-up
        drops the cap.

        :param fresh: The run the launch assembled and stamped.
        :ptype fresh: EvalRun
        :param template: The template the launch read.
        :ptype template: EvalTemplate
        :param profile: The host's profile.
        :ptype profile: HostProfile
        :return: The reopened run, its lease still the stored one (the take-up's write grants the next), and its
            cases in its own order.
        :rtype: tuple[EvalRun, list[EvalTestCase]]
        :raises ValidationFailedError: The two would not measure alike; every difference is named.
        """
        stored = self.run
        if differences := _resume_drift(stored, fresh, template, profile):
            remedy = (
                "launch the template again (run_launch) to measure as things stand now, or restore what moved and call "
                "job_resume again"
            )
            if stored.template_digest is None and _edited_since(template, stored):
                remedy += (
                    "; a run stored before runs froze their template's measured content is taken up only while its "
                    "template is unedited since the run was created"
                )
            raise ValidationFailedError(
                f"run '{stored.id}' is not taken up again: launched now, it would measure otherwise than it did — "
                f"{'; '.join(differences)}",
                remedy=remedy,
            )
        first = RunLifetime(
            holder=stored.lease.holder if stored.lease is not None else None,
            epoch=self.epoch - 1,
            started_at=stored.created_at,
            labels=dict(stored.subject_snapshot.labels),
            run_labels=list(stored.labels),
            max_cost_usd=stored.max_cost_usd,
            goal_check_proof_rules=stored.goal_check_proof_rules,
        )
        ended = ended_lifetimes(
            stored.lifetimes or [first],
            status=stored.status,
            at=stored.completed_at or utc_now_iso(),
            reason=run_ending_reason(stored),
            lost=stored.lost,
            budget_stops=stored.budget_stops,
            completeness=stored.completeness,
            cells_stored=len(self.rows),
        )
        holding = RunLifetime(
            holder=self.holder,
            epoch=self.epoch,
            started_at=utc_now_iso(),
            labels=dict(fresh.subject_snapshot.labels),
            run_labels=list(stored.labels),
            max_cost_usd=fresh.max_cost_usd,
            template_edited_at=template.updated_at if _edited_since(template, stored) else None,
            goal_check_proof_rules=fresh.goal_check_proof_rules,
        )
        reopened = stored.model_copy(
            update={
                "status": "running",
                "lost": None,
                "cancellation_reason": None,
                "cancel_requested_at": None,
                "cancel_request_reason": None,
                "completed_at": None,
                "budget_stop_reason": None,
                "budget_stops": None,
                "failed_because": None,
                "exhausted_because": None,
                "completeness": None,
                # The cap it was stopped under is replayed as named, so a cap it inherited keeps saying so unless the
                # take-up raised it.
                "max_cost_usd": fresh.max_cost_usd,
                "max_cost_usd_origin": stored.max_cost_usd_origin
                if fresh.max_cost_usd == stored.max_cost_usd
                else fresh.max_cost_usd_origin,
                # The metered-call ceiling is replayed as the run's launch named it, or inherited again from the host,
                # which may have raised it since: the run records the one it now runs under.
                "max_metered_calls": fresh.max_metered_calls,
                "max_metered_calls_origin": fresh.max_metered_calls_origin,
                "lifetimes": [*ended, holding],
                # A run stored before runs froze their template's measured content passed only with its template
                # unedited since it began, so what the template measures now is what it measured: frozen from here.
                "template_digest": stored.template_digest or fresh.template_digest,
                # The checks were re-proved alike under the rules in force (`goal_check_proofs` is pinned), so the run
                # reads as proved under the newer of its own rules and those; never moved back by an older build.
                "goal_check_proof_rules": _proof_rules_after(stored, fresh),
                # Admitted on estimated rates in any holding is admitted on an estimate: the flag stays set once set. A
                # take-up that drops the cap leaves nothing to read against an estimate, and the field is not recorded
                # for a run with no cap.
                "cap_checked_on_estimates": None
                if fresh.max_cost_usd is None
                else True
                if stored.cap_checked_on_estimates or fresh.cap_checked_on_estimates
                else stored.cap_checked_on_estimates,
            }
        )
        return reopened, list(self.cases)

    def seed(self, cap: EvalRunCostCap, metered: MeteredCallLedger) -> None:
        """Count what the run's stored rows spent toward its cost cap, an unpriced one as unpriced, and the metered
        calls they landed toward its metered-call ceiling, with the calls they were refused toward its record of
        refusals: both ceilings bound the whole run, not each holding of it.

        :param cap: The cap the launch built for this holding.
        :ptype cap: EvalRunCostCap
        :param metered: The metered-call ledger the launch built for this holding.
        :ptype metered: MeteredCallLedger
        :raises ValidationFailedError: The stored rows have landed the metered calls the run's ceiling allows, so the
            take-up would run nothing.
        """
        for spend in _counted_spend(self.rows):
            cap.record(spend)
        metered.land_earlier(
            calls=sum(row.metered_calls or 0 for row in self.rows),
            refused=sum(row.metered_calls_refused or 0 for row in self.rows),
        )
        if metered.ceiling_reached and metered.ceiling is not None:
            raise _metered_ceiling_reached(self.run, landed=metered.landed_calls, ceiling=metered.ceiling)


def _metered_ceiling_reached(run: EvalRun, *, landed: int, ceiling: int) -> ValidationFailedError:
    """The refusal of a run whose stored cells have landed the metered calls its ceiling allows.

    A take-up names a cost cap and never a metered-call ceiling: one the run inherited moves with the host's setting,
    and one its launch chose moves only with a new launch.
    """
    remedy = (
        "launch its template again (run_launch) with a higher max_metered_calls: a resume keeps the ceiling the run's "
        "launch chose, and cannot raise it"
        if run.max_metered_calls_origin == "chosen"
        else f"raise the host's max_metered_calls above {landed} and call job_resume again; a resume cannot raise a "
        "metered-call ceiling itself"
    )
    return ValidationFailedError(
        f"run '{run.id}' has landed {landed} metered call(s) against its metered-call ceiling of {ceiling}, so taking "
        "it up would run nothing — at least that many, since a row stored before rows counted their metered calls "
        "counts none",
        remedy=remedy,
    )


@dataclass(frozen=True)
class _Read:
    """What a take-up reads before it decides: the run settled, its etag, its rows, its cases and its launch's size."""

    run: EvalRun
    etag: str | None
    rows: _StoredRows
    cases: tuple[EvalTestCase, ...]
    missing_cases: list[str]
    group_size: int


def _read(storage: EvalStorage, run_id: str, scope_id: str) -> _Read:
    """Settle the run as every reader does, then read it with its etag, its rows, its cases and its launch's runs.

    :raises NotFoundError: There is no such run in the scope.
    """
    run, etag = read_to_take_up(storage, RUN_JOB, run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id, remedy=FIND_RUN)
    rows = _stored_rows(storage, run)
    loaded = {case_id: storage.load_test_case(case_id, scope_id) for case_id in dict.fromkeys(run.test_case_ids)}
    group_size = (
        len(storage.launch_group_run_ids(run.launch_group_id, scope_id)) if run.launch_group_id is not None else 1
    )
    return _Read(
        run=run,
        etag=etag,
        rows=rows,
        cases=tuple(case for case in loaded.values() if case is not None),
        missing_cases=[case_id for case_id, case in loaded.items() if case is None],
        group_size=group_size,
    )


def _status_not_resumable(run: EvalRun) -> ValidationFailedError:
    """The refusal of a run whose status is not one :data:`_RESUMABLE_RUN_STATUSES` names, saying why its status is
    not."""
    if run.status not in TERMINAL_RUN_STATUSES:
        holder = f" by process {run.lease.holder} until {run.lease.expires_at}" if run.lease is not None else ""
        return ValidationFailedError(
            f"run '{run.id}' is still {run.status}{holder}; a live run is never taken up",
            remedy="cancel it (job_cancel) or wait for it to end, then call job_resume",
        )
    if run.status == "failed":
        return ValidationFailedError(
            f"run '{run.id}' failed ({run.failed_because or 'see its error_details'}); taking it up again would run "
            "what failed it again",
            remedy="fix what failed it, which run_get names, and launch its template again (run_launch)",
        )
    return ValidationFailedError(
        f"run '{run.id}' {run.status}; there is nothing left of it to run",
        remedy="launch its template again (run_launch) to measure again",
    )


def _why_not_taken_up(
    run: EvalRun,
    rows: Sequence[StoredCell],
    *,
    cap: float | None,
    metered_ceiling: int | None,
    live_here: bool = False,
    missing_cases: Sequence[str] = (),
) -> list[ValidationFailedError]:
    """Why a run cannot be taken up as it stands, each with how to proceed: the one rule a take-up refuses by and a
    listing of resumable runs notes.

    :param run: The run, settled.
    :ptype run: EvalRun
    :param rows: Every row stored for it.
    :ptype rows: Sequence[StoredCell]
    :param cap: The cost cap it would be taken up under.
    :ptype cap: float | None
    :param metered_ceiling: The metered-call ceiling it would be taken up under; ``None`` when none bounds it.
    :ptype metered_ceiling: int | None
    :param live_here: Whether its job in this process is still unwinding.
    :ptype live_here: bool
    :param missing_cases: The cases it froze that its scope no longer holds.
    :ptype missing_cases: Sequence[str]
    :return: Every reason, in the order a take-up meets them; empty when it can be taken up. A run whose status is not
        resumable has that reason alone.
    :rtype: list[ValidationFailedError]
    """
    if run.status not in _RESUMABLE_RUN_STATUSES:
        return [_status_not_resumable(run)]
    reasons: list[ValidationFailedError] = []
    if live_here:
        reasons.append(
            ValidationFailedError(
                f"run '{run.id}' ended, and its job in this process is still unwinding",
                remedy="call job_resume again once its job has stopped (job_poll reads it done)",
            )
        )
    spent = sum(amount for amount in _counted_spend(rows) if amount is not None)
    if run.status == "budget_stopped" and cap is not None and spent >= cap:
        reasons.append(
            ValidationFailedError(
                f"run '{run.id}' was stopped by its budget ({run.budget_stop_reason}), and its stored rows have spent "
                f"${round_half_even(spent, 4)} — at least that, since a cell in flight when it stopped may have spent more — against "
                f"a cap of ${round_half_even(cap, 4)}",
                remedy=f"call job_resume with max_cost_usd above ${round_half_even(spent, 4)}",
            )
        )
    if run.template_id is None:
        reasons.append(
            ValidationFailedError(
                f"run '{run.id}' is ad hoc, so there is no template to launch it from again",
                remedy="only a run launched from a template is resumed (job_resume); run this eval again to measure "
                "again",
            )
        )
    if missing_cases:
        reasons.append(
            ValidationFailedError(
                f"run '{run.id}' froze case(s) {', '.join(missing_cases)}, which scope {run.scope_id!r} no longer "
                "holds, so its cells cannot run",
                remedy="launch its template again (run_launch) over the cases the scope holds",
            )
        )
    landed = sum(row.metered_calls or 0 for row in rows)
    if metered_ceiling is not None and landed >= metered_ceiling:
        reasons.append(_metered_ceiling_reached(run, landed=landed, ceiling=metered_ceiling))
    return reasons


def _refuse_what_cannot_be_taken_up(read: _Read, *, cap: float | None, live_here: bool) -> None:
    """Refuse a run that cannot be taken up, with how to proceed (:func:`_why_not_taken_up`).

    Its metered-call ceiling is the launch's to resolve, so it is refused when the launch seeds the ceiling's ledger
    (:meth:`RunResumption.seed`), by the same rule.

    :raises ValidationFailedError: The run is live, completed, failed, stopped by its budget under a cap its spend has
        reached, ad hoc, or its cases are gone: the first reason a take-up meets.
    """
    reasons = _why_not_taken_up(
        read.run, read.rows.every, cap=cap, metered_ceiling=None, live_here=live_here, missing_cases=read.missing_cases
    )
    if reasons:
        raise reasons[0]


async def _run_resumption(
    host: LaunchHost, run_id: str, scope_id: str, *, max_cost_usd: float | None = None
) -> RunResumption:
    """Read a run as a take-up of it reads it, refuse it when it cannot be taken up, and plan its cells — writing,
    replaying and starting nothing.

    The run is settled as every reader settles it (written lost when its lease lapsed), refused when it cannot be
    taken up, and its cells planned (:func:`_planned_cells`) under its lease's next epoch.

    :param host: The host, whose job manager would hold it.
    :ptype host: LaunchHost
    :param run_id: The run.
    :ptype run_id: str
    :param scope_id: Its scope.
    :ptype scope_id: str
    :param max_cost_usd: The cost cap to take it up under (see :func:`resume_run`); ``None`` keeps the one it recorded.
    :ptype max_cost_usd: float | None
    :return: What a take-up of it starts from.
    :rtype: RunResumption
    :raises NotFoundError: There is no such run.
    :raises ValidationFailedError: It cannot be taken up (see :func:`resume_run`).
    """
    eval_host = host.eval_host
    leases = host.job_manager.leases
    read = await run_blocking(eval_host.blocking_executor, _read, eval_host.storage, run_id, scope_id)
    run = read.run
    cap = max_cost_usd if max_cost_usd is not None else run.max_cost_usd
    _refuse_what_cannot_be_taken_up(read, cap=cap, live_here=host.job_manager.is_active(run_id))
    return RunResumption(
        run=run,
        cases=read.cases,
        stored=_planned_cells(read.rows),
        rows=read.rows.every,
        holder=leases.holder_id if leases is not None else None,
        epoch=next_epoch(run.lease, run.lost, (lifetime.epoch for lifetime in run.lifetimes)),
        # One arm of an interleaved launch played its cases in the order its group's id gave every arm; a group of one
        # never interleaved, and played them in its own run's order.
        cell_order_seed=run.launch_group_id if run.arm_schedule == "interleaved" and read.group_size > 1 else None,
        etag=read.etag,
        max_cost_usd=cap,
    )


class TakenUp[R](NamedTuple):
    """A take-up started, and what admitting it told its caller beside starting it.

    :ivar record: The run, or the launch, as its take-up started it.
    :ivar notices: Each sentence admission told the caller and did not refuse for, such as a served-model check
        that could not be made because the provider's model list could not be read; empty when there was none.
    """

    record: R
    notices: tuple[str, ...]


@dataclass(frozen=True)
class TakeUp:
    """Runs taken up again together, as every take-up takes them up: one run's (:func:`take_up_run`), a launch's
    (:func:`resume_launch`) and a sweep's arms' (:func:`~threetears.evals.ops.sweeps.sweep_resume`).

    Each run was settled, refused when it cannot be taken up, and planned (:func:`_run_resumption`); a take-up of
    several runs also replayed and compared each one's launch in the call (:func:`take_up_arms`), so one that would
    measure otherwise refused them all before any was started. Here each is handed to a group: prepared into one its
    caller starts (:meth:`prepare`), its reopened record written as that group starts (:meth:`reopen`), or started in
    a group of its own (:meth:`start`).

    :ivar host: The host, whose launcher wires each run and whose job manager holds it.
    :ivar arms: Each run's take-up, as planned and checked, by its arm's label, in arm order.
    """

    host: LaunchHost
    arms: Mapping[str, RunResumption]

    @property
    def resumptions(self) -> list[RunResumption]:
        """Every run's take-up, in arm order.

        :return: The take-ups.
        :rtype: list[RunResumption]
        """
        return list(self.arms.values())

    @property
    def notices(self) -> tuple[str, ...]:
        """What checking every run's take-up told the caller beside admitting it, each once, in arm order.

        :return: The notices; empty when there was nothing to tell.
        :rtype: tuple[str, ...]
        """
        return tuple(dict.fromkeys(notice for resuming in self.arms.values() for notice in resuming.notices))

    async def prepare(self, label: str, group: LaunchGroup) -> list[EvalRun]:
        """Prepare one arm's run into ``group``, its launch replayed and compared, beside the other arms its caller
        takes up or launches with it; the caller admits and starts the group, reopening the run as it starts
        (:meth:`reopen`).

        :param label: The arm's label.
        :ptype label: str
        :param group: The group its caller starts.
        :ptype group: LaunchGroup
        :return: The run, prepared.
        :rtype: list[EvalRun]
        :raises ValidationFailedError: It would measure otherwise now (every difference named), or a launch of it would
            be refused now.
        :raises NotFoundError: Its template is gone.
        """
        return await (await admit_resumed_run(self.host, self.arms[label], launch_group=group)).start()

    async def reopen(self, runs: Sequence[EvalRun]) -> None:
        """Write each run of ``runs`` that is taken up again, reopened, under its next epoch, conditionally on the
        record its take-up read; a run launched beside them is written by its group's start, and is left alone here.

        :param runs: The prepared runs of the group about to start.
        :ptype runs: Sequence[EvalRun]
        :raises ConflictError: A run's record moved since it was read: another take-up won, or another writer wrote it.
        """
        eval_host = self.host.eval_host
        leases = self.host.job_manager.leases
        by_run = {resuming.run.id: resuming for resuming in self.arms.values()}

        def write() -> None:
            for run in runs:
                if (resuming := by_run.get(run.id)) is not None:
                    write_reopened(
                        eval_host.storage, RUN_JOB, run, leases=leases, etag=resuming.etag, epoch=resuming.epoch
                    )

        await run_blocking(eval_host.blocking_executor, write)

    async def start(self, label: str) -> TakenUp[EvalRun]:
        """Take one arm's run up again in a group of its own: its launch replayed and compared, the run reopened,
        started.

        :param label: The arm's label.
        :ptype label: str
        :return: The run, reopened and started, and what its admission told the caller
            (:attr:`AdmittedLaunch.notices <threetears.evals.run.launch.AdmittedLaunch.notices>`).
        :rtype: TakenUp[EvalRun]
        :raises ValidationFailedError: It would measure otherwise now (every difference named), or a launch of it would
            be refused now.
        :raises NotFoundError: Its template is gone.
        :raises ConflictError: Another take-up of it won, or another writer moved it; nothing was started.
        :raises AdmissionRefusedError: The host has no room for another run.
        """
        resuming = self.arms[label]
        admitted = await admit_resumed_run(self.host, resuming, attach=self.reopen)
        (started,) = await admitted.start()
        log.info(
            "eval.resume_run run=%s scope=%s taken up at epoch %d: %d cell(s) done, %d to run again over their rows",
            started.id,
            started.scope_id,
            resuming.epoch,
            len(resuming.stored.done),
            len(resuming.stored.replaced),
        )
        return TakenUp(started, admitted.notices)


async def resume_run(host: LaunchHost, run_id: str, scope_id: str, *, max_cost_usd: float | None = None) -> EvalRun:
    """Take a run up again: the cells it has not measured run, in its own order, and it ends as the same run.

    Resumable: a run that was lost (its holder gone, or shut down), one an operator cancelled, one its budget stopped
    — under a cap above what its stored rows spent, so a cost-capped stop needs a raised ``max_cost_usd`` — and one
    its provider account or its model's provider stopped (``exhausted``), which, like a budget, is resumable once the
    account is topped up or the model is served.
    A run that completed or failed is not, nor one still live.

    In order: the run is settled as every reader settles it (written lost when its lease lapsed); refused when it
    cannot be taken up; its cells planned (:func:`_run_resumption`); its launch replayed from what it recorded, through
    its kind's launcher (:func:`~threetears.evals.run.launch.admit_resumed_run`), and refused when what that launch
    assembles would measure otherwise (:func:`_resume_drift`); reopened under its lease's next epoch, conditionally on
    the record as read; and started in a group of its own, ordered as it was — by its launch's group when that
    interleaved several arms, else by its own id (:meth:`TakeUp.start`).

    :param host: The host, whose launcher wires the run and whose job manager holds it.
    :ptype host: LaunchHost
    :param run_id: The run.
    :ptype run_id: str
    :param scope_id: Its scope.
    :ptype scope_id: str
    :param max_cost_usd: The cost cap to take it up under, above what its stored rows spent; ``None`` keeps the cap
        it recorded, whatever the host's own reads now (a run recorded uncapped inherits the host's). Not part of the
        measurement: each holding records the cap it ran under. A cap above the host's ceiling is refused, as a
        launch's is.
    :ptype max_cost_usd: float | None
    :return: The run, reopened and started.
    :rtype: EvalRun
    :raises NotFoundError: There is no such run, or its template is gone.
    :raises ValidationFailedError: It cannot be taken up (see above), it would measure otherwise now (every
        difference named), or a launch of it would be refused now.
    :raises ConflictError: Another take-up of it won, or another writer moved it, while this one prepared; nothing was
        started.
    :raises AdmissionRefusedError: The host has no room for another run.
    """
    return (await take_up_run(host, run_id, scope_id, max_cost_usd=max_cost_usd)).record


async def take_up_run(
    host: LaunchHost, run_id: str, scope_id: str, *, max_cost_usd: float | None = None
) -> TakenUp[EvalRun]:
    """:func:`resume_run`, answering what its admission told the caller beside the run it started.

    What a transport's resume returns to its caller (``job_resume``): a served-model check skipped because the
    provider's model list could not be read is said there, not only logged.

    :param host: As :func:`resume_run` takes it.
    :ptype host: LaunchHost
    :param run_id: As :func:`resume_run` takes it.
    :ptype run_id: str
    :param scope_id: As :func:`resume_run` takes it.
    :ptype scope_id: str
    :param max_cost_usd: As :func:`resume_run` takes it.
    :ptype max_cost_usd: float | None
    :return: The run, reopened and started, and the notices.
    :rtype: TakenUp[EvalRun]
    :raises NotFoundError: As :func:`resume_run` raises it.
    :raises ValidationFailedError: As :func:`resume_run` raises it.
    :raises ConflictError: As :func:`resume_run` raises it.
    :raises AdmissionRefusedError: As :func:`resume_run` raises it.
    """
    resuming = await _run_resumption(host, run_id, scope_id, max_cost_usd=max_cost_usd)
    return await TakeUp(host, {run_id: resuming}).start(run_id)


async def _check_take_up(host: LaunchHost, resuming: RunResumption) -> tuple[str, ...]:
    """Replay a planned run's launch and compare it, as its take-up will, then release what the launch built.

    For a take-up of several runs that must refuse before any of them is started: each run's launch is replayed
    through its kind's launcher and compared with the stored run (:func:`_resume_drift`) here, in the call, and the
    clients it built are released. Nothing is written and nothing started; the take-up replays it again as it starts.

    :param host: The host, whose launcher wires the run.
    :ptype host: LaunchHost
    :param resuming: The run, as :func:`_run_resumption` planned it.
    :ptype resuming: RunResumption
    :return: What its admission told the caller beside admitting it (:attr:`LaunchGroup.notices
        <threetears.evals.run.launch.LaunchGroup.notices>`).
    :rtype: tuple[str, ...]
    :raises ValidationFailedError: It would measure otherwise now (every difference named), or a launch of it would be
        refused now.
    :raises NotFoundError: Its template is gone.
    """
    group = LaunchGroup(candidate_models=[resuming.run.candidate_model])
    try:
        await (await admit_resumed_run(host, resuming, launch_group=group)).start()
        return tuple(group.notices)
    finally:
        await group.abandon()


async def take_up_arms(
    host: LaunchHost, arms: Sequence[tuple[str, str]], scope_id: str, *, max_cost_usd: float | None = None
) -> tuple[TakeUp, list[ValidationFailedError]]:
    """Plan and check the take-up of every arm's run of a sweep or a launch, collecting each arm's refusal.

    Each run is planned (:func:`_run_resumption`) and its launch replayed and compared (:func:`_check_take_up`), so a
    caller refuses the whole take-up, naming every arm's every difference, before any arm is started.

    :param host: The host.
    :ptype host: LaunchHost
    :param arms: Each arm's label and its run's id, in arm order.
    :ptype arms: Sequence[tuple[str, str]]
    :param scope_id: Their scope.
    :ptype scope_id: str
    :param max_cost_usd: The cost cap each run is taken up under; ``None`` keeps each run's own.
    :ptype max_cost_usd: float | None
    :return: The take-up of every arm that passed, each carrying what checking it told the caller
        (:attr:`RunResumption.notices`), and each arm's refusal, its message naming the arm.
    :rtype: tuple[TakeUp, list[ValidationFailedError]]
    :raises NotFoundError: A run is gone, or its template is.
    """
    planned: dict[str, RunResumption] = {}
    refusals: list[ValidationFailedError] = []
    for label, run_id in arms:
        try:
            resuming = await _run_resumption(host, run_id, scope_id, max_cost_usd=max_cost_usd)
            notices = await _check_take_up(host, resuming)
        except ValidationFailedError as refused:
            refusals.append(ValidationFailedError(f"arm {label!r}: {refused.message}", remedy=refused.remedy))
            continue
        planned[label] = replace(resuming, notices=notices)
    return TakeUp(host, planned), refusals


def refused_as_one(job: str, refusals: Sequence[ValidationFailedError]) -> ValidationFailedError:
    """The refusals of a take-up of several arms, as one refusal naming every one of them.

    :param job: The job taken up, as a refusal names it (``sweep '<id>'``).
    :ptype job: str
    :param refusals: Each arm's refusal, its message naming the arm.
    :ptype refusals: Sequence[ValidationFailedError]
    :return: The one refusal; its remedy is each distinct remedy, in order.
    :rtype: ValidationFailedError
    """
    return ValidationFailedError(
        f"{job} is not taken up again, so none of its arms is: {'; '.join(refused.message for refused in refusals)}",
        remedy="; ".join(dict.fromkeys(refused.remedy for refused in refusals if refused.remedy)),
    )


def _refuse_a_job_not_taken_up(
    kind: str, record_id: str, outcome: str, detail: str | None, *, live_here: bool, start_again: str
) -> None:
    """Refuse a take-up of a sweep's or a launch's job that has not ended, or ended with nothing to take up.

    Taken up: one lost (its process gone), one cancelled by an operator, and one whose own budget ran out
    (``timed_out``), which is a ceiling, not a defect. Not: one still running, one whose job in this process is still
    unwinding, one that completed, and one that failed, which would fail again.

    :param kind: The job's kind, as a refusal names it.
    :ptype kind: str
    :param record_id: Its record's id.
    :ptype record_id: str
    :param outcome: How its record ended, or ``running``.
    :ptype outcome: str
    :param detail: Why it ended, as its record says.
    :ptype detail: str | None
    :param live_here: Whether its job in this process is still unwinding.
    :ptype live_here: bool
    :param start_again: The action that starts the work again, named in a remedy.
    :ptype start_again: str
    :raises ValidationFailedError: It cannot be taken up.
    """
    if outcome not in _RESUMABLE_JOB_OUTCOMES:
        if outcome == "running":
            raise ValidationFailedError(
                f"{kind} '{record_id}' is still running; a live {kind} is never taken up",
                remedy="cancel it (job_cancel) or wait for it to end, then call job_resume",
            )
        if outcome == "failed":
            raise ValidationFailedError(
                f"{kind} '{record_id}' failed ({detail or 'see job_poll'}); taking it up again would run what failed "
                "it again",
                remedy=f"fix what failed it, which job_poll names, and start it again ({start_again})",
            )
        raise ValidationFailedError(
            f"{kind} '{record_id}' {outcome}; there is nothing left of it to run",
            remedy=f"start it again ({start_again}) to measure again",
        )
    if live_here:
        raise ValidationFailedError(
            f"{kind} '{record_id}' ended, and its job in this process is still unwinding",
            remedy="call job_resume again once its job has stopped (job_poll reads it done)",
        )


class _JobEnding(Protocol):
    """A sweep's or a launch's record, as a take-up of its job reads how it ended."""

    @property
    def id(self) -> str: ...

    @property
    def outcome(self) -> str: ...

    @property
    def detail(self) -> str | None: ...


async def read_job_to_take_up[R: EvalDocumentModel](
    host: LaunchHost, kind: LeasedJobKind[R], record_id: str, scope_id: str, *, start_again: str
) -> tuple[R, str | None]:
    """Read a sweep's or a launch's record as a take-up of its job reads it, refusing one that cannot be taken up.

    The record is settled as every reader settles it (:func:`~threetears.evals.kernel.job_leases.read_to_take_up`),
    then read with the token its reopen is written conditionally on. Taken up: a job lost (its process gone), one
    cancelled by an operator, and one whose own budget ran out (``timed_out``), which is a ceiling, not a defect. Not:
    one still running, one whose job in this process is still unwinding, one that completed, and one that failed,
    which would fail again.

    :param host: The host, whose job manager would hold the job.
    :ptype host: LaunchHost
    :param kind: The record's type: :data:`~threetears.evals.kernel.job_leases.SWEEP_JOB` or
        :data:`~threetears.evals.run.launch_job.LAUNCH_JOB`.
    :ptype kind: LeasedJobKind[R]
    :param record_id: The record.
    :ptype record_id: str
    :param scope_id: Its scope.
    :ptype scope_id: str
    :param start_again: The action that starts the work again, named in a remedy.
    :ptype start_again: str
    :return: The record, settled, and its token.
    :rtype: tuple[R, str | None]
    :raises NotFoundError: There is no such record in the scope.
    :raises ValidationFailedError: Its job cannot be taken up.
    """
    eval_host = host.eval_host
    record, etag = await run_blocking(
        eval_host.blocking_executor, read_to_take_up, eval_host.storage, kind, record_id, scope_id
    )
    if record is None:
        raise NotFoundError(kind.name, record_id, remedy=FIND_JOB)
    job = cast(_JobEnding, record)
    _refuse_a_job_not_taken_up(
        kind.name,
        job.id,
        job.outcome,
        job.detail,
        live_here=host.job_manager.is_task_active(job.id),
        start_again=start_again,
    )
    return record, etag


async def resume_launch(
    host: LaunchHost, launch_id: str, scope_id: str, *, max_cost_usd: float | None = None
) -> TakenUp[EvalLaunch]:
    """Take a launch up again as the same launch: its unfinished runs taken up together, as one group under its id.

    A launch's job that was lost, cancelled, or ran out of its preparation's budget after its runs started is taken up
    again. Every arm of a launch has a run from the moment its group starts, so there is no arm to launch anew: each
    run that has not ended ``completed`` or ``failed`` is taken up (:func:`_run_resumption`), and every one is replayed
    and compared in the call (:func:`take_up_arms`), so one run that would measure otherwise refuses the whole take-up,
    naming every difference of every run, before anything is spent. Then the launch's record is reopened under its
    lease's next epoch, conditionally on the record as read, and its job starts the runs as one group whose id is the
    launch's own, so an interleaved launch's arms take their remaining cells in turn in the order they always had,
    and waits for every run, ending as a launch does.

    :param host: The host, whose launcher wires the runs and whose job manager holds them.
    :ptype host: LaunchHost
    :param launch_id: The launch.
    :ptype launch_id: str
    :param scope_id: Its scope.
    :ptype scope_id: str
    :param max_cost_usd: The cost cap every run is taken up under; ``None`` keeps each run's own (see
        :func:`resume_run`). Each run's holding records the cap it ran under.
    :ptype max_cost_usd: float | None
    :return: The launch's record as its job started, and what checking its runs' take-up told the caller beside
        admitting them (a served-model check that could not be made), each once.
    :rtype: TakenUp[EvalLaunch]
    :raises NotFoundError: There is no such launch, or a run's template is gone.
    :raises ValidationFailedError: The launch cannot be taken up (still running, completed, failed, or it started no
        run), every run it started has ended, or a run cannot be taken up — every run's refusal named.
    :raises ConflictError: Another take-up of the launch or of one of its runs won, or another writer moved it;
        nothing was started.
    :raises AdmissionRefusedError: The host has no room for the runs.
    """
    eval_host, manager = host.eval_host, host.job_manager
    storage = eval_host.storage
    launch, etag = await read_job_to_take_up(host, LAUNCH_JOB, launch_id, scope_id, start_again="run_launch")
    if not launch.run_ids:
        raise ValidationFailedError(
            f"launch '{launch.id}' ended before it started any run ({launch.detail}), so there is nothing to take up",
            remedy="launch the template again (run_launch)",
        )
    runs = await run_blocking(eval_host.blocking_executor, storage.load_eval_runs, launch.run_ids, scope_id)
    by_id = {run.id: run for run in runs}
    unfinished = [
        (by_id[run_id].candidate_model, run_id)
        for run_id in launch.run_ids
        if run_id in by_id and by_id[run_id].status not in ("completed", "failed")
    ]
    if not unfinished:
        raise ValidationFailedError(
            f"every run launch '{launch.id}' started has ended completed or failed, so there is nothing to take up",
            remedy="read each run with run_get",
        )
    take_up, refusals = await take_up_arms(host, unfinished, scope_id, max_cost_usd=max_cost_usd)
    if refusals:
        raise refused_as_one(f"launch '{launch.id}'", refusals)
    # The launch's own id orders the runs' cells as it always did, and an interleaved launch takes them in turn.
    group = LaunchGroup(
        candidate_models=[resuming.run.candidate_model for resuming in take_up.resumptions],
        arm_schedule=launch.arm_schedule,
        group_id=launch.id,
    )

    async def form() -> tuple[LaunchGroup, None]:
        return group, None

    async def prepare(group: LaunchGroup, _formed: None) -> list[EvalRun]:
        prepared: list[EvalRun] = []
        for label in take_up.arms:
            prepared += await take_up.prepare(label, group)
        return prepared

    admitted = await admit_group(
        host,
        len(take_up.arms),
        settings=host.settings(),
        form=form,
        prepare=prepare,
        event="eval.resume_launch",
        attach=take_up.reopen,
    )
    reopened = launch.model_copy(update=reopened_job_fields())
    epoch = next_epoch(launch.lease, launch.lost)
    work = resumed_launch_work(host, admitted, reopened, epoch=epoch, etag=etag)
    try:
        started = await start_job_work(manager, work, store=storage, executor=eval_host.blocking_executor)
    except ConflictError as moved:
        raise taken_up_meanwhile(LAUNCH_JOB, launch.id) from moved
    log.info(
        "eval.resume_launch launch=%s scope=%s taken up at epoch %d: %d run(s) taken up",
        launch.id,
        scope_id,
        epoch,
        len(take_up.arms),
    )
    return TakenUp(started, take_up.notices)


#: The statuses a run :func:`resume_run` takes up has ended with: ``cancelled`` (lost, or cancelled by an operator),
#: ``budget_stopped`` and ``exhausted``. ``completed`` and ``failed`` are never taken up, nor a live run.
_RESUMABLE_RUN_STATUSES: frozenset[EvalRunStatus] = frozenset({"cancelled", "budget_stopped", "exhausted"})

#: How a resumable run's last holding ended: lost (its process went away), cancelled by an operator, stopped by its
#: budget, or stopped by its provider account or by its model not being served (``EvalRun.exhausted_because``).
ResumableEnding = Literal["lost", "cancelled", "budget_stopped", "exhausted"]


class ResumableJob(EvalBaseModel):
    """One run :func:`resume_run` would take up, as a host offers it: how it ended, how far it got, what it spent."""

    job_id: str = Field(description="The job id job_resume takes: 'run:<run id>'.")
    run_id: str
    scope_id: str
    template_id: str
    candidate_model: str
    status: EvalRunStatus = Field(
        description="Its stored status: cancelled (lost, or by an operator), budget_stopped or exhausted."
    )
    ended: ResumableEnding = Field(
        description="How its last holding ended: lost (its process went away), cancelled by an operator, stopped by "
        "its budget, or stopped by its provider account or its model not being served (exhausted)."
    )
    reason: str | None = Field(
        default=None,
        description="Why, in its record's words: the loss's cause, the cancellation's reason, the budget stop's, or "
        "the account's refusal.",
    )
    cells_done: int = Field(ge=0, description="Cells whose stored row counts as done; a resume runs the rest.")
    cells_expected: int = Field(ge=0, description="Every cell of its matrix: its cases times its repeats.")
    spent_usd: float = Field(
        description="What its stored rows spent, every attempt once: what its cost cap counts on a resume. A lower "
        "bound when a holding was lost with cells in flight."
    )
    unpriced: int = Field(ge=0, description="Attempts its rows record that nobody priced, which spent_usd leaves out.")
    max_cost_usd: float | None = Field(description="The cost cap it ran under; None for uncapped.")
    lifetimes: int = Field(ge=1, description="The process lifetimes it has spanned: 1 until it is first resumed.")
    note: str | None = Field(
        default=None,
        description="What a resume needs beyond the job id: for a run its budget stopped at its cost cap, a "
        "max_cost_usd above what it spent; for one whose stored cells reached its metered-call ceiling, a higher "
        "ceiling, which a resume cannot name. None when job_resume needs nothing more.",
    )


#: How a sweep's or a launch's job that :func:`~threetears.evals.ops.job_resume` takes up has ended: ``cancelled``
#: (lost, or cancelled by an operator) or ``timed_out`` (its own budget ran out, a ceiling rather than a defect).
_RESUMABLE_JOB_OUTCOMES: frozenset[str] = frozenset({"cancelled", "timed_out"})


class ResumableGroup(EvalBaseModel):
    """One sweep or launch a take-up would continue as a whole, as a host offers it: how it ended, and its arms."""

    job_id: str = Field(description="The job id job_resume takes: 'sweep:<sweep id>' or 'launch:<launch id>'.")
    kind: Literal["sweep", "launch"]
    scope_id: str
    template_id: str
    ended: Literal["lost", "cancelled", "timed_out"] = Field(
        description="How its last holding ended: lost (its process went away), cancelled by an operator, or timed out "
        "(its own budget ran out)."
    )
    reason: str | None = Field(default=None, description="Why, in its record's words: the loss's cause, or its detail.")
    arms: dict[str, str] = Field(
        description="Each arm by its label (a launch's by its model), in arm order: its run's stored status; "
        "not_launched for an arm of a sweep that never launched, which a resume launches; or not_found for a run the "
        "scope no longer holds, which a resume refuses."
    )
    runs: list[ResumableJob] = Field(
        description="Each arm's run a resume takes up again with it: its cells, its spend and its cap. A run listed "
        "here is not listed again among the runs."
    )
    note: str | None = Field(
        default=None,
        description="What keeps job_resume from taking it up as it stands, when something does: an arm that never "
        "launched, of a sweep stored before sweeps recorded their arms' launch arguments. None when nothing does.",
    )


class ResumableJobs(EvalBaseModel):
    """The runs, sweeps and launches in the scopes asked about that a host may offer to resume (:func:`resumable_jobs`)."""

    jobs: list[ResumableJob] = Field(
        description="Each run job_resume would take up on its own, newest first within a scope; a run of a listed "
        "sweep or launch is listed under it instead."
    )
    groups: list[ResumableGroup] = Field(
        default_factory=list,
        description="Each sweep and each launch job_resume would take up as a whole, newest first within a scope.",
    )
    unreadable: int = Field(
        default=0,
        description="Rows that matched which this build's model cannot read (runs, sweeps or launches), left out; "
        "runs_unreadable names the runs.",
    )
    drift: Literal["checked_on_resume"] = Field(
        default="checked_on_resume",
        description="Whether each run would still measure as it did is not read here: that means replaying its launch "
        "through the host's kind. job_resume checks it before anything is spent, and refuses naming every difference.",
    )


def _ended(run: EvalRun) -> ResumableEnding:
    """How a resumable run's last holding ended."""
    if run.lost is not None:
        return "lost"
    if run.status == "budget_stopped":
        return "budget_stopped"
    return "exhausted" if run.status == "exhausted" else "cancelled"


def _offered(run: EvalRun, template_id: str, stored: _StoredRows) -> ResumableJob:
    """One resumable run as it is offered, read off its record and its stored rows."""
    rows = stored.every
    spend = _counted_spend(rows)
    spent = sum(amount for amount in spend if amount is not None)
    cap = run.max_cost_usd
    # The take-up's own rules, against the cap and the metered-call ceiling the run recorded: one it inherited moves
    # with the host's setting, which a take-up reads afresh. Its cases and its job's state are not read here.
    ceiling = run.max_metered_calls if run.max_metered_calls_origin != "none_declared" else None
    reasons = _why_not_taken_up(run, rows, cap=cap, metered_ceiling=ceiling or None)
    note = "; ".join(f"{reason.message}: {reason.remedy}" for reason in reasons) or None
    return ResumableJob(
        # The run job's id, in the shape job_poll and job_resume read.
        job_id=f"run:{run.id}",
        run_id=run.id,
        scope_id=run.scope_id,
        template_id=template_id,
        candidate_model=run.candidate_model,
        status=run.status,
        ended=_ended(run),
        reason=run_ending_reason(run),
        cells_done=len(stored.done),
        cells_expected=run.expected_cells,
        spent_usd=spent,
        unpriced=sum(1 for amount in spend if amount is None),
        max_cost_usd=cap,
        lifetimes=max(1, len(run.lifetimes)),
        note=note,
    )


def _ended_job(record: EvalSweep | EvalLaunch) -> Literal["lost", "cancelled", "timed_out"]:
    """How a resumable sweep's or launch's last holding ended."""
    if record.lost is not None:
        return "lost"
    return "timed_out" if record.outcome == "timed_out" else "cancelled"


def _offered_group(
    record: EvalSweep | EvalLaunch,
    arms: Sequence[tuple[str, str | None]],
    runs: Mapping[str, EvalRun],
    offered: Mapping[str, ResumableJob],
    *,
    note: str | None,
) -> ResumableGroup | None:
    """One resumable sweep or launch as it is offered; ``None`` when it has no arm left to take up or launch."""
    states = {
        label: "not_launched" if run_id is None else runs[run_id].status if run_id in runs else "not_found"
        for label, run_id in arms
    }
    taken = [offered[run_id] for _, run_id in arms if run_id is not None and run_id in offered]
    if not taken and "not_launched" not in states.values():
        return None
    kind: Literal["sweep", "launch"] = "sweep" if isinstance(record, EvalSweep) else "launch"
    return ResumableGroup(
        job_id=f"{kind}:{record.id}",
        kind=kind,
        scope_id=record.scope_id,
        template_id=record.template_id,
        ended=_ended_job(record),
        reason=record.lost.cause if record.lost is not None else record.detail,
        arms=states,
        runs=taken,
        note=note,
    )


def _never_launched_note(sweep: EvalSweep) -> str | None:
    """Why a take-up of a sweep refuses an arm that never launched, when one of its arms is one: the sweep was stored
    before sweeps recorded their arms' launch arguments."""
    stranded = [
        arm.label for arm in sweep.arms if arm.run_id is None and (sweep.arguments is None or arm.arguments is None)
    ]
    if not stranded:
        return None
    return (
        f"arm(s) {', '.join(map(repr, stranded))} never launched, and the sweep was stored before sweeps recorded "
        "their arms' launch arguments, so job_resume refuses it: launch those arms with run_launch, and resume each "
        "other arm by its run's job id"
    )


def resumable_jobs(storage: EvalStorage, scopes: Iterable[str]) -> ResumableJobs:
    """The runs, sweeps and launches in ``scopes`` that a take-up would continue, for a host to offer: nothing is
    resumed here.

    A run lost (its holder gone, or shut down), one an operator cancelled, one its budget stopped and one its provider
    account or its model's provider stopped (``exhausted``); never one that completed or failed, one still live, an ad hoc run, which has no
    template to launch it from again, nor an archived run, which an operator declined by archiving it. A sweep or a
    launch whose every run is archived is left out with them. A sweep or a launch lost, cancelled or timed out, with an
    arm left to take up or to launch, is listed once, as a whole (``groups``), with each arm's state and each run it
    takes up again; those runs are not listed again on their own. A record still stored running whose lease has lapsed
    is read lost, as every read reads it, and listed. A host reads this at boot, after
    :func:`~threetears.evals.run.lifecycle.sweep_abandoned_runs` has settled what its last process left running, and
    offers each to an operator: nothing is taken up until someone asks (``job_resume``).

    What a take-up would refuse on what it reads beyond the record is not checked here: whether a run would still
    measure as it did, which means replaying its launch through the host's kind (``job_resume`` checks it before
    anything is spent, naming every difference), and whether its frozen cases are still in its scope. A run its budget
    stopped at its cap carries a ``note`` naming the cap it needs, and a sweep whose never-launched arm cannot be
    launched carries one naming it.

    :param storage: The jobs' store.
    :ptype storage: EvalStorage
    :param scopes: The scopes to read, which the host names, as it names them to the boot sweep.
    :ptype scopes: Iterable[str]
    :return: The runs, the sweeps and launches, and how many rows matched that this build cannot read.
    :rtype: ResumableJobs
    :raises UnreadableRunError: A run read, but writing it lost was refused by a model; or a run a listed sweep or
        launch names cannot be read.
    :raises StorageError: The store failed to answer.
    """
    jobs: list[ResumableJob] = []
    groups: list[ResumableGroup] = []
    unreadable = 0
    for scope_id in scopes:
        read = storage.query_eval_runs_in_status(scope_id, _RESUMABLE_RUN_STATUSES)
        sweeps, unread_sweeps = storage.query_job_records_in_outcome(EvalSweep, scope_id, _RESUMABLE_JOB_OUTCOMES)
        launches, unread_launches = storage.query_job_records_in_outcome(EvalLaunch, scope_id, _RESUMABLE_JOB_OUTCOMES)
        unreadable += len(read.unreadable) + unread_sweeps + unread_launches
        # An archived run is one an operator declined: it is never offered, nor a sweep or launch whose every run is.
        resumable = {run.id: run for run in read.runs if run.template_id is not None and not run.archived}
        named = [arm.run_id for sweep in sweeps for arm in sweep.arms if arm.run_id is not None]
        named += [run_id for launch in launches for run_id in launch.run_ids]
        runs = dict(resumable)
        runs.update((run.id, run) for run in storage.load_eval_runs([i for i in named if i not in runs], scope_id))
        # Each run's rows read a page at a time and kept as the scalars an offer reads, one run after another, so the
        # listing never holds a run's rows whole, nor every run's at once.
        offered = {
            run.id: _offered(run, run.template_id, _stored_rows(storage, run))
            for run in sorted(resumable.values(), key=lambda run: run.created_at, reverse=True)
            if run.template_id is not None
        }
        grouped: set[str] = set()
        candidates: list[tuple[EvalSweep | EvalLaunch, list[tuple[str, str | None]], str | None]] = [
            (sweep, [(arm.label, arm.run_id) for arm in sweep.arms], _never_launched_note(sweep)) for sweep in sweeps
        ]
        candidates += [
            (launch, [(runs[run_id].candidate_model, run_id) for run_id in launch.run_ids if run_id in runs], None)
            for launch in launches
            if launch.run_ids
        ]
        for record, arms, note in sorted(candidates, key=lambda candidate: candidate[0].created_at, reverse=True):
            launched = [runs[run_id] for _, run_id in arms if run_id is not None and run_id in runs]
            if launched and all(run.archived for run in launched):
                continue
            if (group := _offered_group(record, arms, runs, offered, note=note)) is not None:
                groups.append(group)
                grouped.update(run_id for _, run_id in arms if run_id is not None)
        jobs += [job for run_id, job in offered.items() if run_id not in grouped]
    return ResumableJobs(jobs=jobs, groups=groups, unreadable=unreadable)


__all__ = [
    "ResumableEnding",
    "ResumableGroup",
    "ResumableJob",
    "ResumableJobs",
    "RunResumption",
    "TakeUp",
    "next_epoch",
    "read_job_to_take_up",
    "refused_as_one",
    "resumable_jobs",
    "resume_launch",
    "resume_run",
    "take_up_arms",
    "take_up_run",
]
