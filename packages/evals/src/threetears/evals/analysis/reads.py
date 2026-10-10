"""The read lenses: comparison sets, pivot, frontier, history, budget, export, cost estimate and run comparisons.

Every operation here answers a question about runs that already exist — which may be compared, how
they aggregate over two coordinates, which variant is cheapest above a bar, how a measure moved over
time, what the program spent, which runs no campaign holds, what a run scored, what changed between
two runs, and what a proposed sweep would cost. The rules each answer applies live in
:mod:`threetears.evals.analysis.reporting`; these functions are the seam between that pure layer and a
caller: they read the runs and results, normalize the caller's raw arguments once, and turn a
refusal of the question into ``ValidationFailedError``. They are module functions whose dependencies
are parameters, so a host calls them with its own store and its own bindings, and every surface the
host serves reaches them through that one delegation.

**What arrives as an argument.** Results, campaigns and single whole runs are read through
:class:`LensStore`, which a host's store satisfies by having the methods; a missing run or campaign
is refused here, as ``NotFoundError``. Runs are listed through ``list_runs`` (:data:`RunLister`)
rather than read here, because listing a scope's runs is the run package's — it owns the status
vocabulary's "every run" default, the archival exclusion and the host's listing elisions — and this
package may not import it. Two loads arrive as the caller's own loader, because each does more than
load-or-refuse-a-missing-id: a template (``load_template``, whose load also refuses a template
presuming world state the host no longer declares) and a run read the way a listing reads it
(``load_run_listed``, which leaves out the payload paths the host's listing elides — the run
package's rule). What a host adds to an answer arrives as a callable too: per-group columns on a run
summary (:data:`RowColumns`), a run's subject detail for the config diff, and how many cases a launch
of a template would run, which depends on where the host's kinds take their cases from.

**Two runs are compared by :func:`compare_two_runs`**, which the ``runs_compare`` operation exposes with
the completeness, clock and cassette disclosures every comparison carries.
:func:`bisect_runs` (which versioned inputs differ between two runs) is internal: the bundle's confound
scan and ``runs_compare``'s disclosures answer its question on the public surface, and nothing exposes it.

The scope a run lives in is ``scope_id`` here — the engine's word for a partition it never
interprets. The host chooses what a scope is and passes it through.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import ValidationError

from threetears.evals.analysis.contention import marked_latency_sentence, withheld_latency
from threetears.evals.analysis.reporting import (
    DEFAULT_WEIGHTING,
    METRIC_COMPOSITE,
    CostEstimate,
    PlannedCost,
    FrontierError,
    PivotError,
    PivotTable,
    compute_comparison_sets,
    compute_frontier,
    compute_pivot,
    normalize_bar,
    pooled_composite_basis,
    pooled_served_models,
    project_score_records,
)
from threetears.evals.analysis.lenses.history import HistoryError, HistoryResult, compute_history
from threetears.evals.analysis.lenses.program_budget import compute_program_budget
from threetears.evals.analysis.lenses.orphaned_runs import compute_orphaned_runs
from threetears.evals.analysis.lenses.export import ExportError, ScoreExport, export_projection
from threetears.evals.analysis.significance import cross_subject_disclosure
from threetears.evals.analysis.completeness import completeness_disclosure
from threetears.evals.analysis.stats import INTERVAL_LEVEL, composite_significance, difference_interval
from threetears.evals.kernel.arguments import normalize_blank
from threetears.evals.kernel.errors import NotFoundError, ValidationFailedError
from threetears.evals.kernel.host.profile import HostProfile
from threetears.evals.kernel.scoring import (
    compute_composite_summary,
    compute_async_delivery_summary,
    compute_cost_summary,
    compute_dimension_summary,
    compute_latency_summary,
    compute_pass_hat_k,
    compute_per_case_composites,
    pass_hat_k_at,
)
from threetears.evals.kernel.status_filter import StatusFilterError, normalize_status_filter
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.kernel.storage import EvalStorage
    from threetears.evals.kernel.campaign import EvalCampaign
    from threetears.evals.schema.models import EvalResult, EvalRun, EvalTemplate

log = get_logger(__name__)

#: Lists a scope's runs: ``list_runs(scope_id, *, status=None, include_archived=False)``.
#:
#: The run package's listing, passed in because this package may not import it. It resolves the
#: status filter with the listing's own default (unspecified means every run), excludes archived
#: runs unless asked, and leaves out the payload paths the host declares a listing omits — the
#: lenses read run scalars only, so a listed run is all they need.
RunLister = Callable[..., "list[EvalRun]"]

#: The host's own columns for each ``(model, run_id)`` group of a run's results, laid over a
#: :func:`run_summary` row after the engine's aggregates. A group the host has nothing for is absent.
RowColumns = Callable[["list[EvalResult]"], Mapping[tuple[str, str], Mapping[str, Any]]]

#: What a bisection reports for an input the OTHER run carried and this one did not — an
#: open family's members are whatever a given run overlaid, so the two runs' name sets are
#: not guaranteed to match. A recorded level ("this run left that knob alone"), never an
#: absence: reading it as unrecorded would make a real difference undecidable.
_NOT_CARRIED = "(not carried by this run)"


class LensStore(Protocol):
    """The storage reads the lenses make — results by scope and by run, campaigns, and one whole run.

    A scope's runs are not listed through it: they arrive through ``list_runs`` and the caller's
    listed-run loader, which carry the run package's listing rules. Structural, so a host's own storage satisfies it by
    having the methods — the engine's :class:`~threetears.evals.kernel.storage.EvalStorage` does. Positional
    parameters are positional-only, which lets the port say ``scope_id`` while an implementation
    names the thing it partitions by.
    """

    def query_eval_results(self, scope_id: str, /) -> list[EvalResult]:
        """Every result in a scope, without its trace payload."""
        ...

    def query_eval_results_by_run(self, run_id: str, scope_id: str, /) -> list[EvalResult]:
        """Every result one run produced."""
        ...

    def list_campaigns(self, scope_id: str, /) -> list[EvalCampaign]:
        """Every campaign in a scope, newest first."""
        ...

    def load_campaign(self, campaign_id: str, scope_id: str, /) -> EvalCampaign | None:
        """Load one campaign within a scope, or ``None`` when it does not resolve there."""
        ...

    def load_eval_run(self, run_id: str, scope_id: str, /) -> EvalRun | None:
        """Load one run within a scope, whole, or ``None`` when it does not resolve there."""
        ...


def _load_campaign(storage: LensStore, campaign_id: str, scope_id: str) -> EvalCampaign:
    """Load a campaign within a scope, refusing an unknown id as the campaign family does.

    Raises:
        NotFoundError: No campaign with that id in the scope.
    """
    campaign = storage.load_campaign(campaign_id, scope_id)
    if campaign is None:
        raise NotFoundError("campaign", campaign_id)
    return campaign


def _load_run(storage: LensStore, run_id: str, scope_id: str) -> EvalRun:
    """Load one run whole, refusing an unknown id as the run family does.

    Raises:
        NotFoundError: No run with that id in the scope.
    """
    run = storage.load_eval_run(run_id, scope_id)
    if run is None:
        raise NotFoundError("run", run_id)
    return run


def _score_delta(score_a: float | None, score_b: float | None) -> float | None:
    """Signed change B − A between two optional scores; ``None`` if either is missing."""
    if score_a is None or score_b is None:
        return None
    return score_b - score_a


def _status_filter(status: str | None) -> str | None:
    """Resolve a caller's run-status filter, refusing an unknown one as caller input.

    The one translation of :class:`~threetears.evals.kernel.status_filter.StatusFilterError`,
    shared by the five COMPARISON surfaces that take a ``status`` —
    ``comparison_sets``, ``pivot``, ``frontier``, ``history``, ``export_results``.
    Written once for the reason the normalizer itself is: five copies of one
    try/except is five places for the refusal to stop being raised, and the
    failure mode of forgetting one is the silent empty answer the refusal exists
    to remove — not an exception a test would notice.

    **``list_runs`` is a sixth status-taking surface and deliberately does NOT
    come through here**. Its ``None`` means "no filter at all", where
    this seam's ``None`` means "default to ``completed``", so routing it through
    the same normalizer would silently narrow an unfiltered listing to completed
    runs — hiding every failed, cancelled and budget-stopped run from the one
    surface an operator opens to find them. It goes through
    ``_listing_status_filter`` in :mod:`threetears.evals.run.reads` instead, which shares this seam's vocabulary
    and refusal and differs only in that default. Said here because this is where
    an auditor checks whether the class is closed: it is, across six surfaces and
    two defaults.

    Surfaced as caller input rather than a server fault, exactly as an unknown
    ``metric`` is: it is a question the caller can restate. REST renders the 422
    it carries; MCP renders it as the ``Error (VALIDATION_FAILED): …`` prose its
    handler decorator produces. One refusal, two spellings of it.

    Args:
        status: Raw filter from the caller, straight off the wire.

    Returns:
        ``None`` to include every run, else the status to filter on.

    Raises:
        ValidationFailedError: The value is not a status a run can carry, nor
            ``"all"``.
    """
    try:
        return normalize_status_filter(status)
    except StatusFilterError as e:
        raise ValidationFailedError(str(e)) from e


def _corpus_and_cohort(list_runs: RunLister, scope_id: str) -> tuple[list[EvalRun], list[EvalRun], set[str]]:
    """Split a scope's runs into everything that EXISTS and everything that COUNTS.

    The reporting layer needs both, and conflating them misreports curation as
    corruption. ``project_score_records`` / ``frontier`` / ``history`` take
    ``known_run_ids`` to tell two exclusion classes apart: a result whose run is
    in the corpus but not the cohort was excluded *on request*, while a result
    whose run is in neither cannot be placed at all and is counted
    ``results_without_run`` — a genuine integrity gap. Handing them the cohort
    as the corpus would report every archived run's results as unplaceable
    data, which is the confident-wrong answer this curation surface exists to
    stop producing.

    **The archived ids are returned rather than re-derived per caller**, because
    the projection uses them to separate the two *deliberate* exclusion classes
    — archived versus status-filtered — and four call sites each computing "the
    corpus minus the cohort" is exactly how those would drift into disagreeing
    about which narrowing dropped a run. One split, computed once, here.

    Args:
        list_runs: The run listing (:data:`RunLister`).
        scope_id: Partition key — the scope whose runs to split.

    Returns:
        ``(corpus, cohort, archived_run_ids)`` — the corpus is every run in the
        scope; the cohort is the corpus minus operator-archived runs; the
        third element is the ids of the runs that difference removed. Callers
        narrow the cohort further (by status, by explicit run ids) and pass the
        corpus's ids as ``known_run_ids`` and this set as ``archived_run_ids``.
    """
    corpus = list_runs(scope_id, include_archived=True)
    return corpus, [run for run in corpus if not run.archived], {run.id for run in corpus if run.archived}


def comparison_sets(
    storage: LensStore,
    scope_id: str,
    *,
    list_runs: RunLister,
    status: str | None = "completed",
    full_windows: bool = False,
    campaign_id: str | None = None,
    run_ids: list[str] | None = None,
    profile: HostProfile,
) -> dict[str, Any]:
    """Group a scope's runs into sets that may honestly be compared.

    Composed over ``list_runs``; the grouping rule itself lives in
    :mod:`threetears.evals.analysis.reporting` so both the REST route and the MCP action
    answer from one implementation rather than each inventing a rule.

    Takes its ``status`` raw, for the same reason :func:`pivot` does — one
    seam normalizes it, rather than each adapter doing so on the way in.

    **The scope's results are read too, not only its runs.** One caveat the
    grouping reports — **at least one PAIR** of a group's runs measured over
    non-overlapping spans of wall-clock time; the badge's condition is
    existential, and its disclosure counts how many of the group's pairs were
    disjoint rather than asserting that all of them were — is derived from the
    results' ``scored_at`` and cannot be answered
    from the run documents: a run is stamped ``created_at`` when it is
    enqueued, and a sweep's arms are enqueued within seconds of each other and
    executed over hours. Reading the results is what makes that badge
    decidable rather than silently clean, and it is the same scope-wide
    read :func:`pivot` and :func:`frontier` already perform.

    **A badge speaks for the runs it was computed over, and a reader is usually
    asking about a smaller set than the scope holds.** A campaign can
    hold three runs on one template while the group holds seven, spanning several days,
    and badge ``cassette_mode_differs`` because some non-members recorded with a
    cassette while every member recorded ``off`` — a caveat earned by runs
    the analysed set does not contain. ``campaign_id`` / ``run_ids`` narrow every
    group to the set actually under analysis, before anything is badged.

    **Resolving a campaign to its members happens HERE, not one layer down.**
    :func:`~threetears.evals.analysis.reporting.compute_comparison_sets` takes run ids and knows
    nothing about campaigns, deliberately: new eval machinery lands host-agnostic,
    and a campaign is a grouping concept
    the reporting layer has no reason to know. Turning a grouping concept into its members is this
    function's one lookup, through the store's ``load_campaign``.

    Args:
        storage: Where the scope's results and ``campaign_id``'s campaign are read (:class:`LensStore`).
        scope_id: Partition key — the scope whose runs to group.
        list_runs: The run listing (:data:`RunLister`), which owns the archival exclusion.
        status: Raw run-status filter, defaulting to ``"completed"`` because
            an in-flight run has no stable case set to compare on. ``"all"``
            groups every run regardless of status; anything that is neither
            is refused rather than answered with an empty grouping.
        full_windows: List every measurement span rather than summarising a
            large group's spans. Surfaces whose consumer renders the text itself
            pass ``True``; the chat surface leaves it off, since that is
            where an unbroken paragraph of spans stops being readable.
        campaign_id: Narrow the grouping to this campaign's member runs. The
            members are taken as the campaign STORES them, unfiltered: this
            method already narrows by ``status`` and inherits the archival
            exclusion from ``list_runs``, and intersecting a second,
            differently-defined narrowing here would leave a caller unable to
            say which one removed a run. A member that does not resolve — a run
            destroyed outside the delete cascade — is logged by the grouping
            rather than refused.
        run_ids: Narrow the grouping to these runs directly, for a caller whose
            set is not a campaign. Mutually exclusive with ``campaign_id``.
        profile: The host whose vocabulary this reads.

    Returns:
        A JSON-safe :class:`~threetears.evals.analysis.reporting.ComparisonSetsResult` —
        ``comparison_sets`` sorted by subject then template, plus
        ``out_of_scope_run_ids`` naming what a scope left out. A group whose
        every member is out of scope is not returned at all.

    Raises:
        ValidationFailedError: ``status`` is neither ``"all"`` nor a status a
            run can carry, or both ``campaign_id`` and ``run_ids`` were supplied
            — each is a question the caller can restate.
        NotFoundError: ``campaign_id`` names no campaign in the scope.
    """
    if campaign_id and run_ids:
        # Refused rather than intersected, or one silently preferred. Two scopes is an
        # ambiguous question, and answering a narrower question than the one asked is
        # the whole failure class this surface exists to end.
        raise ValidationFailedError(
            "comparison_sets takes campaign_id OR run_ids, not both — they are two different "
            "scopes, and answering one would silently drop the other"
        )
    # The campaign load raises NotFoundError before the scope-wide reads below, so an
    # unknown campaign costs one lookup rather than a full corpus scan answered empty.
    scope_run_ids = _load_campaign(storage, campaign_id, scope_id).run_ids if campaign_id else run_ids

    runs = list_runs(scope_id, status=_status_filter(status))
    results = storage.query_eval_results(scope_id)
    return compute_comparison_sets(
        runs, results=results, full_windows=full_windows, scope_run_ids=scope_run_ids, profile=profile
    ).model_dump(mode="json")


def pivot(
    storage: LensStore,
    scope_id: str,
    *,
    list_runs: RunLister,
    row_factor: str,
    column_factor: str,
    metric: str | None = None,
    weighting: str | None = None,
    subject_id: str | None = None,
    status: str | None = "completed",
    predicted_cost: CostEstimate | Sequence[PlannedCost] | Mapping[str, Any] | None = None,
    profile: HostProfile,
) -> PivotTable:
    """Aggregate a scope's observations over any two coordinates.

    Composed over ``list_runs`` plus the scope's results, projected
    through :func:`~threetears.evals.analysis.reporting.project_score_records`; the
    aggregation rule itself lives in :mod:`threetears.evals.analysis.reporting` so the
    REST route and the MCP action answer from one implementation.

    **Arguments arrive raw.** Blank-normalization, the status filter and the
    subject-filter assembly all happen here rather than at each adapter,
    because they were duplicated line-for-line across the REST route and the
    MCP handler — so a new parameter had to be remembered in both, and the
    parity gate only proves the surfaces agree about the capabilities it
    already covers. Weighting in particular was defaulted three times over
    (a REST ``Query`` default, an adapter ``normalize_blank``, and this
    method's own fallback), which is three places for one answer to change.

    Args:
        storage: Where the scope's results are read (:class:`LensStore`).
        scope_id: Partition key — the scope to aggregate over.
        list_runs: The run listing (:data:`RunLister`).
        row_factor: Coordinate to use as the row axis.
        column_factor: Coordinate to use as the column axis; the axis the
            Simpson's guard pools over rows.
        metric: Which observation-level measure to aggregate. ``None`` or
            blank takes the composite.
        weighting: Aggregation weighting; ``None`` or blank takes the
            disclosed default (equal per scenario).
        subject_id: Optional single-subject filter. Required to pivot a
            judge-mediated measure over a multi-subject scope.
        status: Raw run-status filter, defaulting to ``"completed"`` for the
            same reason as :func:`comparison_sets` — an in-flight run's cells
            are still arriving. ``"all"`` aggregates over every run.
        predicted_cost: The estimate the caller made before these runs, as
            :func:`~threetears.evals.analysis.reporting.compute_estimate_cost` returned it — the model, or its JSON
            form as a caller across a wire holds it — or the planned costs of a launch its host's pricer priced
            (:class:`~threetears.evals.analysis.reporting.PlannedCost`, which
            :func:`~threetears.evals.ops.launch_estimate`'s ``LaunchEstimate`` is). Each cost cell at a planned model
            then carries that model's predicted cost per observation beside the cost it observed. ``None`` shows
            observed cost alone.
        profile: The host whose vocabulary this reads.

    Returns:
        The :class:`~threetears.evals.analysis.reporting.PivotTable`.

    Raises:
        ValidationFailedError: The pivot cannot be answered honestly — an
            unknown metric, weighting, run status, axis or filter coordinate,
            or a cross-subject pooling that would average measurements derived
            from different rubrics; or a ``predicted_cost`` that is not an estimate, or
            that was handed to a pivot of another metric or with no model axis.
    """
    estimate: CostEstimate | list[PlannedCost] | None
    try:
        if predicted_cost is None:
            estimate = None
        elif isinstance(predicted_cost, Sequence):
            estimate = [PlannedCost.model_validate(planned) for planned in predicted_cost]
        else:
            estimate = CostEstimate.model_validate(predicted_cost)
    except ValidationError as e:
        raise ValidationFailedError(f"predicted_cost is not a cost estimate: {e}") from e
    metric = normalize_blank(metric, METRIC_COMPOSITE)
    weighting = normalize_blank(weighting, DEFAULT_WEIGHTING)
    filters = {"subject_id": subject_id} if subject_id else None
    status = _status_filter(status)

    # Every run in the scope is read, then narrowed here rather than in
    # the query, because the projection needs both sets to tell the two
    # exclusion classes apart: a result whose run this filter removed was
    # excluded on request, while one whose run is absent entirely cannot be
    # placed at all. Filtering in storage discards the distinction before
    # the projection can see it, which is what made `results_without_run`
    # report the routine `status="completed"` case as an integrity gap.
    # Mirrors `query_eval_runs`' own predicate: exact equality, and a blank
    # status means no filter. Archived runs leave the cohort the same way a
    # status filter removes one — excluded on request, still placeable — but
    # they are counted apart from it, because the two narrowings have
    # different remedies and a disclosure that names the wrong one sends the
    # reader after a fix that cannot work.
    all_runs, cohort, archived_run_ids = _corpus_and_cohort(list_runs, scope_id)
    runs = [run for run in cohort if run.status == status] if status else cohort
    results = storage.query_eval_results(scope_id)
    projection = project_score_records(
        runs, results, known_run_ids={run.id for run in all_runs}, archived_run_ids=archived_run_ids, profile=profile
    )
    try:
        table = compute_pivot(
            projection.records,
            row_factor=row_factor,
            column_factor=column_factor,
            metric=metric,
            weighting=weighting,
            filters=filters,
            exclusions=projection.exclusions,
            # From the projection rather than re-derived here: one place decides which
            # runs came up short, so the table's caveat and the rows it qualifies can
            # never describe different sets.
            completeness_disclosures=projection.completeness_disclosures,
            predicted_cost=estimate,
            profile=profile,
        )
    except PivotError as e:
        # Surfaced as caller-input rather than a server fault: every case is a
        # question the caller can restate (a different axis, a subject filter),
        # so a 500 would misattribute it and an empty table would read as
        # "no data" — the one answer that is definitely wrong.
        raise ValidationFailedError(str(e)) from e
    return table


def frontier(
    storage: LensStore,
    scope_id: str,
    *,
    list_runs: RunLister,
    bar: float | str | None = None,
    subject_id: str | None = None,
    status: str | None = "completed",
    profile: HostProfile | None = None,
    control_variant_key: str | None = None,
) -> dict[str, Any]:
    """Rank each subject's variants on quality x cost x latency, cheapest above bar.

    Composed over ``list_runs`` plus the scope's results and delegated
    to :func:`~threetears.evals.analysis.reporting.compute_frontier`, so the REST route and the
    MCP action answer from one implementation — the same shape as
    :func:`pivot`.

    **Arguments arrive raw.** The status filter and the bar coercion happen
    here, at the one seam, rather than at each adapter: the bar reaches REST
    as a validated float and MCP as the string a tool argument is, and
    normalizing in both places is exactly how the two surfaces drift.

    Args:
        storage: Where the scope's results are read (:class:`LensStore`).
        scope_id: Partition key — the scope to rank over.
        list_runs: The run listing (:data:`RunLister`).
        bar: The pass^k threshold the verdict is made against. ``None`` or
            blank computes the frontier without a verdict rather than against
            an invented default. A string is parsed; a non-number is
            refused.
        subject_id: Optional single-subject filter; other subjects' results
            are counted as filtered-out rather than dropped silently.
        status: Raw run-status filter, defaulting to ``"completed"`` for the
            same reason as :func:`pivot`. ``"all"`` ranks over every run.
        profile: The host whose sweepable declarations each point's production-replicating cost is read
            against (#571): every point and verdict then names what each of its runs set away from the subject's
            production configuration, read off the WHOLE run. ``None`` leaves that disclosure ``None`` — nobody
            checked, never "nothing moved".
        control_variant_key: The variant each contestant's boundary (guardrail) dimensions are held against;
            ``None`` checks none, and the answer says so per subject.

    Returns:
        A JSON-safe :class:`~threetears.evals.analysis.reporting.FrontierResult` dict.

    Raises:
        ValidationFailedError: The bar is not a number or is outside
            ``[0, 1]``, or ``status`` is neither ``"all"`` nor a status a run
            can carry — each a question the caller can restate.
    """
    try:
        parsed_bar = normalize_bar(bar)
    except FrontierError as e:
        raise ValidationFailedError(str(e)) from e
    subject = (subject_id or "").strip() or None
    status = _status_filter(status)

    # Every run is read then narrowed here rather than in the query, for the
    # exclusion-accounting reason spelled out in :func:`pivot`.
    all_runs, cohort, archived_run_ids = _corpus_and_cohort(list_runs, scope_id)
    runs = [run for run in cohort if run.status == status] if status else cohort
    if profile is not None:
        # A listing elides host payload, and a payload-carried lever read off it would report as the subject's
        # own setting, so a run whose footing is read is read whole (`run_summary` does the same).
        runs = [(storage.load_eval_run(run.id, scope_id) or run) if run.elided_payload_paths else run for run in runs]
    results = storage.query_eval_results(scope_id)
    try:
        result = compute_frontier(
            runs,
            results,
            bar=parsed_bar,
            subject_id=subject,
            known_run_ids={run.id for run in all_runs},
            archived_run_ids=archived_run_ids,
            profile=profile,
            control_variant_key=(control_variant_key or "").strip() or None,
        )
    except FrontierError as e:
        raise ValidationFailedError(str(e)) from e
    return result.model_dump(mode="json")


def history(
    storage: LensStore,
    scope_id: str,
    *,
    list_runs: RunLister,
    metric: str | None = None,
    min_absolute_change: float = 0.0,
    min_relative_change: float = 0.0,
    subject_id: str | None = None,
    status: str | None = "completed",
    profile: HostProfile,
) -> HistoryResult:
    """Series one measure over time per contestant, flagging real regressions.

    Composed over ``list_runs`` plus the scope's results and delegated
    to :func:`~threetears.evals.analysis.reporting.compute_history`, so the REST route and the MCP
    action answer from one implementation — the same shape as :func:`pivot`
    and :func:`frontier`.

    **Arguments arrive raw.** Blank-normalization and the status filter happen
    here, at the one seam, for the reason spelled out in :func:`pivot`.

    Args:
        storage: Where the scope's results are read (:class:`LensStore`).
        scope_id: Partition key — the scope to series over.
        list_runs: The run listing (:data:`RunLister`).
        metric: Which measure to track. ``None`` or blank takes the composite.
        min_absolute_change: Smallest absolute move a regression flag counts,
            in the measure's own unit; ``0.0`` lets significance alone flag.
        min_relative_change: Smallest move relative to the baseline a flag
            counts, as a fraction; ``0.0`` lets significance alone flag.
        subject_id: Optional single-subject filter; other subjects' results
            are counted as filtered-out rather than dropped silently.
        status: Raw run-status filter, defaulting to ``"completed"`` for the
            same reason as :func:`pivot`. ``"all"`` series over every run.
        profile: The host whose vocabulary this reads.

    Returns:
        The :class:`~threetears.evals.analysis.reporting.HistoryResult`.

    Raises:
        ValidationFailedError: The metric is not one this surface can series,
            or ``status`` is neither ``"all"`` nor a status a run can carry —
            each a question the caller can restate.
    """
    metric = normalize_blank(metric, METRIC_COMPOSITE)
    subject = (subject_id or "").strip() or None
    status = _status_filter(status)

    # Every run is read then narrowed here rather than in the query, for the
    # exclusion-accounting reason spelled out in :func:`pivot`.
    all_runs, cohort, archived_run_ids = _corpus_and_cohort(list_runs, scope_id)
    runs = [run for run in cohort if run.status == status] if status else cohort
    results = storage.query_eval_results(scope_id)
    try:
        result = compute_history(
            runs,
            results,
            metric=metric,
            min_absolute_change=min_absolute_change,
            min_relative_change=min_relative_change,
            subject_id=subject,
            known_run_ids={run.id for run in all_runs},
            archived_run_ids=archived_run_ids,
            profile=profile,
        )
    except HistoryError as e:
        raise ValidationFailedError(str(e)) from e
    return result


def program_budget(storage: LensStore, scope_id: str, *, list_runs: RunLister) -> dict[str, Any]:
    """Report program-lens spend over a scope, excluding no run.

    Composed over ``list_runs`` plus the scope's results and delegated
    to :func:`~threetears.evals.analysis.reporting.compute_program_budget` — the same shape as
    :func:`pivot` and :func:`history`, with one deliberate difference: it
    takes **no status filter**. The quality surfaces default to
    ``status="completed"`` because an in-flight or failed run has no stable
    quality signal; budget never filters, because the spend was real whatever
    the run's fate. Passing every run is the whole contract of this view.

    Args:
        storage: Where the scope's results are read (:class:`LensStore`).
        scope_id: Partition key — the scope whose spend to total.
        list_runs: The run listing (:data:`RunLister`).

    Returns:
        A JSON-safe :class:`~threetears.evals.analysis.reporting.ProgramBudget` dict —
        per-run spend with a cumulative series, plus the incomplete-run
        accounting that names the spend a quality view would have dropped.
    """
    # Archived runs are INCLUDED here, unlike every quality surface. Archiving is
    # a measurement curation, not a financial one: a run retired because its
    # observation is junk still spent its dollars, and this is the lens whose
    # whole job is to count the spend a quality view drops. Excluding them would
    # also push their cost into `unattributed_cost_usd`, which is documented as
    # nonzero only on an integrity gap.
    runs = list_runs(scope_id, include_archived=True)
    results = storage.query_eval_results(scope_id)
    return compute_program_budget(runs, results).model_dump(mode="json")


def orphaned_runs(storage: LensStore, scope_id: str, *, list_runs: RunLister) -> dict[str, Any]:
    """Report the scope's runs that no campaign holds, with their spend.

    Membership is curated rather than queried, so a run reaches an analysis
    only because an operator attached it — and nothing reported the runs that
    never were. Shared verbatim by every surface that serves it.

    Archived runs are **included** in the scan, for the reason
    :func:`program_budget` includes them: this view reports spend, and an
    archived run's dollars were still spent. The rows say which are archived
    so a deliberate retirement is not read as an oversight.

    Args:
        storage: Where the scope's results and every campaign are read (:class:`LensStore`).
        scope_id: Partition key — the scope whose runs to check.
        list_runs: The run listing (:data:`RunLister`).

    Returns:
        A JSON-safe :class:`~threetears.evals.analysis.reporting.OrphanedRunsResult`.
    """
    runs = list_runs(scope_id, include_archived=True)
    results = storage.query_eval_results(scope_id)
    # Every campaign in the scope is every campaign that can hold one of its runs:
    # membership outside a campaign's own scope is refused at attachment.
    campaigns = storage.list_campaigns(scope_id)
    return compute_orphaned_runs(runs, results, [campaign.run_ids for campaign in campaigns]).model_dump(mode="json")


def export_results(
    storage: LensStore,
    scope_id: str,
    *,
    list_runs: RunLister,
    fmt: str | None = None,
    status: str | None = "completed",
    run_ids: list[str] | None = None,
    profile: HostProfile,
) -> ScoreExport:
    """Serialize the projection's flat rows for a scope as CSV or JSON.

    Composed over ``list_runs`` plus the scope's results, projected
    through :func:`~threetears.evals.analysis.reporting.project_score_records` — the same
    projection ``pivot`` reads — and serialized at one seam
    (:func:`~threetears.evals.analysis.reporting.export_projection`) so every surface emits
    byte-identical exports.

    **Arguments arrive raw.** The format and status are normalized here, at
    the one seam, for the reason spelled out in :func:`pivot`. The projection
    is wired exactly as :func:`pivot` wires it — every run read, narrowed by
    status here rather than in the query — so a JSON export's exclusion counts
    distinguish "excluded on request" from "unplaceable".

    Args:
        storage: Where the scope's results are read (:class:`LensStore`).
        scope_id: Partition key — the scope to export.
        list_runs: The run listing (:data:`RunLister`).
        fmt: ``"csv"`` (default) or ``"json"``. ``None`` or blank takes CSV,
            the flat-rows-for-analysis form.
        status: Raw run-status filter, defaulting to ``"completed"`` for the
            same reason as :func:`pivot`. ``"all"`` exports every run's rows.
        run_ids: Optional — restrict the export to these run ids. When given,
            only those runs' results are fetched (each query is run-id-indexed
            in storage) rather than the whole scope, so a targeted export of
            a handful of runs reads a handful of runs' rows. What it saves is
            rows, not row width: the stored result carries no trace payload at
            all — that lives in a sibling :class:`~threetears.evals.schema.models.EvalTrace`
            document no export path reads. Unknown ids are silently absent (they
            contribute no rows). ``None`` exports every run, as before.

            **A named id is honoured even when the run is archived**, matching
            ``EvalService.get_run`` and every other by-id surface: archiving withholds a
            run from the *default* cohort, not from a caller who asked for it by
            name. Only the unnamed case — the one that expressed no preference —
            is narrowed to the cohort.
        profile: The host whose vocabulary this reads.

    Returns:
        The :class:`~threetears.evals.analysis.reporting.ScoreExport`: its body is CSV text or a JSON
        :class:`~threetears.evals.analysis.reporting.ScoreProjection`, and beside it the row count,
        the exclusions and the completeness disclosures a CSV body has no place for.

    Raises:
        ValidationFailedError: The format is not one this surface can emit,
            or ``status`` is neither ``"all"`` nor a status a run can carry —
            each a question the caller can restate.
    """
    fmt = normalize_blank(fmt, "csv")
    status = _status_filter(status)
    requested = {rid.strip() for rid in run_ids if rid and rid.strip()} if run_ids else None

    all_runs, cohort, archived_run_ids = _corpus_and_cohort(list_runs, scope_id)
    # An explicitly named id is selected from the CORPUS, not the cohort: naming a run
    # is deliberate, and every other by-id surface (get_run, run_summary, compare_two_runs,
    # bisect_runs) reads an archived run regardless. Narrowing through the cohort here
    # would answer a run the caller asked for by name with an empty CSV — and CSV
    # carries no exclusion channel to say why. The archive filter still applies to the
    # unnamed case, which is the one that did not ask.
    runs = cohort if requested is None else [run for run in all_runs if run.id in requested]
    if status:
        runs = [run for run in runs if run.status == status]

    # With a run_ids filter, pull only the requested runs' results — each
    # query is run-id-indexed — instead of every result in the scope. The
    # projection places a subset natively: known_run_ids stays the whole
    # corpus, so a result whose run was filtered out still reads as "excluded
    # on request", never "unplaceable".
    # Analytic export reads scores + flat coordinates only, and there is nothing
    # left here to project away: the stored eval_result carries neither the
    # per-turn trace nor the OTel spans — they live in sibling eval_trace
    # documents this path never touches. Narrowing the run set narrows the ROW
    # COUNT hauled off the partition; the width is already minimal.
    if requested is None:
        results = storage.query_eval_results(scope_id)
    else:
        results = [res for run in runs for res in storage.query_eval_results_by_run(run.id, scope_id)]
    # The archived ids are supplied only on the path that actually applied the
    # archive filter. With an explicit `run_ids` selection the runs come from
    # the CORPUS, so archival narrowed nothing and an archived run reaching
    # here is in scope — calling any later exclusion of it archival would send
    # the reader to un-archive a run that is already selected. Today that path
    # also reads results per selected run, so no exclusion of either class can
    # arise on it at all; the argument is written to the narrowing that
    # happened rather than to that query shape, which is not its contract.
    projection = project_score_records(
        runs,
        results,
        known_run_ids={run.id for run in all_runs},
        archived_run_ids=archived_run_ids if requested is None else None,
        profile=profile,
    )
    try:
        return export_projection(projection, fmt=fmt)
    except ExportError as e:
        raise ValidationFailedError(str(e)) from e


def run_summary(
    storage: LensStore,
    run_id: str,
    scope_id: str,
    *,
    load_run_listed: Callable[[str, str], EvalRun],
    row_columns: RowColumns,
    profile: HostProfile,
    rubric_threshold: int = 3,
) -> dict[str, Any]:
    """Compose a run's verdict numbers — pass^k, latency, cost — per model.

    Loads the run (404 if missing) plus its results, then composes the
    query-time aggregators
    (:func:`~threetears.evals.kernel.scoring.compute_pass_hat_k`,
    :func:`~threetears.evals.kernel.scoring.compute_latency_summary`,
    :func:`~threetears.evals.kernel.scoring.compute_async_delivery_summary`,
    :func:`~threetears.evals.kernel.scoring.compute_cost_summary`) into one
    JSON-serializable structure. The aggregators key on the
    ``(model, eval_run_id)`` tuple, which is not JSON-safe — so the
    per-group numbers are flattened into a list of rows, one per
    ``(model, run_id)`` that produced results.

    Each row also carries the host's own columns for its group (``row_columns``) — a host's
    rollup of what its kind reported — laid over the aggregates last.

    Args:
        storage: Where the run's results are read (:class:`LensStore`).
        run_id: The eval run to summarize.
        scope_id: Partition key — the scope the run lives in.
        load_run_listed: Loads one run the way a listing does (its scalars, not its host
            payload), raising ``NotFoundError`` for an unknown id.
        row_columns: The host's per-group columns (:data:`RowColumns`).
        profile: The host whose sweepable declarations say which inputs the run moved off the
            subject's production configuration — the disclosure every prod-cost row carries.
        rubric_threshold: Minimum rubric score counted as a pass (default 3).

    Returns:
        ``{"run_id", "status", "candidate_kind", "candidate_model", "k_runs", "rubric_threshold",
        "rows": [{"model", "run_id", "pass_hat_k", "k", "n_cases_at_k", "pass_hat_k_curve",
        "n_test_cases", "n_cannot_tell_excluded", "mean_total_ms", "median_total_ms",
        "p95_total_ms", "max_total_ms", "mean_llm_ms", "mean_tool_ms", "n_total_ms",
        "n_llm_ms", "n_tool_ms", "total_cost_usd",
        "mean_cost_usd", "n_cost_usd", "total_prod_cost_usd",
        "mean_prod_cost_usd",
        "n_prod_cost_usd", "prod_cost_footing", "n_results", "async_deliveries", "async_deliveries_substituted",
        "async_delivery_mean_elapsed_ms", "async_delivery_median_elapsed_ms",
        "async_delivery_p95_elapsed_ms", "async_delivery_elapsed_n", ...host columns}, ...],
        "dimension_rows": [{"model", "run_id", "dim", "mean_score",
        "min_score", "max_score", "n"}, ...]}``.
        Each row's engine keys are the ones named above; the host's ``row_columns`` for the
        row's group are merged over them last, so a host passing none adds none; which keys a
        host adds, and when each is absent, is that host's to document.
        ``pass_hat_k`` is pass^k — the chance that ``k`` attempts at a case all
        pass, estimated without bias per case and averaged over the
        ``n_cases_at_k`` cases scored at least ``k`` times
        (:func:`~threetears.evals.kernel.scoring.compute_pass_hat_k`); ``k`` is
        the deepest iteration *attempted*. ``pass_hat_k_curve`` is every depth
        from 1 to the deepest scored case, each point with its own case count,
        so two rows at different ``k`` are compared at a depth both reached
        rather than headline against headline. Latency keys are absent on
        rows whose group had no measured latency, and absent individually
        when only that component went unmeasured — each mean travels with
        its own denominator (``n_total_ms`` / ``n_llm_ms`` / ``n_tool_ms``),
        which is what to read rather than ``n_results``, since the cost
        aggregate applies last and owns that key. The program-cost pair
        (``total_cost_usd`` / ``mean_cost_usd``) is absent when no result in
        the group was priced, and ``n_cost_usd`` is its denominator — a result
        whose spend went unpriced is omitted from it rather than counted as a
        zero. The prod-cost keys
        (``total_prod_cost_usd`` / ``mean_prod_cost_usd`` /
        ``n_prod_cost_usd``) are absent together when no result in the group
        measured a production-replicating cost, and ``n_prod_cost_usd`` is
        the denominator ``mean_prod_cost_usd`` was computed over — a result
        that observed no production-role cost is omitted from it rather than
        counted as a zero, and so is one that took no turn: a result the harness
        faulted, or a call the model refused straight away. The program-cost pair
        keeps both, because those dollars were spent. ``prod_cost_footing`` travels with the
        prod-cost keys and is absent with them: which inputs the run held away from the subject's
        production configuration (``moved``), which could not be checked (``unchecked``), which held
        (``held``), ``moved_nothing`` and the ``sentence`` to print beside the figure — read off the host's
        declarations (:meth:`~threetears.evals.kernel.host.sweepables.SweepableRegistry.production_footing`),
        since the figure is what production spends only for a run that moved nothing (#571). The six
        async-delivery keys are the engine's rollup of each result's ``async_deliveries``
        (:func:`~threetears.evals.kernel.scoring.compute_async_delivery_summary`): absent together on a
        group none of whose results watched for background work, the durations absent when no real
        delivery measured one, and the 95th percentile absent below 13 durations.
        ``measure_latency`` and ``cell_concurrency`` are the run's own record of whether its launch declared
        latency under test and how many of its cells executed at once (None on a run stored before either was
        recorded, whose cells executed one at a time); ``latency_disclosure`` is the line to render beside the
        latency keys when any of them was read under concurrency (the results' ``execution_mode``), null
        otherwise — the figures describe this run as it ran and are never compared with another run's.
        ``completeness`` and ``completeness_disclosure``
        are both null when the run carries no completeness record (it has not
        reached a terminal state), and the disclosure alone is null when the
        run delivered its whole matrix — it is the sentence to render when a
        short run's rates would otherwise read as comparable;
        ``dimension_rows`` is empty when the run's template carries no
        judge-scored rubric dimensions.

    Raises:
        NotFoundError: No run with that id in the scope.
    """
    # The summary reads the run's scalars and never its payload.
    run = load_run_listed(run_id, scope_id)
    results = storage.query_eval_results_by_run(run_id, scope_id)

    pass_hat = compute_pass_hat_k(results, rubric_threshold=rubric_threshold)
    latency = compute_latency_summary(results)
    deliveries = compute_async_delivery_summary(results)
    cost = compute_cost_summary(results)
    # Read off the WHOLE run: the listed copy elides host payload, and a payload-carried lever read
    # off it would report as the subject's own setting.
    whole = run if not run.elided_payload_paths else storage.load_eval_run(run_id, scope_id)
    footing = profile.sweepables.production_footing(whole, results) if whole is not None else None
    footing_disclosure = (
        {**footing.model_dump(), "moved_nothing": footing.moved_nothing, "sentence": footing.sentence()}
        if footing is not None
        else None
    )
    dimensions = compute_dimension_summary(results)
    host_columns = row_columns(results)

    rows: list[dict[str, Any]] = []
    # Cost is reported for every group that has results; pass^k, cost, and
    # the host's columns share the same group set, while
    # latency is a subset (measured-only).
    for key in sorted(cost.keys()):
        model, group_run_id = key
        row: dict[str, Any] = {"model": model, "run_id": group_run_id}
        row.update(pass_hat.get(key, {}))
        if key in latency:
            row.update(latency[key])
        row.update(deliveries.get(key, {}))
        # Cost lands after latency deliberately: both emit `n_results`, and
        # cost's is the one that means "results in this group". Latency's
        # per-component denominators travel as n_total_ms / n_llm_ms /
        # n_tool_ms precisely so this overwrite costs no information.
        row.update(cost[key])
        if "mean_prod_cost_usd" in row:
            row["prod_cost_footing"] = footing_disclosure
        row.update(host_columns.get(key, {}))
        rows.append(row)

    # Per-dimension breakdown — flattened from the (model, run_id, dim) key
    # into JSON-safe rows, one per scored dimension per model.
    dimension_rows: list[dict[str, Any]] = []
    for model, group_run_id, dim in sorted(dimensions.keys()):
        dimension_rows.append(
            {
                "model": model,
                "run_id": group_run_id,
                "dim": dim,
                **dimensions[(model, group_run_id, dim)],
            }
        )

    return {
        "run_id": run.id,
        "status": run.status,
        # Which kind of candidate the rows below measured, so a reader does not have to
        # load the template (which may have been repointed since) to know what a row is.
        "candidate_kind": run.candidate_kind,
        "candidate_model": run.candidate_model,
        "k_runs": run.k_runs,
        "rubric_threshold": rubric_threshold,
        # The models that actually SCORED, beside the model that was scored. A
        # summary carrying only ``candidate_model`` and a run-level judge invites the reading
        # that one judge produced every number below, which is false for any run whose
        # dims override the pin — and the dimension_rows are exactly where that
        # difference lands.
        "effective_judges": run.effective_judges,
        "effective_judges_source": run.effective_judges_source,
        # How much of the matrix the numbers below actually rest on. A run can
        # read ``completed`` while short, and every rate here is then computed
        # over a denominator its siblings do not share — so the record travels
        # WITH the numbers rather than being a detail on the run document.
        "completeness": run.completeness.to_dict() if run.completeness else None,
        "completeness_disclosure": completeness_disclosure(run.completeness),
        # Whether its launch declared latency under test and how many cells it ran at once (#701), and the
        # line the latency columns are read under when any of them was read under concurrency.
        "measure_latency": run.measure_latency,
        "cell_concurrency": run.cell_concurrency,
        "latency_disclosure": marked_latency_sentence(len(withheld_latency(results, profile.measures)), len(results)),
        "rows": rows,
        "dimension_rows": dimension_rows,
    }


def bisect_runs(
    storage: LensStore,
    run_a_id: str,
    run_b_id: str,
    scope_id: str,
    *,
    profile: HostProfile,
) -> dict[str, Any]:
    """Diff the versioned inputs between two runs — the "what changed" answer.

    Compares every input that determines an eval's score, so a regression
    can be attributed. The set itself is declared once in
    the host's sweepable declarations and shared with
    the analysis bundle's confound scan, which asks the same question over N
    runs instead of two — a second copy of the list here is how this surface
    came to be missing three of them. List/dict-valued inputs are compared as
    sorted sets so element order doesn't register as a difference.

    Both subject fields are reported, and the pair is informative: same
    ``subject_id`` with a differing name is a rename, not a different subject.

    Args:
        storage: Where the runs and their results are read (:class:`LensStore`).
        run_a_id: First run id (the baseline), in ``scope_id``.
        run_b_id: Second run id (the candidate), in ``scope_id``.
        scope_id: Partition key — the scope the runs live in.
        profile: The host whose vocabulary this reads.

    A field whose comparison cannot be decided lands in neither bucket. Two
    runs that both recorded no subject identity are not thereby the same
    subject — comparing two absences for equality manufactures an
    observation, the same ``missing != zero`` discipline applied to usage
    rows and covariates. Those fields land in ``unknown``, and ``details``
    still carries both raw values so a reader can see why.

    Returns:
        ``{"run_a", "run_b", "differs": [field, ...], "same": [field, ...],
        "unknown": [field, ...], "details": {field: {"a": ..., "b": ...}}}``.
        JSON-safe.

    Raises:
        NotFoundError: Either run id is missing in the scope.
    """
    run_a = _load_run(storage, run_a_id, scope_id)
    run_b = _load_run(storage, run_b_id, scope_id)

    results_a = storage.query_eval_results_by_run(run_a_id, scope_id)
    results_b = storage.query_eval_results_by_run(run_b_id, scope_id)

    sweepables = profile.sweepables
    values_a = sweepables.read_all(run_a, results_a)
    values_b = sweepables.read_all(run_b, results_b)

    differs: list[str] = []
    same: list[str] = []
    unknown: list[str] = []
    details: dict[str, dict[str, Any]] = {}
    # UNION, not run A's keys: an open family resolves whatever a given run carried, so two
    # runs of one host legitimately answer under different names. A name only one run has is a
    # real difference (that run overlaid a knob the other left alone), never an absence — a
    # name NEITHER carried is in no union and is never compared, which is the correct silence.
    for field_name in sorted(set(values_a) | set(values_b)):
        value_a = values_a.get(field_name, _NOT_CARRIED)
        value_b = values_b.get(field_name, _NOT_CARRIED)
        # A dimension this host does not have is not a difference, an agreement OR an unknown —
        # the bisection's three answers are all claims about what the two runs did, and this host
        # never had the thing. Reported as undecidable it would send an operator to record a
        # judge model for a subject nothing scored with a model. BOTH values go in: one arm
        # recording a level refutes the declaration, and omitting on the strength of the arm
        # that agreed would drop the very difference this surface exists to name.
        if profile.omits_apparatus(field_name, [(run_a, value_a), (run_b, value_b)]):
            continue
        # A run whose rig had no such seat (a code-only run beside a judged one) reads at UNSEATED_LEVEL,
        # so the two are a difference rather than an unknown.
        if value_a is not _NOT_CARRIED:
            value_a = profile.apparatus_level(run_a, field_name, value_a)
        if value_b is not _NOT_CARRIED:
            value_b = profile.apparatus_level(run_b, field_name, value_b)
        details[field_name] = {"a": value_a, "b": value_b}
        if sweepables.is_indeterminate(field_name, value_a, value_b):
            unknown.append(field_name)
        elif value_a == value_b:
            same.append(field_name)
        else:
            differs.append(field_name)

    return {
        "run_a": run_a_id,
        "run_b": run_b_id,
        "differs": differs,
        "same": same,
        "unknown": unknown,
        "details": details,
    }


def compare_two_runs(
    storage: LensStore,
    run_a_id: str,
    run_b_id: str,
    scope_id: str,
    *,
    load_template: Callable[[str], EvalTemplate],
    subject_detail: Callable[[EvalRun], dict[str, dict[str, Any]]],
    rubric_threshold: int = 3,
) -> dict[str, Any]:
    """Diff two runs into the side-by-side compare view.

    Composes, for the two runs' arms, BOTH quality metrics — pass^k
    (reliability) and mean composite (continuous quality) — with their
    deltas, plus an effect size, its p-value and a significance flag on the
    composite. A run carries one arm, so this is one row, A's arm against
    B's, whether or not they ran one model. Also returns a per-template rollup and each run's
    subject detail for the config diff (``subject_detail``).

    **Whether the samples were paired is a fact about this row and is
    emitted with it.** Pairing needs the two runs to have actually scored the
    same frozen ``test_case_id`` s; a shared ``template_id`` only makes that
    possible, and two runs of one template whose case sets do not intersect
    are compared unpaired (:data:`~threetears.evals.analysis.stats.UNPAIRED_TEST_NAME`). Deriving pairing from the template — as
    ``comparison_basis`` invites — names a test that did not run, so every
    row carries ``paired`` and no surface re-derives it.

    **Composite comparisons are withheld across subjects.** Rubric
    dimensions come from each subject's own self-description, so two
    subjects' composites are different measurements wearing one number.
    When :func:`~threetears.evals.analysis.reporting.cross_subject_disclosure` returns a
    sentence, every composite *comparison* — the delta, the effect size, the
    p and the verdict — is withheld here rather than on each surface, so
    REST and MCP cannot answer differently about the same run pair. The
    per-run composites stay, to be read separately.

    Pure composition over the two run loads + the query-time aggregators; no
    new storage method.

    Args:
        storage: Where the runs and their results are read (:class:`LensStore`).
        run_a_id: Baseline run id, in ``scope_id``.
        run_b_id: Candidate run id, in ``scope_id``.
        scope_id: Partition key — the scope both runs live in.
        load_template: Loads a template by id, raising ``NotFoundError`` for an unknown one —
            the per-template rollup's display name, which falls back to the id.
        subject_detail: The host's one-entry ``{subject label: detail}`` map for a run, which
            the config diff aligns run A's and run B's subjects by.
        rubric_threshold: Minimum rubric score counted as a pass^k pass.

    Returns:
        ``{"run_id_a", "run_id_b", "comparison_basis",
        "composite_comparability", "rubric_threshold", "comparison": {"arm": {...},
        "per_template": [...]}, "subject_detail_a",
        "subject_detail_b"}``. ``comparison_basis`` says only whether the
        two runs share a template — it is NOT the significance basis.
        ``composite_comparability`` is the withholding disclosure, or
        ``None`` when the composites are comparable.
        The ``arm`` row carries ``model_{a,b}``, ``served_models_{a,b}`` (which models the responses named as
        having answered each run's candidate calls — one, pooled or unrecorded — never the requested id), ``k``, ``pass_hat_k_{a,b,delta}``,
        ``composite_{a,b,delta}``, ``composite_basis_{a,b}`` (what each composite was meaned over, ragged
        when its results carried different dimension sets), ``composite_bases_differ``,
        ``composite_interval`` (at ``interval_level``),
        ``n_cases_{a,b}``, ``n_left_out_{a,b}``, ``count_{a,b}``, ``paired``, ``n_pairs``,
        ``hedges_g``, ``p``, ``significant`` (nulls where a run scored nothing,
        a test is undefined, or the composites are not comparable).
        ``composite_{a,b}`` and the delta are over the cases the test read —
        when paired, only those both runs scored, which is not a run's own mean
        when ``n_left_out`` is above 0; ``count_{a,b}`` is each run's own case count.
        JSON-safe.

    Raises:
        NotFoundError: Either run id is missing in the scope.
    """
    run_a = _load_run(storage, run_a_id, scope_id)
    run_b = _load_run(storage, run_b_id, scope_id)
    results_a = storage.query_eval_results_by_run(run_a_id, scope_id)
    results_b = storage.query_eval_results_by_run(run_b_id, scope_id)

    pass_a = compute_pass_hat_k(results_a, rubric_threshold=rubric_threshold)
    pass_b = compute_pass_hat_k(results_b, rubric_threshold=rubric_threshold)
    comp_a = compute_composite_summary(results_a)
    comp_b = compute_composite_summary(results_b)
    per_case_a = compute_per_case_composites(results_a)
    per_case_b = compute_per_case_composites(results_b)

    # Paired significance is only honest when the runs share a template
    # (hence shared frozen test cases).
    # A shared template makes pairing POSSIBLE; the intersection below is
    # what makes it real, and the two come apart when one template's cases
    # were re-frozen between the runs.
    shared_template = run_a.template_id is not None and run_a.template_id == run_b.template_id
    shared_cases = sorted(set(run_a.test_case_ids) & set(run_b.test_case_ids)) if shared_template else []

    # Composite quality is comparable within a subject and never across
    # one. Decided once, here, so both surfaces inherit the same withholding
    # instead of each deciding for itself and drifting.
    comparability = cross_subject_disclosure(run_a.subject_snapshot.subject_id, run_b.subject_snapshot.subject_id)
    composites_comparable = comparability is None

    # A run carries one arm, so the comparison is ONE row: A's arm against B's, whatever model
    # each ran. Joining the two runs' rows by model name — the shape this had while a run could
    # carry several — pairs nothing when the two arms ran different models, which is now the
    # ordinary case: two half-empty rows and no comparison at all.
    model_a, model_b = run_a.candidate_model, run_b.candidate_model
    count_a = comp_a.get((model_a, run_a_id), {}).get("n_cases", 0)
    count_b = comp_b.get((model_b, run_b_id), {}).get("n_cases", 0)
    # pass^k on both arms at ONE depth: the shallower of the two arms' own. A delta between one
    # run's pass^3 and another's pass^1 subtracts two different quantities, and the deeper arm's
    # curve holds its value at the shallower depth, so nothing is lost by reading it there.
    entry_a = pass_a.get((model_a, run_a_id))
    entry_b = pass_b.get((model_b, run_b_id))
    depths = [entry["k"] for entry in (entry_a, entry_b) if entry is not None]
    common_k = min(depths) if depths else None
    pass_hat_k_a = (
        pass_hat_k_at(entry_a["pass_hat_k_curve"], common_k)["pass_hat_k"]
        if entry_a is not None and common_k is not None
        else None
    )
    pass_hat_k_b = (
        pass_hat_k_at(entry_b["pass_hat_k_curve"], common_k)["pass_hat_k"]
        if entry_b is not None and common_k is not None
        else None
    )
    n_pairs: int | None = None
    if shared_cases:
        # Pair only cases scored in BOTH runs.
        paired_cases = [
            tc for tc in shared_cases if (model_a, run_a_id, tc) in per_case_a and (model_b, run_b_id, tc) in per_case_b
        ]
        sample_a = [per_case_a[(model_a, run_a_id, tc)] for tc in paired_cases]
        sample_b = [per_case_b[(model_b, run_b_id, tc)] for tc in paired_cases]
        paired = True
        n_pairs = len(paired_cases)
    else:
        sample_a = [v for (_m, r, _tc), v in per_case_a.items() if r == run_a_id]
        sample_b = [v for (_m, r, _tc), v in per_case_b.items() if r == run_b_id]
        paired = False
    # The means and the delta are over the cases the test read, so the figures a reader is shown are the
    # figures the test saw: paired, only the cases both runs scored, and each side says how many of its
    # own it left out. The same rule a campaign contrast follows (`bundle._compare`).
    composite_a = sum(sample_a) / len(sample_a) if sample_a else None
    composite_b = sum(sample_b) / len(sample_b) if sample_b else None
    # What each composite above was meaned over, read over the same cases (#638): a side pooling results scored on
    # different dimension sets is ragged, and two sides meaned over different sets differ partly in what was
    # averaged. Disclosed, not withheld — the sets can differ for a reason the reader knows to be harmless.
    cases_a = set(paired_cases) if shared_cases else {tc for (_m, r, tc) in per_case_a if r == run_a_id}
    cases_b = set(paired_cases) if shared_cases else {tc for (_m, r, tc) in per_case_b if r == run_b_id}
    basis_a = pooled_composite_basis(
        [r for r in results_a if r.model == model_a and r.test_case_id in cases_a] if sample_a else []
    )
    basis_b = pooled_composite_basis(
        [r for r in results_b if r.model == model_b and r.test_case_id in cases_b] if sample_b else []
    )
    n_scored_a = sum(1 for (_m, r, _tc) in per_case_a if r == run_a_id)
    n_scored_b = sum(1 for (_m, r, _tc) in per_case_b if r == run_b_id)
    # Which models answered each run's candidate calls, as the responses named them (#684): an arm is keyed by
    # the model its launch asked for, so two runs of one floating alias can be answered by different models and
    # still read as one model here. Never the requested id standing in for a response that named none.
    served_a = pooled_served_models(results_a)
    served_b = pooled_served_models(results_b)
    interval: tuple[float, float] | None = None
    if composites_comparable:
        hedges_g, significant, p_value = composite_significance(sample_a, sample_b, paired=paired)
        interval = difference_interval(sample_a, sample_b, paired=paired)
    else:
        # Not computed and then dropped: a t-test on two subjects'
        # composites has no referent, so there is no number to withhold.
        hedges_g, significant, p_value = None, None, None
    arm: dict[str, Any] = {
        "model_a": model_a,
        "model_b": model_b,
        # Each run's served models (`ServedModelReading` as a dict: served_models, n_results, n_unrecorded and
        # state one / pooled / unrecorded); None where the run's candidate made no call.
        "served_models_a": served_a.model_dump() if served_a is not None else None,
        "served_models_b": served_b.model_dump() if served_b is not None else None,
        # The depth both pass^k values below are read at.
        "k": common_k,
        "pass_hat_k_a": pass_hat_k_a,
        "pass_hat_k_b": pass_hat_k_b,
        "pass_hat_k_delta": _score_delta(pass_hat_k_a, pass_hat_k_b),
        # Why a side has no pass^k when none of its attempts had a criterion to pass (#688); None otherwise.
        "pass_hat_k_unmeasured_reason_a": entry_a.get("pass_hat_k_unmeasured_reason") if entry_a else None,
        "pass_hat_k_unmeasured_reason_b": entry_b.get("pass_hat_k_unmeasured_reason") if entry_b else None,
        "composite_a": composite_a,
        "composite_b": composite_b,
        "composite_delta": _score_delta(composite_a, composite_b) if composites_comparable else None,
        # What each composite was meaned over (`CompositeBasis` as a dict: dimensions, bases, ragged), and whether
        # the two sides' dimension sets differ — then the delta is partly a difference in what was averaged.
        "composite_basis_a": basis_a.model_dump() if basis_a is not None else None,
        "composite_basis_b": basis_b.model_dump() if basis_b is not None else None,
        "composite_bases_differ": basis_a is not None
        and basis_b is not None
        and (basis_a.ragged or basis_b.ragged or basis_a.bases != basis_b.bases),
        # The interval on `composite_delta` from the same test as `p`, at the engine's interval level. None
        # wherever no test ran: too few cases, no spread, or composites that are not comparable.
        "composite_interval": None if interval is None else list(interval),
        "interval_level": INTERVAL_LEVEL,
        # The cases each composite above was read over, and how many of each run's scored cases the test
        # left out because the other run did not score them.
        "n_cases_a": len(sample_a),
        "n_cases_b": len(sample_b),
        "n_left_out_a": n_scored_a - len(sample_a),
        "n_left_out_b": n_scored_b - len(sample_b),
        "count_a": count_a,
        "count_b": count_b,
        # How the samples line up, which is what the test that ran on them was:
        # pairing needs cases scored in both runs, not merely one template id in both.
        "paired": paired,
        "n_pairs": n_pairs,
        # Hedges' g (g_z when paired): Cohen's d with its small-sample bias removed. Renamed from
        # `cohens_d` when the estimator changed, so a reader of the old key cannot take the new number for it.
        "hedges_g": hedges_g,
        "p": p_value,
        "significant": significant,
    }

    per_template = _compare_per_template(
        run_a, run_b, arm, load_template=load_template, composites_comparable=composites_comparable
    )

    return {
        "run_id_a": run_a.id,
        "run_id_b": run_b.id,
        # Whether the two runs share a TEMPLATE — nothing more. It is
        # deliberately not the significance basis: a
        # shared template with a non-intersecting case set still tests
        # unpaired, so read each row's own `paired` for that.
        "comparison_basis": "shared-template-intersection" if shared_template else "independent",
        "composite_comparability": comparability,
        # The 1–5 level a criterion had to reach in both pass^k figures (#642).
        "rubric_threshold": rubric_threshold,
        "comparison": {"arm": arm, "per_template": per_template},
        "subject_detail_a": subject_detail(run_a),
        "subject_detail_b": subject_detail(run_b),
    }


def _compare_per_template(
    run_a: EvalRun,
    run_b: EvalRun,
    arm: dict[str, Any],
    *,
    load_template: Callable[[str], EvalTemplate],
    composites_comparable: bool = True,
) -> list[dict[str, Any]]:
    """Roll the arm comparison up to one row per distinct template.

    Runs are single-template (``EvalResult`` carries no template), so each
    run contributes to its own ``template_id``. A shared template yields one
    row with both runs' scores; differing templates yield a row each with
    the other side null. Ad-hoc runs (``template_id is None``) contribute no row.

    Args:
        run_a: Baseline run.
        run_b: Candidate run.
        arm: The arm row.
        load_template: Loads a template by id, for its display name.
        composites_comparable: ``False`` withholds ``composite_delta`` here
            too: rolling up by template does not make two subjects'
            composites comparable, so the rollup withholds wherever the arm
            row does.
    """
    rows: list[dict[str, Any]] = []
    for template_id in dict.fromkeys(t for t in (run_a.template_id, run_b.template_id) if t is not None):
        in_a = template_id == run_a.template_id
        in_b = template_id == run_b.template_id
        pass_a = arm["pass_hat_k_a"] if in_a else None
        pass_b = arm["pass_hat_k_b"] if in_b else None
        comp_a = arm["composite_a"] if in_a else None
        comp_b = arm["composite_b"] if in_b else None
        rows.append(
            {
                "template_id": template_id,
                "template_name": _template_name(load_template, template_id),
                "pass_hat_k_a": pass_a,
                "pass_hat_k_b": pass_b,
                "pass_hat_k_delta": _score_delta(pass_a, pass_b),
                "composite_a": comp_a,
                "composite_b": comp_b,
                "composite_delta": _score_delta(comp_a, comp_b) if composites_comparable else None,
            }
        )
    return rows


def _template_name(load_template: Callable[[str], EvalTemplate], template_id: str) -> str:
    """Best-effort human name for a template id (falls back to the id)."""
    try:
        return load_template(template_id).name
    except NotFoundError:
        # A run can outlive its template; degrade to the id, but leave a
        # trace so a persistently-missing template is diagnosable.
        log.info("compare_two_runs: template %s not found; using id as display name", template_id)
        return template_id


if TYPE_CHECKING:

    def _eval_storage_satisfies_the_port(storage: EvalStorage) -> None:
        """Hold the engine's own store to this consumer's port, so a drifted signature fails typecheck."""
        store: LensStore = storage
        del store


__all__ = [
    "LensStore",
    "RowColumns",
    "RunLister",
    "compare_two_runs",
    "comparison_sets",
    "export_results",
    "frontier",
    "history",
    "orphaned_runs",
    "pivot",
    "program_budget",
    "run_summary",
]
