"""Variation generation for eval templates.

Generates concrete :class:`~threetears.evals.schema.models.EvalTestCase` documents
from a template's :class:`~threetears.evals.schema.models.VariationAxis` declarations.
Each test case freezes one combination of axis values; existing test cases
with the same variation_params are reused (not duplicated).

Three axis generator types:

* **enum** — deterministic; emits every value in ``axis.values`` once.
* **sample** — random subset of ``axis.values`` (without replacement, up to
  ``n_variations`` or the list size, whichever is smaller).
* **llm** — LLM-driven; asks the supplied client for novel values,
  deduplicated against the test cases already persisted for this template.
  The counts record which model wrote them (``VariationCounts.variation_model``),
  read off the client. Those calls are outside every run's cost cap and
  metered-call ceiling, so they go through an
  :class:`~threetears.evals.kernel.out_of_run.OutOfRunBudget` instead, and are
  planned and admitted by the CALLER, not here: the plan is
  :func:`~threetears.evals.kernel.out_of_run.plan_variation_calls`, the caller
  admits every axis's call together (refused together when the out-of-run cap
  cannot pay for them) and hands the admissions in as an
  :class:`~threetears.evals.kernel.out_of_run.AdmittedGeneration`, which this
  only makes and ledgers. A launch admits in its call, before any run starts,
  so a generation it cannot pay for is refused before the launch returns.

Cross-axis combination is the Cartesian product, truncated (or sampled
without replacement) to ``n_variations``. Existing test cases are reused
when their ``variation_params`` match; only new combinations get persisted.
A new case's stratum is the value of the axis the template nominates
(``VariationAxis.stratum``), when it nominates one.

Where the store calls run
-------------------------

Every store call — the dedup read, each case saved, each ledger row the budget writes — runs on the
host's blocking executor; only the model calls stay on the loop, because the writer's client is bound
to it, so a host cannot move the whole generation to a thread.

Preview mode
------------

``preview=True`` produces the same shapes but skips persistence — useful
for a "what would this generate?" web UI without committing rows.
"""

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING, NamedTuple, Protocol

from threetears.evals.kernel.errors import ValidationFailedError
from threetears.evals.schema.hashing import canonical_json
from threetears.evals.kernel.identity import compute_content_hash
from threetears.evals.schema.models import EvalTemplate, EvalTestCase, VariationAxis, VariationCounts
from threetears.evals.kernel.offload import run_blocking
from threetears.evals.kernel.capped_completion import refresh_prices
from threetears.evals.kernel.out_of_run import (
    AdmittedGeneration,
    OutOfRunBudget,
    existing_axis_values,
    plan_variation_calls,
)
from threetears.evals.schema.out_of_run_spend import OutOfRunSpend
from threetears.evals.kernel.provider import extract_json
from threetears.evals.schema.logs import get_logger

if TYPE_CHECKING:
    from concurrent.futures import Executor

    from threetears.evals.kernel.storage import EvalStorage
    from threetears.evals.schema.completion import VariationLLM

log = get_logger(__name__)


class EvalTestCaseStore(Protocol):
    """The one read and one write generation needs.

    Cut to what :func:`generate_variations` calls. Dedup reads the template's
    existing cases in the scope being generated into, and each new combination is
    written back; nothing else about a store is generation's business.

    Structural, so a host's own storage satisfies it by having the methods —
    :class:`~threetears.evals.kernel.storage.EvalStorage` does, with no
    inheritance and no registration.

    Named for the document it stores rather than shortened to ``TestCaseStore``:
    pytest collects any class whose name begins with ``Test``, so the short form
    emits a collection warning in every suite that imports it.

    ``scope_id`` — the engine's word for a partition it never interprets — is
    positional-only, so an implementation's own parameter name never has to
    match the port's.
    """

    def query_test_cases(self, scope_id: str, /, *, template_id: str | None = None) -> list[EvalTestCase]:
        """Every test case in a scope, optionally narrowed to one template."""
        ...

    def save_test_case(self, test_case: EvalTestCase, /) -> None:
        """Write a test case; raises ``StorageError`` (``ConflictError`` on a lost ``if_match``) rather than returning a flag."""
        ...


def _canonical_key(params: dict[str, str]) -> str:
    """Return a stable canonical string for a variation_params dict.

    Sorted by key to handle insertion-order differences. Used to compare two
    test cases for dedup at generation time.

    Shares the canonicaliser behind :attr:`EvalTestCase.content_hash` on
    purpose: dedup and content identity must agree about when two cases are the
    same generated artifact, or generation would mint a second case that hashes
    identically to the first.
    """
    return canonical_json(params)


class GeneratedVariations(NamedTuple):
    """The cases a generation produced, the counts the launch records beside them, and what its calls spent."""

    cases: list[EvalTestCase]
    counts: VariationCounts
    #: One ledger row per ``llm`` axis call the generation made, in order — written through its budget
    #: as each call ended. Empty when no axis is ``llm``.
    spend: tuple[OutOfRunSpend, ...] = ()


async def generate_variations(
    template: EvalTemplate,
    n_variations: int,
    *,
    storage: EvalTestCaseStore,
    scope_id: str,
    blocking_executor: Executor | None,
    generation: AdmittedGeneration | None = None,
    rng: random.Random | None = None,
    preview: bool = False,
) -> GeneratedVariations:
    """Generate up to ``n_variations`` test cases for a template.

    Idempotent: same template + same RNG state + same existing test
    cases produces the same set of variations. LLM-generated values
    deduplicate against existing test cases for the template.

    Args:
        template: The :class:`EvalTemplate` whose variation_axes drive
            generation.
        n_variations: Target number of test cases. The final count may be
            smaller (axis exhaustion) but never larger.
        storage: Storage for dedup-lookup and persistence.
        scope_id: Scope partition key for the test cases.
        blocking_executor: Where every ``storage`` read and write runs, off the event loop — the host's
            ``EvalHost.blocking_executor``, or ``None`` for the loop's default executor. The model calls
            stay on the loop, because the writer's client is bound to it, so a host cannot move the
            whole generation to a thread and this is how the store calls leave it. No default, as the
            host's own has none: the default executor is the one a host's liveness probes may share.
            The ``budget``'s ledger writes go where the budget's own ``blocking_executor`` says.
        generation: The ``llm`` axes' calls, admitted by the caller before this is called
            (:class:`~threetears.evals.kernel.out_of_run.AdmittedGeneration`: the writer, the budget that admitted
            them, each axis's admitted call). Required exactly when an axis declares ``generator='llm'``, and
            refused otherwise: a writer nothing calls would be recorded as the model that wrote the cases. This
            admits nothing; each call is made on the writer it was priced on and ledgered through its budget.
        rng: Optional :class:`random.Random` for deterministic sampling.
            Defaults to a fresh ``random.Random()``.
        preview: When True, returns generated variations *without*
            persisting or deduplicating against storage — useful for web
            UI preview before committing.

    Returns:
        The :class:`EvalTestCase` documents — persisted (or simulated when
        ``preview=True``) in the order generated, existing test cases with
        matching ``variation_params`` reused — and the requested / kept /
        reused counts, so a run holding fewer cases than it asked for says so,
        with the model that wrote the ``llm`` axes' values (``None`` when
        no axis is ``llm``) — and the ledger row of every call it made.

    Raises:
        ValueError: When an axis declares ``llm`` generator but no
            ``generation`` was supplied, one was supplied for a template with
            no ``llm`` axis, its admitted calls are not one per ``llm`` axis,
            or ``n_variations`` is non-positive.
        ValidationFailedError: When an axis yields no values. The launch used
            to fall back to the template's stored cases here, which froze a case
            set nobody asked for under a run that reported generating one.
    """
    if n_variations <= 0:
        raise ValueError(f"n_variations must be positive; got {n_variations}.")
    llm_axes = [axis.name for axis in template.variation_axes if axis.generator == "llm"]
    if generation is not None and not llm_axes:
        raise ValueError(
            f"template {template.id!r} has no llm-generated axis, so nothing would call the generation's writer; "
            "it would be recorded as the model that wrote cases it never wrote — pass generation=None"
        )
    if generation is None and llm_axes:
        raise ValueError(
            f"template {template.id!r} has its {', '.join(map(repr, llm_axes))} axis values written by a model, and "
            "no admitted generation was supplied: plan the calls with plan_variation_calls, admit them under an "
            "out-of-run budget, and pass generation=AdmittedGeneration(...) (a launch's is request.generation)"
        )
    if generation is not None and sorted(generation.calls) != sorted(llm_axes):
        raise ValueError(
            f"the generation admitted calls for axes {sorted(generation.calls)} and template {template.id!r} writes "
            f"{sorted(llm_axes)} with a model; admit exactly the calls plan_variation_calls plans for it"
        )
    rng = rng or random.Random()

    # Load existing test cases for dedup unless previewing.
    existing: list[EvalTestCase] = []
    seen_by_params: dict[str, EvalTestCase] = {}
    if not preview:
        existing = await run_blocking(blocking_executor, storage.query_test_cases, scope_id, template_id=template.id)
        for tc in existing:
            seen_by_params[_canonical_key(tc.variation_params)] = tc

    spend: list[OutOfRunSpend] = []

    # Per-axis value lists.
    per_axis_values: list[list[str]] = []
    for axis in template.variation_axes:
        if axis.generator == "llm":
            # Present by construction: an llm axis was refused above without an admitted generation.
            assert generation is not None
            recorded = await generation.budget.generate(generation.writer, generation.calls[axis.name])
            spend.append(recorded.spend)
            values = _parse_llm_axis_values(axis, recorded.result, n_variations, existing_axis_values(axis, existing))
        else:
            values = _values_for_axis(axis, n_variations, rng=rng)
        if not values:
            raise ValidationFailedError(
                f"variation axis {axis.name!r} ({axis.generator}) produced no values, so no case set can be "
                "generated for it",
                remedy=_LLM_AXIS_REMEDY,
            )
        per_axis_values.append(values)

    # Cross-axis Cartesian product, truncated.
    combos = list(itertools.product(*per_axis_values))
    if len(combos) > n_variations:
        # Shuffle for sample-like variety, then take the first n.
        rng.shuffle(combos)
        combos = combos[:n_variations]

    # The axis whose value is each case's stratum, when the template nominates one. A reused case keeps the
    # stratum it was written with: a stored case never changes, and dedup is by content, so a case written
    # before its axis was nominated reads as declaring none rather than being minted a second time.
    stratum_axis = next((axis.name for axis in template.variation_axes if axis.stratum), None)
    out: list[EvalTestCase] = []
    reused = 0
    for combo in combos:
        params: dict[str, str] = {axis.name: str(combo[i]) for i, axis in enumerate(template.variation_axes)}
        key = _canonical_key(params)
        if key in seen_by_params:
            out.append(seen_by_params[key])
            reused += 1
            continue
        tc = EvalTestCase(
            template_id=template.id,
            scope_id=scope_id,
            variation_params=params,
            stratum=params[stratum_axis] if stratum_axis is not None else None,
            content_hash=compute_content_hash(params),
        )
        if not preview:
            await run_blocking(blocking_executor, storage.save_test_case, tc)
        seen_by_params[key] = tc
        out.append(tc)
    return GeneratedVariations(
        out,
        VariationCounts(
            requested=n_variations,
            kept=len(out),
            reused=reused,
            variation_model=generation.writer.model_name if generation is not None else None,
        ),
        tuple(spend),
    )


async def price_variations(
    template: EvalTemplate,
    n_variations: int,
    *,
    storage: EvalTestCaseStore,
    scope_id: str,
    blocking_executor: Executor | None,
    llm: VariationLLM,
    budget: OutOfRunBudget,
) -> list[float | None]:
    """Price the calls :func:`generate_variations` would make, refusing exactly as it would — and make none.

    What a host's own pre-flight asks before it lets any work start. The engine's battery prices its
    templates' generations itself, from the same plan
    (:func:`~threetears.evals.kernel.out_of_run.plan_variation_calls`) through ``budget.quote``.
    Commits nothing to ``budget``.

    Args:
        template: The template whose ``llm`` axes would be written.
        n_variations: The cases the generation would ask for.
        storage: Where the template's existing cases are read — the values each axis's prompt excludes.
        scope_id: The scope the cases would be generated into.
        blocking_executor: Where the ``storage`` read runs, off the event loop — as
            :func:`generate_variations` takes it.
        llm: The client the axes would be written with.
        budget: The budget the generation would be admitted under.

    Returns:
        Each ``llm`` axis's call ceiling, in declaration order.

    Raises:
        ValueError: ``n_variations`` is non-positive, or the template has no ``llm`` axis.
        ValidationFailedError: The calls cannot be priced under an enforced cap, or are priced above it.
    """
    if n_variations <= 0:
        raise ValueError(f"n_variations must be positive; got {n_variations}.")
    existing = await run_blocking(blocking_executor, storage.query_test_cases, scope_id, template_id=template.id)
    planned = plan_variation_calls(template, n_variations, existing)
    if not planned:
        raise ValueError(f"template {template.id!r} has no llm-generated axis, so a generation of it makes no call")
    await refresh_prices(llm)
    return budget.quote(llm, "variation", list(planned.values()))


# =============================================================================
# Per-axis generators
# =============================================================================


def _values_for_axis(axis: VariationAxis, n_variations: int, *, rng: random.Random) -> list[str]:
    """The values of an axis no model writes: every ``enum`` value, or a ``sample`` of them."""
    if axis.generator == "enum":
        return list(axis.values)
    if axis.generator == "sample":
        if not axis.values:
            return []
        n = min(n_variations, len(axis.values))
        return rng.sample(axis.values, n)
    raise ValueError(f"Unknown axis generator: {axis.generator!r}")


#: What a caller does about an llm axis that produced nothing usable.
_LLM_AXIS_REMEDY = "give the axis values, or for an llm axis launch again with another variation_model"


def _parse_llm_axis_values(
    axis: VariationAxis, response: object, n_variations: int, existing_values: set[str]
) -> list[str]:
    """Read up to ``n_variations`` novel values for ``axis`` off the model's reply.

    Values already in ``existing_values`` are excluded; the rest are returned in the order produced,
    deduplicated. Robust (via :func:`~threetears.evals.kernel.provider.extract_json`) to
    code-fenced output and surrounding commentary, and to duplicate values within the response
    (kept once).

    Returns an empty list when the reply's values are all blank, non-strings or already present; the caller refuses
    an axis that yields nothing.

    Raises:
        ValidationFailedError: The reply is not a JSON object, or carries no ``values`` array — refused here, naming
            which and what came back, rather than read as a reply that offered nothing new.
    """
    text = getattr(response, "content", "") or ""

    try:
        payload = extract_json(text)
    except ValueError as unparsed:
        raise ValidationFailedError(
            f"variation axis {axis.name!r} (llm) produced no values: the model's reply is not a JSON object "
            f"({unparsed})",
            remedy=_LLM_AXIS_REMEDY,
        ) from unparsed
    raw_values = payload.get("values")
    if not isinstance(raw_values, list):
        raise ValidationFailedError(
            f"variation axis {axis.name!r} (llm) produced no values: the model's reply carries no 'values' array "
            f"(its keys: {sorted(payload)})",
            remedy=_LLM_AXIS_REMEDY,
        )

    out: list[str] = []
    seen: set[str] = set(existing_values)
    for v in raw_values:
        if not isinstance(v, str):
            continue
        stripped = v.strip()
        if not stripped:
            continue
        if stripped in seen:
            continue
        seen.add(stripped)
        out.append(stripped)
        if len(out) >= n_variations:
            break
    return out


if TYPE_CHECKING:

    def _eval_storage_satisfies_the_port(storage: EvalStorage) -> None:
        """Hold the engine's own store to this consumer's port, so a drifted signature fails typecheck."""
        store: EvalTestCaseStore = storage
        del store


__all__ = [
    "EvalTestCaseStore",
    "GeneratedVariations",
    "generate_variations",
    "price_variations",
]
