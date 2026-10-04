"""Variation generation for eval templates.

Generates concrete :class:`~threetears.evals.contracts.models.EvalTestCase` documents
from a template's :class:`~threetears.evals.contracts.models.VariationAxis` declarations.
Each test case freezes one combination of axis values; existing test cases
with the same variation_params are reused (not duplicated).

Three axis generator types:

* **enum** — deterministic; emits every value in ``axis.values`` once.
* **sample** — random subset of ``axis.values`` (without replacement, up to
  ``n_variations`` or the list size, whichever is smaller).
* **llm** — LLM-driven; asks the supplied client for novel values,
  deduplicated against the test cases already persisted for this template.

Cross-axis combination is the Cartesian product, truncated (or sampled
without replacement) to ``n_variations``. Existing test cases are reused
when their ``variation_params`` match; only new combinations get persisted.

Preview mode
------------

``preview=True`` produces the same shapes but skips persistence — useful
for a "what would this generate?" web UI without committing rows.
"""

from __future__ import annotations

import itertools
import random
from typing import TYPE_CHECKING, NamedTuple, Protocol

from threetears.evals.contracts.errors import ValidationFailedError
from threetears.evals.contracts.hashing import canonical_json
from threetears.evals.contracts.identity import compute_content_hash
from threetears.evals.contracts.models import EvalTemplate, EvalTestCase, VariationAxis, VariationCounts
from threetears.evals.contracts.provider import JSON_OBJECT_RESPONSE_FORMAT, extract_json
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.evals.contracts.provider import SimulatorLLM

log = get_logger(__name__)


class EvalTestCaseStore(Protocol):
    """The one read and one write generation needs.

    Cut to what :func:`generate_variations` calls. Dedup reads the template's
    existing cases in the scope being generated into, and each new combination is
    written back; nothing else about a store is generation's business.

    Structural, so a host's own storage satisfies it by having the methods —
    :class:`~threetears.evals.contracts.storage.EvalStorage` does, with no
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
    """The cases a generation produced, and the counts the launch records beside them."""

    cases: list[EvalTestCase]
    counts: VariationCounts


async def generate_variations(
    template: EvalTemplate,
    n_variations: int,
    *,
    storage: EvalTestCaseStore,
    scope_id: str,
    llm: SimulatorLLM | None = None,
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
        llm: Optional :class:`SimulatorLLM` for ``llm`` axes. Required
            when any axis declares ``generator='llm'``.
        rng: Optional :class:`random.Random` for deterministic sampling.
            Defaults to a fresh ``random.Random()``.
        preview: When True, returns generated variations *without*
            persisting or deduplicating against storage — useful for web
            UI preview before committing.

    Returns:
        The :class:`EvalTestCase` documents — persisted (or simulated when
        ``preview=True``) in the order generated, existing test cases with
        matching ``variation_params`` reused — and the requested / kept /
        reused counts, so a run holding fewer cases than it asked for says so.

    Raises:
        ValueError: When an axis declares ``llm`` generator but no
            ``llm`` client was supplied, or when ``n_variations`` is
            non-positive.
        ValidationFailedError: When an axis yields no values. The launch used
            to fall back to the template's stored cases here, which froze a case
            set nobody asked for under a run that reported generating one.
    """
    if n_variations <= 0:
        raise ValueError(f"n_variations must be positive; got {n_variations}.")
    rng = rng or random.Random()

    # Load existing test cases for dedup unless previewing.
    existing: list[EvalTestCase] = []
    seen_by_params: dict[str, EvalTestCase] = {}
    if not preview:
        existing = storage.query_test_cases(scope_id, template_id=template.id)
        for tc in existing:
            seen_by_params[_canonical_key(tc.variation_params)] = tc

    # Per-axis value lists.
    per_axis_values: list[list[str]] = []
    for axis in template.variation_axes:
        values = await _values_for_axis(axis, n_variations, existing=existing, llm=llm, rng=rng)
        if not values:
            raise ValidationFailedError(
                f"variation axis {axis.name!r} ({axis.generator}) produced no values, so no case set can be "
                "generated for it — give the axis values, or for an llm axis read the log line above for "
                "what the model returned"
            )
        per_axis_values.append(values)

    # Cross-axis Cartesian product, truncated.
    combos = list(itertools.product(*per_axis_values))
    if len(combos) > n_variations:
        # Shuffle for sample-like variety, then take the first n.
        rng.shuffle(combos)
        combos = combos[:n_variations]

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
            content_hash=compute_content_hash(params),
        )
        if not preview:
            storage.save_test_case(tc)
        seen_by_params[key] = tc
        out.append(tc)
    return GeneratedVariations(out, VariationCounts(requested=n_variations, kept=len(out), reused=reused))


# =============================================================================
# Per-axis generators
# =============================================================================


async def _values_for_axis(
    axis: VariationAxis,
    n_variations: int,
    *,
    existing: list[EvalTestCase],
    llm: SimulatorLLM | None,
    rng: random.Random,
) -> list[str]:
    """Dispatch to the axis-specific generator, returning a list of values."""
    if axis.generator == "enum":
        return list(axis.values)
    if axis.generator == "sample":
        if not axis.values:
            return []
        n = min(n_variations, len(axis.values))
        return rng.sample(axis.values, n)
    if axis.generator == "llm":
        if llm is None:
            raise ValueError(f"Axis {axis.name!r} requires generator='llm' but no llm client was supplied.")
        existing_values = {
            tc.variation_params.get(axis.name, "") for tc in existing if tc.variation_params.get(axis.name)
        }
        return await _llm_axis_values(llm, axis, n_variations, existing_values)
    raise ValueError(f"Unknown axis generator: {axis.generator!r}")


async def _llm_axis_values(
    llm: SimulatorLLM,
    axis: VariationAxis,
    n_variations: int,
    existing_values: set[str],
) -> list[str]:
    """Ask the LLM for ``n_variations`` novel values for ``axis``.

    The call runs in JSON-object mode (``response_format={"type":
    "json_object"}``) — the same structured-output robustness the other eval LLM
    callers get — and asks for ``{"values": [...]}``. The object
    envelope is required because ``json_object`` mode forbids a bare top-level
    array, which is why this caller was the one left on hand-rolled array
    scraping until now. Values already in ``existing_values`` are
    excluded; the rest are returned in the order produced, deduplicated.

    Robust (via :func:`~threetears.evals.contracts.provider.extract_json`) to code-fenced output
    and surrounding commentary, and to duplicate values within the response
    (kept once).

    Returns an empty list when the response is not a JSON object or carries no
    ``values`` array — the runner caller treats that as a soft failure and logs it.
    """
    existing_block = (
        "(none — produce any novel values)"
        if not existing_values
        else "\n".join(f"- {v}" for v in sorted(existing_values))
    )
    system_prompt = (
        "You generate values for an eval variation axis. Return ONLY a JSON "
        'object of the form {"values": ["...", "..."]} — a JSON array of strings '
        'under the key "values", with no other keys and no commentary. Each value '
        "must be distinct from every value already shown to you."
    )
    user_prompt = (
        f"Axis name: {axis.name}\n"
        f"Axis description: {axis.description or '(no description)'}\n"
        f"Existing values to exclude:\n{existing_block}\n\n"
        f'Produce {n_variations} new distinct values as a JSON object: {{"values": [...]}}.'
    )
    response = await llm.generate(system=system_prompt, user=user_prompt, response_format=JSON_OBJECT_RESPONSE_FORMAT)
    text = getattr(response, "content", "") or ""

    try:
        payload = extract_json(text)
    except ValueError:
        log.warning("LLM axis %r: response was not a JSON object. Raw preview: %r", axis.name, text[:200])
        return []
    raw_values = payload.get("values")
    if not isinstance(raw_values, list):
        log.warning("LLM axis %r: JSON object carried no 'values' array. Keys: %r", axis.name, sorted(payload))
        return []

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


__all__ = [
    "EvalTestCaseStore",
    "GeneratedVariations",
    "generate_variations",
]
