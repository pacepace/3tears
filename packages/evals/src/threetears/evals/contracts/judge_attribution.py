"""Which model scored each dim of a run, and what a run may claim about it.

The judge-attribution rule, stated once for every consumer: the context key may hash only
``recorded`` attribution, the comparison badge may claim a role difference only from it, and
the MCP renderer must caveat anything else. It is a derivation over two fields
:class:`~threetears.evals.contracts.models.EvalRun` stores, not part of the stored shape, so it
lives beside the model rather than inside it; the model reads it through
:attr:`~threetears.evals.contracts.models.EvalRun.attribution_state`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeIs, get_args

__all__ = [
    "JudgeAttributionSource",
    "JudgeAttributionState",
    "attribution_state",
    "judges_sharing_a_candidate_model",
]


#: Whether a run's per-dim judge attribution was captured at launch or reconstructed
#: afterwards from stored ``JudgeConfig`` records. The distinction is load-bearing, not
#: cosmetic: only ``recorded`` attribution may enter the context key. A reconstruction
#: *infers* each dim's config from stored records as of the run's start — a strong
#: inference, since configs supersede rather than replace, but not a record of what
#: scored it: it cannot see a config authored and archived inside the run's own window,
#: it resolves at the start instant rather than as the run progressed, and it is blind to
#: a config hard-deleted rather than archived. Hashing an inference would make an
#: inferred condition indistinguishable from a measured one — the same reason
#: ``DerivedContextIdentity`` is returned as a value and never written back onto a run.
JudgeAttributionSource = Literal["recorded", "derived"]

#: What a run can say about which model scored each dim. Three states, not two: the
#: pair ``(effective_judges, effective_judges_source)`` can express a fourth,
#: incoherent combination (a map with no source), and every consumer must collapse
#: that the same way or they disagree about what a run is allowed to assert.
JudgeAttributionState = Literal["recorded", "derived", "absent"]


def _is_attribution_source(value: str) -> TypeIs[JudgeAttributionSource]:
    """Whether ``value`` is a provenance a run can record for its judge attribution."""
    return value in get_args(JudgeAttributionSource)


def attribution_state(
    effective_judges: dict[str, str] | None,
    source: str | None,
) -> JudgeAttributionState:
    """Collapse a run's attribution pair into the one state every consumer reads.

    Stated once because it is an *eligibility* rule, not a formatting detail, and
    its consumers depend on it agreeing: the context key may hash only ``recorded``
    attribution, the comparison badge may claim a role difference only from
    ``recorded`` attribution, and the MCP renderer must caveat anything else. Each
    of those re-derived the rule from the raw pair and they did not all derive it
    alike — the badge tested
    the map's truthiness alone, so an absent map read as an observed difference and
    a reconstruction could certify sameness on the same surface where
    ``derive_context_identity`` refuses to hash it.

    Incoherent pairs resolve to ``absent``, which is the fail-closed direction: a
    map whose provenance is unknown is one nothing may hash, badge, or present
    uncaveated. Nothing writes that combination today; the collapse is here so a
    future writer bug degrades to "cannot say" rather than to "measured".

    Args:
        effective_judges: ``{dim_id: resolved_model}``, or ``None``/empty.
        source: The recorded provenance, or ``None``.

    Returns:
        ``recorded`` — captured at launch, the only state that may enter a context
        key. ``derived`` — reconstructed afterwards, displayable but never
        hashable. ``absent`` — the run cannot say.
    """
    if not effective_judges:
        return "absent"
    if source is None or not _is_attribution_source(source):
        return "absent"
    return source


def judges_sharing_a_candidate_model(
    effective_judges: dict[str, str] | None, candidate_models: Sequence[str]
) -> dict[str, str]:
    """The dims whose judge is one of the run's candidate models: a model grading its own output.

    Derived from what the run already records rather than stored beside it, so a change to the
    rule can never disagree with rows written under the old one. Compared on the model ids as
    recorded, so a candidate named by an alias the judge's id does not spell is not caught.

    Args:
        effective_judges: ``{dim_id: resolved_model}`` as the run recorded it, or ``None``.
        candidate_models: The run's candidate models.

    Returns:
        ``{dim_id: model}`` for each dim judged by a candidate's model; empty when none is, or
        when the run recorded no attribution.
    """
    candidates = set(candidate_models)
    return {dim: model for dim, model in (effective_judges or {}).items() if model in candidates}
