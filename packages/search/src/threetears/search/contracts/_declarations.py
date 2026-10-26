"""What every capability declaration shares, stated once (SR-B2, SR-B4, SR-E4).

A search declaration and an answer declaration differ in their fields and
agree on three rules: how a criterion key is answered from the declared
buckets, that a key sits in one bucket only, and how a bare metered unit is
qualified with its provider. Each rule lives here, so a change to it -- a new
disposition, a second spelling of a unit -- lands for both kinds or neither.

Private to the contracts package: the two declaration modules import it, so
neither has to import the other, and nothing here is surface a consumer calls.
"""

from __future__ import annotations

from threetears.search.contracts.criteria import Disposition

__all__ = ["disposition_from", "qualify_unit", "refuse_contradictions"]


def qualify_unit(provider: str, metered_unit: str | None) -> str | None:
    """Qualify a provider's bare metered unit for :attr:`Spend.provider_unit`.

    The one place ``"<provider>:<unit>"`` is composed, so a second spelling
    cannot appear beside the first and compare unequal -- which would make two
    spends from the same provider refuse to sum.

    :param provider: the product name
    :ptype provider: str
    :param metered_unit: the provider's own bare name for its unit, or None
    :ptype metered_unit: str | None
    :return: ``"<provider>:<metered_unit>"``, or ``None`` when no unit is metered
    :rtype: str | None
    """
    if metered_unit is None:
        return None
    return f"{provider}:{metered_unit}"


def disposition_from(
    criterion_key: str,
    *,
    pushdown: tuple[str, ...],
    namespaced: tuple[str, ...],
    local: tuple[str, ...],
    unsatisfiable: tuple[str, ...],
) -> Disposition:
    """Answer how a declaration handles ``criterion_key``, from its buckets.

    :param criterion_key: a well-known or ``<namespace>:<name>`` key
    :ptype criterion_key: str
    :param pushdown: well-known keys the provider's API expresses
    :ptype pushdown: tuple[str, ...]
    :param namespaced: provider-specific keys the adapter accepts
    :ptype namespaced: tuple[str, ...]
    :param local: well-known keys the adapter filters for; empty where a kind
        cannot filter at all
    :ptype local: tuple[str, ...]
    :param unsatisfiable: well-known keys the provider cannot honour
    :ptype unsatisfiable: tuple[str, ...]
    :return: the declared disposition; ``ignored-unknown`` for anything undeclared
    :rtype: Disposition
    """
    if criterion_key in pushdown or criterion_key in namespaced:
        return "pushdown"
    if criterion_key in local:
        return "local"
    if criterion_key in unsatisfiable:
        return "unsatisfied"
    return "ignored-unknown"


def refuse_contradictions(provider: str, buckets: tuple[tuple[str, ...], ...]) -> None:
    """Refuse a key claimed under two different dispositions.

    A key repeated within one bucket is not a contradiction; a key in two is,
    because :func:`disposition_from` would then answer by bucket order, which
    is not an answer.

    :param provider: the declaring provider, named in the refusal
    :ptype provider: str
    :param buckets: the declaration's disposition buckets
    :ptype buckets: tuple[tuple[str, ...], ...]
    :raises ValueError: when a key appears in more than one bucket
    """
    seen: dict[str, int] = {}
    clashes: set[str] = set()
    for index, bucket in enumerate(buckets):
        for key in bucket:
            if key in seen and seen[key] != index:
                clashes.add(key)
            seen[key] = index
    if clashes:
        raise ValueError(
            f"provider {provider!r} declares {sorted(clashes)} under more than one disposition; "
            f"a criterion has exactly one"
        )
