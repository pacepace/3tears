"""Named, provenanced scores -- never a single ``score`` field (D1, SR-A4).

A candidate carries a *set* of score entries. Each is named, carries its
scale semantics, names its source, and states whether it is comparable
across providers. Tavily's relevance lives in [0, 1]; SearXNG's
engine-fusion weight is on a different scale entirely; a consumer that
needs three orthogonal judgments gets three entries. A comparable
relevance exists only when Select produced one -- provider-native scores
are non-comparable by construction (:meth:`ScoreEntry.provider_native`
forces the flag off).
"""

from __future__ import annotations

from typing import Final

from threetears.search.contracts._base import ContractModel

__all__ = [
    "SCALE_RANK",
    "SCALE_UNBOUNDED",
    "SCALE_UNIT_INTERVAL",
    "ScoreEntry",
]

#: scale semantics: value lies in [0, 1], higher is better.
SCALE_UNIT_INTERVAL: Final[str] = "unit-interval"

#: scale semantics: value is an unbounded weight, higher is better,
#: meaningful only relative to siblings from the same source.
SCALE_UNBOUNDED: Final[str] = "unbounded"

#: scale semantics: value is a 1-based ordinal rank, lower is better.
SCALE_RANK: Final[str] = "rank"


class ScoreEntry(ContractModel):
    """One named judgment about one candidate.

    ``comparable`` answers exactly one question: may this entry be compared
    against a same-named entry from a *different provider*? Provider-native
    scores answer no (SR-A4); only a pipeline stage that normalised across
    providers may answer yes.
    """

    #: what is being judged -- ``relevance``, ``freshness``, a stage's own
    #: vocabulary. Open; consumers select by name.
    name: str
    #: the judgment.
    value: float
    #: scale semantics of ``value`` -- open vocabulary with named
    #: well-known values (:data:`SCALE_UNIT_INTERVAL`,
    #: :data:`SCALE_UNBOUNDED`, :data:`SCALE_RANK`).
    scale: str
    #: who produced the judgment: a provider instance name, or a pipeline
    #: stage name for derived scores.
    source: str
    #: whether same-named entries from different providers may be compared.
    #: Defaults to False; provider-native entries must never set it.
    comparable: bool = False

    @classmethod
    def provider_native(cls, *, name: str, value: float, scale: str, provider_instance: str) -> ScoreEntry:
        """Build a provider-native score entry, non-comparable by construction.

        :param name: score name as the contract exposes it
        :ptype name: str
        :param value: the provider's reported value
        :ptype value: float
        :param scale: scale semantics of ``value``
        :ptype scale: str
        :param provider_instance: the provider instance that reported it
        :ptype provider_instance: str
        :return: the entry, with ``comparable`` forced to False (SR-A4)
        :rtype: ScoreEntry
        """
        return cls(name=name, value=value, scale=scale, source=provider_instance, comparable=False)
