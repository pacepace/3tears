"""The metadata projection -- structure rides ``ToolResult.metadata`` (D22).

Structured results cross the tool border under one named key,
:data:`SEARCH_RESULTS_METADATA_KEY`, following the
``OBJECT_HANDLE_METADATA_KEY`` precedent in ``media-contracts``: an
explicit border projection (``to_metadata`` / ``from_metadata``), never an
implicit dump at the call site. The payload embeds ``schema_version``
(D13); changes are additive within a family minor, and a reader meeting a
version newer than it knows refuses loudly, naming both versions, rather
than best-effort misreading.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from pydantic import Field

from threetears.search.contracts._base import ContractModel
from threetears.search.contracts.candidate import Candidate, CandidateSet
from threetears.search.contracts.criteria import CriterionDisposition
from threetears.search.contracts.spend import Spend

__all__ = [
    "SEARCH_RESULTS_METADATA_KEY",
    "SEARCH_RESULTS_SCHEMA_VERSION",
    "SearchResultsMetadata",
]

#: the key under which Bind places :class:`SearchResultsMetadata` (as
#: :meth:`SearchResultsMetadata.to_metadata`) in ``ToolResult.metadata``
#: (D22). Consumers read structure from here; prose stays prose.
SEARCH_RESULTS_METADATA_KEY: Final[str] = "search_results"

#: current schema version of the metadata payload (D13). Bumped additively
#: within a family minor; a reader refuses versions above its own.
SEARCH_RESULTS_SCHEMA_VERSION: Final[int] = 1


class SearchResultsMetadata(ContractModel):
    """The structured search result as it rides the tool border.

    Everything a structure-reading consumer needs, without re-parsing
    prose: the query, the typed candidates, the per-criterion dispositions
    (SR-B2), and spend -- present on success *and* on the failed results
    Bind renders, so accounting survives the wire (SR-E3, D10).
    """

    #: payload schema version (D13). Serialized first-class so a reader
    #: can refuse before parsing anything else.
    schema_version: int = SEARCH_RESULTS_SCHEMA_VERSION
    #: the query this result answers (user content -- D11).
    query: str
    #: the typed candidates, in the order the producing layer emitted them.
    candidates: tuple[Candidate, ...] = ()
    #: per-criterion dispositions (SR-B2, SR-B3).
    dispositions: tuple[CriterionDisposition, ...] = ()
    #: what the call consumed (SR-E1); carried on failures too (SR-E3).
    spend: Spend = Field(default_factory=Spend)

    @classmethod
    def from_candidate_set(cls, *, query: str, candidate_set: CandidateSet) -> SearchResultsMetadata:
        """Build the projection from what Call returned.

        :param query: the query the candidate set answers
        :ptype query: str
        :param candidate_set: Call's typed result
        :ptype candidate_set: CandidateSet
        :return: the border projection, at the current schema version
        :rtype: SearchResultsMetadata
        """
        return cls(
            query=query,
            candidates=candidate_set.candidates,
            dispositions=candidate_set.dispositions,
            spend=candidate_set.spend,
        )

    def to_metadata(self) -> dict[str, Any]:
        """Project to the JSON-safe dict placed under the named key.

        :return: a JSON-safe representation that survives the NATS/JSON
            round-trip intact (SR-L4)
        :rtype: dict[str, Any]
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_metadata(cls, data: Mapping[str, Any]) -> SearchResultsMetadata:
        """Reconstruct the projection from its :meth:`to_metadata` dict.

        :param data: the JSON-safe payload read from ``ToolResult.metadata``
        :ptype data: Mapping[str, Any]
        :return: the reconstructed projection
        :rtype: SearchResultsMetadata
        :raises ValueError: when the payload's ``schema_version`` is newer
            than this reader understands -- refused loudly with both
            versions named, never best-effort misread (D13, D26)
        """
        raw_version = data.get("schema_version", SEARCH_RESULTS_SCHEMA_VERSION)
        if isinstance(raw_version, int) and raw_version > SEARCH_RESULTS_SCHEMA_VERSION:
            raise ValueError(
                f"search results metadata is schema_version {raw_version}, but this reader "
                f"understands at most {SEARCH_RESULTS_SCHEMA_VERSION}; refusing rather than misreading"
            )
        return cls.model_validate(dict(data))
