"""The metadata projection -- structure rides ``ToolResult.metadata`` (D22).

Structured results cross the tool border under one named key,
:data:`SEARCH_RESULTS_METADATA_KEY`, following the
``OBJECT_HANDLE_METADATA_KEY`` precedent in ``media-contracts``: an
explicit border projection (``to_metadata`` / ``from_metadata``), never an
implicit dump at the call site. The payload embeds ``schema_version``
(D13); changes are additive within a family minor, and a reader meeting a
version newer than it knows refuses loudly, naming both versions, rather
than best-effort misreading.

**A grounded answer rides its own key**, :data:`ANSWER_METADATA_KEY`, beside
the search key and never inside it (``answers-requirements-design.md`` AD8): an
answer is text plus the candidates it cites, a different shape from a
candidate list, and a reader of either key must be able to trust what it
reads there. Its payload, :class:`AnswerMetadata`, versions itself the same
way and refuses the same way.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from pydantic import Field, model_validator

from threetears.search.contracts._base import ContractModel
from threetears.search.contracts.answer import GroundedAnswer
from threetears.search.contracts.candidate import Candidate, CandidateSet
from threetears.search.contracts.criteria import CriterionDisposition
from threetears.search.contracts.errors import FailureRecord
from threetears.search.contracts.spend import Spend

__all__ = [
    "ANSWER_METADATA_KEY",
    "ANSWER_SCHEMA_VERSION",
    "SEARCH_RESULTS_METADATA_KEY",
    "SEARCH_RESULTS_SCHEMA_VERSION",
    "AnswerMetadata",
    "SearchResultsMetadata",
]

#: the key under which Bind places :class:`SearchResultsMetadata` (as
#: :meth:`SearchResultsMetadata.to_metadata`) in ``ToolResult.metadata``
#: (D22). Consumers read structure from here; prose stays prose.
SEARCH_RESULTS_METADATA_KEY: Final[str] = "search_results"

#: current schema version of the metadata payload (D13). Bumped additively
#: within a family minor; a reader refuses versions above its own.
SEARCH_RESULTS_SCHEMA_VERSION: Final[int] = 1

#: the key under which a grounded answer's :class:`AnswerMetadata` rides
#: ``ToolResult.metadata`` (AD8, D22), built only by ``bind_answer``'s module.
ANSWER_METADATA_KEY: Final[str] = "answer"

#: current schema version of the answer payload (D13). Bumped additively
#: within a family minor; a reader refuses versions above its own.
ANSWER_SCHEMA_VERSION: Final[int] = 1


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
    #: provider-reported degradations carried through to the border, so a
    #: structure-reading consumer learns about a partial answer without
    #: parsing prose (SR-L2, P8).
    notices: tuple[str, ...] = ()
    #: the typed failure, when this projection describes a failed call. This
    #: is how D10 holds across the wire: nothing raises, and the far side
    #: still reads *which* of the seven classes said no -- rather than
    #: string-matching an error prefix, which is the defect the gutting
    #: removes. ``None`` on every successful call.
    failure: FailureRecord | None = None

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
            notices=candidate_set.notices,
        )

    @classmethod
    def from_failure(cls, *, query: str, failure: FailureRecord) -> SearchResultsMetadata:
        """Build the projection for a call that failed.

        The candidates are empty and the spend is the failure's own, so a
        structure-reading consumer gets the accounting for a broken call by
        reading the same key it reads on a working one (SR-E3, D10).

        :param query: the query the failed call was answering
        :ptype query: str
        :param failure: the typed failure's wire record
        :ptype failure: FailureRecord
        :return: the border projection, at the current schema version
        :rtype: SearchResultsMetadata
        """
        return cls(query=query, spend=failure.spend, failure=failure)

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


class AnswerMetadata(ContractModel):
    """One grounded answer as it rides the tool border (AD8, D22).

    Built only by ``threetears.search.bind_answer``'s projections, which
    stamp the version they write (AD8). Exactly one of :attr:`answer` and
    :attr:`failure` is set: a structure
    reader learns which happened from the payload itself, and the spend is
    present either way, so accounting survives the wire on a refusal as on a
    success (SR-E3, D10).
    """

    #: payload schema version (D13). Serialized first-class so a reader can
    #: refuse before parsing anything else.
    schema_version: int = ANSWER_SCHEMA_VERSION
    #: the question the payload answers (user content -- D11).
    question: str
    #: the provider's whole answer -- text, sources, anchors, notices,
    #: dispositions -- on success; ``None`` on a failure.
    answer: GroundedAnswer | None = None
    #: what the call consumed (SR-E1): the answer's own on success, the
    #: failure's on a failure (SR-E3).
    spend: Spend = Field(default_factory=Spend)
    #: the typed failure, when the call failed; ``None`` on success.
    failure: FailureRecord | None = None

    @model_validator(mode="after")
    def _one_outcome(self) -> AnswerMetadata:
        """Refuse a payload that is both an answer and a failure, or neither.

        :return: the validated payload
        :rtype: AnswerMetadata
        :raises ValueError: unless exactly one of ``answer`` and ``failure`` is set
        """
        if (self.answer is None) == (self.failure is None):
            raise ValueError("an answer payload carries exactly one of an answer and a failure")
        return self

    def to_metadata(self) -> dict[str, Any]:
        """Project to the JSON-safe dict placed under :data:`ANSWER_METADATA_KEY`.

        :return: a JSON-safe representation that survives the NATS/JSON
            round-trip intact (SR-L4)
        :rtype: dict[str, Any]
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_metadata(cls, data: Mapping[str, Any]) -> AnswerMetadata:
        """Reconstruct the projection from its :meth:`to_metadata` dict.

        The version is checked before anything else is validated (D13): a
        newer payload may carry fields this reader would refuse as unknown,
        and the honest refusal names the version, not the field.

        :param data: the JSON-safe payload read from ``ToolResult.metadata``
        :ptype data: Mapping[str, Any]
        :return: the reconstructed projection
        :rtype: AnswerMetadata
        :raises ValueError: when the payload's ``schema_version`` is newer
            than this reader understands -- refused loudly with both versions
            named, never best-effort misread (D13, D26)
        """
        raw_version = data.get("schema_version", ANSWER_SCHEMA_VERSION)
        if isinstance(raw_version, int) and raw_version > ANSWER_SCHEMA_VERSION:
            raise ValueError(
                f"answer metadata is schema_version {raw_version}, but this reader "
                f"understands at most {ANSWER_SCHEMA_VERSION}; refusing rather than misreading"
            )
        return cls.model_validate(dict(data))
