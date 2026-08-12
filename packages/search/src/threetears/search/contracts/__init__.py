"""The search lingua franca -- types, protocols, errors, keys (search-spec.md §3.1).

The leaf within the leaf. Importing this package pulls nothing beyond
stdlib, pydantic, and ``3tears-media-contracts`` (its one family
dependency by vocabulary, not by import: facets and carrier names key on
its published taxonomy). Nothing here imports ``threetears.core``,
``threetears.agent.*``, langchain, or NATS; nothing reads environment
variables; every payload type JSON round-trips with no callables, files,
or ports in it (SR-L4) -- ports are parameters, never payload.

Layer names (Adapter, Call, Aggregate, Extract, Select, Bind) are module
vocabulary and never appear as type names: contract types are named for
what they are, so a future re-cut of the layers stays cheap.
"""

from __future__ import annotations

from threetears.search.contracts._canonical import (
    CANONICAL_FORM_VERSION,
    canonical_digest,
    canonicalize,
)
from threetears.search.contracts.capabilities import (
    PRICING_FREE_SELF_HOSTED,
    PRICING_PER_REQUEST,
    PRICING_PER_WEIGHTED_UNIT,
    ProviderCapabilities,
    get_capabilities,
    list_capabilities,
    register_capabilities,
)
from threetears.search.contracts.budget import BudgetDecision, BudgetPort
from threetears.search.contracts.candidate import (
    Candidate,
    CandidateSet,
    ContentSlot,
    Locator,
)
from threetears.search.contracts.corpus import Corpus, CorpusEntry
from threetears.search.contracts.criteria import (
    CRITERION_CARRIER,
    CRITERION_DOMAINS_EXCLUDE,
    CRITERION_DOMAINS_INCLUDE,
    CRITERION_LANGUAGE,
    CRITERION_MAX_RESULTS,
    CRITERION_MIN_RESOLUTION,
    CRITERION_RIGHTS_CLASS,
    CRITERION_TIME_RANGE,
    WELL_KNOWN_CRITERIA,
    Criterion,
    CriterionDisposition,
    Disposition,
)
from threetears.search.contracts.errors import (
    FAILURE_CLASSES,
    AuthFailed,
    FailureRecord,
    LocalCapExceeded,
    MalformedResponse,
    QuotaExhausted,
    RateLimited,
    SearchFailure,
    TimedOut,
    TransportFailed,
)
from threetears.search.contracts.facets import (
    FACET_EXTRACTION_STATUS,
    FACET_HAS_DOWNLOADABLE_DATA,
    FACET_HEIGHT,
    FACET_LOCATOR_KIND,
    FACET_MEDIA_CATEGORY,
    FACET_RIGHTS_STATUS,
    FACET_WIDTH,
)
from threetears.search.contracts.fidelity import (
    FIDELITY_BYTES,
    FIDELITY_CONTENT,
    FIDELITY_SNIPPET,
)
from threetears.search.contracts.limiter import RateLimitDecision, RateLimiterPort
from threetears.search.contracts.metadata import (
    SEARCH_RESULTS_METADATA_KEY,
    SEARCH_RESULTS_SCHEMA_VERSION,
    SearchResultsMetadata,
)
from threetears.search.contracts.provenance import (
    EGRESS_DIRECT,
    PRODUCER_API_PROVIDER,
    PRODUCER_MODEL_MEDIATED,
    Provenance,
)
from threetears.search.contracts.provider import SearchProvider
from threetears.search.contracts.request import SearchRequest
from threetears.search.contracts.scores import (
    SCALE_RANK,
    SCALE_UNBOUNDED,
    SCALE_UNIT_INTERVAL,
    ScoreEntry,
)
from threetears.search.contracts.spend import Spend
from threetears.search.contracts.transport import FetchTransport, SearchTransport, TransportResponse

__all__ = [
    "CANONICAL_FORM_VERSION",
    "CRITERION_CARRIER",
    "CRITERION_DOMAINS_EXCLUDE",
    "CRITERION_DOMAINS_INCLUDE",
    "CRITERION_LANGUAGE",
    "CRITERION_MAX_RESULTS",
    "CRITERION_MIN_RESOLUTION",
    "CRITERION_RIGHTS_CLASS",
    "CRITERION_TIME_RANGE",
    "EGRESS_DIRECT",
    "FACET_EXTRACTION_STATUS",
    "FACET_HAS_DOWNLOADABLE_DATA",
    "FACET_HEIGHT",
    "FACET_LOCATOR_KIND",
    "FACET_MEDIA_CATEGORY",
    "FACET_RIGHTS_STATUS",
    "FACET_WIDTH",
    "FAILURE_CLASSES",
    "FIDELITY_BYTES",
    "FIDELITY_CONTENT",
    "FIDELITY_SNIPPET",
    "PRICING_FREE_SELF_HOSTED",
    "PRICING_PER_REQUEST",
    "PRICING_PER_WEIGHTED_UNIT",
    "PRODUCER_API_PROVIDER",
    "PRODUCER_MODEL_MEDIATED",
    "SCALE_RANK",
    "SCALE_UNBOUNDED",
    "SCALE_UNIT_INTERVAL",
    "SEARCH_RESULTS_METADATA_KEY",
    "SEARCH_RESULTS_SCHEMA_VERSION",
    "WELL_KNOWN_CRITERIA",
    "AuthFailed",
    "BudgetDecision",
    "BudgetPort",
    "Candidate",
    "CandidateSet",
    "ContentSlot",
    "Corpus",
    "CorpusEntry",
    "Criterion",
    "CriterionDisposition",
    "Disposition",
    "FailureRecord",
    "FetchTransport",
    "LocalCapExceeded",
    "Locator",
    "MalformedResponse",
    "Provenance",
    "ProviderCapabilities",
    "QuotaExhausted",
    "RateLimitDecision",
    "RateLimited",
    "RateLimiterPort",
    "ScoreEntry",
    "SearchFailure",
    "SearchProvider",
    "SearchRequest",
    "SearchResultsMetadata",
    "SearchTransport",
    "Spend",
    "TimedOut",
    "TransportFailed",
    "TransportResponse",
    "canonical_digest",
    "canonicalize",
    "get_capabilities",
    "list_capabilities",
    "register_capabilities",
]
