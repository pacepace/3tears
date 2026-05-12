"""Memory type classification and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "MemoryConfig",
    "MemoryType",
]


class MemoryType(str, Enum):
    """Extracted memory category."""

    PREFERENCE = "preference"
    FACT = "fact"
    DECISION = "decision"
    TOPICAL_CONTEXT = "topical_context"
    RELATIONAL_CONTEXT = "relational_context"


@dataclass
class MemoryConfig:
    """Configuration for memory retrieval and extraction."""

    # Retrieval thresholds
    similarity_threshold: float = 0.4
    detail_threshold: float = 0.85
    context_budget: int = 15

    # Memory types
    enabled_types: list[str] = field(
        default_factory=lambda: [
            "preference",
            "fact",
            "decision",
            "topical_context",
            "relational_context",
        ]
    )

    # Three-signal hybrid ranking (memories + media_content)
    signal_weights: dict[str, float] = field(
        default_factory=lambda: {"semantic": 0.55, "keyword": 0.15, "recency": 0.30}
    )
    recency_half_life_hours: float = 24.0

    # Two-signal hybrid ranking (chunks — no recency)
    chunk_signal_weights: dict[str, float] = field(default_factory=lambda: {"semantic": 0.80, "keyword": 0.20})

    # Candidate fetch multipliers
    top_k: int = 10
    candidate_multiplier: int = 3
    media_top_k: int = 5
    media_candidate_multiplier: int = 3
    chunk_candidate_k: int = 15

    # MMR parameters
    mmr_lambda: float = 0.7

    # FTS constraints
    fts_min_query_len: int = 3
    fts_max_query_len: int = 500

    # Extraction settings
    extraction_min_user_message_length: int = 20
    extraction_min_assistant_response_length: int = 100
    extraction_min_conversation_turns: int = 3
    extraction_rate_limit_cooldown_seconds: int = 300

    # Extraction similarity search
    similar_memory_top_k: int = 5
    similar_memory_threshold: float = 0.55
