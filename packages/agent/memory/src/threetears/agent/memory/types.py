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

    # Retrieval thresholds. These two are NOT on the same scale, and reading
    # them as a pair is the mistake to avoid: ``similarity_threshold`` is a
    # cosine floor deciding whether a candidate is admitted at all, while
    # ``detail_threshold`` gates the composite ``hybrid_score`` deciding whether
    # an admitted item renders in full or as a summary. Admission deliberately
    # ignores recency (see ``collections._admit_and_rank``); the detail cut does
    # not, so an aged item renders summary-only however relevant it is — which
    # is what the pull-not-push doctrine wants, since the agent reaches full
    # content through ``memory_recall`` / ``chunk_recall``.
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

    # Salience decay + reinforcement (v024 substrate; luminara defaults).
    # Injectable so metallm resolves per-user values and passes them in;
    # the defaults reproduce the luminara reference exactly.
    salience_half_life_days: float = 60.0
    salience_floor: float = 0.1
    salience_access_bump: float = 0.05
    salience_ambient_floor: float = 0.2
    salience_seed: float = 0.5

    # What ambient recall (``MemoryRetriever``) is handed, beyond the ranking.
    # Each default is "as before", so a consumer that sets none sees no change.
    #
    # ``ambient_recent_cap``: at most this many of the memories ambient recall
    # returns were written in the last ``ambient_recent_days``. The ranking's
    # recency term lifts everything written in the last day or two above
    # everything older, so a burst of new memories fills the list and the long
    # history never shows; the cap leaves the rest of the list to it. Which
    # recent memories take the capped places is decided on how well each
    # answers the message (semantic and keyword), never on how new it is.
    # ``None`` is no cap.
    ambient_recent_cap: int | None = None
    ambient_recent_days: float = 7.0
    # Memories carrying any of these tags are searchable and never surface in
    # ambient recall: e.g. ``origin.UNATTENDED_TAG``, a memory written on a
    # turn nobody was present for.
    ambient_excluded_tags: tuple[str, ...] = ()
    # Whether surfacing a memory in ambient recall raises its salience. On, a
    # memory that surfaced keeps surfacing while one that did not sinks below
    # ``salience_ambient_floor`` and never does again. Off, only a recall the
    # agent asked for counts (the ``salience_bump`` of ``memory_recall`` and
    # ``memory_search``).
    ambient_bumps_salience: bool = True

    # One event written twice. A new memory whose source message is one of the
    # turn's ``near_message_ids`` (this turn and the one before) replaces a
    # stored memory from those same messages when their cosine similarity is
    # at least this. ``None`` leaves the rule off. Lower than the dedup
    # thresholds on purpose: two memories from one turn that say nearly the
    # same thing are one event, where two from different months may be two.
    same_event_threshold: float | None = None

    # Dream consolidation (A5; luminara defaults). Injectable per-user like
    # the salience knobs. ``cluster_threshold`` is the cosine floor that
    # groups near-duplicate memories (luminara clustering.py:57);
    # ``min_cluster_size`` is the smallest cluster worth a gist; the gist
    # seeds its salience at ``gist_salience_seed``.
    consolidation_cluster_threshold: float = 0.85
    consolidation_min_cluster_size: int = 2
    consolidation_gist_salience_seed: float = 0.5
