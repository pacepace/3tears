"""Bulk-register product model rows in the shared capabilities registry.

Every product that talks to LLMs stores its own per-row model metadata
somewhere -- metallm has the ``models`` table, 14-eng-ai-bot has its
hub equivalent, the agent products will too.  All of them want the
same downstream observability: cost-per-call recorded on
``threetears_llm_cost_usd_total``, latency histogram on
``threetears_llm_latency_seconds``, ``llm.usage`` OTel spans tagged
with ``llm.tier``.  All of those flows depend on
:class:`threetears.models.capabilities.ModelCapabilities` being
registered for the model_name the factory is invoked with.

This module lets every product feed its rows into the registry with a
single call instead of each one writing its own dict-to-Pydantic
conversion.  Stable kwarg shape across products -- pre-existing
callers (the per-provider ``register_capabilities()`` calls done at
import time by the LangChain provider modules) keep working
unchanged.

Typical usage from a FastAPI lifespan startup::

    from threetears.models.registry_loader import register_model_capabilities_bulk

    rows = await pool.fetch(
        '''
        SELECT m.name_api AS model_name,
               p.name AS provider_name,
               m.cost_per_1m_prompt_tokens AS cost_per_1m_input_tokens,
               m.cost_per_1m_completion_tokens AS cost_per_1m_output_tokens,
               m.context_window_tokens AS context_window,
               m.supports_streaming, m.supports_vision
          FROM models m
          JOIN providers p ON m.provider_id = p.provider_id
        '''
    )
    register_model_capabilities_bulk(dict(r) for r in rows)

The registry is process-local and refreshed only at startup (or on an
explicit re-call); admin edits to per-row costs require a process
restart to take effect on cost accounting.  Hot-reload-on-admin-edit
is filed as a follow-up.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Iterable, Mapping

from threetears.models.capabilities import (
    ModelCapabilities,
    register_capabilities,
)
from threetears.models.enums import ModelStatus, ModelTier, ModelType

__all__ = [
    "register_model_capabilities_bulk",
]


_PER_MILLION = Decimal(1_000_000)


def _per_token(per_million: Any) -> Decimal | None:
    """convert a per-1M-tokens cost (admin-friendly) to per-token Decimal.

    accepts ``None`` / numeric / numeric-string / ``Decimal`` inputs
    so callers can pass straight asyncpg row values without coercion.
    returns ``None`` for absent data so the registry's
    ``cost_per_input_token`` / ``cost_per_output_token`` fields stay
    ``None`` and the UsageTrackingCallback skips the cost increment
    rather than recording a zero (which would dilute averages).

    :param per_million: cost per 1M tokens in USD
    :ptype per_million: Any
    :return: cost per token as Decimal, or ``None``
    :rtype: Decimal | None
    """
    if per_million is None:
        return None
    if isinstance(per_million, Decimal):
        result = per_million / _PER_MILLION
    else:
        # str() round-trip avoids float-induced precision drift on
        # asyncpg numeric values that arrive as Python float on some
        # codecs.
        result = Decimal(str(per_million)) / _PER_MILLION
    return result


def _resolve_enum(value: Any, enum_cls: Any, default: Any) -> Any:
    """coerce a row value to an enum member, with a default fallback.

    accepts: ``None`` (returns default), already-an-enum-instance
    (returns as-is), string (looks up by value).  Unknown strings
    fall back to the default rather than raising so a single
    bad row never blocks startup.

    :param value: row value (enum, string, or None)
    :ptype value: Any
    :param enum_cls: target enum class
    :ptype enum_cls: Any
    :param default: fallback value
    :ptype default: Any
    :return: enum member
    :rtype: Any
    """
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    try:
        return enum_cls(value)
    except ValueError, KeyError:
        return default


def register_model_capabilities_bulk(
    rows: Iterable[Mapping[str, Any]],
    *,
    default_model_type: ModelType = ModelType.CHAT,
    default_model_tier: ModelTier = ModelTier.MEDIUM,
    default_model_status: ModelStatus = ModelStatus.ACTIVE,
) -> int:
    """register every row in the shared 3tears capabilities registry.

    Each row is a mapping with a flexible set of keys; the loader
    accepts both the metallm column names (``cost_per_1m_prompt_tokens``,
    ``context_window_tokens``) and the canonical 3tears names
    (``cost_per_1m_input_tokens``, ``context_window``).  Either set
    works; the loader picks whichever is present.

    Required:
      - ``model_name`` (or ``name_api`` or ``model_id``) -- the
        registry key.  This must match what callers pass to
        :func:`threetears.models.create_chat_model`.

    Common optional keys:
      - ``provider_name``
      - ``cost_per_1m_input_tokens`` / ``cost_per_1m_prompt_tokens``
      - ``cost_per_1m_output_tokens`` /
        ``cost_per_1m_completion_tokens``
      - ``context_window`` / ``context_window_tokens``
      - ``supports_streaming``
      - ``supports_tools``
      - ``supports_vision``
      - ``model_type`` (defaults to CHAT)
      - ``model_tier`` (defaults to STANDARD)
      - ``model_status`` (defaults to ACTIVE)

    Rows missing the model-name key are silently skipped (so a NULL
    column doesn't blow up startup); the count returned reflects
    only successfully registered rows.

    :param rows: iterable of row dicts
    :ptype rows: Iterable[Mapping[str, Any]]
    :param default_model_type: applied when the row omits ``model_type``
    :ptype default_model_type: ModelType
    :param default_model_tier: applied when the row omits ``model_tier``
    :ptype default_model_tier: ModelTier
    :param default_model_status: applied when the row omits ``model_status``
    :ptype default_model_status: ModelStatus
    :return: count of rows registered
    :rtype: int
    """
    count = 0
    for row in rows:
        # Resolve canonical model name -- the registry key.  Try the
        # metallm column name first, then alternates, so the same
        # helper works for both metallm and the hub without a
        # per-product adapter.
        name_value = row.get("model_name") or row.get("name_api") or row.get("model_id")
        if not name_value:
            continue
        model_name = str(name_value)

        caps = ModelCapabilities(
            model_name=model_name,
            model_type=_resolve_enum(
                row.get("model_type"),
                ModelType,
                default_model_type,
            ),
            model_tier=_resolve_enum(
                row.get("model_tier"),
                ModelTier,
                default_model_tier,
            ),
            model_status=_resolve_enum(
                row.get("model_status"),
                ModelStatus,
                default_model_status,
            ),
            provider_name=row.get("provider_name"),
            context_window=(row.get("context_window") or row.get("context_window_tokens")),
            max_output_tokens=row.get("max_output_tokens"),
            supports_streaming=row.get("supports_streaming"),
            supports_tools=row.get("supports_tools"),
            supports_vision=row.get("supports_vision"),
            cost_per_input_token=_per_token(
                row.get("cost_per_1m_input_tokens") or row.get("cost_per_1m_prompt_tokens"),
            ),
            cost_per_output_token=_per_token(
                row.get("cost_per_1m_output_tokens") or row.get("cost_per_1m_completion_tokens"),
            ),
            cost_per_cache_read_token=_per_token(row.get("cost_per_1m_cache_read_tokens")),
            cost_per_cache_write_token=_per_token(row.get("cost_per_1m_cache_write_tokens")),
        )
        register_capabilities(model_name, caps)
        count += 1
    return count
