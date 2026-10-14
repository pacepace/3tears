"""The completion client: a model id in, the engine's completion port out, over each provider's own HTTP API.

``Model("deepseek/deepseek-v4-flash")`` is a working client: it goes to OpenRouter on ``OPENROUTER_API_KEY``, and
``Model("claude-haiku-5-5", provider="anthropic")`` and ``Model("gpt-5-nano", provider="openai")`` go to those APIs
directly (``base_url=`` takes any OpenAI-compatible endpoint). It implements
:class:`~threetears.evals.schema.CompletionClient` and :class:`~threetears.evals.schema.BoundCompletionClient`, so
it is a judge's client, a role factory's, or a runner's own (``model.text``, ``model.json``, ``model.prompt``). It
depends on ``httpx`` and on no SDK, no LangChain and no other 3tears package (``tests/test_providers_boundary.py``).
An app already on another client wraps it in the :class:`~threetears.evals.schema.CompletionClient` port instead,
and :func:`~threetears.evals.testing.check_completion_conformance` checks it.

What a completion carries, member by member:

* ``content`` is the reply's text blocks, never a thinking block; under a schema, the JSON the model wrote.
* ``input_tokens``, ``output_tokens`` and ``reasoning_tokens`` are the provider's counts, ``None`` when it reported
  none, never ``0``.
* ``served_model`` is what the reply named, ``None`` when it named a requested OpenRouter alias back
  (``~…``, ``-latest``, ``openrouter/…``); ``served_provider`` is the upstream OpenRouter routed to.
* ``stop_reason`` is the engine's word: Anthropic's ``end_turn``, ``stop_sequence`` and ``tool_use`` and the
  OpenAI shape's ``stop`` finish; ``max_tokens`` and ``length`` are ``max_tokens``; a refusal is
  ``content_filter``; OpenRouter's ``error`` is ``error``; an absent or unknown reason, Anthropic's ``pause_turn``
  and ``model_context_window_exceeded``, is ``truncated``, logged at WARNING with ``provider_stop_reason``, the raw
  value.
* ``temperature`` and ``sent_settings`` are what the served request carried: the closest settings the model takes
  (:mod:`threetears.evals.negotiation` chooses them, reads a refusal and resends; nothing else is retried).
* ``cost_usd``: on OpenRouter its own reported figure, and under bring-your-own-key that figure plus
  ``cost_details.upstream_inference_cost`` (a BYOK reply without it is unpriced); on Anthropic and OpenAI, tokens
  times OpenRouter's catalogue rates for the matching ``vendor/model`` row. ``price_source`` names which, in
  :class:`PriceSource`'s words (``openrouter:reported``, ``openrouter:reported+upstream``,
  ``openrouter-catalogue:<id>``). Unpriced is ``None`` with
  ``pricing_failure`` saying why, never ``0``. The catalogue refreshes every ``refresh_after`` (six hours by
  default); a catalogue-priced cost whose refresh failed carries ``pricing_stale_since``, when its rates expired.

**This module is the package's public root** for the client. A host imports from here and from no module below it,
and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.providers.catalogue import Catalogue
from threetears.evals.providers.failures import Provider, ProviderError, describe_failure
from threetears.evals.providers.model import Clients, Completion, Model
from threetears.evals.providers.wire import PriceSource

__all__ = [
    "Catalogue",
    "Clients",
    "Completion",
    "Model",
    "PriceSource",
    "Provider",
    "ProviderError",
    "describe_failure",
]
