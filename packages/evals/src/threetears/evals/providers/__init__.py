"""The shipped provider adapter: a chat model 3tears-models builds, as the engine's completion client.

Install with ``pip install "3tears-evals[models]"``. The engine names no provider and imports nothing of this root
(``tests/test_providers_boundary.py`` holds both directions): a host either brings its own client, implementing
:class:`~threetears.evals.schema.CompletionClient`, or binds to the three names here. :func:`models_client` builds
one client for one model; :func:`models_clients` is the role factory a host hands
:class:`~threetears.evals.kernel.host.EvalHost` as ``clients``; :class:`ModelsFailureDescriber` is the
``failure_describer`` beside it.

**Why one adapter, over 3tears-models, and not one per SDK.** The family already has a provider layer: models'
factories for Anthropic, OpenAI and OpenRouter with capability metadata, and a registry of price sources whose
catch-all is OpenRouter's public catalogue, which prices Anthropic and OpenAI models too and bakes no prices in.
Three SDK adapters and a price table here would duplicate all of that and then drift from it. So the adapter
builds through ``create_chat_model``, prices through ``threetears.models.price_lookup``, and adds what the port
needs that models does not do for it: retries off, a private breaker, the stop reason in the engine's words, the
served model as the reply named it, and refusals where a setting cannot be carried. It imports only public names
of models and the public roots of this package.

**What it promises, member by member** (each provider's row is pinned by
``tests/test_models_adapter_dimensions.py``, so a LangChain or models upgrade that starts hiding or exposing a
member fails there, and the commit that moves the pin says so):

* ``content`` is the reply's text blocks (``AIMessage.text``), never a thinking block or a tool call, and under a
  schema the raw JSON text the model wrote, never a parsed object.
* ``input_tokens`` and ``output_tokens`` come from ``usage_metadata`` on every provider, ``None`` when the reply
  carried none; never models' ``extract_usage``, which coerces a missing count to 0 and estimates.
* ``reasoning_tokens``: OpenAI's and OpenRouter's reasoning split from ``usage_metadata``; Anthropic's thinking
  tokens from the response's own usage, which LangChain leaves in ``response_metadata``; ``None`` when unsplit.
* ``model`` is the id the client was built for; ``served_model`` the one the reply named. On OpenRouter, LangChain
  fills an empty reply model with the requested id, so a requested alias (``~…``, ``-latest``, ``openrouter/…``)
  echoed back reads ``None``; the serving upstream is kept beside it as ``served_provider``.
* ``stop_reason``: Anthropic's ``stop_reason`` and the OpenAI-shaped ``finish_reason`` mapped onto the engine's five
  words. A refusal (Anthropic's ``refusal``, OpenAI's ``refusal`` beside empty content) is ``content_filter``; an
  absent or unknown reason, and Anthropic's ``pause_turn``, is ``truncated``, cut short with the cause unnamed,
  never a finish; only OpenRouter's own ``error`` finish reason, a provider-reported failure, is ``error``.
* ``temperature`` is what the built model sends, read back from it: LangChain drops a temperature OpenAI's
  ``gpt-5`` family refuses, and a thinking budget on Anthropic is built with none.
* ``cost_usd`` and ``price_source``: OpenRouter's own reported figure (``openrouter:reported``; under
  bring-your-own-key that is OpenRouter's fee alone, since LangChain drops the flag that would tell the upstream's
  charge apart, so a BYOK call is understated); on Anthropic and OpenAI the
  tokens times models' catalogue rates for the requested model (``<source>:<matched id>``), the catalogue read once
  per client. Unpriced is ``None``, never ``0``, and the completion's ``pricing_failure`` (and the client's
  ``last_pricing_failure``) says why. Cache-read discounts are not in the catalogue's rates, so a cached Anthropic
  prompt is priced at the full input rate: an overstatement, never an understatement.

What the request carries: retries off on every provider (the Anthropic and OpenAI SDKs retry twice by default,
OpenAI an exhausted quota too, and the OpenRouter SDK, given no retry setting of LangChain's, backs off on a 5XX for
up to an hour), streaming off, a breaker that never opens, the output cap and the reasoning
setting as asked or a refusal by name (:func:`models_client` lists them per provider), and a ``response_format``
through models' structured-output translation.

**The Claude CLI route is refused, for now.** models routes an Anthropic subscription token to the Claude CLI,
whose wrapper reports every completion as finished (it synthesises ``stop``) and drops the served model and the
reasoning tokens. A judge on that route cannot see its own truncation, which is a correctness hazard rather than a
missing nicety, and the CLI retries internally with no way to turn it off. The route opens in a later release, once
3tears-models carries the CLI's real stop reason and served model.

**This module is the package's public root** for the adapter. A host imports from here and from no module below
it, and only the names in ``__all__``.
"""

from __future__ import annotations

from threetears.evals.providers.adapter import (
    PROVIDER_KEY_VARIABLES,
    SUPPORTED_PROVIDERS,
    ModelsClientRefused,
    ModelsCompletion,
    ModelsCompletionClient,
    SupportedProvider,
    models_client,
)
from threetears.evals.providers.factory import ModelsClients, models_clients
from threetears.evals.providers.failures import ModelsFailureDescriber

__all__ = [
    "PROVIDER_KEY_VARIABLES",
    "SUPPORTED_PROVIDERS",
    "ModelsClientRefused",
    "ModelsClients",
    "ModelsCompletion",
    "ModelsCompletionClient",
    "ModelsFailureDescriber",
    "SupportedProvider",
    "models_client",
    "models_clients",
]
