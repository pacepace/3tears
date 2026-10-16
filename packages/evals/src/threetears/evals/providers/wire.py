"""The three wires the client speaks: what a request carries, and what a reply says, per provider.

Pure: nothing here sends or logs. A request is built from the settings the negotiation chose
(:class:`~threetears.evals.negotiation.WireSettings`) under the facts it chose them by, and a reply is read into
:class:`Reply`, every count ``None`` where the provider reported none, never ``0``:

* **Anthropic's Messages API**: ``max_tokens`` (required), ``thinking`` as
  :func:`~threetears.evals.negotiation.anthropic_thinking` says (a budget as ``{"type": "enabled",
  "budget_tokens": N}``, which ``max_tokens`` must exceed), and one ``output_config`` holding the effort word and a
  strict schema's ``format``. No JSON mode exists there, so a format it cannot enforce is not sent.
* **OpenAI's chat completions** (and any OpenAI-compatible endpoint): ``reasoning_effort``, a strict
  ``json_schema`` ``response_format``, JSON mode where a schema cannot be enforced; ``max_completion_tokens`` on
  OpenAI itself, ``max_tokens`` on another endpoint. It takes no reasoning budget: the negotiation sends one as its
  word and names it unsent.
* **OpenRouter**: the OpenAI shape with ``reasoning.max_tokens`` for a budget or ``reasoning.effort`` for a word
  (never both: OpenRouter takes one), ``provider.require_parameters`` exactly when the request is enforced, and
  ``usage.include`` so the reply carries its cost.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Literal

from threetears.evals.negotiation import ProviderFamily, SettingFacts, WireSettings, anthropic_thinking
from threetears.evals.providers.failures import Provider, ProviderError
from threetears.evals.schema import StopReason

__all__ = [
    "BASE_URLS",
    "PriceSource",
    "FAMILIES",
    "KEY_VARIABLES",
    "Directive",
    "Reply",
    "declared_facts",
    "directive_of",
    "read_reply",
    "reported_cost",
    "request_body",
]


class PriceSource(StrEnum):
    """The words the shipped client writes as a completion's ``price_source``: who priced the call.

    Each member is the string itself (a ``StrEnum``), so a stored ``price_source`` compares equal to it. A host reading
    spend back compares against these rather than spelling the words.

    :ivar OPENROUTER_REPORTED: OpenRouter's own figure for the call (``usage.cost``).
    :ivar OPENROUTER_REPORTED_UPSTREAM: Under bring-your-own-key, OpenRouter's fee plus the upstream's charge
        (``usage.cost`` plus ``cost_details.upstream_inference_cost``).
    :ivar OPENROUTER_CATALOGUE: A direct call's tokens priced by OpenRouter's catalogue row. Written as a prefix: the
        source is ``openrouter-catalogue:<catalogue id>``, ending ``;cache-approximate`` when a cache read or
        write was priced at a rate the row does not list for it.
    """

    OPENROUTER_REPORTED = "openrouter:reported"
    OPENROUTER_REPORTED_UPSTREAM = "openrouter:reported+upstream"
    OPENROUTER_CATALOGUE = "openrouter-catalogue"


#: The request shape each provider speaks, as the shared negotiation names it.
FAMILIES: Final[Mapping[Provider, ProviderFamily]] = {
    "openrouter": "router",
    "anthropic": "anthropic",
    "openai": "openai",
}

#: The environment variable each provider's key is read from when none is passed.
KEY_VARIABLES: Final[Mapping[Provider, str]] = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}

#: Each provider's API root.
BASE_URLS: Final[Mapping[Provider, str]] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "openai": "https://api.openai.com/v1",
}

#: Claude 5.5 models take no temperature, on any route: none is ever sent to one.
_NO_TEMPERATURE: Final = re.compile(r"claude-[a-z]+-5[.-]5")

#: Anthropic's stop reasons, in the engine's words. ``model_context_window_exceeded`` is the window filling and
#: ``pause_turn`` a server-tool pause: output stopped early for a cause that is neither the cap nor a filter.
_ANTHROPIC_STOPS: Final[Mapping[str, StopReason]] = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "end_turn",
    "max_tokens": "max_tokens",
    "model_context_window_exceeded": "truncated",
    "pause_turn": "truncated",
    "refusal": "content_filter",
}

#: OpenAI-shaped finish reasons, in the engine's words. OpenRouter's own ``error`` is a provider-reported failure.
_OPENAI_STOPS: Final[Mapping[str, StopReason]] = {
    "stop": "end_turn",
    "tool_calls": "end_turn",
    "function_call": "end_turn",
    "length": "max_tokens",
    "content_filter": "content_filter",
    "error": "error",
}

#: The prefix of OpenRouter's own routers (``openrouter/auto``), which it resolves on its side as it does a ``~``
#: pointer and a ``-latest`` alias: a reply naming the requested id back names the pointer, not the model.
_ROUTERS: Final = "openrouter/"


def declared_facts(model_id: str) -> SettingFacts:
    """What the client knows of a model before any request, whatever its route: a Claude 5.5 takes no temperature.

    :param model_id: The model id.
    :ptype model_id: str
    :return: The facts; every one unknown but that.
    :rtype: SettingFacts
    """
    return SettingFacts(accepts_temperature=False) if _NO_TEMPERATURE.search(model_id.lower()) else SettingFacts()


@dataclass(frozen=True, slots=True)
class Directive:
    """A call's reply format, checked: its kind, and a schema's name, body and strictness.

    :ivar kind: ``json_schema``, ``json_object``, or ``None`` for free text.
    :ivar schema: The JSON schema a ``json_schema`` directive carries.
    :ivar name: The schema's name.
    :ivar strict: Whether the schema is to be enforced.
    """

    kind: Literal["json_schema", "json_object"] | None
    schema: Mapping[str, Any] | None = None  # Any: a JSON schema
    name: str = "response"
    strict: bool = False

    def asks_strict(self, provider: Provider) -> bool:
        """Whether the call asks for enforcement: a strict schema, and on OpenRouter any schema.

        :param provider: The provider.
        :ptype provider: Provider
        :return: Whether the request is asked enforced.
        :rtype: bool
        """
        if provider == "openrouter":
            return self.kind == "json_schema"
        return self.kind == "json_schema" and self.strict


def directive_of(response_format: Mapping[str, Any] | None) -> Directive:  # Any: a JSON directive
    """The call's ``response_format``, checked.

    :param response_format: ``None``, ``{"type": "json_object"}``, or ``{"type": "json_schema", "json_schema":
        {"name", "strict", "schema"}}``.
    :ptype response_format: Mapping[str, Any] | None
    :return: The directive.
    :rtype: Directive
    :raises ValueError: Any other directive, or a ``json_schema`` one without its schema.
    """
    if response_format is None:
        return Directive(None)
    kind = response_format.get("type")
    if kind == "json_object":
        return Directive("json_object")
    spec = response_format.get("json_schema")
    if kind == "json_schema" and isinstance(spec, Mapping) and isinstance(spec.get("schema"), Mapping):
        return Directive("json_schema", spec["schema"], str(spec.get("name") or "response"), spec.get("strict") is True)
    raise ValueError(
        f"response_format {kind!r} is not one the client sends: pass None, {{'type': 'json_object'}}, or a "
        "json_schema directive carrying its schema at ['json_schema']['schema']"
    )


def request_body(
    provider: Provider,
    model: str,
    *,
    system: str,
    user: str,
    sent: WireSettings,
    facts: SettingFacts,
    directive: Directive,
    direct_openai: bool,
    json_mode: bool,
) -> dict[str, Any]:  # Any: a JSON body
    """The request body that carries ``sent`` for a call with ``directive``.

    :param provider: The provider.
    :ptype provider: Provider
    :param model: The model id.
    :ptype model: str
    :param system: The system message; an empty one is not sent.
    :ptype system: str
    :param user: The one user message.
    :ptype user: str
    :param sent: The settings the request carries (:func:`~threetears.evals.negotiation.closest_settings`).
    :ptype sent: WireSettings
    :param facts: What was known of the model when they were chosen.
    :ptype facts: SettingFacts
    :param directive: The call's reply format.
    :ptype directive: Directive
    :param direct_openai: The endpoint is OpenAI's own, which takes ``max_completion_tokens``.
    :ptype direct_openai: bool
    :param json_mode: On OpenRouter, whether the model may be sent JSON mode when a schema cannot be enforced.
    :ptype json_mode: bool
    :return: The JSON body.
    :rtype: dict[str, Any]
    :raises ValueError: ``sent`` carries a reasoning budget the provider cannot take as built: one at or above
        Anthropic's ``max_tokens`` (naming both), or one on OpenAI's chat completions, which have no budget
        parameter. :func:`~threetears.evals.negotiation.closest_settings` never chooses either; refused here
        rather than clamped or dropped.
    """
    if provider == "anthropic":
        return _anthropic_body(model, system, user, sent, facts, directive)
    body: dict[str, Any] = {  # Any: a JSON body
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}],
    }
    if sent.max_tokens is not None:
        body["max_completion_tokens" if direct_openai else "max_tokens"] = sent.max_tokens
    if sent.temperature is not None:
        body["temperature"] = sent.temperature
    routed = provider == "openrouter"
    if sent.reasoning_max_tokens is not None:
        if not routed:
            raise ValueError(
                f"OpenAI's chat completions take no reasoning budget, and these settings carry one "
                f"({sent.reasoning_max_tokens}): send its effort word (closest_settings chooses it)"
            )
        # OpenRouter takes a budget or a word, not both: the budget is the bound, so it wins.
        body["reasoning"] = {"max_tokens": sent.reasoning_max_tokens}
    elif sent.reasoning_effort is not None:
        if routed:
            body["reasoning"] = {"effort": sent.reasoning_effort}
        else:
            body["reasoning_effort"] = sent.reasoning_effort
    if directive.kind == "json_schema" and directive.schema is not None and facts.structured_output is not False:
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": directive.name, "strict": directive.strict, "schema": dict(directive.schema)},
        }
    elif directive.kind is not None and (provider == "openai" or json_mode):
        # A schema the model cannot take, or JSON mode asked: JSON mode, never the same schema unenforced.
        body["response_format"] = {"type": "json_object"}
    if routed:
        if directive.asks_strict(provider):
            body["provider"] = {"require_parameters": sent.strict_output}
        body["usage"] = {"include": True}
    return body


def _anthropic_body(
    model: str, system: str, user: str, sent: WireSettings, facts: SettingFacts, directive: Directive
) -> dict[str, Any]:  # Any: a JSON body
    body: dict[str, Any] = {  # Any: a JSON body
        "model": model,
        "max_tokens": sent.max_tokens,
        "messages": [{"role": "user", "content": user}],
    }
    if system:
        body["system"] = system
    if sent.temperature is not None:
        body["temperature"] = sent.temperature
    budget = sent.reasoning_max_tokens
    if budget is not None and (sent.max_tokens is None or sent.max_tokens <= budget):
        raise ValueError(
            f"Anthropic's max_tokens ({sent.max_tokens}) must exceed the thinking budget ({budget}), which it "
            "includes: ask for a max_tokens above the budget"
        )
    if (thinking := anthropic_thinking(sent, facts)) is not None:
        body["thinking"] = thinking
    output_config: dict[str, Any] = {}  # Any: a JSON object
    if sent.reasoning_effort not in (None, "none"):
        output_config["effort"] = sent.reasoning_effort
    if directive.kind == "json_schema" and directive.schema is not None and facts.structured_output is not False:
        output_config["format"] = {"type": "json_schema", "schema": dict(directive.schema)}
    if output_config:
        body["output_config"] = output_config
    return body


@dataclass(frozen=True, slots=True, kw_only=True)
class Reply:
    """What a reply says, read off its body: text, counts, who served it, and why it stopped.

    :ivar content: The reply's text: text blocks only, never a thinking block.
    :ivar input_tokens: Prompt tokens, cache reads and writes included; ``None`` when unreported.
    :ivar output_tokens: Completion tokens, reasoning included; ``None`` when unreported.
    :ivar reasoning_tokens: The reasoning share of ``output_tokens``; ``None`` when unsplit.
    :ivar cache_tokens: Prompt tokens read from or written to a cache (0 when none were reported).
    :ivar cache_read_tokens: The share of ``cache_tokens`` read from a cache.
    :ivar cache_write_tokens: The share of ``cache_tokens`` written to one, both lifetimes.
    :ivar cache_write_1h_tokens: The one-hour share of ``cache_write_tokens``, or ``None`` when the reply wrote to a
        cache without splitting it by lifetime.
    :ivar served_model: The model the reply named; ``None`` when it named none, or named an alias back.
    :ivar served_provider: The upstream OpenRouter routed to; ``None`` elsewhere.
    :ivar stop_reason: Why it stopped, in the engine's words; an absent or unknown reason is ``truncated``.
    :ivar provider_stop_reason: The reason as the provider named it, or ``None``.
    :ivar usage: The reply's ``usage`` object, which OpenRouter's cost is read from.
    """

    content: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cache_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cache_write_1h_tokens: int | None
    served_model: str | None
    served_provider: str | None
    stop_reason: StopReason
    provider_stop_reason: str | None
    usage: Mapping[str, Any]  # Any: a JSON object


def _mapping(value: object) -> Mapping[str, Any]:  # Any: a JSON object
    return value if isinstance(value, Mapping) else {}


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _named(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _is_alias(model_id: str) -> bool:
    return model_id.startswith("~") or model_id.endswith("-latest") or model_id.startswith(_ROUTERS)


def read_reply(provider: Provider, model: str, status: int, body: Mapping[str, Any]) -> Reply:  # Any: JSON
    """A successful reply's members.

    :param provider: The provider that answered.
    :ptype provider: Provider
    :param model: The model id the request named.
    :ptype model: str
    :param status: The HTTP status, for an error the body carries in place of a reply.
    :ptype status: int
    :param body: The parsed JSON body.
    :ptype body: Mapping[str, Any]
    :return: The reply.
    :rtype: Reply
    :raises ProviderError: The body carries an error in place of a reply (OpenRouter answers some failures with a 200),
        or no choice.
    """
    if isinstance(body.get("error"), Mapping):
        code = _count(body["error"].get("code"))
        message = _named(body["error"].get("message"))
        raise ProviderError(provider, code if code is not None else status, message, body=dict(body))
    if provider == "anthropic":
        return _read_anthropic(body)
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise ProviderError(provider, status, "the reply carried no choice", body=dict(body))
    choice = choices[0]
    reply = _mapping(choice.get("message"))
    content = reply.get("content")
    text = content if isinstance(content, str) else ""
    if isinstance(content, list):
        text = "".join(
            part.get("text", "") for part in content if isinstance(part, Mapping) and part.get("type") == "text"
        )
    finish = _named(choice.get("finish_reason"))
    raw = (_named(choice.get("native_finish_reason")) or finish) if provider == "openrouter" else finish
    if _named(reply.get("refusal")):
        stop: StopReason = "content_filter"
    else:
        stop = _OPENAI_STOPS.get(finish, "truncated") if finish else "truncated"
    usage = _mapping(body.get("usage"))
    prompt_details = _mapping(usage.get("prompt_tokens_details"))
    read = _count(prompt_details.get("cached_tokens")) or 0
    written = _count(prompt_details.get("cache_write_tokens")) or 0
    served = _named(body.get("model"))
    if served is not None and provider == "openrouter" and served == model and _is_alias(model):
        served = None
    upstream = _named(body.get("provider")) if provider == "openrouter" else None
    return Reply(
        content=text,
        input_tokens=_count(usage.get("prompt_tokens")),
        output_tokens=_count(usage.get("completion_tokens")),
        reasoning_tokens=_count(_mapping(usage.get("completion_tokens_details")).get("reasoning_tokens")),
        cache_tokens=read + written,
        cache_read_tokens=read,
        cache_write_tokens=written,
        cache_write_1h_tokens=0 if not written else None,
        served_model=served,
        served_provider=upstream,
        stop_reason=stop,
        provider_stop_reason=raw,
        usage=usage,
    )


def _read_anthropic(body: Mapping[str, Any]) -> Reply:  # Any: JSON
    blocks = body.get("content")
    text = "".join(
        block.get("text", "")
        for block in (blocks if isinstance(blocks, list) else [])
        if isinstance(block, Mapping) and block.get("type") == "text"
    )
    raw = _named(body.get("stop_reason"))
    usage = _mapping(body.get("usage"))
    fresh = _count(usage.get("input_tokens"))
    read = _count(usage.get("cache_read_input_tokens")) or 0
    written = _count(usage.get("cache_creation_input_tokens")) or 0
    cache = read + written
    lifetimes = usage.get("cache_creation")
    one_hour = _count(lifetimes.get("ephemeral_1h_input_tokens")) if isinstance(lifetimes, Mapping) else None
    return Reply(
        content=text,
        input_tokens=fresh + cache if fresh is not None else None,
        output_tokens=_count(usage.get("output_tokens")),
        reasoning_tokens=_count(_mapping(usage.get("output_tokens_details")).get("thinking_tokens")),
        cache_tokens=cache,
        cache_read_tokens=read,
        cache_write_tokens=written,
        cache_write_1h_tokens=0 if not written else one_hour,
        served_model=_named(body.get("model")),
        served_provider=None,
        stop_reason=_ANTHROPIC_STOPS.get(raw, "truncated") if raw else "truncated",
        provider_stop_reason=raw,
        usage=usage,
    )


def reported_cost(usage: Mapping[str, Any]) -> tuple[float | None, str | None, str | None]:  # Any: JSON
    """OpenRouter's own figure for a call: the cost, where it came from, and why it is unpriced.

    Under bring-your-own-key (``is_byok``) ``cost`` is OpenRouter's fee alone and the upstream bills the key
    separately, which the reply states as ``cost_details.upstream_inference_cost``. So three cases: ``is_byok`` is
    ``True``, the call cost their sum, and a reply missing the upstream figure is unpriced rather than understated;
    ``is_byok`` is ``False``, the cost is ``cost``; ``is_byok`` is absent (or not a bool), the reply cannot say whether
    ``cost`` is the whole price or the fee alone, so it is unpriced. A reply with no cost is unpriced, never ``0``.

    :param usage: The reply's ``usage`` object.
    :ptype usage: Mapping[str, Any]
    :return: ``(cost, price_source, None)``, or ``(None, None, why)``.
    :rtype: tuple[float | None, str | None, str | None]
    """
    cost = usage.get("cost")
    if not isinstance(cost, int | float) or isinstance(cost, bool):
        return None, None, "OpenRouter's reply carried no cost"
    byok = usage.get("is_byok")
    if not isinstance(byok, bool):
        return None, None, "OpenRouter's reply carried no is_byok, so its cost may be OpenRouter's fee alone"
    if not byok:
        return float(cost), PriceSource.OPENROUTER_REPORTED, None
    upstream = _mapping(usage.get("cost_details")).get("upstream_inference_cost")
    if not isinstance(upstream, int | float) or isinstance(upstream, bool):
        return (
            None,
            None,
            (
                "a bring-your-own-key call whose reply carried no cost_details.upstream_inference_cost: OpenRouter's "
                "fee alone would understate it"
            ),
        )
    return float(cost) + float(upstream), PriceSource.OPENROUTER_REPORTED_UPSTREAM, None
