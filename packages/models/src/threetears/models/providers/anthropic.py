"""anthropic chat factory returning a configured ``langchain_anthropic.ChatAnthropic``.

LangChain-native shape (3tears v0.6.0+): :func:`create_anthropic_chat`
returns a fully-configured ``ChatAnthropic`` instance. Capability metadata
for known Anthropic model ids is registered with the module-level
:func:`~threetears.models.capabilities.register_capabilities` registry at
import time so consumers can ``get_capabilities(model_id)`` without
instantiating the provider.

Tool-name translation: the Anthropic Messages API validates tool names
against ``^[a-zA-Z0-9_-]{1,128}$`` and rejects the dot. Canonical 3tears
tool names use the dotted form (``threetears.calculator``,
``3tears.admin.agent_management``), so :func:`create_anthropic_chat`
returns a :class:`_NameTranslatingChatAnthropic` subclass -- a thin binding
of the shared
:class:`~threetears.models.providers._name_translation_mixin.NameTranslatingChatMixin`
(identical hooks across the OpenAI / OpenRouter / Anthropic wrappers) --
that translates dot-to-underscore on outgoing tool specs / history and
underscore-to-dot on incoming ``tool_calls``. Application code never sees
the wire form.

Structured output is the same class of wire quirk:
:func:`anthropic_structured_output_kwargs` builds the Anthropic-native
directive so no consumer has to know the provider's spelling of "return
json matching this schema". Dispatch across providers lives in
:mod:`threetears.models.providers.structured_output`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import PrivateAttr

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.providers._name_translation_mixin import NameTranslatingChatMixin
from threetears.models.providers.structured_output import StructuredOutputSchemaError

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.language_models import BaseChatModel

__all__ = [
    "ANTHROPIC_PROVIDER_NAME",
    "anthropic_structured_output_kwargs",
    "create_anthropic_chat",
    "strip_v1_suffix",
]


ANTHROPIC_PROVIDER_NAME = "anthropic"


def strip_v1_suffix(url: str) -> str:
    """strips trailing ``/v1`` or ``/v1/`` from URL.

    Anthropic SDK auto-appends ``/v1``, so passing a URL ending with
    ``/v1`` would cause doubled path segments.

    :param url: base URL to clean
    :ptype url: str
    :return: URL with ``/v1`` suffix removed if present
    :rtype: str
    """
    if url.endswith("/v1"):
        return url[:-3]
    if url.endswith("/v1/"):
        return url[:-4]
    return url


def anthropic_structured_output_kwargs(
    json_schema: dict[str, Any],
    *,
    name: str = "response",
    strict: bool = True,
) -> dict[str, Any]:
    """builds the Anthropic-native structured-output bind kwargs.

    Anthropic spells the directive ``output_config={"format": {"type":
    "json_schema", "schema": ...}}``. A directly-bound ``output_config``
    is forwarded verbatim to the Anthropic SDK ``messages.create``, so
    the schema is pre-transformed through :func:`anthropic.transform_schema`
    -- that is what enforces the API's json-schema limits (e.g. injecting
    ``additionalProperties: false``) instead of failing at request time.

    Returns bind KWARGS rather than a bound model on purpose: the caller
    may be applying this to a model that is already a ``bind_tools``
    ``RunnableBinding``, which exposes ``.bind(**kwargs)`` but none of the
    wrapper class's own methods.

    ``name`` and ``strict`` are accepted for signature parity across the
    per-provider builders and are unused here: Anthropic's format block
    has no name field, and the schema constraint is always strict (the
    ``additionalProperties: false`` that ``transform_schema`` injects).

    ``transform_schema`` validates LOCALLY, before any API call, and
    signals rejection with bare builtin exceptions rather than a typed
    SDK error, so a malformed schema is translated here into
    :class:`~threetears.models.providers.structured_output.StructuredOutputSchemaError`.
    Left bare, the raise is unclassifiable by callers -- an
    ``AssertionError`` in particular reads as an internal invariant
    failure, so a consumer's error classifier routes it to its
    provider-unavailable branch and a circuit breaker keyed on that
    verdict takes the provider out of service for every caller over one
    caller's bad schema.

    Note on ``python -O``: the ``AssertionError`` here is NOT stripped
    under ``-O``. It comes from :func:`typing.assert_never`, which ends in
    an explicit ``raise AssertionError(...)`` statement rather than an
    ``assert`` statement, and only the latter is elided by ``-O``; the
    transform module contains no ``assert`` statement at all. This wrap is
    therefore load-bearing under every interpreter flag. Were that ever to
    change, the degradation is still safe: the untranslated schema would
    reach the provider, the provider would answer with its own 400, and a
    consumer's error classifier maps that to a bad-request verdict rather
    than an outage -- so the schema is rejected either way and the circuit
    breaker stays untouched either way.

    :param json_schema: json-schema the response must satisfy
    :ptype json_schema: dict[str, Any]
    :param name: schema name (unused by Anthropic; present for parity)
    :ptype name: str
    :param strict: strict-schema flag (unused by Anthropic; present for parity)
    :ptype strict: bool
    :return: kwargs to pass to ``model.bind(**kwargs)``
    :rtype: dict[str, Any]
    :raises StructuredOutputSchemaError: when json_schema cannot be
        translated into Anthropic's structured-output schema form
    """
    # imported inside the function so the module-scope import stays
    # metadata-only: threetears.models eagerly imports this module to
    # populate the capability registry and must not require any provider
    # SDK to be installed for that side-effect import to succeed.
    from anthropic import transform_schema

    try:
        transformed = transform_schema(json_schema)
    except (AssertionError, ValueError, AttributeError, TypeError, RecursionError) as exc:
        # the exact tuple anthropic.lib._parse._transform.transform_schema
        # raises on caller-supplied schema data, each verified against a
        # malformed schema: AssertionError from the assert_never on an
        # unrecognized "type"; ValueError when no type/anyOf/oneOf/allOf
        # is present; AttributeError from .items() when "properties" or
        # "$defs" is not a mapping; TypeError from the {**schema} unpack
        # when a nested property/item value is not a mapping or when the
        # schema itself is not a mapping; RecursionError because the
        # transform recurses once per nesting level with no depth guard,
        # so caller json nested past the interpreter frame limit exhausts
        # the stack (reachable well inside what json.loads accepts).
        # Deliberately explicit rather than a broad catch so a genuine
        # defect in this module still surfaces as itself.
        raise StructuredOutputSchemaError(
            "supplied json schema was rejected by anthropic structured-output "
            f"translation: {type(exc).__name__}: {exc}",
        ) from exc

    return {
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": transformed,
            },
        },
    }


def _build_translating_chat_class() -> type[ChatAnthropic]:
    """build the :class:`ChatAnthropic` subclass with name-translation hooks.

    Defined inside a function so the langchain-anthropic import stays
    lazy -- :mod:`threetears.models.providers.anthropic` is imported
    eagerly at package load to populate the capability registry, and
    we must not require ``langchain-anthropic`` to be installed for
    that side-effect import to succeed (it is an optional dependency
    selected via ``3tears-models[anthropic]``).

    :return: name-translating ChatAnthropic subclass
    :rtype: type[ChatAnthropic]
    """
    from langchain_anthropic import ChatAnthropic

    class _NameTranslatingChatAnthropic(NameTranslatingChatMixin, ChatAnthropic):
        """``ChatAnthropic`` with dot<->underscore tool-name translation at the
        wire boundary (Anthropic validates names against
        ``^[a-zA-Z0-9_-]{1,128}$`` and rejects the dot).

        All translation hooks live in :class:`NameTranslatingChatMixin`
        (mixed in ahead of ``ChatAnthropic`` so ``super()`` resolves to it);
        this subclass only supplies the per-instance reverse-map slot.

        :ivar _name_reverse_map: underscored-wire -> canonical-dotted map,
            populated at ``bind_tools`` time.
        :ptype _name_reverse_map: dict[str, str]
        """

        _name_reverse_map: dict[str, str] = PrivateAttr(default_factory=dict)

    return _NameTranslatingChatAnthropic


def create_anthropic_chat(
    model_name: str,
    api_key: str,
    *,
    base_url: str | None = None,
    timeout: int = 120,
    max_retries: int = 2,
    **extra_kwargs: object,
) -> BaseChatModel:
    """creates a configured ``ChatAnthropic`` for Anthropic models.

    Returns the :class:`_NameTranslatingChatAnthropic` subclass, not
    a vanilla ``ChatAnthropic``. Application code interacts with it
    exactly the same way (it IS a ``ChatAnthropic``); the only
    difference is the wire-side tool-name translation that hides
    Anthropic's strict tool-name regex from the rest of the
    codebase.

    :param model_name: Anthropic model identifier (e.g. ``claude-sonnet-4-20250514``)
    :ptype model_name: str
    :param api_key: Anthropic API key
    :ptype api_key: str
    :param base_url: optional custom API base URL; trailing ``/v1`` is stripped
    :ptype base_url: str | None
    :param timeout: request timeout in seconds
    :ptype timeout: int
    :param max_retries: maximum retry attempts for failed requests
    :ptype max_retries: int
    :param extra_kwargs: additional keyword arguments forwarded to ``ChatAnthropic``
    :ptype extra_kwargs: object
    :return: configured ``ChatAnthropic`` (the name-translating subclass), or a Claude
        **subscription**-backed model when ``api_key`` is an OAuth token (``sk-ant-oat…``).
    :rtype: BaseChatModel
    """
    # A Claude subscription OAuth token (``claude setup-token``) routes to the CLI/Agent-SDK backend
    # instead of the HTTP API — the SAME Anthropic model ids, no separate provider. An API key
    # (``sk-ant-api…``) takes the ChatAnthropic path below. Imported lazily so the optional
    # ``langchain-claude-code`` dep is only pulled when a subscription token is actually used.
    from threetears.models.providers._claude_cli import create_subscription_chat, is_subscription_token

    if is_subscription_token(api_key):
        return create_subscription_chat(model_name, api_key, **extra_kwargs)

    chat_cls = _build_translating_chat_class()

    cleaned_base_url = strip_v1_suffix(base_url) if base_url else None

    kwargs: dict[str, object] = {
        "model_name": model_name,
        "api_key": api_key,
        "timeout": timeout,
        "max_retries": max_retries,
    }
    if cleaned_base_url is not None:
        kwargs["base_url"] = cleaned_base_url
    kwargs.update(extra_kwargs)

    model: ChatAnthropic = chat_cls(**kwargs)
    return model


# -- capability registration -------------------------------------------------

# canonical Anthropic chat models. extend with additional ids by calling
# register_capabilities() externally at host-app boot time.
_ANTHROPIC_CAPABILITIES: dict[str, ModelCapabilities] = {
    "claude-opus-4-8": ModelCapabilities(
        model_name="claude-opus-4-8",
        provider_name=ANTHROPIC_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        context_window=200_000,
        max_output_tokens=64_000,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        requires_alternating_roles=False,
        supports_anthropic_cache_control=True,
        supports_openai_auto_cache=False,
        min_cacheable_tokens=1024,
        cache_ttl_seconds=300,
    ),
    "claude-sonnet-5": ModelCapabilities(
        model_name="claude-sonnet-5",
        provider_name=ANTHROPIC_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        context_window=200_000,
        max_output_tokens=64_000,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        requires_alternating_roles=False,
        supports_anthropic_cache_control=True,
        supports_openai_auto_cache=False,
        min_cacheable_tokens=1024,
        cache_ttl_seconds=300,
    ),
    "claude-sonnet-4-6": ModelCapabilities(
        model_name="claude-sonnet-4-6",
        provider_name=ANTHROPIC_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        context_window=200_000,
        max_output_tokens=64_000,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        requires_alternating_roles=False,
        supports_anthropic_cache_control=True,
        supports_openai_auto_cache=False,
        min_cacheable_tokens=1024,
        cache_ttl_seconds=300,
    ),
    "claude-haiku-4-5-20251001": ModelCapabilities(
        model_name="claude-haiku-4-5-20251001",
        provider_name=ANTHROPIC_PROVIDER_NAME,
        model_type=ModelType.CHAT,
        model_tier=ModelTier.SMALL,
        model_status=ModelStatus.ACTIVE,
        context_window=200_000,
        max_output_tokens=8_192,
        supports_streaming=True,
        supports_tools=True,
        supports_vision=True,
        requires_alternating_roles=False,
        supports_anthropic_cache_control=True,
        supports_openai_auto_cache=False,
        min_cacheable_tokens=1024,
        cache_ttl_seconds=300,
    ),
}


for _model_id, _caps in _ANTHROPIC_CAPABILITIES.items():
    register_capabilities(_model_id, _caps)
