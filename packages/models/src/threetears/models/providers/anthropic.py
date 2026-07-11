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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import PrivateAttr

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.providers._name_translation_mixin import NameTranslatingChatMixin

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.language_models import BaseChatModel

__all__ = [
    "ANTHROPIC_PROVIDER_NAME",
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
