"""The provider adapter's vocabulary: which providers it builds a client for, and how it refuses a setting.

The client itself, :func:`models_client`, builds a chat model through :func:`threetears.models.create_chat_model`;
this module names the providers it supports and resolves a model id to one through models' capability registry,
refusing an id the registry cannot place, or a provider the adapter does not build for, by name
(:class:`ModelsClientRefused`).
"""

from __future__ import annotations

from typing import Final, Literal, get_args

from threetears.models import get_capabilities

__all__ = [
    "SUPPORTED_PROVIDERS",
    "ModelsClientRefused",
    "SupportedProvider",
    "resolve_provider",
]

#: The providers the adapter builds a client for, as models' registry names them. The Claude CLI route (the
#: ``anthropic`` provider given a subscription token) is refused even though models routes it: see
#: :func:`models_client`.
SupportedProvider = Literal["anthropic", "openai", "openrouter"]

#: Every :data:`SupportedProvider`, in the order the docs list them.
SUPPORTED_PROVIDERS: Final[tuple[SupportedProvider, ...]] = get_args(SupportedProvider)


class ModelsClientRefused(ValueError):
    """A setting the adapter cannot carry to the provider, refused before anything is sent.

    Raised at construction for a provider, model, reasoning word or budget, or provider keyword the adapter cannot
    send as asked; by :meth:`ModelsCompletionClient.generate` for a ``response_format`` it cannot translate; and
    by :class:`ModelsClients` for a role with no model or a provider with no key, at the first call that needs
    one. A ``ValueError``, so a caller already catching one keeps catching it. The message names the setting, why
    it cannot be sent, and what to pass instead.

    Attributes:
        setting: The parameter or keyword refused, as the caller wrote it.
        provider: The provider the client was being built for, or ``None`` when none was resolved yet.
    """

    def __init__(self, setting: str, why: str, *, provider: str | None = None) -> None:
        """Name the refused setting and why.

        Args:
            setting: The parameter or keyword refused.
            why: Why it cannot be sent and what to pass instead, as a clause.
            provider: The provider the client was being built for, when one was resolved.
        """
        self.setting = setting
        self.provider = provider
        where = f" on {provider}" if provider else ""
        super().__init__(f"{setting} refused{where}: {why}")


def resolve_provider(model_id: str, provider: str | None) -> SupportedProvider:
    """The provider a model id is built on: ``provider`` when given, else the one models' registry names.

    Args:
        model_id: The model id as the host names it.
        provider: An explicit provider, which overrides the registry.

    Returns:
        The provider.

    Raises:
        ModelsClientRefused: The registry cannot place ``model_id`` and no ``provider`` was given, or the provider
            is not one the adapter supports.
    """
    if provider is None:
        capabilities = get_capabilities(model_id)
        if capabilities is None or not capabilities.provider_name:
            raise ModelsClientRefused(
                "model_id",
                f"3tears-models' capability registry cannot place {model_id!r}; pass provider= (one of "
                f"{', '.join(SUPPORTED_PROVIDERS)}), or register the model's capabilities "
                "(threetears.models.register_capabilities) first",
            )
        provider = capabilities.provider_name
    for supported in SUPPORTED_PROVIDERS:
        if provider == supported:
            return supported
    raise ModelsClientRefused(
        "provider",
        f"{provider!r} is not a provider this adapter builds a client for ({', '.join(SUPPORTED_PROVIDERS)}); bring "
        "a completion client of your own for it",
    )
