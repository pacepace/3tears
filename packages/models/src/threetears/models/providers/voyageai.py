"""VoyageAI embedding factory backed by ``langchain_voyageai``.

LangChain-native shape (3tears v0.6.0+): :func:`create_voyageai_embedding`
returns a configured ``VoyageAIEmbeddings`` instance, which already
inherits from :class:`langchain_core.embeddings.Embeddings`. This module
applies a Python 3.14 compatibility shim that the upstream voyageai SDK
needs (see :mod:`._voyageai_compat`) before it imports the upstream package.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.providers._voyageai_compat import apply_voyageai_compat


def _voyageai_installed() -> bool:
    """whether the voyageai extra's package can be imported, found without importing it.

    :return: True when ``langchain_voyageai`` is installed
    :rtype: bool
    """
    try:
        return find_spec("langchain_voyageai") is not None
    except ValueError, ImportError:
        # NOSILENT: a module set to None in sys.modules, or one a finder refuses, is one this process cannot import
        return False


# the shim goes in before the upstream package loads, or its multimodal models crash the import; only where the
# extra is installed, so a process without it gets no stub module in sys.modules
if _voyageai_installed():
    apply_voyageai_compat()

_VOYAGEAI_IMPORT_ERROR: ImportError | None = None
try:
    from langchain_voyageai import VoyageAIEmbeddings
except ImportError as _exc:  # NOSILENT: the voyageai extra is optional; the factory names it
    _VOYAGEAI_IMPORT_ERROR = _exc

__all__ = [
    "VOYAGEAI_PROVIDER_NAME",
    "create_voyageai_embedding",
]


VOYAGEAI_PROVIDER_NAME = "voyageai"


def create_voyageai_embedding(
    api_key: str,
    *,
    model_name: str = "voyage-3-lite",
    base_url: str | None = None,
    embedding_dimensions: int | None = None,
    **extra_kwargs: Any,
) -> VoyageAIEmbeddings:
    """creates a configured ``VoyageAIEmbeddings`` instance.

    the Python 3.14 voyageai compat shim is applied when this module loads,
    before the upstream package is imported, so the legacy Pydantic-v1
    multimodal models do not explode at import time.

    :param api_key: API key for VoyageAI authentication
    :ptype api_key: str
    :param model_name: VoyageAI embedding model identifier
    :ptype model_name: str
    :param base_url: optional custom API base URL
    :ptype base_url: str | None
    :param embedding_dimensions: optional output vector dimensionality
    :ptype embedding_dimensions: int | None
    :param extra_kwargs: additional keyword arguments forwarded to ``VoyageAIEmbeddings``
    :ptype extra_kwargs: Any
    :return: configured ``VoyageAIEmbeddings`` instance
    :rtype: VoyageAIEmbeddings
    :raises ImportError: when the ``voyageai`` extra is not installed
    """
    if _VOYAGEAI_IMPORT_ERROR is not None:
        raise ImportError(
            "VoyageAI embeddings need the voyageai extra: install 3tears-models[voyageai]",
        ) from _VOYAGEAI_IMPORT_ERROR
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
    }
    if base_url is not None:
        kwargs["base_url"] = base_url
    if embedding_dimensions is not None:
        kwargs["output_dimension"] = embedding_dimensions
    kwargs.update(extra_kwargs)

    model: VoyageAIEmbeddings = VoyageAIEmbeddings(**kwargs)
    return model


# -- capability registration -------------------------------------------------

# voyage-4 family is the current embedding lineup (verified live against the
# VoyageAI API 2026-06-13: both ids embed at 1024 dims). the voyage-3 family
# was retired -- the platform default moved to voyage-4. Pricing is NOT baked
# here: 3tears does not bundle vendor prices (they change) and VoyageAI's
# embedding models are not on OpenRouter, so consuming products admin-enter
# their cost. The cost_* fields stay at their None default.
_VOYAGEAI_CAPABILITIES: dict[str, ModelCapabilities] = {
    "voyage-4": ModelCapabilities(
        model_name="voyage-4",
        provider_name=VOYAGEAI_PROVIDER_NAME,
        model_type=ModelType.EMBEDDING,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        embedding_dimensions=1024,
        max_embedding_tokens=32_000,
        supports_batch_embedding=True,
    ),
    "voyage-4-large": ModelCapabilities(
        model_name="voyage-4-large",
        provider_name=VOYAGEAI_PROVIDER_NAME,
        model_type=ModelType.EMBEDDING,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        embedding_dimensions=1024,
        max_embedding_tokens=32_000,
        supports_batch_embedding=True,
    ),
}


for _model_id, _caps in _VOYAGEAI_CAPABILITIES.items():
    register_capabilities(_model_id, _caps)
