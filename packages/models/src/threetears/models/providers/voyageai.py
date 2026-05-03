"""VoyageAI embedding factory backed by ``langchain_voyageai``.

LangChain-native shape (3tears v0.6.0+): :func:`create_voyageai_embedding`
returns a configured ``VoyageAIEmbeddings`` instance, which already
inherits from :class:`langchain_core.embeddings.Embeddings`. The factory
applies a Python 3.14 compatibility shim that the upstream voyageai SDK
needs (see :mod:`._voyageai_compat`).
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from threetears.models.capabilities import ModelCapabilities, register_capabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType

if TYPE_CHECKING:
    from langchain_voyageai import VoyageAIEmbeddings

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
    **extra_kwargs: object,
) -> VoyageAIEmbeddings:
    """creates a configured ``VoyageAIEmbeddings`` instance.

    applies the Python 3.14 voyageai compat shim before importing the
    upstream package so the legacy Pydantic-v1 multimodal models do not
    explode at import time.

    :param api_key: API key for VoyageAI authentication
    :ptype api_key: str
    :param model_name: VoyageAI embedding model identifier
    :ptype model_name: str
    :param base_url: optional custom API base URL
    :ptype base_url: str | None
    :param embedding_dimensions: optional output vector dimensionality
    :ptype embedding_dimensions: int | None
    :param extra_kwargs: additional keyword arguments forwarded to ``VoyageAIEmbeddings``
    :ptype extra_kwargs: object
    :return: configured ``VoyageAIEmbeddings`` instance
    :rtype: VoyageAIEmbeddings
    """
    from threetears.models.providers._voyageai_compat import apply_voyageai_compat

    apply_voyageai_compat()

    from langchain_voyageai import VoyageAIEmbeddings

    kwargs: dict[str, object] = {
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

_VOYAGEAI_CAPABILITIES: dict[str, ModelCapabilities] = {
    "voyage-3": ModelCapabilities(
        model_name="voyage-3",
        provider_name=VOYAGEAI_PROVIDER_NAME,
        model_type=ModelType.EMBEDDING,
        model_tier=ModelTier.LARGE,
        model_status=ModelStatus.ACTIVE,
        embedding_dimensions=1024,
        max_embedding_tokens=32_000,
        supports_batch_embedding=True,
        cost_per_input_token=Decimal("0.00000006"),
    ),
    "voyage-3-lite": ModelCapabilities(
        model_name="voyage-3-lite",
        provider_name=VOYAGEAI_PROVIDER_NAME,
        model_type=ModelType.EMBEDDING,
        model_tier=ModelTier.SMALL,
        model_status=ModelStatus.ACTIVE,
        embedding_dimensions=512,
        max_embedding_tokens=32_000,
        supports_batch_embedding=True,
        cost_per_input_token=Decimal("0.00000002"),
    ),
}


for _model_id, _caps in _VOYAGEAI_CAPABILITIES.items():
    register_capabilities(_model_id, _caps)
