"""compatibility shim for voyageai SDK on Python 3.14+.

the voyageai SDK (v0.3.7) uses Pydantic v1 field constraints (min_items)
in its multimodal_embeddings module, which are incompatible with Python
3.14 where Pydantic v1 compat layer rejects unenforced constraints.

this module stubs out the broken multimodal_embeddings module before
voyageai loads. we do not use multimodal embeddings — only regular text
embeddings via VoyageAIEmbeddings.aembed_documents().

TODO: remove this module when voyageai merges Pydantic v2 migration
      (PR #50: https://github.com/voyage-ai/voyageai-python/pull/50)
      and releases a fixed version. the stub will become a harmless
      no-op (voyageai will load its own module, overwriting the stub
      in sys.modules) but should be cleaned up for clarity.

tested against: voyageai==0.3.7, langchain-voyageai==0.3.3
"""

from __future__ import annotations

import sys
import types

__all__ = [
    "apply_voyageai_compat",
]


def _needs_patch() -> bool:
    """checks whether the voyageai multimodal compat patch is needed.

    only patches on Python 3.14+ where the Pydantic v1 compat layer
    rejects min_items constraints. skips if voyageai is already loaded
    (someone else solved it) or if the patch was already applied.

    :return: True if patch should be applied
    :rtype: bool
    """
    if sys.version_info < (3, 14):
        return False

    if "voyageai" in sys.modules:
        return False

    if "voyageai.object.multimodal_embeddings" in sys.modules:
        return False

    return True


def apply_voyageai_compat() -> None:
    """stubs voyageai.object.multimodal_embeddings to bypass Pydantic v1 crash.

    injects a stub module into sys.modules providing all class names that
    voyageai._base imports from multimodal_embeddings. the stub classes
    are inert placeholders — multimodal embedding functionality is not
    available, but regular text embeddings work correctly.

    safe to call multiple times — second call is a no-op.
    safe to call on Python < 3.14 — returns immediately.
    """
    if not _needs_patch():
        return

    stub = types.ModuleType("voyageai.object.multimodal_embeddings")
    stub.__doc__ = "stub module replacing broken Pydantic v1 multimodal models"

    # placeholder class for all multimodal types imported by voyageai._base
    class _MultimodalStub:
        """placeholder for voyageai multimodal class unavailable on Python 3.14."""

    _names = (
        "MultimodalEmbeddingsObject",
        "MultimodalInputRequest",
        "MultimodalInput",
        "MultimodalInputSegmentText",
        "MultimodalInputSegmentImageURL",
        "MultimodalInputSegmentImageBase64",
        "MultimodalInputSegmentVideoURL",
        "MultimodalInputSegmentVideoBase64",
    )
    for name in _names:
        setattr(stub, name, _MultimodalStub)

    sys.modules["voyageai.object.multimodal_embeddings"] = stub
