"""The toy host's PRODUCT side: the extraction request its own code sends, outside any eval.

Everything else in this package is the toy host's eval wiring. This module is the code the product
runs in production, and it exists so the toy host can show the one thing an eval of a real product
has to prove before any of its numbers mean anything: that the eval constructs the call production
constructs (the fidelity pattern, :mod:`threetears.evals.run.fidelity`).

**One constructor.** :func:`extraction_request` is the only function that turns *(model, invoice)*
into what the provider receives. :func:`extract_invoice` — production — calls it, and so does the
candidate kind (``kind.py``) — the eval. ``fidelity.py`` registers that as a contract, and the
canary in ``tests/test_fidelity_adoption.py`` checks both callers' source still reaches it.

It imports nothing from the eval side at run time, so it cannot depend on the thing measuring it:
``kind.py`` imports this module, never the reverse.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from packages.evals.tests.fixtures.toyhost.kind import ExtractionResult, ToyDocument


@dataclass(frozen=True)
class ExtractionRequest:
    """What one extraction call hands the provider."""

    #: The extractor model the call is bound to.
    model: str
    #: The prompt text — the bytes on the wire.
    prompt: str
    #: The invoice the prompt was built from. Carried because the toy provider is scripted per
    #: document; a real provider would read only ``prompt``.
    document: ToyDocument


class ExtractionClient(Protocol):
    """The provider an extraction request is sent to."""

    async def extract(self, request: ExtractionRequest) -> ExtractionResult:
        """Send one request.

        Args:
            request: The request, as :func:`extraction_request` built it.

        Returns:
            The extracted fields and the call's own usage.
        """
        ...


def extraction_request(*, model: str, document: ToyDocument) -> ExtractionRequest:
    """Build the one request an extraction sends — the fidelity contract's constructor.

    Args:
        model: The extractor model.
        document: The invoice to extract from.

    Returns:
        The request.
    """
    fields = ", ".join(document.key)
    page = "\n".join(f"{name}: {value}" for name, value in document.key.items())
    return ExtractionRequest(
        model=model,
        prompt=f"Extract {fields} from invoice {document.document_id}.\n\n{page}",
        document=document,
    )


async def extract_invoice(client: ExtractionClient, *, model: str, document: ToyDocument) -> ExtractionResult:
    """Extract one invoice, the way the product does it outside any eval.

    Args:
        client: The provider.
        model: The extractor model.
        document: The invoice.

    Returns:
        What the provider returned.
    """
    return await client.extract(extraction_request(model=model, document=document))


__all__ = ["ExtractionClient", "ExtractionRequest", "extract_invoice", "extraction_request"]
