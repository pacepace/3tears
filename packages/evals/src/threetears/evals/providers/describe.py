"""What the engine may know of a failed call through the client: :func:`describe_failure`, the host's
``failure_describer``.

Apart from :mod:`threetears.evals.providers.failures` because it builds the engine's
:class:`~threetears.evals.schema.ProviderFailure` and reads ``httpx``'s and pydantic's errors, which a host that
imports :class:`~threetears.evals.providers.ProviderError` only to catch it never needs.
"""

from __future__ import annotations

import httpx
import pydantic

from threetears.evals.providers.failures import ProviderError
from threetears.evals.schema import ProviderFailure

__all__ = ["describe_failure"]


def describe_failure(exc: BaseException) -> ProviderFailure:
    """What the engine may know of a call through the client that raised: the host's ``failure_describer``.

    Total over ``BaseException``. A :class:`ProviderError` is its own text (the provider, the status, the
    message; never the body) with its ``owner``. An ``httpx`` timeout or transport error names the host it
    was talking to and the error's type. A reply :meth:`~threetears.evals.providers.Model.json` could not validate
    names the schema and the first error, leaving out the reply itself. Anything else is withheld, since its text is
    unknown, as :func:`~threetears.evals.kernel.withhold_failure_detail` does. Every failure but a
    :class:`ProviderError` is the ``candidate``'s: nothing in it says the account or the provider declined the call.

    :param exc: Whatever the client raised, or anything else the engine caught around the call.
    :return: The failure as the engine may record it.
    """
    if isinstance(exc, ProviderError):
        return ProviderFailure(str(exc), payload_withheld=False, owner=exc.owner)
    if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
        host = exc.request.url.host if _has_request(exc) else "the provider"
        what = "timed out" if isinstance(exc, httpx.TimeoutException) else "could not be sent"
        return ProviderFailure(
            f"the request to {host} {what} ({type(exc).__name__})", payload_withheld=False, owner="candidate"
        )
    if isinstance(exc, TimeoutError):
        return ProviderFailure(f"the call timed out ({type(exc).__name__})", payload_withheld=False, owner="candidate")
    if isinstance(exc, pydantic.ValidationError):
        first = exc.errors()[0] if exc.error_count() else None
        where = ".".join(str(part) for part in first["loc"]) if first else ""
        said = f"; first: {where or '/'}: {first['msg']}" if first else ""
        return ProviderFailure(
            f"the reply did not validate as {exc.title} ({exc.error_count()} errors{said}; the reply is withheld)",
            payload_withheld=True,
            owner="candidate",
        )
    return ProviderFailure(
        f"{type(exc).__name__} (its text is withheld: the client does not know what it carries)",
        payload_withheld=True,
        owner="candidate",
    )


def _has_request(exc: httpx.TransportError | httpx.TimeoutException) -> bool:
    """Whether ``exc`` carries the request it failed on: ``httpx`` raises ``RuntimeError`` reading one it lacks."""
    try:
        _ = exc.request  # the property raises when the error was built without a request
    except RuntimeError:
        # NOSILENT: an error built without its request names no host; the description says "the provider" instead.
        return False
    return True
