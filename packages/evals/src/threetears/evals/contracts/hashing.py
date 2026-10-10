"""The content-addressing primitive every eval key is built on.

A leaf on purpose: it imports nothing from :mod:`threetears.evals` and nothing from
any host package, so both the identity predicates and the host contract can reach it without
either importing the other. That mattered the moment a host-supplied value became
content-addressed — :class:`~threetears.evals.contracts.host.values.SweepableValue` hashes what a host
handed over, and :mod:`threetears.evals.contracts.identity` hashes what the engine composed from those
values, so the two layers must agree byte-for-byte on what "the same content" means. One
implementation is how they agree; two would be a drift nobody could see until two runs that
swept the same thing stopped sharing a key.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


class UnhashableContentError(TypeError):
    """A value reaching an identity key cannot be JSON-encoded, so it cannot be content-addressed."""


def canonical_json(payload: Any) -> str:
    """Render ``payload`` as canonical JSON — the stable pre-image for every key here.

    Sorted keys make dict insertion order irrelevant and tight separators strip
    incidental whitespace, so two structurally equal payloads always produce
    identical bytes.

    **A value JSON cannot encode raises, and used not to.** A ``default=str`` fallback stood here
    on the reasoning that a key computation must not be what crashes a run launch. It failed
    silently in both directions instead: a value with a stable ``str()`` — a ``datetime`` reaching
    a subject's ``goals`` or ``tool_configs``, both of which are hashed verbatim — was addressed
    by its RENDERING rather than its content, and an object with the default ``__repr__`` embedded
    its address and minted a new key every process. The second is a wrong split on a durable
    grouping key, invisible, with nothing logging or counting it. Every caller here passes
    ``model_dump(mode="json")`` output or a plain structure, so an unencodable value is a defect in
    that caller; raising sends it to the caller, which is where a fix belongs rather than a fallback.

    Args:
        payload: Any JSON-encodable structure.

    Returns:
        The canonical JSON encoding.

    Raises:
        UnhashableContentError: ``payload`` holds something JSON cannot encode.
    """
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except TypeError as e:
        raise UnhashableContentError(
            f"a value reaching an eval identity key is not JSON-encodable: {e}. It cannot be content-addressed, and "
            "rendering it with str() would address either its formatting or its memory address — supply a JSON-safe "
            "value at the caller instead."
        ) from e


def canonical_digest(payload: Any) -> str:
    """Return the full sha256 hex digest of ``payload``'s canonical JSON.

    Full length rather than a truncated prefix: these are durable cross-run
    grouping keys living in a JSONB column where the extra characters cost
    nothing, and collision headroom matters more than display brevity.

    Args:
        payload: Any JSON-encodable structure.

    Returns:
        A 64-character lowercase hex digest.
    """
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def bytes_digest(payload: bytes) -> str:
    """Return the full sha256 hex digest of raw bytes, with no JSON encoding step.

    The companion to :func:`canonical_digest` for a value that arrives as bytes rather than as
    a structure — an opaque blob a host supplies, where there is no structure to canonicalise
    and running it through JSON would address the *encoding* of the blob rather than the blob.

    Args:
        payload: The raw bytes.

    Returns:
        A 64-character lowercase hex digest.
    """
    return hashlib.sha256(payload).hexdigest()


__all__ = ["UnhashableContentError", "bytes_digest", "canonical_digest", "canonical_json"]
