"""The content-addressing primitive every eval key is built on.

A leaf on purpose: it imports nothing from :mod:`threetears.evals` and nothing from
any host package, so both the identity predicates and the host contract can reach it without
either importing the other. That mattered the moment a host-supplied value became
content-addressed — :class:`~threetears.evals.schema.values.SweepableValue` hashes what a host
handed over, and :mod:`threetears.evals.kernel.identity` hashes what the engine composed from those
values, so the two layers must agree byte-for-byte on what "the same content" means. One
implementation is how they agree; two would be a drift nobody could see until two runs that
swept the same thing stopped sharing a key.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final


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

    :param payload: Any JSON-encodable structure.
    :ptype payload: Any
    :return: The canonical JSON encoding.
    :rtype: str
    :raises UnhashableContentError: ``payload`` holds something JSON cannot encode.
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

    :param payload: Any JSON-encodable structure.
    :ptype payload: Any
    :return: A 64-character lowercase hex digest.
    :rtype: str
    """
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


#: The significant figures :func:`round_figures` keeps. Two builds of one Python that compute a statistic through
#: their own C maths (``math.lgamma``, ``math.exp``, ``NormalDist().inv_cdf``) can disagree in its last bit or two,
#: which is past the 15th figure; twelve leaves three figures of headroom above that and below nothing a reader acts on.
IDENTITY_FIGURES: Final[int] = 12


def round_figures(value: float) -> float:
    """``value`` to :data:`IDENTITY_FIGURES` significant figures: the form a computed float is compared or served in.

    A float a computation produced carries bits past the 15th figure that depend on the Python build that computed
    it, so the same evidence compared, hashed or served in full would differ between two machines. Rounded here, it
    does not. A typed float (``0.7``, ``2.5``) has fewer figures than this and comes back unchanged.

    ``-0.0`` becomes ``0.0``, since the two are equal and a sign on nothing is not a difference. NaN and the
    infinities come back as they are: each has one value, which JSON writes as one token (``NaN``, ``Infinity``,
    ``-Infinity``), so they are already deterministic.

    :param value: The float.
    :ptype value: float
    :return: The float to twelve significant figures.
    :rtype: float
    """
    rounded = float(f"{value:.{IDENTITY_FIGURES}g}")
    return 0.0 if rounded == 0 else rounded


def _figures_rounded(payload: Any) -> Any:
    """``payload`` with every float, at any depth of dicts, lists and tuples, as :func:`round_figures` gives it."""
    if isinstance(payload, float):
        return round_figures(payload)
    if isinstance(payload, dict):
        return {key: _figures_rounded(value) for key, value in payload.items()}
    if isinstance(payload, list | tuple):
        return [_figures_rounded(value) for value in payload]
    return payload


def identity_json(payload: Any) -> str:
    """Render ``payload`` as canonical JSON for comparing it by identity: every float rounded to twelve figures.

    :func:`canonical_json` with one change: every float, at any depth, is first :func:`round_figures`'d, so a
    document holding computed statistics renders the same on every Python build that computed them. Nothing else
    differs: sorted keys, tight separators, a bool stays a bool and an int an int, and a value JSON cannot encode
    raises as it does there. **For what is hashed or compared, never for what is stored or shown**: it is the
    pre-image of an identity, and the document itself keeps what it holds.

    :param payload: Any JSON-encodable structure.
    :ptype payload: Any
    :return: The canonical JSON encoding, floats rounded.
    :rtype: str
    :raises UnhashableContentError: ``payload`` holds something JSON cannot encode.
    """
    return canonical_json(_figures_rounded(payload))


def identity_digest(payload: Any) -> str:
    """Return the full sha256 hex digest of ``payload``'s :func:`identity_json`.

    :param payload: Any JSON-encodable structure.
    :ptype payload: Any
    :return: A 64-character lowercase hex digest.
    :rtype: str
    :raises UnhashableContentError: ``payload`` holds something JSON cannot encode.
    """
    return hashlib.sha256(identity_json(payload).encode("utf-8")).hexdigest()


def bytes_digest(payload: bytes) -> str:
    """Return the full sha256 hex digest of raw bytes, with no JSON encoding step.

    The companion to :func:`canonical_digest` for a value that arrives as bytes rather than as
    a structure — an opaque blob a host supplies, where there is no structure to canonicalise
    and running it through JSON would address the *encoding* of the blob rather than the blob.

    :param payload: The raw bytes.
    :ptype payload: bytes
    :return: A 64-character lowercase hex digest.
    :rtype: str
    """
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "IDENTITY_FIGURES",
    "UnhashableContentError",
    "bytes_digest",
    "canonical_digest",
    "canonical_json",
    "identity_digest",
    "identity_json",
    "round_figures",
]
