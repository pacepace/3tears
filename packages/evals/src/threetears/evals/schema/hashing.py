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
    :return: The canonical JSON encoding.
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
    :return: A 64-character lowercase hex digest.
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
    does not, except, rarely, a value that falls exactly on a 12th-figure rounding boundary, where the two builds' last bits round to
    neighbouring twelfth figures. A typed float (``0.7``, ``2.5``) has fewer figures than this and comes back
    unchanged.

    ``-0.0`` becomes ``0.0``, since the two are equal and a sign on nothing is not a difference. NaN and the
    infinities come back as they are: each has one value, which JSON writes as one token (``NaN``, ``Infinity``,
    ``-Infinity``), so they are already deterministic.

    :param value: The float.
    :return: The float to twelve significant figures.
    """
    rounded = float(f"{value:.{IDENTITY_FIGURES}g}")
    return 0.0 if rounded == 0 else rounded


def round_floats(payload: Any) -> Any:
    """``payload`` with every float, at any depth of dicts, lists and tuples, as :func:`round_figures` gives it.

    The one pass a JSON-able value takes to be the same across the Python builds that computed it: before it is hashed
    (:func:`identity_json`), and where a document code computed is served (a campaign page's JSON, the writer's
    message). A typed float, a bool, an int and a string come back as they were; a tuple comes back a list, as JSON
    writes it.

    :param payload: Any JSON-able structure.
    :return: A copy with every float rounded; nothing is changed in place.
    """
    if isinstance(payload, float):
        return round_figures(payload)
    if isinstance(payload, dict):
        return {key: round_floats(value) for key, value in payload.items()}
    if isinstance(payload, list | tuple):
        return [round_floats(value) for value in payload]
    return payload


def identity_json(payload: Any) -> str:
    """Render ``payload`` as canonical JSON for comparing it by identity: every float rounded to twelve figures.

    :func:`canonical_json` with one change: every float, at any depth, is first :func:`round_figures`'d, so a
    document holding computed statistics renders the same across the Python builds that computed them. Nothing else
    differs: sorted keys, tight separators, a bool stays a bool and an int an int, and a value JSON cannot encode
    raises as it does there. **For what is hashed or compared, never for what is stored or shown**: it is the
    pre-image of an identity, and the document itself keeps what it holds.

    :param payload: Any JSON-encodable structure.
    :return: The canonical JSON encoding, floats rounded.
    :raises UnhashableContentError: ``payload`` holds something JSON cannot encode.
    """
    return canonical_json(round_floats(payload))


def identity_digest(payload: Any) -> str:
    """Return the full sha256 hex digest of ``payload``'s :func:`identity_json`.

    :param payload: Any JSON-encodable structure.
    :return: A 64-character lowercase hex digest.
    :raises UnhashableContentError: ``payload`` holds something JSON cannot encode.
    """
    return hashlib.sha256(identity_json(payload).encode("utf-8")).hexdigest()


def _canonical_input(payload: Any) -> Any:
    """``payload`` as :func:`input_json` writes it: every whole float as the integer it equals."""
    if isinstance(payload, float):
        # A whole number is one value however a store types it; -0.0 is 0. NaN and the infinities are not whole.
        return int(payload) if payload.is_integer() else payload
    if isinstance(payload, dict):
        return {key: _canonical_input(value) for key, value in payload.items()}
    if isinstance(payload, list | tuple):
        return [_canonical_input(value) for value in payload]
    return payload


def input_json(payload: Any) -> str:
    """Render stored inputs as canonical JSON: the pre-image of an identity over evidence, never over a statistic.

    For values a store holds and hands back (a result's recorded scores, a run's configuration, a declaration), which
    are exact: parsing JSON to a double is correctly rounded on every platform, so nothing here is rounded and no
    maths library is read. Two documents holding the same inputs render the same bytes whatever store returned them:

    - **Key order is not read.** Every mapping's keys are sorted, at every depth, so a store that reorders keys (a
      ``jsonb`` column does) moves nothing. A list keeps its order, as every JSON store keeps it.
    - **A null is a value.** ``{"a": null}`` and ``{}`` render differently: in a map a host writes, whether a key
      was written can carry meaning. A caller for whom an unset field and an absent one are one value (a typed record, where both
      hydrate to ``None``) leaves the unset field out before it renders.
    - **Numbers.** A bool is ``true`` or ``false``, never a number. An integer is written in decimal, at any size.
      A float holding a whole number is written as the integer it equals, at any magnitude: ``1.0`` is ``1``,
      ``-0.0`` is ``0``, ``1e22`` is ``10000000000000000000000``. That is how a ``jsonb`` column hands a whole number
      back whatever it was written as, so a value written as a float and read back as an integer renders alike. Any
      other finite float is written as Python's ``repr``: the shortest decimal string that parses back to the same
      IEEE-754 double (``sys.float_repr_style`` is ``short``), which Python computes itself and not through the
      platform's C library, so it is the same string on every platform.
    - **NaN and the infinities** are written as the fixed tokens ``NaN``, ``Infinity`` and ``-Infinity``. No JSON
      store holds one, so they reach here only from a store that keeps Python values; each is one value with one
      token, so it is rendered and not refused, and an assembly over such a store still fingerprints.
    - **Strings** are written with every non-ASCII character escaped (``\\uXXXX``), so the bytes do not depend on an
      encoding. They are **not** Unicode-normalised: two spellings of one visible text (composed and decomposed) are
      two values, as they are two values in the store.
    - **Anything JSON cannot encode is refused** (a ``datetime``, a ``Decimal``, a set, an object):
      :class:`UnhashableContentError`, as :func:`canonical_json` refuses it.

    A float **computed** here would carry its platform's last bits into the key, which is what
    :func:`identity_json` rounds away; this function is for values nothing computed.

    :param payload: Any JSON-encodable structure of stored values.
    :return: The canonical JSON encoding.
    :raises UnhashableContentError: ``payload`` holds something JSON cannot encode.
    """
    return canonical_json(_canonical_input(payload))


def input_digest(payload: Any) -> str:
    """Return the full sha256 hex digest of ``payload``'s :func:`input_json`.

    :param payload: Any JSON-encodable structure of stored values.
    :return: A 64-character lowercase hex digest.
    :raises UnhashableContentError: ``payload`` holds something JSON cannot encode.
    """
    return hashlib.sha256(input_json(payload).encode("utf-8")).hexdigest()


def bytes_digest(payload: bytes) -> str:
    """Return the full sha256 hex digest of raw bytes, with no JSON encoding step.

    The companion to :func:`canonical_digest` for a value that arrives as bytes rather than as
    a structure — an opaque blob a host supplies, where there is no structure to canonicalise
    and running it through JSON would address the *encoding* of the blob rather than the blob.

    :param payload: The raw bytes.
    :return: A 64-character lowercase hex digest.
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
    "input_digest",
    "input_json",
    "round_figures",
    "round_floats",
]
