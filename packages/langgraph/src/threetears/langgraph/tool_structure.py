"""the structured-tool-result channel for the streaming faces.

a tool that produces structure produces it once, in-process, on
``ToolMessage.artifact`` -- the typed projection a
``response_format="content_and_artifact"`` tool returns beside its prose.
that structure reaches the model's context and the MCP face, and until this
module it died before the client: the two streaming faces carry a tool's
completion as name + status + timing and nothing else, so a frontend wanting
to draw citation cards had to re-parse the prose the model read.

design: ``docs/stream-protocol-structured-results.md`` (D-S1 (a), D-S3 (c),
D-S4, D-S5). what this module owns is the *decision* -- given an artifact and
a size bound, what does the wire carry -- and the vocabulary that decision is
spelled in. it owns no schema of its own: an inline payload is the artifact
**verbatim**, which is what keeps the streaming faces from becoming a second
result shape (success check 14, and D-S4's one hard "must not": a narrowed
payload MUST NOT wear the full projection's key).

three kinds, one discriminator:

- :data:`STRUCTURED_KIND_INLINE` -- the artifact rides the event whole. the
  common case, and the only case that needs nothing from the host.
- :data:`STRUCTURED_KIND_HANDLE` -- the artifact is over the bound and the
  host stored it out-of-band; the event carries a small reference the client
  resolves. **no producer in this package** (see below).
- :data:`STRUCTURED_KIND_OMITTED` -- the artifact is over the bound (or does
  not survive JSON) and no host store was available. the event says so, in
  the payload, under a key nothing will mistake for a projection. a truncated
  answer that does not admit it is the silent-partial-answer defect; an
  omission that says its own size and reason is a client's cue to fetch the
  result some other way.

**why the handle kind exists with nothing producing it here.** the stream is
not a store (D7 / D12 / D14): this package must not persist a byte to make a
frame smaller. the host holds the store -- and the family already has the
port for exactly this shape,
:class:`threetears.langgraph.offload.ToolResultOffloader`, injected on
``config["configurable"]`` and already moving oversized tool *content*
out-of-band for the model's context window. wiring the same port to this
seam is a live decision, not a missing implementation; the discriminator
ships now so a host that turns it on later does not move the wire.

**``structured_kind`` is a ``str``, deliberately, not a ``Literal``.** a
closed vocabulary on a wire model means a reader predating a fourth kind
*rejects* the event rather than ignoring the value -- the 2026-08-13 lesson
(an additive optional on an ``extra="forbid"`` envelope refusing every call
from a lagging pod for three days) one level down, in the value rather than
the field. the constants below are the vocabulary; validation is the
reader's business, and an unknown kind is a thing to skip, not to fail on.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "DEFAULT_STRUCTURED_INLINE_MAX_CHARS",
    "OMISSION_KEY",
    "OMISSION_REASON_OVER_BOUND",
    "OMISSION_REASON_UNSERIALIZABLE",
    "STRUCTURED_KIND_HANDLE",
    "STRUCTURED_KIND_INLINE",
    "STRUCTURED_KIND_OMITTED",
    "NO_STREAM_STRUCTURE",
    "StreamStructure",
    "StructuredToolResultFields",
    "structure_for_stream",
]

#: the artifact rides the event whole -- ``structured`` IS the artifact.
STRUCTURED_KIND_INLINE = "inline"

#: ``structured`` is a small host-minted reference to the artifact, stored
#: out-of-band. nothing in this package produces it (module docstring).
STRUCTURED_KIND_HANDLE = "handle"

#: the artifact could not ride and was not stored; ``structured`` carries the
#: reason under :data:`OMISSION_KEY` and nothing else.
STRUCTURED_KIND_OMITTED = "omitted"

#: the one key an omission payload is written under. deliberately NOT the
#: projection's key (D-S4): a reader that finds this dict must not be able to
#: hand it to a projection parser and get something that parses while
#: under-reporting.
OMISSION_KEY = "omitted"

#: the artifact serialized to more than the caller's bound.
OMISSION_REASON_OVER_BOUND = "over_inline_bound"

#: the artifact holds something ``json.dumps`` refuses. carrying a partial
#: encode would be inventing a second shape; saying so is the whole answer.
OMISSION_REASON_UNSERIALIZABLE = "not_json_serializable"

#: default inline ceiling, in characters of the artifact's JSON encoding.
#: twice :data:`threetears.langgraph.offload.DEFAULT_OFFLOAD_THRESHOLD_CHARS`,
#: on the reasoning that a websocket frame tolerates more than a context
#: window does, and measured in characters for the same reason that seam is:
#: it is the unit the encode already produces. **this number is a
#: placeholder with an owner** -- open question 1 of the design is metallm's
#: to answer with its real per-frame budget, and every construction site
#: reads it from here or from its own caller so answering it moves one line.
#:
#: **The answer has a ceiling, and it is lower than it looks.** Above this
#: budget sits a limit nobody chose: a room frame crosses NATS, and an
#: oversized publish is refused client-side against the broker's advertised
#: ``max_payload`` -- 1 MB on an untuned broker, which works back through the
#: event -> ``Frame`` -> ``RoomFrame`` nesting to roughly 780,000 characters of
#: artifact. The nesting cost is not a constant (measured at 1.20x on a
#: body-heavy payload and 1.34x on a metadata-heavy one), so anyone converting
#: a frame budget into a bound here re-measures the shape they actually send
#: with ``scripts/measure-structured-result-sizes.py`` rather than applying
#: either ratio. What the two figures establish is only that the multiplier is
#: neither 1.0 nor stable. Past the ceiling the answer stops being a bigger
#: number and becomes a handle; see ``docs/structured-result-tiers.md``.
DEFAULT_STRUCTURED_INLINE_MAX_CHARS = 16384


class StructuredToolResultFields(BaseModel):
    """the two fields every streaming face carries a tool's structure in.

    a mixin rather than two hand-copied field pairs, because the two faces
    are the same contract rendered twice and check 14's rule is that a second
    result shape must not be *possible*, not merely absent today. inheriting
    here means a field added to the channel appears on both faces in the same
    commit, and ``tests/enforcement`` can hold that to be true.

    :ivar structured: the tool's structure as the wire carries it -- the
        artifact verbatim when inline, a host reference when a handle, an
        omission record when neither. ``None`` for the overwhelming majority
        of tools, which produce no structure at all and pay one JSON null.
    :ivar structured_kind: which of the three the payload is
        (:data:`STRUCTURED_KIND_INLINE` / :data:`STRUCTURED_KIND_HANDLE` /
        :data:`STRUCTURED_KIND_OMITTED`), or ``None`` alongside a ``None``
        payload. a plain ``str``: see the module docstring for why this is
        not a ``Literal``.
    """

    structured: dict[str, Any] | None = Field(default=None)
    structured_kind: str | None = Field(default=None)


@dataclass(frozen=True)
class StreamStructure:
    """the decision :func:`structure_for_stream` reached, ready to splat.

    :ivar structured: the payload for
        :attr:`StructuredToolResultFields.structured`.
    :ivar structured_kind: the discriminator for
        :attr:`StructuredToolResultFields.structured_kind`.
    """

    structured: dict[str, Any] | None
    structured_kind: str | None

    def as_fields(self) -> dict[str, Any]:
        """render the pair as kwargs for either streaming face.

        :return: ``{"structured": ..., "structured_kind": ...}``
        :rtype: dict[str, Any]
        """
        return {"structured": self.structured, "structured_kind": self.structured_kind}


#: the answer for every tool that produces no structure: both fields ``None``,
#: no allocation, no branch at the call site.
NO_STREAM_STRUCTURE = StreamStructure(structured=None, structured_kind=None)


def structure_for_stream(
    artifact: Any,
    *,
    max_chars: int = DEFAULT_STRUCTURED_INLINE_MAX_CHARS,
) -> StreamStructure:
    """decide what a streaming face carries for one tool result.

    the artifact is forwarded, never rebuilt: an inline payload is the
    artifact's own JSON encoding decoded back, so the streaming faces render
    the same contract the in-process and MCP faces do and no new construction
    site is born (success check 14 / D-S4). the round trip is not a
    re-shaping -- it is the encode this function already performs to measure
    the bound, kept instead of thrown away, which buys two things a
    ``dict(artifact)`` shallow copy does not: the payload is exactly the bytes
    the size was measured against, and no nested container stays aliased to
    the caller's artifact, so a tool mutating its own result after the
    decision cannot rewrite what the wire already accounted for.

    a non-mapping artifact -- which is every content-format tool, i.e. most
    of them -- is not structure and is not reported as an omission; it is
    simply absent, and costs one ``None``.

    :param artifact: the tool's ``ToolMessage.artifact``, or ``None``.
    :ptype artifact: Any
    :param max_chars: inline ceiling, in characters of the JSON encoding.
        a non-positive bound disables the inline path entirely (everything
        becomes an omission), which is the honest reading of "nothing fits".
    :ptype max_chars: int
    :return: the payload + discriminator pair.
    :rtype: StreamStructure
    """
    if not isinstance(artifact, Mapping) or not artifact:
        return NO_STREAM_STRUCTURE

    try:
        encoded = json.dumps(dict(artifact), ensure_ascii=False)
    except TypeError, ValueError:
        return _omission(reason=OMISSION_REASON_UNSERIALIZABLE, size_chars=None, limit_chars=max_chars)

    if len(encoded) > max_chars:
        return _omission(reason=OMISSION_REASON_OVER_BOUND, size_chars=len(encoded), limit_chars=max_chars)

    payload: dict[str, Any] = json.loads(encoded)
    return StreamStructure(structured=payload, structured_kind=STRUCTURED_KIND_INLINE)


def _omission(*, reason: str, size_chars: int | None, limit_chars: int) -> StreamStructure:
    """build the payload that says structure exists and did not ride.

    :param reason: one of the ``OMISSION_REASON_*`` constants.
    :ptype reason: str
    :param size_chars: the encoded size, when it is known (it is not, when
        the encode itself failed).
    :ptype size_chars: int | None
    :param limit_chars: the bound the size was measured against.
    :ptype limit_chars: int
    :return: an omission-kinded structure.
    :rtype: StreamStructure
    """
    record: dict[str, Any] = {"reason": reason, "limit_chars": limit_chars}
    if size_chars is not None:
        record["size_chars"] = size_chars
    return StreamStructure(
        structured={OMISSION_KEY: record},
        structured_kind=STRUCTURED_KIND_OMITTED,
    )
