#!/usr/bin/env python
"""Measure what a search projection costs on each wire that carries it.

Written for ``docs/structured-result-tiers.md`` §1 and §4, whose argument
rests on sizes. Prose sizes rot and cannot be re-checked; this can be re-run.

Three numbers per shape, because three different layers bound three different
things and only the first is the one anybody has been quoting:

``artifact``
    ``len(json.dumps(artifact))`` -- the projection's own JSON encoding. This
    is the unit ``structure_for_stream`` measures its inline bound in.
``scriob``
    the bytes NATS actually publishes on a shared chat room: the event JSON
    nested in a :class:`~threetears.channels.frames.Frame`, nested again in a
    :class:`~threetears.channels.presence.wire.RoomFrame`. Two levels of JSON
    string-escaping sit between the projection and the broker.
``metallm``
    the bytes NATS publishes on its cross-worker WS fanout: the ws message
    JSON nested in metallm's ``_WsFanoutEnvelope``. One level of escaping.
    Modelled here rather than imported -- the envelope is metallm-side -- so
    it is the shape as of metallm#287, not a live read.

The escaping is why this script exists rather than a multiplication: a
projection is mostly quoted keys and quoted strings, so each nesting inflates
it by a factor no one guessed correctly in advance, and the inflation differs
between a metadata-heavy payload and a body-heavy one.

Run: ``uv run python scripts/measure-structured-result-sizes.py``
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from threetears.channels.frames import Frame
from threetears.channels.presence.wire import RoomFrame
from threetears.search.bind import project_metadata
from threetears.search.contracts.candidate import Candidate, CandidateSet, ContentSlot, Locator
from threetears.search.contracts.provenance import Provenance
from threetears.search.contracts.scores import ScoreEntry

#: NATS refuses a larger publish client-side, before anything leaves the
#: process (``nats/aio/client.py`` checks the server-advertised value and
#: raises ``MaxPayloadError``). 1 MB is the default nobody has tuned -- the
#: same figure ``threetears.nats.pipe`` sizes its own chunks against.
NATS_DEFAULT_MAX_PAYLOAD = 1_048_576

RETRIEVED = datetime(2026, 8, 18, 12, 0, 0, tzinfo=UTC)

#: A provider snippet at a realistic length. Per-candidate cost is dominated
#: by this, which is why the script reports a sensitivity curve rather than
#: one number: "chars per candidate" is a function, not a constant.
SNIPPET = (
    "Counts rose 12% across the surveyed watersheds, with the steepest gains in the "
    "lower basin where three reintroduction sites have now been active for a full decade."
)

#: Extracted page text carries quotes, newlines and non-ASCII punctuation --
#: all of which JSON escapes, and all of which filler like ``"x" * n`` does
#: not. Using filler understates the body-heavy rows by about a fifth.
_PARA = (
    'The survey team reported that "counts rose 12% across the lower basin", a figure the '
    "regional office called “consistent with the decade trend” — though the "
    "underlying transect data has not yet been published.\n\n"
)


def page_text(chars: int) -> str:
    """Build extracted page text of roughly ``chars`` characters.

    :param chars: how much text to produce
    :ptype chars: int
    :return: text with the escaping burden real extracted content carries
    :rtype: str
    """
    return (_PARA * (chars // len(_PARA) + 1))[:chars]


def candidate(index: int, *, body_chars: int = 0, snippet_chars: int = len(SNIPPET)) -> Candidate:
    """Build one fully-populated candidate, as an adapter would emit it.

    Every field the Tavily adapter fills is filled: locators, provenance with
    provider ids, title, snippet, publication date, a provenanced score,
    both fidelity marks, and the facets an image or extraction result carries.
    A sparser candidate measures smaller; this is the honest upper end of an
    ordinary web result.

    :param index: distinguishes the candidate's identity and provider id
    :ptype index: int
    :param body_chars: extracted page text to attach, or 0 for none
    :ptype body_chars: int
    :param snippet_chars: how long the provider snippet is
    :ptype snippet_chars: int
    :return: the candidate
    :rtype: Candidate
    """
    url = f"https://example.test/survey/2026/otter-population-report-{index}"
    return Candidate(
        identity=url,
        locators=(Locator(url=url, rel="canonical"),),
        provenance=Provenance(
            query="otter population survey 2026",
            provider_instance="tavily",
            provider_ids={"result_id": f"tv-{index:04d}"},
            retrieved_at=RETRIEVED,
        ),
        title=f"2026 otter population survey -- regional report {index}",
        snippet=(SNIPPET * (snippet_chars // len(SNIPPET) + 1))[:snippet_chars],
        published_at=RETRIEVED,
        scores=(
            ScoreEntry.provider_native(name="relevance", value=0.82, scale="unit-interval", provider_instance="tavily"),
        ),
        fidelity_available="content",
        fidelity_achieved="content" if body_chars else "snippet",
        content=(
            ContentSlot(
                text=page_text(body_chars),
                origin="provider-response",
                mime_type="text/plain",
                size_bytes=body_chars,
            )
            if body_chars
            else None
        ),
        facets={"media_category": "web", "extraction_status": "ok"},
    )


def artifact_for(count: int, *, body_chars: int = 0, snippet_chars: int = len(SNIPPET)) -> dict[str, Any]:
    """Project a candidate set through the real border projection.

    Goes through :func:`threetears.search.bind.project_metadata` rather than
    dumping a model, so what is measured is what a tool actually puts on
    ``ToolMessage.artifact``.

    :param count: how many candidates the set holds
    :ptype count: int
    :param body_chars: extracted page text per candidate, or 0 for none
    :ptype body_chars: int
    :param snippet_chars: provider snippet length per candidate
    :ptype snippet_chars: int
    :return: the artifact dict, under its named key
    :rtype: dict[str, Any]
    """
    candidates = tuple(candidate(i, body_chars=body_chars, snippet_chars=snippet_chars) for i in range(count))
    return project_metadata("otter population survey 2026", CandidateSet(candidates=candidates))


def scriob_wire_bytes(artifact: dict[str, Any]) -> int:
    """Bytes published to NATS for one shared-room chat frame.

    The nesting is real, not modelled: ``StreamingResponse`` serializes the
    event, ``WsStreamTransport`` wraps it in a ``Frame``, and
    ``RoomFanout.broadcast`` wraps that in a ``RoomFrame`` before publishing.

    :param artifact: the projection to carry
    :ptype artifact: dict[str, Any]
    :return: size of the published payload in bytes
    :rtype: int
    """
    event_json = json.dumps(
        {
            "type": "tool_call_end",
            "correlation_id": "0199a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b",
            "tool_name": "threetears.web_search",
            "success": True,
            "elapsed_ms": 840,
            "structured": artifact,
            "structured_kind": "inline",
        }
    )
    frame = Frame(type="tool_call_end", room="acme:story:chat:0199", payload=event_json)
    room_frame = RoomFrame(room_id="acme:story:chat:0199", payload=frame.model_dump_json(), origin_pod="pod-a")
    return len(room_frame.model_dump_json().encode("utf-8"))


def metallm_wire_bytes(artifact: dict[str, Any]) -> int:
    """Bytes published to NATS for one cross-worker WS fanout message.

    :param artifact: the projection to carry
    :ptype artifact: dict[str, Any]
    :return: size of the published payload in bytes
    :rtype: int
    """
    message = json.dumps(
        {
            "type": "tool_invocation_end",
            "tool_name": "threetears.web_search",
            "tool_status": "completed",
            "tool_duration_ms": 840,
            "conversation_id": "0199a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b",
            "structured": artifact,
            "structured_kind": "inline",
        }
    )
    envelope = json.dumps(
        {"target_kind": "room", "target_id": "0199a1b2-c3d4-7e5f-8a9b-0c1d2e3f4a5b", "payload": message}
    )
    return len(envelope.encode("utf-8"))


SHAPES: tuple[tuple[str, int, int], ...] = (
    ("10 results, metadata only", 10, 0),
    ("20 results, metadata only", 20, 0),
    ("50 results, metadata only", 50, 0),
    ("1 result, 20 KB extracted text", 1, 20 * 1024),
    ("1 result, 100 KB extracted text", 1, 100 * 1024),
    ("8 results, 100 KB each (a research corpus)", 8, 100 * 1024),
    ("20 results, 100 KB each (2 MB of corpus)", 20, 100 * 1024),
)


def main() -> None:
    """Print the size table, the per-candidate curve, and where the ceilings sit.

    :return: nothing
    :rtype: None
    """
    print(f"{'shape':<44}{'artifact':>11}{'scriob':>11}{'metallm':>11}{'nesting':>10}")
    worst_inflation = 1.0
    for label, count, body in SHAPES:
        artifact = artifact_for(count, body_chars=body)
        size = len(json.dumps(artifact, ensure_ascii=False))
        scriob = scriob_wire_bytes(artifact)
        metallm = metallm_wire_bytes(artifact)
        inflation = scriob / size
        worst_inflation = max(worst_inflation, inflation)
        over = "  OVER" if scriob > NATS_DEFAULT_MAX_PAYLOAD else ""
        print(f"{label:<44}{size:>11,}{scriob:>11,}{metallm:>11,}{inflation:>9.2f}x{over}")

    print("\nper-candidate cost is a function of snippet length (20 candidates, no bodies):")
    for snippet in (100, 200, 400, 800):
        artifact = artifact_for(20, snippet_chars=snippet)
        per = len(json.dumps(artifact, ensure_ascii=False)) / 20
        print(f"  snippet {snippet:>4} chars -> {per:>6,.0f} chars per candidate")

    print("\nwhere the ceilings sit, measured in artifact characters:")
    print(f"  {'the inline bound #355 ships with':<44}{16_384:>11,}")
    ceiling = int(NATS_DEFAULT_MAX_PAYLOAD / worst_inflation)
    print(f"  {'untuned NATS max_payload, after nesting':<44}{ceiling:>11,}")


if __name__ == "__main__":
    main()
