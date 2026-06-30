"""Producer-side helper: stream a large tool result to the object store.

A producing tool -- one that generates an artifact too big to cross NATS (a
pcap, a database dump, a rendered report) -- calls this from inside its
``execute`` to push the bytes out-of-band and get back a small
:class:`~threetears.media.contracts.ObjectHandle` to return in
``ToolResult.metadata``. The bytes never travel with the handle.

The object store and the VERIFIED caller identity both ride on the per-call
:class:`~threetears.agent.tools.call_scope.ToolCallScope` that the tool server
installs around every dispatch. This helper reads them through
:func:`~threetears.agent.tools.call_scope.current_scope`, builds the
tenant-scoped key from the verified ``customer_id`` (NEVER an LLM-supplied
value), streams via the injected store, and returns the handle.

It fails closed -- raising :class:`ProduceObjectError` -- when invoked outside
a call scope, when the pod was wired with no object store, or when the call
context carries no verified customer. A tool must never silently produce an
unscoped or untenanted object.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, AsyncIterator
from uuid import uuid7

from threetears.media.contracts import (
    OBJECT_HANDLE_METADATA_KEY,
    ObjectHandle,
    build_object_key,
)
from threetears.observe import get_logger

from threetears.agent.tools.call_scope import current_scope
from threetears.agent.tools.context_envelope import CallContext

__all__ = [
    "ProduceObjectError",
    "object_handle_result_metadata",
    "stream_result_to_object_store",
]

_log = get_logger(__name__)


class ProduceObjectError(RuntimeError):
    """A producing tool could not stream a result to the object store.

    Carries a structural reason only (no call scope / no object store / no
    verified customer) -- never object bytes or key material.
    """


def _scope_label(context: CallContext) -> str:
    """Derive the owning-context key segment from the verified call context.

    Prefers the most specific verified owner present: the engagement (the
    authorized owning context when a call runs inside one), then the
    conversation, then the agent. The chosen value lands as the ``scope``
    segment of the object key, so derivatives of one engagement /
    conversation co-locate under a readable prefix.

    :param context: the verified per-call identity envelope
    :ptype context: CallContext
    :return: the owning-context label (``engagement-`` / ``conversation-`` /
        ``agent-`` + id)
    :rtype: str
    :raises ProduceObjectError: when the context carries no owning identifier
    """
    if context.engagement_id is not None:
        label = f"engagement-{context.engagement_id}"
    elif context.conversation_id is not None:
        label = f"conversation-{context.conversation_id}"
    elif context.agent_id is not None:
        label = f"agent-{context.agent_id}"
    else:
        raise ProduceObjectError(
            "cannot scope a produced object: the call context carries no engagement, conversation, or agent identifier"
        )
    return label


async def stream_result_to_object_store(
    body: AsyncIterator[bytes],
    *,
    filename: str,
    content_type: str,
    category: str,
    summary: str | None = None,
    size_hint: int | None = None,
    created: datetime | None = None,
) -> ObjectHandle:
    """Stream a large tool result to the pod's object store; return its handle.

    Reads the object store + verified identity off the current call scope,
    builds the tenant-scoped key (``<customer_id>/<scope>/<category>/...``)
    from the VERIFIED ``customer_id``, streams ``body`` to it, and returns the
    :class:`ObjectHandle` the tool puts in ``ToolResult.metadata``. The
    streamed bytes never cross NATS.

    :param body: async iterator yielding the artifact's bytes in chunks; the
        store consumes it without buffering the whole object
    :ptype body: AsyncIterator[bytes]
    :param filename: original filename + extension, kept in the key for human
        readability and correct download naming
    :ptype filename: str
    :param content_type: MIME type stored on the object
    :ptype content_type: str
    :param category: object kind (``reports`` / ``evidence`` / ``scans`` ...),
        a readable key segment grouping like artifacts
    :ptype category: str
    :param summary: short human/model-facing description of the artifact; rides
        on the handle so the model sees it without the bytes
    :ptype summary: str | None
    :param size_hint: optional advisory total length forwarded to the store
        (an impl may use it to pick a single PUT below its multipart
        threshold). It is NOT trusted for the handle: the bytes actually
        streamed are counted and that true count is recorded on the handle, so
        a wrong or absent hint never writes a false size. ``None`` for
        unknown-length producers (the store streams regardless)
    :ptype size_hint: int | None
    :param created: creation timestamp driving the ``YYYY/MM/DD`` key
        partition; defaults to now (UTC). injectable for deterministic tests
    :ptype created: datetime | None
    :return: the handle describing the stored object; its ``size_bytes`` is the
        actual number of bytes streamed, not the advisory hint
    :rtype: ObjectHandle
    :raises ProduceObjectError: when called outside a call scope, when the pod
        has no object store wired, or when the context has no verified customer
    """
    scope = current_scope()
    if scope is None:
        raise ProduceObjectError(
            "stream_result_to_object_store called outside a ToolServer call "
            "scope; a producing tool runs inside enter_call_scope"
        )
    store = scope.object_store
    if store is None:
        raise ProduceObjectError(
            "the current call scope carries no object store; the tool pod was "
            "not wired with one (no object store configured)"
        )
    customer_id = scope.context.customer_id
    if customer_id is None:
        raise ProduceObjectError(
            "the call context carries no verified customer_id; refusing to produce an untenanted object"
        )
    object_id = uuid7()
    key = build_object_key(
        customer_id=customer_id,
        scope=_scope_label(scope.context),
        category=category,
        object_id=object_id,
        created=created if created is not None else datetime.now(UTC),
        filename=filename,
    )
    # count the bytes actually streamed so the handle records the TRUE size,
    # not the caller's advisory hint (which may be wrong or absent).
    streamed = 0

    async def _counted() -> AsyncIterator[bytes]:
        nonlocal streamed
        async for chunk in body:
            streamed += len(chunk)
            yield chunk

    await store.put(key, _counted(), content_type=content_type, size=size_hint)
    _log.info(
        "streamed produced object to store",
        extra={
            "extra_data": {
                # object_id is the safe handle; the key is NOT logged (it
                # embeds customer_id + the owning-context label).
                "object_id": str(object_id),  # convert at border: produced-object log extra_data field
                "category": category,
                "content_type": content_type,
                "size_bytes": streamed,
            }
        },
    )
    return ObjectHandle(
        object_id=object_id,
        s3_key=key,
        mime_type=content_type,
        size_bytes=streamed,
        summary=summary,
        category=category,
    )


def object_handle_result_metadata(handle: ObjectHandle) -> dict[str, Any]:
    """Build the ``ToolResult.metadata`` dict that carries a produced object.

    A producing tool returns ``ToolResult(content=<summary>, metadata=...)``;
    pass the handle from :func:`stream_result_to_object_store` here to get the
    metadata the agent's catalog seam recognises (the handle under
    :data:`~threetears.media.contracts.OBJECT_HANDLE_METADATA_KEY`). Merge extra
    keys in if the tool also carries its own metadata.

    :param handle: the handle for the streamed object
    :ptype handle: ObjectHandle
    :return: ``{OBJECT_HANDLE_METADATA_KEY: handle.to_metadata()}``
    :rtype: dict[str, Any]
    """
    return {OBJECT_HANDLE_METADATA_KEY: handle.to_metadata()}
