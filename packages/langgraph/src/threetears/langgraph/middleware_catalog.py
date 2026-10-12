"""3tears produced-object catalog middleware for ``langchain.agents.create_agent``.

The framework-aligned successor to the old ``tool_node`` media-catalog seam (the
hand-rolled node's ``_maybe_catalog_object`` branch, removed with ``nodes.py``).
Re-expressed onto :meth:`langchain.agents.middleware.AgentMiddleware.awrap_tool_call`:
after a tool runs, when it streamed a large artifact to the object store and returned
an :class:`~threetears.media.contracts.ObjectHandle` in its result artifact (under
``OBJECT_HANDLE_METADATA_KEY``), the seam persists a catalog record via the injected
:class:`~threetears.langgraph.catalog.ObjectCataloger`, stamped with the VERIFIED
per-call identity read off ``config["configurable"]`` / ``call_context``.

Opt-in and soft-fail by construction, exactly as the original seam: no cataloger ->
no-op; a missing verified conversation/customer identity, a malformed handle, or a
catalog error is logged and tolerated -- the object stays in the store but
uncataloged, so the hub-side reconciler garbage-collects it past its grace window. A
failed catalog must NEVER break the tool result. ``GraphBubbleUp`` propagates.

The real seam is asynchronous -- the :class:`ObjectCataloger` contract's ``catalog``
is ``async def``, matching the async-only original ``tool_node``. The synchronous
:meth:`~ObjectCatalogMiddleware.wrap_tool_call` mirror cannot drive an async
cataloger, so on the sync tool-call path it skips the side-effect (leaving the
object to the reconciler, consistent with the seam's tolerate-failure contract) and
warns when a cataloger was in fact configured -- it never breaks the tool result.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp

from threetears.media.contracts import OBJECT_HANDLE_METADATA_KEY, ObjectHandle
from threetears.observe import get_logger

__all__ = ["ObjectCatalogMiddleware"]

log = get_logger(__name__)


async def _maybe_catalog_object(
    configurable: dict[str, Any],
    tool_name: str,
    tool_artifact: Any,
    success: bool,
) -> None:
    """catalog a produced object (an ObjectHandle in the tool artifact) via the cataloger.

    Fires only when ALL hold: the tool succeeded, a
    :class:`~threetears.langgraph.catalog.ObjectCataloger` is injected on
    ``configurable``, the artifact carries an ``OBJECT_HANDLE_METADATA_KEY``
    descriptor, and the VERIFIED conversation + customer identity are known
    (read from ``call_context`` / the offload-seam keys).

    SOFT-FAIL: a malformed handle or a catalog error is logged and swallowed --
    the object is already in the store but uncataloged, so it becomes an orphan
    the hub-side reconciler garbage-collects past its grace window. A failed
    catalog must never break the tool result. ``GraphBubbleUp`` propagates (it is
    a control-flow signal, not a catalog error).

    :param configurable: the ``config["configurable"]`` dict for this run.
    :ptype configurable: dict[str, Any]
    :param tool_name: name of the tool whose result is being processed.
    :ptype tool_name: str
    :param tool_artifact: the tool's structured artifact (``ToolMessage.artifact``);
        the handle descriptor rides under ``OBJECT_HANDLE_METADATA_KEY`` when present.
    :ptype tool_artifact: Any
    :param success: whether the tool invocation succeeded; a failed tool is
        never cataloged (it produced no object).
    :ptype success: bool
    :return: nothing
    :rtype: None
    """
    cataloger = configurable.get("object_cataloger")
    handle_data = tool_artifact.get(OBJECT_HANDLE_METADATA_KEY) if isinstance(tool_artifact, dict) else None
    if not (success and cataloger is not None and isinstance(handle_data, dict)):
        return
    call_context = configurable.get("call_context")
    conversation_id = configurable.get("conversation_id")
    if conversation_id is None and call_context is not None:
        conversation_id = call_context.conversation_id
    user_id = configurable.get("user_id")
    if user_id is None and call_context is not None:
        user_id = call_context.user_id
    customer_id = call_context.customer_id if call_context is not None else None
    if conversation_id is None or customer_id is None:
        log.warning(
            "produced object not cataloged: missing verified conversation/customer identity",
            extra={
                "extra_data": {
                    "tool_name": tool_name,
                    "has_conversation": conversation_id is not None,
                    "has_customer": customer_id is not None,
                },
            },
        )
        return
    try:
        handle = ObjectHandle.from_metadata(handle_data)
    except (KeyError, ValueError, TypeError) as exc:
        log.warning(
            "produced object handle malformed; not cataloged",
            extra={"extra_data": {"tool_name": tool_name, "error": str(exc)}},
        )
        return
    try:
        await cataloger.catalog(
            handle,
            conversation_id=conversation_id,
            customer_id=customer_id,
            user_id=user_id,
        )
    except GraphBubbleUp:
        raise
    except Exception as exc:
        object_id_log = str(handle.object_id)  # convert at border: catalog-failed log extra_data field
        log.warning(
            "produced object catalog failed; object will be reconciled if it stays uncataloged",
            extra={"extra_data": {"tool_name": tool_name, "object_id": object_id_log, "error": str(exc)}},
            exc_info=True,
        )


class ObjectCatalogMiddleware(AgentMiddleware):
    """Catalog a produced object handle a tool returns, as a soft-fail side-effect.

    The ``create_agent`` successor to the ``tool_node`` media-catalog seam. On every
    tool call it runs the tool, then -- when the result artifact carries an
    :class:`~threetears.media.contracts.ObjectHandle` and a cataloger is injected --
    persists a catalog record under the verified identity. The tool result flows
    through unchanged; a catalog failure is logged and tolerated. Opt-in: no cataloger
    on ``config["configurable"]`` -> no-op.
    """

    name = "ObjectCatalogMiddleware"

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> Any:
        """Run the tool, then catalog a produced object handle in its artifact.

        :param request: the tool-call request (``tool_call`` + ``runtime.config``).
        :ptype request: ToolCallRequest
        :param handler: async callable that executes the tool call.
        :ptype handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
        :return: the handler result, unchanged (catalog is a pure side-effect).
        :rtype: ToolMessage | Command
        """
        result = await handler(request)
        if not isinstance(result, ToolMessage):
            return result
        configurable = request.runtime.config.get("configurable", {})
        tool_name = request.tool_call["name"]
        success = result.status != "error"
        await _maybe_catalog_object(configurable, tool_name, result.artifact, success)
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Any,
    ) -> Any:
        """Sync mirror: cannot drive the async cataloger, so skip the side-effect.

        The :class:`~threetears.langgraph.catalog.ObjectCataloger` contract is
        ``async def``; a synchronous tool-call path cannot await it. Consistent with
        the seam's tolerate-failure contract (a catalog failure must never break the
        tool result), the side-effect is skipped -- the object is left to the hub-side
        reconciler -- and a warning is logged when a cataloger was configured.

        :param request: the tool-call request.
        :ptype request: ToolCallRequest
        :param handler: sync callable that executes the tool call.
        :ptype handler: Callable[[ToolCallRequest], ToolMessage | Command]
        :return: the handler result, unchanged.
        :rtype: ToolMessage | Command
        """
        result = handler(request)
        if not isinstance(result, ToolMessage):
            return result
        configurable = request.runtime.config.get("configurable", {})
        artifact = result.artifact
        has_handle = isinstance(artifact, dict) and isinstance(artifact.get(OBJECT_HANDLE_METADATA_KEY), dict)
        if configurable.get("object_cataloger") is not None and has_handle:
            log.warning(
                "produced object catalog skipped: a cataloger is configured but the "
                "agent ran on the synchronous tool-call path; the cataloger is "
                "async-only, so the object is left to the reconciler",
                extra={"extra_data": {"tool_name": request.tool_call["name"]}},
            )
        return result
