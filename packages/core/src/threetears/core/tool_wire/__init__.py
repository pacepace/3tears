"""the tool-call wire protocol: what a tool pod and the registry say to each other.

both ends parse these messages, so they live in core, which both depend on, and
neither end depends on the other to read what it sends.

public surface:

- the call context (``context``): :class:`CallContext`, the identity + trace
  envelope one tool call carries, and :func:`bind_log_context`, its projection
  onto the canonical log tags.
- the REST declaration (``rest``): :class:`RestAffordance`, a tool's authored
  inbound HTTP address, on the :class:`PathTemplateBinding` it shares with the
  outbound descriptor; :class:`ParameterBinding`, :class:`HttpMethod`,
  :class:`RestAffordanceError`, :data:`QUERY_METHODS`, :data:`CACHEABLE_METHODS`
  and :data:`MAX_POINTER_AGE_SECONDS`. it rides here because
  :attr:`ToolManifestEntry.face_rest` carries it.
- the messages (``messages``): registration (:class:`ToolManifestEntry`,
  :class:`RegistrationManifest`, :class:`RefusedTool`,
  :class:`RegistrationResponse`), the call (:class:`CallRequest`,
  :class:`CallAccepted`, :class:`CallResponse`) and :class:`HeartbeatMessage`.

pod policy and machinery -- which refusals are final, readiness polling, the
probe acknowledgement, ``ToolServer`` itself -- stay in
``threetears.agent.tools.server``. each name here keeps its old path:
``context_envelope`` re-exports the context, ``http_operation`` the REST
affordance, and ``server`` the messages.
"""

from __future__ import annotations

from threetears.core.tool_wire.context import CallContext, bind_log_context
from threetears.core.tool_wire.messages import (
    CallAccepted,
    CallRequest,
    CallResponse,
    HeartbeatMessage,
    RefusedTool,
    RegistrationManifest,
    RegistrationResponse,
    ToolManifestEntry,
)
from threetears.core.tool_wire.rest import (
    CACHEABLE_METHODS,
    MAX_POINTER_AGE_SECONDS,
    QUERY_METHODS,
    HttpMethod,
    ParameterBinding,
    PathTemplateBinding,
    RestAffordance,
    RestAffordanceError,
)

__all__ = [
    "CACHEABLE_METHODS",
    "MAX_POINTER_AGE_SECONDS",
    "QUERY_METHODS",
    "CallAccepted",
    "CallContext",
    "CallRequest",
    "CallResponse",
    "HeartbeatMessage",
    "HttpMethod",
    "ParameterBinding",
    "PathTemplateBinding",
    "RefusedTool",
    "RegistrationManifest",
    "RegistrationResponse",
    "RestAffordance",
    "RestAffordanceError",
    "ToolManifestEntry",
    "bind_log_context",
]
