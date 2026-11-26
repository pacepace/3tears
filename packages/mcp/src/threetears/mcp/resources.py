"""MCP resources, mirroring tools: a resource or a template of them, each with a required permission.

a server serves resources when it is given a :class:`ResourceRegistry` or a :data:`ResourceProvider`::

    resources = ResourceRegistry()
    resources.register(McpResource(uri="app://config", name="config", required_permission="app.read", reader=config))
    resources.register(
        McpResourceTemplate(
            uri_template="app://sessions/{name}/status",
            name="session status",
            required_permission="app.read",
            reader=status,  # called as status(name="...")
        )
    )
    server = McpServer(..., resources=resources)

``resources/list`` and ``resources/templates/list`` answer what the caller may read; ``resources/read`` reads one,
after resolving the caller and asking the authorizer for its permission; ``resources/subscribe`` (a stateful
session, on the handshake-era protocols) and a 2026-07-28 ``subscriptions/listen`` stream that names resources ask
the same. :meth:`~threetears.mcp.server.McpServer.notify_resource_updated` tells the subscribers a resource changed.

a refusal is a JSON-RPC error (the protocol has no failed result for resources) whose ``data`` is the error model's
detail (:func:`~threetears.mcp.errors.error_detail`): ``UNKNOWN_RESOURCE`` with ``INVALID_PARAMS``,
``IDENTITY_UNAVAILABLE``, ``PERMISSION_DENIED`` and ``AUTHZ_ERROR`` with :data:`RESOURCE_REFUSED`, and
``HANDLER_ERROR`` (a reader that raised) with ``INTERNAL_ERROR``.
"""

from __future__ import annotations

import base64
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import mcp.types as mcp_types
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, INVALID_REQUEST
from mcp.server.context import ServerRequestContext
from mcp.server.subscriptions import ListenHandler
from mcp.shared.exceptions import MCPError
from mcp.shared.uri_template import UriTemplate
from threetears.observe import contain, get_logger

from threetears.mcp.auth import Authorizer, GrantCacheUnavailableError, Identity, IdentityProvider
from threetears.mcp.errors import (
    AUTHZ_ERROR,
    HANDLER_ERROR,
    IDENTITY_UNAVAILABLE,
    PERMISSION_DENIED,
    UNKNOWN_RESOURCE,
    ErrorCode,
    error_detail,
)
from threetears.mcp.http_server import current_transport_pushes
from threetears.mcp.push import Pushes

__all__ = [
    "RESOURCE_REFUSED",
    "McpResource",
    "McpResourceTemplate",
    "ResourceContent",
    "ResourceProvider",
    "ResourceReader",
    "ResourceRegistry",
    "ResourceService",
    "identify_or_refuse",
    "refuse_unknown_resource",
]

log = get_logger(__name__)

#: the JSON-RPC error code of a resource request the caller's identity or permission refused; its ``data`` says
#: which (``IDENTITY_UNAVAILABLE``, ``PERMISSION_DENIED``, ``AUTHZ_ERROR``). in the protocol's implementation-defined
#: band, clear of the codes the SDK allocates there (-32000, -32001)
RESOURCE_REFUSED: Final = -32010


@dataclass(frozen=True, slots=True)
class ResourceContent:
    """one content item of a read, when a reader answers more than one, or a URI or type of its own.

    :ivar uri: the item's URI
    :ivar text: its text; exactly one of ``text`` and ``blob`` is given
    :ivar blob: its bytes
    :ivar mime_type: its media type
    """

    uri: str
    text: str | None = None
    blob: bytes | None = None
    mime_type: str | None = None

    def __post_init__(self) -> None:
        """refuse an item that is neither text nor bytes, or both.

        :return: nothing
        :rtype: None
        :raises ValueError: when not exactly one of ``text`` and ``blob`` is given
        """
        if (self.text is None) == (self.blob is None):
            raise ValueError(f"ResourceContent {self.uri!r}: give exactly one of text and blob")


ResourceReader = Callable[..., Awaitable["str | bytes | ResourceContent | Iterable[ResourceContent]"]]
"""reads a resource: called with no arguments for a :class:`McpResource`, and with the URI's template variables as
keywords for a :class:`McpResourceTemplate`. text is answered as text, bytes as a blob, both under the URI read and
the resource's media type; :class:`ResourceContent` items as they are."""


def _check_named(kind: str, name: str, required_permission: str) -> None:
    """refuse an empty name or permission, as :class:`~threetears.mcp.tool.McpTool` does.

    :param kind: what is being built, for the message
    :ptype kind: str
    :param name: its name
    :ptype name: str
    :param required_permission: its permission
    :ptype required_permission: str
    :return: nothing
    :rtype: None
    :raises ValueError: when either is empty
    """
    if not name:
        raise ValueError(f"{kind}.name must be non-empty")
    if not required_permission:
        raise ValueError(
            f"{kind} {name!r}: required_permission must be non-empty; use a sentinel like 'public' if it truly has "
            "no permission gate (the Authorizer decides)",
        )


@dataclass(frozen=True, slots=True)
class McpResource:
    """one resource at a fixed URI.

    :ivar uri: the URI a client reads
    :ivar name: its name, listed
    :ivar required_permission: the permission a caller must hold to list, read or subscribe to it
    :ivar reader: reads it, called with no arguments
    :ivar description: what it holds
    :ivar mime_type: its media type
    :ivar title: a human-readable display name
    """

    uri: str
    name: str
    required_permission: str
    reader: ResourceReader
    description: str | None = None
    mime_type: str | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        """refuse an empty URI, name or permission.

        :return: nothing
        :rtype: None
        :raises ValueError: when one is empty
        """
        if not self.uri:
            raise ValueError("McpResource.uri must be non-empty")
        _check_named("McpResource", self.name, self.required_permission)

    def listed(self) -> mcp_types.Resource:
        """the ``resources/list`` entry.

        :return: the SDK's resource
        :rtype: mcp.types.Resource
        """
        return mcp_types.Resource(
            uri=self.uri, name=self.name, description=self.description, mime_type=self.mime_type, title=self.title
        )


@dataclass(frozen=True, slots=True)
class McpResourceTemplate:
    """resources whose URIs follow one RFC 6570 template, read with the URI's variables.

    :ivar uri_template: the template, e.g. ``app://sessions/{name}/status``
    :ivar name: its name, listed
    :ivar required_permission: the permission a caller must hold to list it, or read or subscribe to a URI of it
    :ivar reader: reads one, called with the URI's variables as keywords
    :ivar description: what its resources hold
    :ivar mime_type: their media type
    :ivar title: a human-readable display name
    """

    uri_template: str
    name: str
    required_permission: str
    reader: ResourceReader
    description: str | None = None
    mime_type: str | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        """refuse an empty name or permission, or a template that does not parse.

        :return: nothing
        :rtype: None
        :raises ValueError: when one is empty, or the template is not a URI template the SDK can match
        """
        _check_named("McpResourceTemplate", self.name, self.required_permission)
        UriTemplate.parse(self.uri_template)

    def match(self, uri: str) -> dict[str, str | list[str]] | None:
        """the variables ``uri`` binds in this template, or ``None`` when it is not one of its URIs.

        :param uri: a URI
        :ptype uri: str
        :return: the variables
        :rtype: dict[str, str | list[str]] | None
        """
        return UriTemplate.parse(self.uri_template).match(uri)

    def listed(self) -> mcp_types.ResourceTemplate:
        """the ``resources/templates/list`` entry.

        :return: the SDK's template
        :rtype: mcp.types.ResourceTemplate
        """
        return mcp_types.ResourceTemplate(
            uri_template=self.uri_template,
            name=self.name,
            description=self.description,
            mime_type=self.mime_type,
            title=self.title,
        )


ResourceProvider = Callable[[], Iterable["McpResource | McpResourceTemplate"]]
"""a callable answering the resources and templates a server serves right now, called on every resource request,
as :data:`~threetears.mcp.tool.ToolProvider` is for tools. :meth:`ResourceRegistry.all` is one."""


class ResourceRegistry:
    """a fixed set of resources and templates, for a server that does not follow a live set."""

    def __init__(self) -> None:
        """start empty."""
        # a read-only snapshot replaced whole on a change; keyed by URI or URI template
        self._entries: Mapping[str, McpResource | McpResourceTemplate] = MappingProxyType({})

    def register(self, entry: McpResource | McpResourceTemplate) -> None:
        """add a resource or a template.

        :param entry: the resource or template
        :ptype entry: McpResource | McpResourceTemplate
        :return: nothing
        :rtype: None
        :raises ValueError: when one with that URI (or URI template) is already registered
        """
        key = entry.uri if isinstance(entry, McpResource) else entry.uri_template
        if key in self._entries:
            raise ValueError(f"resource conflict: {key!r} already registered")
        self._entries = MappingProxyType({**self._entries, key: entry})

    def all(self) -> list[McpResource | McpResourceTemplate]:
        """every resource and template, in registration order.

        :return: them
        :rtype: list[McpResource | McpResourceTemplate]
        """
        return list(self._entries.values())


def _refused(code: ErrorCode, message: str, *, jsonrpc_code: int = RESOURCE_REFUSED) -> MCPError:
    """the JSON-RPC error a refused resource request answers, the error model's detail as its data.

    :param code: the error model's code
    :ptype code: ErrorCode
    :param message: what went wrong
    :ptype message: str
    :param jsonrpc_code: the JSON-RPC code
    :ptype jsonrpc_code: int
    :return: the error to raise
    :rtype: mcp.shared.exceptions.MCPError
    """
    return MCPError(jsonrpc_code, message, data=error_detail(code, message))


class ResourceService:
    """what :class:`~threetears.mcp.server.McpServer` answers resource requests with.

    :param provider: the resources and templates as they stand now
    :ptype provider: ResourceProvider
    :param identity_provider: resolves the caller of each request
    :ptype identity_provider: IdentityProvider
    :param authorizer: asked for each resource's permission
    :ptype authorizer: Authorizer
    :param pushes: the connections and listen streams updates go to
    :ptype pushes: Pushes
    """

    def __init__(
        self,
        *,
        provider: ResourceProvider,
        identity_provider: IdentityProvider,
        authorizer: Authorizer,
        pushes: Pushes,
    ) -> None:
        """capture what each request is answered from.

        :param provider: the resources and templates
        :ptype provider: ResourceProvider
        :param identity_provider: resolves the caller
        :ptype identity_provider: IdentityProvider
        :param authorizer: asked for permissions
        :ptype authorizer: Authorizer
        :param pushes: where updates go
        :ptype pushes: Pushes
        :return: nothing
        :rtype: None
        """
        self._provider = provider
        self._identity_provider = identity_provider
        self._authorizer = authorizer
        self._pushes = pushes
        self._listen = ListenHandler(pushes.bus)

    def _entries(self) -> tuple[list[McpResource], list[McpResourceTemplate]]:
        """the resources and templates the provider answers now.

        :return: the resources, and the templates
        :rtype: tuple[list[McpResource], list[McpResourceTemplate]]
        :raises MCPError: ``INTERNAL_ERROR`` when the provider fails, after logging it
        """
        try:
            entries = list(self._provider())
        except Exception as exc:
            contain(exc, log=log, what="MCP resources: the resource provider failed", level="error")
            raise MCPError(INTERNAL_ERROR, "the server's resource set could not be read") from exc
        resources = [entry for entry in entries if isinstance(entry, McpResource)]
        templates = [entry for entry in entries if isinstance(entry, McpResourceTemplate)]
        return resources, templates

    async def _identify(self) -> Identity:
        """the caller of this request.

        :return: the identity
        :rtype: Identity
        :raises MCPError: ``IDENTITY_UNAVAILABLE`` when it cannot be resolved
        """
        return await identify_or_refuse(self._identity_provider)

    async def _allows(self, identity: Identity, permission: str) -> bool:
        """whether the authorizer grants ``permission`` to ``identity``.

        :param identity: the caller
        :ptype identity: Identity
        :param permission: the permission
        :ptype permission: str
        :return: the decision
        :rtype: bool
        :raises MCPError: ``AUTHZ_ERROR`` when the authorizer cannot decide
        """
        try:
            return await self._authorizer.allows(identity, permission)
        except GrantCacheUnavailableError as exc:
            raise _refused(AUTHZ_ERROR, "authorization is temporarily unavailable; retry") from exc
        except Exception as exc:
            contain(exc, log=log, what="MCP resources: authorizer raised; denying by default")
            raise _refused(AUTHZ_ERROR, "authorizer evaluation failed") from exc

    async def _permitted(self, identity: Identity, permissions: Iterable[str]) -> frozenset[str]:
        """the permissions of ``permissions`` the caller holds, each asked once.

        :param identity: the caller
        :ptype identity: Identity
        :param permissions: the permissions
        :ptype permissions: Iterable[str]
        :return: those granted
        :rtype: frozenset[str]
        """
        granted: set[str] = set()
        for permission in dict.fromkeys(permissions):
            if await self._allows(identity, permission):
                granted.add(permission)
        return frozenset(granted)

    def _find(self, uri: str) -> tuple[McpResource | McpResourceTemplate, dict[str, str | list[str]]]:
        """the resource at ``uri``, or the first template it matches with its variables.

        :param uri: the URI
        :ptype uri: str
        :return: the entry and the variables to read it with
        :rtype: tuple[McpResource | McpResourceTemplate, dict[str, str | list[str]]]
        :raises MCPError: ``UNKNOWN_RESOURCE`` when nothing matches
        """
        resources, templates = self._entries()
        for resource in resources:
            if resource.uri == uri:
                return resource, {}
        for template in templates:
            variables = template.match(uri)
            if variables is not None:
                return template, variables
        raise refuse_unknown_resource(uri)

    async def _gate(self, uri: str) -> tuple[McpResource | McpResourceTemplate, dict[str, str | list[str]]]:
        """the entry at ``uri``, once the caller is resolved and holds its permission.

        an unknown URI is answered only to a caller with an identity, so a caller without one learns nothing of
        what exists.

        :param uri: the URI
        :ptype uri: str
        :return: the entry and its variables
        :rtype: tuple[McpResource | McpResourceTemplate, dict[str, str | list[str]]]
        :raises MCPError: as :meth:`_identify`, :meth:`_find` and :meth:`_allows`, or ``PERMISSION_DENIED``
        """
        identity = await self._identify()
        entry, variables = self._find(uri)
        if not await self._allows(identity, entry.required_permission):
            log.info(
                "MCP resources: denied",
                extra={
                    "extra_data": {
                        "uri": uri,
                        "principal_id": str(identity.principal_id),
                        "required_permission": entry.required_permission,
                    }
                },
            )
            raise _refused(PERMISSION_DENIED, f"identity lacks required permission {entry.required_permission!r}")
        return entry, variables

    async def list_resources(self) -> mcp_types.ListResourcesResult:
        """the resources the caller may read.

        :return: the listing
        :rtype: mcp.types.ListResourcesResult
        """
        identity = await self._identify()
        resources, _ = self._entries()
        granted = await self._permitted(identity, (resource.required_permission for resource in resources))
        return mcp_types.ListResourcesResult(
            resources=[resource.listed() for resource in resources if resource.required_permission in granted]
        )

    async def list_templates(self) -> mcp_types.ListResourceTemplatesResult:
        """the templates whose resources the caller may read.

        :return: the listing
        :rtype: mcp.types.ListResourceTemplatesResult
        """
        identity = await self._identify()
        _, templates = self._entries()
        granted = await self._permitted(identity, (template.required_permission for template in templates))
        return mcp_types.ListResourceTemplatesResult(
            resource_templates=[template.listed() for template in templates if template.required_permission in granted]
        )

    async def read(self, uri: str) -> mcp_types.ReadResourceResult:
        """read the resource at ``uri``.

        :param uri: the URI
        :ptype uri: str
        :return: its contents
        :rtype: mcp.types.ReadResourceResult
        :raises MCPError: when refused, or ``HANDLER_ERROR`` when the reader raises (logged)
        """
        entry, variables = await self._gate(uri)
        try:
            contents = _contents(uri, entry.mime_type, await entry.reader(**variables))
        except Exception as exc:
            contain(exc, log=log, what="MCP resources: reader exception", extra={"uri": uri})
            raise _refused(HANDLER_ERROR, f"{type(exc).__name__}: {exc}", jsonrpc_code=INTERNAL_ERROR) from exc
        return mcp_types.ReadResourceResult(contents=contents)

    async def subscribe(self, ctx: ServerRequestContext[Any, Any], uri: str) -> mcp_types.EmptyResult:
        """record a stateful session's subscription to ``uri``.

        :param ctx: the request's context
        :ptype ctx: mcp.server.context.ServerRequestContext
        :param uri: the URI
        :ptype uri: str
        :return: nothing to answer
        :rtype: mcp.types.EmptyResult
        :raises MCPError: when refused, or ``INVALID_REQUEST`` when the connection cannot be pushed to (stateless)
        """
        await self._gate(uri)
        if not current_transport_pushes() or not self._pushes.subscribe(ctx.session, uri):
            raise MCPError(
                INVALID_REQUEST,
                "resource subscriptions need a stateful session (the server's transport is stateless)",
            )
        return mcp_types.EmptyResult()

    def unsubscribe(self, ctx: ServerRequestContext[Any, Any], uri: str) -> mcp_types.EmptyResult:
        """drop a session's subscription to ``uri``.

        :param ctx: the request's context
        :ptype ctx: mcp.server.context.ServerRequestContext
        :param uri: the URI
        :ptype uri: str
        :return: nothing to answer
        :rtype: mcp.types.EmptyResult
        """
        self._pushes.unsubscribe(ctx.session, uri)
        return mcp_types.EmptyResult()

    async def listen(
        self, ctx: ServerRequestContext[Any, Any], params: mcp_types.SubscriptionsListenRequestParams
    ) -> mcp_types.SubscriptionsListenResult:
        """serve a 2026-07-28 listen stream, once the caller holds the permission of every resource it names.

        :param ctx: the request's context
        :ptype ctx: mcp.server.context.ServerRequestContext
        :param params: the listen request
        :ptype params: mcp.types.SubscriptionsListenRequestParams
        :return: the stream's closing result
        :rtype: mcp.types.SubscriptionsListenResult
        :raises MCPError: when the caller has no identity, or a named resource refuses it
        """
        await self._identify()
        for uri in params.notifications.resource_subscriptions or ():
            await self._gate(uri)
        return await self._listen(ctx, params)


async def identify_or_refuse(identity_provider: IdentityProvider) -> Identity:
    """the caller of a request that has no failed result to answer with (a resource request, a listen stream).

    :param identity_provider: resolves the caller
    :ptype identity_provider: IdentityProvider
    :return: the identity
    :rtype: Identity
    :raises MCPError: ``IDENTITY_UNAVAILABLE`` (:data:`RESOURCE_REFUSED`) when it cannot be resolved
    """
    try:
        return await identity_provider.identify()
    except RuntimeError as exc:
        contain(exc, log=log, what="MCP request: identity resolution failed")
        raise _refused(IDENTITY_UNAVAILABLE, "caller identity could not be resolved") from exc


def refuse_unknown_resource(uri: str) -> MCPError:
    """the error a URI no resource answers is refused with.

    :param uri: the URI
    :ptype uri: str
    :return: the error to raise
    :rtype: mcp.shared.exceptions.MCPError
    """
    return _refused(UNKNOWN_RESOURCE, f"no resource at {uri!r}", jsonrpc_code=INVALID_PARAMS)


def _contents(
    uri: str, mime_type: str | None, answer: str | bytes | ResourceContent | Iterable[ResourceContent]
) -> list[mcp_types.TextResourceContents | mcp_types.BlobResourceContents]:
    """a reader's answer as the protocol's contents.

    :param uri: the URI read
    :ptype uri: str
    :param mime_type: the resource's media type
    :ptype mime_type: str | None
    :param answer: what the reader answered
    :ptype answer: str | bytes | ResourceContent | Iterable[ResourceContent]
    :return: the contents
    :rtype: list[mcp.types.TextResourceContents | mcp.types.BlobResourceContents]
    :raises TypeError: when the reader answered something else
    """
    if isinstance(answer, str):
        items = [ResourceContent(uri=uri, text=answer, mime_type=mime_type)]
    elif isinstance(answer, bytes):
        items = [ResourceContent(uri=uri, blob=answer, mime_type=mime_type)]
    elif isinstance(answer, ResourceContent):
        items = [answer]
    else:
        items = list(answer)
        if not all(isinstance(item, ResourceContent) for item in items):
            raise TypeError("a resource reader answers str, bytes, a ResourceContent or an iterable of them")
    contents: list[mcp_types.TextResourceContents | mcp_types.BlobResourceContents] = []
    for item in items:
        if item.text is not None:
            contents.append(mcp_types.TextResourceContents(uri=item.uri, text=item.text, mime_type=item.mime_type))
        elif item.blob is not None:
            contents.append(
                mcp_types.BlobResourceContents(
                    uri=item.uri, blob=base64.b64encode(item.blob).decode("ascii"), mime_type=item.mime_type
                )
            )
    return contents
