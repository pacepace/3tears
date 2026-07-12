"""``HttpApiTool`` -- one imported REST/OpenAPI operation, modelled as a TearsTool.

an imported external API is a *set* of tools, one per operation. this module is
the tool that IS one operation: on :meth:`HttpApiTool.execute` it fills the
operation's ``path_template`` from the call arguments, splits the remainder into
query params or a JSON body per the HTTP method, resolves the upstream credential
from a ``scheme://locator`` reference at call time, sends exactly one request over
the injected **gu-task-01** :class:`threetears.core.http_client.TracedHttpClient`
(traced + retried + circuit-broken -- this module never opens a raw ``httpx``
client of its own), and returns a **passthrough** :class:`ToolResult`.

REUSE (governs this epic): this tool is a thin adapter over TWO existing
primitives and hand-rolls neither an HTTP client, a retry loop, nor a credential
cache:

- transport -- :class:`threetears.core.http_client.TracedHttpClient` (injected);
- secrets -- :func:`threetears.core.security.secret_refs.resolve_secret` /
  :func:`threetears.core.security.secret_refs.validate_ref`.

the passthrough result serves two faces (see CLAUDE.md "Datasource Honesty" /
"Governed Guidance"): the **agent face** reads ``content`` only (the SDK
ToolMessage path DROPS ``metadata``), so ``content`` is the self-sufficient raw
upstream body. the **API face** carries ``metadata`` through the ``CallResponse``
envelope, so the HTTP ``status`` / ``headers`` / ``content_type`` live in
``metadata["http"]`` for :func:`map_tool_result_to_http` to reconstruct. do NOT
move the status into ``content``: the two fields serve different faces.

face flags (``face_api`` / ``face_mcp`` / ``face_platform_tool``) are authored
per capability source and stamped by the hub-side ``ApiToolPod`` (gu-task-24), not
baked here -- the :class:`~threetears.agent.tools.base_tool.TearsTool` defaults
apply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.utils import tool_error
from threetears.core.http_client import UpstreamHttpError
from threetears.core.security.secret_refs import (
    SecretResolutionError,
    resolve_secret,
    validate_ref,
)

if TYPE_CHECKING:
    from threetears.core.http_client import TracedHttpClient

__all__ = ["HttpApiTool", "HttpOperationDescriptor"]

# HTTP verbs that carry their non-path arguments as query params rather than a
# JSON request body. everything else (POST / PUT / PATCH) sends a JSON body.
_QUERY_METHODS: frozenset[str] = frozenset({"GET", "DELETE", "HEAD", "OPTIONS"})

# a ``{name}`` placeholder inside a path template.
_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")

# the duck-typed name of the circuit breaker's OPEN signal. agent-tools cannot
# import ``threetears.models.circuit_breaker.CircuitOpenError`` (no models
# dependency), so the classifier matches on the exception's class name.
_CIRCUIT_OPEN_EXC_NAME = "CircuitOpenError"


@dataclass(frozen=True)
class HttpOperationDescriptor:
    """one imported API operation, as the OpenAPI parser (gu-task-23) emits it.

    the descriptor is transport-free: ``upstream_base_url`` + the per-source
    circuit breaker live on the shared
    :class:`threetears.core.http_client.TracedHttpClient` (one per source), not
    duplicated on each per-operation descriptor. ``path_params`` is derived from
    ``path_template`` at construction and need not be supplied by the caller.

    :ivar method: HTTP verb (``GET`` / ``POST`` / ...); compared case-insensitively
    :ivar path_template: request path with ``{name}`` placeholders
        (e.g. ``/users/{id}``), joined onto the client's ``upstream_base_url``
    :ivar param_schema: JSON Schema of the operation's parameters + body; passed
        through unchanged as the tool's ``input_schema`` (powers ``run``'s coercion)
    :ivar credentials_ref: ``scheme://locator`` upstream credential reference, or
        ``None`` when the operation needs no credential
    :ivar name: stable namespaced tool name (``mcp_name``)
    :ivar version: stable tool version (``mcp_version``)
    :ivar description: human-readable operation description
    :ivar path_params: placeholder names extracted from ``path_template``; derived
        in ``__post_init__`` -- any supplied value is replaced
    """

    method: str
    path_template: str
    param_schema: dict[str, Any]
    credentials_ref: str | None
    name: str
    version: str
    description: str
    path_params: frozenset[str] = field(default=frozenset())

    def __post_init__(self) -> None:
        """derive ``path_params`` from ``path_template``.

        :return: nothing
        :rtype: None
        """
        derived = frozenset(_PLACEHOLDER_RE.findall(self.path_template))
        object.__setattr__(self, "path_params", derived)


class HttpApiTool(TearsTool):
    """a single imported REST/OpenAPI operation exposed as a TearsTool.

    constructed from an :class:`HttpOperationDescriptor` plus an injected
    :class:`threetears.core.http_client.TracedHttpClient` (shared across every
    operation of the same source). :meth:`execute` builds and sends exactly one
    upstream request over that client and returns a passthrough
    :class:`~threetears.agent.tools.base_tool.ToolResult`.

    :param descriptor: the operation this tool represents
    :ptype descriptor: HttpOperationDescriptor
    :param http_client: the shared traced transport for the operation's source
    :ptype http_client: TracedHttpClient
    """

    def __init__(
        self,
        *,
        descriptor: HttpOperationDescriptor,
        http_client: TracedHttpClient,
    ) -> None:
        """capture the descriptor + client; validate the credential ref shape.

        the credential reference (when present) is validated for shape at
        build time via
        :func:`threetears.core.security.secret_refs.validate_ref` -- a malformed
        or unknown-scheme ref fails here rather than on the first call.
        resolution stays a use-time concern (:meth:`execute`).

        :param descriptor: the operation this tool represents
        :ptype descriptor: HttpOperationDescriptor
        :param http_client: the shared traced transport for the source
        :ptype http_client: TracedHttpClient
        :return: nothing
        :rtype: None
        :raises SecretResolutionError: when ``descriptor.credentials_ref`` is a
            malformed / unknown-scheme reference
        """
        self._descriptor = descriptor
        self._http_client = http_client
        if descriptor.credentials_ref is not None:
            validate_ref(descriptor.credentials_ref)

    async def execute(self, **kwargs: Any) -> ToolResult:
        """build and send one upstream request; return a passthrough result.

        fills ``path_template`` from ``kwargs`` (a missing required placeholder
        is a guard-clause failure, no request sent), splits the remaining args
        into query params (GET / DELETE / HEAD / OPTIONS) or a JSON body
        (POST / PUT / PATCH), resolves the upstream credential into the
        outbound ``Authorization`` header, and sends the request over the
        injected traced client. the response is passed through: ``content`` is
        the raw body, ``metadata["http"]`` carries status / headers /
        content-type, and a non-2xx/3xx status is a failed-but-not-raised
        result. retry-exhausted ``UpstreamHttpError`` and a tripped
        ``CircuitOpenError`` become failed results rather than escaping the mesh.

        :param kwargs: coerced call arguments (path params + query / body fields)
        :ptype kwargs: Any
        :return: passthrough execution result
        :rtype: ToolResult
        """
        descriptor = self._descriptor
        args = dict(kwargs)

        filled_path = descriptor.path_template
        for placeholder in descriptor.path_params:
            if placeholder not in args:
                return ToolResult(
                    success=False,
                    content="",
                    error=tool_error(
                        descriptor.name,
                        descriptor.method,
                        f"missing required path parameter {placeholder!r}",
                    ),
                )
            value = args.pop(placeholder)
            filled_path = filled_path.replace("{" + placeholder + "}", quote(str(value), safe=""))

        method = descriptor.method.upper()
        params = args or None if method in _QUERY_METHODS else None
        json_body = None if method in _QUERY_METHODS else (args or None)

        auth_headers: dict[str, str] | None = None
        if descriptor.credentials_ref is not None:
            try:
                secret = resolve_secret(descriptor.credentials_ref)
            except SecretResolutionError as exc:
                return ToolResult(
                    success=False,
                    content="",
                    error=tool_error(descriptor.name, method, str(exc)),
                )
            auth_headers = {"Authorization": f"Bearer {secret.get_secret_value()}"}

        try:
            response = await self._http_client.request(
                method,
                filled_path,
                headers=auth_headers,
                params=params,
                json=json_body,
            )
        except UpstreamHttpError as exc:
            result = self._result_from_upstream_error(exc)
        except Exception as exc:
            if type(exc).__name__ == _CIRCUIT_OPEN_EXC_NAME:
                result = ToolResult(
                    success=False,
                    content="",
                    error="upstream circuit open",
                )
            else:
                raise
        else:
            result = self._passthrough_result(response)
        return result

    def _passthrough_result(self, response: Any) -> ToolResult:
        """wrap a received upstream response as a passthrough ``ToolResult``.

        ``success`` is a 2xx/3xx status; ``content`` is the raw body;
        ``metadata["http"]`` carries status / headers / content-type for the API
        face. no credential value ever reaches here (headers are the *response*
        headers).

        :param response: the received upstream response (``httpx.Response``)
        :ptype response: Any
        :return: passthrough result
        :rtype: ToolResult
        """
        status = response.status_code
        success = 200 <= status < 400
        metadata = {
            "http": {
                "status": status,
                "headers": dict(response.headers),
                "content_type": response.headers.get("content-type", ""),
            }
        }
        result = ToolResult(
            success=success,
            content=response.text,
            metadata=metadata,
            error=None if success else f"upstream {status}",
        )
        return result

    def _result_from_upstream_error(self, exc: UpstreamHttpError) -> ToolResult:
        """convert a retry-exhausted ``UpstreamHttpError`` to a failed result.

        the last upstream body still rides in ``content``; the last status (or
        ``None`` when no response was ever received) rides in
        ``metadata["http"]``.

        :param exc: the retry-exhausted upstream error
        :ptype exc: UpstreamHttpError
        :return: failed passthrough result
        :rtype: ToolResult
        """
        result = ToolResult(
            success=False,
            content=exc.body.decode("utf-8", errors="replace"),
            metadata={
                "http": {
                    "status": exc.status_code,
                    "headers": {},
                    "content_type": "",
                }
            },
            error=f"upstream request failed after retries (status {exc.status_code})",
        )
        return result

    def mcp_schema(self) -> MCPToolDefinition:
        """return the MCP definition, deriving ``input_schema`` from the descriptor.

        ``input_schema`` IS ``descriptor.param_schema`` (same object) so
        :meth:`~threetears.agent.tools.base_tool.TearsTool.run`'s coercion runs
        against the parser-provided schema unchanged.

        :return: tool definition with name, version, description, input schema
        :rtype: MCPToolDefinition
        """
        result = MCPToolDefinition(
            name=self._descriptor.name,
            version=self._descriptor.version,
            description=self._descriptor.description,
            input_schema=self._descriptor.param_schema,
        )
        return result

    def mcp_name(self) -> str:
        """return the descriptor's namespaced tool name.

        :return: namespaced tool name
        :rtype: str
        """
        return self._descriptor.name

    def mcp_version(self) -> str:
        """return the descriptor's version.

        :return: version string
        :rtype: str
        """
        return self._descriptor.version
