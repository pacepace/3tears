"""path-template binding, shared by the outbound and inbound HTTP faces.

one tool, one set of code, reachable over whichever faces its author
declares. three of those faces are booleans on
:class:`~threetears.agent.tools.base_tool.TearsTool` because their address is
derived: the mesh subject, the API operation and the MCP tool name all fall
out of ``mcp_name()``. a REST face cannot be a boolean -- an inbound HTTP
address has to come from somewhere -- so it is declared as a
:class:`RestAffordance`.

**this module ships INERT.** it declares; nothing here serves. no routing, no
OpenAPI document, no cache headers, no dispatch. the Hub builds the serving
half against this published declaration.

REUSE, and what "inbound differs" actually means. the outbound half of this
idea already ships one directory down, in
:class:`~threetears.agent.tools.builtin.http_api_tool.HttpOperationDescriptor`:
a frozen dataclass of method + path template whose ``{name}`` placeholders are
derived from the template at construction. that derivation, the placeholder
regex and the query-versus-body method split are lifted here into
:class:`PathTemplateBinding` and both faces now inherit them, so an inbound
declaration cannot drift from the outbound descriptor that already works.

what the inbound half deliberately does NOT inherit, and why:

- **no ``credentials_ref``.** outbound resolves an upstream secret to
  authenticate ITSELF to a third party. inbound is authorized per caller by
  the platform's own ACL; there is no secret to hold, and a field for one
  would invite an inbound tool to carry a bearer token.
- **no ``param_schema``.** outbound carries the schema because the imported
  operation IS the only place it exists. inbound already has one --
  ``mcp_schema().input_schema`` -- and restating it beside the declaration
  would be a second place to say the same thing.
- **no ``name`` / ``version`` / ``description``.** the tool declares those
  through ``mcp_name()`` / ``mcp_version()`` / ``mcp_schema()``.
- **a closed method vocabulary.** outbound must accept whatever verb a
  third-party spec carries (``PROPFIND`` and worse), so
  :class:`HttpOperationDescriptor` stays permissive. inbound is authored in
  this codebase, so an unknown verb is a typo and fails at declaration time.

**location, not just path.** ``mcp_schema().input_schema`` is one flat
object; binding some of its properties to path segments leaves the rest
needing a home, and a GET has no body to put them in. :meth:`RestAffordance.bind`
resolves that with the same constant the outbound tool already uses --
:data:`QUERY_METHODS` -- so the two faces agree on where an argument lives.
header and cookie binding are deliberately absent inbound: inbound headers
carry the platform's own concerns (authorization, correlation), not tool
arguments.

**what is validated here, and what is NOT.** 3tears can check *intra*-tool
coherence and does: the method is in the vocabulary, every ``{name}`` in the
template is a property of the tool's own schema, a declared cache version
parameter actually rides in the URL, and shared cacheability is only claimed
for a method that has a shared-cache story. those fail at class definition or
at registration -- the posture
:class:`~threetears.core.collections.schema_backed.PartitionEnforcementError`
takes, and for the same reason.

3tears CANNOT check *inter*-tool coherence: template collision across pods,
prefix ownership within a customer, or whether a resolved tool has an ingress
principal at all. those need the full ``platform.namespaces`` view, so they
belong to the Hub-side serving shard, which refuses on collision.

**scalar coercion is a known, open gap, and it is NOT closed here.**
:mod:`threetears.agent.tools._coercion` engages only for declared types
``object`` and ``array``; every other field passes through untouched, and
``run`` performs no schema validation. URL segments are always strings, so a
tool declaring ``{"page": {"type": "integer"}}`` would receive ``"5"`` over
REST and ``5`` over a JSON body -- a per-face divergence inside one tool,
which is precisely what "one tool, one set of code" exists to prevent.

The gap is assigned to the SERVING shard, not to this one, deliberately:
widening ``_coercion`` to scalars changes the arguments every existing tool
receives on every existing face, and a declaration-only shard that ships
inert must not carry a behaviour change for code nothing has called yet. The
serving shard parses path and query segments against the declared schema
BEFORE dispatch, so the tool's ``execute`` sees the same Python types
whichever face the call arrived on. :meth:`RestAffordance.bind` gives it the
property names to parse; the schema it parses against is
``mcp_schema().input_schema``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from threetears.core.http_cache import CacheClass, narrow_cache_class

__all__ = [
    "CACHEABLE_METHODS",
    "QUERY_METHODS",
    "HttpMethod",
    "ParameterBinding",
    "PathTemplateBinding",
    "RestAffordance",
    "RestAffordanceError",
]

#: a ``{name}`` placeholder inside a path template.
_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")

#: HTTP verbs that carry their non-path arguments as query params rather than
#: a request body. everything else (POST / PUT / PATCH) uses a body. this is
#: the constant the outbound :class:`HttpApiTool` has always split on; the
#: inbound face reuses it rather than inventing a second rule.
QUERY_METHODS: frozenset[str] = frozenset({"GET", "DELETE", "HEAD", "OPTIONS"})

#: verbs whose responses a shared cache may hold. a declaration may only claim
#: something wider than ``PRIVATE`` for one of these -- there is no shared-cache
#: story for a POST, and claiming one is how a mutation gets replayed from an
#: edge.
CACHEABLE_METHODS: frozenset[str] = frozenset({"GET", "HEAD"})


class HttpMethod(StrEnum):
    """the closed vocabulary an inbound tool face may declare.

    spelled once, validated at the producer -- the same posture the audit
    vocabulary takes. it covers every member of :data:`QUERY_METHODS`: a
    five-verb list of GET/POST/PUT/PATCH/DELETE would leave HEAD and OPTIONS
    undeclarable while the binding rule still branched on them, making half
    that constant unreachable.

    HEAD and OPTIONS are admitted rather than refused because the serving
    shard has to answer them anyway -- HEAD is a GET without a body and
    OPTIONS is what a browser preflight asks -- and a face that cannot
    declare them forces the server to synthesise an address the author never
    wrote. an OPTIONS declaration does NOT replace CORS preflight handling,
    which is the server's concern and not a tool's.
    """

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class RestAffordanceError(TypeError):
    """raised when a REST declaration is incoherent with its own tool.

    a ``TypeError`` subclass raised at class-definition or registration time,
    mirroring
    :class:`~threetears.core.collections.schema_backed.PartitionEnforcementError`:
    an authored declaration that cannot be served is a programming error, and
    the cheapest place to learn that is where it was written -- not on the
    first request from outside.

    resolution: name only declared schema properties in the path template,
    pick a method from :class:`HttpMethod`, and claim shared cacheability
    only for a method a shared cache may hold.
    """


@dataclass(frozen=True)
class PathTemplateBinding:
    """method plus path template, with ``{name}`` placeholders derived.

    the structure both HTTP faces share. subclasses add what their own
    direction needs and inherit the placeholder derivation, so an inbound
    declaration and an outbound descriptor can never disagree about which
    segments of a template are parameters.

    :ivar method: HTTP verb; subclasses decide how strictly it is policed
    :ivar path_template: request path with ``{name}`` placeholders (e.g.
        ``/surveys/{survey_id}``)
    :ivar path_params: placeholder names extracted from ``path_template``;
        always derived at construction. accepted as a keyword argument for
        compatibility and then ignored, exactly as it was before this base
        class existed -- ``HttpOperationDescriptor`` has declared it since it
        was introduced and has always overwritten whatever was passed. it
        stays accepted rather than becoming ``init=False`` so no existing
        construction site raises ``TypeError``; supplying it has never had an
        effect and still has none
    """

    method: str
    path_template: str
    # keyword-only so subclasses may keep declaring REQUIRED fields after it:
    # a plain defaulted field here would put a default ahead of them and
    # dataclass construction would refuse the subclass outright.
    path_params: frozenset[str] = field(default=frozenset(), kw_only=True)

    def __post_init__(self) -> None:
        """derive ``path_params`` from ``path_template``, discarding any passed value.

        :return: nothing
        :rtype: None
        """
        derived = frozenset(_PLACEHOLDER_RE.findall(self.path_template))
        object.__setattr__(self, "path_params", derived)

    @property
    def binds_remainder_to_query(self) -> bool:
        """whether non-path arguments ride in the query string.

        :return: True when the method carries arguments as query params
        :rtype: bool
        """
        return self.method.upper() in QUERY_METHODS


@dataclass(frozen=True)
class ParameterBinding:
    """where each of a tool's declared properties rides in an HTTP request.

    the three sets partition the tool's ``input_schema`` properties. header
    and cookie locations are deliberately unrepresented inbound: those carry
    the platform's own concerns, not tool arguments.

    :ivar path: properties bound to ``{name}`` segments of the template
    :ivar query: properties bound to the query string
    :ivar body: properties bound to the JSON request body
    """

    path: frozenset[str]
    query: frozenset[str]
    body: frozenset[str]


@dataclass(frozen=True)
class RestAffordance(PathTemplateBinding):
    """a tool's authored declaration that it may be served as a REST resource.

    off by default -- ``TearsTool.face_rest`` is ``None`` until a tool
    authors one, because every face is surface to defend. like the three
    boolean faces it governs *reach* only: ACL still governs authorization,
    and a declared REST resource is per-caller authorized exactly as the mesh
    call to the same tool is.

    :ivar method: HTTP verb; must be a :class:`HttpMethod` member, normalized
        to upper case at construction
    :ivar path_template: request path with ``{name}`` placeholders, each of
        which must name a property of the tool's own ``mcp_schema()``
    :ivar path_params: placeholder names derived from ``path_template``
    :ivar cache: how far a response may travel, as a NARROWING of whatever
        the resolved resource already is. defaults to
        :attr:`~threetears.core.http_cache.CacheClass.INHERIT`; see
        :meth:`resolve_cache_class`
    :ivar cache_version_param: the path placeholder carrying the resource's
        version, when the resource is immutable under that version. naming
        one says "a rebuild is a NEW address, not a purge", which is the only
        cache-invalidation scheme a shared edge honours without a purge API.
        must be one of ``path_params``: an HTTP shared cache keys on URL, so
        a version outside the URL cannot key anything
    """

    cache: CacheClass = CacheClass.INHERIT
    cache_version_param: str | None = None

    def __post_init__(self) -> None:
        """normalize the method, then check everything knowable without a schema.

        the schema-dependent half (template placeholders naming real
        properties) needs the tool instance and runs at registration; see
        :meth:`validate_for_schema`.

        :return: nothing
        :rtype: None
        :raises RestAffordanceError: when the method is outside
            :class:`HttpMethod`, when shared cacheability is claimed for a
            method a shared cache may not hold, or when
            ``cache_version_param`` names something absent from the path
        """
        object.__setattr__(self, "method", self.method.upper())
        super().__post_init__()
        if self.method not in {member.value for member in HttpMethod}:
            msg = (
                f"REST affordance declares method {self.method!r}, which is not in the "
                f"closed vocabulary {sorted(member.value for member in HttpMethod)}"
            )
            raise RestAffordanceError(msg)
        if self.cache is not CacheClass.INHERIT and self.cache is not CacheClass.PRIVATE:
            if self.method not in CACHEABLE_METHODS:
                msg = (
                    f"REST affordance declares cache class {self.cache.value!r} on method "
                    f"{self.method!r}; only {sorted(CACHEABLE_METHODS)} have a shared-cache story"
                )
                raise RestAffordanceError(msg)
        if self.cache_version_param is not None and self.cache_version_param not in self.path_params:
            msg = (
                f"REST affordance names cache_version_param {self.cache_version_param!r}, which is "
                f"not a placeholder in path template {self.path_template!r}; a shared cache keys on "
                "URL, so a version outside the URL cannot key it"
            )
            raise RestAffordanceError(msg)

    def bind(self, input_schema: dict[str, Any]) -> ParameterBinding:
        """split a tool's declared properties into path, query and body.

        placeholders in the template bind to path segments; everything else
        rides in the query string for a :data:`QUERY_METHODS` verb and in the
        JSON body otherwise -- the same split the outbound
        :class:`HttpApiTool` performs, read from the same constant.

        :param input_schema: the tool's ``mcp_schema().input_schema``
        :ptype input_schema: dict[str, Any]
        :return: which properties ride where
        :rtype: ParameterBinding
        """
        properties = frozenset(input_schema.get("properties", {}))
        remainder = properties - self.path_params
        if self.binds_remainder_to_query:
            binding = ParameterBinding(path=self.path_params, query=remainder, body=frozenset())
        else:
            binding = ParameterBinding(path=self.path_params, query=frozenset(), body=remainder)
        return binding

    def validate_for_schema(self, input_schema: dict[str, Any], *, tool_name: str) -> None:
        """check the template against the tool's own declared properties.

        the half of intra-tool coherence that needs the tool: a ``{name}``
        naming a property the tool never declares is an address that can be
        parsed and never dispatched.

        :param input_schema: the tool's ``mcp_schema().input_schema``
        :ptype input_schema: dict[str, Any]
        :param tool_name: namespaced tool name, for the failure message
        :ptype tool_name: str
        :return: nothing
        :rtype: None
        :raises RestAffordanceError: when a placeholder names a property the
            tool's input schema does not declare
        """
        properties = frozenset(input_schema.get("properties", {}))
        missing = self.path_params - properties
        if missing:
            msg = (
                f"tool {tool_name!r} declares REST path template {self.path_template!r} whose "
                f"placeholder(s) {sorted(missing)} are not properties of its own mcp_schema(); "
                f"declared properties are {sorted(properties)}"
            )
            raise RestAffordanceError(msg)

    def resolve_cache_class(self, inherited: CacheClass) -> CacheClass:
        """narrow the resource's own classification by this declaration.

        the ONLY sanctioned way to turn a declaration into an effective cache
        class. a declaration wider than the resource is clamped rather than
        honoured, because a tool REST read is per-caller authorized by
        construction and a class attribute must not be able to publish one to
        a shared edge.

        the caller supplies ``inherited`` because 3tears cannot derive it: it
        comes from the resolved resource namespace and its customer scope,
        which only the serving side holds.

        :param inherited: the resolved resource's own classification
        :ptype inherited: CacheClass
        :return: effective class, never more exposed than ``inherited``
        :rtype: CacheClass
        :raises ValueError: when ``inherited`` is
            :attr:`~threetears.core.http_cache.CacheClass.INHERIT`
        """
        return narrow_cache_class(inherited, self.cache)
