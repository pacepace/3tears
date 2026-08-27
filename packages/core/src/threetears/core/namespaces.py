"""canonical namespace-name builder.

namespace-task-01 phase 9.5 locks the shape of every
``platform.namespaces.name`` value to the canonical form:

    ``{plural_prefix}.<segment1>.<segment2>...``

segments are joined with ``.`` as separator. every segment value is
sanitized through :func:`sanitize_segment` — any ``.`` character in a
raw segment becomes ``-`` before the segment is emitted into the
final name, so model names like ``claude-sonnet-4.5`` round-trip as
``claude-sonnet-4-5`` in the namespace name without collapsing the
segment boundary into the separator.

the ``namespace_type`` column keeps its singular form; the values it
admits, and the plural prefix each one leads its name with, are
:data:`PLURAL_PREFIX_BY_NAMESPACE_TYPE` below rather than a second
list here that can fall behind it.
only the ``name`` column moves to the plural-prefix-with-dots shape;
action strings on roles (``memory.read``, ``datasource.read``,
``model.invoke``, ``workspace.read_file_matching:*``) also stay
singular, because action strings are a distinct axis from namespace
names.

this module lives in :mod:`threetears.core` so every downstream
package (``agent-memory``, ``agent-tools``, ``agent-workspace`` on the
3tears side; ``3tears.hub.datasources`` / ``3tears.hub.channels`` /
``3tears.hub.agents`` / ``3tears.hub.gateway`` on the platform side)
imports the one builder. bespoke string interpolation against the
old colon-separated shape is banned — every per-domain helper
delegates to :func:`build_namespace_name`.

per CLAUDE.md's NO-SHIMS rule: the old
colon-separated-singular-prefix shape disappears from every name
construction site in the same release that lands this builder. there
is NO back-compat parser accepting both shapes. hub migration v040
translates persisted legacy data into the new canonical form at
install time; after v040 runs, the legacy shape does not exist in
the database.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from threetears.nats.subjects import sanitize_subject_segment

__all__ = [
    "HITL_NAMESPACE_TYPE",
    "NAMESPACE_NAME_SEPARATOR",
    "PLURAL_PREFIX_AGENT",
    "PLURAL_PREFIX_API_KEY",
    "PLURAL_PREFIX_AUDIT",
    "PLURAL_PREFIX_CHANNEL",
    "PLURAL_PREFIX_CONVERSATION",
    "PLURAL_PREFIX_CUSTOMER",
    "PLURAL_PREFIX_DATASOURCE",
    "PLURAL_PREFIX_HITL",
    "PLURAL_PREFIX_KNOWLEDGE",
    "PLURAL_PREFIX_MEMORY",
    "PLURAL_PREFIX_MODEL",
    "PLURAL_PREFIX_SHARED",
    "PLURAL_PREFIX_SHARED_AGENT",
    "PLURAL_PREFIX_SYSTEM",
    "PLURAL_PREFIX_TOOL",
    "PLURAL_PREFIX_WORKSPACE",
    "PLURAL_PREFIX_BY_NAMESPACE_TYPE",
    "HitlSessionNamespace",
    "ToolNamespaceName",
    "build_agent_namespace_name",
    "build_hitl_namespace_name",
    "build_namespace_name",
    "build_tool_namespace_name",
    "build_tool_namespace_name_or_none",
    "namespace_contains",
    "parse_tool_namespace_name",
    "sanitize_segment",
]


#: canonical separator between the plural prefix and each sanitized
#: segment. segment boundaries never contain ``.`` because
#: :func:`sanitize_segment` maps every inner ``.`` to ``-`` first.
NAMESPACE_NAME_SEPARATOR = "."


# per-namespace-type plural prefixes. keep this dict sorted by value
# so future readers can eyeball the full taxonomy. the dict is the
# single source of truth for the mapping; per-domain helpers pull
# their prefix from it so adding a new namespace type is a one-line
# change here plus a new helper wrapper.
PLURAL_PREFIX_AGENT = "agents"
PLURAL_PREFIX_API_KEY = "api_keys"
PLURAL_PREFIX_AUDIT = "audits"
PLURAL_PREFIX_CHANNEL = "channels"
PLURAL_PREFIX_CONVERSATION = "conversations"
PLURAL_PREFIX_CUSTOMER = "customers"
PLURAL_PREFIX_DATASOURCE = "datasources"
# DELIBERATE deviation from the pluralized convention, on the same
# grounds as ``knowledge`` directly below: "hitl" is an initialism
# (human in the loop) rather than a countable noun, so a pluralized
# prefix reads as a typo.
PLURAL_PREFIX_HITL = "hitl"
# DELIBERATE deviation from the pluralized convention (``agents`` /
# ``datasources`` / ``memories``): "knowledge" is a mass noun with no
# natural plural, so the prefix stays singular, mirroring the existing
# non-pluralized ``shared`` and ``system`` precedents.
PLURAL_PREFIX_KNOWLEDGE = "knowledge"
PLURAL_PREFIX_MEMORY = "memories"
PLURAL_PREFIX_MODEL = "models"
PLURAL_PREFIX_SHARED = "shared"
PLURAL_PREFIX_SHARED_AGENT = "shared_agents"
PLURAL_PREFIX_SYSTEM = "system"
PLURAL_PREFIX_TOOL = "tools"
PLURAL_PREFIX_WORKSPACE = "workspaces"


#: singular ``namespace_type`` column value carried by the rows a
#: human-in-the-loop display session is authorized against. published
#: as a named constant, following the ``DATASOURCE_NAMESPACE_TYPE`` /
#: ``MEMORY_NAMESPACE_TYPE`` / ``INTENTION_NAMESPACE_TYPE`` precedent,
#: because the row is written by a consumer of this package rather
#: than by anything here, so the value crosses a distribution boundary
#: and both sides have to read it from one place.
HITL_NAMESPACE_TYPE = "hitl"


#: mapping from singular ``namespace_type`` column value to the
#: plural prefix that leads the canonical name. the full closed set
#: is pinned by the CHECK constraint on
#: ``platform.namespaces.namespace_type``; in the 3tears hub repo that
#: constraint lives in the squashed init at
#: ``v001_initial_schema.sql`` (widened to admit ``knowledge`` by
#: ``v011_knowledge_substrate``).
PLURAL_PREFIX_BY_NAMESPACE_TYPE: dict[str, str] = {
    "agent": PLURAL_PREFIX_AGENT,
    "api_key": PLURAL_PREFIX_API_KEY,
    "audit": PLURAL_PREFIX_AUDIT,
    "channel": PLURAL_PREFIX_CHANNEL,
    "conversation": PLURAL_PREFIX_CONVERSATION,
    "customer": PLURAL_PREFIX_CUSTOMER,
    "datasource": PLURAL_PREFIX_DATASOURCE,
    HITL_NAMESPACE_TYPE: PLURAL_PREFIX_HITL,
    "knowledge": PLURAL_PREFIX_KNOWLEDGE,
    "memory": PLURAL_PREFIX_MEMORY,
    "model": PLURAL_PREFIX_MODEL,
    "shared": PLURAL_PREFIX_SHARED,
    "shared_agent": PLURAL_PREFIX_SHARED_AGENT,
    "system": PLURAL_PREFIX_SYSTEM,
    "tool": PLURAL_PREFIX_TOOL,
    "workspace": PLURAL_PREFIX_WORKSPACE,
}


def sanitize_segment(value: str | UUID) -> str:
    """replace any ``.`` in a namespace-name segment with ``-``.

    the canonical name shape uses ``.`` as the separator between the
    plural prefix and each segment; a raw segment value that itself
    contains ``.`` (e.g. ``claude-sonnet-4.5`` as a model name) must
    be sanitized before interpolation so the separator is not
    overloaded. the replacement is one-way — downstream consumers
    read the sanitized value as the namespace-name segment, and no
    code path reconstructs the original dotted form from the segment
    alone.

    delegates to :func:`threetears.nats.sanitize_subject_segment`, the
    ONE implementation of this rule. the copies this replaced (a
    private ``_sanitize`` in ``subjects.py``, a private ``_seg`` in
    ``subject_permissions.py``) had drifted apart from nothing except
    being three, and the grant side could reach neither this one nor
    each other's without a Shape-A underscore violation.

    **the delegation is a coincidence of rules, not one rule.** the callee's
    contract is "safe as a NATS subject token"; this function's is "safe as a
    ``platform.namespaces.name`` segment", and those values are PERSISTED, so a
    change to the sanitizer rewrites what new rows key on while old rows keep the
    old shape. the two rules agree on every character today, which is why sharing
    the implementation is right. if the subject rule ever widens -- a new NATS
    token restriction, or a relaxation -- do NOT follow it here by inheritance:
    fork this function, keep the persisted shape, and say why. the dedup exists to
    stop three copies drifting silently, not to make a storage-key rule track a
    wire-format rule automatically.

    :param value: raw segment value (may contain dots); a UUID renders
        as its 36-character canonical string
    :ptype value: str | UUID
    :return: sanitized value safe to concatenate with the separator
    :rtype: str
    """
    return sanitize_subject_segment(value)


def build_namespace_name(plural_prefix: str, *segments: str) -> str:
    """build a canonical namespace name from plural prefix + segments.

    every segment is passed through :func:`sanitize_segment` before
    interpolation. the final form is
    ``{plural_prefix}.<seg1>.<seg2>...``. callers supply the plural
    prefix via the ``PLURAL_PREFIX_*`` constants (or the
    :data:`PLURAL_PREFIX_BY_NAMESPACE_TYPE` lookup) rather than
    string-literal constants so a future prefix rename is a one-line
    change in this module.

    :param plural_prefix: per-type plural prefix (e.g. ``models``,
        ``workspaces``); typically one of the ``PLURAL_PREFIX_*``
        module constants
    :ptype plural_prefix: str
    :param segments: ordered segment values (each sanitized before
        interpolation)
    :ptype segments: str
    :return: canonical dot-separated namespace name
    :rtype: str
    """
    sanitized_segments = [sanitize_segment(s) for s in segments]
    parts = [plural_prefix, *sanitized_segments]
    return NAMESPACE_NAME_SEPARATOR.join(parts)


def build_agent_namespace_name(agent_id: UUID) -> str:
    """build the canonical name of an agent's OWN namespace row.

    shape is ``agents.<canonical uuid>``. the uuid renders through
    :func:`sanitize_segment`, which leaves it untouched -- the hyphens
    in ``xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`` are not separators and
    the value carries no ``.``.

    this lives here, beside the generic builder, because it is asked
    from BOTH sides of an ownership question and the two sides must
    agree exactly. the hub WRITES this name when it provisions an
    agent; the rbac evaluator READS it to find out which namespace row
    a calling agent IS, so it can answer whether that row owns the
    namespace under evaluation. a second spelling of the rule in either
    place makes every ownership comparison miss, and it misses
    SILENTLY -- an agent simply stops being recognised as the owner of
    its own storage.

    :param agent_id: agent unique identifier
    :ptype agent_id: UUID
    :return: canonical namespace name in the form ``agents.<uuid>``
    :rtype: str
    """
    # convert at border: namespace name segment
    return build_namespace_name(PLURAL_PREFIX_AGENT, str(agent_id))


def namespace_contains(node: str, name: str) -> bool:
    """true iff ``name`` is ``node`` itself or lives beneath it.

    the ONE containment rule for dot-segmented hierarchical names in
    this codebase -- ``platform.namespaces.name`` values, and the mcp
    names those are built from, which share the shape. every other
    place that used to ask "is this name under that one" with a raw
    prefix test delegates here.

    the rule is::

        name == node or name.startswith(node + NAMESPACE_NAME_SEPARATOR)

    which is segment-aware BY CONSTRUCTION rather than by convention:
    the only character that may follow the node is the separator
    itself, so ``pentest`` reaches ``pentest.sqlmap`` and can never
    reach ``pentestimposter.sqlmap``. a raw ``name.startswith(node)``
    reaches both, which is why every ``allowed_namespaces`` value used
    to be written with a trailing dot -- a value-level workaround for a
    gap in the comparison. the workaround is not accommodated here: a
    node written ``pentest.`` matches nothing, so the old shape fails
    visibly instead of silently.

    **comparison is exact.** no case folding, no whitespace stripping,
    no tolerance for a trailing separator. a node carrying stray
    whitespace or the wrong case therefore grants NOTHING, which is the
    fail-closed direction; normalizing it would make the value grant
    more than it says while its author reads it as written. refusing a
    malformed node belongs at write time -- the platform's
    ``role_assignments_scope_shape`` CHECK does that -- because this
    function runs inside the rbac evaluator's inner loop, where raising
    would turn one bad row into a failure of every authorization
    question the caller asks.

    an empty ``node`` contains NOTHING. under a raw prefix test it
    would contain every name there is, so the emptiest possible value
    would be the widest possible grant; the explicit refusal is what
    stops that.

    :param node: the container -- a name, or a bare plural prefix
        (``tools``), written WITHOUT a trailing separator
    :ptype node: str
    :param name: the candidate name being tested for membership
    :ptype name: str
    :return: whether ``name`` is ``node`` or sits beneath it
    :rtype: bool
    """
    result = False
    if node and name:
        result = name == node or name.startswith(node + NAMESPACE_NAME_SEPARATOR)
    return result


#: leading component every tool namespace name carries, and the one
#: :func:`build_hitl_namespace_name` swaps out. structural rather than
#: conventional: :func:`build_tool_namespace_name` is the one place the
#: prefix is applied, so its absence means the caller passed something
#: that is not a tool namespace name.
_TOOL_NAME_PREFIX = PLURAL_PREFIX_TOOL + NAMESPACE_NAME_SEPARATOR


@dataclass(frozen=True, slots=True)
class ToolNamespaceName:
    """the components :func:`parse_tool_namespace_name` recovers.

    the two fields are NOT symmetric in what they promise, and the
    naming says so. ``mcp_name`` is EXACT -- it is the value the
    builder was handed, character for character, which is what
    un-flattening the name bought. ``version_segment`` is the
    SANITIZED version as it appears in the name, and the natural
    version it came from is NOT recoverable from it: ``1-0-0`` is
    equally the sanitization of ``1.0.0`` and a literal ``1-0-0``.

    a caller that needs the natural version reads it from the row's
    ``metadata``, where the emitter persists it beside the mcp name.
    the field is called ``version_segment`` rather than ``version`` so
    that using it as a semver string reads wrong at the call site.

    :ivar mcp_name: the tool's mcp name, exactly as built
    :ivar version_segment: the sanitized version component
    """

    mcp_name: str
    version_segment: str


def build_tool_namespace_name(mcp_name: str, version: str) -> str:
    """build the canonical ``platform.namespaces.name`` for a tool row.

    the shape is ``tools.<mcp_name>.<sanitized version>``. the mcp name
    is interpolated VERBATIM -- its dots become segment boundaries
    rather than being flattened to hyphens -- and the version is
    sanitized and placed last.

    **the mcp name is not sanitized, and that is the whole point.**
    a flattened name renders ``pentest.sqlmap`` as
    ``tools.pentest-sqlmap.<version>``, which is not under
    ``tools.pentest`` by any segment-aware rule, so a subtree grant on
    a provider node reaches none of its tools and
    :func:`namespace_contains` has nothing to bite on. flattening also
    COLLIDES: ``a.b`` and ``a-b`` render one name, so two distinct
    tools share one row and one of them stops being addressable.

    **the price, stated: the separator is now overloaded**, so the name
    can no longer be split into a fixed number of components. the
    grammar stays unambiguous only because the version is sanitized and
    is ALWAYS the last component, which is what lets
    :func:`parse_tool_namespace_name` recover the split right to left.
    that invariant is the reason the two functions ship together and
    are pinned against each other by an enforcement test.

    **bulk deregistration by ``LIKE`` is now a SUBTREE sweep, not a
    version sweep.** under the flattened shape
    ``LIKE 'tools.<flattened-mcp>.%'`` could only match one tool's
    versions, because no other row shared that component. under this
    shape ``LIKE 'tools.a.b.%'`` matches ``tools.a.b.<version>`` AND
    every version of a distinct tool named ``a.b.c``, so a
    deregistration written that way deletes rows belonging to another
    tool. delete by the exact name, or by ``(mcp_name, version)`` read
    from ``metadata``; do not compose a prefix ``LIKE``.

    :param mcp_name: tool mcp name (e.g. ``pentest.sqlmap``), which may
        contain ``.`` and is interpolated unchanged
    :ptype mcp_name: str
    :param version: tool version (e.g. ``1.0`` or ``1.0.0``), sanitized
        into the final component
    :ptype version: str
    :return: canonical namespace name string
    :rtype: str
    :raises ValueError: if ``mcp_name`` or ``version`` is empty, if
        ``mcp_name`` carries an empty component, or if ``mcp_name`` is
        already rooted at the ``tools`` prefix
    """
    if not mcp_name:
        raise ValueError("mcp_name must be non-empty")
    if not version:
        raise ValueError("version must be non-empty")
    # an already-rooted value is the caller passing a BUILT name back
    # in. doubling it would mint ``tools.tools.*``, a row nothing
    # resolves and nothing complains about, so refuse at the one layer
    # that applies the prefix.
    if namespace_contains(PLURAL_PREFIX_TOOL, mcp_name):
        raise ValueError(
            f"mcp_name is already rooted at {PLURAL_PREFIX_TOOL!r}: {mcp_name!r}",
        )
    if not all(mcp_name.split(NAMESPACE_NAME_SEPARATOR)):
        raise ValueError(
            f"mcp_name has an empty component: {mcp_name!r}",
        )
    return NAMESPACE_NAME_SEPARATOR.join(
        (PLURAL_PREFIX_TOOL, mcp_name, sanitize_segment(version)),
    )


def build_tool_namespace_name_or_none(mcp_name: str, version: str) -> str | None:
    """build a tool namespace name, or ``None`` when the pair is malformed.

    :func:`build_tool_namespace_name` REFUSES a malformed pair, which is
    the right contract where the pair is AUTHORED -- an emitter, a seed,
    a migration -- because a name nobody can build is a bug at its
    source and should say so.

    It is the wrong contract where the pair arrives from OUTSIDE: a URL
    path segment split into candidates, a dispatch envelope's tool name,
    an operator-typed column. There a malformed value is an ordinary
    "no such tool", and raising turns a 404 into a 500 and turns a DENY
    on the tool-dispatch hot path into an exception. Both are worse
    answers than a miss, and the second is worse than the first.

    So the catch is spelled ONCE, here, rather than at each of those
    call sites -- where one of them would eventually be written as a
    bare ``except``, or forgotten entirely and found by whoever sends
    the first request carrying a doubled dot.

    ``None`` rather than an empty string: an empty name is a lookup
    value that means something (and :func:`namespace_contains` reads an
    empty node as containing nothing), so it cannot double as "there is
    no name".

    :param mcp_name: tool mcp name, from an untrusted or arbitrary source
    :ptype mcp_name: str
    :param version: tool version, from an untrusted or arbitrary source
    :ptype version: str
    :return: the canonical namespace name, or ``None`` when the pair
        cannot compose one
    :rtype: str | None
    """
    result: str | None = None
    try:
        result = build_tool_namespace_name(mcp_name, version)
    except ValueError:
        result = None
    return result


def parse_tool_namespace_name(name: str) -> ToolNamespaceName:
    """split a tool namespace name back into its components.

    the inverse of :func:`build_tool_namespace_name` for the mcp name,
    and a LOSSY inverse for the version -- see
    :class:`ToolNamespaceName` for which half promises what.

    the split runs RIGHT TO LEFT: the last component is the version,
    and everything between the ``tools.`` prefix and it is the mcp
    name, rejoined on the separator. that direction is not a
    convenience. the mcp name may carry any number of dots, so counting
    components from the left cannot find the boundary; the version
    carries none, because the builder sanitizes it, so counting one
    component from the right always can.

    :param name: a canonical tool namespace name
    :ptype name: str
    :return: the recovered mcp name and sanitized version component
    :rtype: ToolNamespaceName
    :raises ValueError: if ``name`` is not a strict descendant of the
        ``tools`` prefix, carries fewer than two components under it,
        or carries an empty component
    """
    # a STRICT descendant: the bare prefix is not a tool namespace
    # name, and neither is a provider node, so
    # :func:`namespace_contains` -- which counts a node as containing
    # itself -- is composed with an inequality.
    if not (namespace_contains(PLURAL_PREFIX_TOOL, name) and name != PLURAL_PREFIX_TOOL):
        raise ValueError(
            f"name must start with {_TOOL_NAME_PREFIX!r}, got {name!r}",
        )
    components = name[len(_TOOL_NAME_PREFIX) :].split(NAMESPACE_NAME_SEPARATOR)
    if not all(components):
        raise ValueError(
            f"name has an empty component: {name!r}",
        )
    if len(components) < 2:
        # one component under the prefix is a PROVIDER node
        # (``tools.pentest``), which names no tool and carries no
        # version. reading it as one would hand back an empty mcp name
        # and a version of ``pentest``.
        raise ValueError(
            f"name must carry at least two components under {_TOOL_NAME_PREFIX!r} "
            f"(an mcp name and a version), got {name!r}",
        )
    return ToolNamespaceName(
        mcp_name=NAMESPACE_NAME_SEPARATOR.join(components[:-1]),
        version_segment=components[-1],
    )


def build_hitl_namespace_name(tool_namespace_name: str, customer_id: UUID) -> str:
    """build the namespace name a human-in-the-loop session authorizes against.

    the shape is the serving tool's namespace name with its plural
    prefix swapped for :data:`PLURAL_PREFIX_HITL` and the customer
    appended, so ``tools.scrape.zone_alpha.1-0-0`` and a customer give
    ``hitl.scrape.zone_alpha.1-0-0.<32 hex chars>``. the tool
    components are lifted verbatim out of the tool namespace name
    rather than rebuilt from an mcp name and a version, so the two
    rows cannot disagree about how a name was sanitized.

    the tool half carries as many components as the mcp name has, and
    they are splatted through rather than counted: an mcp name is
    interpolated unflattened by
    :func:`build_tool_namespace_name`, so ``a.b.c`` contributes three
    components where ``a`` contributes one. nothing here depends on
    how many there are.

    **both halves of the name are load-bearing, and each closes a
    different leak.**

    the TOOL half carries the network zone, because a zone is a
    distinct registered tool rather than a pod attribute. a name built
    from the customer alone isolates tenants while leaving every zone
    reachable by anyone entitled to that tenant anywhere, so an
    operator granted a customer on the general internet would reach
    that same customer's display inside a firewalled network. the same
    argument at a finer grain is why that half keeps the tool's
    VERSION: ``tools.<mcp>.<version>`` is its own row with its own
    assignments, so a name that dropped the version would let a
    version bump leave somebody able to call a tool but not to attach
    to the display it raised, or the reverse.

    the CUSTOMER half is what makes the row tenant-scoped, and the
    tool's own namespace row is not a substitute: those materialize
    with ``customer_id`` NULL, and the shipped evaluator counts a
    customer-scoped membership or group only when its customer matches
    the namespace's. against a NULL-customer row every customer-scoped
    group is therefore dropped, so the only grant that survives there
    is one held by a PLATFORM-scoped group, which is not one tenant's.
    a per-tenant entitlement is not expressible on a tool namespace
    at all.

    the tool components stay READABLE here rather than becoming the
    digest the owner-routed subjects use. a namespace name is a
    database row value a human administers, so the wildcard-injection
    argument that put a digest in a subject token does not reach it.
    this is the same split ``Subjects.room`` in ``3tears-nats``
    already makes: the digest goes in the subject, the raw identity
    rides the envelope.

    the customer is spelled as its FULL hex, unlike the eight-character
    forms ``memories.`` and ``intentions.`` names use. those rows are
    resolved by an (owner agent, customer) pair and their names are
    display handles; this one is resolved BY NAME, so a truncated hex
    is a real chance of two tenants sharing one row, which is the
    exposure the name exists to prevent.

    two sessions share a name exactly when they share both the tool
    namespace name and the customer. the mcp name behind it no longer
    collapses -- :func:`build_tool_namespace_name` interpolates it
    verbatim, so ``a.b`` and ``a-b`` are distinct rows -- but the
    VERSION still passes through :func:`sanitize_segment`, so two tool
    rows whose versions differ only by a ``.`` against a ``-`` share
    one name. that is a property of the tool row rather than something
    this builder could repair.

    :param tool_namespace_name: the serving tool's canonical namespace
        name, as ``threetears.agent.tools.server.tool_namespace_name``
        builds it
    :ptype tool_namespace_name: str
    :param customer_id: the customer whose session this is
    :ptype customer_id: UUID
    :return: canonical hitl namespace name
    :rtype: str
    :raises ValueError: if ``tool_namespace_name`` does not carry the
        ``tools.`` prefix, or if any component after that prefix is
        empty
    """
    # a STRICT descendant of ``tools``: the bare prefix is not itself a
    # tool namespace name, so :func:`namespace_contains` -- which counts
    # a node as containing itself -- is composed with an inequality
    # rather than replaced by a second prefix test.
    if not (namespace_contains(PLURAL_PREFIX_TOOL, tool_namespace_name) and tool_namespace_name != PLURAL_PREFIX_TOOL):
        raise ValueError(
            f"tool_namespace_name must start with {_TOOL_NAME_PREFIX!r}, got {tool_namespace_name!r}",
        )
    tool_components = tool_namespace_name[len(_TOOL_NAME_PREFIX) :].split(NAMESPACE_NAME_SEPARATOR)
    if not all(tool_components):
        # an empty component would render two consecutive separators,
        # leaving a name whose customer is no longer the last thing
        # after a dot. refuse rather than mint it.
        raise ValueError(
            f"tool_namespace_name has an empty component: {tool_namespace_name!r}",
        )
    return build_namespace_name(PLURAL_PREFIX_HITL, *tool_components, customer_id.hex)


@dataclass(frozen=True, slots=True)
class HitlSessionNamespace:
    """which namespace a human-in-the-loop display session authorizes against.

    built by whatever terminates an operator's connection, out of the
    two facts it is the first thing on the path to hold together: the
    serving tool's namespace name, which the pod publishes on its
    attach reply, and the customer being served, which the pod never
    sees. it STATES a namespace and evaluates nothing, because this
    package cannot see the roles, the groups or the assignments; the
    isolating is done by the platform's own evaluator reading the row
    this names. anything here that looked like a permission check
    would be a second authorization concept beside the shipped one.

    one row per (serving tool, customer), NOT one per session. every
    session a customer runs against one zone at one tool version
    resolves to the same namespace, which is what makes a single role
    assignment the durable grant rather than something minted per
    display.

    the tool namespace name is validated at construction, so an
    unusable descriptor cannot sit in a session waiting to fail at
    the moment somebody attaches.

    :ivar tool_namespace_name: the serving tool's canonical namespace
        name; the zone and the version both live in it
    :ivar customer_id: the customer whose session this is
    """

    tool_namespace_name: str
    customer_id: UUID

    def __post_init__(self) -> None:
        """reject a tool namespace name the builder could not use.

        :raises ValueError: per :func:`build_hitl_namespace_name`
        """
        build_hitl_namespace_name(self.tool_namespace_name, self.customer_id)

    @property
    def namespace_name(self) -> str:
        """canonical name of the row to authorize against.

        :return: the name :func:`build_hitl_namespace_name` renders
        :rtype: str
        """
        return build_hitl_namespace_name(self.tool_namespace_name, self.customer_id)

    @property
    def namespace_type(self) -> str:
        """``namespace_type`` column value the named row carries.

        :return: :data:`HITL_NAMESPACE_TYPE`
        :rtype: str
        """
        return HITL_NAMESPACE_TYPE
