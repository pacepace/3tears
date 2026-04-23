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

the ``namespace_type`` column keeps its singular form
(``memory`` / ``datasource`` / ``tool`` / ``channel`` / ``shared_agent``
/ ``model`` / ``agent`` / ``shared`` / ``system`` / ``workspace``).
only the ``name`` column moves to the plural-prefix-with-dots shape;
action strings on roles (``memory.read``, ``datasource.read``,
``model.invoke``, ``workspace.read_file_matching:*``) also stay
singular, because action strings are a distinct axis from namespace
names.

this module lives in :mod:`threetears.core` so every downstream
package (``agent-memory``, ``agent-tools``, ``agent-workspace`` on the
3tears side; ``aibots.hub.datasources`` / ``aibots.hub.channels`` /
``aibots.hub.agents`` / ``aibots.hub.gateway`` on the platform side)
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

__all__ = [
    "NAMESPACE_NAME_SEPARATOR",
    "PLURAL_PREFIX_AGENT",
    "PLURAL_PREFIX_CHANNEL",
    "PLURAL_PREFIX_DATASOURCE",
    "PLURAL_PREFIX_MEMORY",
    "PLURAL_PREFIX_MODEL",
    "PLURAL_PREFIX_SHARED",
    "PLURAL_PREFIX_SHARED_AGENT",
    "PLURAL_PREFIX_SYSTEM",
    "PLURAL_PREFIX_TOOL",
    "PLURAL_PREFIX_WORKSPACE",
    "PLURAL_PREFIX_BY_NAMESPACE_TYPE",
    "build_namespace_name",
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
PLURAL_PREFIX_CHANNEL = "channels"
PLURAL_PREFIX_DATASOURCE = "datasources"
PLURAL_PREFIX_MEMORY = "memories"
PLURAL_PREFIX_MODEL = "models"
PLURAL_PREFIX_SHARED = "shared"
PLURAL_PREFIX_SHARED_AGENT = "shared_agents"
PLURAL_PREFIX_SYSTEM = "system"
PLURAL_PREFIX_TOOL = "tools"
PLURAL_PREFIX_WORKSPACE = "workspaces"


#: mapping from singular ``namespace_type`` column value to the
#: plural prefix that leads the canonical name. the full closed set
#: is pinned by hub migration v018's CHECK constraint on
#: ``platform.namespaces.namespace_type`` (expanded by v037 to admit
#: ``model``).
PLURAL_PREFIX_BY_NAMESPACE_TYPE: dict[str, str] = {
    "agent": PLURAL_PREFIX_AGENT,
    "channel": PLURAL_PREFIX_CHANNEL,
    "datasource": PLURAL_PREFIX_DATASOURCE,
    "memory": PLURAL_PREFIX_MEMORY,
    "model": PLURAL_PREFIX_MODEL,
    "shared": PLURAL_PREFIX_SHARED,
    "shared_agent": PLURAL_PREFIX_SHARED_AGENT,
    "system": PLURAL_PREFIX_SYSTEM,
    "tool": PLURAL_PREFIX_TOOL,
    "workspace": PLURAL_PREFIX_WORKSPACE,
}


def sanitize_segment(value: str) -> str:
    """replace any ``.`` in a namespace-name segment with ``-``.

    the canonical name shape uses ``.`` as the separator between the
    plural prefix and each segment; a raw segment value that itself
    contains ``.`` (e.g. ``claude-sonnet-4.5`` as a model name) must
    be sanitized before interpolation so the separator is not
    overloaded. the replacement is one-way — downstream consumers
    read the sanitized value as the namespace-name segment, and no
    code path reconstructs the original dotted form from the segment
    alone.

    :param value: raw segment value (may contain dots)
    :ptype value: str
    :return: sanitized value safe to concatenate with the separator
    :rtype: str
    """
    return value.replace(".", "-")


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
