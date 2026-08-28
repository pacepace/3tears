"""the one rule that decides which tool names a registering pod may claim.

Registration used to answer that question three different ways. A token-bearing
pod's tools were filtered against ``tool_pods.allowed_namespaces`` -- a text
column naming what a pod was *permitted to register*, which is registration
authority rather than ownership. A tokenless pod's tools were not filtered at
all, and that path is every agent's in-process ``ToolServer``, not a rare
in-hub case. A registry constructed with no authenticator filtered nothing
either.

Here the question is asked once, of the namespace GRAPH: **who owns the most
specific provider node that contains this name.** The graph is the same
``namespaces`` rows a grant names and a schema is derived from, so registration
and authorization stop being able to disagree.

The rule, stated once:

* the most specific ``tool_provider`` node containing the offered name decides.
  Containment is :func:`threetears.core.namespaces.namespace_contains`, the one
  implementation, so ``tools.pentest`` reaches ``tools.pentest.sqlmap`` and can
  never reach ``tools.pentestimposter.sqlmap``;
* **most-specific wins, explicitly** rather than by registration order or by
  whichever row the directory happened to list first. ``tools.aibots.admin``
  decides for ``aibots.admin.list_pods`` even when ``tools.aibots`` exists and
  is owned by somebody else, and a pod holding only the parent is refused;
* a name under NO provider node is claimable only by a pod that owns no
  provider node. That is what preserves the old positive filter: a pod bound to
  a provider stays inside it, and cannot wander into territory the graph has
  not spoken about.

**A registry that cannot see the graph enforces nothing, and says so rather than
pretending.** With an empty directory no node contains anything, so an unbound
pod is admitted -- which is exactly what open mode did before. The point of this
module is that the PATH is now one path; its answer differs only because the
data it reads is absent.
"""

from __future__ import annotations

from collections.abc import Iterable

from threetears.core.namespaces import (
    PLURAL_PREFIX_TOOL,
    build_tool_provider_node_name,
    namespace_contains,
)

__all__ = [
    "most_specific_container",
    "rooted_tool_name",
    "tool_is_registrable",
]


def rooted_tool_name(tool_name: str) -> str | None:
    """the canonical namespace name a manifest's tool name sits at, or ``None``.

    A manifest carries BARE mcp names (``pentest.sqlmap``) while a provider node
    is a ``namespaces.name`` (``tools.pentest``). Comparing the two directly
    would never match, so the name is rooted through the one builder --
    :func:`~threetears.core.namespaces.build_tool_provider_node_name`.

    **A name ALREADY rooted at ``tools.`` is refused rather than accepted
    unchanged**, and this is the one place the two sides of the comparison are
    treated differently. An OWNERSHIP entry may legitimately be held either way
    -- as the bare stem an operator wrote or as the canonical row it was
    materialized into -- so the builder accepts both there. A MANIFEST name may
    not: a pod offering ``tools.pentest.sqlmap`` has put a namespace name where an
    mcp name belongs, and admitting it would enter a catalog full name that no
    dispatcher resolves. Refusing it also closes the evasion the other reading
    would open, where a pod owning nothing dodges the containment check by
    pre-rooting the name it wants.

    A name the builder refuses is ``None`` rather than an exception: the value is
    unvalidated publisher text arriving on a network message, and one malformed
    entry must not raise out of a loop over a pod's whole manifest. The caller
    treats ``None`` as not registrable, which is the fail-closed direction.

    :param tool_name: the mcp name as the manifest wrote it
    :ptype tool_name: str
    :return: the rooted canonical name, or ``None`` when the name composes none
    :rtype: str | None
    """
    result: str | None = None
    if not namespace_contains(PLURAL_PREFIX_TOOL, tool_name):
        try:
            result = build_tool_provider_node_name(tool_name)
        except ValueError:
            result = None
    return result


def most_specific_container(provider_nodes: Iterable[str], name: str) -> str | None:
    """the deepest provider node that contains ``name``, or ``None``.

    Two nodes that both contain one name are necessarily one inside the other --
    containment is segment-aware, so the containers of a single name form a
    chain -- which means no two candidates can share a length and there is no
    tie to break. Length is therefore a faithful stand-in for depth here, and
    the comparison needs no segment count.

    :param provider_nodes: every ``tool_provider`` node name the graph holds
    :ptype provider_nodes: Iterable[str]
    :param name: the canonical name being placed
    :ptype name: str
    :return: the most specific containing node, or ``None`` when none contains it
    :rtype: str | None
    """
    result: str | None = None
    for node in provider_nodes:
        if namespace_contains(node, name) and (result is None or len(node) > len(result)):
            result = node
    return result


def tool_is_registrable(
    *,
    tool_name: str,
    owned_nodes: Iterable[str],
    provider_nodes: Iterable[str],
) -> bool:
    """whether the pod owning ``owned_nodes`` may register ``tool_name``.

    See the module docstring for the rule and why each half of it is there.

    ``owned_nodes`` is compared by NAME rather than by pod id deliberately: the
    caller resolved it from the same graph the directory came from, so a name
    appearing in both is a node this pod owns. Keeping pod ids out of the
    comparison keeps the registry from needing an identity it has no way to
    verify.

    :param tool_name: the mcp name the manifest offers
    :ptype tool_name: str
    :param owned_nodes: canonical names of the provider nodes this pod owns
    :ptype owned_nodes: Iterable[str]
    :param provider_nodes: canonical names of every provider node in the graph
    :ptype provider_nodes: Iterable[str]
    :return: whether the name may be registered by this pod
    :rtype: bool
    """
    owned = tuple(owned_nodes)
    rooted = rooted_tool_name(tool_name)
    result = False
    if rooted is not None:
        container = most_specific_container(provider_nodes, rooted)
        result = container in owned if container is not None else not owned
    return result
