"""the one rule that decides which tool names a registering pod may claim.

Registration used to answer that question three different ways. A token-bearing
pod's tools were filtered against the host's own tool-pod row -- a text column
then named ``allowed_namespaces``, naming what a pod was *permitted to
register*, which is registration authority rather than ownership. A tokenless pod's tools were not filtered at
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

**Ownership is half of admission; the other half is who a copy would serve.**
:func:`admit_copy` asks both. A copy that serves EVERY caller -- a Tool Pod's -- is
admitted only from a verified publisher, and under no provider node only from the
platform. A copy an agent serves in-process serves only that agent, so it needs no
platform standing, and still may not sit inside somebody else's provider node.

**A registry with no authenticator enforces nothing, and says so rather than
pretending.** It has no identity and no graph to decide with, so :func:`admit_copy`
admits every well-formed name, and the registration handler logs the mode once at
startup.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from threetears.core.namespaces import (
    PLURAL_PREFIX_TOOL,
    build_tool_provider_node_name,
    namespace_contains,
)
from threetears.nats import Subjects

__all__ = [
    "CopyAudience",
    "PublisherStanding",
    "RefusalCode",
    "admit_copy",
    "audience_of",
    "most_specific_container",
    "rooted_tool_name",
    "tool_is_registrable",
]


class RefusalCode(StrEnum):
    """why the registry refused a copy of a tool, or a whole manifest.

    The wire carries the string value (``RefusedTool.code`` and ``RegistrationResponse.error_code``)
    and a caller branches on it; the accompanying reason is prose for a person.

    :cvar OWNED_ELSEWHERE: the name sits under a provider node this publisher does not own
    :cvar NOT_PLATFORM_SHARED: a copy that would serve every caller, under no provider node, from
        a publisher that is not the platform
    :cvar UNVERIFIED_PUBLISHER: the publisher's identity was not verified -- no credential where
        one is required, or one that failed verification
    :cvar POD_ID_MISMATCH: a VERIFIED publisher registered under a pod id that is not its own; the
        whole manifest is refused
    :cvar INVALID_TOOL_NAME: the name composes no namespace node (empty, or already rooted at
        ``tools.``)
    :cvar OWNERSHIP_GRAPH_UNAVAILABLE: the host could not read the ownership graph; nothing is
        admitted unfiltered, and the pod's next heartbeat retries
    :cvar INVALID_MANIFEST: the manifest itself was malformed or failed validation
    :cvar NO_TOOLS_ADMITTED: every tool the manifest offered was refused; each one's own code is
        in ``refused_tools``
    :cvar PUBLISHER_VERIFICATION_UNAVAILABLE: the host's authenticator could not READ the store it
        verifies publishers against (it raised, rather than answering ``None``). Nothing was
        decided about the publisher, so this is not ``UNVERIFIED_PUBLISHER``: that one says the
        credential failed, this one says the check could not run. Temporary; the pod's next
        heartbeat retries
    :cvar CATALOG_UNAVAILABLE: the manifest was judged, but the catalog could not record the
        outcome -- its durable write failed. Temporary; the pod's next heartbeat retries
    """

    OWNED_ELSEWHERE = "OWNED_ELSEWHERE"
    NOT_PLATFORM_SHARED = "NOT_PLATFORM_SHARED"
    UNVERIFIED_PUBLISHER = "UNVERIFIED_PUBLISHER"
    POD_ID_MISMATCH = "POD_ID_MISMATCH"
    INVALID_TOOL_NAME = "INVALID_TOOL_NAME"
    OWNERSHIP_GRAPH_UNAVAILABLE = "OWNERSHIP_GRAPH_UNAVAILABLE"
    INVALID_MANIFEST = "INVALID_MANIFEST"
    NO_TOOLS_ADMITTED = "NO_TOOLS_ADMITTED"
    PUBLISHER_VERIFICATION_UNAVAILABLE = "PUBLISHER_VERIFICATION_UNAVAILABLE"
    CATALOG_UNAVAILABLE = "CATALOG_UNAVAILABLE"


class CopyAudience(StrEnum):
    """who one copy of a tool would be served to.

    :cvar AGENT: an agent's in-process copy, callable only by that agent
    :cvar EVERYONE: a Tool Pod's copy, callable by every caller
    """

    AGENT = "agent"
    EVERYONE = "everyone"


@dataclass(frozen=True)
class PublisherStanding:
    """what the registry verified about the pod publishing a manifest.

    :param verified: whether the publisher's identity was verified -- a tool pod's token, an
        agent's own identity token naming the agent the pod id names
    :ptype verified: bool
    :param platform_shared: whether the verified publisher is the PLATFORM, which alone may
        serve every caller a name no provider node contains. set by the host's authenticator
        from verified identity, never from anything the manifest says
    :ptype platform_shared: bool
    :param owned_nodes: canonical names of the provider nodes the publisher owns
    :ptype owned_nodes: tuple[str, ...]
    :param enforced: ``False`` only in open mode -- a registry with no authenticator, which has
        no identity or ownership data to decide with and enforces nothing
    :ptype enforced: bool
    """

    verified: bool
    platform_shared: bool
    owned_nodes: tuple[str, ...]
    enforced: bool = True

    @classmethod
    def unenforced(cls) -> PublisherStanding:
        """the standing every publisher has in open mode.

        :return: a standing that admits every well-formed name to every audience
        :rtype: PublisherStanding
        """
        return cls(verified=False, platform_shared=False, owned_nodes=(), enforced=False)


def audience_of(pod_id: str) -> CopyAudience:
    """who a copy registered under ``pod_id`` would serve.

    Read from the pod id with :meth:`~threetears.nats.Subjects.agent_inprocess_owner_id`, the
    same reading routing uses, so what admission judges and what routing serves cannot differ.

    :param pod_id: the registering pod's id, already validated as routable
    :ptype pod_id: str
    :return: the audience
    :rtype: CopyAudience
    :raises ValueError: when ``pod_id`` is dotted but names no agent
    """
    owner = Subjects.agent_inprocess_owner_id(pod_id)
    return CopyAudience.EVERYONE if owner is None else CopyAudience.AGENT


def admit_copy(
    *,
    tool_name: str,
    audience: CopyAudience,
    standing: PublisherStanding,
    provider_nodes: Iterable[str],
) -> RefusalCode | None:
    """whether one publisher may register one copy of ``tool_name`` for ``audience``.

    In order:

    1. a name that composes no node is :attr:`RefusalCode.INVALID_TOOL_NAME`, in every mode;
    2. open mode (``standing.enforced`` false) admits everything else;
    3. an unverified publisher is refused (:attr:`RefusalCode.UNVERIFIED_PUBLISHER`), whatever the
       audience. The 0.55 and 0.56 releases admitted an unverified publisher's agent-scoped copy
       while agents moved to an SDK that signs; 0.57.0 ended that
       (``test_unsigned_agent_concession_expires.py`` holds it);
    4. a name under a provider node needs that node's owner (:attr:`RefusalCode.OWNED_ELSEWHERE`),
       whatever the audience -- an agent owns no provider node, so its own copy of a name inside
       one is refused exactly as :func:`tool_is_registrable` always refused it;
    5. an agent-scoped copy of any other name is admitted: it serves only its own agent;
    6. a copy serving everyone under no provider node needs the platform
       (:attr:`RefusalCode.NOT_PLATFORM_SHARED`).

    :param tool_name: the mcp name the manifest offers
    :ptype tool_name: str
    :param audience: who the copy would serve
    :ptype audience: CopyAudience
    :param standing: what was verified about the publisher
    :ptype standing: PublisherStanding
    :param provider_nodes: canonical names of every provider node in the graph
    :ptype provider_nodes: Iterable[str]
    :return: the refusal, or ``None`` when the copy is admitted
    :rtype: RefusalCode | None
    """
    rooted = rooted_tool_name(tool_name)
    result: RefusalCode | None = None
    if rooted is None:
        result = RefusalCode.INVALID_TOOL_NAME
    elif not standing.enforced:
        result = None
    elif not standing.verified:
        result = RefusalCode.UNVERIFIED_PUBLISHER
    else:
        container = most_specific_container(provider_nodes, rooted)
        if container is not None:
            result = None if container in standing.owned_nodes else RefusalCode.OWNED_ELSEWHERE
        elif audience is CopyAudience.EVERYONE and not standing.platform_shared:
            result = RefusalCode.NOT_PLATFORM_SHARED
    return result


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
