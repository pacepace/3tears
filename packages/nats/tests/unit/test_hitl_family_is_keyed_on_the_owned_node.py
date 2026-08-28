"""The human-in-the-loop family is keyed on the node a pod OWNS, not on a tool leaf.

**The defect this closes, stated as the two values that never met.** A tool pod's
human-in-the-loop grants are minted at CONNECT, from the provider nodes recorded for
its ``tool_pods`` row -- and that column holds tool-name NODES (``pentest``,
``aibots.admin``), compared on a segment boundary by
:func:`threetears.core.namespaces.namespace_contains`. The pod's own consumer, meanwhile,
derived its family from the tool's REGISTERED namespace name
(``tools.pentest.sqlmap.1-0-0``), which does not exist at connect time and is not a prefix
of anything the row holds. The family is a SHA-256 digest, so the two produced different
subjects and the mismatch is invisible: the pod subscribes successfully and receives
nothing, forever.

**Why the family is keyed on the node rather than on the leaf.** The node is the only value
both ends can hold. The grant side has it at connect (it is the row). The pod learns the
canonical form of it -- ``tools.<node>``, the provider namespace it owns -- on its
registration reply. A leaf name cannot be reduced to its node by any rule: ``aibots.admin``
is two components and ``pentest`` is one, so ``tools.aibots.admin.thing.1-0-0`` and
``tools.pentest.sqlmap.1-0-0`` are the same shape and split differently.

So :meth:`Subjects.tool_provider_node` roots a bare node at the ``tools.`` prefix and leaves
an already-rooted one alone, and every builder that digests a tool applies it. That is what
lets the grant (built from the stem) and the consumer (holding the canonical name) derive
one family.
"""

from __future__ import annotations

import pytest

from threetears.nats.subject_permissions import Principal, build_permissions
from threetears.nats.subjects import Subjects, set_default_namespace

_NS = "3tears"

#: the node as a pod's own declaration holds it: no ``tools.`` prefix, no version,
#: no trailing separator. this is what the auth callout hands ``build_permissions``.
_STEM = "pentest"

#: a MULTI-COMPONENT node, because that is the case no leaf-side rule can recover. a pod
#: authorized at ``aibots.admin`` serves ``tools.aibots.admin.<tool>.<version>``, and
#: nothing in that name says where the node ends and the tool begins.
_DEEP_STEM = "aibots.admin"

#: the canonical namespace name of the provider node the pod OWNS -- the value its
#: registration reply carries back, and the only form its consumer ever sees.
_OWNED_NODE = f"tools.{_STEM}"

#: a tool the pod actually serves, under that node. deliberately NOT the family key: it is
#: unknown at connect, and one pod serves many of them.
_LEAF = f"tools.{_STEM}.sqlmap.1-0-0"

_POD = "01947100-0000-7000-8000-0000000000f1"
_SESSION = "session-42"


@pytest.fixture(autouse=True)
def _bind_namespace() -> None:
    """Bind the subject namespace so every rendered subject carries a known prefix."""
    set_default_namespace(_NS)


class TestTheNodeAndItsStemDeriveOneFamily:
    """rooting is what makes the mint's value and the consumer's value the same string."""

    def test_the_stem_and_the_rooted_node_derive_one_forward_family(self) -> None:
        """the mint holds ``pentest``; the pod holds ``tools.pentest``; one family.

        this is the whole re-key. before it the two derived different digests, and a
        different digest is a different subject -- which is a subscription that receives
        nothing rather than an error anybody sees.

        :return: none
        :rtype: None
        """
        assert Subjects.hitl_forward_family(_STEM) == Subjects.hitl_forward_family(_OWNED_NODE)

    def test_the_stem_and_the_rooted_node_derive_one_pipe_family(self) -> None:
        """the display stream rides its own family and must root identically.

        a session's control plane and its display stream are owner-routed on the same key
        and must derive DIFFERENT families (``serve_owner`` queue-groups on the subject),
        but each family must still be one string across the two processes.

        :return: none
        :rtype: None
        """
        assert Subjects.hitl_pipe_family(_STEM) == Subjects.hitl_pipe_family(_OWNED_NODE)

    def test_the_two_families_remain_distinct(self) -> None:
        """rooting must not collapse the control plane onto the display stream.

        :return: none
        :rtype: None
        """
        assert Subjects.hitl_forward_family(_STEM) != Subjects.hitl_pipe_family(_STEM)

    def test_rooting_is_idempotent_rather_than_doubling(self) -> None:
        """an already-rooted node must not become ``tools.tools.pentest``.

        both forms reach these builders -- the mint passes the stem, the consumer passes
        the canonical name -- so doubling one of them re-opens the mismatch by the other
        door.

        :return: none
        :rtype: None
        """
        assert Subjects.tool_provider_node(_STEM) == _OWNED_NODE
        assert Subjects.tool_provider_node(_OWNED_NODE) == _OWNED_NODE

    def test_a_multi_component_node_roots_whole(self) -> None:
        """``aibots.admin`` is ONE node, and its dots are segment boundaries, not data.

        :return: none
        :rtype: None
        """
        assert Subjects.tool_provider_node(_DEEP_STEM) == f"tools.{_DEEP_STEM}"
        assert Subjects.hitl_forward_family(_DEEP_STEM) == Subjects.hitl_forward_family(f"tools.{_DEEP_STEM}")

    def test_an_empty_node_is_refused(self) -> None:
        """an empty node would collapse every pod onto one family.

        :return: none
        :rtype: None
        """
        with pytest.raises(ValueError, match="owned_node must be non-empty"):
            Subjects.tool_provider_node("")

    def test_a_node_that_is_only_the_prefix_is_refused(self) -> None:
        """``tools`` alone is the whole tool tree, not a node anybody owns.

        rooting it would yield the bare prefix, and a family derived from it would be
        shared by every provider on the platform -- so a pod holding it would take a SHARE
        of every other pod's session traffic through ``serve_owner``'s queue group.

        :return: none
        :rtype: None
        """
        with pytest.raises(ValueError, match="names the whole tool tree"):
            Subjects.tool_provider_node("tools")


class TestTheGrantAdmitsWhatTheOwnedNodeAddresses:
    """the property the grant exists for, asserted across the process boundary it spans."""

    def _pod(self, *stems: str) -> object:
        """permissions for a tool pod whose row authorizes ``stems``.

        :param stems: declared provider stems, exactly as the column holds them
        :ptype stems: str
        :return: the resolved allow-list
        :rtype: PrincipalPermissions
        """
        return build_permissions(Principal.TOOL_POD, pod_id=_POD, tool_namespaces=stems)

    def test_the_control_plane_grant_admits_the_owned_nodes_subject(self) -> None:
        """minted from the stem, addressed by the owned node, one subject.

        :return: none
        :rtype: None
        """
        perm = self._pod(_STEM)
        granted = str(Subjects.forward_scoped_wildcard(Subjects.hitl_forward_family(_OWNED_NODE)))
        assert granted in perm.subscribe  # type: ignore[attr-defined]
        served = Subjects.forward_scoped(Subjects.hitl_forward_family(_OWNED_NODE), _SESSION)
        assert served.path.rsplit(".", 1)[0] == granted.rsplit(".", 1)[0]

    def test_the_display_stream_grant_admits_the_owned_nodes_subject(self) -> None:
        """the pipe family half of the same property.

        :return: none
        :rtype: None
        """
        perm = self._pod(_STEM)
        granted = str(Subjects.forward_scoped_wildcard(Subjects.hitl_pipe_family(_OWNED_NODE)))
        assert granted in perm.subscribe  # type: ignore[attr-defined]

    def test_the_byte_pipe_grant_admits_the_owned_nodes_stream(self) -> None:
        """the pipe SUBJECT digests the tool too, so it re-keys with the families.

        the grant names the pod's own half exactly and wildcards only the per-attach
        nonce; the owner renders the concrete subject from the node it owns.

        :return: none
        :rtype: None
        """
        perm = self._pod(_STEM)
        granted = str(Subjects.pipe_pod_wildcard(_OWNED_NODE, _POD, "down"))
        assert granted in perm.publish  # type: ignore[attr-defined]
        concrete = Subjects.pipe(_OWNED_NODE, _POD, "nonce-1", "down")
        assert concrete.path.rsplit(".", 2)[0] == granted.rsplit(".", 2)[0]

    def test_a_multi_component_node_is_granted_whole(self) -> None:
        """the case a leaf-side split gets wrong, asserted end to end.

        :return: none
        :rtype: None
        """
        perm = self._pod(_DEEP_STEM)
        granted = str(Subjects.forward_scoped_wildcard(Subjects.hitl_forward_family(f"tools.{_DEEP_STEM}")))
        assert granted in perm.subscribe  # type: ignore[attr-defined]

    def test_the_leaf_name_is_not_the_key_and_holds_no_grant(self) -> None:
        """a consumer that still passes the registered tool name is granted NOTHING.

        recorded as an assertion rather than as prose because it is the pre-existing
        behaviour: the leaf digest is a family this pod does not hold, and subscribing it
        is silent. anything reaching for the leaf is reaching for a dead subject.

        :return: none
        :rtype: None
        """
        perm = self._pod(_STEM)
        leaf = str(Subjects.forward_scoped_wildcard(Subjects.hitl_forward_family(_LEAF)))
        assert leaf not in perm.subscribe  # type: ignore[attr-defined]

    def test_one_nodes_grant_is_not_anothers(self) -> None:
        """the family segment still separates providers after the re-key.

        :return: none
        :rtype: None
        """
        perm = self._pod(_STEM)
        peer = str(Subjects.forward_scoped_wildcard(Subjects.hitl_forward_family("tools.other-provider")))
        assert peer not in perm.subscribe  # type: ignore[attr-defined]
