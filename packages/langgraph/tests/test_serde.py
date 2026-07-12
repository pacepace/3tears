"""security + round-trip tests for the checkpoint serializer.

the checkpoint serializer runs in the trusted hub, but checkpoint blobs are
written by sandboxed agents into a shared L2 bucket -- so deserialization must
NEVER reconstruct an arbitrary ``(module, name)`` from a stored blob. these
tests pin the strict-msgpack behavior (a poisoned ext is refused, never
executed) and confirm legitimate langchain/langgraph types still revive.
"""

from __future__ import annotations

import ormsgpack
from langchain_core.messages import AIMessage, HumanMessage

from threetears.langgraph.serde import UUIDSafeSerializer

# langgraph ext type tag for a positional-args constructor ext
# (langgraph.checkpoint.serde.jsonplus.EXT_CONSTRUCTOR_POS_ARGS).
_EXT_CONSTRUCTOR_POS_ARGS = 1


def test_langchain_messages_round_trip() -> None:
    """legitimate message types (in SAFE_MSGPACK_TYPES) survive a round-trip."""
    serializer = UUIDSafeSerializer()
    original = [HumanMessage(content="hello"), AIMessage(content="hi there")]

    tag, data = serializer.dumps_typed(original)
    revived = serializer.loads_typed((tag, data))

    assert isinstance(revived[0], HumanMessage)
    assert isinstance(revived[1], AIMessage)
    assert revived[0].content == "hello"
    assert revived[1].content == "hi there"


def test_poisoned_ext_does_not_execute_code(tmp_path) -> None:
    """a poisoned checkpoint ext (os.system) is refused, never invoked.

    an attacker who can write an L2 checkpoint blob crafts an ext that would
    reconstruct ``os.system("touch <marker>")``. strict msgpack mode blocks the
    unregistered ``(os, system)`` type: the ext-hook returns the raw args instead
    of importing + calling, so the marker file is never created.
    """
    marker = tmp_path / "pwned"
    inner = ormsgpack.packb(("os", "system", (f"touch {marker}",)))
    blob = ormsgpack.packb(ormsgpack.Ext(_EXT_CONSTRUCTOR_POS_ARGS, inner))

    serializer = UUIDSafeSerializer()
    result = serializer.loads_typed(("msgpack", blob))

    # os.system was NEVER invoked -- no side effect on the filesystem
    assert not marker.exists()
    # strict mode returns the raw args (tup[2]) rather than the constructed call;
    # msgpack normalizes the tuple to a list on the wire
    assert result == [f"touch {marker}"]
    assert not isinstance(result, str)


def test_poisoned_ext_for_unregistered_type_is_blocked() -> None:
    """a benign but unregistered type is blocked too (the gate is general).

    ``collections.OrderedDict`` is not in SAFE_MSGPACK_TYPES; under the permissive
    default it would be constructed, under strict mode the ext-hook returns the
    raw args instead.
    """
    inner = ormsgpack.packb(("collections", "OrderedDict", ([("x", 1)],)))
    blob = ormsgpack.packb(ormsgpack.Ext(_EXT_CONSTRUCTOR_POS_ARGS, inner))

    serializer = UUIDSafeSerializer()
    result = serializer.loads_typed(("msgpack", blob))

    # blocked: the raw args come back, not a constructed OrderedDict
    assert not isinstance(result, dict)
