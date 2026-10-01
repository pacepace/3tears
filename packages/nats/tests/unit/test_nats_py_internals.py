"""the nats-py private surface ``threetears.nats`` depends on is present in the installed nats-py.

Nothing else fails when it is not: every use of it is behind a call that a health probe, a
subscribe or a pull-consumer stop catches and degrades. A nats-py release that renames one of
these attributes must fail here, by name, before anything ships against it.
"""

from __future__ import annotations

import asyncio
import importlib.metadata

from nats.aio.client import Client
from nats.aio.msg import Msg
from nats.aio.subscription import Subscription

from threetears.nats._nats_py_internals import (  # noqa: SLF001 - module is private by design; this is its test
    PRIVATE_SURFACE,
    PrivateAttribute,
    missing_private_attributes,
    take_queued_messages,
)


def test_the_installed_nats_py_has_every_private_attribute_the_wrapper_uses() -> None:
    """each declared attribute exists, with the declared kind, on a real nats-py object."""
    missing = missing_private_attributes()
    assert not missing, (
        f"nats-py {importlib.metadata.version('nats-py')} no longer provides what "
        f"threetears.nats._nats_py_internals depends on: {missing}. Read the module docstring "
        "before moving the nats-py cap."
    )


def test_the_check_reports_an_attribute_nats_py_does_not_have() -> None:
    """non-vacuity: the check above can fail, for an absent attribute and for a changed kind."""
    surface = (
        PrivateAttribute("Client", "_renamed_in_a_future_release", list),
        PrivateAttribute("Subscription", "_id", str),
        PrivateAttribute("Client", "_pending", "method"),
    )
    assert missing_private_attributes(surface) == [
        "Client._renamed_in_a_future_release: absent",
        "Subscription._id: expected str, found int",
        "Client._pending: no longer a coroutine method",
    ]


def test_the_declared_surface_covers_every_class_the_wrapper_reaches_into() -> None:
    """the surface names the three nats-py classes the wrapper reaches into, and is not empty."""
    assert {attribute.owner for attribute in PRIVATE_SURFACE} == {"Client", "Subscription", "PullSubscription"}


def test_taking_the_queue_keeps_nats_py_pending_bytes_right() -> None:
    """taking queued messages empties the queue and the byte count nats-py tracks beside it."""

    async def scenario() -> None:
        subscription = Subscription(Client(), id=3, subject="inbox")
        for data in (b"one", b"three"):
            msg = Msg(_client=Client(), subject="inbox", data=data)
            subscription._pending_queue.put_nowait(msg)  # noqa: SLF001 -- seeding what the read loop would
            subscription._pending_size += len(data)  # noqa: SLF001 -- seeding what the read loop would

        taken = take_queued_messages(subscription)

        assert [msg.data for msg in taken] == [b"one", b"three"]
        assert subscription._pending_queue.empty()  # noqa: SLF001 -- the property under test
        assert subscription._pending_size == 0  # noqa: SLF001 -- the property under test

    asyncio.run(scenario())
