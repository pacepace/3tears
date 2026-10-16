"""the nats-py private surface ``threetears.nats`` depends on is present in the installed nats-py.

Nothing else fails when it is not: every use of it is behind a call that a health probe, a
subscribe or a pull-consumer stop catches and degrades. A nats-py release that renames one of
these attributes must fail here, by name, before anything ships against it.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import string
from datetime import timedelta

from nats.aio.client import Client

from threetears.nats._nats_py_internals import (
    NUID_LENGTH,
    PRIVATE_SURFACE,
    CsprngNuid,
    PrivateAttribute,
    missing_private_attributes,
    take_queued_messages,
)

from .wire_server import wire_server


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
    """taking queued messages empties the queue and the byte count nats-py tracks beside it.

    The messages arrive the way they do in production: a server delivers them to a subscription
    with no callback, and nats-py's read loop queues them and counts their bytes. The queue and the
    count are then read through nats-py's public ``pending_msgs`` and ``pending_bytes``.
    """

    async def scenario() -> None:
        async with wire_server() as server:
            connection = Client()
            await connection.connect(server.url, allow_reconnect=False)
            try:
                subscription = await connection.subscribe("inbox")
                await connection.flush()
                for data in (b"one", b"three"):
                    await server.deliver("inbox", data)
                async with asyncio.timeout(2.0):
                    while subscription.pending_msgs < 2:
                        await asyncio.sleep(0.01)
                assert subscription.pending_bytes == len(b"one") + len(b"three")

                taken = take_queued_messages(subscription)

                assert [msg.data for msg in taken] == [b"one", b"three"]
                assert subscription.pending_msgs == 0
                assert subscription.pending_bytes == 0
            finally:
                await connection.close()

    asyncio.run(scenario())


_BASE62 = set((string.digits + string.ascii_uppercase + string.ascii_lowercase).encode())


def test_a_csprng_token_is_what_nats_py_slices_and_extends() -> None:
    """22 base62 bytes, in a fresh mutable bytearray: nats-py slices at 22 and extends the result."""
    nuid = CsprngNuid()
    first, second = nuid.next(), nuid.next()

    assert isinstance(first, bytearray)
    assert len(first) == NUID_LENGTH == 22
    assert set(first) <= _BASE62
    assert first is not second
    first.extend(b"ab")  # nats-py appends token_hex(2) to a request token
    assert len(second) == NUID_LENGTH


def test_csprng_tokens_share_no_prefix_and_no_step() -> None:
    """10,000 draws: no token repeats, no 12-character prefix repeats, no constant step between them.

    nats-py's NUID holds one 12-character prefix and adds a fixed increment, so one observed token
    predicts the next; these must not.
    """
    nuid = CsprngNuid()
    tokens = [bytes(nuid.next()) for _ in range(10_000)]

    assert len(set(tokens)) == len(tokens)
    assert len({token[:12] for token in tokens}) == len(tokens)
    alphabet = sorted(_BASE62)

    def value(token: bytes) -> int:
        number = 0
        for char in token[12:]:
            number = number * 62 + alphabet.index(char)
        return number

    steps = {value(b) - value(a) for a, b in zip(tokens, tokens[1:], strict=False)}
    assert len(steps) > 9_000


def test_nats_py_own_nuid_fails_the_same_checks() -> None:
    """non-vacuity: the sequential NUID these tests replace shares its prefix across draws."""
    from nats.nuid import NUID

    nuid = NUID()
    tokens = [bytes(nuid.next()) for _ in range(100)]
    assert len({token[:12] for token in tokens}) == 1


def test_the_declared_surface_names_the_nuid_override() -> None:
    """the NUID override is declared, so a nats-py release that renames ``_nuid`` fails by name."""
    assert PrivateAttribute("Client", "_nuid", __import__("nats.nuid", fromlist=["NUID"]).NUID) in PRIVATE_SURFACE


def test_a_threetears_connection_names_its_inboxes_from_the_csprng() -> None:
    """every inbox a NatsClient's connection names is unpredictable from any other it named.

    Through the front door: a NatsClient connects (to an in-process server, so no JetStream check)
    and its connection's inboxes are read. Under nats-py's own NUID all of them share one prefix.
    """
    from threetears.nats import NatsClient

    async def scenario() -> None:
        async with wire_server() as server:
            client = await NatsClient.connect(
                nats_url=server.url,
                nats_subject_namespace="3tears",
                client_name="csprng-check",
                verify_jetstream=False,
            )
            try:
                tokens = [client.raw.new_inbox().removeprefix("_INBOX.") for _ in range(200)]
            finally:
                await client.shutdown(drain_timeout=timedelta(seconds=1))
            assert all(len(token) == NUID_LENGTH for token in tokens)
            assert len({token[:12] for token in tokens}) == len(tokens)

    asyncio.run(scenario())
