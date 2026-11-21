"""a change-log read the server refused is told from one that timed out on a broker that is down.

A NATS permissions violation leaves the connection up and answers the refused request with nothing: the read ends
at its deadline, exactly as on an unreachable broker. The violation itself reaches only the connection's error
callback. The client records the refused subject there (``NatsClient.publish_refused_since``), and the change log
raises ``KvReadRefusedError`` for a read whose subject was refused, so a pod's reader can treat it as the wiring
error it is rather than an outage to wait out.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from nats.errors import Error as NatsError
from nats.errors import TimeoutError as NatsTimeoutError

from threetears.nats import KvError, KvReadRefusedError, NatsClient

_NAMESPACE = "3tears"
_LOG = "epoch-changes"
_STREAM = f"KV_{_NAMESPACE}-{_LOG}"


def _violation(subject: str) -> Exception:
    """the error nats-py hands ``error_cb`` for a refused publish: its parser lowercases the server's text."""
    return NatsError(f'nats: Permissions Violation for Publish to "{subject}"'.lower())


async def _client_over(
    js: Any, options_out: dict[str, Any] | None = None
) -> tuple[NatsClient, Callable[[Exception], Awaitable[None]]]:
    """a client connected over a stand-in connection whose JetStream context is ``js``, and its ``error_cb``."""
    captured: dict[str, Any] = {}
    raw = MagicMock()
    raw.jetstream.return_value = js

    async def _fake_establish(servers: list[str], options: dict[str, Any], nats_url: str) -> Any:
        captured["options"] = options
        return raw

    client = await NatsClient.connect(
        establish_connection=_fake_establish,
        nats_url="nats://localhost:4222",
        nats_subject_namespace=_NAMESPACE,
        client_name="tool-pod-x",
        verify_jetstream=False,
    )
    error_cb: Callable[[Exception], Awaitable[None]] = captured["options"]["error_cb"]
    if options_out is not None:
        options_out.update(captured["options"])
    return client, error_cb


class _UnansweredJetStream:
    """a JetStream context whose direct get is never answered: the server, when it refuses, says so on error_cb."""

    def __init__(self) -> None:
        self.on_error: Callable[[Exception], Awaitable[None]] | None = None
        self.refuse = True

    async def get_last_msg(self, stream: str, subject: str, *, direct: bool = False) -> Any:
        if self.refuse and self.on_error is not None:
            await self.on_error(_violation(f"$JS.API.DIRECT.GET.{stream}.{subject}"))
        raise NatsTimeoutError


async def test_a_read_the_server_refused_raises_read_refused_naming_the_subject() -> None:
    js = _UnansweredJetStream()
    client, js.on_error = await _client_over(js)
    log = client.change_log(_LOG)

    with pytest.raises(KvReadRefusedError) as refused:
        await log.last("s-ns_wrong.scope_epochs.e")
    assert (
        refused.value.subject
        == f"$js.api.direct.get.{_STREAM}.$kv.{_NAMESPACE}-{_LOG}.s-ns_wrong.scope_epochs.e".lower()
    )
    assert isinstance(refused.value, KvError), "still a KvError for a caller that tells nothing apart"


async def test_a_read_that_only_timed_out_is_an_ordinary_failure() -> None:
    js = _UnansweredJetStream()
    js.refuse = False
    client, js.on_error = await _client_over(js)

    with pytest.raises(KvError) as failed:
        await client.change_log(_LOG).last("s-ns_x.scope_epochs.e")
    assert not isinstance(failed.value, KvReadRefusedError)


async def test_a_refusal_of_another_key_or_before_the_read_is_not_this_reads() -> None:
    js = _UnansweredJetStream()
    js.refuse = False
    client, on_error = await _client_over(js)
    await on_error(_violation(f"$JS.API.DIRECT.GET.{_STREAM}.$KV.{_NAMESPACE}-{_LOG}.s-ns_x.scope_epochs.e"))
    assert client.publish_refused_since("s-ns_x.scope_epochs.e", 0.0) is not None
    assert client.publish_refused_since("s-ns_x.scope_epochs.e", time.monotonic() + 1) is None, "only since the read"

    with pytest.raises(KvError) as failed:
        await client.change_log(_LOG).last("s-ns_x.other_table.e")
    assert not isinstance(failed.value, KvReadRefusedError), "another key's refusal is not this read's"


async def test_a_new_credential_clears_the_refusals_the_last_one_heard() -> None:
    """a reconnect mints a fresh credential, which may name the subject: a remembered refusal no longer holds."""
    options: dict[str, Any] = {}
    js = _UnansweredJetStream()
    client, on_error = await _client_over(js, options)
    subject = f"$JS.API.DIRECT.GET.{_STREAM}.$KV.{_NAMESPACE}-{_LOG}.s-ns_x.scope_epochs.e"
    await on_error(_violation(subject))
    assert client.publish_refused_since("s-ns_x.scope_epochs.e", 0.0) is not None

    await options["reconnected_cb"]()

    assert client.publish_refused_since("s-ns_x.scope_epochs.e", 0.0) is None
