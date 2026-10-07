"""Integration test: a credential renewal loses nothing in flight, against a real auth-callout.

A connection authenticated through the NATS auth-callout holds a user JWT that expires, and the
client renews it before it does (:meth:`threetears.nats.NatsClient.renew_credential`). This proves,
against a real nats-server running config-mode ``auth_callout`` and the real
:class:`~threetears.nats.AuthCalloutResponder` minting the pod's user JWT, that the renewal drops
nothing the connection was carrying when it began:

- a request the pod sent before the renewal, whose reply the responder publishes WHILE the renewal
  is under way, completes -- the incident this reproduces is the cobalt L3 read that failed closed
  with ``DataLayerUnavailableError: NATS request failed`` 6.6s after a renewal reconnect;
- a subscription keeps receiving: every message published before, during and after the renewal
  arrives, and arrives exactly ONCE -- the new connection and the old one are both subscribed while
  the renewal hands over, and neither may double a message;
- a reply the pod OWES for a request it received before the renewal is delivered after it. NATS
  lets only the connection that received a request answer it, so the answer must leave on that
  connection;
- a KV key watch sees the write made while the renewal was under way, and the one after it.

The "while the renewal is under way" window is held open deterministically: the callout resolver
parks the renewal's authorization until the test has published into the window, the way a hub
under load holds a real renewal for a few hundred milliseconds. That is the window the old
renewal -- a reconnect of the one connection -- spent disconnected.

Gated on docker: a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import aclosing
from datetime import timedelta
from pathlib import Path
from typing import Any

import nats
import pytest

from threetears.core.testing.containers import check_docker_available
from threetears.nats import (
    AuthCalloutRequest,
    AuthCalloutResponder,
    GrantPolicy,
    IncomingMessage,
    JsResource,
    NatsClient,
    PrincipalPermissions,
    PrincipalResolver,
    ResolvedPrincipal,
    Subject,
    account_public_key,
    decode_auth_request,
    generate_account_seed,
)
from threetears.nats.kv_watch import KvKeyUpdate
from threetears.nats.subjects import get_default_namespace, set_default_namespace

pytestmark = pytest.mark.integration

_NS = "renewlive"
_TOKEN = "renewal-live-identity"  # noqa: S105 - the test resolver's only admissible credential
_ADMIN_PW = "admin-pw"  # noqa: S105 - ephemeral testcontainer credential
_INBOX = "_INBOX_renewal_live"
_BUCKET_SUFFIX = "cfg"
_BUCKET = f"{_NS}-{_BUCKET_SUFFIX}"
_KEY = "watched"
_QUERY_SUBJECT = "svc.query"
_EVENTS = "events"
_CALLS_SUBJECT = "pod.calls"
_POD_OUT = "podout.seq"
#: the TTL the pod schedules its renewal against, and the TTL the responder mints. Long enough that
#: the credential never actually expires during the test; the renewal fires early because the
#: schedule subtracts its margins and the longest request from it.
_TTL_SECONDS = 100
#: the longest request the pod declares. Small, so the renewal fires within seconds.
_LONGEST_REQUEST_SECONDS = 5.0
#: how long the callout resolver holds the renewal's authorization open.
_WINDOW_SECONDS = 1.5
_WAIT = 20.0


class _GatedResolver(PrincipalResolver):
    """admits the one test credential; parks every authorization after the first until released.

    :param gate: set by the test to release a parked authorization
    :ptype gate: asyncio.Event
    """

    def __init__(self, gate: asyncio.Event) -> None:
        """bind the gate.

        :param gate: set by the test to release a parked authorization
        :ptype gate: asyncio.Event
        :return: nothing
        :rtype: None
        """
        self._gate = gate
        self.calls = 0
        self.renewal_started = asyncio.Event()
        self.renewal_admitted = asyncio.Event()

    async def resolve(self, request: AuthCalloutRequest) -> ResolvedPrincipal | None:
        """admit the test credential, holding the renewal's authorization until the gate opens.

        :param request: the decoded authorization request
        :ptype request: AuthCalloutRequest
        :return: the pod principal, or ``None`` for any other credential
        :rtype: ResolvedPrincipal | None
        """
        self.calls += 1
        if request.bootstrap_token != _TOKEN:
            return None
        if self.calls > 1:
            self.renewal_started.set()
            await self._gate.wait()
            self.renewal_admitted.set()
        return ResolvedPrincipal(conn_id="renewal-live", name="renewal-live")


class _PodGrant(GrantPolicy):
    """the pod's least-privilege grant: its inbox, three app subjects, and one watched key."""

    def permissions(self, principal: ResolvedPrincipal) -> PrincipalPermissions:
        """the pod's allow-list.

        Publishing to a requester's inbox is NOT granted: the pod may answer a request only through
        ``allow_responses``, which the server scopes to the connection that received the request.

        :param principal: the admitted principal
        :ptype principal: ResolvedPrincipal
        :return: the allow-list minted into the pod's user JWT
        :rtype: PrincipalPermissions
        """
        return PrincipalPermissions(
            publish=(_QUERY_SUBJECT, _POD_OUT),
            subscribe=(f"{_INBOX}.>", f"{_EVENTS}.>", _CALLS_SUBJECT),
            allow_responses=True,
            inbox_prefix=_INBOX,
            js_resources=(JsResource.kv_key_read(_BUCKET, key=_KEY),),
        )


def _server_config(account_pub: str) -> str:
    """a JetStream nats-server that delegates every connection but the admin's to the callout.

    ``timeout`` is the server's wait for a callout answer; it is raised above the window the
    resolver holds open, so the parked authorization is answered rather than timed out.

    :param account_pub: the account public key that signs callout responses
    :ptype account_pub: str
    :return: the nats-server configuration text
    :rtype: str
    """
    admin = {"user": "admin", "password": _ADMIN_PW}
    return (
        "port: 4222\n"
        "jetstream { store_dir: /tmp/js-store }\n"
        "authorization {\n"
        "  timeout: 10\n"
        "  auth_callout {\n"
        f'    issuer: "{account_pub}"\n'
        "    auth_users: [ admin ]\n"
        "  }\n"
        f"  users: [ {json.dumps(admin)} ]\n"
        "}\n"
    )


@contextlib.contextmanager
def _nats_with_config(config_text: str, conf_dir: Path) -> Iterator[str]:
    """start a nats-server with ``config_text``; yield its URI.

    :param config_text: the server configuration
    :ptype config_text: str
    :param conf_dir: a directory to mount the configuration from
    :ptype conf_dir: Path
    :return: the server's URI, yielded
    :rtype: Iterator[str]
    """
    from threetears.core.testing.fixtures import nats_server, nats_server_uri  # noqa: PLC0415

    (conf_dir / "nats.conf").write_text(config_text)
    container = (
        nats_server().with_volume_mapping(str(conf_dir), "/etc/nats", "ro").with_command(["-c", "/etc/nats/nats.conf"])
    )
    container.start()
    try:
        yield nats_server_uri(container)
    finally:
        container.stop()


async def test_a_renewal_loses_nothing_in_flight(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """requests, owed replies, subscriptions and a KV watch all ride through a renewal intact."""
    if not check_docker_available():
        pytest.skip("Docker not available")

    account_seed = generate_account_seed()
    gate = asyncio.Event()
    resolver = _GatedResolver(gate)
    responder = AuthCalloutResponder(
        nc=None,  # driven through build_response from the admin connection below
        account_seed=account_seed,
        resolver=resolver,
        policy=_PodGrant(),
        account_name="$G",
        user_jwt_ttl_seconds=_TTL_SECONDS,
    )
    previous_ns = get_default_namespace()

    with _nats_with_config(_server_config(account_public_key(account_seed)), tmp_path) as uri:
        admin = await nats.connect(uri, user="admin", password=_ADMIN_PW, max_reconnect_attempts=0)
        pod: NatsClient | None = None
        tasks: list[asyncio.Task[Any]] = []
        try:

            async def _on_auth(msg: Any) -> None:
                request = decode_auth_request(bytes(msg.data).decode())
                await msg.respond((await responder.build_response(request)).encode())

            await admin.subscribe("$SYS.REQ.USER.AUTH", cb=_on_auth)

            # the service the pod queries: it answers each query only once the renewal has begun,
            # so the reply is in flight exactly while the old renewal had the pod disconnected.
            async def _on_query(msg: Any) -> None:
                await resolver.renewal_started.wait()
                await msg.respond(b"answer:" + bytes(msg.data))

            await admin.subscribe(_QUERY_SUBJECT, cb=_on_query)
            await admin.flush()

            admin_js = admin.jetstream()
            admin_kv = await admin_js.create_key_value(bucket=_BUCKET, direct=True)
            await admin_kv.put(_KEY, b"before")

            pod = await NatsClient.connect(
                nats_url=uri,
                nats_subject_namespace=_NS,
                client_name="renewal-live",
                auth_token=lambda: _TOKEN,
                inbox_prefix=_INBOX,
                startup_timeout=timedelta(seconds=15),
            )
            assert resolver.calls == 1

            # === a subscription, a pending reply the pod owes, and a key watch =====================
            received: Counter[bytes] = Counter()

            async def _on_event(msg: IncomingMessage) -> None:
                received[bytes(msg.data)] += 1

            await pod.subscribe(Subject.raw(f"{_EVENTS}.>"), cb=_on_event)

            answer_owed = asyncio.Event()

            async def _on_call(msg: IncomingMessage) -> None:
                # the pod does its work across the renewal and answers after it has completed.
                await answer_owed.wait()
                assert msg.reply_subject is not None
                await pod_client().publish_raw_reply(reply_subject=msg.reply_subject, payload=b"done")

            def pod_client() -> NatsClient:
                assert pod is not None
                return pod

            await pod.subscribe(Subject.raw(_CALLS_SUBJECT), cb=_on_call)
            await pod.flush()

            bucket = await pod.kv_bucket(name=_BUCKET_SUFFIX, create_if_missing=False, direct=True)
            seen: list[KvKeyUpdate] = []
            watch_saw: dict[bytes, asyncio.Event] = {
                v: asyncio.Event() for v in (b"before", b"during", b"after", b"retired")
            }

            async def _watch() -> None:
                async with aclosing(bucket.watch_key(key=_KEY, heartbeat=timedelta(seconds=1))) as updates:
                    async for update in updates:
                        seen.append(update)
                        if update.value in watch_saw:
                            watch_saw[update.value].set()

            tasks.append(asyncio.create_task(_watch()))
            await asyncio.wait_for(watch_saw[b"before"].wait(), timeout=_WAIT)

            # a steady stream of events from before the renewal until after it, to catch a message
            # dropped in the handover or delivered twice by the overlap.
            published: list[bytes] = []
            published_at: dict[bytes, float] = {}
            stop_stream = asyncio.Event()

            async def _stream() -> None:
                seq = 0
                while not stop_stream.is_set():
                    body = f"seq-{seq}".encode()
                    await admin.publish(f"{_EVENTS}.stream", body)
                    published.append(body)
                    published_at[body] = time.time()
                    seq += 1
                    # fast enough that the moment both connections are subscribed carries traffic
                    await asyncio.sleep(0.001)

            tasks.append(asyncio.create_task(_stream()))

            # === in flight when the renewal starts ================================================
            owed = asyncio.create_task(admin.request(_CALLS_SUBJECT, b"work", timeout=_WAIT))
            tasks.append(owed)
            query = asyncio.create_task(
                pod.request_raw(
                    subject=Subject.raw(_QUERY_SUBJECT),
                    payload=b"q1",
                    timeout=timedelta(seconds=10),
                )
            )
            tasks.append(query)

            replaced = pod.raw
            with caplog.at_level(logging.INFO, logger="threetears.nats"):
                pod.renew_credential(ttl_seconds=lambda: _TTL_SECONDS, longest_request_seconds=_LONGEST_REQUEST_SECONDS)

                # === the renewal window: its authorization is parked ============================
                await asyncio.wait_for(resolver.renewal_started.wait(), timeout=_WAIT)
                await admin.publish(f"{_EVENTS}.window", b"during-window")
                await admin_kv.put(_KEY, b"during")
                await admin.flush()
                await asyncio.sleep(_WINDOW_SECONDS)
                gate.set()
                await asyncio.wait_for(resolver.renewal_admitted.wait(), timeout=_WAIT)

                # the renewal completes on its own; give the handover a moment to finish.
                for _ in range(100):
                    if pod.is_connected:
                        break
                    await asyncio.sleep(0.05)
                await asyncio.sleep(1.0)

                # === after the renewal ==========================================================
                answer_owed.set()
                await admin.publish(f"{_EVENTS}.after", b"after-renewal")
                await admin_kv.put(_KEY, b"after")
                await admin.flush()

                assert await asyncio.wait_for(query, timeout=_WAIT) == b"answer:q1"
                reply = await asyncio.wait_for(owed, timeout=_WAIT)
                assert reply.data == b"done"
                await asyncio.wait_for(watch_saw[b"during"].wait(), timeout=_WAIT)
                await asyncio.wait_for(watch_saw[b"after"].wait(), timeout=_WAIT)

                # === after the replaced connection is retired ===================================
                # everything that was bound to it -- the watch's consumer, the KV handle, the
                # subscription's old half -- carries on from the successor.
                assert pod.raw is not replaced
                for _ in range(400):
                    if replaced.is_closed:
                        break
                    await asyncio.sleep(0.05)
                assert replaced.is_closed, "the replaced connection was never retired"
                await admin_kv.put(_KEY, b"retired")
                await admin.publish(f"{_EVENTS}.final", b"after-retirement")
                await admin.flush()
                await asyncio.wait_for(watch_saw[b"retired"].wait(), timeout=_WAIT)
                assert await bucket.get(key=_KEY) == b"retired"
                second = await pod.request_raw(
                    subject=Subject.raw(_QUERY_SUBJECT), payload=b"q2", timeout=timedelta(seconds=10)
                )
                assert second == b"answer:q2"

                stop_stream.set()
                await admin.flush()
                for _ in range(100):
                    if b"after-retirement" in received and all(body in received for body in published):
                        break
                    await asyncio.sleep(0.05)

            assert b"during-window" in received
            assert b"after-renewal" in received
            assert b"after-retirement" in received
            missing = [body for body in published if body not in received]
            timeline = sorted(
                [(published_at[body], f"lost {body.decode()}") for body in missing[:5]]
                + [(r.created, r.getMessage()[:60]) for r in caplog.records if r.name.startswith("threetears.nats")]
            )
            assert not missing, f"{len(missing)} of {len(published)} streamed messages lost: {timeline}"
            doubled = sorted(body for body, count in received.items() if count > 1)
            assert not doubled, f"delivered more than once: {doubled[:5]}"
            assert [u.value for u in seen].count(b"during") == 1

            violations = [r.getMessage() for r in caplog.records if "permissions violation" in r.getMessage().lower()]
            assert not violations, violations
            # positive control: the capture sees this package's log lines, so the absence above is
            # evidence rather than a logger nobody was listening to.
            assert "NATS credential renewed" in caplog.text
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with contextlib.suppress(BaseException):
                    await task
            if pod is not None:
                await pod.shutdown(drain_timeout=timedelta(seconds=2))
            await admin.close()
            set_default_namespace(previous_ns)


#: renewals the stress test drives back to back.
_STRESS_RENEWALS = 100


async def test_many_renewals_under_a_busy_subscription_lose_and_double_nothing(tmp_path: Path) -> None:
    """every renewal hands the subscription over while messages are streaming onto it.

    Each handover opens a moment when both connections are members of the subscription's group and
    a moment when the old member is being removed. A message the server routes to the old member in
    that second moment is the one that can fall between the connections: nats-py's own drain forgets
    the subscription on a round trip that can overtake its UNSUB. One such loss in a few dozen
    renewals is what this test exists to see, so it drives many renewals, with a short hold, under a
    publisher that never pauses.
    """
    if not check_docker_available():
        pytest.skip("Docker not available")

    account_seed = generate_account_seed()
    gate = asyncio.Event()
    gate.set()  # nothing is parked: the renewals run as fast as the callout answers
    resolver = _GatedResolver(gate)
    responder = AuthCalloutResponder(
        nc=None,
        account_seed=account_seed,
        resolver=resolver,
        policy=_PodGrant(),
        account_name="$G",
        user_jwt_ttl_seconds=_TTL_SECONDS,
    )
    previous_ns = get_default_namespace()

    with _nats_with_config(_server_config(account_public_key(account_seed)), tmp_path) as uri:
        admin = await nats.connect(uri, user="admin", password=_ADMIN_PW, max_reconnect_attempts=0)
        pod: NatsClient | None = None
        stream: asyncio.Task[None] | None = None
        try:

            async def _on_auth(msg: Any) -> None:
                request = decode_auth_request(bytes(msg.data).decode())
                await msg.respond((await responder.build_response(request)).encode())

            await admin.subscribe("$SYS.REQ.USER.AUTH", cb=_on_auth)
            await admin.flush()

            pod = await NatsClient.connect(
                nats_url=uri,
                nats_subject_namespace=_NS,
                client_name="renewal-stress",
                auth_token=lambda: _TOKEN,
                inbox_prefix=_INBOX,
                startup_timeout=timedelta(seconds=15),
            )
            received: Counter[bytes] = Counter()

            async def _on_event(msg: IncomingMessage) -> None:
                received[bytes(msg.data)] += 1

            await pod.subscribe(Subject.raw(f"{_EVENTS}.>"), cb=_on_event)
            await pod.flush()

            published: list[bytes] = []
            stop = asyncio.Event()

            async def _stream() -> None:
                seq = 0
                while not stop.is_set():
                    body = f"seq-{seq}".encode()
                    await admin.publish(f"{_EVENTS}.stress", body)
                    published.append(body)
                    seq += 1
                    await asyncio.sleep(0)

            # and the other direction: the pod publishes a sequence while its connection is replaced
            # under it, and the order it arrives in must be the order it was sent in.
            sent_by_pod: list[int] = []
            seen_from_pod: list[int] = []

            async def _on_pod_out(msg: Any) -> None:
                seen_from_pod.append(int(msg.data))

            await admin.subscribe(_POD_OUT, cb=_on_pod_out)
            await admin.flush()

            async def _pod_stream() -> None:
                assert pod is not None
                seq = 0
                while not stop.is_set():
                    await pod.publish_raw(subject=Subject.raw(_POD_OUT), payload=str(seq).encode())
                    sent_by_pod.append(seq)
                    seq += 1
                    await asyncio.sleep(0)

            stream = asyncio.create_task(_stream())
            pod_stream = asyncio.create_task(_pod_stream())
            first = pod.raw
            for _ in range(_STRESS_RENEWALS):
                await pod.renew_connection(retire_after=timedelta(seconds=0.2))
                await asyncio.sleep(0.05)
            stop.set()
            await stream
            await pod_stream
            await pod.flush()
            await admin.flush()
            for _ in range(200):
                if all(body in received for body in published) and len(seen_from_pod) >= len(sent_by_pod):
                    break
                await asyncio.sleep(0.05)

            assert pod.raw is not first
            assert first.is_closed, "the first connection was never retired"
            assert len(published) > 10 * _STRESS_RENEWALS, "the stream was too slow to exercise the handovers"
            missing = [body for body in published if body not in received]
            assert not missing, f"{len(missing)} of {len(published)} messages lost across renewals: {missing[:5]}"
            doubled = sorted(body for body, count in received.items() if count > 1)
            assert not doubled, f"{len(doubled)} delivered more than once: {doubled[:5]}"
            assert len(sent_by_pod) > 10 * _STRESS_RENEWALS, "the pod published too slowly to exercise the handovers"
            assert seen_from_pod == sent_by_pod, (
                "the pod's publishes arrived lost, doubled or out of order across renewals: "
                f"{[(i, seq) for i, seq in enumerate(seen_from_pod) if i >= len(sent_by_pod) or sent_by_pod[i] != seq][:5]}"
            )
        finally:
            if stream is not None and not stream.done():
                stream.cancel()
                with contextlib.suppress(BaseException):
                    await stream
            if pod is not None:
                await pod.shutdown(drain_timeout=timedelta(seconds=2))
            await admin.close()
            set_default_namespace(previous_ns)
