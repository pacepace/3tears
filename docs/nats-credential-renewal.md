# NATS credential renewal: make-before-break

Decision record, 2026-09-30. Code: `packages/nats/src/threetears/nats/credential_renewal.py` (the
schedule) and `NatsClient.renew_connection` in `packages/nats/src/threetears/nats/client.py` (the
handover). Live proof: `packages/nats/tests/integration/test_credential_renewal_live.py`.

## Long logins, and a kick for "now" (owner ruling Q17)

nats-server takes authority away from a live connection in exactly one way: it closes the
connection when its user JWT's `exp` passes (`client.authExpired`). There is no in-place
re-authorization -- a second `CONNECT` drops every subscription on 2.14 -- and no refresh of a live
connection's permissions. So a short TTL used to be the fence: every pod renewed every few minutes
so that a revoked or superseded one would be cut off within one TTL, and every renewal was a
handover carrying its own residual risk.

The TTL is now a **backstop**: `PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS` is a day. Access is
taken away when it must be:

- **Kick.** Every auth-callout request carries the requesting server's id and the client's id
  (`AuthCalloutRequest.connection`). The consumer records them for every admission
  (`AdmissionRecorder`; a recorder that raises denies the connection) and, when a principal must
  lose access, closes the connection through the server's system account --
  `kick_connection(system_client, connection)`, `$SYS.REQ.SERVER.<server_id>.KICK {"cid": N}`.
  Measured live: the connection closes about a millisecond after the kick is sent, on 2.12.6 and
  2.14.2 (`test_connection_kick_live.py`).
- **Refuse the reconnect.** A kicked nats-py client sees EOF and reconnects through the callout,
  which is what decides whether it comes back. A kick without a refusing callout only costs the
  principal a reconnect.
- **The system account.** Only a SYSTEM-account user can send a kick; a client of any other
  account finds no responder, which a kick would read as a server that is gone.
  `require_system_account` proves the login before the first kick. A system user in
  server-config callout mode must be listed in `auth_callout.auth_users`, or it goes through the
  callout like any other connection -- verified live on both versions.
- **The backstop renewal** is the make-before-break handover below, about once a day per pod.

What a kick cannot bound: a connection whose kick is lost for good, or whose server is unreachable
from the kicking client (a partition makes it answer "no responders", indistinguishable from a
server that restarted). Those keep their credential until the backstop expiry, and are refused at
their next connect.

## A rolling restart is a handover too

A server in lame-duck mode stops accepting connections, sends its clients a lame-duck notice, and
closes them over its lame-duck duration. nats-py only calls `lame_duck_mode_cb`; the close that
follows is its ordinary reconnect, which drops what is in flight exactly as the old renewal did.
Every connection a `NatsClient` opens now carries that callback: on the current connection it
starts a move to a successor through `renew_connection`, retried every `REAUTH_RETRY_SECONDS`
while the connection is still current and open (another renewal, or the server's own close, ends
it). The old connection is kept for `longest_request_seconds` or until the server closes it.
Proven live on a two-node 2.14.2 cluster (`test_lame_duck_handover_live.py`): the pod moved about
90ms after the notice, and not one request or subscribed message was lost, where the reconnect
lost several requests and most of the feed.

## A run whose order is its meaning stays on one connection

NATS orders messages per publisher, and a publisher is a connection: a client mid-handover is two
publishers, and on two nodes of a cluster they travel two routes. `_settle_publishes` orders the old
connection's publishes on its own server only. A run whose order is its meaning -- a token stream,
whose tokens carry no sequence number -- takes a `PublishPin` (`NatsClient.publish_pin()`) and
passes it to every `publish_raw` of the run; every publish then leaves on the connection the first
one used, held open for the longest request. Reproduced live: after a handover onto the other node,
a publish on the successor overtook the tail of a run still leaving on the old connection; the
pinned run itself arrived in order.

## The problem

A pod that connects through the auth-callout holds a user JWT with a finite TTL (a day by default
since Q17; five minutes when this handover was built). At its `exp` the server closes the connection (`client.authExpired`: `-ERR 'User
Authentication Expired'`, then `closeConnection`), and nats-py routes that error to a terminal
close that forever-reconnect never sees. So the client must renew before then.

It used to renew by reconnecting its one connection (`_process_op_err` -> `_attempt_reconnect`).
That is a real disconnect. From the moment the transport closes until the new one has replayed its
subscriptions -- TCP connect, INFO/CONNECT, the auth-callout round trip to the hub, SUB replay --
the server holds no interest for the pod, and core NATS drops what is published to it:

- the reply to a request the pod sent before the renewal (the request mux subscription is gone);
- a message on any subject the pod subscribes;
- a push-consumer delivery (a KV watch's, a result waiter's).

And a reply the pod OWES for a request it received cannot be sent after the reconnect at all:
nats-server grants `allow_responses` to the connection that received the request, so the new
connection's publish is refused as a permissions violation while the publish call reports success.

On cobalt-prod every agent pod renewed every 210s, and a turn whose L3 read straddled one failed:
the read was sent ~0.4s before the renewal, its reply landed in the gap, and it failed closed 7s
later (`timeout_ms` 5000 + 2s) as `DataLayerUnavailableError: NATS request failed`.

The pre-renewal drain (`before_renewal=ToolServer.drain_before_reauth`, `drain_grace_seconds`) did
not cover this and could not. It waited only for replies the in-process ToolServer owed, never for
the pod's own outbound requests; `drain_grace_seconds` only fed the cadence check. A request of any
length is lost if its reply lands in the gap -- the cadence check reasoned about requests LONGER
than the renewal interval, which was never the failure.

## The mechanism

A credential is per connection and stays valid until its own `exp`. So the renewal opens a second
connection, which the auth-callout mints a fresh credential, and hands everything over before the
first one is closed:

1. **Open the successor** with the options the first connection used (bounded by
   `REAUTH_CONNECT_TIMEOUT_SECONDS`). A failure changes nothing; the loop retries in
   `REAUTH_RETRY_SECONDS` while the current credential is still valid.
2. **Subscribe every subscription on it, in the same queue group**, and round-trip it. Both
   connections are now members.
3. **Settle the old connection's publishes, then switch.** New publishes wait (the publish gate)
   while the old connection makes an ordered round trip, so a message published on the successor never
   overtakes one already published on the old connection. Then the successor is current.
4. **End each subscription's old half without dropping anything**, and rebind durable push
   consumers.
5. **Retire the old connection** after `longest_request_seconds` (never later than `ttl - leeway`
   less the drain bound): until then a reply to a request sent on it still arrives there, a reply
   owed for a request received there still leaves there, and a JetStream message received there is
   acknowledged there. Then `drain()` and close.

Anything else bound to a connection follows the current one on its own: `NatsKvBucket` rebinds its
handle before its next operation (one `STREAM.INFO`); a `watch_key` whose connection closed while the
client lives on replaces its consumer on the successor (last-per-subject redelivers the latest
value); a `JetStreamResultWaiter` rebuilds on the current connection (`DeliverPolicy.ALL` still finds
the answer); a `JetStreamPullConsumer` rebinds its durable before its next fetch. Per-call
subscriptions (a gateway token stream, a key listing) finish on the connection they started on,
which is held open for the longest request.

### Why every subscription holds a queue group

While both connections are subscribed, a plain subscription would receive every message twice; one
unsubscribed before the other is subscribed would receive nothing in between. nats-server delivers a
message to every plain subscriber and to ONE member of each queue group, so a subscription made
without a group joins one of its own (`_solo.<uuid7>`): to every other subscriber it behaves exactly
like a plain subscription, and across the handover the server gives each message to exactly one of
the two members. Verified against nats-server source (`client.go`, `canSubscribeInternal`): a plain
subscribe allow-list entry admits any queue name unless the allow list itself names queues, which no
grant here does. Reverting to plain subscriptions makes the live test deliver messages twice.

The price is ordering across the handover: while both are members, two consecutive messages can go
to different connections. Core NATS orders per publisher per connection only.

### The schedule

The successor opens `ttl - REAUTH_LEEWAY_SECONDS - REAUTH_BUFFER_SECONDS - longest` after the current
connection was established (90s at TTL 300 with a 120s completion); the old connection is retired
`longest` later, `REAUTH_BUFFER_SECONDS` short of `ttl - leeway`. A TTL at or below
`longest + REAUTH_MARGIN_SECONDS` cannot hold the longest request across a renewal:
`unsafe_renewal_reason` names it every cycle, and the Hub refuses to start with it
(`MINIMUM_SAFE_NATS_USER_JWT_TTL_SECONDS`, the same inequality).

### One lifecycle model

Which connections a client holds, what each is for, and where the client is in its life are one
state machine, `_ConnectionLifecycle` in `client.py`. Each connection has one role -- `candidate`
(a renewal opened it; not yet current), `current`, or `retiring` (replaced; held for the work it
carries) -- and the client one phase: `running`, `renewing` (one renewal at a time), `abandoned`
or `closed`. Every transition is a method of that class and checks the whole state first:

- a candidate is registered in the same step that opens it, so every sweep (abandon, shutdown, the
  renewal's own exit) sees it and no cancellation can orphan it;
- the step that makes a candidate current refuses once the client was abandoned or shut down, so a
  renewal mid-handover when a refusal lands cannot leave an abandoned client with an open
  connection;
- the renewal's exit closes any candidate never made current, on every path, cancellation included.

The callbacks each connection carries read its role: only the current connection's refusals count
against `is_healthy`, and only its reconnects run the consumer's hooks.

## Three nats-py defects the handover works around

The first two reproduced against a real nats-server.

- **`Subscription.drain()` can lose messages.** `_send_unsubscribe` queues the `UNSUB` in the
  pending buffer, but `Client._send_ping` writes the `PING` straight to the transport, so the `PING`
  reaches the server first -- wire order observed `PING`, `UNSUB`. The `PONG` then proves nothing
  about the `UNSUB`; the drain forgets the subscription while the server may still route to it, and
  `_process_msg` drops what arrives for an unknown sid. With a shared queue group that message
  reached no other member: 1 of 11858 streamed messages was lost across a few dozen renewals under
  load. The handover (`_stop_routing_then_drain`) sends the `UNSUB` on its own and makes an ordered
  round trip (`_round_trip`: the pending buffer is written out before the `PING`) before running
  nats-py's drain.
- **A flush that times out or is cancelled kills the read loop.** The cancelled future stays in
  `_pongs`; the next `PONG` raises `InvalidStateError` in `_process_pong`, and `_read_loop`'s
  catch-all logs "nats: encountered error" and exits. The connection still reports connected and
  never reads again, and a later `flush` returns without a round trip. Every wrapper round trip
  (`ping`, `flush`, the handover's) goes through `_round_trip`, which shields the `PONG` future, so a
  timeout abandons the wait and never the future. The one flush left inside nats-py's own `drain`
  is bounded by its own timeout and never cancelled by the handover.
- **A forced flush has no bound and swallows cancellation.** `Client._flush_pending(force_flush=True)`
  waits for the flusher's `transport.drain()` under `flush_timeout`, which defaults to none, inside
  `except asyncio.CancelledError: pass`. On a backpressured socket a round trip built on it outlived
  its timeout -- `ping(timeout)` did not answer, the handover's settle held the publish gate shut --
  and a cancellation from shutdown or a bounded drain was discarded. `_round_trip` never awaits the
  flusher: it hands the pending buffer and its `PING` to the transport in one synchronous step,
  queues the `PONG` future in the same step, wakes the flusher without waiting, and waits only for
  the `PONG`, under the caller's timeout.

## Alternatives rejected

- **Re-authenticate in place.** NATS has no in-band re-authentication. nats-server does process a
  second `CONNECT` (`client.go`, `processConnect`), but it first drops every subscription the
  connection holds (`if !firstConnect { c.clearAccountSubs(false) }`) and runs the auth-callout on
  the client's read loop -- the same gap, shorter, on a server-version-specific path.
- **Swap the transport inside one nats-py client.** Keeps every object that holds the nats-py
  client valid, but means driving a second transport, parser, read loop and PING/PONG queue through
  nats-py internals (`_transport`, `_ps`, `_pongs`, `_flusher`, `_process_connect_init`): a fork of
  nats-py's connection core in all but name.
- **Bridge the gap with a temporary side connection** that forwards the old connection's
  subscriptions into the client while it reconnects. Doubles delivery on both edges of the gap, and
  a request received on the bridge cannot be answered from the main connection.
- **Idempotent retry of reads across a reconnect.** Covers reads only; a write whose reply was lost
  was applied, and retrying it applies it twice. The handover removes the loss instead.
- **Longer TTLs alone.** Fewer renewals, the same loss at each one, and a wider window before a
  revoked or superseded principal is cut off. Adopted with the kick (above), which closes that
  window, and with this handover, which removes the loss.

## What a renewal can still lose

- A request, owed reply or per-call subscription that outlives `longest_request_seconds` after the
  handover: the old connection is retired under it. That is the declared longest request being
  wrong; `SYNC_REPLY_BUDGET_SECONDS` routes longer tool calls to the durable result stream.
- Anything still on the old connection when it is drained at retirement goes through nats-py's
  `Client.drain()`, which carries the first defect above for each remaining subscription. What is
  left there by then is past its declared bound, and the push consumers that could be affected
  (key watches, result waiters) recover from their streams.
- In a NATS cluster that places the successor on a different server than the old connection, a
  message published just before the handover and one just after travel different routes; the
  settle orders them on one server only. A run that must stay ordered is pinned (above).

## A renewal refused on purpose, and one that is not

Owner ruling Q16 (2026-09-30). A connection the auth-callout refuses learns only
`-ERR 'Authorization Violation'`: nats-server's `client.authViolation` sends that fixed text
"regardless of the authErr override", and the reason the callout gave reaches only the server log
(`auth_callout.go`: "auth callout service returned an error"). So the two cases are told apart out
of band:

- **Refused on purpose** -- the hub's fence refuses a superseded pod-session. The resolver returns a
  `RefusedPrincipal`; the responder denies with the typed reason (`"superseded"`) and publishes a
  `CredentialRefusal` to the principal's own inbox (`{inbox_prefix}.credential-refused`), which the
  pod still holds a subscription on over its still-valid connection. The pod
  (`NatsClient.abandon_on_refusal`, armed by the SDK with its pod-session and current generation)
  closes every connection at once and stops the renewal; its supervisor restarts it.
- **Anything else** -- the callout unreachable, timed out, or erroring. No refusal is published.
  The renewal is retried every 5s on the current connection, which is kept until its own credential
  expires; a candidate's refusals do not count against `is_healthy`, so nothing restarts a pod
  whose connection still works.
