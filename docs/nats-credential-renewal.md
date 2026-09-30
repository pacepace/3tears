# NATS credential renewal: make-before-break

Decision record, 2026-09-30. Code: `packages/nats/src/threetears/nats/credential_renewal.py` (the
schedule) and `NatsClient.renew_connection` in `packages/nats/src/threetears/nats/client.py` (the
handover). Live proof: `packages/nats/tests/integration/test_credential_renewal_live.py`.

## The problem

A pod that connects through the auth-callout holds a user JWT with a finite TTL (300s on the
platform). At its `exp` the server closes the connection (`client.authExpired`: `-ERR 'User
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

## Two nats-py defects the handover works around

Both reproduced against a real nats-server.

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
- **Longer TTLs.** Fewer renewals, the same loss at each one, and a wider window before a revoked
  or superseded principal is cut off.

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
  settle orders them on one server only.

## Fencing a superseded pod

A superseded agent pod (identity fencing, hub `resilience-task-05`) is refused when its successor
connection presents a stale generation. The renewal then fails and is retried; each refused attempt
counts toward `is_healthy`, which trips after three, and the liveness probe restarts the pod -- the
old connection is never renewed and expires at its own `exp` at the latest.
