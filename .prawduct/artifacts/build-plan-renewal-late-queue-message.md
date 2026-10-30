---
artifact: build-plan
version: 1
scope: renewal-late-queue-message
branch: fix/renewal-late-queue-message
depends_on: []
governed_by: []
last_validated: 2026-10-30
---

# Build plan: a credential renewal loses no queue-group message at the handover

Bugfix, medium. Found by `test_many_renewals_under_a_busy_subscription_lose_and_double_nothing` failing once in the
#1050 CI run (1 of 45,272 messages lost), reproduced under load (6 stress loops in parallel).

## Goals

- A renewal's handover loses no message routed to the subscription, under load included.
- The stress test proves what it claims (no test-side loss).

## Chunks

### Chunk 01: The old half keeps receiving until its connection retires

The old half was forgotten (nats-py drain) at the PONG that proved its UNSUB; a message delivered to it just after
was dropped. Now `_stop_routing` (UNSUB + ordered round trip) and the receiver runs until the replaced connection is
drained at retirement. Unit test `test_a_message_the_server_delivers_to_the_old_half_after_its_round_trip_still_arrives`
fails on the old code. Done.

### Chunk 02: The stress test proves the admin's subscription with a probe

nats-py's own flush writes its PING ahead of a SUB still in the pending buffer; the pod's first message was once lost
to that. Done.

### Chunk 03: The remaining handover loss (open)

Still failing, rarely: 3 of 216 stress runs after chunk 01 (7 of 192 before), one message each, always at the
boundary where the old connection stops and the new one starts receiving. An observer subscription saw every lost
message (the server received it); a spy on every pod connection's raw parser input found none of them (no pod
connection was sent it); no error, disconnect or slow consumer anywhere. Keeping the old member subscribed until
retirement made it much worse (15 of 72), so removing a queue member is the dangerous step. Next: server tracing
narrowed to one subject, the queue-group selection and retry path, nats-server issues and changelog, a valid
no-renewal control; if the cause is nats-server, a minimal reproduction and a 3tears workaround (never remove a queue
member until its replacement is proven receiving).
