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

### Chunk 03: The old half stays in its group until the hold is over

Root cause, traced on the server (nats-server 2.14.2 and 2.15.2, `-V`): the server matches each published message to
a queue group's members as they stand when it matches, and delivers a moment later; under CPU load 8 ms passed
between a match and its delivery, against 2 ms between the successor's `SUB` and the old member's `UNSUB`. The
delivery found the one member of its match gone and dropped the message: never tried against the successor, no error
anywhere (trace: seq-155191 parsed at .018223, `SUB` m34 .020076, `UNSUB` m33 .022330, delivered only to an observer
.026869). Minimal reproduction, plain nats-py and a plain server, no 3tears: `qhandover.py` (losses in 10 of 12 runs
in a Docker network; 12/12 clean with no handovers, 6/6 with only additions, 6/6 with 20 ms between the `SUB` and the
`UNSUB`).

Workaround: each old half stays in its group for `retire_after` (the longest request the client declares) before
`_stop_routing`, so no match that old remains when it leaves. It narrows the race to a match outliving the hold; no
client can see another publisher's pending match, so it is not a proof. Test
`test_the_old_half_stays_in_its_group_until_the_hold_is_over` fails with the hold removed. Stress (6 parallel loops):
0 messages lost in 216 runs (was 7 of 192, then 3 of 216 after chunk 01). The stress test now retries a renewal whose
successor connect timed out (the callout answers on the test's saturated event loop), as the renewal loop does; 3 of
216 failed on that before.

Upstream: Pace decides whether to file the reproduction with nats-server.

### Chunk 04: A renewal that fails after subscribing its successor loses nothing routed there

Two strict stress runs lost a block of thousands of messages, each after successor-connect timeouts. A successor's
subscription is a member of its group from its `SUB`, but nothing received from it until the handover finished; a
renewal whose round trip on the successor then timed out closed the successor outright, dropping everything routed
to it. Now a successor's subscription receives from its `SUB` (`Subscription.subscribe_on`), and a renewal that will
not complete stops routing to each one and drains the candidate rather than closing it (`_discard_candidate`, logged
as a WARNING). Test `test_a_renewal_that_fails_after_subscribing_its_successor_loses_nothing_routed_there` fails on
the earlier code. A first cut tracked receivers by `id()` and lost messages when an id was reused (the stress run
caught it); receivers are tracked by object.

The stress harness, strict: every one of the 100 renewals goes through (up to 5 tries for a timed-out successor
connect, as the renewal loop retries) and nothing is lost. The callout answers on its own thread, loop and
connection (it starved on the test's saturated loop); the final wait runs until nothing arrives for 5 s, capped at
120 s, which fails with what is outstanding. The gated test waits for the handover itself, not a fixed second.

Proof: 216 of 216 strict stress runs clean (6 parallel loops). The give-up path did not fire in that run, so the run
shows no block loss with the fix and the harness fixes; the unit test is what pins the give-up path.
