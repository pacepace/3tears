# Changelog

All notable changes to the 3tears platform packages are recorded here.
This project follows semantic versioning across all workspace
packages (bumped in lock-step).

## Unreleased

**Fix: four names were public in practice and absent from the declared surface (`3tears`).**
`threetears.core` resolves `Keyset`, `Page`, `decode_cursor` and `encode_cursor` through its lazy
PEP 562 map -- they import, and this package's own pagination tests use them through
`from threetears.core import ...` -- while `__all__` listed only `CursorError` beside them. So
`import *` missed them and anything reading `__all__` did not know they existed.

The test meant to catch this asserted `set(__all__) <= set(_LAZY)`, a subset, while its module
docstring claimed to pin a "three-way agreement". A name reachable but undeclared passed. It now
asserts set EQUALITY in both directions, because `_LAZY` is the mechanism and `__all__` is the
declaration, and a name in one and not the other is a surface nobody decided on.

Found by enabling ruff's `F` rules, which recognised the four as undeclared re-exports.

**`F` is now on repo-wide, and clearing it found two more real defects.** A flag set and never read
in a shipped datasources driver (`error_raised`, whose apparent intent was to label a latency
histogram by outcome -- deleted rather than wired, because adding a metric label is a decision
about dashboards, not a lint fix); and an exclusion set that a tools enforcement test built and
then ignored, its return statement carrying a hardcoded set instead, so the module its own comment
named was never actually excluded.

The rest was volume rather than substance: 149 unused imports, mostly one copy-paste pattern in one
package's test suite, plus three dead locals. Every finding was cleared BEFORE the rule was turned
on -- a gate that fails on a clean checkout is a gate people learn to skip.

**Two ledgers are keyed on `path:line:symbol`**, so an autofix of that size invalidated both:
fifteen entries across `_underscore_exemptions.txt` and `_fake_parity_exemptions.txt` were
realigned, and five rationales that carry-forward could not match were recovered from git rather
than rewritten from scratch.

**Fix: an undefined name reached a container past both lint and type checking (`3tears-scrape`
sidecar).** `suppress` was used after an earlier change removed its import. `./scripts/lint.sh`
passed, `./scripts/typecheck.sh` passed, and the container would have raised `NameError` on the
first hung window-manager call.

**Both gates were blind by configuration, not by accident.** Ruff's default rule set is in force
nowhere in this repo: the root selects `SLF` alone, and a nested `ruff.toml` is a full override
rather than an extension, so the sidecar's restated the same narrow set -- F821 (undefined name)
was running in no directory at all. And mypy's file list covers `src` trees only, so it never looks
at this one. The only thing between that `NameError` and a booted container was
`./scripts/test-sidecar.sh`, and it caught this instance solely because a test happened to cover
that branch.

`packages/scrape/sidecar/ruff.toml` now selects `["SLF", "F"]`. Adopting all of F there cost two
unused imports, and the rule was verified against the real defect: removing the import again yields
`F821 Undefined name 'suppress'`.

**This is the second time this directory's lint has let something through that then ran** -- the
first was a formatter autofix writing a syntax error, recorded under v0.19.3 -- which is why the
answer is a rule rather than more care. Running F across the whole repository additionally found
**two live undefined names** in a channels test's annotations, now imported; they were harmless only
because `from __future__ import annotations` never evaluates them, which is the same mechanism that
had already broken the operator WebSocket route on this branch.

Root-wide adoption is not done here: it reports 159 findings, almost all unused imports, which is a
sweep rather than a fix and not one to start while closing a release.

**A hung window-manager call no longer leaks a process.** `asyncio.wait_for` cancels the
`communicate()` await and leaves the CHILD running, so a call that never answered held a process for
the life of a container meant to run long and unattended. It is now killed and reaped, and both are
asserted -- the reap had been claimed in a comment only, and deleting it left the suite green.

**An operator now sees their target and nothing else (`3tears-scrape` sidecar).** Chromium must own
at least one window or it exits, and the warm-up render disposes of its own tab, so exactly one
window always survived doing nothing -- showing the new-tab page, which on this image renders a
search engine's home page. A person summoned to clear one challenge arrived at a display holding
their target next to something that looked exactly like a usable browser.

Not tidiness. That window belongs to the DEFAULT browser context, so it was the one place on the
display where what somebody types is not isolated per target -- which is the promise the rest of
the surface keeps. The people doing this work are not the people who should have to work out that
a window is scenery.

It is hidden at startup with `wmctrl` (skip_taskbar, skip_pager) and `xdotool` (iconify), both
added to the image. Both are needed: a minimised window a taskbar still lists is still one click
away. Chromium is also launched straight onto `about:blank`, so no window in the container's life
shows anything worth clicking, not even before the hide runs. Hidden rather than closed -- closing
the last window exits the browser and takes the container's purpose with it -- and every
window-manager call is best-effort for the same reason.

**An openbox rule was tried first and does not work**, recorded because the next person will reach
for it too: openbox applies `<application>` rules when a window is first *mapped*, and Chromium
maps before its page title arrives, so a title-matched rule never fires. Observed on the live
display, where the window stayed viewable with an empty `_NET_WM_STATE`. Removed rather than left
in place.

**Fix: the operator's WebSocket route had never once worked (`3tears-scrape`).** It shipped with
FastAPI imported inside `build_operator_router`, to keep the `hitl` extra genuinely optional. That
does not compose with `from __future__ import annotations`: every annotation in the module is a
string at runtime, and FastAPI resolves a handler's annotations against the handler's own
`__globals__` -- the defining module's namespace -- where a name bound only as a local variable
does not exist. FastAPI treated `websocket` as a request field, failed to validate it, and closed
every upgrade with **1008**.

1008 is the worst code it could have picked, because it is also what a refused token gets. A dead
route and a rejected operator were indistinguishable from outside, and nothing exercised the route
end to end -- the tests covered the relay and the token extraction in isolation, which both worked
perfectly. A first attempt at a regression test asserted the close code and passed while the route
was still dead.

Fixed structurally rather than worked around: the route wiring moved to `operator_routes.py`,
which imports FastAPI at module scope so annotations resolve, and `operator.py` imports that
module lazily instead of importing FastAPI lazily. The extra is exactly as optional as before --
nothing reaches it until a caller asks for a router -- and the shape that caused the bug is gone.
The tests now assert the injected authorizer was **consulted**, which is the only evidence the
handler ran.

**The operator page and the noVNC client now ship from the router, and the sidecar is down to one
job (`3tears-scrape`).** `build_operator_router()` serves the page at its own mount root, the
vendored client beneath it, and the RFB WebSocket beside both. Mount it under any prefix at any
depth; every URL it emits is relative, so it never learns where it ended up.

**Served at the root, and that is load-bearing.** Relative URLs resolve against the directory the
page came from: at `.../hitl/` the WebSocket resolves to `.../hitl/ws`, and at `.../hitl` it
resolves to `.../ws` -- one directory too high, no route, and an operator who sees only "Failed to
connect". A request without the trailing slash is redirected to the one with it, and a test mounts
under a deep prefix rather than at the root, because at the root every wrong answer works.

The static client tree is mounted BESIDE the routes rather than over them. A mount matches
everything beneath it, including an upgrade `StaticFiles` cannot serve -- it dies on its own
`assert scope["type"] == "http"`, reaching the operator as a 500 that looks like a dead display.
The sidecar hit exactly that, because there the tree was the socket's parent and only registration
order kept them apart. Here nothing the mount could claim is a route, so ordering does not matter,
which is why moving it to the router root fails a test.

**What the sidecar lost, and it is more than the relay.** `websockify` and Debian's `novnc` are
gone from the image, with the noVNC root, the client path, the operator page, the WebSocket relay,
the token subprotocol handling and `authorize_token`. `EXPOSE` is down to the API port and
`VncSession` carries only a display. The AGPL container is now what the design says it is: Xvfb,
Chromium, nodriver, `x11vnc`. `x11vnc` binds loopback, which on Kubernetes means reachable by the
MIT container sharing the pod's network namespace and by nothing else -- that binding IS the
access control on the display port rather than a hardening extra.

`SessionManager` stays, deliberately, and so does `authorize(session_id, token)`. The tab
machinery drives nodriver and cannot move; and removing the last capability check from endpoints
that hand back raw cookie jars is a security reduction, not a cleanup, so it is a separate
decision rather than a consequence of this one.

Every property that lived in a deleted test moved with the thing it described: the token comes
from the fragment and never the query string, no URL the page emits is absolute, the operator
arrives connected on a desktop that scales, and a refused operator never causes the display to be
resolved. That last one needed rewriting rather than copying -- the original watched
`asyncio.open_connection`, which now happens inside the relay, so it watches the injected display
collaborator instead.

**Control messages for a session now find the pod holding its display (`3tears-scrape`).**
Everything that acts on a display -- putting a target in front of the operator, taking a cleared
one back, ending the session -- has to reach one specific pod. The two obvious arrangements are
both bad: addressing pods directly makes the caller track which pod is which and re-track it
after every reschedule, and ingress stickiness makes the routing layer responsible for a fact it
cannot see. So messages are addressed to the SESSION and find their own way, over
`threetears.nats.serve_owner` and `forward`, on a subject derived from the session id.

Four messages act on a live session: open a tab, complete a tab, close the session, read its
state. **Opening a session is deliberately not one of them**, and the asymmetry is the mechanism
rather than a gap -- taking the claim is what MAKES a pod the owner, so routing it to an owner
would be circular. A caller opens a session by claiming it and addresses everything afterwards
to the session it now owns. That is pinned by a test, because it is the one message somebody
will reasonably expect to find.

**A completed tab is sealed before it goes anywhere.** The raw export is the cookie jar of a
target a human has just cleared: a live credential. It travels one loopback hop from the sidecar,
which holds no key by design, and it stops at the pod -- the first point in the path that holds a
key at all. What goes on the bus is ciphertext with an expiry, because a bus is a place other
subscribers can be granted a read of. The raw key is removed from the reply rather than
overwritten, so a jar cannot survive by an exception unwinding past the reassignment, and the
test asserts against every payload the bus carried rather than against the reply alone -- checking
the reply would only prove the field was renamed.

**A pod that has lost its claim refuses every message from the moment it loses it.** Dropping
the subscription is prompt but not instantaneous, and in the gap a message can still arrive at a
pod that has stopped being the owner; acting on one would drive a display the new owner is using.
Ownership is therefore re-checked per message, with the subscription as the optimisation and the
check as the mechanism. Serving refuses to start at all on a claim already lost, since
subscribing would advertise ownership this pod does not have.

Nothing here frames its own errors: `forward` already carries a handler's exception back as a
type name and message, and a second envelope on top would be a second thing that can disagree
with the first.

**One pod holds a session's display, and knows when it stops (`3tears-scrape`).** A session is
one operator working one display, and on Kubernetes that display lives in exactly one pod while
the operator's WebSocket lands on whichever pod the ingress routed it to. Nothing stopped a
second pod deciding it also served the session, and the cost is not a race that resolves: two
Xvfb displays, two browsers, and a human driving whichever one their socket reached while the
other collects half a solve. `claim_session` is the claim, and it refuses rather than queues --
the holder is a person working a page, so waiting would hold a caller open for minutes to hours
and tell nobody anything.

**It is built on `KVLease` rather than `nats_distributed_lock`, which is the closer-looking
fit.** The lock is one context manager, owns its own heartbeat, and its own docstring pairs it
with `serve_owner`. It is still the wrong primitive here for one specific reason: its heartbeat
is an unconditional `bucket.put`. A holder that stalls long enough for its entry to expire, and
whose key another pod then wins, overwrites the winner's entry on its next heartbeat -- two
holders, no error. `LeaseHandle.refresh` is a compare-and-swap against the recorded holder and
raises `LeaseLost` instead. What KVLease lacks is only the renewal loop, and that loop is short
because its whole job is to react to the exception the lock cannot raise.

**A claim is given up on evidence or on time, never on one failed call.** `LeaseLost` is
authoritative and acts immediately. An unreachable coordination layer is not evidence of
anything, so a blip is ridden out -- but a claim un-renewed past its TTL has expired whether or
not this pod noticed, and another pod may already hold it, so the deadline gives it up. Holding
on *because* renewals are failing is the exact inversion of the safe reading.

Three sharp edges are refused rather than reinterpreted. A sub-second TTL truncates to zero at
the coordination layer, writing an entry stale the instant it lands, so it raises. A refresh
interval longer than the TTL lapses a live claim under its own holder, so it raises. And a
deployment with no lease -- which the compose file in this repo is -- still runs, but says so at
WARNING: silence there means two operators on two displays believing they share one, with
nothing anywhere saying why.

Releasing is best-effort by design. The likeliest reason a release fails is that the
coordination layer is unreachable, which is the same reason the claim was just given up -- so it
is the ordinary path out of a lost claim, and raising would replace the operator's real failure
with a cleanup error exactly when the original was the informative one. The TTL frees the entry
regardless; that is what it is for.

**The noVNC client now ships in the wheel, under a licence notice that says so
(`3tears-scrape`).** The human-handover router hands a platform a working display instead of
instructions for installing one. A seam that requires the consumer to go and fetch noVNC
separately is homework, not a seam -- and there is a correctness argument on top of the
ergonomic one that is the stronger of the two: the operator page does
`import RFB from "./core/rfb.js"` and passes `wsProtocols`, and RFB's constructor options have
changed across releases, so a platform-supplied tree turns "did you install the right noVNC"
into a bug class diagnosed from outside the process. Owning both pins them together.

noVNC v1.7.0 is vendored unmodified at `src/threetears/scrape/operator_assets/novnc/`: `core/`
and `vendor/pako/` only, roughly 740K rather than 2.8M, because `core/` imports nothing outside
itself but pako and references no image, stylesheet or translation. The operator page replaces
noVNC's own UI, so `app/`, `po/` and `vnc.html` are not shipped.

**The licence obligation is met specifically rather than by a line in a README**, since noVNC is
MPL-2.0 and this package is MIT. Redistributing MPL-2.0 files inside an MIT wheel is permitted;
what it requires is that the licence text and copyright notice travel with the files, that the
source stays identifiable, and that any modification is marked. So the upstream `LICENSE.txt`,
`AUTHORS` and every text they reference ship beside the code and again in the wheel's
`dist-info/licenses/`; `novnc-provenance.json` records the version, the source archive and a
digest of the tree; and the declared expression is now `MIT AND MPL-2.0 AND
LicenseRef-noVNC-DES` rather than plain `MIT`. That makes `3tears-scrape` the only compound
entry in a family where every other package declares plain `MIT`, which is correct: it is the
only one redistributing somebody else's files. `LicenseRef-noVNC-DES` is
`core/crypto/des.js`, which carries two bespoke permissive grants matching no listed SPDX
licence. The operator page sits as a sibling of the vendored directory rather than inside it, so
no file under `novnc/` can be read as a modified noVNC file.

`modified: false` is checked, not promised: a test recomputes the tree digest, and the fix for a
failure is to mark the modification, never to restamp the digest.

**Two separate mechanisms would each have shipped a dead display**, and both are now held by
test. The stock Python `.gitignore` carries a bare `lib/` rule; with no leading slash it matches
at any depth, so it matched `vendor/pako/lib/` -- every zlib module noVNC's compressed-encoding
decoders import. Twelve files, absent from the repository, with the working tree looking
complete. A `!` re-inclusion fixes that for git and does not fix it for hatchling, which reads
ignore files with its own matcher and does not honour the negation, so the wheel was still built
without them; `[tool.hatch.build.targets.wheel] artifacts` is what covers that half. Nothing
readable from the source tree distinguishes the two cases, which is why one test builds a real
wheel and looks inside it.

**A blocked scrape target now backs off, instead of being hammered forever
(`3tears-scrape`).** Telling a bot wall apart from a site redesign already stopped a
walled target burning its recipe, but it did not make one cheap: the target was still
fetched on every poll, and every one of those fetches produced a page that failed
extraction and therefore got classified. The classifier's verdict cache does not bound
that, because it keys on a digest of the page's visible text and a real interstitial
renders a per-request id into exactly that text -- so the cache misses on every poll,
forever. The only thing that bounds either cost is not fetching the target, which is why
the fetch rate and the classification rate are two separate claims and both are now
tested over many polls rather than one.

`TargetCircuit` gates the fetch off durable state on the health row
(`circuit_state`, `consecutive_fetch_failures`, `blocked_until`, shipped unwritten in
`v010`). Repeated blocks trip it open, an open circuit suppresses the fetch, and each
probe that finds the wall still standing doubles the wait up to a ceiling -- a decay,
not a floor. A transport failure shares the circuit with a wall, since both mean the
content did not arrive, but only a wall stamps `last_blocked_at`, so that column keeps
meaning "walled" rather than drifting into "something went wrong". A suppressed poll
persists nothing: no observation was made, and a row per suppressed poll would write
more the harder the backoff worked.

**No new state machine was written.** The three states, the failure threshold, the
OPEN-to-HALF_OPEN promotion and what a probe's outcome does are all
`threetears.models.circuit_breaker.CircuitBreaker`'s, reached through a new
`CircuitBreaker.restore()` classmethod (`3tears-models`): the durable row is hydrated
into a real breaker, the transition is driven by calling it, and the resulting state is
written back. That seam is the point -- a consumer keeping circuit state in its own store
should not also keep its own copy of the rules, because a second copy is a second copy
that can disagree. `restore()` deliberately does not restore an in-flight probe: that
belongs to the process that issued it and no other process can observe it.

The collaborators below are optional and injected, never constructed, because each belongs to
infrastructure `3tears-scrape` does not own: a per-target `CircuitBreakerLike` lookup for a
free in-process fast-fail before any I/O (the same structural seam `core.http_client`
already uses, taken as a lookup because one `TargetCircuit` serves many targets and a shared
breaker would let one walled target fast-fail the rest -- `CircuitBreakerRegistry.get` fits
it directly, though a long-lived process should inject a bounded lookup, since that registry
never evicts and a scrape target is a far larger key space than the provider name it was
built for), a `WindowedCounter` so several pods polling one target reach the threshold
together instead of each carrying a share that never gets there, a capacity-one `TokenBucket`
for the cross-pod single-probe admission a restored breaker structurally cannot give, and a
`ReprobeScheduler`. With every one of them absent the circuit still decays a blocked target's fetch
rate off the health row alone.

Two consequences of the durable and in-process circuits running on different clocks are
handled rather than left latent. A suppressed fetch resolves an in-process probe that will
now never happen -- but only where one was genuinely admitted, since reporting a failure to
a breaker that admitted nothing trips it on fetches nobody attempted, after which the wrong
circuit answers and the caller is told to retry in seconds when the truth is hours. And
because `restore()` cannot carry an in-flight flag across a process, a restored HALF_OPEN
breaker consults no timer, so the promotion writes `blocked_until` as the probe's own
reservation: a caller that dies between the fetch and the outcome report leaves a HALF_OPEN
row, and without the reservation that row would be fetched on every poll with every
individual state transition still correct.

`3tears-scrape[reprobe]` is a new extra carrying the `ReprobeScheduler`: `reprobe.py` books the next
probe as a `3tears-scheduled-jobs` `relative_delay` job rather than sleeping, for a caller
that is event-driven rather than polling (a poller's next poll already is the re-probe).
The job id is derived from the target, so re-booking replaces the outstanding probe --
with a random id every superseded booking would survive and eventually fire, turning the
longest backoff into the biggest burst. The extra is optional because scheduled-jobs
brings NATS and APScheduler with it, and nothing in the default install imports it.

**A suppressed poll reports `validation_status: "backoff"`, not `"blocked"`.** In this
package `"blocked"` is a fact about the target -- a bot wall stood where the content should
be -- and the same circuit also opens on repeated transport failures, so the old value told
a consumer that a host which had simply stopped answering was challenging it. The status
now describes the poll rather than the target, which is the same reason a suppressed poll
persists no extraction row: it observed nothing. Whether a target was ever walled remains
`last_blocked_at`'s question to answer.

The probe reservation is honoured whether or not a `TokenBucket` is configured. The bucket
and the reservation look like two answers to one question and are not: the bucket bounds how
many pods probe at once and refills at a constant rate, while the reservation bounds how
often a stuck target is probed and decays. Deferring to the bucket swapped the decay for a
floor, and only in the deployments that had configured one.

`TargetCircuit.release_probe()` closes the permitted path's version of a hazard the
suppressed path already handled. A permitted decision can promote the in-process breaker and
mark its probe in flight; that flag is cleared only by an outcome, so a caller raising
between the fetch and the report left the breaker holding it for the life of the process,
fast-failing the target ahead of the durable row and answering "retry in about 0s" forever.
The durable side needs no equivalent, because its own promotion already stamped a
reservation that outlives the process that abandoned it. `breaker_for` is correspondingly
typed to `ProbeObservableBreaker` -- the three-call protocol plus a readable `state` -- since
a probe this module cannot see admitted is a probe it cannot release, and a breaker that
could not answer that was previously accepted at the seam and wedged at runtime.

The same release covers a cancelled fetch. `driver.render` is guarded by `except Exception`,
which a `CancelledError` is not, and it is the longest await in the call -- so cancellation
is where a strand most often lands. The cancellation is not persisted as a fetch outcome:
a shutdown is not evidence about the target, and a durable failure would back it off across
every pod and outlive the process that was cancelled. The in-process breaker does take the
failure, because the three-call protocol has no "never mind", but that is seconds-scale,
process-local, and dies with the process anyway.

`"backoff"` is deliberately NOT added to the `ValidationStatus` Literal, whose four values
are the domain of what gets stored on `ScrapeExtraction`. A suppressed poll stores nothing,
so admitting it would declare a storable value that can never be stored; the four existing
values each describe a page we did or did not receive, where `"backoff"` describes a fetch
we declined to attempt. The scrape README and the design doc record the distinction, since
the tool's JSON payload is where a consumer meets both.

The OPEN-to-HALF_OPEN promotion now books a re-probe as well as stamping a reservation. A
`relative_delay` job is terminal, so the job booked at trip time is spent once it fires: a
dispatcher that fires it, promotes the row, and then dies before reporting an outcome left a
HALF_OPEN row with a live reservation and nothing left that would ever revisit it. That is
the crash the reservation was invented for, and it was solved for a poller -- whose next poll
is the re-probe -- and silently not for the event-driven caller `reprobe.py` exists to serve.
Bookings are keyed by target, so the outcome report that normally follows replaces this one.
A recovery is the exception, since closing the circuit books nothing and so cannot supersede
the outstanding booking. `ReprobeScheduler` therefore gains `cancel_reprobe`, called when the
circuit closes: without it the last booking survives and fires against a target that already
came back, which is a whole poll including its eval loop rather than a bare fetch, and it
leaves a job row behind for every target that ever tripped. The cancel deletes rather than
expires, and is safe to issue blind -- `Collection.delete` is idempotent and returns `True` on
every path, so a booking that was never made costs one no-op rather than an exception, and a
caller closing a circuit never has to find out which it was. Nothing is logged above DEBUG for
the same reason: the return value cannot tell a real cancellation from a no-op, and a close
happens for every target that recovers, including the many that never tripped.

**A success reported against a circuit the breaker leaves OPEN no longer erases its backoff.**
`CircuitBreaker` answers a success from a request it never admitted by leaving the circuit
open, which this module adopts rather than overrides. Clearing `blocked_until` on the way past
turned that conservative answer into its opposite: a missing window restores as nought seconds
remaining, so the next check found an open breaker whose recovery had elapsed, promoted it and
probed -- the state column still reading OPEN while the backoff it names had been discarded.
Reachable across a fleet with nothing failing to persist, when one pod trips the row while
another's already-permitted fetch is in flight. The window is now cleared only by an actual
close, which writes no transition rule: the window is this module's storage, not a state.

**`TargetCircuit.forget_target()` is the retention story, and it is manual on purpose.** Both
tables this writes are keyed by target and upserted rather than appended, so neither grows
with time or poll count -- but both grow with distinct targets, and an ad-hoc target id is
derived from `(url, field_schema)`, so a long-lived process accumulates a row per URL it has
ever scraped. There was previously no way to reclaim any of it. It is not automatic because a
health row is not garbage: it carries the fingerprint that stops a target being re-classified
on every poll, so evicting one for a target still being polled costs exactly the LLM calls
this design exists to avoid, and no TTL can distinguish a retired target from a quiet one.
Only the caller knows which is which.

The three `BackoffPolicy` defaults now carry their reasoning rather than appearing as bare
numbers -- three failures because two in a row is ordinary bad luck, fifteen minutes because
it is sized against the poll interval it protects rather than any vendor's undocumented
cooldown, six hours because a doubling curve otherwise passes a day within a working shift
and a target blocked overnight should be probed by morning without anyone intervening.

**A human can now clear a wall the scraper cannot, and their work is reused
(`3tears-scrape`, `3tears-core`).** The rest of the human-in-the-loop path, plus two
capabilities that are not scrape-specific and are not in this package.

The sidecar's Xvfb display is reachable on demand: `x11vnc` starts when a person arrives and
stops when they leave, bound to loopback, with the display number a parameter so a display pool
is later configuration rather than a rewrite. The operator reaches it through one origin that
also authenticates it. (This entry described that origin as the sidecar's own port, with
`websockify` still running beside it; a later entry in this same release moved both the client
and the relay into the MIT container and removed `websockify` outright. The arrangement above is
what the display path looked like partway through, not what ships.) On top of that sits one
session against that one display -- a bounded number
of targets at a time, each in its own isolated browser context so a second target cannot see
the first's cookies, behind a hard TTL and a token this container minted. The token proves
only that; deciding who was entitled to it belongs where identity lives, which is not here.

Nothing is held while a target waits for a person. It is reported and forgotten, and
re-driven from `url` plus `nav_steps` when an operator actually arrives, so waiting costs no
container resources at all.

When they finish, the context's cookies and `localStorage` are exported, sealed with
`core.security.encryption` under an operator master key, and stored on the health row with an
expiry. Later unattended renders send them, so the target extracts normally without anyone
watching. The sidecar holds no key and seals nothing -- the container driving a browser for
arbitrary targets is the one you least want holding a decryption key. Every way the stored
state can fail (wrong key, tamper, format change, missing or passed expiry, no key configured)
degrades to "this target needs a human again", never to sending a dead cookie and believing
the answer. `localStorage` is named because the export always captured it while the apply path
dropped it, so a site keeping its session there came back needing a human anyway.

Two seams make that loop usable by a platform, which owns the queue and the operator.
`ScrapeTargetHealthCollection.list_walled()` answers "which targets need a person" -- the only
non-primary-key query in the package, filtered so a host that merely stopped answering does
not queue somebody who arrives with nothing to clear. `TargetCircuit.record_human_cleared()`
lifts the suppression afterwards, which nothing else can: `record_reachable` reports a FETCH
that succeeded, and a success from a request the breaker never admitted deliberately leaves
the circuit open. Without it the solve is stored and the next poll is still suppressed.

**`threetears.core.egress` is new, and deliberately in core.** An `EgressDriver` seam with
`direct`, any proxy URL, and a SOCKS constructor covering TOR and most VPN sidecars, wired
into `core.http_client` -- which already called itself "the one transport" and already had a
transport seam, and httpx proxying IS a transport. A driver answers both what httpx needs and
what a browser needs, because a deployment whose API calls proxy while its scrapes go out
direct is worse off than one with no proxying: it believes it has the property. An unknown
driver name raises rather than falling back to direct, and an exit with no address is refused
at construction -- both are the same silent failure, an exit that reports itself as `tor` and
leaves by the container's own IP. Nothing here starts a daemon.

Egress is per target, not per container. The sidecar's `EGRESS_PROXY` is a default applied at
browser launch, and a render may override it with `egress_proxy`, which gets its own browser
context -- `Target.createBrowserContext` accepts a `proxyServer`, where the `--proxy-server`
flag is process-wide. `last_egress` on the health row (migration `v011`) records which exit an
observation was CONFIGURED to come from -- reported by the fetcher rather than assumed by the
caller, so a dropped proxy argument shows up as a mismatch. It is not evidence that traffic
left that way: a per-context proxy Chromium accepted and then ignored would still be recorded
under the name it was asked for, and confirming otherwise needs an observer outside the
process. What it buys is telling "walled" apart from "walled from this exit", so a working
alternative is not left untried. `None` means no exit was configured, which is a different fact
from choosing the default route; that choice is `DirectEgress` and records as `direct`.

Egress is wired separately on the drivers and on `ScrapeTool`, and `ScrapeTool` warns on either
half being wrong. Drivers proxied with an unproxied gate leaks the container's address on the
`robots.txt` read in front of every fetch; the gate proxied with an unproxied driver leaks it on
the page fetch itself. Both are invisible otherwise, because both halves work. The backends that
honour an exit are `ApiDriver` and `NodriverSidecarDriver`, plus the wrappers that delegate to
them -- the warning names any driver that cannot.

**`robots.txt` is honoured, both halves on by default.** `Crawl-delay` is waited between
fetches of an origin; a `Disallow` is escalated for a person rather than fetched unattended or
silently skipped. A human working the page over VNC is not an automated agent, which is what
makes that escalation close rather than dead-end -- and `Crawl-delay` deliberately does not get
that exemption, because load on someone's server is caused equally by either. Every unusable
robots file means "allowed": treating an unreachable text file as a refusal lets one bad
response stop a scrape silently. Parsing is `urllib.robotparser`, so no new dependency.

A disallowed target becomes a queue item rather than a dead end.
`ScrapeTool` builds a gate unless one is passed. There are two ways off: `robots=None`
removes the gate entirely, and a gate whose policy has both behaviours disabled stays in
place, keeps its overrides, and does not fetch the file at all -- so turning politeness off
does not leave a request going out to every new origin purely to discard the answer.
Migration `v012` adds `robots_blocked_at`/`robots_blocked_reason`, and `list_walled()` now
answers with BOTH kinds of target a human is needed for -- a bot wall and a robots refusal --
which widens what that method returns. The robots columns are deliberately not the circuit's:
a policy decision is not a fetch failure, and counting it as one would back off a site that
works perfectly. A block is stamped once rather than per poll, and is cleared both when a
human clears the target and when the file stops disallowing us, so the queue empties as well
as fills.

The default robots fetcher leaves by the tool's configured exit. On-by-default politeness with
an unproxied read would have disclosed the container's real address to every origin
immediately before the proxied fetch that was meant to hide it -- a deployment with one exit
configured and two in reality.

**The VNC display is now operable by a human, which it was not (`3tears-scrape` sidecar).** Four
defects, every one invisible to a green suite because every test asserted the contexts were
isolated and the slots accounted for, and every one of those assertions was true. The windows
existed, held the right cookies, and could not be used.

There was no window manager: bare Xvfb maps windows in creation order with no titlebars and no
click-to-focus, so with four slots only the last target opened was reachable. `openbox` fixes
that, and `tint2` gives a taskbar, without which switching is blind and a minimised window has
nowhere to come back from. openbox's four virtual desktops with the mousewheel bound to
`GoToDesktop` are collapsed to one and the gesture unbound -- a stray scroll moved the operator
to an empty desktop where the panel was still drawn but held nothing, which reads as a missing
taskbar rather than a switched desktop.

And the client URL handed to every operator since the VNC shipped carried `resize=scale`, which
`vnc_lite.html` does not parse: it reads `scale` and does not know the word `resize`. The
parameter had never once done anything, so an operator on a laptop scrolled a fixed 1920x1080
desktop to reach the taskbar at the bottom of it. The page served is now `vnc.html` with
`autoconnect=true&resize=scale`.

**`UI_SCALE` is the display-scaling setting a desktop OS would offer.** Chromium's
`--force-device-scale-factor` plus `Xft.dpi` for openbox and tint2, so the page and the
furniture around it scale together -- either alone reads as a rendering fault. It affects only
what a person looks at: extraction renders never go through it, because a scraped page's layout
must not depend on an operator's comfort setting. Live per-site adjustment is Chromium's own
`Ctrl +`, which persists in the pinned profile.

**`docker-compose.yml` names the tag `docker buildx bake` writes.** It said
`nodriver-sidecar:latest` while bake tags `aibots/...`, so building and then running started
whatever stale image carried the bare name -- silently, with every command reporting success. It
cost a verification run against a nine-day-old container in which the feature under test did not
exist.

**A cancelled poll gives back the crawl-delay turn it took (`3tears`, `3tears-scrape`).**
`TokenBucket.claim` consumes and had no inverse, so the only recovery was refill over time: a
caller cancelled between taking a turn and doing the work held that key's shared budget down for
nothing. Invisible once, and compounding under repeated cancellation -- a pod restarting in a
loop can hold a key near zero while doing nothing at all. `TokenBucket.refund()` is the inverse,
capped at capacity so a double refund cannot mint budget the bucket never had, and it never
raises: it exists to be called from a handler that is already unwinding, where an exception
would replace a self-healing throughput dip with a lost error.

`ScrapeTool` returns the origin's turn when a poll is cancelled after claiming it. Fire-and-forget
rather than awaited, because an `await` inside a cancellation handler re-raises before reaching
the store.

**`ScrapeTool.execute` has one probe guard instead of two.** The two adjacent
`except BaseException` blocks had no `await` between them, so there was no live gap -- but that
shape produced four stranded-probe bugs in a row, each fixed as a symptom, because every new
`await` had to be placed against whichever guard its author happened to be reading. The render is
now `_render_once`, returning `(page, error)`, and one guard covers the whole permitted path, so
the compensation has exactly one home. `execute` is 45 lines shorter.

**A poll that never fetches no longer spends the site's fleet-wide budget.** `TokenBucket.claim`
consumes atomically, and the crawl-delay pacer was being claimed inside `RobotsGate.check` --
which is a question, not a commitment. The circuit can suppress the fetch afterwards, the caller
can change its mind, the driver can be missing. Because the token is shared across pods, polling
one walled target inside its backoff drained a token per poll and delayed every SIBLING target
on that origin: a target behaving perfectly, slowed by one that is not. Taking the turn is now
`claim_fleet_turn`, called only once the fetch is committed, which is the rule `note_fetched`
already enforced for the local clock -- the site pays when we actually visit it.

`EgressRegistry.health()` survives a driver that raises. It is a diagnostic over a
`runtime_checkable` Protocol that invites foreign implementations, so one broken driver replaced
the whole report with an exception -- leaving an operator asking "which exit is down" with no
answer at the moment one already was.

`threetears.core.egress` is reachable from `threetears.core` directly. It was described as new
public API while being absent from the package's lazy export map, so it could only be imported
by its full module path.

**A driver that cannot use a human's solve now says so, once per site.** Only the nodriver
sidecar can apply exported cookies and storage; the other five accept the parameter and render
unauthenticated. In silence that is a trap -- the caller gets a successful page back and learns
nothing until extraction fails on a login wall and the target is escalated to a person who
already cleared it.

The cardinality is the whole design, and both obvious choices are wrong. Per render is a storm:
`MultiDocumentDriver` forwards a solve to its inner driver once per document, so one listing
would emit a warning per document, and a warning that repeats that way teaches its reader to
filter it. Per driver instance is silence: `ScrapeTool` builds its driver map once and reuses it
for the life of the process, so the first target would warn and every later one would be dropped
quietly. An origin is what a solve actually belongs to, so it is the unit -- one report per site,
bounded so a long-lived process cannot accumulate origins forever. `NodriverDownloadDriver`
carries its own remedy, since the general advice ("use the nodriver sidecar driver") names the
thing it already is; its constraint is the `/v1/download` endpoint, which carries no session
state.

**A driver written before this release keeps working.** `ScrapeDriver` ships as a pluggable
contract, and `session_state` was being passed on EVERY fetch -- so an out-of-tree driver
written against 0.19.x raised `TypeError` on every call, including the overwhelming majority
carrying no stored solve at all. It is passed only when one exists. The egress half of the
same change had reasoned about exactly this consumer and reads its attribute through a
`getattr`; the asymmetry is what made this an oversight rather than a decision.

**`egress` accepts a NAME, not just a constructed driver.** `ScrapeTool(egress="tor",
egress_registry=...)` resolves through `EgressRegistry`, which is what that class was built
for and, until now, what nothing did with it. An unknown name raises rather than falling back
to the default route: a deployment that asked for TOR and silently got the container's own
address would look correct in every log line while being wrong about the one property it
configured.

`3tears[socks]` declares `httpx[socks]`, which `SocksEgress` and `WarpEgress` need. It was
reachable only as an ImportError on the first request, which makes a packaging requirement
look like a runtime bug.

**Cloudflare WARP is a named exit, not a configuration people have to discover.**
`WarpEgress()` on `warp-cli mode proxy`'s own default SOCKS port, with commented compose
plumbing beside TOR's. It was expressible via `SocksEgress` all along, which is true and is not
the same thing: a backend nobody can find by name is one the next person reimplements, and the
port is the part everyone gets wrong. WARP and TOR are for opposite problems -- WARP changes
the address a site sees and is far less challenged, while TOR is for non-attribution and raises
the challenge rate.

**An exit can be asked whether it is up.** `EgressDriver.health()` returns an `EgressHealth`
carrying the address the exit actually presents, because a proxy that is listening but
forwarding directly answers a connectivity check perfectly while providing none of the property
it was configured for. `EgressRegistry.health()` sweeps every registered exit at once.

This closes a detection gap rather than adding a convenience. A dead `tor` or `warp` daemon
fails every render transport-side; each target's circuit then opens and backs off for hours,
and those targets are correctly EXCLUDED from the walled queue, since unreachability
deliberately never stamps `last_blocked_at`. Every individual signal behaves correctly and the
aggregate is invisible: "all my targets broke at once" and "one daemon died" produce identical
evidence until something asks the exits directly. A driver that cannot answer reports
unreachable with a reason rather than defaulting to healthy -- an exit nobody can check must
not be the one that looks fine.

**Two wrapping drivers stopped swallowing a human's solve.** `NetworkCaptureDriver` delegates
to an inner driver that is typically the nodriver sidecar -- the one backend that can apply a
session -- and dropped `session_state` on the way, so a solved session was discarded exactly
where it would have worked and the capture returned the login wall's XHR.
`MultiDocumentDriver` forwards it now too. A wrapper that silently withholds a credential makes
the capability depend on which wrapper happens to be in the way.

**A fleet pacer no longer overrides what a site asked for.** With a `delay_pacer` injected,
`RobotsGate` returned the bucket's answer alone and discarded the parsed `Crawl-delay`
entirely -- so a site asking for 30s between requests was fetched at whatever rate the bucket
happened to carry, and only in the fleet deployments where several pods make that delay
matter most. Both constraints bind now: the wait is the longer of the fleet's turn and this
origin's own clock. That branch was previously the module's only untested one, which is how
it stayed wrong.

`RobotsGate` also stops growing forever. Both per-origin stores are caches reconstructible
from a re-fetch, so they self-bound at `max_origins` on a least-recently-used basis, and
`forget()` retires one origin explicitly. The circuit's equivalent state gets a manual lever
instead, because evicting a circuit row would discard a judgement nothing can rebuild.

**Completing a HITL tab no longer holds the session lock across the browser.** The export and
context-dispose are two CDP round-trips, and a wedged browser holding that lock blocked
`reap()` and `close()` -- defeating the hard TTL exactly when it matters, since the stuck
context is the one still holding a target's authenticated session. Popping the tab is the
claim and stays under the lock; the round-trips happen outside it under their own timeouts,
and a failed export costs the state rather than the slot.

**`CamoufoxDriver` says when it drops a session state.** It accepts the parameter and cannot
apply it -- a real gap rather than an inapplicability, since it drives a browser. Silence
meant a caller handed over a session a person had spent real time solving, got a successful
render back, and learned nothing until extraction failed on a login wall and the target was
escalated to a human who had already done the work. Now a warning, and a documented one.

**`fire_and_forget` is now supported public API on `3tears`.** It moves from
`threetears.core._bridge` into `threetears.core`'s export surface because a sibling
distribution needs to schedule a coroutine it must not await, and the alternative was every
consumer reaching across a package boundary into a module whose underscore says it may change.
`_bridge`'s other exports -- `sync_await`, `drain`, `shutdown` -- are deliberately not promoted
and carry no compatibility promise: they drive the bridge's lifecycle, which belongs to whoever
owns the loop.

**The gate can now see the sidecar, and the SLF001 ledger is checked in both directions.**
`nodriver` is AGPL-3.0 and never enters the workspace venv, so the workspace suite carries
`--ignore` for `packages/scrape/sidecar` -- which meant ruff formatted the sidecar's source
while nothing executed the result. An autofix wrote `except OSError, ProcessLookupError:` into
`hitl.py`, a syntax error that passed lint, passed mypy (the sidecar is outside its file list
too) and passed the entire workspace suite. `scripts/test-sidecar.sh` runs that suite against
the sidecar's own interpreter and `check-all.sh` calls it, so it is separate but not optional.

`tests/enforcement/_underscore_exemptions.txt` records why each exempted private access was
judged acceptable, and nothing read it back: the underscore walker scans `packages/*/src` and
never enters a `tests/` tree. It had rotted in both directions -- entries pointing at code that
had moved or gone, and accesses with no entry at all. Both directions are now checked, and they
have to be separate checks, because a missing entry is not a stale one. The reconciliation, the
AST walking and the ruff-config discovery live in
`threetears.enforcement.underscore_access` alongside the walkers whose exemptions they describe,
with thin shells in `tests/enforcement/`.

That discovery reads every ruff config rather than the root `pyproject.toml` alone. A nested
`ruff.toml` is a full override, so a checker built on the root cannot see what the subtree
exempts -- which is how a set of reviewed sidecar entries were deleted with nothing noticing.
Regenerate with `uv run python scripts/regen-underscore-exemptions.py`, which carries rationales
forward by `(path, enclosing scope, symbol)` so a line shift loses nothing and two accesses of
one name keep their own reasons; hand-editing the line numbers is what
the checks exist to catch.
## v0.19.3 -- 2026-07-26

**`NoRespondersError` and `RequestTimeoutError`, so callers can tell "nobody is
subscribed" from "nobody answered."** `RequestError`'s own docstring had said
"distinct subclasses may be added later if callers need to disambiguate"; a caller
now does. The two are different operational facts -- the first points at a service
that never started, was never deployed, or is subscribed on another
subject/namespace, the second at one that is present and wedged -- and they send an
operator to different places. Collapsed into a bare `RequestError` carrying only a
message string, the only way to tell them apart was matching on that text, which is
why `forward.py` had already resorted to inspecting `__cause__`.

The caller that forced it: an agent registering with the agent router at boot. On a
cold rollout the router may not have subscribed yet -- an ordinary race worth
waiting out -- whereas a transport or response-decode failure is not. Without the
distinction a retry either over-waits a permanent failure and then misreports it as
unavailability, or reaches around this wrapper into `nats.errors` directly, which
consumers' own enforcement tests forbid.

Both subclass `RequestError`, so every existing `except RequestError` catches them
unchanged. This refines the hierarchy; nothing is renamed and no call site must
move.

## v0.19.2 -- 2026-07-25

**Image builds use uv, not pip.** `threetears-base` installed with pip and every
consumer image inherited that. The platform has been uv-only everywhere else
since the beginning; the Dockerfiles were the last holdout, and that is exactly
where it hurt. pip backtracks across the cross-product of every published version
when a dependency graph is under-constrained, then reports `ResolutionImpossible`
against whichever node it happened to be holding rather than the package actually
in conflict. One such message named an innocent, correctly-installed package and
cost most of a day. uv resolved the identical set in seconds and named the real
conflict.

The uv binary is copied into the **runtime** image, not just the wheel builder,
so every downstream consumer installs into the shared venv with the same resolver
instead of drifting back to pip.

Consumers can now render `uv.lock` at build time with `uv export --frozen` and
retire hand-frozen constraints files entirely.

**`bump-version.sh` moves the intra-family bounds.** v0.19.1 bounded every
sibling dependency to its own minor line, but the bump script did not know about
those bounds. Releasing 0.20.0 would have left every package at 0.20.0 while
requiring siblings `<0.20.0` — a family that excludes itself, unresolvable the
moment anyone installed it. Both the bump path and `--verify` now handle bounds,
so a stale-bound release fails pre-flight instead of shipping.

## v0.19.1 -- 2026-07-25

**Every intra-family dependency is now version-bounded.** The packages release in
lockstep but declared each other with no bound at all -- 84 bare entries such as
`"3tears-observe"` across 25 packages. Each is now `>=0.19.0,<0.20.0`.

Unbounded siblings let pip resolve a MIXED family, which fails in two ways that
are both very expensive to diagnose:

- **A mixed install builds clean and breaks at runtime.** pip paired
  `3tears-object-store` 0.18.0 with an otherwise-0.19.0 family in a consumer
  image; 0.18.0 predates `build_object_key`'s `path=` parameter.
- **Resolution explodes and blames the wrong package.** Across ~17 published
  versions and ~25 mutually-unbounded packages, pip backtracks the cross-product
  and reports `ResolutionImpossible` against whichever node it was holding. One
  such failure named `3tears-agent-tools` as having "no matching distributions
  available" while that package was entirely innocent -- the real cause was a
  stale `protobuf` pin in a consumer's constraints file, three levels away.

Bounding makes a mixed family unresolvable rather than merely unlikely, and
collapses the search space so pip names the package that actually conflicts.

Also corrected five bounds that existed but had gone stale -- `registry`
admitting `3tears-agent-acl>=0.1.0` and `3tears-agent-tools>=0.5.0`,
`enforcement` admitting `3tears>=0.5.0`, `datasources` admitting
`3tears>=0.9.1,<1.0`, and `channels` admitting `3tears-agent-wake>=0.9.0`. Those
are worse than unbounded, because they look deliberate.

`tests/enforcement/test_intra_family_version_bounds.py` now enforces both halves:
no sibling may be unbounded, and no bound may name a line other than the
declaring package's own.

**Consumers should pin the whole family to `0.19.1` exactly.**

## v0.19.0 -- 2026-07-25

**New package: `3tears-geo`.** Slippy-map tile geometry in application code. Every
off-the-shelf tile server assumes PostGIS and calls `ST_AsMVT`; YugabyteDB ships no
postgis extension, so the work happens in Python -- shapely for geometry,
`mapbox-vector-tile` for encoding. Tile addressing is Web Mercator EPSG:3857 with XYZ
orientation (`y` increasing southward, **not** TMS), stated explicitly because the two
conventions differ only in that axis and confusing them renders a mirrored map that
looks plausible.

The package carries: tile addressing and bounds math, a fixed SQL-to-MVT attribute
coercion (NULL becomes an omitted key, since MVT has no null and collapsing it to zero
shades unmeasured regions as though they were surveyed), WKB/EWKB decoding, two zoom
bands, MVT encoding, a per-pod feature cache with a SQLite R-Tree, and `TileCollection`.

Low zoom is **not** simplified high zoom. A z4 tile spans a large fraction of a country,
so rendering it by dropping individual features leaves an arbitrary sample of whichever
survived -- a different, misleading dataset rather than a coarse view of the same one.
The aggregate band rolls rows up to a declared coarser geography and emits real totals;
individual features appear only above the declared crossover.

**New primitive: `DerivedCollection` (`3tears`).** A collection whose key is derived
from a request and whose value is computed on miss. `BaseCollection` caches by primary
key, which serves reads whose identity is already discrete and does not serve reads
whose identity is continuous -- a bounding box, a time window or an offset/limit page
names a region rather than a row, so no two callers produce the same key and the
cross-pod hit rate is zero. Those reads are annotated `# cache-bypass: not by-pk`
throughout this codebase. The fix is quantization: collapse the request onto a discrete
grid and the cell becomes a primary key the existing three tiers already handle
unmodified. Geographic tiles are one instance; hour buckets and pagination pages are
others.

Misses are single-flighted twice: an in-process `asyncio.Lock` per key, and
`nats_distributed_lock` across pods. Derivation is expensive by definition -- if it were
cheap there would be nothing to cache -- so an unguarded miss on a popular key is a
stampede. An integration test against real NATS caught exactly that during development:
a peer-wait budget shorter than a derivation meant every loser duplicated the winner's
work, reintroducing the stampede underneath the lock meant to prevent it.

**`geo:` block on `DatasourceConfig` (`3tears-datasources`).** A product declares its
tileable layers alongside its connection details and writes no map plumbing. Sensitivity
is deliberately not a new field: a datasource already records how exposed it is via
`visibility` and a nullable `customer_id`, and a second place to say it is a second
place for it to be wrong, so a layer may only narrow what it inherits.

**`build_object_key` extended (`3tears-media-contracts`).** Optional customer (absent
yields a grantable `shared/` prefix, mirroring `platform.datasources.customer_id` being
nullable for platform-shared rows) and a caller-supplied deterministic path. A CDN
deriving a storage key from a request URL cannot perform a lookup to translate `z/x/y`
into an opaque object id. The existing key shape is unchanged and its tests pass
untouched.

**`Subjects.datasource_tile_epoch`.** Tile versions per (datasource, layer). A single
global version would discard every layer's edge cache worldwide whenever any one layer
was reseeded.

### Also in this release

**Fix: `3tears-scrape` 0.18.0 was built and then dropped before upload.** The v0.18.0
release published 26 of its 27 packages. A step in `release.yml` deleted the scrape
artifacts from `dist/` between build and publish -- correct when it was written, since
the project did not yet exist on PyPI and so had no trusted-publisher entry -- and its
own comment said to remove it for the release where scrape shipped. Nothing enforced
that, and the only person who would ever have read the comment was someone already
editing `release.yml`, which cutting a release does not require. PyPI has carried
`3tears-scrape` as a reserved name with zero files since.

No version is bumped and no tag moves. The withhold step is gone, and `release.yml`
gains a `workflow_dispatch` path so an already-tagged version can be republished:
`skip-existing` means every artifact already on PyPI is skipped, so the only possible
effect is that a genuinely absent one uploads. The operator types the version, and the
existing lockstep verify holds the run to it.

**Guard: `scripts/verify-dist-complete.sh`.** Asserts `dist/` carries an sdist and a
wheel for every workspace member and fails the build naming any that are missing. It
reads the member globs out of the root `pyproject.toml` rather than restating them, so
a workspace tier added later is covered without touching the script. Verified against
the real failure: delete the scrape artifacts from a full build and it fails with
exactly that name. The republish procedure is documented in `CLAUDE.md` rather than
only in a workflow comment, which is the defect that caused this.

**Feature: per-target fetch health (`threetears.scrape`)** -- the eval loop has always
remembered which extraction strategy won for a target, and nothing at all about the
fetch: whether the page came back, whether it resembled the page the strategy was learned
against, or whether a bot wall was served instead of content. All three of those failures
currently increment the same counter and get the same response, which is why a blocked
target burns through its failure threshold and spends an LLM candidate round learning to
extract data from a challenge page, discarding a recipe that was never broken.

New `ScrapeTargetHealth` entity and `scrape_target_health` table (migration `v010`),
keyed by `target_id`. A separate table rather than columns on `scrape_recipes`, because
health exists for targets that never had a recipe: one blocked before it ever extracted
successfully has real health and no strategy, and giving it a strategy-less recipe row
would need a guard so the reuse path never mistook that empty strategy for a real one.

`content_fingerprint` is a digest of the page's readable text, stamped whenever an
extraction validates, and it is the comparison value that lets a redesigned page be told
apart from an unchanged one. Fingerprinting text rather than markup is deliberate: a site
that reformats its template has not changed what it says, and a fingerprint that flipped on
that would claim the site changed on every deploy the site makes. The circuit, backoff and
sealed-session columns are created now because the shape is settled and one `CREATE` beats
several `ALTER`s against a table this young.

`run_eval_loop` and `run_eval_loop_multi_row` take an optional `health_collection`;
omitted, as every existing caller omits it, nothing is written and nothing else differs.

**Feature: a failed page is classified before it is acted on (`threetears.scrape`)** -- the
fix for the recipe destruction described above. When a stored strategy stops matching, the
eval loop now asks what the page actually is before deciding what the failure meant, and
routes on the answer: a bot wall leaves the recipe byte-identical and persists an extraction
with the new `validation_status` value `"blocked"`; a page that genuinely changed regenerates
on the **first** failure instead of the third; anything else keeps today's behaviour exactly.

Detection is a question, not a marker list. Matching a vendor's current interstitial markup
is a hand-written parser for one page as it looks this week, and vendors reword these pages,
so a fixture set captured today specifies nothing about tomorrow -- and the rot is silent in
the worst direction, a stale marker meaning a blocked page is read as "the site changed" and
its recipe burned. New `threetears.scrape.challenge` holds the verdict model and the prompt;
it contains no vendor string, so a wall it has never seen classifies on meaning.

Cost, stated honestly rather than as a slogan. Classification is never the first question
asked. A page whose readable text is identical to the one the strategy last validated
against provably has not changed and provably is not a new wall, so it counts the failure for
zero model calls, exactly as today. A page already classified reuses its verdict from the new
`classified_fingerprint` / `classified_verdict` / `classified_evidence` columns. Only a page
that is both different and unseen costs a call -- one, in exchange for the entire
candidate-generation round a blocked target burns today, and for regenerating two polls sooner
when the site really did change. A cached `"changed"` verdict also records that regeneration
has already been tried against that exact page, which is what stops an unlearnable page burning
a candidate round every poll.

The verdict cache bounds cost only for a wall that renders the same bytes each time. The
fingerprint digests visible text, and a real Cloudflare interstitial renders a per-request Ray
ID into exactly that, so such a target costs one classification **per poll** rather than one
while walled. Still cheaper than the candidate round it replaces, and it no longer destroys the
recipe, but not a bounded cost. Normalising ids out of the fingerprint was rejected: it puts
vendor-shaped pattern matching back into the one place this design removed it, and would
suppress genuine content changes that happen to look like ids. What bounds a walled target is
not fetching it every poll, which is the circuit backoff still to come.

Both entry points take an optional `page_status` (real evidence for the classifier, though
rarely decisive since most walls return 200) and `classifier_model_id`. A classifier that
cannot answer degrades to precisely today's behaviour: an unanswerable question is never
more destructive than not having asked one.

`ScrapeTool.__init__` gains a matching optional `health_collection`, and forwards
`page_status` from the page it just rendered. Both are keyword-only and default to the old
behaviour, so no existing construction changes. Without this the feature had no caller in
this repo at all: the tool was holding the status and passing neither, which makes a
parameter plumbing rather than a capability.

**Operator-visible log strings changed** in the strategy collapse, which is worth stating
because anything grepping them will stop matching. The per-strategy reuse and no-survivor
messages are now prefixed by the shape's own label rather than hand-written per function, so
`scrape row recipe reuse: ...` reads `scrape row eval loop recipe reuse: ...`, and the regex
row variant reports `rows_matched=` where it said `matches=`. Same events, same fields, same
frequency. A spent classification call also now logs before and after, so a fresh
`content`/`empty`/`other` verdict is no longer invisible -- previously only the free cached
path announced itself, which is backwards for the one branch that costs money.

The four recipe-reuse paths were restructured into pure validators plus one shared commit,
and the four "no structurally valid candidates" branches into one shared helper, so the
classification hook exists once per family rather than eight times. The regex strategies were
not an afterthought here -- a regex target behind a wall loses its recipe exactly as a CSS one
does, and hooking only the two paths originally named would have left that intact.

`v010` gained the three `classified_*` columns rather than a `v011` adding them, since the
table had not shipped and no database outside a test container had run it -- verified, not
assumed: no `scrape_*` table and no `3tears_scrape` migration row existed in any local
database. A database that HAD applied an earlier `v010` would not pick the columns up, since
the version is already recorded as applied, and would need dropping.

**The four cached-recipe strategies are now one implementation.** CSS or regex, single record
or many rows, used to be eight functions: a reuse checker and a ~90-line regeneration body
apiece, differing only in which generator and validator they called and how they wrapped a
winning candidate. Adding the classification hook meant touching all of them, which is what
made the duplication expensive rather than untidy. They are now a `_StrategyShape` record of
the genuine differences plus one shared reuse cycle and one shared regeneration body -- 197
lines lighter, and a fifth shape would inherit the classification routing, the judging and
the persistence by construction.

One behaviour needed care and is pinned by two tests: the row shapes surfaced the survivor
capturing the most records when the judge confirmed nothing, while the single-record shapes
took the first proposed. The shared body uses max-by-record-count for both, which is only
equivalent because every single-record survivor holds exactly one record and `max` returns
the first maximal element. Both tests fail if that rule is changed in either direction.

**Tooling: every workspace package is now strict-mypy checked.** `scripts/typecheck.sh` went
from 13 targets over 315 files to 28 over 578 -- the whole workspace. The 144 errors that had
kept 14 packages out are fixed, not suppressed.

`models` supplied 116 of them, and 79 were one pattern: kwargs dicts typed `dict[str, object]`
splatted into third-party constructors. `object` claims more safety than a pass-through
forward has, and those values are arbitrary by construction (some arrive via `**extra_kwargs`),
so `Any` is the accurate type. The rest were real: `NameTranslatingChatMixin` declared
`invoke`/`ainvoke` returning `BaseMessage` where the base returns `AIMessage`, and `bind_tools`
taking `list[BaseTool]` where the base takes a far wider `Sequence` -- LSP violations in both
directions, against bodies that only ever return what `super()` returned and a translator that
already accepted both tool shapes.

`conversations` and `agent-workspace` shared one cause worth naming: 16 raw-SQL call sites did
`await self.l3_pool.fetch(...)` where `BaseCollection` documents `l3_pool` as legitimately
`None` and tells callers to guard. Those were latent `AttributeError`s, not type noise. A new
`BaseCollection.required_l3_pool` gives all of them one guard that names the actual mistake.
`langgraph` had the same shape in its offload middleware, and `mcp` accepted any JSON value as
a bearer token on a truthiness test.

`threetears.knowledge` was already clean and simply never listed -- the invisible version of
this gap, since an unlisted package looks exactly like a passing one from outside.

**`langchain-claude-code` is now a dev dependency.** It was the only optional provider adapter
missing from dev, against a block whose own comment says they are installed "so their test
suites exercise the real langchain integrations rather than getting skipped". Its absence was
silently skipping 32 tests across five modules and leaving `_claude_cli.py` entirely unchecked
(mypy resolved its base class to `Any`). Only the Claude Code CLI binary and Node are runtime
requirements; the Python packages import fine without them.

**Behaviour change, all four `threetears.scrape` collections** -- `deserialize` now returns
`datetime` where it returned `str` for every TIMESTAMPTZ column. `BaseCollection` documents
`deserialize` as where a subclass restores typed fields, and this one was a bare `json.loads`
against a `serialize` that writes `default=str`, so a row read through L2 differed in type
from the identical row read through L1 or L3. Invisible while such a row is only read, since
the entity accessors already parsed on the way out. Not invisible when one is written back:
an update fences on the row's own `date_updated` as an optimistic lock against a TIMESTAMPTZ
column, and a string bound there fails at the asyncpg border. Both read-modify-write paths in
the package were exposed, by different routes: `enrichment.enrich_extraction` rebuilds its row
through `create()`, so it binds no fence and would have failed on `retrieved_at` entering the
upsert's VALUES as a string; the new health merge rebuilds as an existing entity and fails on
the fence itself, where its own non-fatal handling would have swallowed the error and quietly
stopped updating fingerprints. Each collection declares its
`datetime_columns`, and a test asserts those match the TIMESTAMPTZ columns the migrations
create, in both directions. A caller that was reading these fields off the raw row dict and
expecting a string will now get a `datetime`; one using the entity accessors sees no change.

**Testing: `packages/scrape` gets its first integration suite** -- `link_selector` shipped
broken because the package had no test that touched a real database, and
`ScrapeCollection` falls back to an in-memory dict that has no schema to violate. The new
suite applies the real migrations to real Postgres and round-trips a row through the
production collection. Both guards were verified to discriminate by deleting a column:
the integration test raises `asyncpg.UndefinedColumnError` and the offline drift guard
names the missing column. The drift guard now also discovers its entity-to-table pairings
from the collections themselves, so a collection added later is guarded the moment it
exists rather than when someone remembers to add a fourth copy of the test.

**Fix: missing `link_selector` DDL column (`threetears.scrape`)** -- `ScrapeTarget`
exposed a persisted `link_selector` field with no matching `scrape_targets` column,
so a `multi_document` target seeded from YAML raised `asyncpg.UndefinedColumnError`
on its first real L3 upsert. Unit tests never caught it: `ScrapeCollection`'s
in-memory L3 fallback ignores schema entirely. Migration `v009` adds the column
(nullable, no-op for existing rows).

The guard that should have caught it is the real fix. `test_migrations_drift.py`
restated each entity's persisted fields as hand-maintained string literals, and
those literals omitted `link_selector` too, so the drift test sat green while the
drift shipped. It now derives the field set by walking `property` descriptors
declared below `BaseEntity`, filtering by declaring class rather than by name --
`ScrapeExtraction.id` shadows `BaseEntity.id` and IS a real column, so a name-based
filter would have silently stopped checking that table's primary key. A named,
currently-empty exemption set covers any future non-persisted property, and a
companion test asserts the derivation never returns an empty set vacuously.

**Behavior change, logging only (`threetears.scrape`)** -- a `ScrapeCollection`
whose registry has no `l3_pool` now emits one WARNING naming the table, instead
of silently using a process-local dict as L3. That silence is why the missing
column was invisible: the fallback ignores schema entirely, so a field with no
DDL column round-trips perfectly and only fails against a real pool. Operators
running without a wired pool will see one new WARNING per table. It is warned
once per table via a class-level set (mirroring
`BaseCollection._warn_missing_nats_client_once`), not per instance, so a
consumer that rebuilds collections each poll cycle still gets one warning
rather than one per cycle. Not an exception: the fallback is legitimate and
every unit test in the package relies on it.

Also in `threetears.scrape`, documentation only: the README had drifted a release
behind (five drivers documented against eight shipped, two extraction strategies
against four, `forms.py` / `request_shape_finder.py` / `page_finder.py` absent from
the module map); durable docstrings anchored to build ids and design docs from the
package's pre-lift home, some of which no longer resolve; and `ScrapeTool`'s MCP
schema excludes five backends without recording why (they need per-target config
its flat input schema cannot carry).

## v0.18.0 -- 2026-07-24

**Feature: `timestamptz` column type (`threetears.core.data`)** -- the declarable
column-type closed set gains `timestamptz` alongside the existing naive `timestamp`.
Until now a product could only declare naive `TIMESTAMP` columns, which forced any
platform layer that speaks timezone-aware datetimes end to end (the datasource broker,
whose deserialization binds aware-UTC) to either coerce or break on product writes.
A product can now declare `column_type="timestamptz"` and get:

- DDL: `build_create_table_sql` renders `TIMESTAMPTZ` (`sql_builder._COLUMN_TYPE_MAP`).
- L1 cache: `collection_factory` maps it to a timezone-aware `DateTime(timezone=True)`
  and to the Python `datetime` field type, so aware datetimes round-trip through L2.

`timestamp` (naive) is unchanged and remains valid; the two are distinct DDL types.
This is additive -- existing declarations keep their exact behavior.

## v0.17.9 -- 2026-07-23

**Feature: provider-native structured output (`threetears.models.providers`)** -- every provider
spells "return json matching this schema" differently: Anthropic takes `output_config`, OpenRouter
a top-level `response_format` plus a `provider` routing block, OpenAI the same `response_format`
nested under `extra_body` (a top-level one makes `ChatOpenAI` switch to `completions.parse()` and,
when streaming, to the beta streaming client -- both break callers that need a raw `AIMessage` and
a working token stream). That is the same class of wire quirk as the tool-name dot restriction the
name-translation mixin already hides, so it is owned in the provider layer rather than by every
consumer.

`structured_output_kwargs(provider_type, json_schema, *, name, strict)` dispatches to the right
per-provider builder and returns bind KWARGS rather than a bound model -- callers frequently apply
structured output to a model that is already a `bind_tools` `RunnableBinding`, which exposes
`.bind(**kwargs)` but none of the provider wrapper's own methods. A provider type with no
translation raises `StructuredOutputUnsupportedError` instead of returning empty kwargs: silently
dropping the directive lets the model answer in prose, which is the exact failure structured output
exists to prevent.

**Every builder rejects a malformed schema LOCALLY, before any provider call**, as
`StructuredOutputSchemaError`. Both error types derive from `StructuredOutputError` so a consumer
can catch every caller-error kind in one clause -- the distinction that matters downstream is not
WHICH way the request was wrong but that the request, not the provider, was at fault. These
failures never reach the network, so a consumer that lets one fall through to a generic handler
reads a local rejection as an unreachable provider and any circuit breaker keyed on that verdict
takes the provider out of service for every caller over one caller's bad schema.

On the Anthropic path that means translating `transform_schema`'s bare builtins (`AssertionError`
from its `assert_never`, plus `ValueError` / `AttributeError` / `TypeError` / `RecursionError`) --
an `AssertionError` in particular reads as an internal invariant failure, not a caller error. The
verbatim-embedding builders (openrouter, openai) had no validating step of their own and the
providers do not supply one: verified live against openrouter -> `google/gemini-2.5-flash`,
`{"type": "not_a_real_json_type"}` came back as the bare string `"Dublin"` and
`{"type": "object", "properties": "oops"}` came back as `{}`, both as SUCCESSFUL completions. They
now call `ensure_valid_json_schema`, which checks json-schema VALIDITY only -- deliberately not any
provider's additional subset rules (e.g. OpenAI strict mode's `additionalProperties: false`), since
enforcing one provider's policy would reject schemas that provider accepts today and would rot the
moment the policy moves. Adds `jsonschema` as a direct dependency of `3tears-models`.

**Refactor: `SchemaBackedCollection._partition_exempt_methods` is now
`partition_exempt_methods`** -- the base class instructs subclasses to extend it, which makes it
public API, not an implementation detail. Every subclass across `agent/acl`, `agent/memory`,
`conversations` (and downstream in the hub) was reading a leading-underscore name across a class
boundary, each site carrying a `noqa: SLF001` and an enforcement exemption to say so. **Breaking
for any subclass that overrides it**: rename the attribute. No back-compat alias -- a subclass that
keeps the old name would silently inherit an unextended allowlist, so it breaks loudly at the
attribute name instead.

## v0.17.8 -- 2026-07-19

**Fix: a pod's L1 cache-invalidation listener crashed on every broadcast for a table it never
locally caches** -- `CollectionRegistry._on_invalidation` already had an early-return for a
`message.table` with no registered `Collection` at all ("unknown-table receipts are expected
during partial rollouts"). This was the same class of expected-not-error case one level deeper:
a pod CAN have a `Collection` registered for a table (so that early-return never fires) while its
own L1 backend (`SQLiteBackend`/`DuckDBBackend`) was never `initialize()`'d with that specific
table's schema -- `collection_factory.create_dynamic_collection` only calls `initialize()` lazily,
per table, the first time that table's Collection is actually instantiated locally, and a
cross-pod invalidation broadcast (`threetears.cache.invalidate`) is heard by EVERY pod regardless
of which tables each one actually caches. Observed live: any agent pod that never touches the
knowledge subsystem's `concepts`/`playbook_entries` tables crashed its own NATS subscribe callback
with `sqlite3.OperationalError: no such table` (or DuckDB's equivalent) on every single write any
OTHER pod made to those tables.

Added `L1Backend.has_table(table)` (both backends already track this via their existing
`_schema_info` dict, so the check is free) and consult it in `_on_invalidation` before calling
`delete_by_id` -- the same "unknown receipts are expected" treatment the unregistered-Collection
case already gets, extended to the one-level-deeper "registered but never locally cached" case.
Prevents the error at the source rather than catching a backend-specific exception after the fact
(SQLite and DuckDB raise different exception types for "no such table", and the base module is
deliberately kept free of a hard DuckDB import since that backend is optional).

**Fix: the WebSocket per-message/per-frame crash-safety nets could themselves crash the
connection against an already-dead socket** -- `8950bae` (v0.17.6) wrapped the chat-message
dispatch in a safety net so a router failure degrades to one error frame instead of crashing the
socket, matching the typed cross-pod frame path's existing per-frame net. Neither net's own
error-frame *send* was itself guarded: if the socket had ALREADY died (a client disconnect
racing an in-flight dispatch -- the exact window a long-running turn/handler runs in), the
notification send raised, uncaught, and the ASGI framework closed the connection with an
unhandled-exception crash instead of degrading gracefully. Observed live: a chat WebSocket
connection crashed during a client-disconnect window, in an error-frame send that only exists
because of the v0.17.6 fix.

Audited every best-effort notification/error-frame send in `WebSocketHandler._message_loop` and
its full typed-frame dispatch tree (`_route_frame`, `_authorize`, `_handle_join`,
`_handle_editor_op`, `_handle_transient`, `_handle_resume`/`_stream_replay`, plus the legacy
chat-message path) -- not just the two outer safety nets -- and routed every one through a new
`_safe_send` helper (try/except, log-and-drop on failure, mirroring the pre-existing
`_close_with_error` pattern). The resume/replay tail additionally now stops attempting further
payloads the moment one send fails, rather than retrying into a wire already known dead. Two
send sites are deliberately left unguarded, with an inline comment explaining why: the very
first "connected" frame on a freshly-accepted socket (nothing to degrade gracefully from yet),
and `_route_standard`/`_route_streaming`'s own response/token sends (already covered by their
surrounding safety net, so a failure there gets ONE well-logged notification attempt rather than
a second, less-specific one).

## v0.17.7 -- 2026-07-18

**Fix: a bound tool's LangGraph HITL interrupt was silently swallowed by the Claude
subscription-CLI backend, and could not simply be re-raised** -- a confirm-mode write tool (the
LangGraph "pause and wait for a human" pattern: the tool body calls
`langgraph.types.interrupt(...)`, which raises `GraphInterrupt`) worked correctly on every
direct-API chat backend, but under the subscription/CLI backend (`create_subscription_chat`,
selected when the credential is a Claude Max/Pro OAuth token rather than an API key) the interrupt
never reached the graph. Found live: a consuming product's write tool staged an edit, the model
said "the tool call was interrupted and requires approval" -- but nothing was actually staged, no
confirmation step was ever reached, and the graph completed the turn as if the tool had simply
failed.

Two independent layers were swallowing the exception, not one:

1. `_wrap_langchain_tool`'s `wrapped()` closure (the in-process handler the CLI subprocess's own
   internal tool-calling loop invokes directly, bypassing LangGraph's own `ToolNode`) caught
   `except Exception` around the tool dispatch with no exclusion for `GraphBubbleUp` --
   `GraphInterrupt` is a plain `Exception` subclass.
2. Even with (1) fixed to re-raise, the *third-party* `mcp` package's own `Server.call_tool`
   request handler (`mcp/server/lowlevel/server.py`, not ours) *also* catches every exception
   unconditionally and converts it to a normal `CallToolResult(isError=True, ...)` -- confirmed
   directly against that dispatch path, not just by reading its source. No exception of any kind
   can survive that boundary, so simply re-raising harder was never going to work.

The fix therefore **captures** the interrupt at the point it occurs (`wrapped()`, the only code
with real-time visibility into the call) instead of trying to propagate it through a boundary that
cannot carry it, and **replays** it from a point that genuinely sits inside LangGraph's own call
stack (`_astream`/`_agenerate`, once the underlying CLI turn completes). Resuming needed its own
mechanism too: this backend has no separate LangGraph "tools" node to replay in isolation (the
whole decide-and-call round-trip lives inside ONE model call), so a resume means calling the model
again from scratch, and a plain `Command(resume=...)` never edits the conversation history -- a
resume-hint message is now appended (via LangGraph's own `__pregel_resuming` configurable flag,
read without disturbing `interrupt()`'s own resolution) asking the model to retry the exact same
tool call, whose own `interrupt()` then resolves normally via LangGraph's ordinary resume-value
matching.

- **`_wrap_langchain_tool`/`_astream`/`_agenerate`**
  (`packages/models/src/threetears/models/providers/_claude_cli.py`). `wrapped()` now captures a
  caught `GraphInterrupt` into `_captured_interrupts_var` (a `ContextVar`, matching this class's
  existing `_tool_results_var` pattern) and reports a benign, non-error result instead of trying to
  propagate; both `_astream` and `_agenerate` set that var fresh per call, append a resume-hint
  message via `_messages_with_resume_hint`/`_is_resume_replay`, and re-raise a combined
  `GraphInterrupt` once the underlying CLI turn completes if anything was captured.
- **`ToolCompletedEvent.tool_status`** (`packages/langgraph/src/threetears/langgraph/events.py`)
  gains a third, honestly-labeled value, `'interrupted'`, alongside `'completed'`/`'failed'` -- an
  interrupt is not a tool failure.
- Regression tests: `TestGraphInterruptCapture` in `test_claude_cli_tool_events.py` proves the
  handler-level capture in isolation; `test_claude_cli_interrupt_resume.py` drives a REAL compiled
  LangGraph graph (only the Claude Agent SDK subprocess is faked) through a full pause -> resume
  round trip for both an accepted and a rejected decision, proving the tool's own `interrupt()`
  call genuinely resolves on replay -- not just that the graph pauses.

## v0.17.6 -- 2026-07-17

**Fix: an empty or malformed chat message over WebSocket could crash the whole connection** --
the typed cross-pod frame path (`join`/`leave`/`editor.op`/the transient `cursor`/`typing`/
`presence` types) already wraps every dispatch in a per-frame safety net: a handler exception
becomes one error frame and the socket keeps serving. The plain chat `message` path, dispatched
just above that safety net in the same loop, had no equivalent -- a router failure (e.g. an
unknown target agent, any downstream dispatch error) propagated all the way out of the message
loop uncaught, and the ASGI framework closed the connection with a 1011 internal-error code
instead of degrading gracefully. Found auditing a consuming product's integration test suite,
where a `test_empty_content_ignored`-style test had never actually exercised this path before
(it always failed earlier, at connect/auth) until an unrelated auth fix let it reach here for
the first time.

- **`WebSocketHandler._message_loop`** (`packages/channels/src/threetears/channels/websocket.py`).
  The `if is_streaming: ... else: ...` chat dispatch is now wrapped in the same
  `except Exception` safety net the typed-frame branch already has: log the failure, send
  `{"type": "error", "message": "internal error handling message"}`, and keep the loop running
  for the next message -- matching design T3-D2's "never a silent drop, never a dead connection"
  posture for every message shape, not just typed frames. Regression test:
  `TestChatMessageDispatchIsCrashSafe::test_router_exception_on_chat_message_does_not_crash_socket`
  in `packages/channels/tests/unit/channels/test_websocket_task03.py`.

## v0.17.5 -- 2026-07-17

**Fix: `tool_search`'s own hit message contradicted its own description** -- the tool's
DESCRIPTION correctly told the model a hit becomes callable starting its NEXT reply, but the
text returned immediately after a hit said the tool was "now available to call" -- a direct
contradiction read mid-round, before any caller has had a chance to compose the hit into its
bound tool set. This drove a calling model to immediately retry the newly-found tool in the same
round and bounce off "No such tool available" (found live, metallm prod conv
`019f6cf5-073a-7b50-bd44-721efb0c7b90`).

- **`create_tool_search_tool`** (`packages/agent/tools/src/threetears/agent/tools/relevance.py`).
  No caller can make a tool available before the next round boundary -- that's a hard
  architectural floor (a model completion already in flight is committed to whatever tool
  schema it was sent with), not something any caller can work around. This is a wording fix
  only: the hit-message return string now matches the description instead of promising
  immediacy no caller can deliver.

**Docs: codified the never-squash-merge convention directly in `CLAUDE.md`.** The
merge_commit-only convention was already followed in practice, but was never written down
anywhere read at session start; a squash merge earlier in this cycle silently diverged `main`'s
file content from a source branch until it was caught and corrected. Added an explicit
"Git / PR Workflow" section (never squash-merge, never force-push, release sequencing) so this
stays a checked rule, not tribal knowledge.

## v0.17.4 -- 2026-07-16

**Fix: every optional tool parameter was advertised to a Claude Max subscription turn as a
required string** -- `memory_search`'s `ids` (`list[str] | None`) arrived at the handler as the
literal string `"[]"`, failing pydantic validation on every call; every OTHER optional filter
(`date_after`, `date_before`, `alias`, ...) arrived populated with an empty-string placeholder
instead of being omitted, degrading `conversation_search`/`chunk_search` results (found live,
same prod conversation as v0.17.3, `019f6cf5-073a-7b50-bd44-721efb0c7b90`).

- **`_SubscriptionChatModel._wrap_langchain_tool`** (`packages/models/src/threetears/models/providers/_claude_cli.py`).
  Pydantic renders an `X | None` field as `anyOf: [{type: X}, {type: null}]` with no top-level
  `type` key; the schema-to-SDK conversion's naive `prop.get("type", "string")` silently defaulted
  every such field to `string`. Separately, handing the SDK a bare `{name: type}` map (rather than
  a full JSON Schema) makes its own schema builder mark every key `required`, forcing the model to
  invent placeholder values for filters it had nothing to fill in. `_wrap_langchain_tool` now
  builds the full `{type: object, properties, required}` schema itself, resolving each property's
  real type and copying `required` verbatim from the source tool schema.

**Fix: `NameMangledToolProxy` never forwarded `config` to a delegate that requires it** -- a real
3tears builtin tool (a `StructuredTool` built by `to_langchain_tool`, e.g. `threetears.calculator`)
raised `TypeError: StructuredTool._arun() missing 1 required keyword-only argument: 'config'` the
instant it was actually invoked through this proxy, on EVERY provider that uses it (`anthropic.py`,
`openrouter.py`, and the Claude Max subscription backend) -- not a claude-cli-specific bug, a
pre-existing gap in shared name-translation infrastructure, found live testing the fixes above.

- **`NameMangledToolProxy._arun`/`_run`** (`packages/models/src/threetears/models/tool_name_translation.py`).
  `BaseTool.arun`/`run` only forward `config` to `_arun`/`_run` when that method's OWN signature
  declares a `RunnableConfig`-typed parameter -- the proxy never declared one, so it never received
  a `config` to forward, and its body called the delegate's `_arun`/`_run` directly with none. Fixed
  by declaring `config`/`run_manager` on the proxy's own methods (so the caller's introspection
  finds and supplies them), then forwarding to the delegate only if its own signature wants them.

## v0.17.3 -- 2026-07-16

**Fix: 3tears builtin tool calls (`web_search`, `calculator`, ...) were silently denied under a
Claude Max subscription turn** -- "Claude requested permissions to use ... but you haven't granted
it yet", with nothing logged anywhere, while the same tool worked fine on every other backend
(found live, prod conversation `019f6cf5-073a-7b50-bd44-721efb0c7b90`).

- **`_SubscriptionChatModel.bind_tools`** (`packages/models/src/threetears/models/providers/_claude_cli.py`).
  Every 3tears builtin's canonical name is dotted (`threetears.web_search`, per
  `BaseAgentTool.mcp_name()`). The base class's own `bind_tools` already tries to auto-approve
  bound tools by deriving `allowed_tools` from each tool's raw dotted name, but the SDK/CLI
  normalizes dots out of tool identities on the wire, so that entry never matched what it was
  meant to auto-approve -- every real call needed (and never got) interactive approval. Every
  other provider wrapper (`anthropic.py`, `openrouter.py`) already applies the same dot-to-underscore
  translation for the identical Anthropic tool-name constraint, but the subscription backend never
  got it. `bind_tools` now substitutes each dotted tool for a `NameMangledToolProxy` before the
  base class derives `allowed_tools`, so the entry matches exactly. Also adds
  `NameMangledToolProxy.canonical_name`, a public accessor for the delegate's un-mangled name.

**Fix: L2 bucket-resolution failures on first open were not degrading like every other L2
transport failure.** `BaseCollection._get_from_l2`/`_save_to_l2`/`_delete_from_l2`
(`packages/core/src/threetears/core/collections/base.py`) already caught `KvError` narrowly around
the `kv.get`/`put`/`delete` call, but the preceding `_ensure_kv()` bucket-resolution call sat
outside that try block -- a regression from an earlier change that split bucket resolution out of
the get/put/delete calls without widening the catch to cover the new call site. When a KV bucket
had never been opened yet (e.g. right after a NATS outage begins) and `_ensure_kv()` raised
`KvError` on the first open attempt, the exception propagated uncaught instead of degrading,
breaking `save_entity()`'s documented "L2 is best-effort, L3 is source of truth" contract for any
collection whose bucket was not already warm. The catch now covers bucket resolution too.

## v0.17.2 -- 2026-07-16

**`skill_report_outcome` tool (`packages/agent/skills`), written 2026-07-13 but left
unmerged on a feature branch until now -- metallm's own skill-outcome-reporting rework
needs it to build.**

- **`skill_report_outcome` tool + `load_skill_report_outcome_tool`
  (`packages/agent/skills/src/threetears/agent/skills/tools.py`).** Lets an agent
  self-report a skill invocation's success/failure via an explicit tool call, replacing
  the retired `[SUCCESS]`/`[FAILED]` post-response text-marker convention.

## v0.17.1 -- 2026-07-16

**Two additions that were written the same day as v0.17.0 but were left unmerged on
feature branches and missed that release. Both land here instead.**

- **`ToolRelevanceIndex` + the `tool_search` meta-tool (`packages/agent/tools`,
  `relevance.py`).** Embeds and ranks a tool catalog against the current turn's query,
  returning the top-k most relevant tools with an LRU cache keyed on the catalog
  identity; a `tool_search` `BaseTool` wrapper lets a model reach anything filtered out
  of the initial top-k on demand. Falls back to the full, unfiltered catalog on any
  embedding failure or when ranking exceeds a configurable latency ceiling -- a
  degraded turn is never worse than today's full-catalog behavior. This is the
  platform primitive metallm's own dynamic tool-relevance selection consumes.
- **`acting_as_principal_id` on `AuditEvent`** (`packages/agent/audit`, `envelope.py`).
  `14-eng-ai-bot-identity`'s impersonation flow (`identity.impersonation.start`/`stop`)
  needs to record both the impersonation TARGET (`actor_user_id`, whose session it is)
  and the ADMIN actually driving it. Previously that producer carried the admin
  identity in `details["acting_as_principal_id"]` -- works on the wire, but isn't a
  typed, Hub-queryable column. Additive only: optional, defaults to `None`, every
  existing producer unaffected.

## v0.17.0 -- 2026-07-15

**Support for `14-eng-ai-bot-identity`, the platform's new NATS-native multi-tenant
identity broker.** Four additions to `packages/core` and `packages/agent/acl`, built and
landed across identity-core's own build (chunks 03/05/13), consumed there via a
temporary local-path override while this release was pending:

- **`jwk_thumbprint()` (`packages/core`, `security/identity_token.py`) now accepts
  `EllipticCurvePublicKey`, not just Ed25519.** Extends the RFC 7638 thumbprint to the
  EC required-member set (`crv`, `kty`, `x`, `y`, via PyJWT's `ECAlgorithm.to_jwk`) --
  needed for DPoP proof validation binding a P-256 client key. The existing Ed25519
  branch is unchanged, verified byte-identical against a pinned vector.
- **`RevocationGuard` (`packages/core`, `coordination/replay_guard.py`), a new sibling to
  `ReplayGuard`.** Where `ReplayGuard.record_unique`'s presence-only sentinel fits a
  single-use nonce or an exact `jti`/`sid` revocation, a `sub` (principal) or
  `customer_id` (tenant) revocation needs a value comparison, not membership: record a
  `revoked_at` timestamp per key, then `is_revoked_before(key, moment=...)`. Fail-closed
  on KV transport failure, same durability posture as `ReplayGuard`.
- **`WindowedCounter` (`packages/core`, `coordination/windowed_counter.py`), a new
  generic throttle primitive.** A windowed attempt counter over a NATS JetStream KV
  bucket (`record_attempt`/`count`/`is_over_threshold`) for a "how many times in the
  last N seconds" shape neither `ReplayGuard` nor `RevocationGuard` express. Fail-open
  vs. fail-closed is a constructor-level caller choice (`fail_open: bool`, default
  `False`), since a throttle counter doesn't always sit on a hard security boundary.
- **`authorize_from_claims` + the impersonation gate schema (`packages/agent/acl`).** A
  claims-aware authorization entry point layering an impersonation deny-list overlay
  on top of the existing `authorize()`: denies unconditionally when
  `act_reason == "impersonation"` and the caller names a sensitive
  `ImpersonationCategory`, otherwise defers as normal. `ImpersonationGateCollection`/
  `ImpersonationGateEntity` add the per-tenant `disabled|requested|enabled` + TTL gate
  schema, with read-time TTL self-revert. Real Hub-side wiring (a live NATS responder
  persisting this collection against Postgres) is not part of this release --
  identity-core's own test suite proves the wire contract against a local fake double.

## v0.16.1 -- 2026-07-15

**Real token-level streaming for the Claude Max subscription backend.** `ClaudeCodeChatModel._astream`
(`langchain-claude-code` 0.1.0) requests `include_partial_messages=True` from the Claude Agent SDK --
which makes the subprocess emit granular `StreamEvent` text deltas -- but the method only ever
consumed the terminal, whole-block `AssistantMessage`, silently dropping every delta. A subscription
turn arrived as one or two large lumps instead of a real token stream.

- **`_SubscriptionChatModel._astream`** (`_claude_cli.py`, alongside the existing `_build_options` /
  `_wrap_langchain_tool` overrides for other upstream gaps in this same package) now consumes
  `StreamEvent` text deltas and yields each one immediately as it arrives, tracked per content-block
  index so the terminal `AssistantMessage` never re-yields (and thereby doubles) text a delta already
  streamed. A block that produces no `StreamEvent` at all (older CLI build, future SDK regression)
  still gets its text emitted whole from the `AssistantMessage` -- strictly additive, never worse than
  before.
- Verified against a real Claude Max subscription session: a response streamed in 13 chunks over
  ~10.6s (visible incremental delivery), versus 1-2 chunks arriving all at once under the prior
  behavior.

## v0.14.1 -- 2026-07-06

**Refreshing NATS connect credentials + per-key tool-pod identity.** A connection's auth
credential is no longer a single string captured at connect and re-presented (stale) on
every reconnect — it is a PROVIDER re-invoked each (re)connect, so a short-lived self-minted
identity token is re-minted fresh and the connection never wedges when the credential
expires mid-session.

- **`NatsClient.connect(auth_token=...)` is now a token PROVIDER** (`Callable[[], str]`,
  invoked by nats-py on every (re)connect) rather than a static `str`. Static-credential
  services wrap their token in `static_token_provider`; self-minting principals pass a
  provider backed by `IdentityMinter`.
- **`IdentityMinter`** (`threetears.core.security`) — holds a custody Ed25519 key and
  self-mints short-lived EdDSA identity JWTs (the stateful counterpart to the pure
  `sign_identity_token`), for a pod/agent/tool-pod to present as its connect credential.
- **`NatsClient.is_healthy`** reports `False` when a connection is stuck in a persistent
  Authorization-Violation reconnect loop (a rejected credential the forever-reconnect rides
  forever without closing), so a `/healthz` keyed on it lets k8s restart the pod.
- **Tool-pod per-key identity, both auth layers.** `ToolServer` accepts an `auth_token`
  provider so a tool pod self-mints its connect JWT for the NATS auth-callout, and carries
  the same JWT on its registration manifest. The registry `ToolPodAuthenticator.verify_pod`
  now takes the RAW JWT (was a token hash); `RegistrationHandler` verifies token-bearing
  manifests and admits tokenless (agent-owned in-process) pods.

**Cross-worker cancellation primitives (additive).** A WS-streaming consumer that runs turns
as fire-and-forget tasks now has a platform primitive to stop one — locally or on whichever
worker holds it — instead of hand-rolling a registry + NATS routing. Purely additive; no
existing signature changed.

- **`threetears.core.KeyedTaskRegistry`** — a per-worker registry of cancellable
  `asyncio.Task`s keyed by `UUID` (`register`/`pop`/`get`/`discard`, identity-guarded). Keeps
  a fire-and-forget task's handle reachable so it can be cancelled by key. `pop` is
  pop-before-cancel (a redelivered cancel is a clean no-op).
- **`threetears.nats.CrossWorkerCanceller` + `TaskCancelEnvelope`** — wraps a
  `KeyedTaskRegistry`; `request_cancel(key, payload)` cancels the task locally when this
  worker owns it, else publishes on a consumer-supplied broadcast `Subject` so the OWNING
  worker cancels. On cancel it invokes a consumer `on_cancel(key, payload)` callback with an
  **opaque** payload — the primitive knows nothing about locks/frames/checkpoints; all product
  semantics stay in the callback and the cancelled coroutine's own `finally`. `registry` is a
  required constructor arg so `threetears.nats` keeps `threetears.core` a type-only dependency.
  (Mirrors the `channels` `RoomFanout` publish-one/act-on-receive-per-pod pattern, specialised
  to cancellation. First consumer: metallm's Stop button.)

**Fixes (enforcement debt on this branch).**
- `IdentityMinter` per-mint session id now uses `uuid7` (time-ordered), satisfying the
  uuidv7 enforcement.
- The provider `NameTranslatingChatMixin`'s `_name_reverse_map` type hint is now
  `TYPE_CHECKING`-guarded, so the (pydantic-required) per-subclass `PrivateAttr` declarations
  are no longer flagged as shadowing a base private. No runtime change.

## v0.13.11 -- 2026-07-02

The **scope-and-objects** framework family: the huge-object offload backend and
the general engagement re-authorization seam that pentest scan scope is built on.

- **Object offload (Path-2).** A streaming S3/MinIO `ObjectStore` with scope-first
  keys (`object-store`); a langgraph offload seam that streams large tool results
  to the store and threads an `ObjectHandle` through the graph, plus tool-authored
  offload summaries via `content_and_artifact` (`langgraph`); the pod-side produce
  seam and a `build_s3_object_store` secret-ref-resolving wiring helper
  (`agent-tools`); the object-catalog NATS subjects — `hub_object_commit` /
  `hub_object_resolve` — and `list_entries` (`nats`, `object-store`); a general
  report tool that renders to the store and a general deliver tool that resolves an
  object id to a presigned URL, both pod-side and identity-token authed against the
  verified tenant (`agent-tools`); and a `BIGINT_TYPE` column tag for int8 columns
  (`core`).
- **Engagement scope (ES-1/ES-5).** The `hub_engagement_scope` NATS subject and the
  pod-side engagement re-authorization resolver + scope-injection seam (`nats`,
  `agent-tools`), and an `engagement_provider` carried on `BootstrapContext` so the
  runtime can auto-stamp the active engagement onto outgoing tool calls
  (`langgraph`). The framework treats `target_type` as an opaque string; the pentest
  domain interprets it.

## v0.13.10 -- 2026-06-29

Fixes the platform-wide **1-hour agent cliff**: every long-lived agent pod went
dead ~1h after boot because the auth-callout's NATS user JWT (default 3600s TTL)
expired while connected, and the NATS server's auth `-ERR` is routed by nats-py
straight to a terminal `close` that bypasses `_attempt_reconnect` — so
forever-reconnect (the network-drop path) never recovers it, and host daemons have
no k8s liveness net.

### Added

- **`NatsClient.reconnect()`** (`threetears.nats`) — a force-reconnect primitive
  that drives nats-py's own `_attempt_reconnect` on a still-connected client
  (synthesizing a `StaleConnectionError` through `_process_op_err`), so the
  transport cycles and the server auth handshake re-runs — under decentralized
  auth this re-runs the Hub auth-callout, minting a **fresh user JWT with full
  TTL** — while subscriptions replay under their original `sid` and the same
  underlying client object is reused (consumers holding `.raw` stay valid).
  Raises on an already-closed client; no-ops when not currently connected (so it
  never trips `_process_op_err`'s connection-closing else-branch).

This is the primitive the SDK uses for **proactive NATS-JWT re-auth**: forcing a
reconnect a margin *before* expiry, while the current JWT is still valid, so the
connection never reaches the terminal auth-expiry close. (The SDK-side re-auth
loop, the Hub's TTL-in-handshake reporting, and the env-overridable TTL knob ride
on this primitive and live in the consumer repos.)

### Fixed

- Three pre-existing over-strict third-party-stub `mypy` errors in
  `threetears.nats.client` (the gate-excluded `nats` package is now mypy-clean):
  the wrapper's float `flush()` timeout (nats-py annotates `int` but waits via
  `asyncio.wait_for`, which accepts float) and `_subscribe_internal`'s `Subject |
  str` subject (it already coerces `str`).

## v0.13.9 -- 2026-06-28

Platform-wide authentication lands and is **enforced**. The NATS bus is
fail-closed (an anonymous connect is rejected); every tool call carries a
Hub-issued, cryptographically-bound caller identity; and RBAC evaluates the
**verified** identity rather than a self-asserted envelope field. Shipped
enforce-only — no warn rung, no `no_auth_user`. Also: first-class
human-in-the-loop interrupts, an `engagement_id` identity dimension, and a
Kubernetes-resilience pass across the NATS + identity-verifier layer.

### Added — platform auth (A: NATS connection auth)

- **Auth-callout connection auth.** A connecting agent/tool pod presents its
  bootstrap token; the Hub's auth-callout responder resolves the principal and
  mints a **least-privilege, per-principal NATS user JWT** (`threetears.nats`:
  `user_jwt`, `auth_callout`, `subject_permissions`). Each principal's pub/sub
  allow-list is scoped to its own identity-bound subjects + reply inbox — no bare
  `>` wildcards, no cross-tenant KV/stream reach.
- Ships **enforce-only**: `no_auth_user` removed, anonymous connect rejected
  (`Authorization Violation`); platform services authenticate with per-service
  static users.

### Added — platform auth (B: identity tokens + crypto binding)

- **Hub-issued identity tokens** — EdDSA/Ed25519 JWS, alg-pinned, published via
  JWKS over NATS request/reply, minted at the bootstrap handshake and attached to
  every outgoing `CallContext` (`threetears.core.security`: `identity_token`,
  `jwks_provider`).
- **Verify-and-re-stamp at the registry proxy AND the tool pod** — the verified
  agent/user/customer overwrite the envelope, so RBAC authorizes the verified
  identity, never a self-asserted one. Fail-closed.
- **Crypto binding (DPoP-style).** A per-pod proof-of-possession key binds each
  call (`cnf` + `ath` + body-hash + single-use nonce, replay-guarded); the proxy
  mints a body-bound `proxy_assertion` (Ed25519 JWS, `aud=pod_id`) the tool pod
  verifies (`threetears.core.security`: `pop`, `proxy_assertion`, `replay_guard`).

### Added — HITL + identity

- **Human-in-the-loop interrupt surfacing** in `threetears.langgraph` streaming:
  a LangGraph `interrupt()` emits a `StreamInterruptEvent` terminal and stashes
  `__interrupt__` instead of an empty end, so an approval gate can pause and
  resume via `Command(resume=)`. Additive — uninterrupted graphs end as before.
- **`engagement_id`** promoted to a first-class typed `CallContext` field.

### Changed — Kubernetes resilience

- **Identity-token refresh lifecycle** — pods re-handshake before expiry reusing
  the pop key (cnf intact), so tool-calling survives past the token TTL.
- Forever-retry startup for critical bindings (never flip ready with a dead
  handler); honest liveness/readiness (real `ping()` + a `jwks_warmed` gate);
  effectively-infinite NATS reconnect; reactive JWKS self-heal on a kid-miss;
  guarded background loops. Built for undefined start order, N replicas, and pod
  movement, not a later resilience pass.

### Fixed

- `timezone_converter` resolves `"now"` itself instead of requiring the caller to
  supply the current datetime — the tool carries the value, the caller never
  infers it.
- Three registry proxy tests import their shared dispatch helper relatively, so
  the canonical full-suite run (`pytest packages/ tests/`) collects cleanly, not
  only per-package.

## v0.13.8 -- 2026-06-24

On cancel (e.g. a datasource tool-call timeout) the Redshift driver aborted the
query by closing the client connection — but closing the **client** socket does
not kill the running **server-side** Redshift query. A real abandoned query ran
on the cluster for **7.4 hours**, leaking a connection-pool slot the whole time
and re-exhausting the small pool faster than it could drain, which silently
stopped an agent from answering.

### Fixed — `3tears-datasources` — `RedshiftDriver` cancellation

- **The driver now captures each connection's `pg_backend_pid()` at open and, on
  cancel, issues `pg_terminate_backend(<pid>)` from a fresh short-lived
  connection** before closing/evicting the poisoned connection. Closing the
  client socket alone left the query running server-side; terminating the backend
  actually stops it. (The DB user need not be a superuser — `pg_terminate_backend`
  on one's own session works where `CANCEL` does not.)
- **Best-effort and non-fatal throughout.** The pid read at open is best-effort
  (a failure only degrades the server-side cancel; the connection stays usable).
  The terminate runs in a worker thread under `wait_for`, logs on success,
  logs + bumps the existing `cancellation.failed` counter on failure, and never
  raises — the client-socket close + evict path runs regardless.
- Pairs with consumers capping each datasource's `query_timeout_seconds` at its
  tool-call timeout: that bounds queries that **respect** `statement_timeout`;
  this terminates the ones that **wedge past** it.

## v0.13.7 -- 2026-06-23

NATS is the **L2** tier in 3tears — ephemeral, with durability riding JetStream
R3 replication plus the consumer's real L3 (git/DB). The JetStream helpers,
however, defaulted to **file** storage, so any consumer running against a
deliberately memory-only NATS deployment failed at first KV/stream creation with
`10047 insufficient storage resources available` (it surfaced as a 500 on the
first collections L2 access — presence join, entry read).

### Fixed — `3tears-nats` / `3tears` — JetStream storage now defaults to memory

- **`NatsClient.kv_bucket` and `NatsClient.ensure_jetstream_stream` now default
  `storage="memory"`** (was `"file"`); `NatsKvBucket.__init__` matches. `"file"`
  remains available as a deliberate, explicit opt-in for the rare object that
  genuinely needs on-disk durability.
- **`core.cache.NatsKvClient` no longer forces the `collections` bucket to
  `file`** — it now uses the `BucketConfig` memory default. This is the bucket
  whose file-backed creation failed on a memory-only account.
- Net effect: a consumer on a memory-only NATS (no file store, `max_file: 0`)
  works out of the box; nothing has to opt into memory. File storage is now the
  conscious exception, matching the L2 contract.

## v0.13.6 -- 2026-06-23

Closes a permanent-staleness race in the cross-pod config-epoch machinery
that any consumer loading local state before subscribing could hit -- it
surfaced as a gateway serving a model catalog that contradicted the admin
API, and the same shape sat latent in the MCP grant cache.

### Fixed — `3tears-epoch` — `threetears.epoch.listener`

- **`EpochListener.subscribe` gains an optional `primed_epoch` parameter so a
  consumer that loaded local state before subscribing can never go permanently
  stale.** `subscribe` primed its per-subject last-seen by reading
  `EpochClient.current()` at subscribe time. A consumer that loads local state (a
  model catalog, a grant cache) and only then subscribes therefore primed
  last-seen to whatever epoch had committed by subscribe time — which can be
  AHEAD of the epoch the loaded state actually reflects. A bump landing in the
  load→subscribe window then pins last-seen past the loaded state, the periodic
  `catch_up` sees `current == last_seen` and never fires, and the consumer serves
  stale state forever with no recovery path. The fix is additive and
  backward-compatible: pass `primed_epoch` = the epoch the loaded state reflects
  (read `current()` BEFORE the load, then load, then subscribe). last-seen is then
  never ahead of the loaded state, so any bump at or after the load is detected
  (broadcast or `catch_up`); worst case is one redundant reload, never permanent
  staleness. Omitting `primed_epoch` preserves the prior `current()`-at-subscribe
  behaviour — correct only when no state was loaded against an earlier epoch.

### Fixed — `3tears-mcp` — `threetears.mcp.auth`

- **`LocalGrantAuthorizer.start` reads the rbac epoch BEFORE reloading the grant
  cache and primes the listener to it.** `start` reloaded the grant cache and
  then subscribed, so a `mcp.rbac` bump committing in that window pinned the
  listener's last-seen past the freshly-loaded grants and the catch-up tick
  (`current == last_seen`) never recovered it — the authorizer could serve a
  permanently-stale grant set, making default-deny RBAC decisions on revoked or
  stale grants. It now reads `current()` before the reload and passes it as
  `primed_epoch`, mirroring the gateway catalog fix. Also asserts the listener is
  non-None in the catch-up loop (only ever spawned under epoch mode), closing a
  latent `union-attr`.

## v0.13.5 -- 2026-06-22

Closes the remaining gaps that surfaced while converging a host app's bespoke
tool loop onto `build_tool_agent`: wire-side tool-name translation leaking
through every non-`astream` chat surface, the agent node force-hoisting a
system message a caller had already assembled, a hook emitter able to abort a
turn, and `SqlL3Backend` dropping namespace + `customer_scope` when it wraps a
scope-aware transport.

### Fixed — `3tears-core` — `threetears.core.backends`

- **`SqlL3Backend` now forwards `namespace` + transport kwargs (e.g.
  `NatsProxyL3Backend`'s `customer_scope`) to a scope-aware wrapped pool instead
  of dropping them.** The collection registry wraps any non-`DurableStore` L3
  transport in `SqlL3Backend` to add the structured CRUD layer — including
  `NatsProxyL3Backend`, which is a raw-SQL-over-NATS transport with no
  `fetch_one`/`upsert`. But `SqlL3Backend`'s raw-SQL methods
  (`fetch`/`fetchrow`/`fetchval`/`execute`) were written to wrap a bare asyncpg
  pool: they silently dropped `namespace` and had no `customer_scope` parameter.
  So an agent-SDK scoped/RBAC read through a wrapped `NatsProxyL3Backend` either
  raised `TypeError: unexpected keyword argument 'customer_scope'` or lost
  namespace scoping; ordinary collection ops survived only via a default-namespace
  fallback. The wrapper now detects a scope-aware pool via an `accepts_scoped_reads`
  capability marker (an identity check, **not** `isinstance` — `NatsProxyL3Backend`
  omits `fetchval` and so does not satisfy the `L3Backend` protocol structurally,
  which would make an `isinstance` gate silently fail) and forwards `namespace` plus
  any extra kwargs generically via `**kwargs`, staying ignorant of NATS-specific
  concepts. A bare asyncpg pool lacks the marker, so its behaviour is unchanged.
  Pre-existing since the per-call `customer_scope` channel landed; it affected the
  agent SDK (knowledge retrieval + RBAC visibility through the proxy), not the hub
  (which passes a raw asyncpg pool that `SqlL3Backend` is built to wrap).

### Fixed — `3tears-models` — `threetears.models.providers`

- **`_NameTranslatingChat{OpenRouter,Anthropic}` now un-mangle tool-call names on
  every public surface — `ainvoke` / `invoke` and `agenerate` / `generate` — not
  just `astream` / `_agenerate`.** The wrappers reverse-translate names from the
  underscored wire form (forced by strict provider validators) back to the
  canonical dotted form. That happened in the public `astream` override and in
  `_agenerate`, but `BaseChatModel.ainvoke` aggregates from the PROTECTED
  `_astream` (not `_agenerate`) whenever `_should_stream()` is true — i.e. a
  streaming callback is attached, as when running under an `astream_events` tap —
  so both overrides were bypassed and underscored names reached consumers whose
  tool-dispatch maps key on the dotted canonical name, causing silent tool-call
  misses. Both providers now override the public `ainvoke` / `invoke` to
  un-mangle the returned message (mirrors the `astream` strategy; overriding
  `_astream` directly would drop `on_chat_model_stream` callbacks), and override
  the batch `agenerate` / `generate` chokepoint (which `abatch` and direct
  callers route through, and which also aggregates from `_astream` under
  streaming) to reverse-translate every generation. All passes are idempotent
  with `_agenerate`'s translation (`reverse_translate_message` keys on the
  underscored wire name, so a second pass is a no-op). Regression tests on both
  providers force the `_astream` aggregation path (`stream=True`) and assert the
  dotted name on each surface.

### Added — `3tears-langgraph` — `threetears.langgraph.nodes`

- **`agent_node` honours a `preassembled_messages` flag.** A host app that has
  already assembled the full message list — system prompt, history, and a
  trailing post-history injection in a deliberate position — was having its
  ordering rewritten by the node's default system-message hoist/merge. With the
  flag set, the node passes the messages through untouched (no hoist, no
  `str()` coercion, no merge), letting the caller own prompt assembly while still
  using the converged loop. Default behaviour is unchanged.

### Fixed — `3tears-langgraph` — `threetears.langgraph.hooks`

- **`_ComposedToolNodeHook` no longer lets a hook emitter abort a turn.** The
  composer now wraps `on_tool_start` / `on_tool_end` / `on_heartbeat` emitter
  dispatch in a guard that logs and swallows exceptions (a dispatch with no run
  context, a transient event-bus error) instead of propagating them out of the
  tool node and crashing the turn. `GraphBubbleUp` (LangGraph's control-flow
  signal for interrupts/commands) is re-raised first so the guard never
  swallows legitimate graph control flow.

## v0.13.4 -- 2026-06-22

Adds the op-log stream-head read a consumer needs to reconcile its
committed-through cursor against an external record when the two diverge.

### Added — `3tears-nats` — `threetears.nats.oplog`

- **`OpLog.last_seq() -> int`** — the stream's current head sequence (one `stream_info()`
  read, O(1)): the seq the next `append` will follow, i.e. the value `expected_last_seq`
  must equal. A consumer that derives the op-log head from an external record (e.g. a git
  `Op-Seq` commit trailer) can clamp to this when the two diverge — a reset/fresh stream
  sitting behind an ahead-of-it external record — instead of wedging on a CAS that can never
  match the shorter stream.

## v0.13.3 -- 2026-06-21

Completes the conversation-folder relationship (referential integrity + helpers),
hardens the write-buffer flush against orphaned writes, and adds a per-dimension
column-coverage probe to datasources.

### Added — `3tears-conversations` — `threetears.conversations`

- **Folder referential integrity (migration v009).** A `UNIQUE(folder_id)` on `folders`
  plus an FK `conversations.folder_id → folders.folder_id` **ON DELETE SET NULL`, so
  deleting a folder auto-unfiles its conversations at the DB level (no consumer can
  forget the unfile). `ConversationsCollection.clear_folder(agent_id, folder_id)` — the
  cache-coherent unfile-all (routes each conversation through `save_entity` so L1/L2 are
  invalidated, vs a raw L3 UPDATE) — and `count_by_folder(agent_id, folder_id)`, the cheap
  per-folder count peer of `find_by_folder`.

### Fixed — `3tears` (core) — `threetears.core.collections.flush`

- **Orphaned writes no longer poison the atomic batch.** `flush_pending` now partitions the
  drained buffer by retry count: only never-failed writes (`retries == 0`) enter the atomic
  batch; any write that has already failed (`retries > 0`) — e.g. an orphan whose FK parent
  was deleted and will never return — routes straight to the per-entity loop. Previously one
  un-satisfiable write aborted the whole transaction every cycle until it exhausted its
  ~100-retry budget, forcing per-entity fallback for ALL co-buffered writes. The per-entity
  safety net + FK-aware re-enqueue is unchanged.

### Added — `3tears-datasources` — `threetears.datasources`

- **Per-dimension column-value coverage.** `Driver.column_value_coverage_by_dimension(schema,
  table, dimension_column, columns)` — the grouped sibling of `column_value_coverage`: one
  `GROUP BY` pass reporting non-null/non-zero coverage per numeric column per dimension value,
  so a caller can see a column loaded for some dimension values but all-zero for others (the
  partial-coverage case the whole-table probe can't see). Concrete on the `Driver` ABC
  (portable SQL routed through `fetch`), so every backend inherits it.

## v0.13.2 -- 2026-06-21

Conversation folders (a reusable grouping primitive lifted from metallm) and a Redshift
connection-concurrency cap. Additive across `3tears-conversations` and `3tears-datasources`.

### Added -- `3tears-conversations` -- `threetears.conversations`

- **Folder system** — `Folder` entity + `FolderCollection`: an app-agnostic, mutable, per-owner
  named container that groups conversations, lifted from metallm's product-side feature so any
  3tears consumer reuses one canonical entity. Scoped per `(agent_id, folder_id)` with a `name` and
  a free-form `metadata` JSONB (app presentation: color/icon/sort_order). Adds the `folders` table
  and a nullable `conversations.folder_id` (migration v008).

### Changed -- `3tears-datasources` -- `threetears.datasources`

- **Cap simultaneously-open Redshift connections.** A burst of N concurrent `fetch()` could open N
  connections past the warehouse user's `CONNECTION LIMIT` even after the 0.13.1 bounded-cache fix.
  An `asyncio.Semaphore` sized to `connection_cache_size`, acquired before opening, now bounds
  concurrently-open connections to the cache size (the executor still bounds concurrent work).

## v0.13.1 -- 2026-06-21

Patch: size the Redshift warm-connection cache as a bounded pool so concurrent
queries reuse warm connections instead of overshooting a tight per-user Redshift
CONNECTION LIMIT.

### Fixed -- `3tears-datasources` -- `threetears.datasources`

- Redshift warm-connection cache is now a bounded pool. `executor_max_workers`
  previously defaulted to 10 while `connection_cache_size` defaulted to 3, so
  concurrent queries past the cache opened a fresh connection every time — which
  overshoots a tight per-user Redshift CONNECTION LIMIT and fails with "too many
  connections" (the cache never acted as a pool). Now `executor_max_workers`
  defaults to 5 and `connection_cache_size` defaults to `executor_max_workers`
  (cache == workers) via a model validator, so queries past the bound queue on the
  executor and reuse warm connections rather than opening doomed ones. Set both
  per datasource to the user's connection limit.

## v0.13.0 -- 2026-06-21

### Changed — `3tears` (core) — BREAKING

- **Neutral L3 store seam (`collections-task-06`).** The collection framework's L3
  (durable) tier extension points were renamed to be storage-agnostic so a non-SQL
  backend (e.g. a git working tree) can be an L3: `fetch_from_postgres` →
  `fetch_from_store`, `save_to_postgres` → `save_to_store`, `delete_from_postgres` →
  `delete_from_store`, `persist_to_postgres` → `persist_to_store`. Behavior unchanged;
  `SchemaBackedCollection` generates the new names. `l3_pool`/`get_l3_pool` and
  `serialize`/`deserialize` are unchanged. **Consumers: see
  `docs/migrating-to-l3-store-seam.md`.**

- **Atomic write-buffer flush (`collections-task-06` L3B-04) — BREAKING for hand-rolled
  overrides.** `flush_pending` now persists a toposorted batch inside ONE backend
  transaction (degrading to the per-entity loop when the backend has no usable
  `transaction()`). To carry the transaction handle, `BaseCollection.persist_to_store`
  calls `save_to_store(data, *, conn=...)`. The base + `SchemaBackedCollection` signatures
  already accept `conn`, so the schema-backed collections are unaffected — but any
  **hand-rolled `save_to_store` override** with the old signature now raises
  `TypeError: ... unexpected keyword argument 'conn'` on every flush. Fix by migrating the
  collection to `SchemaBackedCollection` (preferred) or threading `conn` through the
  override. **Consumers: see `docs/migrating-to-l3-store-seam.md`.**

### Fixed — `3tears` (core)

- **L2 serde now round-trips `NUMERIC` columns as `Decimal`.** The schema-backed
  L2 (JSON) codec handled `UUID`/`datetime`/`bytes` but not `Decimal`, so
  serializing any row with a `NUMERIC_TYPE` value (a money/metric column) raised
  `TypeError: Object of type Decimal is not JSON serializable` — and the decode
  side had no `NUMERIC` branch either, so a value would have come back as a bare
  string/number rather than `Decimal`. `json_default` now emits `Decimal` as its
  exact decimal string and `decode_l2_value` rehydrates a `NUMERIC` column back to
  `Decimal` (via `Decimal(str(value))`, tolerating a float/int producer too — app
  code that computes a cost as a `float` round-trips losslessly without the
  binary-float expansion `Decimal(float)` would introduce). Surfaced by metallm
  migrating its cost-bearing collections onto `SchemaBackedCollection`.

### Changed — `3tears-nats` — BREAKING

- **`deadletter_on_error` → `deadletter_on_failure`.** The `NatsClient.subscribe` /
  `subscribe_typed` parameter (and the subscribe log field of the same name) was renamed so
  benign config field names no longer read as errors in log/alert greps. Behavior is
  identical — it still controls whether a callback/validation failure republishes to
  `{ns}.deadletter.{subject}`. Update any call site that passes the keyword explicitly
  (`grep -rn deadletter_on_error`); callers relying on the `True` default need no change.

### Added — `3tears` (core)

- `threetears.core.backends.L3Backend` (raw-SQL transport) and `DurableStore` (SQL-free
  structured ops: `fetch_one`/`upsert`/`delete`/`scan`) protocols; `SqlL3Backend`
  implementing both over an asyncpg pool; `DurableStoreCollection` (a collection whose L3
  tier is a `DurableStore` — the base a git-backed collection subclasses); and
  `parse_rowcount`, the one framework-owned asyncpg status-tag parser.

### Added — `3tears-scheduled-jobs` (new package)

- The generic, multipod-safe scheduled-jobs core, generalized from
  agent-wake's tick machinery onto a payload-agnostic, consumer-neutral
  surface. `threetears.scheduled_jobs.tick` — the pure-async tick engine
  body a consumer's scheduler (e.g. APScheduler) invokes per interval;
  it enumerates due jobs via `ScheduleStore.list_due_for_tick` (a
  deliberate `__SPANS_PARTITIONS__` cross-partition scan) and claims each
  via an optimistic-CAS on `next_fire_at = expected_next_fire`, so two
  ticks across pods can never double-fire one job.
- `threetears.scheduled_jobs.reschedule` — the next-fire computation
  (interval / one-shot / terminal), with `coalesce` / `catch_up`
  missed-fire policies.
- Store protocols (`ScheduleStore` / `FireStore` / `DueSchedule`) the
  tick engine talks to, plus a default three-tier store keyed on an
  opaque `kind` (TEXT) + `payload` (JSONB): the `scheduled_jobs` +
  `job_fires` tables (partition column `partition_key`, composite PKs,
  `ON DELETE CASCADE` fire history), `ScheduledJobCollection` /
  `JobFireCollection`, and the v001 migration. The platform never
  inspects `kind` / `payload`.
- `config` (tick limits / policy defaults), `events` (lifecycle event
  names), and `metrics` (the `threetears_scheduled_jobs_` Prometheus
  instruments — fires / failures / tick-duration / drift — with the
  forbidden-label cardinality guard preserved). `prometheus_client`
  stays an optional extra; the emitter no-ops gracefully when absent.

### Changed — `3tears-agent-wake` — BREAKING

- **Tick engine delegates to `3tears-scheduled-jobs` (S-2).** The cross-pod tick
  pump (lock acquire/degrade-open, due-scan, optimistic-CAS claim, per-fire
  isolation, drift) and the reschedule math now live ONCE in the generic
  scheduled-jobs core; `threetears.agent.wake.tick` is a thin adapter over it.
  The wake-facing contract is UNCHANGED: `wake_tick_job(pool, nats_client,
  dispatch_callback)`, the wake-shaped `DispatchCallback`, `WakeTrigger`,
  `WakeDispatchResult`, the schedule/fire schema, the richer `FireStatus`, and
  the webhook / `[SILENT]` handling all stay put. The cross-pod lock key stays
  `"agent_wake_tick"`.
- **Removed `threetears.agent.wake.reschedule`** (and its private
  `_compute_next_fire_at`). The identical math is now public at
  `threetears.scheduled_jobs.compute_next_fire_at` — same positional signature.
- **Dropped the direct `3tears-nats` dependency** (added `3tears-scheduled-jobs`).
  The cross-pod lock now belongs to the scheduled-jobs core; wake reaches NATS
  only transitively. No code change for consumers that pass a `nats_client`
  through `wake_tick_job` (still typed `Any`).
- **Tick Prometheus metrics moved to the `threetears_scheduled_jobs_*` family.**
  The per-fire / drift / tick-duration counters the tick used to emit on the
  `threetears_agent_wake_*` instruments now come from the scheduled-jobs emitter;
  the CAS-miss failure reason changed `conv_busy` → `claim_lost`, and the
  per-fire `execution_mode` label is no longer on the tick fire counter. The
  genuinely wake-specific `threetears_agent_wake_yield_duration_seconds` is
  preserved (re-emitted by the adapter). Webhook / rate-limit / schedule-cap
  metrics are unchanged. The `EVENT_FIRE_SKIPPED_BUSY` log's `extra_data` keys
  changed (`conversation_id`/`fire_source`/`execution_mode` → `job_id`/
  `partition_key`). **Operators: update dashboards/alerts that key on the old
  `agent_wake` tick metrics or the `conv_busy` reason.**
- **Consumers: see `docs/migrating-agent-wake-to-scheduled-jobs.md`.**

### Added — `3tears-nats`

- **Owner-routed request forward** (`threetears.nats.forward` / `serve_owner`) — a
  generic, payload-agnostic primitive for "send a request to whichever pod currently
  *serves* a key, and get its reply." It is the messaging half of a single-writer
  pattern; it does NOT elect a leader (a separate `nats_distributed_lock` / `KVLease`
  decides who serves — the consumer ties them together). `serve_owner(nats, key,
  handler)` is an async context manager a consumer runs *while it holds the key*: it
  subscribes the key's forward subject in a **queue group keyed by that subject**, so a
  brief two-owner overlap during lease handoff still dispatches each request to exactly
  one owner. `forward(nats, key, payload, *, timeout) -> bytes` requests that subject and
  returns the owner's reply bytes. Payload + reply are opaque `bytes`.
- **Typed forward errors.** No current owner (no subscriber / timeout in the handoff
  window) raises `NoOwnerError`; an owner whose handler *raised* surfaces to the caller as
  `ForwardedHandlerError` carrying the original exception's **type name + message** (so a
  consumer can map a forwarded failure back onto its own typed exception). Both subclass
  `ForwardError` (← `NatsClientError`). Wire framing is a 1-byte tag (`0x00` ok + verbatim
  reply bytes; `0x01` err + UTF-8 JSON `{type, message}`), keeping an empty/arbitrary
  handler reply unambiguous from an error frame.
- **`Subjects.forward(key)`** — derives `{ns}.forward.{sha256hex(key)}`; the key is hashed
  (not `_sanitize`-mapped) so arbitrary app keys carrying `.`/spaces/`*`/`>` map to a
  subject-safe, collision-resistant, deterministic token, mirroring `Subjects.room`.
## v0.12.3 -- 2026-06-20

Broker isolation, NATS durability, prompt-cache cost accounting, and Redshift
hardening. Additive across `3tears`, `3tears-models`, `3tears-nats`, and
`3tears-langgraph`; one fixed-path change to the Redshift connection lifecycle.

### Added -- `3tears-models` -- `threetears.models`

- `UsageRecord` carries `cache_read_tokens` and `cache_creation_tokens`, surfaced
  as `llm.cache_read_tokens` / `llm.cache_creation_tokens` OTel span attributes,
  so prompt-cache hits and writes are tracked per call.
- `registry_loader` populates `cost_per_cache_read_token` and
  `cost_per_cache_write_token` from the capabilities registry, so cache-aware
  cost can be computed downstream instead of billing cached input at the full
  input rate.

### Added -- `3tears-nats` -- `threetears.nats`

- Bounded redelivery + dead-letter in the durable consumer factory
  (`resilience-task-01` RES-01-01/03): a message that fails past its redelivery
  budget is routed to a dead-letter subject instead of redelivering forever.
- Agent-deregister subject on `Subjects`, so a pod can announce its teardown.

### Added -- `3tears` -- `threetears.core`

- Per-call `customer_scope` channel on `NatsProxyL3Backend` reads -- the proxy
  carries the caller's customer scope per read rather than per connection, the
  substrate for conversation-scoped RBAC pool reads (broker isolation).
- Centralize JSONB through native binding under the codec (`collections-task-04`,
  Option B): a single binding path for JSONB columns, plus an enforcement drift
  guard (`test_jsonb_native_binding`) so a new column cannot silently bypass it.

### Added -- `3tears-langgraph` -- `threetears.langgraph`

- Turn-level keepalive on `StreamingResponse` (`long-response-task-01` LRT-02):
  a long single response emits periodic keepalives so the stream does not idle
  out mid-generation.

### Added / Fixed -- `3tears-datasources` -- `threetears.datasources`

- Per-column value-coverage probe: classifies a column as unloaded when every
  value is zero across the table -- the `UNLOADED_COLUMN` source the hub mirrors
  into datasource read results -- with driver-coverage tests.
- Redshift: re-apply `search_path` on every connection acquisition, so a pooled
  connection no longer serves a stale path left by a prior caller.

### Changed -- `3tears` -- `threetears.enforcement`

- The fake-parity walker accepts an inline `# parity-exempt: <rationale>` marker
  (matching the cache/underscore exemption style), removing the line-shift
  fragility of the prior line-numbered exemption file.

## v0.12.2 -- 2026-06-17

Additive: add the documented-schema digest entity + collection to
`3tears-datasources` — the materialized, by-pk schema/concept summary the hub
publishes per datasource and agent pods read at conversation start (the
foundation for schema priming). No behavior change to existing datasource
collections.

### Added — `3tears-datasources` — `threetears.datasources`

- `DataSourceSchemaDigest` entity + `DataSourceSchemaDigestCollection`, a
  three-tier collection keyed by `datasource_id` for a by-pk hot-L1 read with
  L2/L3 fallback and cross-pod invalidation. The table has no `id` column, so
  `primary_key_column = "datasource_id"` (the `BaseCollection` default would
  emit `WHERE id = ?` / `ON CONFLICT (id)` and break every by-pk read +
  invalidation). One row per datasource; the `tables` projection is JSONB.

### Fixed — `3tears-datasources` — `threetears.datasources`

- JSONB write double-encode: a pre-`json.dumps`'d string bound as `::jsonb` was
  re-encoded by the text-format jsonb codec into a scalar. Digest writes now
  text-cast (`::text::jsonb`) so the value lands as a real JSONB array. Covered
  by a real-L1 round-trip test (no-codec test pools gave a false green).

## v0.12.1 -- 2026-06-16

Patch: stop the OpenRouter wrapper from logging streaming tool-call
continuations as junk tool names. Every DeepSeek tool turn produced a
per-chunk WARNING storm (`dropped invalid_tool_calls entry with junk name:
None`) that buried real signal; the dropped entries were harmless to tool
arguments (the chunk merge re-derives from `tool_call_chunks`), but the
noise was severe. No behavior change to tool dispatch.

### Fixed — `3tears-models` — `threetears.models`

- `filter_invalid_tool_calls` now treats a nameless `invalid_tool_calls`
  entry (`name` None / absent / empty) as a normal streaming-continuation
  fragment — kept, never logged. Only a concrete, undispatchable name claim
  is rejected: a non-empty string failing the canonical tool-name regex
  (the genuine junk case, e.g. a quote-garbage name leaked from XML-shaped
  tool-call text) or a non-string / non-dict value. Genuine-junk rejection
  is unchanged. Verified by a local A/B on a real DeepSeek-over-OpenRouter
  tool turn: 12 `junk name: None` warnings before, 0 after.

## v0.12.0 -- 2026-06-15

Durable channel-answer delivery and native Slack rendering. A finished
agent answer is published to a durable JetStream subject and delivered
out-of-band, so an answer that takes minutes — or completes while the
channel adapter is restarting — is delivered, never lost. Agent markdown
now renders into native Slack Block Kit instead of arriving as raw text.

### Added — `3tears-channels` — `threetears.channels`

- `markdown_to_slack_blocks` — converts GitHub-flavored markdown into native
  Slack Block Kit: mrkdwn emphasis/links, `header` blocks, native `table`
  blocks (numeric columns right-aligned), code fences, and dividers, bounded
  to Slack's per-message limits. `SlackAdapter` now always renders answers as
  blocks with a plain-text fallback, and `post_message` delivers a finished
  answer out-of-band on the bot token.
- `ChannelDeliveryMessage` — the durable channel-delivery envelope, with a
  NATS-KV-valid `dedup_key` making at-least-once delivery post at-most-once.

### Added — `3tears-nats` — `threetears.nats`

- JetStream durable-delivery helpers on `NatsClient`: `ensure_jetstream_stream`
  (create-or-reconcile), `jetstream_publish` (PubAck-awaited), and
  `jetstream_subscribe_durable` (manual-ack consumer).
- `Subjects.channels_deliver` / `channels_deliver_wildcard` — the
  `{ns}.channels.deliver.{channel_type}` delivery subject family.

## v0.11.0 -- 2026-06-13

The governed-knowledge layer: agents answer data questions with curated,
scoped business knowledge instead of guessing. Concepts (a business term →
its data binding) and playbook entries (procedures) merge across the
platform / customer / user scope ladder; datasources are shareable across
customers with origin lineage; the model registry becomes a single source
of truth.

### Added — `3tears` (core) — `threetears.knowledge`

- Governed-knowledge merge: `merge_concept_views` / `merge_entry_views`
  resolve the three-scope shadow ladder (user > customer > platform, D4),
  flag ambiguity when same-name definitions compete with no declared shadow
  (D5), and honour the `always_inject` invariant (KNW-25). One shared
  `resolve_shadow_chains` walk, so the hub eval fingerprint and a live SDK
  turn agree byte-for-byte on the effective view.
- `ConceptSnapshot.datasource_table_ref` + `build_table_ref` — a concept's
  bound table renders as its agent-usable `schema.table` name (one source
  of truth for the format), never the raw `datasource_table_id` UUID the
  agent has no tool to resolve.
- `EntryEnforcement` constraint on playbook-entry snapshots; draft-command
  wire models + tool `BootstrapContext` for the correction-harvest surface.
- `repoint_user_rows` + `MemoryRepointResult` — the user-merge repoint
  primitives (`threetears.agent.memory`, `threetears.conversations`).

### Added — `3tears-agent-acl`

- Shared caller-visibility SQL: `three_scope_visibility_clause` +
  `customer_scope_visibility_clause` — one copy of the security SQL that
  admits a row iff it passes the platform/customer/user read rule. Every
  RBAC-scoped list composes it; no per-row Python visibility filter.

### Added — `3tears-datasources`

- Platform-sharing: a flat datasource PK, visibility, and origin lineage
  (`origin_datasource_id`) so a customer datasource inherits a
  platform-shared datasource's schema docs + governed knowledge.

### Added — `3tears-models`

- Single source of truth for model ids + capabilities, with a no-literal
  guard that keeps stale model strings out of the codebase.

### Added — `3tears-nats`

- `hub_channel_installs` subject so the Slack adapter fetches its active
  installs over NATS (sandboxed; no DB credentials cross the wire).

### Fixed

- `threetears.langgraph` — `NOSTREAM_TAG` + `replace_content` keep internal
  model calls out of the user-facing stream; the bound-model cache degrades
  gracefully on an unhashable model.
- `threetears.knowledge` — `EntryEnforcement.canonical_sql` is truly
  optional; hardened the core by-pk read + langgraph injection.

## v0.10.5 -- 2026-06-03

A reusable keyset (seek) paginator in `threetears.core` for paging large,
append-heavy ordered lists without `LIMIT`/`OFFSET` drift.

### Added — `3tears` (core)

- `threetears.core.pagination` — a shared cursor-pagination primitive. `Keyset`
  builds the `ORDER BY` clause and the composite row-value seek predicate
  (`(a, b) < ($1::text::t1, $2::text::t2)`) for a sort key + direction;
  `encode_cursor`/`decode_cursor` give an opaque, URL-safe base64-JSON cursor;
  `Keyset.page` trims the `+1` sentinel and emits the next cursor. The caller
  owns the SQL (columns are a trusted allow-list, never user input). Replaces
  ad-hoc `OFFSET` (which skips/repeats rows as the list grows under you) and
  hand-rolled "list-since" cursors. Exported from `threetears.core`:
  `Keyset`, `Page`, `CursorError`, `encode_cursor`, `decode_cursor`.
- Cursor values round-trip through JSON, so non-native key types (`datetime`,
  `UUID`, `Decimal`) serialize to strings; the keyset binds them as `text` and
  casts (`$1::text::timestamptz`) so drivers like asyncpg accept the string and
  Postgres parses it — the paginator pages by a timestamp key, the common case.

## v0.10.4 -- 2026-06-03

Single-node NATS resilience: the platform now survives a NATS restart on
ephemeral JetStream storage instead of silently losing the wake heartbeat.

### Fixed — `3tears-agent-wake`

- `wake_tick_job` degrades open when the cross-pod lock cannot be acquired
  (`KvError` -- the bucket/stream is gone after a NATS restart on ephemeral
  storage -- distinct from `LockHeld`): the tick body runs anyway, since
  per-schedule mutual exclusion is the Postgres optimistic-CAS in
  `WakeScheduleCollection.claim_and_reschedule`, not the lock. A NATS wipe no
  longer silences the wake heartbeat for hours until a process restart. Worst
  case under a NATS outage: every pod runs the due-scan and contends on the
  CAS (the handled `SKIPPED_BUSY` path) -- no double-fires, no data loss.

### Fixed — `3tears-nats`

- `NatsKvBucket` self-heals a vanished stream/bucket. A single-node NATS
  restart on ephemeral JetStream storage wipes every stream and KV bucket;
  the client caches bucket handles, so every op then failed forever
  (`nats: no response from stream`) until the process restarted. The bucket
  now retains its open config and, on a transport failure (not KeyNotFound /
  CAS-conflict), re-opens once -- recreating the bucket when `create_if_missing`
  -- and retries the op. The handle heals in place, so the client bucket cache
  needs no flush; a second failure surfaces as `KvError` as before.

## v0.10.3 -- 2026-06-02

Three platform features consumed by metallm: a per-schedule wake
conversation-history switch, conversation-search date filters, and
tool-result dedup (the foundation for bounding agent context bloat).
Plus a cron-scheduling correctness fix.

### Added — `3tears-agent-wake`

- `agent_wake_schedules.include_conversation_history` (BOOLEAN NOT NULL
  DEFAULT true, migration v006): per-schedule switch for whether a fire
  carries the conversation's recent history into the wake's LLM context.
  Threaded through the entity, collection, `WakeTrigger`, tick, the
  create/update/response API models, and the `wake_schedule_create` /
  `wake_schedule_update` tools. Independent of the attached skill's
  `prompt_mode` (persona) — the two compose.

### Fixed — `3tears-agent-wake`

- `CronTrigger.from_crontab` no longer adopts the host's local timezone:
  fire times are stored/compared in UTC, so a non-UTC host fired cron
  schedules at the wrong wall-clock instant. Now pinned to `_tz(config)`
  (UTC by default), matching every other schedule type.

### Added — `3tears-conversations`

- `ConversationsCollection.search` gains `date_field` (`"created"` |
  `"updated"`, allow-listed to a real column — never interpolated, raises
  `ValueError` otherwise) plus inclusive `date_after` / `date_before`
  bounds.

### Added — `3tears-agent-tools`

- Tool results dedup on `(tool, input)`: `ContextItemCollection`
  `upsert_tool_result` (sharing the extracted `_upsert_keyed` codepath
  with `upsert_variable`) on a new `ix_context_items_tool_result_key`
  partial-unique index (migration v004, non-destructive legacy-key
  suffix first). `context.save_tool_result(input_fingerprint=)` keys
  `tool_name + ':' + sha256(input)` and upserts; the shared
  `make_tool_result_dedup_key` lets storage and lookup agree (consumed by
  metallm's per-tool TTL result reuse).

## v0.10.2 -- 2026-06-01

Single-feature release on top of v0.10.1. `DatasourceConfig` now
threads `allowed_schemas` onto the connection's `search_path` at
open time so agents can write unqualified table names in their SQL
instead of fully qualifying every reference. Closes the Hub-side
pairing of the long-standing "agent must qualify every table" UX
papercut.

### Added — `3tears-datasources`

- `RedshiftConnectionConfig`, `PostgresConnectionConfig`, and
  `YugabyteConnectionConfig` carry a new `allowed_schemas: list[str]`
  field (default `[]` means "leave the backend default in place").
- Shared helpers `build_search_path_value` /
  `build_set_search_path_sql` in
  `threetears.datasources.drivers._util` with identifier-quoting
  for adversarial schema names.
- Redshift driver issues `SET search_path TO "<schemas>"` via
  `cursor.execute` after the existing `SET statement_timeout` block
  on every connection open.
- asyncpg driver passes `server_settings={"search_path": "..."}`
  through `create_pool`, landing the value in the pgwire STARTUP
  packet so it survives `DISCARD ALL` reset on pool release. An
  `init=` callback would NOT — that was the trip-wire surfaced by
  the live testcontainer pass.
- Coverage: 8 new unit tests (4 per driver), 4 new live integration
  tests against Redshift and the asyncpg testcontainer.

## v0.10.1 -- 2026-05-29

Single-fix release on top of v0.10.0. `RedshiftDriver` now runs
`ROLLBACK` on a query error before returning the connection to its
cache so a single bad SELECT no longer poisons the cached session
for the rest of the consumer's conversation.

### Fixed — `3tears-datasources`

- `RedshiftDriver._acquire_and_run` catches the query exception,
  runs `conn.rollback()` through the existing sync bridge, and
  releases the rolled-back connection back to the cache. Cancel
  path stays as-is (`asyncio.CancelledError` is `BaseException`-
  rooted and propagates through the dedicated `_on_cancel`
  callback, not double-handled here). If the rollback itself
  raises, the connection is evicted instead of released and a
  WARNING is logged; the ORIGINAL query exception is what
  propagates to callers in every branch. Coverage: three new
  unit tests (mocked-cursor positive / rollback-failure / two-
  fetch end-to-end) plus one new live integration test against
  `central-reporting` gated on `OTS_REDSHIFT_PASSWORD`.

  Background: `redshift_connector` uses the DB-API default of
  `autocommit=False`. A failed statement leaves the connection's
  implicit transaction in `aborted` state and the server then
  rejects every subsequent statement on that connection with
  `25P02: current transaction is aborted, commands ignored until
  end of transaction block` until an explicit `ROLLBACK` runs.
  Without the rollback, the agent's tool loop on a typo'd SELECT
  spins through its recursion budget retrying because every retry
  inherits the same poisoned cached connection.

## v0.10.0 -- 2026-05-23

The long-running-agent foundation release. Three new platform features
land in lock-step: a tool-eligibility flag pair on the existing
`3tears-agent-tools` base class, a brand-new `3tears-agent-skills`
package for procedural memory, and a brand-new `3tears-agent-wake`
package for scheduled + webhook-triggered fires. Two existing packages
gain supporting capabilities: `3tears-nats` exposes a distributed-lock
primitive lifted from metallm; `3tears-channels` ships a generic
`WebhookReceiver` framework with a pluggable verifier registry.

The first consumer is metallm's long_running + skills work (separate
release on the metallm side that pins this 3tears version).

### Added — `3tears-agent-tools`

- `TearsTool.tool_eligible: bool = True` and `TearsTool.skill_eligible:
  bool = False` class attributes decouple "is this tool in the agent's
  default tool surface?" from "is this tool discoverable in the skills
  catalog?". The defaults preserve pre-v0.10.0 behaviour for every
  existing tool. Subclasses opt-in to the new visibility states.
- New `agent_tools_platform` PLATFORM-scope migration adds
  `tool_eligible` + `skill_eligible` BOOLEAN columns to `namespaces`
  with `DEFAULT TRUE` / `DEFAULT FALSE` so existing rows keep their
  pre-shard semantics.
- `ToolNamespaceEmitter` / `ToolServer.publish_registration` stamps the
  flags onto the namespace row and emits a structured WARNING when a
  tool registers with both flags False (would be invisible to every
  agent surface).
- `agent-acl.NamespaceCollection` gains
  `list_tool_namespaces_for_actor(...)` (default surface =
  `tool_eligible=True` ∩ ACL) and
  `list_skill_eligible_tool_namespaces(...)` (skills catalog UNION
  source). Eligibility filters AFTER ACL — eligibility decides
  VISIBILITY; ACL decides AUTHORIZATION.
- `agent-acl.builtin_roles` ships the `PlatformBuiltinToolUser` role
  definition + canonical pre-check `mcp_name` list (`http_get`,
  `loki_query`, `postgres_query`) + idempotent
  `ensure_platform_builtin_tool_user_role` bootstrap helper. The
  deploying app seeds the `role_assignments` rows post-registration
  (per-version namespace UUIDs only exist after `ToolNamespaceEmitter`
  runs).

### Added — `3tears-agent-skills` (new package)

- `agent_skills` + `agent_skill_invocations` tables (partition column
  `agent_id`, composite PK + standalone UNIQUE on bare id for
  cross-package FKs). FTS-maintained `search_vector` (weighted A/B/C
  over `name || trigger_keywords || body`) for `skill_list` query
  filtering — NOT for auto-load (auto-load via classifier is
  explicitly out of scope per the v1 design).
- `AgentSkillCollection` + `AgentSkillInvocationCollection` with the
  full method surface (find_by_name_for_user, list_for_user, bump_use_count,
  increment_outcome_counts, record, list_for_skill, set_message_id,
  set_outcome).
- Seven `TearsTool` factories: `skill_create`, `skill_list`,
  `skill_get`, `skill_update`, `skill_delete`, `skill_invoke`,
  `skill_introspect` (the last returns the minimal-token shape for
  cheap discovery). Per-user cap of 200 prose skills; ACL probe on
  every tool name in `tool_additions`; first-invoke-wins enforcement
  on `skill_invoke` (with consumer-supplied state probe + setter
  Callable hooks).
- `compose_turn_context(active_skill, base_system_prompt,
  base_tool_names, *, acl_permits) -> ComposedTurnContext` — pure
  per-turn composition function. `prompt_mode='additive'` appends body
  to base prompt; `prompt_mode='replace'` substitutes (consumer
  layers per-user additions like NSFW / jailbreak on top in either
  mode). `tool_additions` ACL-gated; `tool_restrictions` subtractive
  without ACL check. One skill per turn maximum (no multi-skill
  composition).
- `SkillRegistryClient` Protocol decouples the package from
  `3tears-agent-acl` / `3tears-agent-tools` dependencies — consumers
  wire concrete bindings via three small Callable hooks
  (`conversation_id_resolver`, `active_skill_probe`,
  `active_skill_setter`) + a three-method Protocol surface
  (`acl_permits`, `list_skill_eligible_tools`, `get_tool_introspect`).

### Added — `3tears-agent-wake` (new package)

- `agent_wake_schedules`, `wake_fires`, `webhook_subscriptions` tables
  (partition column `conversation_id`; nullable `skill_id` FK on
  schedules; nullable `default_skill_id` FK on webhook subscriptions —
  single skill per wake / per subscription per the v1 design;
  `webhook_subscriptions.endpoint_secret_ciphertext` BYTEA Fernet-
  encrypted, decrypted via `EncryptionService` Protocol). All
  migrations idempotent; cross-package FKs land via post-creation
  guarded ALTER blocks.
- `wake_tick_job(pool, nats_client, dispatch_callback, *, wake_config)`
  — pure-async tick body the consumer's APScheduler
  `IntervalTrigger(seconds=60)` job invokes. Atomic CAS claim per
  schedule via `WakeScheduleCollection.claim_and_reschedule` (two
  ticks cannot fire the same schedule). Missed-fire policies
  `'coalesce'` (default) and `'catch_up'`; drift-recording via
  `wake_fires.scheduled_fire_at` + `wake_fires.actual_fired_at`.
  Per-fire skip emits `EVENT_FIRE_SKIPPED_BUSY`. Wake-yield
  cooperative-interrupt support via `wake_fires.status='yielded'` +
  yield-duration histogram.
- `_compute_next_fire_at(schedule, now)` covers all seven schedule
  types (cron / daily_at / one_shot / random_window /
  relative_delay / interval + the existing). DST-correct via stdlib
  `zoneinfo` (spring-forward + fall-back integration tests pinned).
- `dispatch_wake(trigger, fire_id, pool, *, handler, wake_config,
  delivery_adapters)` — sole entry point every wake source flows
  through (tick + webhook). Resolves attached skill (single-skill
  per PLACEMENT §1.3); resolves `context_from` single-hop
  same-conversation chain with 16KB truncation; invokes the consumer's
  `HandlerCallback`; detects `[SILENT]` prefix on response
  (case-insensitive, whitespace-tolerant); routes delivery to each
  target via the supplied `DeliveryAdapter` Protocol mapping
  (silent fires skip delivery; raised adapter exceptions caught +
  logged WARNING, fire still marked success because the LLM produced
  output). `_check_rate_limit` enforced at step 1 (per-conv per-day +
  per-user per-day; per-subscription per-hour on the webhook path).
- Fourteen `TearsTool` factories: six wake-schedule
  (`wake_schedule_create` / `_update` / `_list` / `_pause` / `_resume`
  / `_delete`) + seven webhook-subscription
  (`webhook_subscription_create` / `_update` / `_list` / `_pause` /
  `_resume` / `_delete` / `_rotate_secret`) + `wake_yield` (gated to
  load only on wake-driven turns via `is_wake_turn()` closure). Skill
  attachment is via the create/update `skill_id` parameter — no
  separate `wake_skill_attach` / `wake_skill_detach` tools. Detach
  semantics use explicit `detach_skill: bool = False` /
  `detach_default_skill: bool = False` / `clear_name: bool = False`
  fields because LangChain `@tool` cannot distinguish "field absent"
  from "explicit null".
- Per-conversation active-schedule cap (`WakeConfig.
  max_schedules_per_conversation = 10` default per PLACEMENT §1.9).
  App-side cycle detection on `context_from_schedule_id` (single-hop
  same-conversation; max-depth 10 defense-in-depth). ACL probe on
  every `skill_id` attached to a wake.
- `WakeConfig` Protocol + `DEFAULT_WAKE_CONFIG` constant — product
  supplies caps, URL allow-lists, named-query registries; platform
  honours.
- Prometheus instruments (prefix `threetears_agent_wake_*` — the
  documented `3tears_agent_wake_*` prefix is rewritten by
  `prometheus_client` because identifiers must match
  `[a-zA-Z_][a-zA-Z0-9_]*`): fires/failures/tick-duration counters,
  drift/yield-duration histograms, rate-limit/cap-rejection counters,
  webhook-received counter, delivery counter. No unbounded-cardinality
  labels (`conversation_id` / `user_id` / `schedule_id` /
  `subscription_id` / `agent_id` / `fire_id` are FORBIDDEN as
  labels). Enforcement test pinned at
  `tests/unit/test_metrics_cardinality.py`.
- Loki event-name constants (`EVENT_TICK_STARTED`, `EVENT_FIRE_*`,
  `EVENT_DELIVERY_*`, `EVENT_WEBHOOK_*`).
- Pydantic v2 request/response models in `api_models` for the wake
  REST surface (consumers import; metallm pins in shard-09 of the
  metallm long_running release). All models declare
  `extra='forbid'`; `pre_check_type` / `no_agent` /
  `pre_check_output` round-trip rejected (anti-patterns per
  PLACEMENT §1.2).

### Added — `3tears-nats`

- `nats_distributed_lock(client, key, *, ttl, heartbeat_interval,
  holder_id) -> AsyncContextManager` lifted from metallm's
  `scheduler_lock`. Atomic NATS KV `bucket.create()` claim; background
  heartbeat task refreshes lease before TTL; raises `LockHeld` on
  conflict; auto-expires on holder crash. Constant-time bucket-TTL
  mismatch check raises `ValueError` rather than silently inheriting
  the first caller's TTL.

### Added — `3tears-channels`

- `WebhookReceiver` framework (optional `[webhook]` extra; depends on
  `fastapi` + `3tears-agent-wake`). `register_verifier(scheme,
  callable)` lets vendor-specific schemes (GitHub `X-Hub-Signature-
  256`, Stripe `Stripe-Signature`, etc.) plug in. Default scheme
  `generic_hmac_sha256` ships with `verify_generic_hmac_sha256`
  (constant-time `hmac.compare_digest`). HTTP status mapping
  202 / 400 / 401 / 403 / 404 / 413 / 429 (with `Retry-After: 60`) /
  500. 1 MiB payload cap enforced BEFORE subscription lookup +
  secret decryption (closes cost-attack vector on unverified
  payloads).
- `verify_generic_hmac_sha256` + `compute_generic_hmac_sha256_signature`
  live at `threetears.agent.wake.hmac_util` (one shared
  implementation; both channels' receiver and agent-wake's adapter
  import from there).
- `webhook_subscriptions.verification_scheme` CHECK constraint opened
  in v005 migration (was hardcoded to the single
  `generic_hmac_sha256` literal; now `~ '^[a-z0-9_]+$' AND length
  BETWEEN 1 AND 64`). Registered schemes are validated at
  receiver-handle time (unknown → 400) since the DB cannot consult
  the live in-process registry.

### Notes

- All 18 workspace packages bumped to 0.10.0 in lock-step (the
  `3tears-agent-skills` + `3tears-agent-wake` packages are new in
  this release; the other 16 keep their existing surfaces with
  the additions documented above).
- Test count: 6,564 unit + 201 integration, all green.
  No new "ours-side" test warnings — the only remaining 67
  warnings are upstream (langgraph `LangChainPendingDeprecationWarning`
  + langchain_core `asyncio.iscoroutinefunction` deprecation).
- Migration ordering: `agent-skills` migrations (v001 + v002) land
  before `agent-wake` migrations (`depends_on=("conversations",
  "agent_skills")` enforces the topological order via the canonical
  `MigrationRunner`). The `agent-tools` PLATFORM-scope migration
  for the eligibility columns runs once at hub startup against the
  shared schema.
- Cross-package dep direction: `channels` → `agent-wake` (via the
  `[webhook]` extra) is the only new directional edge. `agent-wake`
  → `agent-skills` (single-skill resolution from
  `AgentSkillCollection`). No circular imports. The `nats`
  distributed-lock primitive is consumed by `agent-wake` (the tick
  body) and by metallm's existing backup job (which becomes a
  re-export when metallm pins this release).
- Backwards compatibility: NO breaking changes. The two new
  `TearsTool` flags default to the pre-v0.10.0 behavior.
  Migration v005 in `agent-wake` opens a previously-stricter
  CHECK constraint (additive); no schema breaks. All new tables
  and columns are additive. Existing consumers continue to work.

## v0.9.1 -- 2026-05-23

### Changed

- **`3tears-datasources` — pluggable secret resolution (Path A).**
  Datasource credentials are no longer named by an env var
  (`password_env` / `credentials_json_env`). They now carry a
  `scheme://locator` *reference* in `password_ref` /
  `credentials_json_ref`, resolved at driver-creation time (Hub-side,
  scoped to one datasource) by a pluggable backend in the new
  `threetears.datasources.secrets` module. The secret value never
  lives in agent.yaml, never lands plaintext in the Hub DB, and never
  sits in a long-lived process variable — it is only ever held inside
  a `SecretStr` and unwrapped at the last moment when handed to the
  backend lib. Shipped backends:
    - `env://NAME` — read process env var `NAME` (the devx backend;
      devx mounts the agent project `.env` into the Hub container so
      every datasource credential resolves on a fresh stack with no
      per-secret hand-listing).
    - `k8s://rel/path` — read a projected-Secret file under
      `AIBOTS_DATASOURCE_SECRETS_DIR` (default `/var/run/secrets/aibots`);
      the prod shape (k8s `Secret` as a volume).
  `vault://`, `aws-secretsmanager://` and `gcp-sm://` are registered
  but raise a clear "not implemented" error so the scheme surface is
  stable for config authors today. Config validators call
  `validate_ref` at load time (shape/scheme check, no env/fs touch);
  resolution stays a use-time concern. This is a hard rename with no
  backwards-compatibility shim.
- **`3tears-datasources` realigned to the monorepo lockstep version.**
  The package had been on an independent `0.1.x` line; it now versions
  with every other workspace package (`0.9.1`). Its README "Versioning
  policy" and CHANGELOG were rewritten accordingly.

### Notes

- Patch bump: the only behavioural change is internal to
  `3tears-datasources` (the credential-reference rename + resolver).
  No other package's public API changed.
- All 17 workspace packages bumped to 0.9.1 in lock-step (the
  `3tears-datasources` package joined the lockstep this release).
- The platform Docker image stamp tracks this tag (`v0.9.1`); the
  devx compose now injects the whole agent `.env` into the Hub
  container generically, retiring the per-secret passthrough.

## v0.9.0 -- 2026-05-20

### Added

- `threetears.models.chunk_merging.merge_chunks` -- canonical merge of
  streamed `AIMessageChunk` lists into a single `AIMessage`. Wraps
  LangChain's `AIMessageChunk.__add__` for the merge, finalizes to a
  concrete `AIMessage`, and preserves `invalid_tool_calls` for
  downstream recovery. Replaces inline duplicates across consumers
  (metallm personality node, 14-eng-ai-bot router,
  14-eng-ai-bot-agents tool loop).
- `threetears.models.chunk_parsing.parse_chunk` -- canonical extractor
  of `(text, reasoning)` per streamed chunk. Covers all three
  observed shapes (OpenAI / OpenRouter string content, Anthropic-direct
  list-of-blocks, OpenRouter / OpenAI reasoning models'
  `additional_kwargs["reasoning_content"]`) and mixed cases. Pure,
  no-I/O hot-path helper.
- `threetears.models.tool_name_validation` -- canonical tool-name
  validator (`is_valid_tool_name`, `validate_tool_name`,
  `filter_invalid_tool_calls`, `ToolNameValidationError`). Pins the
  3tears tool-name regex (`^[a-zA-Z0-9_.-]{1,64}$`) covering every
  observed provider validator plus the dotted canonical form.

### Fixed

- Closes the metallm 2026-05-19 prod incident (conv
  `019e3e26-9870-7a03-8f04-8cc6a4f5f418`) where a misbehaving
  model response surfaced a tool-call name with an embedded
  XML-attribute fragment (`memory_recall" name="memory_recall`).
  The junk name reached metallm's dispatch layer through the
  chat-model wrapper unfiltered and was persisted as an
  unrecoverable invocation. The OpenRouter and Anthropic provider
  wrappers now call `filter_invalid_tool_calls` on every streamed
  chunk and every `_agenerate` result, dropping junk entries with
  one `WARNING` log per drop (name truncated to 80 chars). This
  blocks `function.name` junk from reaching downstream dispatch in
  any 3tears consumer.

### Notes

- v0.9.0 is a minor bump because it establishes new wrapper-layer
  contracts that downstream consumers can rely on: clean tool
  names guaranteed at the chat-model boundary, plus the canonical
  chunk-parsing / chunk-merging utilities. Bugfix patch would have
  been wrong given the new public API surface.
- All 16 workspace packages bumped to 0.9.0 in lock-step.
- No backwards-incompatible changes. Existing consumers that
  inline their own chunk parsing / merging continue to work; the
  new utilities are opt-in.
