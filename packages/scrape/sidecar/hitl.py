"""Make the sidecar's existing X display reachable by a human, on demand.

The container already runs a real headful Chromium against Xvfb -- what it has never had is
any way for a person to SEE that display, let alone drive it. This module is that path, and
the session that sits on it.

Two layers, and the split is deliberate. :class:`VncLifecycle` is the display: it starts
``x11vnc`` and ``websockify`` in front of it, serves noVNC's static client, and stops both
again. :class:`SessionManager` is what a human actually works in: one session against the one
display, a bounded number of targets in it at a time, each in its own isolated browser
context, behind a token and a hard TTL.

Still no authorization here, and there will not be any: the sidecar holds no identity and
cannot evaluate a policy. It honours a token it minted; deciding who should have been given
one happens on the MIT side.

**Started on demand, not at boot.** An idle VNC surface is an attack surface that exists for
the 99% of the container's life when nobody is looking at it. The processes come up when a
person arrives and go away when they leave, so the steady state is the same container that
ran before this module existed.

**Loopback only, with one way in.** ``x11vnc`` binds ``127.0.0.1`` and is never published;
``websockify`` is the sole path from outside, which is what makes "who may connect" a
question answerable at one place later rather than at two. That single seam is the reason
the two processes are separate rather than x11vnc's own ``-http`` mode.

**The display number is a parameter from the first line.** One Xvfb display means one
operator at a time; more than that needs a display pool (``:100``, ``:101``, ...), each with
its own Chromium and its own ``x11vnc``. Parameterising now costs nothing and makes that pool
a configuration change rather than a rewrite -- but this is single-display, and says so.

Runs inside the AGPL-3.0 sidecar container and imports nothing from 3tears, exactly like
:mod:`main`. The boundary is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import shutil
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("nodriver_sidecar.hitl")

#: Where Debian's ``novnc`` package puts the client. Verified by installing the package in
#: the real base image rather than recalled: the tree holds ``vnc.html``, ``vnc_lite.html``
#: and ``vnc_auto.html`` and NO ``index.html``, which is why :attr:`VncSession.path` names a
#: page explicitly. Serving the directory alone would 404 at ``/``.
NOVNC_ROOT = os.environ.get("NOVNC_ROOT", "/usr/share/novnc")

#: The noVNC page to hand a caller. ``vnc.html`` with ``autoconnect``, not ``vnc_lite.html``.
#:
#: The lite page was chosen first because it connects on load where the full client waits behind
#: a settings sidebar, and for a human summoned to clear one challenge "it is already connected"
#: is the whole difference. ``autoconnect=true`` buys that from the full client too, and the
#: lite page cannot do the thing that matters more: it parses ``scale`` and nothing else, so it
#: has no way to ask the server to match the viewport. An operator on a laptop was left
#: scrolling a 1920x1080 desktop to reach a taskbar at the bottom of it.
#:
#: Verified against the installed tree rather than recalled: ``vnc_lite.html`` reads
#: ``scale``, and ``vnc.html`` reads ``resize``. The ``resize=scale`` this once handed out was
#: inert on the lite page -- a parameter that had never once done anything.
NOVNC_PAGE = "vnc.html"

#: RFB port ``x11vnc`` listens on, loopback only. Not published by the container and not
#: configurable per session, because there is exactly one display.
_RFB_PORT = 5900

#: How long to wait for each process to start listening before calling it a failure. Both are
#: local process spawns, so this is generous; the cost of being wrong is a session that
#: reports success and shows a black rectangle.
_START_TIMEOUT_SECONDS = 10.0

#: Poll interval while waiting for a port to accept a connection.
_POLL_INTERVAL_SECONDS = 0.1

#: How many targets may occupy one session at once. A bounded working set is the point: a
#: target holds its slot from the moment it is opened until a human says it is done, including
#: while it sits in the background being slow, so this is a real ceiling on concurrent tabs
#: rather than a number that is usually not reached.
DEFAULT_MAX_SLOTS = int(os.environ.get("HITL_MAX_SLOTS", "4"))

#: Hard session lifetime. A ceiling rather than an idle timeout, because an operator who walks
#: away mid-solve leaves a live browser holding a target's authenticated session, and "idle"
#: cannot tell that apart from "reading carefully".
DEFAULT_SESSION_TTL_SECONDS = float(os.environ.get("HITL_SESSION_TTL_SECONDS", "1800"))

#: How often the reaper checks. Well under the TTL, and not so often that an unattended
#: container spends its life waking up.
REAPER_INTERVAL_SECONDS = float(os.environ.get("HITL_REAPER_INTERVAL_SECONDS", "30"))

#: Budget for replaying a target's nav steps when pulling it into a session. Generous: a human
#: is already waiting by this point, and failing the replay costs them the whole target.
_NAV_STEP_TIMEOUT_SECONDS = 30.0

#: Ceiling on each CDP round-trip taken while completing a tab. These run OUTSIDE the session
#: lock so a wedged browser cannot block the reaper, but they still need their own bound, or a
#: caller waits forever on a request that has already given up its slot.
_COMPLETE_TAB_TIMEOUT_SECONDS = 20.0


class VncUnavailable(RuntimeError):
    """Raised when the VNC path cannot be brought up.

    Deliberately distinct from "not running": a caller that asked for a session and did not
    get one needs to know it failed, where a caller checking :meth:`VncLifecycle.health` on a
    stopped session is asking a question with a legitimate negative answer.
    """


@dataclass(frozen=True)
class VncSession:
    """Where a human should point their browser, and what is behind it."""

    #: Port ``websockify`` listens on. The only port a human ever reaches.
    web_port: int
    #: The X display being shared, e.g. ``":99"``.
    display: str
    #: Path to the noVNC client, including the query string that makes it self-connect.
    path: str


class VncLifecycle:
    """Starts and stops the two processes that put the X display in a browser.

    One instance per sidecar process, matching the one display it has. Not reentrant across
    event loops and not intended to be: the container runs a single uvicorn worker against a
    single Chromium against a single Xvfb, and pretending otherwise here would be modelling a
    deployment that does not exist.
    """

    def __init__(self, *, display_num: int | None = None, web_port: int = 6080) -> None:
        """
        :param display_num: X display to share; defaults to ``DISPLAY_NUM``, then 99. The
            same display ``entrypoint.sh`` started Xvfb on and Chromium is drawing to
        :ptype display_num: int | None
        :param web_port: port ``websockify`` serves noVNC and the RFB proxy on
        :ptype web_port: int
        """
        self._display_num = display_num if display_num is not None else int(os.environ.get("DISPLAY_NUM", "99"))
        self._web_port = web_port
        self._x11vnc: asyncio.subprocess.Process | None = None
        self._websockify: asyncio.subprocess.Process | None = None

    @property
    def display(self) -> str:
        """The X display this shares, in ``:N`` form."""
        return f":{self._display_num}"

    @property
    def web_port(self) -> int:
        """The port websockify serves on. The only port a human ever reaches."""
        return self._web_port

    @property
    def client_path(self) -> str:
        """Where to point a browser, query string included.

        Public because callers legitimately need it before a session is running -- the session
        endpoint publishes it in its response -- and reaching into a private for it made every
        stand-in for this class have to know about the underscore.
        """
        return self._novnc_path()

    def health(self) -> bool:
        """Whether both processes are running right now.

        Both, not either: websockify alive with x11vnc dead is a page that loads and never
        paints, which is indistinguishable from a broken display to the person looking at it
        and is the specific failure this chunk exists to avoid shipping.
        """
        return self._alive(self._x11vnc) and self._alive(self._websockify)

    async def start(self) -> VncSession:
        """Bring up the VNC path, or return the running one.

        Idempotent by contract: a second caller gets the session the first one started rather
        than a second ``x11vnc`` fighting over the RFB port. That matters because "open a
        session" is the operation a human-facing queue will retry.

        :return: where to point a browser
        :rtype: VncSession
        :raises VncUnavailable: when either process fails to come up
        """
        if self.health():
            return self._session()

        # A half-dead pair is not a running session and not a clean stopped one. Tearing the
        # remains down first means `start` has exactly two outcomes rather than three.
        await self.stop()

        # The guard is around the whole sequence, not around each await, because "exactly two
        # outcomes" is a property of the METHOD. `_await_port` cleans up after itself, but it
        # is not the only thing here that can fail after x11vnc is already up and listening:
        # `_websockify_argv()` resolves a binary and raises when the image lacks it, and a
        # cancelled start raises without going through any of it. Both would otherwise leave
        # an x11vnc holding the RFB port behind a lifecycle that reports not-running.
        #
        # The client tree is checked first, before anything is spawned at all: a missing one is
        # a certain failure, and finding out before there are processes to clean up is better
        # than finding out after.
        self._require_novnc()

        try:
            self._x11vnc = await self._spawn(self._x11vnc_argv(), what="x11vnc")
            await self._await_port(_RFB_PORT, what="x11vnc")

            self._websockify = await self._spawn(self._websockify_argv(), what="websockify")
            await self._await_port(self._web_port, what="websockify")
        except BaseException:
            await self.stop()
            raise

        log.info(
            "hitl: vnc session up on display %s, noVNC on port %d",
            self.display,
            self._web_port,
        )
        return self._session()

    async def stop(self) -> None:
        """Stop both processes and leave nothing listening.

        Safe to call when nothing is running, and safe to call twice -- teardown is the path
        an error handler takes, so it must not be able to raise a second error on top of the
        first one.
        """
        for proc, what in ((self._websockify, "websockify"), (self._x11vnc, "x11vnc")):
            await self._terminate(proc, what=what)
        self._websockify = None
        self._x11vnc = None

    def _session(self) -> VncSession:
        """Describe the running session."""
        return VncSession(web_port=self._web_port, display=self.display, path=self._novnc_path())

    def _novnc_path(self) -> str:
        """The client URL path, with the query string that makes it connect on load.

        ``path=websockify`` tells the client which endpoint to open its WebSocket against;
        websockify serves the RFB proxy on that path and the static tree on every other one.
        Without it the page loads and waits for a human to fill in a form, which is a worse
        experience than the black rectangle it resembles.
        """
        # ``resize=scale``, not ``remote``, and the reason is a property of Xvfb rather than a
        # preference. Xvfb creates ONE mode at startup and reports `maximum 1920 x 1080` with a
        # single entry in its mode list, so there is nothing for a client-driven resize to
        # resize to -- `resize=remote` is accepted and silently does nothing. Verified with
        # `xrandr` against the running container, not assumed.
        #
        # Scaling costs some sharpness on text an operator has to read, which is a real cost on
        # this screen of all screens. Making the desktop genuinely follow the viewport needs an
        # X server that can resize -- TigerVNC's Xvnc, which would replace both Xvfb and x11vnc
        # -- and that is a container change rather than a flag.
        return f"/{NOVNC_PAGE}?path=websockify&autoconnect=true&resize=scale"

    def _x11vnc_argv(self) -> list[str]:
        """``x11vnc`` invocation.

        - ``-localhost`` binds 127.0.0.1, so websockify is the only route in.
        - ``-nopw`` because there is no password to check: the sidecar authenticates nobody
          (it cannot -- it holds no identity), and pretending otherwise with a shared
          password would be security theatre over a loopback socket. The real gate is the
          token check that goes in front of websockify in a later chunk.
        - ``-forever`` so the first disconnect does not end the session; a human who closes a
          tab by accident should be able to come back.
        - ``-shared`` so a second viewer does not evict the first.
        - ``-noxdamage`` because Xvfb's DAMAGE extension reports are unreliable enough to
          leave stale rectangles on screen, and a stale screen is exactly the failure mode
          nobody notices until they have already clicked the wrong thing.
        - ``-xrandr resize`` so a server-side geometry change is picked up rather than leaving
          viewers on a stale size. It does NOT enable a client-driven resize here: Xvfb has one
          fixed mode, so there is nothing to resize to. Kept because it costs nothing and
          becomes correct the moment the X server is one that can resize.
        """
        return [
            self._require("x11vnc"),
            "-display",
            self.display,
            "-rfbport",
            str(_RFB_PORT),
            "-localhost",
            "-nopw",
            "-forever",
            "-shared",
            "-noxdamage",
            "-xrandr",
            "resize",
            "-quiet",
        ]

    def _websockify_argv(self) -> list[str]:
        """``websockify`` invocation: serve noVNC's tree and proxy to the RFB port.

        Positional form is ``[source_addr:]source_port [target_addr:target_port]``, verified
        against the installed binary's own usage line rather than recalled.
        """
        return [
            self._require("websockify"),
            "--web",
            NOVNC_ROOT,
            f"0.0.0.0:{self._web_port}",
            f"127.0.0.1:{_RFB_PORT}",
        ]

    @staticmethod
    def _require_novnc() -> None:
        """Fail if the client the returned path points at is not actually on disk.

        ``_require`` guards the two binaries, but the noVNC tree is an unguarded distro
        layout, and its failure mode is worse than a missing binary: both processes come up,
        ``start`` returns happily, and the path it hands back 404s. To the person told to open
        it that is indistinguishable from the black rectangle everything else here is written
        to avoid -- and unlike a crash it produces no log line anywhere.
        """
        page = os.path.join(NOVNC_ROOT, NOVNC_PAGE)
        if not os.path.isfile(page):
            raise VncUnavailable(
                f"the noVNC client is not at {page}; the image was built without the novnc "
                f"package, or the distro moved its asset tree"
            )

    @staticmethod
    def _require(binary: str) -> str:
        """Resolve *binary* or fail with the reason, rather than with ``FileNotFoundError``.

        The image installs both, so an absence here means the container was built without the
        VNC packages -- worth saying plainly, because the alternative message sends whoever
        reads it looking at this code instead of at the Dockerfile.
        """
        found = shutil.which(binary)
        if found is None:
            raise VncUnavailable(
                f"{binary} is not installed in this container; the image was built without VNC support"
            )
        return found

    async def _spawn(self, argv: list[str], *, what: str) -> asyncio.subprocess.Process:
        """Launch a process, surfacing a failure to launch as :class:`VncUnavailable`."""
        try:
            return await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.DEVNULL,
                # DEVNULL, not PIPE. A pipe nobody reads is a 64 KiB ceiling on the child's
                # life: websockify logs a line per connection and this session is built for
                # reconnects (-forever, -shared), so a long-lived operator session would
                # eventually fill it and block websockify inside a write -- surfacing as a
                # page that loads and never paints, which is the exact failure this module
                # exists to prevent. Draining it properly means a reader task per process for
                # output nothing consumes; discarding it is the honest trade, and the
                # diagnosis that matters (did it listen?) comes from the port wait.
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError as exc:
            raise VncUnavailable(f"could not start {what}: {exc}") from exc

    async def _await_port(self, port: int, *, what: str) -> None:
        """Wait until *port* accepts a connection, or tear down and fail.

        Waiting on the PORT rather than on the process, because a process that started and
        then exited is the failure this catches, and a bare ``sleep`` would report success
        for it. On timeout everything comes down: a half-started pair left running is the
        state that makes the next ``start`` ambiguous.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _START_TIMEOUT_SECONDS
        while loop.time() < deadline:
            try:
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
            except OSError:
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                continue
            writer.close()
            await self._close_quietly(writer)
            del reader
            return
        await self.stop()
        raise VncUnavailable(f"{what} did not start listening on port {port} within {_START_TIMEOUT_SECONDS:.0f}s")

    @staticmethod
    async def _close_quietly(writer: asyncio.StreamWriter) -> None:
        """Await a probe socket's close without letting its teardown fail the start.

        The connection existed, which is the entire question being asked; a reset while
        closing a socket we opened only to prove a listener exists says nothing about the
        listener.
        """
        try:
            await writer.wait_closed()
        except OSError:
            return

    @staticmethod
    def _alive(proc: asyncio.subprocess.Process | None) -> bool:
        """Whether *proc* exists and has not exited."""
        return proc is not None and proc.returncode is None

    async def _terminate(self, proc: asyncio.subprocess.Process | None, *, what: str) -> None:
        """Stop one process, escalating to a kill, and never raising.

        ``SIGTERM`` then ``SIGKILL``: ``x11vnc`` with ``-forever`` does not exit on its own,
        and a stop that leaves it holding the RFB port makes the next start fail with a
        message about a port rather than about what actually happened.
        """
        if not self._alive(proc):
            return
        assert proc is not None  # narrowed by `_alive`
        try:
            proc.terminate()
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except TimeoutError:
            log.warning("hitl: %s ignored SIGTERM; killing it", what)
            try:
                proc.kill()
                await proc.wait()
            except OSError:
                # ProcessLookupError is an OSError subclass, so this covers "already gone".
                log.warning("hitl: %s could not be killed; it may already be gone", what)
        except OSError:
            # Already dead between the liveness check and the signal. Nothing to do, and
            # nothing worth telling a caller who asked for it to be stopped.
            return


# ---------------------------------------------------------------------------
# The HITL session.
#
# One display means one session, so this is a single-session manager rather
# than a pool -- stated rather than implied, because the shape a reader expects
# from "session manager" is a dict of many and that is exactly the thing this
# is not, until a display pool exists to make it true.
# ---------------------------------------------------------------------------


def _constant_time_equal(known: str, supplied: str) -> bool:
    """Constant-time compare that cannot be turned into a 500 by the caller.

    ``secrets.compare_digest`` raises ``TypeError`` on a non-ASCII ``str``, and both operands
    here come straight from a caller: the id is a UTF-8 path segment and the token is a
    header. Sending one non-ASCII character would otherwise turn a refusal into an unhandled
    exception -- a 500 where a 404 belongs, which is both a worse answer and a signal that the
    input reached somewhere it should not have. Comparing UTF-8 bytes keeps the timing
    property and accepts anything a caller can send.
    """
    return secrets.compare_digest(known.encode("utf-8"), supplied.encode("utf-8"))


class SessionUnavailable(RuntimeError):
    """A session was asked for and cannot be given.

    Separate from :class:`VncUnavailable`, which is about the display plumbing. This is about
    the session policy: something is already in progress, or a slot budget is spent.
    """


class SessionNotFound(RuntimeError):
    """The session id or token does not match a live session.

    One exception for both, deliberately: telling a caller "that session exists but your token
    is wrong" confirms the id, and the id is the thing a guesser is trying to learn.
    """


@dataclass
class HitlTab:
    """One target occupying one slot of a session."""

    tab_id: str
    target_id: str
    url: str
    #: nodriver Tab, or ``None`` while this is still a slot reservation whose navigation is in
    #: flight. Typed loosely because nodriver is AGPL-isolated to this container and this
    #: module is deliberately importable without it for the lifecycle tests.
    tab: Any
    #: The isolated browser context, or ``None`` for a reservation not yet navigated.
    context_id: Any
    opened_at: float
    #: Cookies and origin storage read out of the context at completion, or ``None`` when the
    #: tab has not been completed or the export failed. Raw and unencrypted: this container
    #: holds no key. The MIT side seals it before anything persists it.
    exported_state: dict[str, Any] | None = None


@dataclass
class HitlSession:
    """One operator working session against the single display."""

    session_id: str
    token: str
    expires_at: float
    max_slots: int
    tabs: dict[str, HitlTab] = field(default_factory=dict)

    def is_expired(self, now: float) -> bool:
        """Whether the hard TTL has passed."""
        return now >= self.expires_at

    def free_slots(self) -> int:
        """Slots not currently occupied by a tab."""
        return self.max_slots - len(self.tabs)


class SessionManager:
    """Creates, tracks and reaps the one live HITL session.

    Holds the VNC lifecycle rather than being held by it: a session IS the display being
    reachable plus the tabs on it, so tying their lifetimes together in one place is what
    stops a session outliving its VNC or a VNC outliving its session.

    The browser is injected as a zero-argument callable rather than an object, because
    ``main`` binds its browser during lifespan startup and this is constructed at import
    time -- taking the browser eagerly would capture ``None`` forever.
    """

    def __init__(
        self,
        *,
        vnc: VncLifecycle | None = None,
        browser_provider: Callable[[], Any] | None = None,
        max_slots: int = DEFAULT_MAX_SLOTS,
        ttl_seconds: float = DEFAULT_SESSION_TTL_SECONDS,
    ) -> None:
        """
        :param vnc: the display lifecycle a session opens and closes
        :ptype vnc: VncLifecycle | None
        :param browser_provider: returns the live nodriver Browser, or None before startup
        :ptype browser_provider: Callable[[], Any] | None
        :param max_slots: how many targets may occupy the session at once
        :ptype max_slots: int
        :param ttl_seconds: hard lifetime; the reaper closes a session past it
        :ptype ttl_seconds: float
        """
        self._vnc = vnc if vnc is not None else VncLifecycle()
        self._browser_provider = browser_provider
        self._max_slots = max_slots
        self._ttl_seconds = ttl_seconds
        self._session: HitlSession | None = None
        self._reaper: asyncio.Task[None] | None = None
        # Every mutation of _session and its tabs runs under this. Slot accounting is a
        # read-then-write, and two concurrent /tab calls that both read "one slot free" would
        # both take it -- which is how a bounded working set stops being bounded.
        self._lock = asyncio.Lock()

    @property
    def vnc(self) -> VncLifecycle:
        """The display lifecycle this manager owns."""
        return self._vnc

    def current(self) -> HitlSession | None:
        """The tracked session, if any, expired or not.

        Deliberately does NOT filter by expiry: a caller cleaning up needs to see a session
        that is past its TTL, because that is exactly the one still holding the display.
        Use :meth:`owns_display` for the "is this resource spoken for" question.
        """
        return self._session

    def owns_display(self, *, now: float | None = None) -> bool:
        """Whether a session is live enough to still own the display.

        Expiry is the whole point. An expired session is refused by :meth:`authorize`, so no
        caller can tear it down through the session API -- and if this also reported the
        display as spoken for, the bare VNC teardown would refuse too and nothing but the
        reaper could ever release it. Which is precisely the task the TTL check exists because
        it might be dead. Two correct-looking guards, one unreachable resource.
        """
        session = self._session
        if session is None:
            return False
        return not session.is_expired(now if now is not None else time.time())

    def authorize(self, session_id: str, token: str, *, now: float | None = None) -> HitlSession:
        """Resolve a session from an id and token, or refuse.

        Compared with :func:`secrets.compare_digest` rather than ``==``: the comparison is
        against a secret, and the timing of a short-circuiting equality is a side channel that
        leaks a prefix. Cheap to do correctly, so there is no reason not to.

        **Expiry is checked here, not only by the reaper.** The reaper polls, so between a
        session expiring and the next tick there is a window where a TTL that has passed is
        still honoured -- and worse, the reaper is a background task that can die, after which
        nothing else enforces the ceiling at all. A hard TTL that depends on a task staying
        alive is not a hard TTL. Checking on the request path makes the reaper an optimisation
        (it frees the display promptly) rather than the mechanism.

        The sidecar authenticates nobody. This checks that the caller holds a token this
        process minted, which is a different and much weaker claim -- who was allowed to be
        given that token is decided on the MIT side, which is the only side that can evaluate
        a policy.

        :raises SessionNotFound: no live session, wrong id, wrong token, or one past its TTL
        """
        moment = now if now is not None else time.time()
        session = self._session
        if session is None:
            raise SessionNotFound("no session is open")
        if not _constant_time_equal(session.session_id, session_id):
            raise SessionNotFound("no such session")
        if not _constant_time_equal(session.token, token):
            raise SessionNotFound("no such session")
        if session.is_expired(moment):
            # Same error as an unknown session, deliberately: an expired id is still an id
            # that existed, and confirming it would leak the thing a guesser is working
            # towards. The reaper will free the display; this refuses to act on it meanwhile.
            raise SessionNotFound("no such session")
        return session

    async def open(self, *, now: float | None = None) -> HitlSession:
        """Open a session and bring up the display.

        Refuses rather than queues when one is already live. One Xvfb display means one
        operator; queueing here would mean holding a request open for however long the first
        operator takes, which is minutes to hours and is not a thing to do to an HTTP caller.
        The refusal names the expiry so a caller knows when to try again.

        :raises SessionUnavailable: a session is already open
        :raises VncUnavailable: the display could not be brought up
        """
        moment = now if now is not None else time.time()
        async with self._lock:
            live = self._session
            if live is not None and not live.is_expired(moment):
                raise SessionUnavailable(
                    f"a session is already open and expires in {live.expires_at - moment:.0f}s; "
                    f"one display means one operator at a time"
                )
            if live is not None:
                # Expired but not yet reaped. Closing it here rather than waiting for the
                # reaper means an operator arriving exactly then is not told "busy" by a
                # session nobody is using.
                await self._close_locked(live)

            # The display comes up BEFORE the session is recorded. A session whose VNC failed
            # is not a session, and recording it first would leave one that must then be
            # rolled back -- the same two-outcomes discipline `start` itself follows.
            await self._vnc.start()

            session = HitlSession(
                session_id=secrets.token_urlsafe(16),
                # 32 bytes, not a uuid: this is the bearer of the whole session and a uuid4 is
                # 122 bits of a format whose shape invites people to treat it as an
                # identifier rather than a secret.
                token=secrets.token_urlsafe(32),
                expires_at=moment + self._ttl_seconds,
                max_slots=self._max_slots,
            )
            self._session = session
            self._ensure_reaper()
            log.info(
                "hitl: session %s open with %d slots, expiring in %.0fs",
                session.session_id,
                session.max_slots,
                self._ttl_seconds,
            )
            return session

    async def open_tab(
        self,
        session: HitlSession,
        *,
        target_id: str,
        url: str,
        nav_steps: Any = None,
        session_state: dict[str, Any] | None = None,
        now: float | None = None,
    ) -> HitlTab:
        """Bring one target into the session as an isolated tab.

        Isolated per tab, not per session: the acceptance criterion for this chunk is that a
        second target cannot see the first one's cookies, and a shared context would hand a
        walled site the credentials a human just earned somewhere else.

        Occupies a slot from here until :meth:`complete_tab`. Backgrounding a slow target
        still holds its slot -- that is what makes the working set bounded rather than merely
        usually small.

        :raises SessionUnavailable: no slots free
        """
        moment = now if now is not None else time.time()
        tab_id = secrets.token_urlsafe(8)

        # The slot is RESERVED under the lock and the CDP work happens outside it. Holding the
        # lock across the navigation would serialise every tab open behind the slowest one,
        # defer the reaper, and hang a teardown -- which contradicts the model this session is
        # built on, where backgrounding a slow target is normal and still holds its slot. The
        # reservation is what keeps the budget honest while the lock is released: a
        # placeholder occupies the slot, so a concurrent caller sees it taken.
        async with self._lock:
            if session.free_slots() <= 0:
                raise SessionUnavailable(
                    f"all {session.max_slots} slots are occupied; complete a tab before opening another"
                )
            browser = self._browser_provider() if self._browser_provider is not None else None
            if browser is None:
                raise SessionUnavailable("the browser is not running")
            session.tabs[tab_id] = HitlTab(
                tab_id=tab_id,
                target_id=target_id,
                url=url,
                tab=None,
                context_id=None,
                opened_at=moment,
            )

        try:
            tab_obj, context_id = await _open_isolated(browser, url, nav_steps, session_state)
        except BaseException:
            # The reservation must not outlive the attempt it was reserving for, or a target
            # that failed to open silently costs the operator a slot for the whole session.
            #
            # Shielded because acquiring the lock is itself an await, and this handler already
            # runs on the cancellation path: a second cancellation delivered while waiting for
            # it would skip the rollback and burn the slot for the rest of the session, which
            # is the exact outcome the rollback exists to prevent.
            await asyncio.shield(self._release_reservation(session, tab_id))
            raise

        async with self._lock:
            reserved = session.tabs.get(tab_id)
            if reserved is None:
                # The session was torn down or reaped while this navigation was in flight.
                # The context exists and nothing is tracking it, so drop it here rather than
                # leaving it to the browser's lifetime.
                await self._dispose_quietly(browser, context_id, tab_id)
                raise SessionNotFound("the session was closed while this tab was opening")
            reserved.tab = tab_obj
            reserved.context_id = context_id
            log.info(
                "hitl: session %s opened tab %s for target %s (%d/%d slots used)",
                session.session_id,
                tab_id,
                target_id,
                len(session.tabs),
                session.max_slots,
            )
            return reserved

    async def complete_tab(self, session: HitlSession, tab_id: str) -> HitlTab:
        """Close a tab a human has finished with, exporting its state, and free its slot.

        The export happens here because this is the last moment the context exists. What comes
        back is the raw cookie and storage state -- this container never seals anything, having
        no key and no business holding one; the MIT side encrypts it before it is stored.

        :raises SessionNotFound: no such tab in this session
        """
        # Only the pop is under the lock. Popping IS the claim: the tab is out of
        # `session.tabs` before anything is awaited, so no second caller can complete it and
        # no reap can double-drop it. The CDP work that follows is two round-trips to a
        # browser that may be wedged, and holding the lock across them would let one stuck
        # context block `reap()` and `close()` -- which is to say, defeat the hard TTL, the
        # one guarantee that does not depend on guessing an operator's intent. The TTL is
        # worth more than the tidiness of one critical section.
        async with self._lock:
            tab = session.tabs.pop(tab_id, None)
            if tab is None:
                raise SessionNotFound(f"no tab {tab_id} in this session")

        try:
            tab.exported_state = await asyncio.wait_for(self._export_state(tab), timeout=_COMPLETE_TAB_TIMEOUT_SECONDS)
        except TimeoutError:
            # Only a hang can reach here. `_export_state` catches its own failures and returns
            # None -- it promises never to raise into the completion -- so a broad catch here
            # would be guarding against nothing while claiming to guard against everything,
            # and would silently start swallowing real errors the day that promise changed.
            log.warning(
                "hitl: exporting state for tab %s timed out after %.0fs; completing it without any",
                tab_id,
                _COMPLETE_TAB_TIMEOUT_SECONDS,
                extra={"extra_data": {"tab_id": tab_id, "session_id": session.session_id}},
            )
            tab.exported_state = None

        try:
            await asyncio.wait_for(self._drop_tab(tab), timeout=_COMPLETE_TAB_TIMEOUT_SECONDS)
        except TimeoutError:
            # `_drop_tab` already swallows its own errors; only a hang reaches here, and the
            # leaked context is bounded by the browser's lifetime exactly as it is there.
            log.warning("hitl: disposing tab %s's context timed out; leaving it to the browser", tab_id)

        log.info(
            "hitl: session %s completed tab %s (%d/%d slots used)",
            session.session_id,
            tab_id,
            len(session.tabs),
            session.max_slots,
        )
        return tab

    async def close(self, session: HitlSession | None = None) -> None:
        """Tear a session down: drop every context, stop the display.

        Idempotent, and safe with no session open, because teardown is what an error handler
        and a reaper both call.
        """
        async with self._lock:
            target = session if session is not None else self._session
            if target is None:
                # Still stop the display: a VNC running with no session is the leak this
                # method exists to prevent, and reaching it means something already went wrong.
                await self._vnc.stop()
                return
            await self._close_locked(target)

    async def reap(self, *, now: float | None = None) -> bool:
        """Close the session if its hard TTL has passed. Returns whether it did.

        A TTL rather than an idle timeout: an operator who walks away mid-solve leaves a live
        browser holding a target's authenticated session, and "idle" cannot tell that apart
        from "reading carefully". A hard ceiling is the only bound that does not depend on
        guessing intent.
        """
        moment = now if now is not None else time.time()
        async with self._lock:
            session = self._session
            if session is None or not session.is_expired(moment):
                return False
            log.warning(
                "hitl: reaping session %s past its TTL with %d tab(s) still open",
                session.session_id,
                len(session.tabs),
            )
            await self._close_locked(session)
            return True

    async def _close_locked(self, session: HitlSession) -> None:
        """Teardown, with the lock already held."""
        for tab in list(session.tabs.values()):
            await self._drop_tab(tab)
        session.tabs.clear()
        if self._session is session:
            self._session = None
        await self._vnc.stop()
        log.info("hitl: session %s closed", session.session_id)

    async def _drop_tab(self, tab: HitlTab) -> None:
        """Dispose one tab's browser context, never raising into a teardown."""
        browser = self._browser_provider() if self._browser_provider is not None else None
        if browser is None or tab.context_id is None:
            # No context yet means this is a reservation whose navigation is still in flight.
            # Its own error path drops it, and disposing nothing here is correct.
            return
        try:
            await _dispose_context(browser, tab.context_id)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- a context that will not dispose must not stop the other tabs being dropped or the display being stopped; the leak is bounded by the browser's own lifetime. Logged with its traceback below
            log.exception(
                "hitl: could not dispose the browser context for tab %s",
                tab.tab_id,
                extra={"extra_data": {"tab_id": tab.tab_id, "target_id": tab.target_id}},
            )

    async def _export_state(self, tab: HitlTab) -> dict[str, Any] | None:
        """Read the cookies and origin storage a human's work left in *tab*'s context.

        Never raises into the completion. A human has just finished real work; losing the
        export costs the reuse this exists for, but failing the completion would also lose
        them the slot and leave the tab open, which is strictly worse.

        Cookies come from the browser context rather than the page, so ``HttpOnly`` ones are
        included -- those are usually the session cookie, and a cookie-jar export that
        silently omitted exactly the credential worth keeping would look like it worked.
        """
        browser = self._browser_provider() if self._browser_provider is not None else None
        if tab.tab is None or tab.context_id is None or browser is None:
            return None
        try:
            return await _export_context_state(browser, tab.tab, tab.context_id)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- an export that fails costs the reuse, where raising would additionally cost the operator their completed tab and its slot; the human's work is already done either way. Logged with its traceback below
            log.exception(
                "hitl: could not export session state for tab %s; the solve will not be reusable",
                tab.tab_id,
            )
            return None

    async def _release_reservation(self, session: HitlSession, tab_id: str) -> None:
        """Drop a slot reservation whose navigation never landed."""
        async with self._lock:
            session.tabs.pop(tab_id, None)

    async def _dispose_quietly(self, browser: Any, context_id: Any, tab_id: str) -> None:
        """Drop an orphaned context, logging rather than raising."""
        try:
            await _dispose_context(browser, context_id)
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- this runs while already unwinding a torn-down session; the leak is bounded by the browser's lifetime and raising here would replace a clear error with a confusing one. Logged with its traceback below
            log.exception("hitl: could not dispose the orphaned context for tab %s", tab_id)

    def _ensure_reaper(self) -> None:
        """Start the reaper loop if it is not already running."""
        if self._reaper is not None and not self._reaper.done():
            return
        self._reaper = asyncio.create_task(self._reap_loop())

    async def _reap_loop(self) -> None:
        """Poll for an expired session until there is none left to reap.

        Exits when no session is open rather than sleeping forever, so a container with no
        operator has no task. `_ensure_reaper` starts a fresh one with the next session.
        """
        try:
            while self._session is not None:
                await asyncio.sleep(REAPER_INTERVAL_SECONDS)
                await self.reap()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- the reaper is a background supervisor; letting it die silently would leave every later session unbounded, so it logs and the next session restarts it. Logged with its traceback below
            log.exception("hitl: the session reaper stopped unexpectedly")

    async def shutdown(self) -> None:
        """Stop the reaper and close whatever is open. For container teardown."""
        if self._reaper is not None:
            self._reaper.cancel()
            with suppress(asyncio.CancelledError):
                await self._reaper
            self._reaper = None
        await self.close()


async def _open_isolated(
    browser: Any, url: str, nav_steps: Any, session_state: dict[str, Any] | None = None
) -> tuple[Any, Any]:
    """Create an isolated context+tab at *url* and replay *nav_steps* in it.

    When *session_state* is given, it is applied to the context and the page is then reloaded,
    so the navigation that matters carries the cookies. Creating the tab at ``about:blank``
    first would be tidier, but the existing helper navigates on creation and reusing it is
    worth more than saving one reload.

    Indirected through ``main`` at call time rather than imported at module scope: ``main``
    imports this module, so importing it back at the top would be a cycle, and this module is
    deliberately importable without nodriver so the lifecycle tests can run outside the
    container.
    """
    import main  # noqa: PLC0415 -- deliberate late import; see docstring

    tab, context_id = await main._create_isolated_tab(browser, url)
    if session_state:
        await _apply_context_state(browser, context_id, session_state)
        # Storage after the cookies and before the reload: the tab is already on the target
        # origin at this point, which is the only place localStorage can be written, and the
        # reload is what makes the page load with both in place.
        await _apply_origin_storage(tab, session_state)
        await tab.reload()
    if nav_steps:
        await main._execute_nav_steps(tab, nav_steps, _NAV_STEP_TIMEOUT_SECONDS, [])
    return tab, context_id


async def _export_context_state(browser: Any, tab: Any, context_id: Any) -> dict[str, Any]:
    """Read cookies and origin storage out of one isolated browser context.

    Split out so tests can substitute it, and because the CDP surface is the part most likely
    to need revisiting: cookie handling has moved between the Network and Storage domains
    across Chrome versions, and this is the seam to change when it moves again.

    **The storage call goes to the BROWSER connection, not the tab's.** Chrome rejects
    ``browserContextId`` on a page session outright -- "browserContextId is only allowed for
    Browser target" -- which is easy to miss because the tab has the context and looks like
    the natural place to ask. Verified against a running container, not assumed. The
    ``localStorage`` read below is the opposite: it is page script, so it belongs to the tab.

    ``localStorage`` is read through the page rather than a CDP domain, because CDP's
    ``DOMStorage`` requires enabling a domain and enumerating frames for something a single
    expression answers. It is best-effort: a page that blocks script evaluation still yields
    its cookies, which are the part that carries a cleared challenge.
    """
    import nodriver as uc  # noqa: PLC0415 -- deliberate late import; nodriver is AGPL-isolated to this container

    cookies = await browser.send(uc.cdp.storage.get_cookies(browser_context_id=context_id))
    exported: dict[str, Any] = {
        "cookies": [
            {
                "name": c.name,
                "value": c.value,
                "domain": c.domain,
                "path": c.path,
                "expires": getattr(c, "expires", None),
                "httpOnly": getattr(c, "http_only", False),
                "secure": getattr(c, "secure", False),
                "sameSite": getattr(getattr(c, "same_site", None), "value", None),
            }
            for c in (cookies or [])
        ],
        "origins": [],
    }
    try:
        local_storage = await tab.evaluate(
            "JSON.stringify(Object.fromEntries(Object.entries(window.localStorage)))",
            await_promise=False,
        )
    except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- storage is a bonus; a page that refuses script evaluation still yields the cookies that actually carry a cleared challenge, and losing the rest must not lose those
        log.debug("hitl: localStorage was not readable for this context")
        return exported
    if isinstance(local_storage, str) and local_storage not in ("", "{}"):
        exported["origins"].append({"origin": tab.url, "localStorage": local_storage})
    return exported


async def _apply_context_state(browser: Any, context_id: Any, state: dict[str, Any]) -> None:
    """Put a previously exported state's COOKIES back into a fresh isolated context.

    Applied BEFORE the navigation, which is the whole point: a cookie set after the page has
    loaded arrives too late to have been sent with the request that was going to be
    challenged.

    Sent on the BROWSER connection for the same reason as the export: ``browserContextId`` is
    rejected on a page session.

    Cookies only. Origin storage cannot be restored here and has its own function --
    :func:`_apply_origin_storage` -- because ``localStorage`` is origin-scoped and only
    writable while a page from that origin is loaded, which is not true yet at this point in
    the sequence. Splitting them keeps that ordering constraint visible instead of hiding it
    inside a function whose name promises to restore everything.
    """
    import nodriver as uc  # noqa: PLC0415 -- deliberate late import; see _export_context_state

    cookies = state.get("cookies") or []
    if not cookies:
        return
    params = [
        uc.cdp.network.CookieParam(
            name=c["name"],
            value=c["value"],
            domain=c.get("domain"),
            path=c.get("path"),
            secure=c.get("secure"),
            http_only=c.get("httpOnly"),
        )
        for c in cookies
    ]
    await browser.send(uc.cdp.storage.set_cookies(cookies=params, browser_context_id=context_id))


async def _apply_origin_storage(tab: Any, state: dict[str, Any]) -> None:
    """Restore exported ``localStorage`` into a tab already sitting on the right origin.

    Separate from the cookie restore and deliberately later in the sequence: ``localStorage``
    belongs to an origin, and the only way to write it is to evaluate script on a page from
    that origin. There is no browser-level CDP call for it the way there is for cookies.

    The export was written while a human sat on the page, so the entries here are theirs. They
    are applied one key at a time, with the key and value JSON-ENCODED into the expression --
    `json.dumps` on each, never raw interpolation. A value is arbitrary text a human's session
    put there, so it is untrusted input being placed into JavaScript source: the encoding is
    what makes that safe, quoting and escaping it including the `U+2028`/`U+2029` line
    terminators that are legal in JSON strings and fatal in JS source. CDP `evaluate` takes an
    expression rather than bound parameters, so encoding is the mechanism available; anything
    added here must go through `json.dumps` for the same reason.

    Best-effort, and it never raises. Cookies are what carry a cleared challenge; storage is a
    bonus, and losing it must not lose them or fail the fetch they were restored for.
    """
    origins = state.get("origins") or []
    if not origins:
        return
    current = str(getattr(tab, "url", "") or "")
    for origin in origins:
        raw = origin.get("localStorage")
        if not raw:
            continue
        # Only into a page actually on that origin. Writing one origin's storage while sitting
        # on another either fails or, worse, silently lands somewhere it was never exported
        # from.
        recorded = str(origin.get("origin") or "")
        if recorded and current and not _same_origin(recorded, current):
            log.debug("hitl: skipping storage for %s; the tab is on %s", recorded, current)
            continue
        try:
            entries = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            log.info("hitl: exported localStorage for %s did not parse; skipping it", recorded)
            continue
        if not isinstance(entries, dict):
            continue
        for key, value in entries.items():
            try:
                await tab.evaluate(
                    f"window.localStorage.setItem({json.dumps(str(key))}, {json.dumps(str(value))})",
                    await_promise=False,
                )
            except Exception:  # noqa: BLE001 -- prawduct:allow prawduct/broad-except -- one unwritable key (quota, a page that blocks script, a disabled store) must not cost the other keys or the cookies that actually carry the cleared challenge. Logged with its traceback below
                # WARNING with the traceback, not a bare DEBUG line. A systematic failure here
                # -- a page that blocks evaluation, an exhausted quota -- silently half-restores
                # the very thing this module exists to make reusable, and at DEBUG it leaves no
                # evidence at all at default level. Per key rather than once, because which key
                # failed is the diagnostic.
                log.warning(
                    "hitl: could not restore localStorage key %r for %s; the solve is only partly applied",
                    key,
                    recorded,
                    exc_info=True,
                )


def _same_origin(a: str, b: str) -> bool:
    """Whether two urls share scheme+host+port, which is what an origin is."""
    from urllib.parse import urlsplit  # noqa: PLC0415 -- deliberate late import, module stays light

    pa, pb = urlsplit(a), urlsplit(b)
    return (pa.scheme, pa.netloc) == (pb.scheme, pb.netloc)


async def _dispose_context(browser: Any, context_id: Any) -> None:
    """Dispose a browser context. Split out so tests can substitute it."""
    import nodriver as uc  # noqa: PLC0415 -- deliberate late import; see _open_isolated

    await browser.send(uc.cdp.target.dispose_browser_context(context_id))
