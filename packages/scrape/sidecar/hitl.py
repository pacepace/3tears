"""Make the sidecar's existing X display reachable by a human, on demand.

The container already runs a real headful Chromium against Xvfb -- what it has never had is
any way for a person to SEE that display, let alone drive it. This module is that path, and
nothing more: it starts ``x11vnc`` against the display and ``websockify`` in front of it,
serving noVNC's static client, and it stops both again. There are no sessions here, no tab
isolation, no tokens and no authorization; those are later work and deliberately absent
rather than half-present.

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
import logging
import os
import shutil
from dataclasses import dataclass

log = logging.getLogger("nodriver_sidecar.hitl")

#: Where Debian's ``novnc`` package puts the client. Verified by installing the package in
#: the real base image rather than recalled: the tree holds ``vnc.html``, ``vnc_lite.html``
#: and ``vnc_auto.html`` and NO ``index.html``, which is why :attr:`VncSession.path` names a
#: page explicitly. Serving the directory alone would 404 at ``/``.
NOVNC_ROOT = os.environ.get("NOVNC_ROOT", "/usr/share/novnc")

#: The noVNC page to hand a caller. ``vnc_lite.html`` rather than ``vnc.html``: the full
#: client opens a settings sidebar and expects the user to connect manually, where the lite
#: page connects to the websockify endpoint that served it. For a human summoned to clear one
#: challenge, "it is already connected" is the whole difference.
NOVNC_PAGE = "vnc_lite.html"

#: RFB port ``x11vnc`` listens on, loopback only. Not published by the container and not
#: configurable per session, because there is exactly one display.
_RFB_PORT = 5900

#: How long to wait for each process to start listening before calling it a failure. Both are
#: local process spawns, so this is generous; the cost of being wrong is a session that
#: reports success and shows a black rectangle.
_START_TIMEOUT_SECONDS = 10.0

#: Poll interval while waiting for a port to accept a connection.
_POLL_INTERVAL_SECONDS = 0.1


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
        # Checked before anything is spawned: a missing client tree is a certain failure, and
        # finding out before there are processes to clean up is strictly better.
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
        return f"/{NOVNC_PAGE}?path=websockify&resize=scale"

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
