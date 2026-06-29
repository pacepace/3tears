"""Cached provider of the Hub's JWKS for identity-token verifiers (registry proxy + tool pods).

The Hub publishes its public JWKS -- the keys identity tokens are signed under -- via NATS
request/reply on :meth:`Subjects.hub_jwks`. Verifiers call :func:`verify_identity_token` with a
JWKS dict on the hot path, so they need it synchronously. This provider fetches the JWKS once at
:meth:`start` and refreshes it on a background interval; ``__call__`` returns the cached dict with
no IO. Public material only -- nothing here is secret.

Fail-closed: before the first successful fetch the cache is an EMPTY JWKS (``{"keys": []}``), so a
token verifies against no key and is rejected. A later refresh FAILURE keeps the last good JWKS
(tolerating a transient Hub blip and overlap-window key rotation), logged but never cleared to
empty -- only the never-yet-fetched initial state is empty.

Self-healing on a Hub re-key: the steady interval would otherwise reject a VALID token signed under a
freshly-rotated key for up to a full ``refresh_interval``. So the provider also exposes
:meth:`refresh_now` -- a verifier whose verification failed with a kid-not-in-cache miss triggers ONE
immediate refresh and re-verifies, so the re-key heals on the FIRST failed-but-valid token.
:meth:`refresh_now` is debounced (one in-flight refresh) and rate-limited (one Hub fetch per short
window) so a flood of bad tokens cannot stampede the Hub. The background refresh loop is a supervisor
loop: no exception type can end it (only cancellation), so the self-heal source never dies. The
:attr:`is_warmed` flag exposes the first-fetch-succeeded state so a consuming verifier can gate its
k8s readiness on the cache being able to verify anything at all.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from threetears.nats import Subjects
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.nats import NatsClient

__all__ = ["CachedHubJwksProvider"]

log = get_logger(__name__)

_EMPTY_JWKS: dict[str, Any] = {"keys": []}


class CachedHubJwksProvider:
    """fetches + caches the Hub JWKS so a verifier's sync ``jwks_provider()`` returns it with no IO.

    :param nats_client: connected canonical :class:`threetears.nats.NatsClient` (the fetch transport)
    :ptype nats_client: NatsClient
    :param request_timeout_seconds: per-fetch request/reply timeout; sourced from the caller's
        config layer (no default here -- timeouts are configuration, not a buried magic number)
    :ptype request_timeout_seconds: float
    :param refresh_interval_seconds: seconds between background refreshes once the cache is warm
    :ptype refresh_interval_seconds: float
    :param initial_retry_interval_seconds: seconds between retries BEFORE the first successful fetch.
        Short so a verifier that started before the Hub's JWKS responder was up warms within seconds
        rather than waiting a full ``refresh_interval`` -- under enforce an empty cache rejects every
        call, so a slow cold-start warm is a multi-minute reject window.
    :ptype initial_retry_interval_seconds: float
    :param reactive_min_interval_seconds: the rate-limit window for :meth:`refresh_now`. After a
        reactive refresh runs, further reactive triggers inside this window are suppressed so a flood
        of bad (kid-miss) tokens cannot stampede the Hub: at most ONE reactive Hub fetch per window,
        on top of the at-most-one-in-flight collapse a concurrency lock provides. Short so a genuine
        Hub re-key still self-heals on the first failed-but-valid token within seconds.
    :ptype reactive_min_interval_seconds: float
    """

    def __init__(
        self,
        nats_client: NatsClient,
        *,
        request_timeout_seconds: float,
        refresh_interval_seconds: float = 300.0,
        initial_retry_interval_seconds: float = 2.0,
        reactive_min_interval_seconds: float = 5.0,
    ) -> None:
        self._nc = nats_client
        self._refresh_interval = refresh_interval_seconds
        self._initial_retry_interval = initial_retry_interval_seconds
        self._request_timeout = request_timeout_seconds
        self._jwks: dict[str, Any] = dict(_EMPTY_JWKS)
        self._task: asyncio.Task[None] | None = None
        #: True after the first SUCCESSFUL fetch; gates the fast cold-start retry vs the steady loop
        self._warmed = False
        #: collapses concurrent reactive triggers onto a single in-flight refresh (no Hub stampede)
        self._reactive_lock = asyncio.Lock()
        self._reactive_min_interval = reactive_min_interval_seconds
        #: monotonic timestamp of the last reactive refresh; rate-limits :meth:`refresh_now`
        self._last_reactive_refresh: float | None = None

    @property
    def is_warmed(self) -> bool:
        """whether the FIRST successful JWKS fetch has populated the cache.

        Before this flips ``True`` the cache is empty and every token verifies against no key (fail
        closed), so a consuming verifier gates its k8s readiness on it: report NOT-READY until warmed
        so the verifier does not accept calls it would immediately fail closed.

        :return: True once at least one fetch has succeeded
        :rtype: bool
        """
        return self._warmed

    def __call__(self) -> dict[str, Any]:
        """return the cached JWKS (sync, no IO); EMPTY until the first successful fetch.

        :return: the current cached JWKS document
        :rtype: dict[str, Any]
        """
        return self._jwks

    async def start(self) -> None:
        """best-effort initial fetch, then a background refresh loop.

        A failed initial fetch does NOT raise -- the verifier must still come up; the cache stays
        empty (fail closed) and the refresh loop retries.

        :return: nothing
        :rtype: None
        """
        await self.refresh()
        self._task = asyncio.create_task(self._refresh_loop())

    async def stop(self) -> None:
        """cancel the background refresh loop.

        :return: nothing
        :rtype: None
        """
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def refresh(self) -> None:
        """fetch the JWKS once; on failure keep the last good cache (never clear to empty).

        Called by :meth:`start` and the background loop; safe to call directly to force a refresh.

        :return: nothing
        :rtype: None
        """
        try:
            response = await self._nc.request_raw(
                subject=Subjects.hub_jwks(),
                payload=b"",
                timeout=timedelta(seconds=self._request_timeout),
            )
            jwks = json.loads(response)
            if not isinstance(jwks, dict) or "keys" not in jwks:
                raise ValueError("hub jwks reply is not a JWKS document")
            if not jwks.get("keys"):
                # a shape-valid but EMPTY keyset can verify NOTHING (every token rejects fail-closed),
                # so treat it like a transient blip: keep the last-good cache (never clear to empty)
                # and stay UN-warmed, so the readiness gate keeps holding traffic and the fast
                # cold-start retry continues until the Hub serves a real key. a keyless Hub JWKS is a
                # misconfiguration, not a reason to report ready.
                log.warning("hub jwks reply has no keys; keeping cached keys, staying un-warmed")
                return
            self._jwks = jwks
            self._warmed = True
            log.debug("hub jwks refreshed: keys=%d", len(jwks.get("keys", [])))
        except Exception as exc:
            # tolerate a transient Hub blip + overlap-window rotation: keep the last good JWKS,
            # logged. only the never-yet-fetched initial state is empty (fail closed). the catch is
            # broadened from (RequestError, ValueError) to Exception on purpose -- once Hub-backed a
            # fetch can raise far beyond those (ConnectionError, TimeoutError, a transport-level
            # error), and NONE of them may end the background refresh loop or escape a reactive
            # trigger. CancelledError is a BaseException (not Exception), so cooperative cancellation
            # still propagates and :meth:`stop` cancels cleanly.
            log.warning(
                "hub jwks refresh failed; keeping cached keys",
                extra={"extra_data": {"reason": type(exc).__name__, "detail": str(exc)}},
            )

    async def refresh_now(self) -> bool:
        """trigger at most ONE immediate JWKS refresh, debounced + rate-limited; for reactive re-key.

        The reactive companion to the steady loop: when token verification fails because the cached
        JWKS holds no key for the token's ``kid`` (a Hub re-key, or a stale cache after a Hub pod
        move), the consuming verifier calls this so a VALID-but-unverifiable token self-heals on the
        FIRST such failure instead of waiting up to a full ``refresh_interval`` for the steady tick.

        Two guards keep a flood of bad tokens from stampeding the Hub:

        - **collapse** -- a concurrency lock serializes triggers, so concurrent callers ride a SINGLE
          in-flight refresh rather than each firing their own Hub request;
        - **rate-limit** -- after a reactive refresh runs, triggers within ``reactive_min_interval``
          are suppressed, so a sustained flood of unverifiable tokens drives at most one Hub fetch
          per window. A genuinely-forged kid then costs one fetch per window, not one per token.

        :return: ``True`` if a refresh actually ran in this call, ``False`` if suppressed by the
            rate-limit (a refresh ran recently; the consumer should still re-verify against the cache
            either way -- a concurrent/recent refresh may already have brought the key)
        :rtype: bool
        """
        # fast-path rate-limit (no lock): a refresh ran very recently, so the cache is already as
        # fresh as a reactive fetch would make it -- skip without contending for the lock.
        last = self._last_reactive_refresh
        if last is not None and (time.monotonic() - last) < self._reactive_min_interval:
            return False
        async with self._reactive_lock:
            # re-check under the lock: a concurrent trigger may have refreshed while we waited for it.
            # collapse onto that refresh's result rather than firing a second, redundant Hub request.
            last = self._last_reactive_refresh
            if last is not None and (time.monotonic() - last) < self._reactive_min_interval:
                return False
            await self.refresh()
            self._last_reactive_refresh = time.monotonic()
            return True

    async def _refresh_loop(self) -> None:
        """refresh the JWKS until cancelled.

        Until the first SUCCESSFUL fetch the loop retries on the short ``initial_retry_interval`` so
        a verifier whose initial fetch raced the Hub's responder at boot warms within seconds (under
        enforce an empty cache rejects every call). After the first success it settles to the steady
        ``refresh_interval``.

        This is a supervisor loop: it must be UNKILLABLE, because a Hub re-key only self-heals while
        it keeps running. :meth:`refresh` already keeps-last-good + logs on the failures it models;
        this belt-and-braces guard wraps the loop body so NO exception type (including a future
        programming error in the body) can end the loop. Only ``asyncio.CancelledError`` (a
        BaseException, not caught here) ends it, so :meth:`stop` still cancels cleanly.
        """
        while True:
            interval = self._refresh_interval if self._warmed else self._initial_retry_interval
            await asyncio.sleep(interval)
            try:
                await self.refresh()
            except Exception as exc:
                log.warning(
                    "hub jwks refresh loop iteration failed; loop continues",
                    extra={"extra_data": {"reason": type(exc).__name__, "detail": str(exc)}},
                )
