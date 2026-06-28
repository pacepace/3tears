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
"""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from threetears.nats import Subjects
from threetears.nats.errors import RequestError
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
    """

    def __init__(
        self,
        nats_client: NatsClient,
        *,
        request_timeout_seconds: float,
        refresh_interval_seconds: float = 300.0,
        initial_retry_interval_seconds: float = 2.0,
    ) -> None:
        self._nc = nats_client
        self._refresh_interval = refresh_interval_seconds
        self._initial_retry_interval = initial_retry_interval_seconds
        self._request_timeout = request_timeout_seconds
        self._jwks: dict[str, Any] = dict(_EMPTY_JWKS)
        self._task: asyncio.Task[None] | None = None
        #: True after the first SUCCESSFUL fetch; gates the fast cold-start retry vs the steady loop
        self._warmed = False

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
            self._jwks = jwks
            self._warmed = True
            log.debug("hub jwks refreshed: keys=%d", len(jwks.get("keys", [])))
        except (RequestError, ValueError) as exc:
            # tolerate a transient Hub blip + overlap-window rotation: keep the last good JWKS,
            # logged. only the never-yet-fetched initial state is empty (fail closed).
            log.warning("hub jwks refresh failed; keeping cached keys: %s", type(exc).__name__)

    async def _refresh_loop(self) -> None:
        """refresh the JWKS until cancelled.

        Until the first SUCCESSFUL fetch the loop retries on the short ``initial_retry_interval`` so
        a verifier whose initial fetch raced the Hub's responder at boot warms within seconds (under
        enforce an empty cache rejects every call). After the first success it settles to the steady
        ``refresh_interval``.
        """
        while True:
            interval = self._refresh_interval if self._warmed else self._initial_retry_interval
            await asyncio.sleep(interval)
            await self.refresh()
