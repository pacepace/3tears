"""slack channel adapter using slack-bolt socket mode.

bridges slack workspace messages to platform via unified channel protocol.
uses slack-bolt AsyncApp with socket mode for real-time message handling
without requiring public HTTP endpoints.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.errors import SlackApiError

from threetears.channels.formatting import (
    markdown_to_slack_blocks,
    plain_text_fallback,
)
from threetears.channels.protocol import (
    Attachment,
    ChannelMessage,
    ChannelResponse,
    ChannelRouter,
)
from threetears.observe import contain, get_logger

__all__ = [
    "SlackAdapter",
]

log = get_logger(__name__)

# cache TTL for the per-user Slack profile (timezone, BCP 47 locale tag,
# display name, email). Slack ``users.info`` is Tier 4 (~100 req/min); a
# 5-minute TTL keeps fetches rare without serving stale tz to a user who
# just travelled (Slack updates ``user.tz`` from the user's profile, which
# a user can change mid-session).
_USER_PROFILE_TTL_SECONDS = 300.0
#: how long a FAILED users.info lookup is remembered, by how many times in a row it has failed for that
#: user: 15 s, then 60 s, then 300 s. a burst of messages from one user does not hammer a struggling
#: endpoint, and the next message after a blip retries instead of carrying an unknown sender for the
#: whole :data:`_USER_PROFILE_TTL_SECONDS`.
_USER_PROFILE_FAILURE_BACKOFF_SECONDS = (15.0, 60.0, 300.0)
#: the longest a Slack ``Retry-After`` is honoured for one user.
_USER_PROFILE_RETRY_AFTER_CAP_SECONDS = 3600.0
#: Slack error codes that no retry will cure: the user does not exist, the app lacks the scope, the token
#: is bad, the account is deactivated. asking again inside the TTL only spends the Tier 4 budget.
_PERMANENT_USERS_INFO_ERRORS = frozenset({"user_not_found", "missing_scope", "not_authed", "account_inactive"})


@dataclass(slots=True)
class _ProfileCacheEntry:
    """one user's cached lookup.

    :param profile: what to serve: the last profile Slack answered if there ever was one, else a
        profile marked ``lookup_failed``
    :ptype profile: _SlackUserProfile
    :param fresh_until: monotonic time before which the entry is served without asking Slack again
    :ptype fresh_until: float
    :param failures: consecutive failed lookups, driving the backoff
    :ptype failures: int
    :param good: the last profile Slack actually answered, kept to serve when a refresh fails
    :ptype good: _SlackUserProfile | None
    """

    profile: _SlackUserProfile
    fresh_until: float
    failures: int = 0
    good: _SlackUserProfile | None = None


def _retry_after_seconds(exc: SlackApiError) -> float | None:
    """the ``Retry-After`` a rate-limited Slack response carried, in seconds.

    :param exc: the failure
    :ptype exc: SlackApiError
    :return: the wait, or ``None`` when the response named none
    :rtype: float | None
    """
    headers = getattr(exc.response, "headers", None) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    try:
        return float(raw) if raw is not None else None
    except TypeError, ValueError:
        return None


# hard bound on the per-user profile cache. entries are keyed by Slack user
# id and were previously only overwritten, never evicted -- so a long-lived
# adapter in a large workspace accumulated one entry per person who had ever
# spoken and never released any of them. an expired entry is not dropped by
# the TTL check either (that check only decides whether to SERVE it), so the
# TTL bounds staleness and this bounds size. LRU: on overflow the least
# recently used entry goes and is simply re-fetched if that user speaks again.
_USER_PROFILE_CACHE_MAX_ENTRIES = 2048


# message subtypes that are not fresh user content: bot echoes and the
# edit / delete mutation events (``message_changed`` / ``message_deleted``)
# that replay a prior message's ``ts``. routing them re-answers content the
# user has since changed or removed.
_IGNORED_MESSAGE_SUBTYPES = frozenset(
    {"bot_message", "message_changed", "message_deleted"},
)


# grace window (seconds) applied to the startup-replay guard. a message
# whose ``ts`` predates this adapter's start by more than the grace is a
# Slack REDELIVERY of an event that arrived while the adapter was down /
# restarting (never acked, so Slack retried it minutes later). answering it
# re-surfaces a stale — possibly since-deleted — question. bolt drops the
# Socket Mode ``retry_attempt`` marker before the listener, so the age of
# the message ``ts`` relative to process start is the available signal. the
# grace absorbs clock skew and messages genuinely in flight at startup.
_STARTUP_REPLAY_GRACE_SECONDS = 30.0


@dataclass(frozen=True, slots=True)
class _SlackUserProfile:
    """the subset of a Slack ``users.info`` user object this adapter carries.

    every field is optional because Slack documents that "data that has not
    been supplied may not be present at all, may be null or may contain the
    empty string" -- so each is normalized to None rather than ``""``.

    :param timezone: IANA timezone name from ``user.tz``
    :ptype timezone: str | None
    :param locale: BCP 47 locale tag from ``user.locale``
    :ptype locale: str | None
    :param display_name: the member's chosen display name, falling back to
        their real name
    :ptype display_name: str | None
    :param email: ``user.profile.email``. requires the ``users:read.email``
        scope IN ADDITION to ``users:read``; without it Slack omits the field
        silently rather than erroring, so None here means "no scope, no
        address on file, or a lookup failure" and the three are
        indistinguishable from the response alone.
    :ptype email: str | None
    :param lookup_failed: ``True`` when ``users.info`` itself failed, so every field is UNKNOWN
        rather than absent; ``False`` for a profile Slack answered (however sparse)
    :ptype lookup_failed: bool
    """

    timezone: str | None = None
    locale: str | None = None
    display_name: str | None = None
    email: str | None = None
    lookup_failed: bool = False


def _non_empty_str(value: Any) -> str | None:
    """``value`` when it is a non-empty string, else None.

    Slack renders absent profile data as a missing key, ``null`` OR the empty
    string, and consumers want one representation of "absent" rather than
    three.

    :param value: raw value read off a Slack API response
    :ptype value: Any
    :return: the string, or None when absent/blank/not a string
    :rtype: str | None
    """
    return value if isinstance(value, str) and value else None


def _profile_from_user_object(user_obj: dict[str, Any]) -> _SlackUserProfile:
    """map a Slack ``users.info`` ``user`` object onto :class:`_SlackUserProfile`.

    display name prefers ``profile.display_name`` (what the member chose to be
    called) over ``profile.real_name``, then the top-level ``real_name`` /
    ``name`` -- a workspace that never set display names would otherwise yield
    no name at all.

    :param user_obj: the ``user`` object from a ``users.info`` response
    :ptype user_obj: dict[str, Any]
    :return: the normalized profile subset
    :rtype: _SlackUserProfile
    """
    raw_profile = user_obj.get("profile")
    profile_obj: dict[str, Any] = raw_profile if isinstance(raw_profile, dict) else {}
    display_name = (
        _non_empty_str(profile_obj.get("display_name"))
        or _non_empty_str(profile_obj.get("real_name"))
        or _non_empty_str(user_obj.get("real_name"))
        or _non_empty_str(user_obj.get("name"))
    )
    return _SlackUserProfile(
        timezone=_non_empty_str(user_obj.get("tz")),
        locale=_non_empty_str(user_obj.get("locale")),
        display_name=display_name,
        email=_non_empty_str(profile_obj.get("email")),
    )


class SlackAdapter:
    """channel adapter bridging slack to platform via socket mode.

    receives inbound slack messages, normalizes them to ChannelMessage,
    routes through ChannelRouter, and delivers responses back to slack.

    :param bot_token: slack bot user OAuth token (xoxb-...)
    :ptype bot_token: str
    :param app_token: slack app-level token for socket mode (xapp-...)
    :ptype app_token: str
    :param router: channel router for processing inbound messages
    :ptype router: ChannelRouter
    :param config: optional adapter configuration overrides
    :ptype config: dict[str, Any] | None
    :param user_profile_cache_max_entries: most user profiles held before the least recently
        used is evicted
    :ptype user_profile_cache_max_entries: int

    host contract: when Slack's ``users.info`` fails and no earlier answer exists for the sender,
    the routed :class:`ChannelMessage` carries ``metadata["profile_lookup_failed"] = True`` and
    ``sender_name`` / ``sender_email`` / ``user_timezone`` / ``user_locale`` are ``None`` because they
    are UNKNOWN, not absent. a host that reconciles identity by address, or stamps a timezone, must
    treat that flag as "do not conclude anything from the missing values". the key is absent on a
    profile Slack answered, however sparse, and on one served stale from an earlier answer.
    """

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        router: ChannelRouter,
        config: dict[str, Any] | None = None,
        *,
        user_profile_cache_max_entries: int = _USER_PROFILE_CACHE_MAX_ENTRIES,
    ) -> None:
        """initialize slack adapter with tokens, router, and optional config.

        :param bot_token: slack bot user OAuth token (xoxb-...)
        :ptype bot_token: str
        :param app_token: slack app-level token for socket mode (xapp-...)
        :ptype app_token: str
        :param router: channel router for processing inbound messages
        :ptype router: ChannelRouter
        :param config: optional adapter configuration overrides
        :ptype config: dict[str, Any] | None
        :param user_profile_cache_max_entries: most user profiles held before the least
            recently used is evicted; the production bound is
            :data:`_USER_PROFILE_CACHE_MAX_ENTRIES`
        :ptype user_profile_cache_max_entries: int
        """
        self._app = AsyncApp(token=bot_token)
        self.app_token = app_token
        self.router = router
        self.config: dict[str, Any] = config if config is not None else {}
        self._handler: AsyncSocketModeHandler | None = None
        # wall-clock time this adapter began listening, set once in
        # :meth:`start`. events whose ``ts`` predates it (minus the grace)
        # are Slack replays of messages that arrived while the adapter was
        # down; :meth:`_is_startup_replay` drops them. ``None`` before start
        # disables the guard (so unit tests that never call ``start`` route
        # normally).
        self._started_at: float | None = None
        # per-user profile cache: maps slack user_id -> (cached_at_monotonic, profile).
        # an entry stops being served once ``cached_at_monotonic`` is older than
        # :data:`_USER_PROFILE_TTL_SECONDS`; the OrderedDict is kept in
        # least-recently-used order and capped at
        # ``user_profile_cache_max_entries`` so the map cannot grow without
        # bound. populated lazily on first message from each user via
        # :meth:`_resolve_user_profile`.
        self._user_profile_cache: OrderedDict[str, _ProfileCacheEntry] = OrderedDict()
        self._user_profile_cache_max_entries = user_profile_cache_max_entries

        self._app.event("message")(self.handle_message_event)

    async def _resolve_user_profile(
        self,
        user_id: str,
    ) -> _SlackUserProfile:
        """resolve the cached :class:`_SlackUserProfile` for a slack user.

        consults the per-instance LRU/TTL cache first; on miss, calls
        ``users.info(user=user_id, include_locale=True)`` and caches
        the result. a FAILED lookup serves the last profile Slack answered for
        that user, or a profile marked ``lookup_failed`` when there never was
        one. the failure is remembered per user for 15, then 60, then 300
        seconds; a rate-limit's ``Retry-After`` is honoured instead; and a
        permanent error (``user_not_found``, ``missing_scope``, ``not_authed``,
        ``account_inactive``) is remembered for the full TTL, since asking again
        cannot change the answer. callers consume the result as the ``ChannelMessage``
        ``user_timezone`` / ``user_locale`` / ``sender_name`` /
        ``sender_email`` fields.

        ONE lookup serves all four. it is deliberately not split per field:
        ``users.info`` is Tier 4 (~100 req/min) and a second call for the
        address would double the budget for a value the first response
        already contains.

        :param user_id: slack platform user identifier (``U…`` form)
        :ptype user_id: str
        :return: the profile subset; every field may be None when slack
            does not surface a value for this user
        :rtype: _SlackUserProfile
        """
        entry = self._user_profile_cache.get(user_id)
        now = time.monotonic()
        if entry is not None and now < entry.fresh_until:
            self._user_profile_cache.move_to_end(user_id)
            return entry.profile
        previous_good = entry.good if entry is not None else None
        failures = entry.failures if entry is not None else 0
        try:
            response = await self._app.client.users_info(
                user=user_id,
                include_locale=True,
            )
            user_obj: dict[str, Any] = response.get("user", {}) if response else {}
            profile = _profile_from_user_object(user_obj)
            if profile.email is None:
                # named explicitly because the usual cause is a MISSING SCOPE
                # and slack reports it by silently omitting the field -- no
                # error, no warning, just an address that never arrives. a
                # host relying on it (to reconcile a chat participant with a
                # known identity) would otherwise see a feature that quietly
                # never fires. once per user per TTL, not per message.
                log.info(
                    "slack users.info returned no email for this user; "
                    "the app needs the users:read.email scope alongside users:read "
                    "(a re-install is required after adding it)",
                    extra={"extra_data": {"user_id": user_id}},
                )
            new_entry = _ProfileCacheEntry(profile, now + _USER_PROFILE_TTL_SECONDS, 0, good=profile)
        except Exception as exc:
            contain(
                exc,
                log=log,
                what="slack users.info lookup failed; serving the last known profile, or an unknown one",
                extra={"user_id": user_id, "error": str(exc)},
            )
            failures += 1
            code = exc.response.get("error") if isinstance(exc, SlackApiError) and exc.response is not None else None
            if code in _PERMANENT_USERS_INFO_ERRORS:
                wait = _USER_PROFILE_TTL_SECONDS
            else:
                retry_after = _retry_after_seconds(exc) if isinstance(exc, SlackApiError) else None
                backoff = _USER_PROFILE_FAILURE_BACKOFF_SECONDS
                wait = (
                    min(max(retry_after, 1.0), _USER_PROFILE_RETRY_AFTER_CAP_SECONDS)
                    if retry_after is not None
                    else backoff[min(failures, len(backoff)) - 1]
                )
            # stale-on-error: a profile Slack once answered beats an unknown one. a PERMANENT error says the
            # user (or our access to them) is gone, so there the old profile is not served.
            served = previous_good if previous_good is not None and code not in _PERMANENT_USERS_INFO_ERRORS else None
            new_entry = _ProfileCacheEntry(
                served if served is not None else _SlackUserProfile(lookup_failed=True),
                now + wait,
                failures,
                good=previous_good,
            )
        self._user_profile_cache[user_id] = new_entry
        self._user_profile_cache.move_to_end(user_id)
        while len(self._user_profile_cache) > self._user_profile_cache_max_entries:
            self._user_profile_cache.popitem(last=False)
        return new_entry.profile

    async def start(self) -> None:
        """start socket mode connection to slack.

        creates AsyncSocketModeHandler and initiates websocket connection.
        records the process-listen baseline used by the startup-replay
        guard (only on first start, so a routing test that calls start
        twice does not shift the baseline forward).
        """
        if self._started_at is None:
            self._started_at = time.time()
        self._handler = AsyncSocketModeHandler(self._app, self.app_token)
        await self._handler.start_async()

    async def stop(self) -> None:
        """stop socket mode connection to slack.

        closes websocket connection if handler exists. safe to call
        before start() or after already stopped.
        """
        if self._handler is not None:
            await self._handler.close_async()

    async def post_message(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> None:
        """post a message to a channel/thread out-of-band (no inbound event).

        used by durable channel-answer delivery: a finished agent answer can
        arrive long after the originating Slack event has closed, so it is
        posted directly through the web client (``chat_postMessage``) on the
        adapter's bot token rather than the event-bound ``say`` closure. the
        markdown body is rendered into native Slack blocks, as in
        :func:`_send_response`.

        :param channel: slack channel id to post into
        :ptype channel: str
        :param text: message text in markdown
        :ptype text: str
        :param thread_ts: thread to reply in; ``None`` posts top-level
        :ptype thread_ts: str | None
        :return: nothing
        :rtype: None
        """
        # always render the agent's markdown into native Slack blocks (tables,
        # headers, mrkdwn) -- Slack does not render GitHub markdown, so posting
        # raw text shows ``**bold**`` / ``| tables |`` literally. ``text`` is the
        # notification/fallback (markdown stripped); ``blocks`` is the rendered
        # body.
        blocks = markdown_to_slack_blocks(text)
        if blocks:
            kwargs: dict[str, Any] = {
                "channel": channel,
                "text": plain_text_fallback(text),
                "blocks": blocks,
            }
        else:
            kwargs = {"channel": channel, "text": text}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        await self._app.client.chat_postMessage(**kwargs)

    def _is_startup_replay(self, event: dict[str, Any]) -> bool:
        """whether an inbound event is a Slack replay from before startup.

        Slack redelivers an event that was never acknowledged (the adapter
        was down or restarting when the message arrived), retrying minutes
        later once the socket reconnects. answering such a replay
        re-surfaces a stale — and possibly since-deleted — question. bolt
        does not expose the Socket Mode ``retry_attempt`` marker to the
        listener, so the guard compares the message ``ts`` against this
        adapter's start time: a message posted before the adapter began
        listening (beyond :data:`_STARTUP_REPLAY_GRACE_SECONDS`) is a
        replay. an unset start baseline (adapter never started) or an
        unparseable ``ts`` disables the guard rather than dropping.

        :param event: raw slack event payload
        :ptype event: dict[str, Any]
        :return: True when the event is a pre-startup replay to drop
        :rtype: bool
        """
        result = False
        if self._started_at is not None:
            raw_ts = event.get("ts")
            posted_at: float | None = None
            if isinstance(raw_ts, str):
                try:
                    posted_at = float(raw_ts)
                except ValueError:
                    posted_at = None
            if posted_at is not None and posted_at < self._started_at - _STARTUP_REPLAY_GRACE_SECONDS:
                log.info(
                    "dropping slack pre-startup replay (message predates adapter start)",
                    extra={
                        "extra_data": {
                            "message_ts": raw_ts,
                            "channel": event.get("channel"),
                            "age_seconds": round(self._started_at - posted_at, 1),
                        },
                    },
                )
                result = True
        return result

    async def handle_message_event(
        self,
        event: dict[str, Any],
        say: Any,
    ) -> None:
        """public slack-event handler for inbound messages.

        registered as the slack-bolt ``message`` event callback in
        :meth:`__init__`. tests exercise this surface directly; the
        name + ``(event, say)`` shape are part of the stability
        contract.

        filters bot messages, normalizes event to ChannelMessage,
        routes through router, and delivers response back to slack.

        :param event: raw slack event payload
        :ptype event: dict[str, Any]
        :param say: slack-bolt say function for replying
        :ptype say: Any
        """
        if event.get("bot_id") or event.get("subtype") in _IGNORED_MESSAGE_SUBTYPES:
            return

        if self._is_startup_replay(event):
            return

        thread_ts = event.get("thread_ts")
        channel_type_raw = event.get("channel_type", "")
        is_dm = channel_type_raw == "im"

        attachments = _map_slack_files(event.get("files", []))

        # fields consumed by standard ChannelMessage attributes
        consumed_keys = {
            "type",
            "user",
            "text",
            "channel",
            "thread_ts",
            "team",
            "files",
        }
        metadata: dict[str, Any] = {k: v for k, v in event.items() if k not in consumed_keys}

        # per-user profile from slack -- looked up via
        # ``users.info(include_locale=True)`` and cached for
        # :data:`_USER_PROFILE_TTL_SECONDS`. populated even on cache
        # miss so consumers (agent runtime stamping
        # ``CallContext.user_timezone``) read uniformly. a message event
        # carries only the sender's ID, so name and address both come from
        # here -- there is nothing on the event to fall back to.
        sender_id = event.get("user", "")
        profile = _SlackUserProfile()
        if sender_id:
            profile = await self._resolve_user_profile(sender_id)

        if profile.lookup_failed:
            # the sender's name, address, zone and locale are unknown, not absent
            metadata["profile_lookup_failed"] = True

        channel_message = ChannelMessage(
            channel_type="slack",
            sender_id=sender_id,
            sender_name=profile.display_name,
            sender_email=profile.email,
            # slack returning an address at all IS the workspace asserting it:
            # ``profile.email`` is workspace/SSO-administered, not a value the
            # sender puts on the wire. deliberately NOT read from the
            # ``is_email_confirmed`` field some responses carry -- that field is
            # absent from slack's documented user object, and an undocumented
            # field is not a thing to gate identity on. the protocol defines this
            # flag as the PLATFORM's assertion rather than proof of mailbox
            # control, and the host applies its own condition on top.
            sender_email_verified=profile.email is not None,
            content=event.get("text", ""),
            channel_id=event.get("channel", ""),
            conversation_id=thread_ts if thread_ts else event.get("ts"),
            workspace_id=event.get("team"),
            # this adapter is bound to ONE slack app / bot; stamp that bot's id so
            # a router serving several bots in the same workspace resolves the
            # target agent unambiguously. the host passes it via config; None (no
            # config) means single-bot and nothing to disambiguate.
            receiving_bot_id=self.config.get("bot_user_id") or None,
            attachments=attachments,
            reply_to_id=thread_ts,
            metadata=metadata,
            timestamp=datetime.now(UTC),
            user_timezone=profile.timezone,
            user_locale=profile.locale,
        )

        response = await self.router.route_inbound(channel_message)

        if response is None:
            return

        await _send_response(
            say=say,
            client=self._app.client,
            response=response,
            event=event,
            thread_ts=thread_ts,
            is_dm=is_dm,
        )


def _map_slack_files(files: list[dict[str, Any]]) -> list[Attachment]:
    """map slack file objects to Attachment dataclass instances.

    :param files: list of slack file metadata dicts
    :ptype files: list[dict[str, Any]]
    :return: list of Attachment dataclass instances
    :rtype: list[Attachment]
    """
    result: list[Attachment] = []
    for f in files:
        result.append(
            Attachment(
                filename=f.get("name", ""),
                content=b"",
                content_type=f.get("mimetype", "application/octet-stream"),
                description=f.get("title"),
            )
        )
    return result


async def _send_response(
    say: Any,
    client: Any,
    response: ChannelResponse,
    event: dict[str, Any],
    thread_ts: str | None,
    is_dm: bool,
) -> None:
    """deliver outbound response back to slack.

    sends text reply and uploads any file attachments. threading behavior
    depends on whether inbound was threaded, DM, or top-level channel message.

    :param say: slack-bolt say function for replying
    :ptype say: Any
    :param client: slack async web client for file uploads
    :ptype client: Any
    :param response: outbound response from router
    :ptype response: ChannelResponse
    :param event: original slack event for context
    :ptype event: dict[str, Any]
    :param thread_ts: thread timestamp if message was in thread
    :ptype thread_ts: str | None
    :param is_dm: whether message is direct message
    :ptype is_dm: bool
    """
    # render the agent's markdown into native Slack blocks (tables, headers,
    # mrkdwn); ``text`` is the markdown-stripped notification fallback.
    blocks = markdown_to_slack_blocks(response.content)
    fallback_text = plain_text_fallback(response.content)

    if is_dm and not thread_ts:
        if blocks:
            await say(text=fallback_text, blocks=blocks)
        else:
            await say(text=response.content)
    else:
        reply_thread_ts = thread_ts if thread_ts else event.get("ts", "")
        if blocks:
            await say(text=fallback_text, blocks=blocks, thread_ts=reply_thread_ts)
        else:
            await say(text=response.content, thread_ts=reply_thread_ts)

    channel_id = event.get("channel", "")
    effective_thread_ts = thread_ts if thread_ts else event.get("ts", "")

    for attachment in response.attachments:
        await client.files_upload_v2(
            channel=channel_id,
            filename=attachment.filename,
            content=attachment.content,
            title=attachment.filename,
            thread_ts=effective_thread_ts,
        )
