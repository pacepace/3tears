# 3tears-channels

A unified message protocol for the 3tears framework, with adapters for Slack, Discord, and WebSocket clients. Write your agent logic once and deliver it across channels.

```bash
pip install 3tears-channels
```

## What you get

- **One message model** -- `ChannelMessage`, `ChannelResponse`, `ChannelDeliveryMessage`, and `Attachment` carry a request and its reply regardless of channel.
- **Routing** -- `ChannelRouter` and `StreamingChannelRouter` dispatch inbound messages and stream responses back.
- **Slack and Discord** -- payload and rich-formatting builders (`build_slack_blocks`, `build_slack_payload`, `build_discord_embed`, `build_discord_payload`) plus a `should_use_rich_formatting` helper.
- **WebSocket** -- `WebSocketHandler`, `WebSocketProtocol`, a `ConnectionRegistry`, and frame primitives for real-time clients.
- **Rooms and presence** -- `RoomFanout`, `RoomState`, `RoomIndexCollection`, and a three-tier-backed `PresenceCollection` with a `PresenceSweeper` for connection liveness.
- **Email** -- `threetears.channels.mail`: `Mailer` over an `EmailTransport`, an SMTP transport reading its settings per send, `EmailTemplate` substitution with an HTML alternative and `List-Unsubscribe` headers, `send_batch` for many recipients with per-recipient failure isolation, and `BounceReceiver` for HMAC-verified delivery reports. A sibling of the channel adapters rather than one of them -- an email recipient has no conversation and usually no account, so `ChannelDeliveryMessage` does not apply.

## Quickstart

```python
from threetears.channels import ChannelRouter, ChannelMessage, build_slack_blocks

router = ChannelRouter(...)
response = await router.dispatch(ChannelMessage(text="hello", channel="slack", ...))

blocks = build_slack_blocks(response)
```

## WebSocket connection lifetime

`WebSocketHandler` ends a connection on its own in two cases, besides the peer disconnecting.

**The peer stops answering.** Every `heartbeat_interval` seconds (config key, default 30) each open
connection is sent `{"type": "ping"}`. A connection that has sent nothing since the previous ping
is closed with code 1011, and its registry entry, rooms and presence are cleaned up as on a
disconnect. Any frame counts as an answer; an otherwise idle client answers with
`{"type": "pong"}`. A connection busy with a frame (a turn in flight) is not judged until it is
reading again.

**The credential expires.** If the `auth_validator`'s claims include `exp` (unix seconds), the
first frame at or after that time, or the first heartbeat tick of an idle connection, is answered
with the same refusal an unauthenticated connection gets:

```json
{"type": "error", "code": "UNAUTHENTICATED", "message": "access token expired"}
```

and the socket is closed 1008. A client handles it like any `UNAUTHENTICATED` refusal: obtain a
fresh credential and reconnect. Leave `exp` out of the claims for a connection that should not
expire.

A connection that is answering pings also keeps its presence row fresh, so `PresenceSweeper`
does not evict a live member.

## License

MIT. See [LICENSE](LICENSE).
