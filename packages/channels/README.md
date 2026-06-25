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

## Quickstart

```python
from threetears.channels import ChannelRouter, ChannelMessage, build_slack_blocks

router = ChannelRouter(...)
response = await router.dispatch(ChannelMessage(text="hello", channel="slack", ...))

blocks = build_slack_blocks(response)
```

## License

MIT. See [LICENSE](LICENSE).
