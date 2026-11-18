# previews

Notification previews for our team-chat app (`previews.py`). Before a message is opened, the phone shows a
preview of it: `preview(body, channel)` builds it for one channel (`lock_screen`, `banner` or `watch`), and
`previews(body)` for all three. Each channel has a character limit; the rules for fitting a message to it are in the
module docstring.

The design team went through a sample of real notifications (names changed) and signed off the preview each one
should get on the channel it was sent to. They are in `data/signed_off.jsonl`: the message `body`, its `channel`, and
the `approved` preview.

Tests: `python -m unittest discover tests` from this directory.
