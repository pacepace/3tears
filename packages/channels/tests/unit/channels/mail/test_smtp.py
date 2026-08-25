"""The SMTP conversation, and the promise that nothing escapes it uncaught.

Promoted from `identity_core/email/smtp.py` together with the decisions its own suite
pinned: whether STARTTLS is negotiated, whether AUTH is attempted, what the message
carries, and how each failure is reported. Every one of those is visible at the
`smtplib.SMTP` boundary, so the relay is faked rather than reached.

The additions this package makes over the original are tested here too -- a
`multipart/alternative` HTML part, `List-Unsubscribe` headers, and a failure recorder
that replaces the original's hard-wired identity audit call.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from threetears.channels.mail import smtp as smtp_module
from threetears.channels.mail.message import EmailMessage, EmailSendError
from threetears.channels.mail.settings import (
    EmailSettingsNotConfiguredError,
    ResolvedEmailSettings,
)
from threetears.channels.mail.smtp import SmtpEmailTransport


def _settings(**overrides: Any) -> ResolvedEmailSettings:
    base: dict[str, Any] = {
        "host": "smtp.example",
        "port": 587,
        "username": "apikey",
        "password": "relay-password",
        "from_address": "no-reply@acme.example",
        "from_name": None,
        "use_starttls": True,
    }
    return ResolvedEmailSettings(**{**base, **overrides})


class _RecordingSmtp:
    """Records the conversation, so a test can assert what was NOT said.

    Stands in for `smtplib.SMTP`, which is a concrete stdlib class rather than a
    protocol this package defines, so there is no parity target to declare against.
    """

    instances: list[_RecordingSmtp] = []

    def __init__(self, host: str, port: int, timeout: float | None = None) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.started_tls = False
        self.logged_in_as: str | None = None
        self.sent: list[Any] = []
        _RecordingSmtp.instances.append(self)

    def __enter__(self) -> _RecordingSmtp:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def starttls(self, context: object = None) -> None:
        self.started_tls = True

    def login(self, username: str, password: str) -> None:
        self.logged_in_as = username

    def send_message(self, mime: Any) -> None:
        self.sent.append(mime)


# parity-with: threetears.channels.mail.settings.EmailSettingsResolver
class _StubResolver:
    """Answers `resolve_for_send` with a fixed value, an exception, or a sequence."""

    def __init__(self, *answers: ResolvedEmailSettings | Exception) -> None:
        self._answers = list(answers)
        self.calls = 0

    async def resolve_for_send(self) -> ResolvedEmailSettings:
        self.calls += 1
        answer = self._answers[min(self.calls - 1, len(self._answers) - 1)]
        if isinstance(answer, Exception):
            raise answer
        return answer


class _RecordedFailures:
    """Captures what a failed send decides to record, without a message bus."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.calls: list[Mapping[str, object]] = []
        self._raises = raises

    async def __call__(self, details: Mapping[str, object]) -> None:
        self.calls.append(details)
        if self._raises is not None:
            raise self._raises


@pytest.fixture(autouse=True)
def _recording_smtp(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingSmtp.instances = []
    monkeypatch.setattr(smtp_module.smtplib, "SMTP", _RecordingSmtp)


def _raising_smtp(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    """Point the transport at a relay that raises `failure` when handed the message.

    :param monkeypatch: the active monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :param failure: what the relay raises from `send_message`
    :ptype failure: Exception
    :return: nothing
    :rtype: None
    """

    class _Raising(_RecordingSmtp):
        def send_message(self, mime: Any) -> None:
            raise failure

    _RecordingSmtp.instances = []
    monkeypatch.setattr(smtp_module.smtplib, "SMTP", _Raising)


def _transport(*answers: ResolvedEmailSettings | Exception, on_failure: Any = None) -> SmtpEmailTransport:
    return SmtpEmailTransport(_StubResolver(*answers), on_failure=on_failure)


def _message(**overrides: Any) -> EmailMessage:
    base: dict[str, Any] = {
        "to": "ada@acme.example",
        "subject": "Reset your password",
        "body_text": "https://example/reset/abc",
    }
    return EmailMessage(**{**base, **overrides})


class TestTheConversationItHas:
    async def test_it_negotiates_starttls_and_authenticates(self) -> None:
        await _transport(_settings()).send(_message())

        client = _RecordingSmtp.instances[-1]
        assert client.started_tls is True
        assert client.logged_in_as == "apikey"
        assert client.host == "smtp.example"
        assert client.port == 587

    async def test_starttls_is_skipped_when_the_relay_does_not_offer_it(self) -> None:
        """An internal relay on port 25 has no TLS to upgrade to. Skipping is the
        operator's declared choice, not a silent downgrade."""
        await _transport(_settings(use_starttls=False, port=25, username=None, password=None)).send(_message())

        assert _RecordingSmtp.instances[-1].started_tls is False

    async def test_authenticating_without_starttls_is_refused_rather_than_sent_in_clear(self) -> None:
        """Fail CLOSED. The settings surface already refuses this pairing, so reaching
        here means the configuration arrived another way -- a direct ops UPDATE, a
        restore from an older dump. Asserts the password never left the process, not
        merely that an exception was raised: `login` must not have been called."""
        with pytest.raises(EmailSendError, match="unencrypted"):
            await _transport(_settings(use_starttls=False, port=25)).send(_message())

        assert _RecordingSmtp.instances[-1].logged_in_as is None

    async def test_an_anonymous_relay_is_not_offered_credentials(self) -> None:
        """AUTH against a relay that does not want it is an error on some servers, so
        the absence of a username has to mean 'do not try', not 'try with None'."""
        await _transport(_settings(username=None, password=None)).send(_message())

        assert _RecordingSmtp.instances[-1].logged_in_as is None

    async def test_a_timeout_is_always_applied(self) -> None:
        """Without one a hung relay holds a worker thread forever, and threads are the
        scarce resource on this path."""
        await _transport(_settings()).send(_message())

        assert _RecordingSmtp.instances[-1].timeout is not None


class TestSettingsAreReadPerSend:
    async def test_settings_changed_between_sends_take_effect_without_a_restart(self) -> None:
        """The least obvious and most operationally valuable property of the original.
        An operator correcting a bad host expects the next send to work, not the next
        deploy."""
        transport = _transport(_settings(host="wrong.example"), _settings(host="right.example"))

        await transport.send(_message())
        await transport.send(_message())

        assert [client.host for client in _RecordingSmtp.instances] == ["wrong.example", "right.example"]


class TestWhatTheMessageCarries:
    async def test_the_configured_sender_and_the_body(self) -> None:
        await _transport(_settings(from_name="Acme Security")).send(_message())

        mime = _RecordingSmtp.instances[-1].sent[0]
        assert mime["From"] == "Acme Security <no-reply@acme.example>"
        assert mime["To"] == "ada@acme.example"
        assert mime["Subject"] == "Reset your password"
        assert "https://example/reset/abc" in mime.get_content()

    async def test_a_bare_address_is_used_when_no_display_name_is_configured(self) -> None:
        await _transport(_settings()).send(_message())

        assert _RecordingSmtp.instances[-1].sent[0]["From"] == "no-reply@acme.example"

    async def test_an_html_part_becomes_a_multipart_alternative_with_text_first(self) -> None:
        """Order matters: a client picks the LAST part it can render, so the plain-text
        alternative has to come first or an HTML-capable client never sees the HTML."""
        await _transport(_settings()).send(_message(body_html="<p>rich</p>"))

        mime = _RecordingSmtp.instances[-1].sent[0]
        assert mime.get_content_type() == "multipart/alternative"
        assert [part.get_content_type() for part in mime.iter_parts()] == ["text/plain", "text/html"]

    async def test_a_text_only_message_stays_a_single_part(self) -> None:
        """A `multipart/alternative` with one part is legal and scores worse with spam
        filters than the plain message it wraps."""
        await _transport(_settings()).send(_message())

        assert _RecordingSmtp.instances[-1].sent[0].get_content_type() == "text/plain"

    async def test_an_unsubscribe_url_becomes_a_list_unsubscribe_header(self) -> None:
        """Survey mail legally needs it, and a mailbox provider that finds one turns
        its own unsubscribe button into a header rather than a spam report."""
        await _transport(_settings()).send(_message(list_unsubscribe_url="https://acme.example/u/abc"))

        mime = _RecordingSmtp.instances[-1].sent[0]
        assert mime["List-Unsubscribe"] == "<https://acme.example/u/abc>"

    async def test_an_https_unsubscribe_url_also_gets_the_one_click_header(self) -> None:
        """RFC 8058 one-click needs both headers, and it is what Gmail and Yahoo
        require of bulk senders."""
        await _transport(_settings()).send(_message(list_unsubscribe_url="https://acme.example/u/abc"))

        assert _RecordingSmtp.instances[-1].sent[0]["List-Unsubscribe-Post"] == "List-Unsubscribe=One-Click"

    async def test_a_mailto_only_unsubscribe_does_not_claim_one_click(self) -> None:
        """RFC 8058 one-click is an HTTPS POST. Claiming it over `mailto:` advertises
        something no client can perform."""
        await _transport(_settings()).send(_message(list_unsubscribe_mailto="unsub@acme.example"))

        mime = _RecordingSmtp.instances[-1].sent[0]
        assert mime["List-Unsubscribe"] == "<mailto:unsub@acme.example>"
        assert mime["List-Unsubscribe-Post"] is None

    async def test_both_unsubscribe_forms_are_offered_together(self) -> None:
        await _transport(_settings()).send(
            _message(
                list_unsubscribe_url="https://acme.example/u/abc",
                list_unsubscribe_mailto="unsub@acme.example",
            )
        )

        header = _RecordingSmtp.instances[-1].sent[0]["List-Unsubscribe"]
        assert header == "<https://acme.example/u/abc>, <mailto:unsub@acme.example>"

    async def test_caller_supplied_headers_are_applied(self) -> None:
        """A provider correlates its bounce callback to a send by a header it can see,
        so the caller needs a way to put one there."""
        await _transport(_settings()).send(_message(headers={"X-Campaign-Id": "wave-3"}))

        assert _RecordingSmtp.instances[-1].sent[0]["X-Campaign-Id"] == "wave-3"

    async def test_a_caller_supplied_header_cannot_replace_the_envelope(self) -> None:
        """`To` and `From` are the send's own decision. A caller that could overwrite
        them could redirect a message the transport believes it addressed elsewhere."""
        with pytest.raises(EmailSendError, match="To"):
            await _transport(_settings()).send(_message(headers={"To": "mallory@evil.example"}))


class TestEveryFailureLeavesAsEmailSendError:
    """The contract a caller depends on. A failure that escapes as anything else is
    caught by nobody, and an operation that merely wanted to send a notification fails
    for a user who did nothing wrong."""

    async def test_an_unconfigured_platform(self) -> None:
        with pytest.raises(EmailSendError):
            await _transport(EmailSettingsNotConfiguredError("nothing configured")).send(_message())

    async def test_a_relay_that_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import smtplib as real_smtplib

        _raising_smtp(monkeypatch, real_smtplib.SMTPRecipientsRefused({}))

        with pytest.raises(EmailSendError):
            await _transport(_settings()).send(_message())

    async def test_an_unforeseen_error_is_still_an_email_send_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The reason the catch is deliberately broad. A narrow `(SMTPException,
        OSError, SSLError)` reads as more precise and lets `HeaderWriteError` -- and
        anything else -- through to a caller that cannot handle it."""
        _raising_smtp(monkeypatch, RuntimeError("something nobody predicted"))

        with pytest.raises(EmailSendError):
            await _transport(_settings()).send(_message())

    async def test_a_database_error_resolving_settings_is_still_an_email_send_error(self) -> None:
        """The settings read is a real IO call on every send, whatever backs it."""
        with pytest.raises(EmailSendError):
            await _transport(OSError("connection reset by peer")).send(_message())


class TestNothingSecretAndNothingPersonalEscapes:
    async def test_the_relay_password_never_appears_in_the_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The message reaches a log and a failure record; the credential must not
        travel with it."""
        _raising_smtp(monkeypatch, OSError("connection reset"))

        with pytest.raises(EmailSendError) as caught:
            await _transport(_settings()).send(_message())

        assert "relay-password" not in str(caught.value)

    async def test_an_exception_naming_the_recipient_is_redacted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Exception text on this path routinely carries an address: `HeaderWriteError`
        names the offending header VALUE and the header in question is `To`."""
        recorder = _RecordedFailures()
        _raising_smtp(monkeypatch, ValueError("cannot fold header To: ada@acme.example"))

        with pytest.raises(EmailSendError) as caught:
            await _transport(_settings(), on_failure=recorder).send(_message())

        assert "ada@acme.example" not in str(caught.value)
        assert "ada@acme.example" not in str(recorder.calls[0])
        assert "[address redacted]" in str(recorder.calls[0]["reason"])

    async def test_the_sender_address_is_redacted_too(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _RecordedFailures()
        _raising_smtp(monkeypatch, ValueError("sender refused: no-reply@acme.example"))

        with pytest.raises(EmailSendError):
            await _transport(_settings(), on_failure=recorder).send(_message())

        assert "no-reply@acme.example" not in str(recorder.calls[0])


class TestTheFailureRecorder:
    """The original hard-wired identity-core's own audit call. A promoted package
    cannot, so the hook is a callback -- and it keeps the original's two properties:
    a failed send is recorded, and recording it never changes what the caller sees."""

    async def test_a_refused_relay_is_recorded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorder = _RecordedFailures()
        _raising_smtp(monkeypatch, ValueError("relay refused"))

        with pytest.raises(EmailSendError):
            await _transport(_settings(), on_failure=recorder).send(_message())

        assert len(recorder.calls) == 1
        assert recorder.calls[0]["smtp_host"] == "smtp.example"
        assert recorder.calls[0]["smtp_port"] == 587

    async def test_unusable_settings_are_recorded_too(self) -> None:
        recorder = _RecordedFailures()

        with pytest.raises(EmailSendError):
            await _transport(EmailSettingsNotConfiguredError("nothing configured"), on_failure=recorder).send(
                _message()
            )

        assert len(recorder.calls) == 1

    async def test_a_recorder_failure_never_displaces_the_send_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The recorder sits between the failure and the raise. If it escaped, a caller
        catching `EmailSendError` would get something else -- and only when something
        was already wrong."""
        recorder = _RecordedFailures(raises=RuntimeError("bus down"))
        _raising_smtp(monkeypatch, ValueError("relay refused"))

        with pytest.raises(EmailSendError):
            await _transport(_settings(), on_failure=recorder).send(_message())

    async def test_a_successful_send_records_nothing(self) -> None:
        recorder = _RecordedFailures()

        await _transport(_settings(), on_failure=recorder).send(_message())

        assert recorder.calls == []
