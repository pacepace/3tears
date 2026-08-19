"""Where the relay credentials come from, and when they are read.

Two properties are load-bearing and neither is obvious from the type. Settings are
resolved PER SEND, so an operator who corrects a wrong password does not wait for a
deploy; and the password arrives as a `scheme://locator` reference resolved at that
moment, so it is never a field on a config object somebody logs.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from threetears.channels.mail import settings as settings_module
from threetears.channels.mail.settings import (
    EmailAuthWithoutTlsError,
    EmailCredentialsIncompleteError,
    EmailSettingsNotConfiguredError,
    ResolvedEmailSettings,
    StaticEmailSettingsResolver,
)


@pytest.fixture
def _resolved_refs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every secret reference resolved, and answer with a known password.

    :param monkeypatch: the active monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the list the resolver appends each requested reference to
    :rtype: list[str]
    """
    seen: list[str] = []

    def _resolve(ref: str) -> SecretStr:
        seen.append(ref)
        return SecretStr("relay-password")

    monkeypatch.setattr(settings_module, "resolve_secret", _resolve)
    return seen


class TestReadPerSend:
    async def test_the_reference_is_resolved_on_every_send(self, _resolved_refs: list[str]) -> None:
        """Not once at construction. A rotated credential must take effect on the next
        send, and a resolver that cached it would keep sending under the old one until
        the process restarted."""
        resolver = StaticEmailSettingsResolver(
            host="smtp.example",
            port=587,
            from_address="no-reply@acme.example",
            username="apikey",
            password_ref="env://SMTP_PASSWORD",
        )

        await resolver.resolve_for_send()
        await resolver.resolve_for_send()

        assert _resolved_refs == ["env://SMTP_PASSWORD", "env://SMTP_PASSWORD"]

    async def test_the_resolved_settings_carry_the_opened_password(self, _resolved_refs: list[str]) -> None:
        resolver = StaticEmailSettingsResolver(
            host="smtp.example",
            port=587,
            from_address="no-reply@acme.example",
            username="apikey",
            password_ref="env://SMTP_PASSWORD",
        )

        resolved = await resolver.resolve_for_send()

        assert resolved == ResolvedEmailSettings(
            host="smtp.example",
            port=587,
            username="apikey",
            password="relay-password",
            from_address="no-reply@acme.example",
            from_name=None,
            use_starttls=True,
        )


class TestTheCredentialNeverBecomesAField:
    def test_the_resolver_repr_carries_the_reference_not_the_password(self, _resolved_refs: list[str]) -> None:
        """A resolver reaches a log, a traceback frame and a debugger. What it holds is
        the reference, which is not a secret; the password exists only inside the
        `ResolvedEmailSettings` a single send is holding."""
        resolver = StaticEmailSettingsResolver(
            host="smtp.example",
            port=587,
            from_address="no-reply@acme.example",
            username="apikey",
            password_ref="env://SMTP_PASSWORD",
        )

        assert "relay-password" not in repr(resolver)
        assert "env://SMTP_PASSWORD" in repr(resolver)


class TestConfigurationsItRefuses:
    def test_a_username_with_no_password_reference(self) -> None:
        """A half-configured relay fails at send time with an authentication error
        nobody traces back to the configuration."""
        with pytest.raises(EmailCredentialsIncompleteError):
            StaticEmailSettingsResolver(
                host="smtp.example", port=587, from_address="no-reply@acme.example", username="apikey"
            )

    def test_a_password_reference_with_no_username(self) -> None:
        with pytest.raises(EmailCredentialsIncompleteError):
            StaticEmailSettingsResolver(
                host="smtp.example",
                port=587,
                from_address="no-reply@acme.example",
                password_ref="env://SMTP_PASSWORD",
            )

    def test_authenticating_to_a_relay_that_will_not_upgrade_to_tls(self) -> None:
        """SMTP AUTH without STARTTLS puts this platform's own mail credential on the
        wire in clear, and whoever reads it can send mail as us."""
        with pytest.raises(EmailAuthWithoutTlsError):
            StaticEmailSettingsResolver(
                host="smtp.example",
                port=25,
                from_address="no-reply@acme.example",
                username="apikey",
                password_ref="env://SMTP_PASSWORD",
                use_starttls=False,
            )

    def test_an_anonymous_relay_without_tls_is_permitted(self) -> None:
        """An internal relay on port 25 that does not authenticate has no secret to
        disclose, so the refusal above must not catch it."""
        resolver = StaticEmailSettingsResolver(
            host="relay.internal", port=25, from_address="no-reply@acme.example", use_starttls=False
        )

        assert resolver is not None


class TestDisabled:
    async def test_a_disabled_configuration_is_not_configured(self) -> None:
        """`enabled=False` is an operator staging a configuration they have not turned
        on. Sending through it anyway would be the surprise."""
        resolver = StaticEmailSettingsResolver(
            host="smtp.example", port=587, from_address="no-reply@acme.example", enabled=False
        )

        with pytest.raises(EmailSettingsNotConfiguredError):
            await resolver.resolve_for_send()
