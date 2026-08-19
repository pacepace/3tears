"""Where a send gets its relay configuration, and when.

**Read per send, never held from startup.** That is the least obvious and most
operationally valuable property of the mailer this package promotes
(`identity_core/email/smtp.py`'s module docstring): an operator who corrects a wrong
SMTP password expects the next send to work, not the next deploy. The read is cheap
relative to an SMTP round-trip, and mail runs at nothing like request volume.

**3tears owns the protocol, not the storage.** identity-core resolves these from its own
`platform_email_settings` table, with the password sealed under
`threetears.core.security.seal`, a residency-routed pool and its own audit trail. None
of that is generic -- a second product has no residency router and no lifecycle-event
bus -- so what moves upstream is the shape a transport depends on
(:class:`EmailSettingsResolver`), and each product keeps the store behind it.
`PlatformEmailSettingsService` already satisfies this protocol structurally, method name
and return shape included.

:class:`StaticEmailSettingsResolver` is the answer for a product that has no settings
table and does not want one. It still honours both disciplines: the password arrives as
a `scheme://locator` secret reference rather than a value, and it is resolved on every
send rather than at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from threetears.core.security.secret_refs import resolve_secret

__all__ = [
    "EmailAuthWithoutTlsError",
    "EmailCredentialsIncompleteError",
    "EmailSettingsNotConfiguredError",
    "EmailSettingsResolver",
    "ResolvedEmailSettings",
    "StaticEmailSettingsResolver",
]


class EmailSettingsNotConfiguredError(Exception):
    """No usable outbound-email configuration exists, so there is nothing to send
    through. Distinct from a relay that refused: nothing was attempted."""


class EmailCredentialsIncompleteError(Exception):
    """Half a credential was supplied: a username with no password, or a password with
    no username.

    Refused where the configuration is written rather than at send time. A
    half-configured relay fails with an authentication error nobody traces back to the
    configuration, and by then the failure is somebody's missing invitation.
    """


class EmailAuthWithoutTlsError(Exception):
    """Credentials were configured on a relay that will not upgrade to TLS.

    SMTP AUTH over an unencrypted connection puts the password on the wire in clear, and
    it is this platform's own mail credential -- whoever reads it can send mail as us,
    including anything carrying a single-use link. An anonymous relay with no STARTTLS
    stays permitted: there is nothing secret to disclose.
    """


@dataclass(frozen=True, slots=True)
class ResolvedEmailSettings:
    """Settings with the password OPENED, built only at send time.

    Deliberately not the type any read or admin path returns: it exists for the moments
    between resolving a send and performing it, which is the only window the plaintext
    should exist in. Field-for-field identical to identity-core's own
    `ResolvedEmailSettings`, so its service satisfies the protocol below unchanged.

    :ivar host: relay hostname
    :ivar port: relay port
    :ivar username: SMTP AUTH username, or ``None`` for an anonymous relay
    :ivar password: SMTP AUTH password, or ``None`` for an anonymous relay
    :ivar from_address: envelope and header sender
    :ivar from_name: optional display name paired with `from_address`
    :ivar use_starttls: whether to upgrade the connection before saying anything else
    """

    host: str
    port: int
    username: str | None
    password: str | None
    from_address: str
    from_name: str | None
    use_starttls: bool


@runtime_checkable
class EmailSettingsResolver(Protocol):
    """What a transport asks for its configuration, once per send."""

    async def resolve_for_send(self) -> ResolvedEmailSettings:
        """Return the configuration this send should use.

        :return: settings including the plaintext password, if one is configured
        :rtype: ResolvedEmailSettings
        :raises EmailSettingsNotConfiguredError: nothing usable is configured
        """
        ...


class StaticEmailSettingsResolver:
    """Fixed relay settings whose password is a secret reference resolved per send.

    For a product that configures mail from its own deployment rather than from an
    operator-editable table. The reference is what this object holds; the password
    exists only inside the :class:`ResolvedEmailSettings` a single send is using, so a
    resolver reaching a log, a traceback frame or a debugger discloses nothing.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        from_address: str,
        username: str | None = None,
        password_ref: str | None = None,
        from_name: str | None = None,
        use_starttls: bool = True,
        enabled: bool = True,
    ) -> None:
        """
        :param host: relay hostname
        :ptype host: str
        :param port: relay port
        :ptype port: int
        :param from_address: envelope and header sender
        :ptype from_address: str
        :param username: SMTP AUTH username, or ``None`` for an anonymous relay
        :ptype username: str | None
        :param password_ref: `scheme://locator` reference to the SMTP AUTH password,
            resolved at each send; an inline password is deliberately not accepted
        :ptype password_ref: str | None
        :param from_name: optional display name paired with `from_address`
        :ptype from_name: str | None
        :param use_starttls: whether a send upgrades the connection before AUTH
        :ptype use_starttls: bool
        :param enabled: ``False`` stages a configuration without arming it; every
            resolution then refuses rather than sending through it
        :ptype enabled: bool
        :return: nothing
        :rtype: None
        :raises EmailCredentialsIncompleteError: a username with no password reference,
            or a password reference with no username
        :raises EmailAuthWithoutTlsError: credentials on a relay that will not upgrade
        """
        if (username is None) != (password_ref is None):
            raise EmailCredentialsIncompleteError(
                "an SMTP username needs a password reference and a password reference needs a "
                "username. Supply both, or neither for an anonymous relay that does not authenticate."
            )
        if username is not None and not use_starttls:
            raise EmailAuthWithoutTlsError(
                "a relay that authenticates must use STARTTLS: without it the SMTP password is sent "
                "in clear to anyone on the network path, and it is this platform's own credential. "
                "Enable STARTTLS, or drop the credentials to use an anonymous relay."
            )
        self._host = host
        self._port = port
        self._from_address = from_address
        self._username = username
        self._password_ref = password_ref
        self._from_name = from_name
        self._use_starttls = use_starttls
        self._enabled = enabled

    def __repr__(self) -> str:
        """Render the resolver without disclosing anything the reference protects.

        :return: a debug representation naming the reference, never a password
        :rtype: str
        """
        return (
            f"StaticEmailSettingsResolver(host={self._host!r}, port={self._port!r}, "
            f"from_address={self._from_address!r}, username={self._username!r}, "
            f"password_ref={self._password_ref!r}, use_starttls={self._use_starttls!r}, "
            f"enabled={self._enabled!r})"
        )

    async def resolve_for_send(self) -> ResolvedEmailSettings:
        """Resolve the password reference and return the settings for one send.

        :return: settings including the plaintext password, if one is configured
        :rtype: ResolvedEmailSettings
        :raises EmailSettingsNotConfiguredError: this configuration is not enabled
        :raises threetears.core.security.secret_refs.SecretResolutionError: the
            reference names no resolvable secret
        """
        if not self._enabled:
            raise EmailSettingsNotConfiguredError("outbound email is configured but not enabled")
        password = None if self._password_ref is None else resolve_secret(self._password_ref).get_secret_value()
        return ResolvedEmailSettings(
            host=self._host,
            port=self._port,
            username=self._username,
            password=password,
            from_address=self._from_address,
            from_name=self._from_name,
            use_starttls=self._use_starttls,
        )
