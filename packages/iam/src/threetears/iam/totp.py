"""TOTP (RFC 6238) second factors, and their one-time backup codes.

Pure functions and sealing. No database, no transport: enrolment records,
credential lookup, and the hash write-back are the caller's, because every
service stores them differently and none of that is protocol.

**The seed is a bearer secret and is sealed at rest.** Unlike a password hash,
a TOTP seed is symmetric -- anyone holding it can generate valid codes forever,
so a database dump of raw seeds is a silent, permanent bypass of the second
factor for every enrolled user. Sealing uses
:func:`threetears.core.security.seal`, which authenticates a version header as
AAD and returns typed errors, rather than a hand-rolled AES-GCM wrapper.

**Backup codes are hashed, not sealed.** They are high-entropy generated tokens
rather than human-chosen secrets, so SHA-256 is the right tool for the same
reason it is for API keys: there is no dictionary to slow an attacker against,
and the verifier needs an equality lookup. Comparison is constant-time per
candidate.

**A one-step skew window each side.** Roughly thirty seconds either way, which
is what authenticator apps and verifiers conventionally allow. It is not a
meaningfully wide replay window, and narrowing it further mostly produces
support tickets from users with drifting phone clocks.

**Only sealing needs the ``core`` extra** (``3tears-iam[core]``). Seed generation,
provisioning URIs, verification and backup codes run on the base install. ``seal_seed``,
``unseal_seed`` and :class:`DecryptionError` are core's sealing, imported at the top of this
module where core is installed -- sealed seeds are stored data, so they are sealed by the one
implementation that opens them again -- and without the extra they raise an
:class:`ImportError` naming it.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from typing import Final, NoReturn

import pyotp
from pydantic import SecretStr

from threetears.iam._core_extra import core_extra_missing
from threetears.observe import contain, get_logger

try:
    from threetears.core.security import DecryptionError, open_secret, seal
except ModuleNotFoundError as _exc:
    # called for its check, not its result: a module missing from inside an installed core is a
    # broken environment, which it re-raises here as it came. Core's own absence returns the error
    # naming the extra, which each sealing call raises afresh from _CORE_ABSENT.
    core_extra_missing(_exc, needed_by=__name__)
    _CORE_ABSENT: ModuleNotFoundError | None = _exc
else:
    _CORE_ABSENT = None

__all__ = [
    "BACKUP_CODE_COUNT",
    "DecryptionError",
    "TotpVerifyResult",
    "generate_backup_codes",
    "generate_seed",
    "hash_backup_code",
    "match_backup_code",
    "provisioning_uri",
    "seal_seed",
    "unseal_seed",
    "verify_code",
]

#: How many one-time backup codes an enrolment issues.
BACKUP_CODE_COUNT: Final[int] = 10

#: Steps of clock skew tolerated either side of now (one step is 30 seconds).
_VALID_WINDOW: Final[int] = 1

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TotpVerifyResult:
    """The outcome of :func:`verify_code`, shaped like :class:`~threetears.iam.passwords.PasswordVerifyResult`.

    :ivar matched: whether the code verified. Always ``False`` when :attr:`secret_fault` is set.
    :ivar secret_fault: set when the STORED secret is unusable -- not base32, or empty --
        saying which, never quoting the secret. Not a wrong code: the user's second factor is
        broken until it is re-enrolled, and read as a wrong code it looks like a user who keeps
        mistyping. For the operator, not the user: the attempt is denied exactly as a wrong code
        is, so the response does not tell an attacker which accounts hold a broken enrolment.
    """

    matched: bool
    secret_fault: str | None = None

    def __bool__(self) -> NoReturn:
        """Refuse truthiness: this used to be a ``bool``, and an object is always true.

        A caller still writing ``if verify_code(...)`` would otherwise accept every code. Read
        :attr:`matched` instead.

        :raises TypeError: always
        """
        raise TypeError("TotpVerifyResult has no truth value; read .matched (and .secret_fault)")


def generate_seed() -> str:
    """Generate a fresh base32 TOTP secret (160 bits, RFC 4226's recommended default)."""
    return pyotp.random_base32()


def provisioning_uri(secret_b32: str, *, account_name: str, issuer_name: str) -> str:
    """The ``otpauth://`` URI an authenticator app scans as a QR code or accepts by hand.

    :param secret_b32: the base32 seed.
    :ptype secret_b32: str
    :param account_name: what the app shows as the account, usually an email address.
    :ptype account_name: str
    :param issuer_name: what the app shows as the issuer. Required and not defaulted: it is
        what tells a user WHICH service a code belongs to when they have thirty of them, and
        a generic default makes every deployment look identical in the list.
    :ptype issuer_name: str
    :return: the provisioning URI.
    :rtype: str
    """
    return pyotp.TOTP(secret_b32).provisioning_uri(name=account_name, issuer_name=issuer_name)


def verify_code(secret_b32: str, code: str) -> TotpVerifyResult:
    """Verify a live TOTP code.

    Never raises for a malformed code or a malformed stored secret: both deny with
    ``matched=False``, so no rejection reason is distinguishable by exception shape or by
    status code. A malformed SECRET also sets :attr:`TotpVerifyResult.secret_fault`, logs at
    ERROR without the secret, and counts on ``contained_errors_total``: it used to read as a
    wrong code, so a broken enrolment locked its user out with nothing to say why.

    :param secret_b32: the stored base32 seed
    :ptype secret_b32: str
    :param code: the code the user typed
    :ptype code: str
    :return: the outcome, naming a broken stored secret when there is one
    :rtype: TotpVerifyResult
    """
    totp = pyotp.TOTP(secret_b32)
    fault: str | None = None
    try:
        # Decoded on its own, before the code is looked at, so the only failure classified as
        # the secret's is the secret's own decoding.
        key = totp.byte_secret()
    except ValueError as exc:  # binascii.Error, the base32 decoder's, is a ValueError
        # The type only: the fault travels to callers, which should never carry decoder text.
        fault = f"the stored TOTP secret is not valid base32 ({type(exc).__name__})"
        _report_secret_fault(exc, fault)
    else:
        if not key:
            fault = "the stored TOTP secret is empty"
            _report_secret_fault(ValueError(fault), fault)
    if fault is not None:
        return TotpVerifyResult(matched=False, secret_fault=fault)
    if not code:
        return TotpVerifyResult(matched=False)
    return TotpVerifyResult(matched=bool(totp.verify(code, valid_window=_VALID_WINDOW)))


def _report_secret_fault(exc: Exception, fault: str) -> None:
    """Log and count a broken stored secret, for the operator. Never the secret itself.

    The base32 decoder's messages ("Non-base32 digit found", "Incorrect padding") do not
    quote their input, and nothing here passes the secret to the log.

    :param exc: the decoder's error, or one built for an empty secret
    :ptype exc: Exception
    :param fault: what is wrong, as the result names it
    :ptype fault: str
    :return: nothing
    :rtype: None
    """
    contain(exc, log=log, what=f"totp verify: {fault}; denied, and the enrolment needs redoing", level="error")


def generate_backup_codes(count: int = BACKUP_CODE_COUNT) -> list[str]:
    """Generate one-time backup codes, shown to the user exactly once at enrolment.

    Only their hashes are ever persisted. The hyphenated shape is for humans copying them off
    a screen under stress, which is the only circumstance in which they are ever used.
    """
    return [f"{secrets.token_hex(4)}-{secrets.token_hex(4)}" for _ in range(count)]


def hash_backup_code(code: str) -> str:
    """SHA-256 hex digest of a backup code, whitespace-stripped.

    Stripping matters: these are read off a screen and pasted, and a trailing space that
    silently invalidates a user's last backup code is a lockout, not a security control.
    """
    return hashlib.sha256(code.strip().encode("utf-8")).hexdigest()


def match_backup_code(code: str, hashes: tuple[str, ...]) -> str | None:
    """The stored hash matching ``code``, or ``None``.

    Returns the matched hash rather than a boolean so the caller can remove exactly that
    entry -- a backup code is single-use, and "which one was it" is needed to burn it.

    Comparison is constant-time per candidate. The loop's LENGTH is not hidden, which is
    fine: how many backup codes remain is not a secret worth protecting, while the codes
    themselves are.
    """
    target = hash_backup_code(code)
    matched: str | None = None
    for stored_hash in hashes:
        if secrets.compare_digest(stored_hash, target):
            matched = stored_hash
    return matched


def seal_seed(seed_b32: str, key: SecretStr) -> str:
    """Seal a TOTP seed for storage. See the module docstring on why this is not optional.

    :raises ImportError: the ``core`` extra is not installed
    """
    if _CORE_ABSENT is not None:
        raise core_extra_missing(_CORE_ABSENT, needed_by="threetears.iam.totp.seal_seed") from _CORE_ABSENT
    return seal(seed_b32, key)


def unseal_seed(sealed: str, key: SecretStr) -> str:
    """Recover a sealed TOTP seed.

    :raises DecryptionError: the token is malformed, of an unknown version, sealed under a
        different key, or tampered with.
    :raises ImportError: the ``core`` extra is not installed
    """
    if _CORE_ABSENT is not None:
        raise core_extra_missing(_CORE_ABSENT, needed_by="threetears.iam.totp.unseal_seed") from _CORE_ABSENT
    return open_secret(sealed, key).get_secret_value()


def __getattr__(name: str) -> object:
    """Name the ``core`` extra when :class:`DecryptionError` is asked for without it.

    With core installed, ``DecryptionError`` is core's own class, imported above, so
    ``except DecryptionError`` here catches what ``unseal_seed`` raises and this hook is never
    reached for it. On the base install the name is absent, so ``dir()`` leaves it out, and
    reaching it raises the error that names the extra.

    :param name: the attribute asked for
    :ptype name: str
    :return: never returns
    :rtype: object
    :raises ImportError: ``DecryptionError`` was asked for and the ``core`` extra is not installed
    :raises AttributeError: for any other name
    """
    if name == "DecryptionError" and _CORE_ABSENT is not None:
        raise core_extra_missing(_CORE_ABSENT, needed_by="threetears.iam.totp.DecryptionError") from _CORE_ABSENT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
